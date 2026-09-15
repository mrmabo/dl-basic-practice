"""完整流程 5：Autoencoder 异常检测（仅用正常训练集学习重构）。"""

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED, BATCH_SIZE, EPOCHS = 42, 64, 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class AnomalyDataset(Dataset):
    def __init__(self, n, anomaly_ratio=0.0, seed=SEED):
        g = torch.Generator().manual_seed(seed)
        self.y = (torch.rand(n, generator=g) < anomaly_ratio).long()
        normal = torch.randn(n, 12, generator=g) * 0.5
        anomaly = torch.randn(n, 12, generator=g) * 1.5 + 3
        self.x = torch.where(self.y[:, None].bool(), anomaly, normal)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 12))

    def forward(self, x):
        return self.decoder(self.encoder(x))  # [B,12] -> [B,3] -> [B,12]


def scores(model, loader):
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            all_scores.append(((model(x) - x) ** 2).mean(1).cpu())
            all_labels.append(y)
    return torch.cat(all_scores), torch.cat(all_labels)


def main():
    set_seed()
    train = AnomalyDataset(1800, 0, 42)
    val = AnomalyDataset(500, 0, 43)
    test = AnomalyDataset(800, 0.25, 44)
    train_loader = DataLoader(train, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, BATCH_SIZE)
    test_loader = DataLoader(test, BATCH_SIZE)
    model = Autoencoder().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), x)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        val_scores, _ = scores(model, val_loader)
        val_loss = val_scores.mean().item()
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best_autoencoder.pt")
        print(
            f"epoch={epoch:02d} train_mse={total/len(train):.5f} val_mse={val_loss:.5f}"
        )
    model.load_state_dict(
        torch.load("best_autoencoder.pt", map_location=DEVICE, weights_only=True)
    )
    val_scores, _ = scores(model, val_loader)
    threshold = torch.quantile(val_scores, 0.99)  # 99%正常样本应低于阈值
    test_scores, y = scores(model, test_loader)
    pred = (test_scores > threshold).long()
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    accuracy = (pred == y).float().mean()
    print(
        f"threshold={threshold:.5f} accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f}"
    )
    print("scores:", test_scores[:10].round(decimals=3).tolist())
    print("pred:", pred[:10].tolist(), "true:", y[:10].tolist())


if __name__ == "__main__":
    main()
