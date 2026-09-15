"""完整流程 2：CNN 图像分类（程序生成条纹图像）。"""

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

SEED, BATCH_SIZE, EPOCHS, LR = 42, 64, 8, 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class StripeDataset(Dataset):
    """类别0=竖条，类别1=横条；x:[N,1,28,28]。"""

    def __init__(self, n=2000):
        g = torch.Generator().manual_seed(SEED)
        self.y = torch.randint(2, (n,), generator=g)
        self.x = torch.zeros(n, 1, 28, 28)
        for i, label in enumerate(self.y):
            pos = int(torch.randint(5, 21, (1,), generator=g))
            if label == 0:
                self.x[i, 0, :, pos : pos + 3] = 1
            else:
                self.x[i, 0, pos : pos + 3, :] = 1
        self.x = (self.x + 0.15 * torch.randn(self.x.shape, generator=g)).clamp(0, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        x = self.features(x)  # [B,1,28,28] -> [B,32,1,1]
        return self.head(x.flatten(1))  # -> [B,2]


def evaluate(model, loader, loss_fn):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    set_seed()
    train_ds, val_ds, test_ds = random_split(
        StripeDataset(), [1400, 300, 300], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE)
    test_loader = DataLoader(test_ds, BATCH_SIZE)
    model = CNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best_cnn.pt")
        print(
            f"epoch={epoch:02d} train_loss={total_loss/len(train_ds):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
    model.load_state_dict(
        torch.load("best_cnn.pt", map_location=DEVICE, weights_only=True)
    )
    print("test:", evaluate(model, test_loader, loss_fn))
    x, y = next(iter(test_loader))
    print(
        "pred:",
        model(x[:8].to(DEVICE)).argmax(1).cpu().tolist(),
        "true:",
        y[:8].tolist(),
    )


if __name__ == "__main__":
    main()
