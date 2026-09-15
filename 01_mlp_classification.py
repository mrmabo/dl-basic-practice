"""完整流程 1：MLP 多分类（合成表格数据，无需下载）。"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

SEED, BATCH_SIZE, EPOCHS, LR = 42, 64, 10, 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class TabularDataset(Dataset):
    def __init__(self, n=2400, n_features=20, n_classes=3):
        g = torch.Generator().manual_seed(SEED)
        centers = torch.randn(n_classes, n_features, generator=g) * 2
        self.y = torch.randint(n_classes, (n,), generator=g)
        self.x = centers[self.y] + 0.8 * torch.randn(n, n_features, generator=g)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_features=20, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(.2),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes))
    def forward(self, x): return self.net(x)  # [B,20] -> [B,3]

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None; model.train(training)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if training: optimizer.zero_grad()
            logits = model(x); loss = criterion(logits, y)
            if training: loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0); correct += (logits.argmax(1) == y).sum().item(); total += x.size(0)
    return total_loss / total, correct / total

def main():
    set_seed(); full = TabularDataset()
    train_ds, val_ds, test_ds = random_split(full, [1600, 400, 400], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True); val_loader = DataLoader(val_ds, BATCH_SIZE); test_loader = DataLoader(test_ds, BATCH_SIZE)
    model = MLP().to(DEVICE); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        if va_loss < best: best = va_loss; torch.save(model.state_dict(), "best_mlp.pt")
        print(f"epoch={epoch:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} val_loss={va_loss:.4f} val_acc={va_acc:.3f}")
    model.load_state_dict(torch.load("best_mlp.pt", map_location=DEVICE, weights_only=True))
    test_loss, test_acc = run_epoch(model, test_loader, criterion); print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")
    x, y = next(iter(test_loader)); pred = model(x[:5].to(DEVICE)).argmax(1).cpu(); print("pred:", pred.tolist(), "true:", y[:5].tolist())

if __name__ == "__main__": main()
