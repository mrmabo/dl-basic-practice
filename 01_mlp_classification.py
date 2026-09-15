"""完整流程 1：使用 UCI Wine 公开数据训练 MLP 多分类模型。"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
BATCH_SIZE = 32
EPOCHS = int(os.getenv("EPOCHS", "30"))
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "mlp_wine" / "wine.data"
CHECKPOINT_PATH = ROOT / "best_mlp.pt"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WineDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def stratified_split(labels, train_ratio=0.6, val_ratio=0.2):
    """按类别划分索引，避免小数据集的某个集合缺少某一类别。"""
    rng = np.random.default_rng(SEED)
    train_indices, val_indices, test_indices = [], [], []

    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)
        train_end = int(len(class_indices) * train_ratio)
        val_end = train_end + int(len(class_indices) * val_ratio)
        train_indices.extend(class_indices[:train_end])
        val_indices.extend(class_indices[train_end:val_end])
        test_indices.extend(class_indices[val_end:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def build_dataloaders():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {DATA_PATH}\n请先运行：python download_01_mlp_wine.py"
        )

    raw = np.loadtxt(DATA_PATH, delimiter=",", dtype=np.float32)
    labels = raw[:, 0].astype(np.int64) - 1
    features = raw[:, 1:]
    train_idx, val_idx, test_idx = stratified_split(labels)

    # 只使用训练集统计量标准化，避免验证集和测试集信息泄漏。
    mean = features[train_idx].mean(axis=0)
    std = features[train_idx].std(axis=0)
    std[std == 0] = 1.0
    features = (features - mean) / std

    train_dataset = WineDataset(features[train_idx], labels[train_idx])
    val_dataset = WineDataset(features[val_idx], labels[val_idx])
    test_dataset = WineDataset(features[test_idx], labels[test_idx])

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
    return train_loader, val_loader, test_loader


class MLP(nn.Module):
    def __init__(self, input_features=13, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, features):
        return self.network(features)  # [B, 13] -> [B, 3]


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            if training:
                optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * features.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += features.size(0)

    return total_loss / total, correct / total


def main():
    set_seed()
    train_loader, val_loader, test_loader = build_dataloaders()
    model = MLP().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f}"
        )

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_acc = run_epoch(model, test_loader, loss_fn)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

    features, labels = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        predictions = model(features[:5].to(DEVICE)).argmax(dim=1).cpu()
    print("pred:", predictions.tolist())
    print("true:", labels[:5].tolist())


if __name__ == "__main__":
    main()
