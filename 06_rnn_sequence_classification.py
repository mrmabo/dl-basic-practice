"""完整流程 6：使用 UCR SyntheticControl 公开数据训练 RNN 分类模型。"""

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "rnn_synthetic_control"
TRAIN_PATH = DATA_DIR / "SyntheticControl_TRAIN.txt"
TEST_PATH = DATA_DIR / "SyntheticControl_TEST.txt"
CHECKPOINT_PATH = ROOT / "best_rnn.pt"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class SequenceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(-1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def load_ucr_file(path):
    raw = np.loadtxt(path, dtype=np.float32)
    labels = raw[:, 0].astype(np.int64) - 1
    features = raw[:, 1:]
    return features, labels


def stratified_train_val_split(labels, val_ratio=0.2):
    rng = np.random.default_rng(SEED)
    train_indices, val_indices = [], []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.array(train_indices), np.array(val_indices)


def build_dataloaders():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {TRAIN_PATH} 或 {TEST_PATH}\n"
            "请先运行：python download_data/download_06_rnn_synthetic_control.py"
        )
    all_train_x, all_train_y = load_ucr_file(TRAIN_PATH)
    test_x, test_y = load_ucr_file(TEST_PATH)
    train_idx, val_idx = stratified_train_val_split(all_train_y)

    mean = all_train_x[train_idx].mean(axis=0)
    std = all_train_x[train_idx].std(axis=0)
    std[std == 0] = 1.0
    train_x = (all_train_x[train_idx] - mean) / std
    val_x = (all_train_x[val_idx] - mean) / std
    test_x = (test_x - mean) / std

    train_dataset = SequenceDataset(train_x, all_train_y[train_idx])
    val_dataset = SequenceDataset(val_x, all_train_y[val_idx])
    test_dataset = SequenceDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
    return train_loader, val_loader, test_loader


class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.head = nn.Linear(32, 6)

    def forward(self, features):
        _, hidden = self.rnn(features)  # [B, 60, 1] -> hidden [1, B, 32]
        return self.head(hidden[-1])  # [B, 6]


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features)
            total_loss += loss_fn(logits, labels).item() * features.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += features.size(0)
    return total_loss / total, correct / total


def main():
    set_seed()
    train_loader, val_loader, test_loader = build_dataloaders()
    model = RNNClassifier().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss / len(train_loader.dataset):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")
    features, labels = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        predictions = model(features[:8].to(DEVICE)).argmax(dim=1).cpu()
    print("pred:", predictions.tolist())
    print("true:", labels[:8].tolist())


if __name__ == "__main__":
    main()
