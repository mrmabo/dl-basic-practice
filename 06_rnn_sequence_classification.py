"""完整流程 6：使用 UCR SyntheticControl 训练固定或可变长度 RNN 分类模型。"""

"""
单个样本：[L_i, 1] + 标量 label + 整数 L_i
      ↓ collate：补齐并组 batch
一个 batch：[B, L_max, 1] + [B] labels + [B] lengths
      ↓ pack_padded_sequence(features, lengths)
RNN 根据真实长度处理 → 每条序列输出一个类别预测
"""
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
VARIABLE_LENGTH = False  # 改为 True：随机截短序列，练习可变长度 RNN
MIN_SEQUENCE_LENGTH = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "rnn_synthetic_control"
TRAIN_PATH = DATA_DIR / "SyntheticControl_TRAIN.txt"
TEST_PATH = DATA_DIR / "SyntheticControl_TEST.txt"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class SequenceDataset(Dataset):
    def __init__(self, features, labels, variable_length=False, seed=SEED):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        if variable_length:
            rng = np.random.default_rng(seed)
            self.lengths = rng.integers(
                MIN_SEQUENCE_LENGTH,
                self.features.size(1) + 1,
                size=len(self.labels),
            )
        else:
            self.lengths = np.full(len(self.labels), self.features.size(1))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        length = int(self.lengths[index])
        sequence = self.features[index, :length].unsqueeze(-1)
        return sequence, self.labels[index], length


def collate_sequences(batch):
    sequences, labels, lengths = zip(*batch)
    padded = pad_sequence(list(sequences), batch_first=True)
    return padded, torch.stack(labels), torch.tensor(lengths, dtype=torch.long)


def load_ucr_file(path):
    raw = np.loadtxt(path, dtype=np.float32)
    labels = raw[:, 0].astype(np.int64) - 1
    features = raw[:, 1:]
    return features, labels


# ===== 进阶练习：原来的手写分层划分（当前不执行） =====
# def stratified_train_val_split(labels, val_ratio=0.2):
#     rng = np.random.default_rng(SEED)
#     train_indices, val_indices = [], []
#     for label in np.unique(labels):
#         indices = np.where(labels == label)[0]
#         rng.shuffle(indices)
#         val_count = max(1, int(len(indices) * val_ratio))
#         val_indices.extend(indices[:val_count])
#         train_indices.extend(indices[val_count:])
#     rng.shuffle(train_indices)
#     rng.shuffle(val_indices)
#     return np.array(train_indices), np.array(val_indices)


def build_dataloaders(variable_length=False):
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {TRAIN_PATH} 或 {TEST_PATH}\n"
            "请先运行：python download_data/download_06_rnn_synthetic_control.py"
        )
    all_train_x, all_train_y = load_ucr_file(TRAIN_PATH)
    test_x, test_y = load_ucr_file(TEST_PATH)
    train_x, val_x, train_y, val_y = train_test_split(
        all_train_x,
        all_train_y,
        test_size=0.20,
        random_state=SEED,
        stratify=all_train_y,
    )

    train_x = np.asarray(train_x, dtype=np.float32)
    val_x = np.asarray(val_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64)
    val_y = np.asarray(val_y, dtype=np.int64)

    mean = train_x.mean()
    std = train_x.std()
    std = std if std > 0 else 1.0
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    train_dataset = SequenceDataset(train_x, train_y, variable_length, seed=SEED)
    val_dataset = SequenceDataset(val_x, val_y, variable_length, seed=SEED + 1)
    test_dataset = SequenceDataset(test_x, test_y, variable_length, seed=SEED + 2)
    train_loader = DataLoader(
        train_dataset,
        BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        BATCH_SIZE,
        collate_fn=collate_sequences,
    )
    test_loader = DataLoader(
        test_dataset,
        BATCH_SIZE,
        collate_fn=collate_sequences,
    )
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

    def forward(self, features, lengths):
        packed = pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)
        return self.head(hidden[-1])  # [B, 6]


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels, lengths in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features, lengths)
            total_loss += loss_fn(logits, labels).item() * features.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += features.size(0)
    return total_loss / total, correct / total


def main():
    mode = "variable" if VARIABLE_LENGTH else "fixed"
    checkpoint_path = ROOT / f"best_rnn_{mode}.pt"
    set_seed()
    train_loader, val_loader, test_loader = build_dataloaders(VARIABLE_LENGTH)
    print(
        f"mode={mode} epochs={EPOCHS} "
        f"train_lengths={train_loader.dataset.lengths[:8].tolist()}"
    )
    model = RNNClassifier().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for features, labels, lengths in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(features, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * features.size(0)
            train_count += features.size(0)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss / train_count:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"mode={mode} test_loss={test_loss:.4f} test_acc={test_acc:.3f}")
    features, labels, lengths = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        predictions = model(features[:8].to(DEVICE), lengths[:8]).argmax(dim=1).cpu()
    print("lengths:", lengths[:8].tolist())
    print("pred:", predictions.tolist())
    print("true:", labels[:8].tolist())


if __name__ == "__main__":
    main()
