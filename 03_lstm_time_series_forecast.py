"""完整流程 3：使用 AirPassengers 公开数据训练 LSTM 一步预测模型。"""

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
LOOKBACK = 12
BATCH_SIZE = 16
EPOCHS = int(os.getenv("EPOCHS", "100"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "lstm_air_passengers" / "airline-passengers.csv"
CHECKPOINT_PATH = ROOT / "best_lstm.pt"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_series():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {DATA_PATH}\n"
            "请先运行：python download_data/download_03_lstm_air_passengers.py"
        )
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    months = [row["Month"] for row in rows]
    values = np.array([float(row["Passengers"]) for row in rows], dtype=np.float32)
    return months, values


class WindowDataset(Dataset):
    def __init__(self, values, lookback=LOOKBACK):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.values) - self.lookback

    def __getitem__(self, index):
        features = self.values[index : index + self.lookback].unsqueeze(-1)
        target = self.values[index + self.lookback].unsqueeze(-1)
        return features, target  # [L, 1], [1]


def build_dataloaders():
    months, raw = load_series()

    # ===== 进阶练习：原来的手写时间边界（当前不执行） =====
    # train_end = int(len(raw) * 0.70)
    # val_end = int(len(raw) * 0.85)
    # train_raw = raw[:train_end]
    # val_raw = raw[train_end:val_end]
    # test_raw = raw[val_end:]
    # test_months = months[val_end:]

    train_raw, temp_raw, _, temp_months = train_test_split(
        raw,
        months,
        train_size=0.70,
        shuffle=False,
    )
    val_raw, test_raw, _, test_months = train_test_split(
        temp_raw,
        temp_months,
        train_size=0.50,
        shuffle=False,
    )

    # 时间序列不能随机打乱；标准化统计量也只能来自较早的训练区间。
    mean = train_raw.mean()
    std = train_raw.std()
    train_normalized = (train_raw - mean) / std
    val_normalized = (val_raw - mean) / std
    test_normalized = (test_raw - mean) / std

    # 验证集和测试集各补上前一段的 LOOKBACK 个历史点来构造第一个窗口。
    val_with_context = np.concatenate([train_normalized[-LOOKBACK:], val_normalized])
    test_with_context = np.concatenate([val_normalized[-LOOKBACK:], test_normalized])
    train_dataset = WindowDataset(train_normalized)
    val_dataset = WindowDataset(val_with_context)
    test_dataset = WindowDataset(test_with_context)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
    return train_loader, val_loader, test_loader, test_months, mean, std


class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, features):
        output, _ = self.lstm(features)  # [B, L, 1] -> [B, L, 32]
        return self.head(output[:, -1])  # [B, 1]


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            total_loss += loss_fn(model(features), targets).item() * features.size(0)
    return total_loss / len(loader.dataset)


def main():
    set_seed()
    train_loader, val_loader, test_loader, test_months, mean, std = build_dataloaders()
    model = LSTMForecaster().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        val_loss = evaluate(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} "
                f"train_mse={train_loss / len(train_loader.dataset):.5f} "
                f"val_mse={val_loss:.5f}"
            )

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    print(f"test_mse(normalized)={evaluate(model, test_loader, loss_fn):.5f}")

    features, targets = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        predictions = model(features[:5].to(DEVICE)).cpu().squeeze(1).numpy()
    predictions = predictions * std + mean
    actual = targets[:5].squeeze(1).numpy() * std + mean
    for month, prediction, true_value in zip(test_months, predictions, actual):
        print(f"month={month} pred={prediction:.1f} true={true_value:.1f}")


if __name__ == "__main__":
    main()
