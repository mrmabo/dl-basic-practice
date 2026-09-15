"""完整流程 4：使用 ETTh1 公开数据训练 Transformer 一步预测模型。"""

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
LOOKBACK = 96
BATCH_SIZE = 64
EPOCHS = int(os.getenv("EPOCHS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_INDEX = FEATURE_NAMES.index("OT")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transformer_etth1" / "ETTh1.csv"
CHECKPOINT_PATH = ROOT / "best_transformer.pt"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_etth1():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {DATA_PATH}\n"
            "请先运行：python download_data/download_04_transformer_etth1.py"
        )
    timestamps, rows = [], []
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            timestamps.append(row["date"])
            rows.append([float(row[name]) for name in FEATURE_NAMES])
    return timestamps, np.asarray(rows, dtype=np.float32)


class ETTH1Dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - LOOKBACK

    def __getitem__(self, index):
        features = self.data[index : index + LOOKBACK]
        target = self.data[index + LOOKBACK, TARGET_INDEX : TARGET_INDEX + 1]
        return features, target  # [L, 7], [1]


def build_dataloaders():
    timestamps, raw = load_etth1()
    train_end = int(len(raw) * 0.70)
    val_end = int(len(raw) * 0.85)
    mean = raw[:train_end].mean(axis=0)
    std = raw[:train_end].std(axis=0)
    std[std == 0] = 1.0
    normalized = (raw - mean) / std

    train_dataset = ETTH1Dataset(normalized[:train_end])
    val_dataset = ETTH1Dataset(normalized[train_end - LOOKBACK : val_end])
    test_dataset = ETTH1Dataset(normalized[val_end - LOOKBACK :])
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
    return train_loader, val_loader, test_loader, timestamps[val_end:], mean, std


class TransformerForecaster(nn.Module):
    def __init__(self, input_features=7, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(input_features, d_model)
        self.position = nn.Parameter(torch.randn(1, LOOKBACK, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, features):
        hidden = self.input_projection(features) + self.position[:, : features.size(1)]
        encoded = self.encoder(hidden)  # [B, L, 7] -> [B, L, 32]
        return self.head(encoded[:, -1])  # [B, 1]


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
    train_loader, val_loader, test_loader, test_times, mean, std = build_dataloaders()
    model = TransformerForecaster().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        val_loss = evaluate(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(
            f"epoch={epoch:02d} "
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
    predictions = predictions * std[TARGET_INDEX] + mean[TARGET_INDEX]
    actual = targets[:5].squeeze(1).numpy() * std[TARGET_INDEX] + mean[TARGET_INDEX]
    for timestamp, prediction, true_value in zip(test_times, predictions, actual):
        print(f"time={timestamp} pred_OT={prediction:.3f} true_OT={true_value:.3f}")


if __name__ == "__main__":
    main()
