"""Stage 2-11: multivariate, multi-step forecasting on the public ETTh1 dataset."""

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
HORIZON = 24
TARGET_INDICES = [1, 2]
BATCH_SIZE = 64
EPOCHS = int(os.getenv("EPOCHS", "20"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "multistep_etth1" / "ETTh1.csv"
CHECKPOINT = ROOT / "checkpoints" / "11_multistep_lstm.pt"


def load_values():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run: python download_data/download_11_etth1.py")
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    columns = [name for name in rows[0] if name != "date"]
    values = np.asarray(
        [[float(row[name]) for name in columns] for row in rows], dtype=np.float32
    )
    return columns, values


class WindowDataset(Dataset):
    def __init__(self, values):
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.values) - LOOKBACK - HORIZON + 1

    def __getitem__(self, index):
        features = self.values[index : index + LOOKBACK]
        targets = self.values[
            index + LOOKBACK : index + LOOKBACK + HORIZON, TARGET_INDICES
        ]
        return features, targets


class MultiStepLSTM(nn.Module):
    def __init__(self, input_features, num_targets):
        super().__init__()
        self.lstm = nn.LSTM(
            input_features, 64, num_layers=2, dropout=0.1, batch_first=True
        )
        self.head = nn.Linear(64, HORIZON * num_targets)
        self.num_targets = num_targets

    def forward(self, features):
        sequence, _ = self.lstm(features)  # [B, 96, 7]→[B, 96, 64]
        prediction = self.head(sequence[:, -1])  # [B, 64]→[B, 48]
        return prediction.view(-1, HORIZON, self.num_targets)


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = count = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, targets in loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(features)
            count += len(features)
    return total / count


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    columns, values = load_values()
    train_end, val_end = int(len(values) * 0.7), int(len(values) * 0.85)
    train_raw, val_raw, test_raw = (
        values[:train_end],
        values[train_end:val_end],
        values[val_end:],
    )
    mean, std = train_raw.mean(0), train_raw.std(0)
    std[std == 0] = 1
    train = (train_raw - mean) / std
    val = (np.concatenate([train_raw[-LOOKBACK:], val_raw]) - mean) / std
    test = (np.concatenate([val_raw[-LOOKBACK:], test_raw]) - mean) / std
    train_loader = DataLoader(WindowDataset(train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(val), BATCH_SIZE)
    test_loader = DataLoader(WindowDataset(test), BATCH_SIZE)

    model = MultiStepLSTM(values.shape[1], len(TARGET_INDICES)).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_mse = run_epoch(model, train_loader, loss_fn, optimizer)
        val_mse = run_epoch(model, val_loader, loss_fn)
        if val_mse < best:
            best = val_mse
            torch.save(model.state_dict(), CHECKPOINT)
        print(f"epoch={epoch:02d} train_mse={train_mse:.4f} val_mse={val_mse:.4f}")

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            predictions.append(model(features.to(DEVICE)).cpu().numpy())
            targets.append(labels.numpy())
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    target_mean, target_std = mean[TARGET_INDICES], std[TARGET_INDICES]
    predictions = predictions * target_std + target_mean
    targets = targets * target_std + target_mean
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    print("targets:", [columns[i] for i in TARGET_INDICES])
    print(f"test_mae={mae:.4f} test_rmse={rmse:.4f}")
    print("first prediction shape:", predictions[0].shape)


if __name__ == "__main__":
    main()
