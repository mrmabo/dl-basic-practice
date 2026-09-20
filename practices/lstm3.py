import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "lstm_air_passengers" / "airline-passengers.csv"
CHECKPOINT = ROOT / "lstm3.pt"
LOOKBACK = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300


def load_series():

    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    months = [row["Month"] for row in rows]
    values = np.array([row["Passengers"] for row in rows], dtype=np.float32)
    return months, values


class WindowsDataset(Dataset):
    def __init__(self, values, lookback=LOOKBACK) -> None:
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.values) - self.lookback

    def __getitem__(self, index) -> Any:
        features = self.values[index : index + self.lookback].unsqueeze(-1)
        targets = self.values[index + self.lookback].unsqueeze(-1)
        return features, targets


def build_dataloaders():
    months, values = load_series()

    train_raw, temp_raw, _, temp_months = train_test_split(
        values, months, train_size=0.7, shuffle=False
    )

    val_raw, test_raw, _, test_months = train_test_split(
        temp_raw, temp_months, train_size=0.5, shuffle=False
    )

    mean = train_raw.mean()
    std = train_raw.std()

    train_norm = (train_raw - mean) / std
    val_norm = (val_raw - mean) / std
    test_norm = (test_raw - mean) / std

    val_with_context = np.concatenate([train_norm[-LOOKBACK:], val_norm])
    # ValueError: zero-dimensional arrays cannot be concatenated 因为缺少个:
    test_with_context = np.concatenate([val_norm[-LOOKBACK:], test_norm])

    train_dataset = WindowsDataset(train_norm)
    val_dataset = WindowsDataset(val_with_context)
    test_dataset = WindowsDataset(test_with_context)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    return train_dataloader, val_dataloader, test_dataloader, test_months, mean, std


class LSTMForcast(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=10, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, values):
        output, _ = self.lstm(values)
        return self.head(output[:, -1, :])


def run_epoch(model, data_loader, loss_fn, optimizer=None):
    isTraining = optimizer is not None
    model.train(isTraining)
    total_loss = 0.0

    with torch.set_grad_enabled(isTraining):
        for values, targets in data_loader:
            values = values.to(DEVICE)
            targets = targets.to(DEVICE)

            if isTraining:
                optimizer.zero_grad()

            pred = model(values)
            mse = loss_fn(pred, targets)

            if isTraining:
                mse.backward()
                optimizer.step()

            total_loss += mse.item() * targets.size(0)

    return total_loss / len(data_loader.dataset)


def main():
    train_dataloader, val_dataloader, test_dataloader, test_months, mean, std = (
        build_dataloaders()
    )
    model = LSTMForcast().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_mse = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_mse = run_epoch(model, train_dataloader, loss_fn, optimizer)
        val_mse = run_epoch(model, val_dataloader, loss_fn)

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} train_mse={train_mse:.4f} val_mse={val_mse:.4f}")

        if best_val_mse > val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), CHECKPOINT)

    test_mse = run_epoch(model, test_dataloader, loss_fn)
    print(f"test_mse={test_mse:.4f}")

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )

    values, targets = next(iter(test_dataloader))

    model.eval()
    with torch.no_grad():
        pred = model(values[5:].to(DEVICE)).squeeze(1).cpu().numpy()

    pred = pred * std + mean
    actual = targets[5:].squeeze(1).numpy() * std + mean

    for month, predication, true in zip(test_months[5:], pred, actual):
        print(f"Month={month} predication={predication:.1f} true={true:.1f}")


main()
