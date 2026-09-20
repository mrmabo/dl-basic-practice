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
CHECKPOINT = ROOT / "lstm.pt"
LOOKBACK = 12
BATCH_SIZE = 16
EPOCHS = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_series():
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    months = [row["Month"] for row in rows]
    # 问题1，这里需要将values 转换成array，并且是float32
    values = np.array([row["Passengers"] for row in rows], dtype=np.float32)

    return months, values


class WindowDateset(Dataset):
    def __init__(self, values, lookback=LOOKBACK) -> None:
        super().__init__()
        # 问题2，这里需要float32
        self.values = torch.tensor(values, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.values) - self.lookback

    def __getitem__(self, index) -> Any:
        # 问题3： unsqueeze(-1) 是将[12] -> [12, 1], 在形状后面加一个为1的维度，因为lstm需要的形状是[Batch_number, seq_length, features_numbers]
        features = self.values[index : index + self.lookback].unsqueeze(-1)
        target = self.values[index + self.lookback].unsqueeze(-1)
        return features, target


def build_dataloaders():
    months, raw_values = load_series()

    train_raw, temp_raw, _, temp_months = train_test_split(
        raw_values, months, train_size=0.7, shuffle=False
    )

    val_raw, test_raw, _, test_months = train_test_split(
        temp_raw, temp_months, train_size=0.5, shuffle=False
    )

    mean = train_raw.mean()
    std = train_raw.std()

    train_norm = (train_raw - mean) / std
    val_norm = (val_raw - mean) / std
    test_norm = (test_raw - mean) / std

    # 问题4： val和test数据集需要之前lookback长度的context。
    val_with_context = np.concatenate([train_norm[-LOOKBACK:], val_norm])
    test_with_context = np.concatenate([val_norm[-LOOKBACK:], test_norm])

    train_dataset = WindowDateset(train_norm)
    val_dataset = WindowDateset(val_with_context)
    test_dataset = WindowDateset(test_with_context)

    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE)

    return train_dataloader, val_dataloader, test_dataloader, test_months, mean, std


class LSTMForecaster(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size=32, batch_first=True, num_layers=10)
        self.forecaster_head = nn.Linear(32, 1)

    def forward(self, features):
        # lstm 模型的形状变换，output[:, -1, :] -> [16, 32]
        output, _ = self.lstm(features)  # [16, 12, 1] -> [16, 12, 32]
        return self.forecaster_head(output[:, -1, :])  # output[:, -1, :] -> [16, 32]


def run_epoch(model, data_loader, loss_fn, optimizer=None):
    isTraining = optimizer is not None
    model.train(isTraining)
    total_loss = 0.0

    with torch.set_grad_enabled(isTraining):
        for features, target in data_loader:
            features = features.to(DEVICE)
            target = target.to(DEVICE)
            if isTraining:
                optimizer.zero_grad()

            legits = model(features)
            loss = loss_fn(legits, target)

            # 问题：又一次忘记了更新训练权重
            if isTraining:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * features.size(0)

    return total_loss / len(data_loader.dataset)


def main():
    train_loader, val_loader, test_loader, test_months, mean, std = build_dataloaders()

    model = LSTMForecaster().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_mse = run_epoch(model, train_loader, loss_fn, optimizer)
        val_mse = run_epoch(model, val_loader, loss_fn)

        if best_val_loss > val_mse:
            best_val_loss = val_mse
            torch.save(model.state_dict(), CHECKPOINT)

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} train_mse={train_mse:.4f} val_mse={val_mse:.4f}")

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )

    test_mse = run_epoch(model, test_loader, loss_fn)
    print(f"test_mse={test_mse:.4f}")

    features, targets = next(iter(test_loader))

    model.eval()
    with torch.no_grad():
        # 问题：由于在创建dateset的增加一个维度，在预测的时候应该将这个维度删除
        pred = model(features[:5].to(DEVICE)).squeeze(1).numpy()
    pred = pred * std + mean
    actual = targets[:5].squeeze(1).numpy() * std + mean
    for month, prediction, true_value in zip(test_months, pred, actual):
        print(f"month={month} pred={prediction:.1f} true={true_value:.1f}")


main()
