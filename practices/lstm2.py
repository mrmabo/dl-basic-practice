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
CHECKPOINT = ROOT / "lstm2.pt"
LOOKBACK = 12
EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_series():
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    months = [row["Month"] for row in rows]

    values = np.array([row["Passengers"] for row in rows], dtype=np.float32)

    return months, values


class WindowDataSet(Dataset):
    def __init__(self, values, lookback=LOOKBACK):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self):
        # 这里需要注意应该先获得values的len在减去lookback
        return len(self.values) - self.lookback

    def __getitem__(self, index) -> Any:
        # 这里的features 和target 的code需要多练习
        features = self.values[index : index + self.lookback].unsqueeze(-1)

        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1) 这个问题应该是.unsqueeze(-1) 而不是.unsqueeze(1) 需要在形状最后加一个为1的维度
        target = self.values[index + self.lookback].unsqueeze(-1)
        return features, target


def build_dataloader():
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

    # TypeError: only integer scalar arrays can be converted to a scalar index, 这个问题是np.concatenate应该传的是一个数组。
    val_with_context = np.concatenate([train_norm[-LOOKBACK:], val_norm])
    test_with_context = np.concatenate([val_norm[-LOOKBACK:], test_norm])

    train_dataset = WindowDataSet(train_norm)
    val_dataset = WindowDataSet(val_with_context)
    test_dataset = WindowDataSet(test_with_context)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    return train_dataloader, val_dataloader, test_dataloader, test_months, mean, std


class LSTMForecast(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, values):
        ## TypeError: tuple indices must be integers or slices, not tuple 这里的问题是 lstm返回的tuple type
        features, _ = self.lstm(values)
        return self.head(features[:, -1, :])


def run_epoch(model, data_loader, loss_fn, optimizer=None):
    isTraining = optimizer is not None
    model.train(isTraining)
    total_loss = 0.0

    with torch.set_grad_enabled(isTraining):
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            if isTraining:
                optimizer.zero_grad()

            pred = model(features)
            loss = loss_fn(pred, targets)

            if isTraining:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * targets.size(0)

    # 这里的应该除以data_loader.dataset的长度，在训练的时候发现mse 值很大，说明平均mse的计算有问题
    return total_loss / len(data_loader.dataset)


def main():
    train_dataloader, val_dataloader, test_dataloader, test_months, mean, std = (
        build_dataloader()
    )

    model = LSTMForecast().to(DEVICE)
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

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )

    test_mse = run_epoch(model, test_dataloader, loss_fn)
    print(f"test_mse={test_mse:.4f}")

    values, targets = next(iter(test_dataloader))

    model.eval()

    with torch.no_grad():
        # 主要要用squeeze和numpy 删除多余的维度
        pred = model(values[5:].to(DEVICE)).squeeze(1).numpy()

    # 根据mean和std，还原数据
    pred = pred * std + mean
    actual = targets[5:].squeeze(1).numpy() * std + mean

    for month, true, predication in zip(test_months, actual, pred):
        print(f"month={month} pred={predication:.1f} true={true:.1f}")


main()
