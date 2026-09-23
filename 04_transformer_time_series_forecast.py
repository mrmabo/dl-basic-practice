"""完整流程 4：使用 ETTh1 训练 Transformer 多步、多目标预测模型。"""

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
LOOKBACK = 96
HORIZON = 24
BATCH_SIZE = 64
EPOCHS = int(os.getenv("EPOCHS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_NAMES = ["HUFL", "OT"]
TARGET_INDICES = [FEATURE_NAMES.index(name) for name in TARGET_NAMES]
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
    return timestamps, np.asarray(
        rows, dtype=np.float32
    )  # timestamps: [N]；数据: [N, 7]


class ETTH1Dataset(Dataset):
    def __init__(
        self,
        data,
        lookback=LOOKBACK,
        horizon=HORIZON,
        target_indices=TARGET_INDICES,
    ):
        self.data = torch.tensor(data, dtype=torch.float32)  # [N, 7] -> [N, 7]
        self.lookback = lookback
        self.horizon = horizon
        self.target_indices = target_indices

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, index):
        forecast_start = index + self.lookback
        forecast_end = forecast_start + self.horizon
        features = self.data[index:forecast_start]  # [N, 7] -> [LOOKBACK, 7]
        targets = self.data[  # [N, 7] -> [HORIZON, num_targets]
            forecast_start:forecast_end, self.target_indices
        ]
        return features, targets  # [LOOKBACK, 7], [HORIZON, num_targets]


def build_dataloaders():
    timestamps, raw = load_etth1()  # timestamps: [N]；raw: [N, 7]

    # ===== 进阶练习：原来的手写时间边界（当前不执行） =====
    # train_end = int(len(raw) * 0.70)
    # val_end = int(len(raw) * 0.85)
    # train_raw = raw[:train_end]
    # val_raw = raw[train_end:val_end]
    # test_raw = raw[val_end:]
    # test_times = timestamps[val_end:]

    train_raw, temp_raw, _, temp_times = train_test_split(
        raw,
        timestamps,
        train_size=0.70,
        shuffle=False,
    )
    val_raw, test_raw, _, test_times = train_test_split(
        temp_raw,
        temp_times,
        train_size=0.50,
        shuffle=False,
    )

    train_raw = np.asarray(train_raw, dtype=np.float32)
    val_raw = np.asarray(val_raw, dtype=np.float32)
    test_raw = np.asarray(test_raw, dtype=np.float32)

    # 时间序列必须按时间顺序划分，不能使用 shuffle 或 stratify。
    mean = train_raw.mean(axis=0)  # [N_train, 7] -> [7]
    std = train_raw.std(axis=0)  # [N_train, 7] -> [7]
    std[std == 0] = 1.0  # [7] -> [7]
    train_normalized = (train_raw - mean) / std  # [N_train, 7] -> [N_train, 7]
    val_normalized = (val_raw - mean) / std  # [N_val, 7] -> [N_val, 7]
    test_normalized = (test_raw - mean) / std  # [N_test, 7] -> [N_test, 7]

    val_with_context = np.concatenate(  # [LOOKBACK, 7] + [N_val, 7]
        [train_normalized[-LOOKBACK:], val_normalized], axis=0
    )  # -> [LOOKBACK + N_val, 7]
    test_with_context = np.concatenate(  # [LOOKBACK, 7] + [N_test, 7]
        [val_normalized[-LOOKBACK:], test_normalized], axis=0
    )  # -> [LOOKBACK + N_test, 7]
    train_dataset = ETTH1Dataset(train_normalized)
    val_dataset = ETTH1Dataset(val_with_context)
    test_dataset = ETTH1Dataset(test_with_context)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
    return train_loader, val_loader, test_loader, test_times, mean, std


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_features=7,
        d_model=32,
        nhead=4,
        num_layers=2,
        horizon=HORIZON,
        num_targets=len(TARGET_NAMES),
    ):
        super().__init__()
        self.horizon = horizon
        self.num_targets = num_targets
        self.input_projection = nn.Linear(
            input_features, d_model
        )  # [..., 7] -> [..., 32]
        self.position = nn.Parameter(  # 可学习位置编码，shape: [1, LOOKBACK, 32]
            torch.randn(1, LOOKBACK, d_model) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(  # [B, L, 32] -> [B, L, 32]
            layer, num_layers=num_layers
        )
        self.head = nn.Linear(  # [B, 32] -> [B, HORIZON * num_targets]
            d_model, horizon * num_targets
        )

    def forward(self, features):
        projected = self.input_projection(features)  # [B, L, 7] -> [B, L, 32]
        position = self.position[
            :, : features.size(1)
        ]  # [1, LOOKBACK, 32] -> [1, L, 32]
        hidden = projected + position  # [B, L, 32] + [1, L, 32] -> [B, L, 32]
        encoded = self.encoder(hidden)  # [B, L, 32] -> [B, L, 32]
        last_hidden = encoded[:, -1]  # [B, L, 32] -> [B, 32]
        predictions = self.head(last_hidden)  # [B, 32] -> [B, HORIZON * num_targets]
        return predictions.reshape(  # [B, HORIZON * num_targets] -> [B, HORIZON, num_targets]
            -1, self.horizon, self.num_targets
        )


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, targets in loader:  # [B, LOOKBACK, 7], [B, HORIZON, num_targets]
            features = features.to(DEVICE)  # shape不变: [B, LOOKBACK, 7]
            targets = targets.to(DEVICE)  # shape不变: [B, HORIZON, num_targets]
            predictions = model(
                features
            )  # [B, LOOKBACK, 7] -> [B, HORIZON, num_targets]
            total_loss += loss_fn(predictions, targets).item() * features.size(
                0
            )  # 标量
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
        train_count = 0
        for (
            features,
            targets,
        ) in train_loader:  # [B, LOOKBACK, 7], [B, HORIZON, num_targets]
            features = features.to(DEVICE)  # shape不变: [B, LOOKBACK, 7]
            targets = targets.to(DEVICE)  # shape不变: [B, HORIZON, num_targets]
            optimizer.zero_grad()
            predictions = model(
                features
            )  # [B, LOOKBACK, 7] -> [B, HORIZON, num_targets]
            loss = loss_fn(predictions, targets)  # 两个同形状Tensor -> 标量MSE
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * features.size(0)
            train_count += features.size(0)

        val_loss = evaluate(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(
            f"epoch={epoch:02d} "
            f"train_mse={train_loss / train_count:.5f} "
            f"val_mse={val_loss:.5f}"
        )

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    print(f"test_mse(normalized)={evaluate(model, test_loader, loss_fn):.5f}")

    features, targets = next(iter(test_loader))  # [B, LOOKBACK, 7], [B, HORIZON, 2]
    model.eval()
    with torch.no_grad():
        predictions = (
            model(features[:1].to(DEVICE)).cpu().numpy()
        )  # [1, LOOKBACK, 7] -> [1, HORIZON, 2]

    target_mean = mean[TARGET_INDICES]  # [7] -> [2]
    target_std = std[TARGET_INDICES]  # [7] -> [2]
    predictions = (
        predictions * target_std + target_mean
    )  # [1, HORIZON, 2] -> [1, HORIZON, 2]
    actual = (
        targets[:1].numpy() * target_std + target_mean
    )  # [1, HORIZON, 2] -> [1, HORIZON, 2]

    for step, timestamp in enumerate(test_times[:HORIZON]):
        values = " ".join(
            f"pred_{name}={predictions[0, step, target_index]:.3f} "
            f"true_{name}={actual[0, step, target_index]:.3f}"
            for target_index, name in enumerate(TARGET_NAMES)
        )
        print(f"time={timestamp} {values}")


if __name__ == "__main__":
    main()
