"""完整流程 5：使用 WDBC 训练普通或 Denoising Autoencoder 异常检测模型。"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
BATCH_SIZE = 32
EPOCHS = int(os.getenv("EPOCHS", "300"))
DENOISING = os.getenv("DENOISING", "0") == "1"
NOISE_STD = float(os.getenv("NOISE_STD", "0.1"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "autoencoder_breast_cancer" / "wdbc.data"
MODE = "denoising" if DENOISING else "standard"
CHECKPOINT_PATH = ROOT / f"best_autoencoder_{MODE}.pt"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class BreastCancerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {DATA_PATH}\n"
            "请先运行：python download_data/download_05_autoencoder_breast_cancer.py"
        )
    raw = np.genfromtxt(DATA_PATH, delimiter=",", dtype=str)
    features = raw[:, 2:].astype(np.float32)
    labels = (raw[:, 1] == "M").astype(np.int64)  # benign=0, malignant=1
    return features, labels


def build_dataloaders():
    features, labels = load_data()

    # ===== 进阶练习：原来的手写异常检测划分（当前不执行） =====
    # rng = np.random.default_rng(SEED)
    # normal_indices = np.where(labels == 0)[0]
    # anomaly_indices = np.where(labels == 1)[0]
    # rng.shuffle(normal_indices)
    # rng.shuffle(anomaly_indices)
    # train_end = int(len(normal_indices) * 0.60)
    # val_end = int(len(normal_indices) * 0.80)
    # train_idx = normal_indices[:train_end]
    # val_idx = normal_indices[train_end:val_end]
    # test_idx = np.concatenate([normal_indices[val_end:], anomaly_indices])
    # rng.shuffle(test_idx)

    normal_features = features[labels == 0]
    normal_labels = labels[labels == 0]
    anomaly_features = features[labels == 1]
    anomaly_labels = labels[labels == 1]

    # Autoencoder 只用正常样本训练和确定阈值；异常样本全部留到测试集。
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        normal_features,
        normal_labels,
        train_size=0.60,
        random_state=SEED,
    )
    val_features, normal_test_features, val_labels, normal_test_labels = (
        train_test_split(
            temp_features,
            temp_labels,
            train_size=0.50,
            random_state=SEED,
        )
    )
    test_features = np.concatenate([normal_test_features, anomaly_features], axis=0)
    test_labels = np.concatenate([normal_test_labels, anomaly_labels], axis=0)
    test_features, test_labels = shuffle(
        test_features,
        test_labels,
        random_state=SEED,
    )

    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    test_features = (test_features - mean) / std

    train_dataset = BreastCancerDataset(train_features, train_labels)
    val_dataset = BreastCancerDataset(val_features, val_labels)
    test_dataset = BreastCancerDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
    return train_loader, val_loader, test_loader


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU()
        )
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 30))

    def forward(self, features):
        return self.decoder(self.encoder(features))  # [B, 30] -> [B, 8] -> [B, 30]


def reconstruction_scores(model, loader):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            scores = ((model(features) - features) ** 2).mean(dim=1)
            all_scores.append(scores.cpu())
            all_labels.append(labels)
    return torch.cat(all_scores), torch.cat(all_labels)


def main():
    set_seed()
    train_loader, val_loader, test_loader = build_dataloaders()
    model = Autoencoder().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for features, _ in train_loader:
            features = features.to(DEVICE)
            model_inputs = features
            if DENOISING:
                model_inputs = features + torch.randn_like(features) * NOISE_STD
            optimizer.zero_grad()
            loss = loss_fn(model(model_inputs), features)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        val_scores, _ = reconstruction_scores(model, val_loader)
        val_loss = val_scores.mean().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch:02d} "
                f"train_mse={train_loss / len(train_loader.dataset):.5f} "
                f"val_mse={val_loss:.5f}"
            )

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    print(f"mode={MODE} noise_std={NOISE_STD if DENOISING else 0.0}")
    val_scores, _ = reconstruction_scores(model, val_loader)
    threshold = torch.quantile(val_scores, 0.75)
    test_scores, labels = reconstruction_scores(model, test_loader)
    predictions = (test_scores > threshold).long()

    true_positive = ((predictions == 1) & (labels == 1)).sum()
    false_positive = ((predictions == 1) & (labels == 0)).sum()
    false_negative = ((predictions == 0) & (labels == 1)).sum()
    accuracy = (predictions == labels).float().mean()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    print(
        f"threshold={threshold:.5f} accuracy={accuracy:.3f} "
        f"precision={precision:.3f} recall={recall:.3f}"
    )
    print("scores:", test_scores[:10].round(decimals=3).tolist())
    print("pred:", predictions[:10].tolist())
    print("true:", labels[:10].tolist())


if __name__ == "__main__":
    main()
