"""完整流程 1：使用 UCI Wine 公开数据训练 MLP 分类或回归模型。"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
BATCH_SIZE = 32
EPOCHS = int(os.getenv("EPOCHS", "30"))
LEARNING_RATE = 1e-3
TASK = os.getenv("TASK", "classification").lower()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "mlp_wine" / "wine.data"
CHECKPOINT_PATH = ROOT / f"best_mlp_{TASK}.pt"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WineDataset(Dataset):
    def __init__(self, features, targets, task=TASK):
        self.features = torch.tensor(features, dtype=torch.float32)
        target_dtype = torch.long if task == "classification" else torch.float32
        self.targets = torch.tensor(targets, dtype=target_dtype)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


# ===== 进阶练习：原来的手写分层划分（当前不执行） =====
# def stratified_split(labels, train_ratio=0.6, val_ratio=0.2):
#     """按类别划分索引，避免小数据集的某个集合缺少某一类别。"""
#     rng = np.random.default_rng(SEED)
#     train_indices, val_indices, test_indices = [], [], []
#
#     for label in np.unique(labels):
#         class_indices = np.where(labels == label)[0]
#         rng.shuffle(class_indices)
#         train_end = int(len(class_indices) * train_ratio)
#         val_end = train_end + int(len(class_indices) * val_ratio)
#         train_indices.extend(class_indices[:train_end])
#         val_indices.extend(class_indices[train_end:val_end])
#         test_indices.extend(class_indices[val_end:])
#
#     rng.shuffle(train_indices)
#     rng.shuffle(val_indices)
#     rng.shuffle(test_indices)
#     return (
#         np.array(train_indices),
#         np.array(val_indices),
#         np.array(test_indices),
#     )


def build_dataloaders():
    if TASK not in {"classification", "regression"}:
        raise ValueError("TASK 必须是 classification 或 regression")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {DATA_PATH}\n"
            "请先运行：python download_data/download_01_mlp_wine.py"
        )

    raw = np.loadtxt(DATA_PATH, delimiter=",", dtype=np.float32)
    if TASK == "classification":
        targets = raw[:, 0].astype(np.int64) - 1
        features = raw[:, 1:]
        stratify_targets = targets
    else:
        # 回归任务使用其余12个属性预测连续的 alcohol 含量。
        targets = raw[:, 1]
        features = raw[:, 2:]
        stratify_targets = None

    # 先划分出 60% 训练集，再把剩余 40% 平分为验证集和测试集。
    # stratify 保证三个集合中的类别比例尽量与完整数据一致。
    train_features, temp_features, train_targets, temp_targets = train_test_split(
        features,
        targets,
        train_size=0.60,
        random_state=SEED,
        stratify=stratify_targets,
    )
    val_features, test_features, val_targets, test_targets = train_test_split(
        temp_features,
        temp_targets,
        train_size=0.50,
        random_state=SEED,
        stratify=temp_targets if TASK == "classification" else None,
    )

    # sklearn的返回类型兼容多种array-like；显式转换后Pylance和后续计算
    # 都能确定这里使用的是NumPy数组。
    train_features = np.asarray(train_features, dtype=np.float32)
    val_features = np.asarray(val_features, dtype=np.float32)
    test_features = np.asarray(test_features, dtype=np.float32)
    train_targets = np.asarray(train_targets)
    val_targets = np.asarray(val_targets)
    test_targets = np.asarray(test_targets)

    # 只使用训练集统计量标准化，避免验证集和测试集信息泄漏。
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1.0
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    test_features = (test_features - mean) / std

    target_mean, target_std = 0.0, 1.0
    if TASK == "regression":
        target_mean = float(train_targets.mean())
        target_std = float(train_targets.std())
        target_std = target_std if target_std > 0 else 1.0
        train_targets = (train_targets - target_mean) / target_std
        val_targets = (val_targets - target_mean) / target_std
        test_targets = (test_targets - target_mean) / target_std

    train_dataset = WineDataset(train_features, train_targets)
    val_dataset = WineDataset(val_features, val_targets)
    test_dataset = WineDataset(test_features, test_targets)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
    return (
        train_loader,
        val_loader,
        test_loader,
        features.shape[1],
        target_mean,
        target_std,
    )


class MLP(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_features),
        )

    def forward(self, features):
        return self.network(features)  # [B, 13] -> [B, 3]


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    metric_total = 0.0
    total = 0

    with torch.set_grad_enabled(training):
        for features, targets in loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            if training:
                optimizer.zero_grad()
            outputs = model(features)
            if TASK == "regression":
                outputs = outputs.squeeze(1)
            loss = loss_fn(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * features.size(0)
            if TASK == "classification":
                metric_total += (outputs.argmax(dim=1) == targets).sum().item()
            else:
                metric_total += (outputs - targets).abs().sum().item()
            total += features.size(0)

    return total_loss / total, metric_total / total


def main():
    set_seed()
    (
        train_loader,
        val_loader,
        test_loader,
        input_features,
        target_mean,
        target_std,
    ) = build_dataloaders()
    output_features = 3 if TASK == "classification" else 1
    model = MLP(input_features, output_features).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss() if TASK == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metric = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_metric = run_epoch(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        metric_name = "acc" if TASK == "classification" else "mae(normalized)"
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"train_{metric_name}={train_metric:.3f} val_loss={val_loss:.4f} "
            f"val_{metric_name}={val_metric:.3f}"
        )

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_metric = run_epoch(model, test_loader, loss_fn)
    if TASK == "classification":
        print(f"test_loss={test_loss:.4f} test_acc={test_metric:.3f}")
    else:
        print(
            f"test_mse(normalized)={test_loss:.4f} "
            f"test_mae={test_metric * target_std:.3f}"
        )

    features, targets = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        outputs = model(features[:5].to(DEVICE)).cpu()
    if TASK == "classification":
        predictions = outputs.argmax(dim=1)
        print("pred:", predictions.tolist())
        print("true:", targets[:5].tolist())
    else:
        predictions = outputs.squeeze(1).numpy() * target_std + target_mean
        actual = targets[:5].numpy() * target_std + target_mean
        print("pred_alcohol:", predictions.round(3).tolist())
        print("true_alcohol:", actual.round(3).tolist())


if __name__ == "__main__":
    main()
