"""完整流程 2：使用 torchvision MNIST 公开数据训练 CNN 图像分类模型。"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

SEED = 42
BATCH_SIZE = 128
EPOCHS = int(os.getenv("EPOCHS", "5"))
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cnn_mnist"
CHECKPOINT_PATH = ROOT / "best_cnn.pt"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def build_dataloaders():
    transform = Compose(
        [
            ToTensor(),
            # 0.1307和0.3081来自MNIST官方训练集，不使用测试集统计量。
            Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )

    try:
        full_train_dataset = MNIST(
            root=DATA_DIR,
            train=True,
            transform=transform,
            download=False,
        )
        test_dataset = MNIST(
            root=DATA_DIR,
            train=False,
            transform=transform,
            download=False,
        )
    except RuntimeError as error:
        raise FileNotFoundError(
            "没有找到torchvision格式的MNIST数据。\n"
            "请先运行：python download_data/download_02_cnn_mnist.py"
        ) from error

    all_indices = np.arange(len(full_train_dataset))
    all_labels = full_train_dataset.targets.numpy()

    # ===== 进阶练习：手写随机划分（当前不执行） =====
    # rng = np.random.default_rng(SEED)
    # shuffled_indices = rng.permutation(all_indices)
    # val_indices = shuffled_indices[:5000]
    # train_indices = shuffled_indices[5000:]

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=5000,
        random_state=SEED,
        stratify=all_labels,
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, images):
        features = self.features(images)  # [B, 1, 28, 28] -> [B, 64, 1, 1]
        return self.classifier(features.flatten(1))  # [B, 64] -> [B, 10]


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    set_seed()
    train_loader, val_loader, test_loader = build_dataloaders()

    model = CNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss / len(train_loader.dataset):.4f} "  # type: ignore
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

    images, labels = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        predictions = model(images[:8].to(DEVICE)).argmax(dim=1).cpu()

    print("pred:", predictions.tolist())
    print("true:", labels[:8].tolist())


if __name__ == "__main__":
    main()
