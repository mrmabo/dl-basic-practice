"""完整流程 2：使用 MNIST 公开数据训练 CNN 图像分类模型。"""

import gzip
import os
import random
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


def read_idx_images(path):
    with gzip.open(path, "rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path.name} 不是有效的 MNIST 图像文件")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data.reshape(count, rows, columns).copy()


def read_idx_labels(path):
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path.name} 不是有效的 MNIST 标签文件")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if len(labels) != count:
        raise ValueError(f"{path.name} 中的标签数量不正确")
    return labels.copy()


class MNISTDataset(Dataset):
    def __init__(self, images, labels, mean, std):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index].float().unsqueeze(0) / 255.0
        image = (image - self.mean) / self.std
        return image, self.labels[index]


def build_dataloaders():
    paths = {
        "train_images": DATA_DIR / "train-images-idx3-ubyte.gz",
        "train_labels": DATA_DIR / "train-labels-idx1-ubyte.gz",
        "test_images": DATA_DIR / "t10k-images-idx3-ubyte.gz",
        "test_labels": DATA_DIR / "t10k-labels-idx1-ubyte.gz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少 MNIST 文件：\n"
            + "\n".join(missing)
            + "\n请先运行：python download_02_cnn_mnist.py"
        )

    all_train_images = read_idx_images(paths["train_images"])
    all_train_labels = read_idx_labels(paths["train_labels"])
    test_images = read_idx_images(paths["test_images"])
    test_labels = read_idx_labels(paths["test_labels"])

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(all_train_labels))
    val_indices = indices[:5000]
    train_indices = indices[5000:]
    train_images = all_train_images[train_indices]
    train_labels = all_train_labels[train_indices]
    val_images = all_train_images[val_indices]
    val_labels = all_train_labels[val_indices]

    train_float = train_images.astype(np.float32) / 255.0
    mean = float(train_float.mean())
    std = float(train_float.std())

    train_dataset = MNISTDataset(train_images, train_labels, mean, std)
    val_dataset = MNISTDataset(val_images, val_labels, mean, std)
    test_dataset = MNISTDataset(test_images, test_labels, mean, std)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)
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
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss / len(train_loader.dataset):.4f} "
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
