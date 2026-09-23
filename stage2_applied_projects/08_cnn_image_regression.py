"""Stage 2-08: use a CNN to regress the numeric value shown by an MNIST image."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

SEED = 42
BATCH_SIZE = 128
EPOCHS = int(os.getenv("EPOCHS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cnn_regression_mnist"
CHECKPOINT = ROOT / "checkpoints" / "08_cnn_regression.pt"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def build_dataloaders():
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    try:
        full_train = MNIST(DATA_DIR, train=True, transform=transform, download=False)
        test_set = MNIST(DATA_DIR, train=False, transform=transform, download=False)
    except RuntimeError as error:
        raise FileNotFoundError(
            "Run: python stage2_applied_projects/download_data/download_08_mnist.py"
        ) from error

    train_set, val_set = random_split(
        full_train,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(SEED),
    )
    return (
        DataLoader(train_set, BATCH_SIZE, shuffle=True),
        DataLoader(val_set, BATCH_SIZE),
        DataLoader(test_set, BATCH_SIZE),
    )


class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, images):
        return self.head(self.features(images).flatten(1)).squeeze(1)


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_abs_error = count = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            targets = labels.float().to(DEVICE)
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch = images.size(0)
            total_loss += loss.item() * batch
            total_abs_error += (predictions - targets).abs().sum().item()
            count += batch
    return total_loss / count, total_abs_error / count


def main():
    set_seed()
    CHECKPOINT.parent.mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = build_dataloaders()
    model = CNNRegressor().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_mse, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)
        val_mse, val_mae = run_epoch(model, val_loader, loss_fn)
        if val_mse < best:
            best = val_mse
            torch.save(model.state_dict(), CHECKPOINT)
        print(f"epoch={epoch:02d} train_mse={train_mse:.4f} train_mae={train_mae:.3f} "
              f"val_mse={val_mse:.4f} val_mae={val_mae:.3f}")

    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    test_mse, test_mae = run_epoch(model, test_loader, loss_fn)
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        predictions = model(images[:8].to(DEVICE)).cpu()
    print(f"test_rmse={test_mse ** 0.5:.3f} test_mae={test_mae:.3f}")
    print("pred:", predictions.round(decimals=2).tolist())
    print("true:", labels[:8].tolist())


if __name__ == "__main__":
    main()
