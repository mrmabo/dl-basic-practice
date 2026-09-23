"""Stage 2-11: ResNet18 transfer learning on CIFAR-10."""

import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor

SEED = 42
BATCH_SIZE = 64
HEAD_EPOCHS = int(os.getenv("HEAD_EPOCHS", "3"))
FINETUNE_EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "2"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "transfer_cifar10"
CHECKPOINT = ROOT / "checkpoints" / "11_transfer_resnet18.pt"


def transforms(training):
    steps = [Resize((128, 128))]
    if training:
        steps.append(RandomHorizontalFlip())
    steps.extend([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return Compose(steps)


def build_dataloaders():
    try:
        augmented = CIFAR10(DATA_DIR, train=True, transform=transforms(True), download=False)
        evaluation = CIFAR10(DATA_DIR, train=True, transform=transforms(False), download=False)
        test_set = CIFAR10(DATA_DIR, train=False, transform=transforms(False), download=False)
    except RuntimeError as error:
        raise FileNotFoundError(
            "Run: python stage2_applied_projects/download_data/download_11_cifar10.py"
        ) from error
    indices = torch.randperm(len(augmented), generator=torch.Generator().manual_seed(SEED))
    train_indices, val_indices = indices[:-5_000], indices[-5_000:]
    train_set = torch.utils.data.Subset(augmented, train_indices)
    val_set = torch.utils.data.Subset(evaluation, val_indices)
    return (DataLoader(train_set, BATCH_SIZE, shuffle=True),
            DataLoader(val_set, BATCH_SIZE), DataLoader(test_set, BATCH_SIZE))


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images); loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item(); count += len(labels)
    return loss_sum / count, correct / count


def train_phase(model, train_loader, val_loader, epochs, optimizer, loss_fn, best):
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        if val_loss < best:
            best = val_loss; torch.save(model.state_dict(), CHECKPOINT)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
    return best


def main():
    random.seed(SEED); torch.manual_seed(SEED); CHECKPOINT.parent.mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = build_dataloaders()
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    best = train_phase(model, train_loader, val_loader, HEAD_EPOCHS, optimizer, loss_fn,
                       float("inf"))

    # Fine-tune layer4 with a smaller learning rate than the new classification head.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4},
    ])
    train_phase(model, train_loader, val_loader, FINETUNE_EPOCHS, optimizer, loss_fn, best)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    test_loss, test_acc = run_epoch(model, test_loader, loss_fn)
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        predictions = model(images[:8].to(DEVICE)).argmax(1).cpu()
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")
    print("pred:", predictions.tolist(), "true:", labels[:8].tolist())


if __name__ == "__main__":
    main()
