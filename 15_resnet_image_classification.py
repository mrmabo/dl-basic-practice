"""Stage 2-15: hand-written ResNet-18 with a CIFAR-10 input stem."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

# config
SEED = 42
BATCH_SIZE = 32
EPOCHS = int(os.getenv("EPOCHS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "resnet_cifar10"
CHECKPOINT = ROOT / "checkpoints" / "15_resnet18.pt"


# dataloader
def build_dataloaders():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_data = CIFAR10(
        DATA_DIR, train=True, transform=train_transform, download=False
    )
    eval_data = CIFAR10(DATA_DIR, train=True, transform=eval_transform, download=False)
    train_set, val_set = random_split(
        train_data, [45000, 5000], generator=torch.Generator().manual_seed(SEED)
    )
    val_set.dataset = eval_data
    test_set = CIFAR10(DATA_DIR, train=False, transform=eval_transform, download=False)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)
    test_loader = DataLoader(test_set, BATCH_SIZE)
    return train_loader, val_loader, test_loader


# model
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, features):
        identity = self.shortcut(features)  # [B,C_out,H_out,W_out]
        hidden = self.conv1(features)  # [B,C_out,H_out,W_out]
        hidden = self.bn1(hidden)  # [B,C_out,H_out,W_out]
        hidden = self.relu(hidden)  # [B,C_out,H_out,W_out]
        hidden = self.conv2(hidden)  # [B,C_out,H_out,W_out]
        hidden = self.bn2(hidden)  # [B,C_out,H_out,W_out]
        hidden = hidden + identity  # [B,C_out,H_out,W_out]
        hidden = self.relu(hidden)  # [B,C_out,H_out,W_out]
        return hidden


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

    def forward(self, images):
        hidden = self.stem(images)  # [B,64,32,32]
        hidden = self.layer1(hidden)  # [B,64,32,32]
        hidden = self.layer2(hidden)  # [B,128,16,16]
        hidden = self.layer3(hidden)  # [B,256,8,8]
        hidden = self.layer4(hidden)  # [B,512,4,4]
        hidden = self.pool(hidden)  # [B,512,1,1]
        hidden = hidden.flatten(1)  # [B,512]
        logits = self.head(hidden)  # [B,10]
        return logits


# training and evaluation
def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, (images, labels) in enumerate(loader, 1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch = images.size(0)
            total_loss += loss.item() * batch
            correct += (logits.argmax(1) == labels).sum().item()
            count += batch
            if training and (step == 1 or step % 100 == 0):
                print(
                    f"  step={step}/{len(loader)} loss={total_loss / count:.4f}",
                    flush=True,
                )
    return total_loss / count, correct / count


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_loader, val_loader, test_loader = build_dataloaders()
    model = ResNet18().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    best = float("inf")
    print(
        f"device={DEVICE} train={len(train_loader.dataset)} val={len(val_loader.dataset)}",
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
        scheduler.step()
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            flush=True,
        )
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_acc = run_epoch(model, test_loader, loss_fn)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")
    images, labels = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        logits = model(images[:5].to(DEVICE))
        predictions = logits.argmax(1).cpu()
    classes = test_loader.dataset.classes
    for predicted, actual in zip(predictions.tolist(), labels[:5].tolist()):
        print(f"pred={classes[predicted]} true={classes[actual]}")


if __name__ == "__main__":
    main()
