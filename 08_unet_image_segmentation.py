"""Stage 2-08: train a small U-Net for Oxford-IIIT Pet foreground segmentation."""

import os
import random
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import pil_to_tensor, to_tensor

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = int(os.getenv("EPOCHS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "unet_oxford_pet"
CHECKPOINT = ROOT / "checkpoints" / "08_unet.pt"


class PetSegmentationDataset(Dataset):
    def __init__(self, split):
        try:
            self.dataset = OxfordIIITPet(
                DATA_DIR, split=split, target_types="segmentation", download=False
            )
        except RuntimeError as error:
            raise FileNotFoundError(
                "Run: python download_data/download_08_oxford_pet.py"
            ) from error

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        image = cast(Image.Image, image)
        mask = cast(Image.Image, mask)
        output_size = (IMAGE_SIZE, IMAGE_SIZE)
        image = image.resize(output_size, Image.Resampling.BILINEAR)
        mask = mask.resize(output_size, Image.Resampling.NEAREST)
        image = to_tensor(image)
        # Original trimaps: 1=pet, 2=background, 3=border. Treat pet and border as foreground.
        mask = (pil_to_tensor(mask) != 2).float()
        return image, mask


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = conv_block(3, 32)
        self.enc2 = conv_block(32, 64)
        self.bridge = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, images):
        skip1 = self.enc1(images)  # [B, 32, 128, 128]
        skip2 = self.enc2(self.pool(skip1))  # [B, 64, 64, 64]
        hidden = self.bridge(self.pool(skip2))  # [B, 128, 32, 32]
        hidden = self.dec2(torch.cat([self.up2(hidden), skip2], dim=1))
        hidden = self.dec1(torch.cat([self.up1(hidden), skip1], dim=1))
        return self.head(hidden)  # [B, 1, 128, 128]


def dice_score(logits, targets):
    predictions = (logits.sigmoid() >= 0.5).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    denominator = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = dice_sum = count = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            loss = loss_fn(logits, masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch = images.size(0)
            loss_sum += loss.item() * batch
            dice_sum += dice_score(logits, masks).item() * batch
            count += batch
    return loss_sum / count, dice_sum / count


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    full_train = PetSegmentationDataset("trainval")
    train_size = len(full_train) - 500
    train_set, val_set = random_split(
        full_train, [train_size, 500], generator=torch.Generator().manual_seed(SEED)
    )
    test_set = PetSegmentationDataset("test")
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)
    test_loader = DataLoader(test_set, BATCH_SIZE)

    model = UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_dice = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, loss_fn)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_dice={train_dice:.3f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.3f}"
        )

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_dice = run_epoch(model, test_loader, loss_fn)
    images, masks = next(iter(test_loader))
    with torch.no_grad():
        predicted_masks = (model(images[:2].to(DEVICE)).sigmoid() >= 0.5).cpu()
    print(f"test_loss={test_loss:.4f} test_dice={test_dice:.3f}")
    print("inference shapes:", images[:2].shape, masks[:2].shape, predicted_masks.shape)


if __name__ == "__main__":
    main()
