"""Stage 2-08: train a four-level U-Net for Oxford-IIIT Pet foreground segmentation."""

import os
import random
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.functional.classification import binary_f1_score
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import pil_to_tensor, to_tensor

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = int(os.getenv("EPOCHS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "unet_oxford_pet"
CHECKPOINT = ROOT / "checkpoints" / "08_unet_full.pt"


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

        mask = (pil_to_tensor(mask) != 2).float()
        return image, mask


def build_dataloaders():
    full_train = PetSegmentationDataset("trainval")
    train_size = len(full_train) - 500
    train_set, val_set = random_split(
        full_train, [train_size, 500], generator=torch.Generator().manual_seed(SEED)
    )
    test_set = PetSegmentationDataset("test")
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)
    test_loader = DataLoader(test_set, BATCH_SIZE)
    return train_loader, val_loader, test_loader


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, images):
        skip1 = self.enc1(images)          # [B, 64, 128, 128]
        pooled1 = self.pool(skip1)         # [B, 64, 64, 64]

        skip2 = self.enc2(pooled1)         # [B, 128, 64, 64]
        pooled2 = self.pool(skip2)         # [B, 128, 32, 32]

        skip3 = self.enc3(pooled2)         # [B, 256, 32, 32]
        pooled3 = self.pool(skip3)         # [B, 256, 16, 16]

        skip4 = self.enc4(pooled3)         # [B, 512, 16, 16]
        pooled4 = self.pool(skip4)         # [B, 512, 8, 8]

        hidden = self.bottleneck(pooled4)  # [B, 1024, 8, 8]

        upsampled4 = self.up4(hidden)      # [B, 512, 16, 16]
        concatenated4 = torch.cat([upsampled4, skip4], dim=1)  # [B, 1024, 16, 16]
        hidden = self.dec4(concatenated4)  # [B, 512, 16, 16]

        upsampled3 = self.up3(hidden)      # [B, 256, 32, 32]
        concatenated3 = torch.cat([upsampled3, skip3], dim=1)  # [B, 512, 32, 32]
        hidden = self.dec3(concatenated3)  # [B, 256, 32, 32]

        upsampled2 = self.up2(hidden)      # [B, 128, 64, 64]
        concatenated2 = torch.cat([upsampled2, skip2], dim=1)  # [B, 256, 64, 64]
        hidden = self.dec2(concatenated2)  # [B, 128, 64, 64]

        upsampled1 = self.up1(hidden)      # [B, 64, 128, 128]
        concatenated1 = torch.cat([upsampled1, skip1], dim=1)  # [B, 128, 128, 128]
        hidden = self.dec1(concatenated1)  # [B, 64, 128, 128]

        logits = self.head(hidden)        # [B, 1, 128, 128]
        return logits


def dice_score(logits, targets):

    predictions = (logits.detach().sigmoid() >= 0.5).long()
    return binary_f1_score(
        preds=predictions,
        target=targets.long(),
        multidim_average="samplewise",
        zero_division=1,
    ).mean()


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
    train_loader, val_loader, test_loader = build_dataloaders()

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
