"""Stage 2-08: study U-Net Figure 1 on Oxford-IIIT Pet segmentation.

Goal: follow every tensor shape in the original 2015 architecture:
[B, 1, 572, 572] -> [B, 2, 388, 388].
Paper: https://arxiv.org/abs/1505.04597 (Figure 1 and Section 2).

Images and trimaps are resized together to 572x572; images become grayscale.
Only the central 388x388 mask is supervised, because valid convolutions remove
context at the edges. Never resize the full mask to the smaller output size.
Pet/border = foreground (1), background = 0; these are pixel classes, not breeds.

This is an architecture practice on a different dataset, not a full reproduction
of the biomedical experiments. It retains Adam and unweighted cross entropy;
elastic augmentation, cell-separation weight maps and overlap-tile inference
are additional paper topics. Inference here predicts only the central region.
Old 128x128 checkpoints are incompatible, so this version uses a new filename.
"""

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
IMAGE_SIZE = 572
OUTPUT_SIZE = 388
BATCH_SIZE = 1
EPOCHS = int(os.getenv("EPOCHS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "unet_oxford_pet"
CHECKPOINT = ROOT / "checkpoints" / "08_unet_paper_shape.pt"


# dataloader: image [1,572,572], target [388,388] with integer class IDs.
def center_crop(tensor, height, width):
    """Crop the last two dimensions; works for masks and batched feature maps."""
    source_height, source_width = tensor.shape[-2:]
    if source_height < height or source_width < width:
        raise ValueError("Cannot center-crop a tensor to a larger spatial size.")
    top = (source_height - height) // 2
    left = (source_width - width) // 2
    return tensor[..., top : top + height, left : left + width]


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
        image = image.convert("L").resize(output_size, Image.Resampling.BILINEAR)
        mask = mask.resize(output_size, Image.Resampling.NEAREST)
        image = to_tensor(image)
        # Original trimaps: 1=pet, 2=background, 3=border. Treat pet and border as foreground.
        mask = (pil_to_tensor(mask).squeeze(0) != 2).long()
        mask = center_crop(mask, OUTPUT_SIZE, OUTPUT_SIZE)  # [92:480, 92:480]
        return image, mask


def conv_block(in_channels, out_channels):
    """Two 3x3 convolutions and ReLUs, as in each block of the original U-Net."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Original Figure 1 channels, valid convolutions and cropped skip tensors.

    Each 3x3 convolution removes two pixels per spatial dimension.
    Each block therefore removes four; pooling halves, up-convolution doubles.
    The output contains two class logits per pixel; CrossEntropyLoss applies
    log-softmax internally, so forward must return raw logits.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = conv_block(1, 64)
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
        self.head = nn.Conv2d(64, 2, kernel_size=1)
        # Section 3: Gaussian initialization with std=sqrt(2/fan_in).
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images):
        if images.shape[1:] != (1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Expected grayscale input [B, 1, 572, 572].")
        # Two valid convolutions in each encoder block:
        skip1 = self.enc1(images)             # 572 -> 570 -> 568; [B,64,568,568]
        skip2 = self.enc2(self.pool(skip1))   # 284 -> 282 -> 280; [B,128,280,280]
        skip3 = self.enc3(self.pool(skip2))   # 140 -> 138 -> 136; [B,256,136,136]
        skip4 = self.enc4(self.pool(skip3))   #  68 ->  66 ->  64; [B,512,64,64]
        hidden = self.bottleneck(self.pool(skip4))  # 32 -> 30 -> 28; [B,1024,28,28]

        # Up-convolution halves channels; concatenate cropped encoder features.
        hidden = self.up4(hidden)             # [B,512,56,56]
        cropped = center_crop(skip4, 56, 56)  # 64 -> 56; remove 4 on each side
        hidden = self.dec4(torch.cat([hidden, cropped], dim=1))  # 1024 channels -> [B,512,52,52]

        hidden = self.up3(hidden)             # [B,256,104,104]
        cropped = center_crop(skip3, 104, 104) # 136 -> 104; remove 16 each side
        hidden = self.dec3(torch.cat([hidden, cropped], dim=1))  # 512 channels -> [B,256,100,100]

        hidden = self.up2(hidden)             # [B,128,200,200]
        cropped = center_crop(skip2, 200, 200) # 280 -> 200; remove 40 each side
        hidden = self.dec2(torch.cat([hidden, cropped], dim=1))  # 256 channels -> [B,128,196,196]

        hidden = self.up1(hidden)             # [B,64,392,392]
        cropped = center_crop(skip1, 392, 392) # 568 -> 392; remove 88 each side
        hidden = self.dec1(torch.cat([hidden, cropped], dim=1))  # 128 channels -> [B,64,388,388]
        return self.head(hidden)              # 1x1 conv -> [B,2,388,388]


def dice_score(logits, targets):
    predictions = (logits.argmax(dim=1) == 1).float()
    targets = (targets == 1).float()
    intersection = (predictions * targets).sum(dim=(1, 2))
    denominator = predictions.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
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
    loss_fn = nn.CrossEntropyLoss()
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
        predicted_masks = model(images[:2].to(DEVICE)).argmax(dim=1).cpu()
    print(f"test_loss={test_loss:.4f} test_dice={test_dice:.3f}")
    print("inference shapes:", images[:2].shape, masks[:2].shape, predicted_masks.shape)


if __name__ == "__main__":
    main()
