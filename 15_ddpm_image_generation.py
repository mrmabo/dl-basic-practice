"""Flow 15: train a compact DDPM to generate MNIST-like images."""

import math
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.utils import save_image

SEED = 42
BATCH_SIZE = 128
EPOCHS = int(os.getenv("EPOCHS", "5"))
TIME_STEPS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ddpm_mnist"
CHECKPOINT = ROOT / "checkpoints" / "15_ddpm.pt"
SAMPLE_PATH = ROOT / "checkpoints" / "15_ddpm_samples.png"


class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Embedding(TIME_STEPS, 64),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.down = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.middle = nn.Conv2d(64, 64, 3, padding=1)
        self.up = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.output = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, noisy_images, time_steps):
        time = self.time_embedding(time_steps)[:, :, None, None]  # [B,64,1,1]
        hidden = torch.nn.functional.silu(self.conv1(noisy_images))
        hidden = torch.nn.functional.silu(self.down(hidden))  # [B,64,14,14]
        hidden = torch.nn.functional.silu(self.middle(hidden) + time)
        hidden = torch.nn.functional.silu(self.up(hidden))  # [B,32,28,28]
        return self.output(hidden)  # predicted noise


def build_schedule():
    betas = torch.linspace(1e-4, 0.02, TIME_STEPS, device=DEVICE)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def add_noise(clean_images, time_steps, alpha_bars):
    noise = torch.randn_like(clean_images)
    alpha_bar = alpha_bars[time_steps][:, None, None, None]
    noisy_images = alpha_bar.sqrt() * clean_images + (1 - alpha_bar).sqrt() * noise
    return noisy_images, noise


def run_epoch(model, loader, optimizer, alpha_bars):
    training = optimizer is not None
    model.train(training)
    total_loss = count = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for clean_images, _ in loader:
            clean_images = clean_images.to(DEVICE)
            time_steps = torch.randint(
                0, TIME_STEPS, (len(clean_images),), device=DEVICE
            )
            noisy_images, true_noise = add_noise(clean_images, time_steps, alpha_bars)
            predicted_noise = model(noisy_images, time_steps)
            loss = torch.nn.functional.mse_loss(predicted_noise, true_noise)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(clean_images)
            count += len(clean_images)
    return total_loss / count


@torch.no_grad()
def sample_images(model, count, betas, alphas, alpha_bars):
    model.eval()
    images = torch.randn(count, 1, 28, 28, device=DEVICE)
    for step in reversed(range(TIME_STEPS)):
        time_steps = torch.full((count,), step, device=DEVICE, dtype=torch.long)
        predicted_noise = model(images, time_steps)
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        mean = (
            images - betas[step] / torch.sqrt(1 - alpha_bar) * predicted_noise
        ) / torch.sqrt(alpha)
        if step > 0:
            previous_alpha_bar = alpha_bars[step - 1]
            variance = betas[step] * (1 - previous_alpha_bar) / (1 - alpha_bar)
            images = mean + torch.sqrt(variance) * torch.randn_like(images)
        else:
            images = mean
    return images.clamp(-1, 1)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    transform = Compose([ToTensor(), Lambda(lambda image: image * 2 - 1)])
    try:
        full_dataset = MNIST(DATA_DIR, train=True, transform=transform, download=False)
    except RuntimeError as error:
        raise FileNotFoundError(
            "Run: python download_data/download_15_ddpm_mnist.py"
        ) from error
    train_set, val_set = random_split(
        full_dataset,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)
    model = NoisePredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    betas, alphas, alpha_bars = build_schedule()
    best_val_loss = math.inf

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, alpha_bars)
        val_loss = run_epoch(model, val_loader, None, alpha_bars)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
        print(
            f"epoch={epoch:02d} train_noise_mse={train_loss:.4f} val_noise_mse={val_loss:.4f}"
        )

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    samples = sample_images(model, 16, betas, alphas, alpha_bars)
    save_image((samples + 1) / 2, SAMPLE_PATH, nrow=4)
    print("generated shape:", samples.shape)
    print(f"samples saved to: {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
