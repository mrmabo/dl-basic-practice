from pathlib import Path

from torchvision.datasets import CIFAR10

OUT = Path(__file__).resolve().parents[1] / "data" / "resnet_cifar10"
OUT.mkdir(parents=True, exist_ok=True)
CIFAR10(OUT, train=True, download=True)
CIFAR10(OUT, train=False, download=True)
print(f"CIFAR-10 saved to {OUT}")
