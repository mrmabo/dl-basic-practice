from pathlib import Path

from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

OUT = Path(__file__).resolve().parents[1] / "data" / "transfer_cifar10"
OUT.mkdir(parents=True, exist_ok=True)
CIFAR10(OUT, train=True, download=True)
CIFAR10(OUT, train=False, download=True)
resnet18(weights=ResNet18_Weights.DEFAULT)
print(f"CIFAR-10 and ResNet18 weights are ready; data saved to {OUT}")
