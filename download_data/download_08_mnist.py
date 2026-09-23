from pathlib import Path

from torchvision.datasets import MNIST

OUT = Path(__file__).resolve().parents[1] / "data" / "cnn_regression_mnist"
OUT.mkdir(parents=True, exist_ok=True)
MNIST(OUT, train=True, download=True)
MNIST(OUT, train=False, download=True)
print(f"MNIST saved to {OUT}")
