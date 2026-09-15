"""下载公开 MNIST 原始 IDX 文件：适合 CNN 图像分类。"""

from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
OUT = Path("data/cnn_mnist")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = OUT / name
        if not target.exists():
            print("Downloading", name)
            urlretrieve(BASE + name, target)
    print("Ready:", OUT.resolve(), "; 已下载四个 MNIST gzip 压缩的 IDX 原始文件")


if __name__ == "__main__":
    main()
