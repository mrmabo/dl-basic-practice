"""使用torchvision下载MNIST公开数据，供CNN训练脚本读取。"""

from pathlib import Path

from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cnn_mnist"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    MNIST(
        root=OUT,
        train=True,
        download=True,
    )
    MNIST(
        root=OUT,
        train=False,
        download=True,
    )

    print("Ready:", OUT.resolve())
    print("训练集和测试集已保存为torchvision MNIST目录结构。")


if __name__ == "__main__":
    main()
