from pathlib import Path

from torchvision.datasets import OxfordIIITPet

OUT = Path(__file__).resolve().parents[1] / "data" / "unet_oxford_pet"
OUT.mkdir(parents=True, exist_ok=True)
OxfordIIITPet(OUT, split="trainval", target_types="segmentation", download=True)
OxfordIIITPet(OUT, split="test", target_types="segmentation", download=True)
print(f"Oxford-IIIT Pet saved to {OUT}")
