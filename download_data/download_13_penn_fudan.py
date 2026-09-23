from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

OUT = Path(__file__).resolve().parents[1] / "data" / "penn_fudan_ped"
OUT.mkdir(parents=True, exist_ok=True)
archive = OUT / "PennFudanPed.zip"
urlretrieve("https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip", archive)
with ZipFile(archive) as zip_file:
    zip_file.extractall(OUT)
archive.unlink()
fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
print(f"Penn-Fudan and Faster R-CNN weights are ready: {OUT}")
