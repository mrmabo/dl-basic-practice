from pathlib import Path
from urllib.request import urlretrieve

OUT = Path(__file__).resolve().parents[1] / "data" / "transformer_etth1" / "ETTh1.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
if not OUT.exists():
    urlretrieve("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", OUT)
print(f"ETTh1 saved to {OUT}")
