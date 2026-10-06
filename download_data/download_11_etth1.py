from pathlib import Path
from urllib.request import urlretrieve

OUT = Path(__file__).resolve().parents[1] / "data" / "multistep_etth1"
OUT.mkdir(parents=True, exist_ok=True)
target = OUT / "ETTh1.csv"
urlretrieve(
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    target,
)
print(f"ETTh1 saved to {target}")
