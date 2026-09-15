"""下载 ETTh1：结构规整的多变量时序 CSV，适合 Transformer。"""
from pathlib import Path
from urllib.request import urlretrieve

URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
OUT=Path("data/transformer_etth1")
def main():
    OUT.mkdir(parents=True,exist_ok=True); target=OUT/"ETTh1.csv"
    if not target.exists(): print("Downloading",URL); urlretrieve(URL,target)
    print("Ready:",target.resolve(),"; CSV 已整理好，可用 OT 作为目标")
if __name__=="__main__": main()
