"""下载 UCI Wisconsin Breast Cancer：小型表格数据，可练异常检测。"""
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

URL="https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
OUT=Path("data/autoencoder_breast_cancer")
def main():
    OUT.mkdir(parents=True,exist_ok=True); archive=OUT/"breast_cancer.zip"
    if not archive.exists(): print("Downloading",URL); urlretrieve(URL,archive)
    with zipfile.ZipFile(archive) as z: z.extractall(OUT)
    print("Ready:",OUT.resolve(),"; 可用良性样本训练，将恶性样本视为异常")
if __name__=="__main__": main()
