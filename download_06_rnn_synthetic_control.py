"""下载 UCR SyntheticControl：600条短序列、6分类，适合 RNN。"""

from pathlib import Path
from urllib.request import urlretrieve
import zipfile

URL = "https://www.timeseriesclassification.com/aeon-toolkit/SyntheticControl.zip"
OUT = Path("data/rnn_synthetic_control")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "SyntheticControl.zip"
    if not archive.exists():
        print("Downloading", URL)
        urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(OUT)
    print("Ready:", OUT.resolve(), "; 仅600条序列，适合入门序列分类")


if __name__ == "__main__":
    main()
