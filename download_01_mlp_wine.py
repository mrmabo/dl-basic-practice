"""下载 UCI Wine：适合 MLP 表格多分类。"""

from pathlib import Path
from urllib.request import urlretrieve
import zipfile

URL = "https://archive.ics.uci.edu/static/public/109/wine.zip"
OUT = Path("data/mlp_wine")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "wine.zip"
    if not archive.exists():
        print("Downloading", URL)
        urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(OUT)
    print("Ready:", OUT.resolve())


if __name__ == "__main__":
    main()
