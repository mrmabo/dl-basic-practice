"""下载 LINQS Cora 引文网络：适合 GCN 节点分类。"""

from pathlib import Path
from urllib.request import urlretrieve
import shutil
import tarfile

URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "gnn_cora"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "cora.tgz"
    if not archive.exists():
        print("Downloading", URL)
        urlretrieve(URL, archive)
    # 只解压本练习需要的三个已知文件，兼容 Python 3.9+，同时避免路径穿越。
    expected_members = {
        "cora/README",
        "cora/cora.content",
        "cora/cora.cites",
    }
    with tarfile.open(archive, "r:gz") as tar:
        members = [
            member for member in tar.getmembers() if member.name in expected_members
        ]
        found_members = {member.name for member in members}
        if found_members != expected_members:
            missing = expected_members - found_members
            raise RuntimeError(f"Cora 压缩包缺少预期文件: {sorted(missing)}")
        for member in members:
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"无法读取 Cora 文件: {member.name}")
            target = OUT / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    print("Ready:", OUT.resolve(), "; cora.content=节点特征/标签，cora.cites=边")


if __name__ == "__main__":
    main()
