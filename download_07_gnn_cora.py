"""下载 LINQS Cora 引文网络：适合 GCN 节点分类。"""
from pathlib import Path
from urllib.request import urlretrieve
import tarfile

URL="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
OUT=Path("data/gnn_cora")
def main():
    OUT.mkdir(parents=True,exist_ok=True); archive=OUT/"cora.tgz"
    if not archive.exists(): print("Downloading",URL); urlretrieve(URL,archive)
    with tarfile.open(archive,"r:gz") as t: t.extractall(OUT,filter="data")
    print("Ready:",OUT.resolve(),"; cora.content=节点特征/标签，cora.cites=边")
if __name__=="__main__": main()
