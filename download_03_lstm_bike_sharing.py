"""下载经典 AirPassengers：单变量月度时序，适合 LSTM 预测。"""
from pathlib import Path
from urllib.request import urlretrieve

URL="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
OUT=Path("data/lstm_air_passengers")
def main():
    OUT.mkdir(parents=True,exist_ok=True); target=OUT/"airline-passengers.csv"
    if not target.exists(): print("Downloading",URL); urlretrieve(URL,target)
    print("Ready:",target.resolve(),"; 只有 Month 和 Passengers 两列")
if __name__=="__main__": main()
