"""完整流程 3：LSTM 单变量时间序列一步预测。"""
import math, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED, LOOKBACK, BATCH_SIZE, EPOCHS = 42, 24, 64, 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class WindowDataset(Dataset):
    def __init__(self, values, lookback=LOOKBACK): self.values, self.lookback = values.float(), lookback
    def __len__(self): return len(self.values)-self.lookback
    def __getitem__(self, i):
        return self.values[i:i+self.lookback].unsqueeze(-1), self.values[i+self.lookback].unsqueeze(-1)  # [L,1], [1]

class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__(); self.lstm = nn.LSTM(1, 32, batch_first=True); self.head = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)  # [B,L,1] -> [B,L,32]
        return self.head(out[:, -1])

def evaluate(model, loader, loss_fn):
    model.eval(); total = 0
    with torch.no_grad():
        for x,y in loader: x,y=x.to(DEVICE),y.to(DEVICE); total += loss_fn(model(x),y).item()*x.size(0)
    return total/len(loader.dataset)

def main():
    set_seed(); g=torch.Generator().manual_seed(SEED); t=torch.arange(3000).float()
    raw = torch.sin(2*math.pi*t/24)+.4*torch.sin(2*math.pi*t/168)+.08*torch.randn(3000,generator=g)
    train_end,val_end=2000,2500; mean,std=raw[:train_end].mean(),raw[:train_end].std(); z=(raw-mean)/std
    train_ds=WindowDataset(z[:train_end]); val_ds=WindowDataset(z[train_end-LOOKBACK:val_end]); test_ds=WindowDataset(z[val_end-LOOKBACK:])
    train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True); val_loader=DataLoader(val_ds,BATCH_SIZE); test_loader=DataLoader(test_ds,BATCH_SIZE)
    model=LSTMForecaster().to(DEVICE); loss_fn=nn.MSELoss(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3); best=float("inf")
    for epoch in range(1,EPOCHS+1):
        model.train(); total=0
        for x,y in train_loader: x,y=x.to(DEVICE),y.to(DEVICE); optimizer.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); optimizer.step(); total+=loss.item()*x.size(0)
        val=evaluate(model,val_loader,loss_fn)
        if val<best: best=val; torch.save(model.state_dict(),"best_lstm.pt")
        print(f"epoch={epoch:02d} train_mse={total/len(train_ds):.5f} val_mse={val:.5f}")
    model.load_state_dict(torch.load("best_lstm.pt",map_location=DEVICE,weights_only=True)); print(f"test_mse(normalized)={evaluate(model,test_loader,loss_fn):.5f}")
    x,y=next(iter(test_loader)); pred=model(x[:5].to(DEVICE)).cpu().detach()*std+mean; actual=y[:5]*std+mean
    print("pred:",pred.squeeze().round(decimals=3).tolist()); print("true:",actual.squeeze().round(decimals=3).tolist())

if __name__ == "__main__": main()
