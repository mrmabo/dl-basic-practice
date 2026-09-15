"""完整流程 4：Transformer 多变量时间序列预测未来一步。"""
import math, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED, LOOKBACK, BATCH_SIZE, EPOCHS = 42, 36, 64, 10
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class MultiSeriesDataset(Dataset):
    def __init__(self, data): self.data=data.float()
    def __len__(self): return len(self.data)-LOOKBACK
    def __getitem__(self,i): return self.data[i:i+LOOKBACK],self.data[i+LOOKBACK,0:1]  # [L,3], [1]

class TransformerForecaster(nn.Module):
    def __init__(self,in_features=3,d_model=32,nhead=4,layers=2):
        super().__init__(); self.input_proj=nn.Linear(in_features,d_model)
        self.pos=nn.Parameter(torch.randn(1,LOOKBACK,d_model)*.02)
        layer=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=64,dropout=.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers); self.head=nn.Linear(d_model,1)
    def forward(self,x):
        h=self.input_proj(x)+self.pos[:,:x.size(1)]  # [B,L,3] -> [B,L,32]
        return self.head(self.encoder(h)[:,-1])      # -> [B,1]

def eval_mse(model,loader,loss_fn):
    model.eval(); total=0
    with torch.no_grad():
        for x,y in loader: x,y=x.to(DEVICE),y.to(DEVICE); total+=loss_fn(model(x),y).item()*x.size(0)
    return total/len(loader.dataset)

def main():
    set_seed(); g=torch.Generator().manual_seed(SEED); t=torch.arange(3600).float()
    raw=torch.stack([torch.sin(t/12)+.1*torch.randn(3600,generator=g),torch.cos(t/12),torch.sin(t/50)],dim=1)
    a,b=2400,3000; mean,std=raw[:a].mean(0),raw[:a].std(0); z=(raw-mean)/std
    train=MultiSeriesDataset(z[:a]); val=MultiSeriesDataset(z[a-LOOKBACK:b]); test=MultiSeriesDataset(z[b-LOOKBACK:])
    train_loader=DataLoader(train,BATCH_SIZE,shuffle=True); val_loader=DataLoader(val,BATCH_SIZE); test_loader=DataLoader(test,BATCH_SIZE)
    model=TransformerForecaster().to(DEVICE); loss_fn=nn.MSELoss(); optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4); best=float("inf")
    for epoch in range(1,EPOCHS+1):
        model.train(); total=0
        for x,y in train_loader: x,y=x.to(DEVICE),y.to(DEVICE); optimizer.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total+=loss.item()*x.size(0)
        vm=eval_mse(model,val_loader,loss_fn)
        if vm<best: best=vm; torch.save(model.state_dict(),"best_transformer.pt")
        print(f"epoch={epoch:02d} train_mse={total/len(train):.5f} val_mse={vm:.5f}")
    model.load_state_dict(torch.load("best_transformer.pt",map_location=DEVICE,weights_only=True)); print(f"test_mse={eval_mse(model,test_loader,loss_fn):.5f}")
    x,y=next(iter(test_loader)); pred=model(x[:5].to(DEVICE)).detach().cpu()*std[0]+mean[0]; true=y[:5]*std[0]+mean[0]
    print("pred:",pred.squeeze().round(decimals=3).tolist()); print("true:",true.squeeze().round(decimals=3).tolist())

if __name__ == "__main__": main()
