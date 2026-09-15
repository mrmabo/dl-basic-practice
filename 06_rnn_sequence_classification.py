"""完整流程 6：原生 RNN 序列分类（判断带噪序列总体趋势）。"""
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

SEED,SEQ_LEN,BATCH_SIZE,EPOCHS=42,30,64,10
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class TrendDataset(Dataset):
    """类别0=下降，类别1=上升；每个样本为 [L,1]。"""
    def __init__(self,n=2400):
        g=torch.Generator().manual_seed(SEED); self.y=torch.randint(2,(n,),generator=g)
        base=torch.linspace(-1,1,SEQ_LEN).repeat(n,1); direction=self.y.float()*2-1
        offset=torch.randn(n,1,generator=g); noise=.25*torch.randn(n,SEQ_LEN,generator=g)
        self.x=(direction[:,None]*base+offset+noise).unsqueeze(-1)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.x[i],self.y[i]

class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn=nn.RNN(input_size=1,hidden_size=32,num_layers=1,batch_first=True,nonlinearity="tanh"); self.head=nn.Linear(32,2)
    def forward(self,x):
        output,hidden=self.rnn(x)  # x:[B,L,1], output:[B,L,32], hidden:[1,B,32]
        return self.head(hidden[-1])  # [B,2]

def evaluate(model,loader,loss_fn):
    model.eval(); loss_sum=correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE); logits=model(x); loss=loss_fn(logits,y)
            loss_sum+=loss.item()*x.size(0); correct+=(logits.argmax(1)==y).sum().item(); total+=x.size(0)
    return loss_sum/total,correct/total

def main():
    set_seed(); train,val,test=random_split(TrendDataset(),[1600,400,400],generator=torch.Generator().manual_seed(SEED))
    train_loader=DataLoader(train,BATCH_SIZE,shuffle=True); val_loader=DataLoader(val,BATCH_SIZE); test_loader=DataLoader(test,BATCH_SIZE)
    model=RNNClassifier().to(DEVICE); loss_fn=nn.CrossEntropyLoss(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3); best=float("inf")
    for epoch in range(1,EPOCHS+1):
        model.train(); total=0
        for x,y in train_loader:
            x,y=x.to(DEVICE),y.to(DEVICE); optimizer.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); total+=loss.item()*x.size(0)
        val_loss,val_acc=evaluate(model,val_loader,loss_fn)
        if val_loss<best: best=val_loss; torch.save(model.state_dict(),"best_rnn.pt")
        print(f"epoch={epoch:02d} train_loss={total/len(train):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
    model.load_state_dict(torch.load("best_rnn.pt",map_location=DEVICE,weights_only=True)); print("test:",evaluate(model,test_loader,loss_fn))
    x,y=next(iter(test_loader)); pred=model(x[:8].to(DEVICE)).argmax(1).cpu(); print("pred:",pred.tolist(),"true:",y[:8].tolist())

if __name__ == "__main__": main()
