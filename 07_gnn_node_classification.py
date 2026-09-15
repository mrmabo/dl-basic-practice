"""完整流程 7：两层 GCN 节点分类（纯 PyTorch，无需 torch_geometric）。"""
import random
import numpy as np
import torch
from torch import nn

SEED,EPOCHS=42,150
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def make_graph(nodes_per_class=100,features=8):
    """创建两社区图。返回节点特征X、归一化邻接矩阵A_hat、标签和三个mask。"""
    g=torch.Generator().manual_seed(SEED); n=nodes_per_class*2
    y=torch.cat([torch.zeros(nodes_per_class,dtype=torch.long),torch.ones(nodes_per_class,dtype=torch.long)])
    centers=torch.stack([-torch.ones(features),torch.ones(features)]); x=centers[y]+.8*torch.randn(n,features,generator=g)
    same=y[:,None]==y[None,:]; probs=torch.where(same,torch.tensor(.12),torch.tensor(.01))
    upper=torch.triu((torch.rand(n,n,generator=g)<probs),diagonal=1); adj=(upper|upper.T).float()
    adj=adj+torch.eye(n)  # self-loop
    degree=adj.sum(1); d_inv_sqrt=degree.pow(-.5)
    a_hat=d_inv_sqrt[:,None]*adj*d_inv_sqrt[None,:]  # D^-1/2 A D^-1/2
    perm=torch.randperm(n,generator=g); train_mask=torch.zeros(n,dtype=torch.bool); val_mask=train_mask.clone(); test_mask=train_mask.clone()
    train_mask[perm[:100]]=True; val_mask[perm[100:150]]=True; test_mask[perm[150:]]=True
    return x,a_hat,y,train_mask,val_mask,test_mask

class GraphConv(nn.Module):
    def __init__(self,in_features,out_features): super().__init__(); self.linear=nn.Linear(in_features,out_features,bias=False)
    def forward(self,x,a_hat): return a_hat@self.linear(x)  # 聚合邻居: [N,N] @ [N,F]

class GCN(nn.Module):
    def __init__(self):
        super().__init__(); self.gcn1=GraphConv(8,32); self.gcn2=GraphConv(32,2); self.dropout=nn.Dropout(.3)
    def forward(self,x,a_hat):
        x=torch.relu(self.gcn1(x,a_hat)); x=self.dropout(x); return self.gcn2(x,a_hat)  # [N,2]

def metrics(model,x,a_hat,y,mask,loss_fn):
    model.eval()
    with torch.no_grad():
        logits=model(x,a_hat); loss=loss_fn(logits[mask],y[mask]); acc=(logits[mask].argmax(1)==y[mask]).float().mean()
    return loss.item(),acc.item()

def main():
    set_seed(); x,a_hat,y,train_mask,val_mask,test_mask=make_graph()
    x,a_hat,y=x.to(DEVICE),a_hat.to(DEVICE),y.to(DEVICE); train_mask,val_mask,test_mask=train_mask.to(DEVICE),val_mask.to(DEVICE),test_mask.to(DEVICE)
    model=GCN().to(DEVICE); loss_fn=nn.CrossEntropyLoss(); optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=5e-4); best=float("inf")
    for epoch in range(1,EPOCHS+1):
        model.train(); optimizer.zero_grad(); logits=model(x,a_hat); loss=loss_fn(logits[train_mask],y[train_mask]); loss.backward(); optimizer.step()
        val_loss,val_acc=metrics(model,x,a_hat,y,val_mask,loss_fn)
        if val_loss<best: best=val_loss; torch.save(model.state_dict(),"best_gcn.pt")
        if epoch==1 or epoch%25==0: print(f"epoch={epoch:03d} train_loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
    model.load_state_dict(torch.load("best_gcn.pt",map_location=DEVICE,weights_only=True)); print("test:",metrics(model,x,a_hat,y,test_mask,loss_fn))
    model.eval()
    with torch.no_grad(): pred=model(x,a_hat).argmax(1)
    ids=torch.where(test_mask)[0][:10]; print("node_ids:",ids.cpu().tolist()); print("pred:",pred[ids].cpu().tolist(),"true:",y[ids].cpu().tolist())

if __name__ == "__main__": main()
