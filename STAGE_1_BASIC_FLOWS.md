# 第一阶段：基础训练闭环（01–07）

## 目标

第一阶段的目标不是记住所有 PyTorch API，而是能够从空白文件写出完整训练闭环：

```text
公开数据 -> Dataset -> DataLoader -> Model -> Loss
-> Backward -> Validation -> Checkpoint -> Test -> Inference
```

## 流程

| 编号 | 项目 | 公开数据 | 核心练习 |
|---|---|---|---|
| 01 | MLP 表格分类 | UCI Wine | 标准化、多分类、分层拆分 |
| 02 | CNN 图像分类 | MNIST | torchvision、random_split、卷积 |
| 03 | LSTM 时序预测 | AirPassengers | 滑动窗口、单步预测 |
| 04 | Transformer 时序预测 | ETTh1 | attention、多变量输入 |
| 05 | Autoencoder 异常检测 | Breast Cancer Wisconsin | 重构误差、阈值 |
| 06 | RNN 序列分类 | UCR SyntheticControl | hidden state、序列分类 |
| 07 | GNN 节点分类 | Cora | 图结构、GCN、布尔 mask |

下载脚本位于 `download_data/download_01_*.py` 至 `download_data/download_07_*.py`。

## 期望达到的效果

完成第一阶段后，应当能够：

- 不看答案写出 Dataset、DataLoader 和训练循环
- 解释模型输入、隐藏层和输出的 shape
- 正确切换 `model.train()` 与 `model.eval()`
- 正确使用 `torch.no_grad()`
- 保存并加载最佳模型
- 完成一批样本和单个样本的推理
- 排查常见的 shape、dtype 和 device 错误
- 知道分类、回归、预测和异常检测的 loss 与 metric 为什么不同

## 进入第二阶段的标准

至少能够从空白文件完成5个第一阶段流程，并能修改输入维度、类别数、预测目标或隐藏层结构。
