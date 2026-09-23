# 第一阶段：PyTorch 基础训练闭环

第一阶段用于建立从空白文件写出完整深度学习流程的肌肉记忆。

## 七个项目

| 编号 | 项目 | 数据集 | 核心知识 |
|---|---|---|---|
| 01 | MLP 表格分类 | UCI Wine | 标准化、多分类、分层拆分 |
| 02 | CNN 图像分类 | MNIST | torchvision、random_split、二维卷积 |
| 03 | LSTM 时序预测 | AirPassengers | 滑动窗口、单步预测 |
| 04 | Transformer 时序预测 | ETTh1 | 多变量输入、attention、多步输出 |
| 05 | Autoencoder 异常检测 | Breast Cancer Wisconsin | 重构误差、阈值 |
| 06 | RNN 序列分类 | UCR SyntheticControl | 序列分类、hidden state |
| 07 | GNN 节点分类 | Cora | 邻接矩阵、GCN、布尔 mask |

## 下载与运行

所有命令从仓库根目录运行：

```bash
python stage1_basic_flows/download_data/download_01_mlp_wine.py
python stage1_basic_flows/download_data/download_02_cnn_mnist.py
python stage1_basic_flows/download_data/download_03_lstm_air_passengers.py
python stage1_basic_flows/download_data/download_04_transformer_etth1.py
python stage1_basic_flows/download_data/download_05_autoencoder_breast_cancer.py
python stage1_basic_flows/download_data/download_06_rnn_synthetic_control.py
python stage1_basic_flows/download_data/download_07_gnn_cora.py
```

```bash
python stage1_basic_flows/01_mlp_classification.py
python stage1_basic_flows/02_cnn_image_classification.py
python stage1_basic_flows/03_lstm_time_series_forecast.py
python stage1_basic_flows/04_transformer_time_series_forecast.py
python stage1_basic_flows/05_autoencoder_anomaly_detection.py
python stage1_basic_flows/06_rnn_sequence_classification.py
python stage1_basic_flows/07_gnn_node_classification.py
```

## 空白重写模板

```python
# imports
# config
# dataset and dataloader
# model
# loss and optimizer
# training
# evaluation
# checkpoint
# inference
```

先完成训练闭环，再检查 shape、dtype、device 和数据泄漏。`building_blocks/` 用于单独练习 attention 等底层模块。
