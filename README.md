# PyTorch 七套完整基础流程

每个脚本都包含：数据生成、`Dataset`、`DataLoader`、模型、训练、验证、保存最佳权重、加载权重、测试与预测。

```bash
pip install torch numpy
python 01_mlp_classification.py
python 02_cnn_image_classification.py
python 03_lstm_time_series_forecast.py
python 04_transformer_time_series_forecast.py
python 05_autoencoder_anomaly_detection.py
python 06_rnn_sequence_classification.py
python 07_gnn_node_classification.py
```

其中 GNN 示例用纯 PyTorch 实现 GCN，不需要额外安装 `torch_geometric`。

## 下载对应的公开真实数据

训练脚本默认使用合成数据，确保无需联网也能练习完整流程。下面每个下载脚本会把真实数据保存到 `data/`，且重复运行时不会重复下载压缩包：

| 流程 | 公开数据 | 下载命令 |
|---|---|---|
| MLP 表格分类 | UCI Wine | `python download_01_mlp_wine.py` |
| CNN 图像分类 | MNIST | `python download_02_cnn_mnist.py` |
| LSTM 时序预测 | AirPassengers（两列 CSV） | `python download_03_lstm_bike_sharing.py` |
| Transformer 多变量预测 | ETTh1（规整 CSV） | `python download_04_transformer_air_quality.py` |
| Autoencoder 异常检测 | UCI Wisconsin Breast Cancer | `python download_05_autoencoder_kdd99.py` |
| RNN 序列分类 | UCR SyntheticControl（600条） | `python download_06_rnn_forda.py` |
| GNN 节点分类 | LINQS Cora | `python download_07_gnn_cora.py` |

这些脚本负责获取和解压原始数据。下一练习阶段，可以把对应训练文件中的合成 `Dataset` 替换成读取真实文件的 `Dataset`，训练循环和模型主体保持不变。

建议练习顺序：先运行并逐段理解；第二天只看模块标题重写；第三天从空白文件重写；第四天修改数据维度、类别数或预测目标。
