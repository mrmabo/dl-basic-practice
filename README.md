# PyTorch 七套完整基础流程

每个脚本都包含：数据生成、`Dataset`、`DataLoader`、模型、训练、验证、保存最佳权重、加载权重、测试与预测。

## 使用 Conda 配置环境

建议使用 Python 3.11。它对 PyTorch、NumPy 以及常用机器学习库的兼容性比较成熟。

创建新的 Conda 环境：

```bash
conda create -n dl-basic-practice python=3.11 -y
```

激活环境：

```bash
conda activate dl-basic-practice
```

进入本项目目录，然后安装依赖：

```bash
python -m pip install -r requirements.txt
```

验证 Python、PyTorch、NumPy 和 CUDA 状态：

```bash
python --version
python -c "import torch, numpy; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('CUDA available:', torch.cuda.is_available())"
```

如果最后显示 `CUDA available: True`，说明当前 PyTorch 可以使用 NVIDIA GPU；显示 `False` 时仍然可以使用 CPU 运行这些练习。

## 运行训练流程

然后可以分别运行七个完整流程：

```bash
python 01_mlp_classification.py
python 02_cnn_image_classification.py
python 03_lstm_time_series_forecast.py
python 04_transformer_time_series_forecast.py
python 05_autoencoder_anomaly_detection.py
python 06_rnn_sequence_classification.py
python 07_gnn_node_classification.py
```

其中 GNN 示例用纯 PyTorch 实现 GCN，不需要额外安装 `torch_geometric`。

## 空白重写练习模板

学习完一个完整流程后，新建一个空白 Python 文件，只保留下面这些标题，然后按照顺序独立补全代码：

```python
# imports
# config
# dataloader
# model
# loss and optimizer
# training
# evaluation
# checkpoint
# inference
```

练习时先不要查看对应的完整脚本。写完并运行后，再与标准代码对照，重点检查：

- 是否完成训练集、验证集和测试集的划分
- 输入、标签以及模型输出的 tensor shape 是否正确
- 训练阶段是否包含清零梯度、前向传播、计算损失、反向传播和更新参数
- 验证和推理阶段是否使用 `model.eval()` 与 `torch.no_grad()`
- 是否能够保存并重新加载验证集表现最好的模型
- 是否输出最终测试指标和少量预测结果

## 下载对应的公开真实数据

训练脚本默认使用合成数据，确保无需联网也能练习完整流程。下面每个下载脚本会把真实数据保存到 `data/`，且重复运行时不会重复下载压缩包：

| 流程 | 公开数据 | 下载命令 |
|---|---|---|
| MLP 表格分类 | UCI Wine | `python download_01_mlp_wine.py` |
| CNN 图像分类 | MNIST | `python download_02_cnn_mnist.py` |
| LSTM 时序预测 | AirPassengers（两列 CSV） | `python download_03_lstm_air_passengers.py` |
| Transformer 多变量预测 | ETTh1（规整 CSV） | `python download_04_transformer_etth1.py` |
| Autoencoder 异常检测 | UCI Wisconsin Breast Cancer | `python download_05_autoencoder_breast_cancer.py` |
| RNN 序列分类 | UCR SyntheticControl（600条） | `python download_06_rnn_synthetic_control.py` |
| GNN 节点分类 | LINQS Cora | `python download_07_gnn_cora.py` |

这些脚本负责获取和解压原始数据。下一练习阶段，可以把对应训练文件中的合成 `Dataset` 替换成读取真实文件的 `Dataset`，训练循环和模型主体保持不变。

建议练习顺序：先运行并逐段理解；第二天只看模块标题重写；第三天从空白文件重写；第四天修改数据维度、类别数或预测目标。
