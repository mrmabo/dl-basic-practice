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

## 第二阶段任务

当前7套流程属于第一阶段，目标是掌握 PyTorch 的通用训练闭环，以及表格、图像、时间序列和图数据的基本处理方式。完成第一阶段后，再增加下面5套流程，将项目扩展为12套完整流程。

### 8. CNN 图像回归

使用 CNN 根据图像预测一个连续值，重点练习：

- 将分类头替换为单个连续值输出
- 使用 `MSELoss`、MAE 和 RMSE
- 对连续目标进行标准化和逆标准化
- 理解分类任务与回归任务在输出头、loss 和 metric 上的区别

### 9. U-Net 图像分割

使用成对的 image 和 mask 完成像素级预测，重点练习：

- 图像与 mask 的同步读取和数据增强
- Encoder、Decoder、Skip Connection 和上采样
- 使用 BCE Loss、Dice Loss 或组合 loss
- 使用 Dice 和 IoU 评估分割结果
- 保存并可视化预测 mask

### 10. Embedding + LSTM 文本分类

使用简单的文本情感分类数据，完成从原始文本到分类结果的完整流程，重点练习：

- tokenize、建立 vocabulary 和 numericalize
- 对不等长序列进行 padding
- `Embedding` 层与 LSTM 文本表示
- 使用 `BCEWithLogitsLoss` 完成二分类
- 从一条新的原始文本开始执行推理

### 11. 预训练模型迁移学习

使用预训练 CNN 完成新的小型图像分类任务，重点练习：

- 加载预训练权重并替换分类头
- 冻结 backbone，只训练新的分类头
- 解冻部分网络进行 fine-tuning
- 为 backbone 和分类头设置不同的 learning rate
- 保存和加载微调后的最佳模型

### 12. 多步时间序列预测

使用多变量历史序列预测未来多个时间步，例如用历史96步预测未来24步，重点练习：

- 按时间顺序划分训练集、验证集和测试集，避免数据泄漏
- 使用滑动窗口构造 `[B, seq_len, features]` 输入
- 生成 `[B, pred_len, targets]` 多步预测结果
- 只使用训练集统计量进行标准化和逆标准化
- 使用 MAE、MSE 和 RMSE 评估预测结果
- 绘制未来多个时间步的真实值与预测值

### 进入第二阶段的标准

完成下面的目标后再开始第二阶段：

- 能够不看答案写出至少5套第一阶段流程的训练主干
- 能够解释关键位置的 tensor shape
- 能够独立检查并修复常见的 shape、dtype 和 device 错误
- 能够改变输入维度、类别数、预测长度、loss 或 metric
- 能够正确保存、加载最佳 checkpoint 并完成 inference

第二阶段建议顺序：U-Net 图像分割 → 文本分类 → 迁移学习 → 图像回归 → 多步时间序列预测。

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
