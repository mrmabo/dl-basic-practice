# PyTorch 七套完整基础流程

每个训练脚本都使用对应的公开数据，并包含：数据读取与预处理、`Dataset`、`DataLoader`、模型、训练、验证、保存最佳权重、加载权重、测试与预测。

## 三层学习路线总览

这个项目按照“基础训练闭环 → 常见应用任务 → 特殊训练范式”逐层扩展；第一层已有可运行代码，第二层和第三层是后续按照本 README 继续实现的任务。

| 层级 | 目标 | 流程 |
|---|---|---|
| 第一层：基础训练闭环 | 熟练 Dataset、DataLoader、模型、loss、训练、验证、checkpoint、测试和 inference | 1～7，当前已实现 |
| 第二层：常见应用任务 | 掌握图像、文本、迁移学习和进阶时序任务的数据与输出结构 | 8～12，待实现 |
| 第三层：特殊训练范式 | 掌握交互式强化学习和生成式扩散模型的不同训练闭环 | 13～14，待实现 |

推荐总顺序：

```text
第一层7套流程全部熟练
→ U-Net图像分割
→ Embedding + LSTM文本分类
→ 迁移学习
→ CNN图像回归
→ 进阶多步时序预测
→ DQN强化学习
→ DDPM扩散模型
→ PPO等更高级强化学习
```

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

验证 Python、PyTorch、torchvision、NumPy、scikit-learn 和 CUDA 状态：

```bash
python --version
python -c "import torch, torchvision, numpy, sklearn; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('NumPy:', numpy.__version__); print('scikit-learn:', sklearn.__version__); print('CUDA available:', torch.cuda.is_available())"
```

如果最后显示 `CUDA available: True`，说明当前 PyTorch 可以使用 NVIDIA GPU；显示 `False` 时仍然可以使用 CPU 运行这些练习。

## 下载公开数据

第一次运行训练流程前，先下载每套流程对应的公开数据：

```bash
python download_data/download_01_mlp_wine.py
python download_data/download_02_cnn_mnist.py
python download_data/download_03_lstm_air_passengers.py
python download_data/download_04_transformer_etth1.py
python download_data/download_05_autoencoder_breast_cancer.py
python download_data/download_06_rnn_synthetic_control.py
python download_data/download_07_gnn_cora.py
```

所有数据都会保存在仓库的 `data/` 目录中，该目录不会提交到 GitHub。\n\n其中MNIST由下载脚本调用`torchvision.datasets.MNIST(download=True)`保存；CNN训练脚本使用`download=False`读取，不再手动解析IDX文件。

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

## 数据集划分策略

现阶段优先练习完整的深度学习流程。表格分类、序列分类以及时间序列流程根据任务需要使用 scikit-learn 的 `train_test_split`；CNN + MNIST 使用 PyTorch 的 `torch.utils.data.random_split`，让图像分类示例保持更完整的 PyTorch 数据流程。

| 脚本 | 划分方式 | 必须保留的规则 |
|---|---|---|
| `01_mlp_classification.py` | 两次 `train_test_split` 得到 60% / 20% / 20% | 分类模式使用 `stratify`；回归模式不使用 |
| `02_cnn_image_classification.py` | 使用 `random_split` 将 MNIST 官方训练集拆为 55,000 条训练数据和 5,000 条验证数据 | 使用固定 `torch.Generator` 保证拆分可复现；官方测试集保持不变 |
| `03_lstm_time_series_forecast.py` | 两次 `train_test_split` 得到 70% / 15% / 15% | 必须设置 `shuffle=False`，保持时间顺序 |
| `04_transformer_time_series_forecast.py` | 两次 `train_test_split` 得到 70% / 15% / 15% | 必须设置 `shuffle=False`，保持时间顺序；窗口同时支持多步和多目标 |
| `05_autoencoder_anomaly_detection.py` | 正常样本划分为训练、验证和测试，异常样本放入测试集 | Autoencoder 的训练集和阈值验证集只包含正常样本 |
| `06_rnn_sequence_classification.py` | 从 UCR 官方训练集划出 20% 验证数据 | 使用 `stratify`；可变长度模式使用 padding 和真实长度 |
| `07_gnn_node_classification.py` | 每个类别划出 20 个训练节点和 30 个验证节点 | 使用布尔 mask 训练 GCN，其余节点用于测试 |

无论使用哪种划分方式，都只使用训练集计算均值和标准差，验证集与测试集不能参与统计量拟合，否则会发生 data leakage。

### 为什么现阶段使用 library method

当前第一目标是熟练掌握：数据进入 `Dataset` / `DataLoader`、模型前向传播、loss、反向传播、验证、checkpoint 和 inference。`train_test_split` 和 `random_split` 都是成熟且经过充分测试的工具。小样本分类数据需要 `stratify` 时使用 `train_test_split`；MNIST 数据量大、类别较均衡，并且 CNN 流程使用 torchvision Dataset，因此使用 `random_split` 更简洁，也更便于理解 PyTorch 的数据流程。

熟练以后完全可以阅读 library method 的实现，并自己写一个简化版本。建议顺序：

1. 先熟练使用 `train_test_split` 的 `train_size`、`test_size`、`random_state`、`shuffle` 和 `stratify`，同时掌握 `random_split` 的 `lengths` 和 `generator`。
2. 使用 `inspect.getsource(train_test_split)` 查看入口实现，再继续阅读它调用的 `ShuffleSplit` 和 `StratifiedShuffleSplit`。
3. 阅读每个脚本中保留的“进阶练习”注释代码，先复制到单独的练习文件中运行，再与当前 library method 做对照实验。
4. 用相同 seed 检查结果可复现，并用类别计数验证 stratified split 是否保持了类别比例。

以后也可以用同样的“先会用 → 再读源码 → 最后写简化版”方法学习 `StandardScaler`、常见 metric、PyTorch 的 `Dataset` 和 `DataLoader`。目标不是复刻 library 的全部边界处理，而是理解核心算法和接口设计。

## Transformer 预测类型练习

`04_transformer_time_series_forecast.py` 的窗口输出形状为 `[horizon, num_targets]`，经过 `DataLoader` 后模型目标与输出均为 `[batch_size, horizon, num_targets]`。

修改 `HORIZON` 和 `TARGET_NAMES` 可以在同一套完整流程中练习四种预测任务：

| 预测类型 | 配置示例 | 模型输出形状 |
|---|---|---|
| 单步、单目标 | `HORIZON = 1`，`TARGET_NAMES = ["OT"]` | `[B, 1, 1]` |
| 多步、单目标 | `HORIZON = 24`，`TARGET_NAMES = ["OT"]` | `[B, 24, 1]` |
| 单步、多目标 | `HORIZON = 1`，`TARGET_NAMES = ["HUFL", "OT"]` | `[B, 1, 2]` |
| 多步、多目标 | `HORIZON = 24`，`TARGET_NAMES = ["HUFL", "OT"]` | `[B, 24, 2]` |

滑动窗口的样本数量为 `len(data) - lookback - horizon + 1`，每个样本使用过去 `lookback` 步作为输入，并把紧接着的 `horizon` 步作为预测目标。

## 其他可切换任务练习

### MLP：表格分类与回归

`01_mlp_classification.py` 默认使用 Wine 类别标签完成三分类；设置 `TASK=regression` 后，使用其余12个属性预测连续的 alcohol 含量。

```powershell
$env:TASK="classification"
python 01_mlp_classification.py

$env:TASK="regression"
python 01_mlp_classification.py
```

分类模式使用 `[B, 3]` logits、`CrossEntropyLoss` 和 accuracy；回归模式使用 `[B, 1]` 输出、`MSELoss` 和 MAE，并且只用训练集统计量标准化和逆标准化回归目标。

### Autoencoder：普通重建与去噪重建

`05_autoencoder_anomaly_detection.py` 默认使用干净输入重建自身；设置 `DENOISING=1` 后，模型接收加入高斯噪声的输入，但仍以原始干净特征作为重建目标。

```powershell
$env:DENOISING="0"
python 05_autoencoder_anomaly_detection.py

$env:DENOISING="1"
$env:NOISE_STD="0.1"
python 05_autoencoder_anomaly_detection.py
```

两种模式都使用干净的验证集重建误差选择 checkpoint，并使用干净测试样本的重建误差计算 anomaly score。

### RNN：固定长度与可变长度序列

`06_rnn_sequence_classification.py` 默认保留全部60个时间步；设置 `VARIABLE_LENGTH=1` 后，每条序列被确定性地截成30至60步，再由 `collate_fn`、`pad_sequence` 和 `pack_padded_sequence` 完成批处理。

```powershell
$env:VARIABLE_LENGTH="0"
python 06_rnn_sequence_classification.py

$env:VARIABLE_LENGTH="1"
python 06_rnn_sequence_classification.py
```

padding 只负责把一个 batch 补成相同长度，RNN 根据真实 `lengths` 读取 packed sequence，因此不会把补出的0当作有效时间步。

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

## 第二层：常见应用任务

当前7套流程属于第一层，目标是掌握 PyTorch 的通用训练闭环，以及表格、图像、时间序列和图数据的基本处理方式；完成第一层后，再增加下面5套流程，将项目扩展为12套完整流程。

### 8. CNN 图像回归

建议脚本：`08_cnn_image_regression.py`

使用 CNN 根据图像预测一个连续值，重点练习：

- 将分类头替换为单个连续值输出
- 使用 `MSELoss`、MAE 和 RMSE
- 对连续目标进行标准化和逆标准化
- 理解分类任务与回归任务在输出头、loss 和 metric 上的区别

### 9. U-Net 图像分割

建议脚本：`09_unet_image_segmentation.py`

使用成对的 image 和 mask 完成像素级预测，重点练习：

- 图像与 mask 的同步读取和数据增强
- Encoder、Decoder、Skip Connection 和上采样
- 使用 BCE Loss、Dice Loss 或组合 loss
- 使用 Dice 和 IoU 评估分割结果
- 保存并可视化预测 mask

### 10. Embedding + LSTM 文本分类

建议脚本：`10_lstm_text_classification.py`

使用简单的文本情感分类数据，完成从原始文本到分类结果的完整流程，重点练习：

- tokenize、建立 vocabulary 和 numericalize
- 对不等长序列进行 padding
- `Embedding` 层与 LSTM 文本表示
- 使用 `BCEWithLogitsLoss` 完成二分类
- 从一条新的原始文本开始执行推理

### 11. 预训练模型迁移学习

建议脚本：`11_cnn_transfer_learning.py`

使用预训练 CNN 完成新的小型图像分类任务，重点练习：

- 加载预训练权重并替换分类头
- 冻结 backbone，只训练新的分类头
- 解冻部分网络进行 fine-tuning
- 为 backbone 和分类头设置不同的 learning rate
- 保存和加载微调后的最佳模型

### 12. 进阶多步时间序列预测

建议脚本：`12_advanced_time_series_forecast.py`

在流程4的直接多步预测基础上继续增加更完整的预测与分析能力，重点练习：

- 按时间顺序划分训练集、验证集和测试集，避免数据泄漏
- 使用滑动窗口构造 `[B, seq_len, features]` 输入
- 生成 `[B, pred_len, targets]` 多步预测结果
- 只使用训练集统计量进行标准化和逆标准化
- 使用 MAE、MSE 和 RMSE 评估预测结果
- 绘制未来多个时间步的真实值与预测值
- 对比 direct、recursive 和 encoder-decoder 多步预测方式

第二层的每套流程仍然必须完整包含：公开数据下载脚本、训练/验证/测试划分、Dataset、DataLoader、模型、loss、metric、checkpoint、inference，以及 README 中的运行命令和 tensor shape。

### 进入第二层的标准

完成下面的目标后再开始第二层：

- 能够不看答案写出至少5套第一层流程的训练主干
- 能够解释关键位置的 tensor shape
- 能够独立检查并修复常见的 shape、dtype 和 device 错误
- 能够改变输入维度、类别数、预测长度、loss 或 metric
- 能够正确保存、加载最佳 checkpoint 并完成 inference

第二层建议顺序：U-Net 图像分割 → 文本分类 → 迁移学习 → 图像回归 → 多步时间序列预测。

## 第三层：特殊训练范式

第三层不再只是替换 Dataset、backbone 或 prediction head，而是学习与普通监督学习明显不同的训练数据来源、目标构造和推理过程。

### 13. DQN 强化学习

建议脚本：`13_dqn_cartpole.py`

建议环境：Gymnasium `CartPole-v1`。

完整流程需要包含：

- Environment 的创建、reset、step 和 episode 终止处理
- state、action、reward、next_state 和 done 的 transition
- epsilon-greedy action selection
- Replay Buffer 的存储与随机采样
- Online Q Network 与 Target Q Network
- Bellman TD target 和 TD loss
- 定期同步 target network
- episode reward、移动平均 reward 和训练曲线
- 保存最佳模型，并用关闭探索的策略完成 evaluation 和 inference

核心训练关系：

```text
与环境交互产生 transition
→ Replay Buffer采样batch
→ Online Network计算Q(s, a)
→ Target Network计算TD target
→ TD loss
→ 更新Online Network
→ 定期同步Target Network
```

完成 DQN 后，再增加 PPO 作为强化学习进阶任务；PPO 应重点练习 trajectory、return、advantage、policy/value network、importance ratio 和 clipped objective。

### 14. DDPM 扩散模型

建议脚本：`14_ddpm_mnist.py`

建议数据：复用 torchvision MNIST，并使用小型 time-conditioned U-Net 预测噪声。

完整流程需要包含：

- beta schedule、alpha 和 cumulative alpha 的预计算
- 为每张图像随机采样 diffusion timestep
- forward diffusion：从干净图像生成带噪图像
- timestep embedding
- 小型 U-Net 噪声预测模型
- 使用真实噪声与预测噪声计算 MSE
- 保存和加载最佳 checkpoint
- reverse diffusion：从纯噪声逐步生成图像
- 保存每轮生成样本和最终生成结果

核心训练关系：

```text
干净图像x0 + 随机时间步t + 随机噪声epsilon
→ 构造带噪图像xt
→ 模型预测epsilon
→ MSE(predicted_epsilon, epsilon)
```

DDPM 完成后，再学习 latent diffusion、VAE latent space、text conditioning、cross-attention 和 classifier-free guidance。

### 进入第三层的标准

完成下面的目标后再开始第三层：

- 能够独立完成至少3套第二层流程
- 能够根据论文描述设计 Dataset、target shape、prediction head、loss 和 metric
- 能够解释 U-Net 的 encoder、decoder、skip connection 和多尺度特征
- 能够独立处理 padding、mask、预训练权重、冻结与微调
- 能够运行并比较多个 seed、baseline 和简单 ablation

第三层建议顺序：DQN → DDPM → PPO；先完成 U-Net 再实现 DDPM，因为 DDPM 的噪声预测网络会复用 U-Net 的多尺度结构。

## 后续任务的统一完成标准

以后从 README 继续实现第二层或第三层任务时，每完成一套流程都要同时完成下面内容：

1. 新增编号训练脚本和对应公开数据下载脚本。
2. 保持 `imports → config → dataloader → model → training → evaluation → checkpoint → inference` 结构。
3. 在关键位置标注单个样本和 batch 的 tensor shape。
4. 只使用训练集拟合标准化参数、词表、阈值或其他统计量。
5. 至少输出一个适合当前任务的 test metric 和一组 inference 结果。
6. 使用环境变量或清晰配置切换有学习价值的任务变体，但避免把训练闭环完全不同的任务强行塞进同一脚本。
7. 更新 `requirements.txt`、运行命令、数据说明和 README 任务状态。
8. 先用较小 epoch 或数据子集完成 smoke test，再运行完整训练。

## 训练流程与公开数据对应关系

下面每个下载脚本会把公开数据保存到 `data/`。训练脚本会直接读取这些数据；如果文件不存在，会提示需要执行的下载命令：

| 流程 | 公开数据 | 下载命令 |
|---|---|---|
| MLP 表格分类/回归 | UCI Wine | `python download_data/download_01_mlp_wine.py` |
| CNN 图像分类 | MNIST | `python download_data/download_02_cnn_mnist.py` |
| LSTM 时序预测 | AirPassengers（两列 CSV） | `python download_data/download_03_lstm_air_passengers.py` |
| Transformer 多步、多目标预测 | ETTh1（规整 CSV） | `python download_data/download_04_transformer_etth1.py` |
| 普通/去噪 Autoencoder 异常检测 | UCI Wisconsin Breast Cancer | `python download_data/download_05_autoencoder_breast_cancer.py` |
| RNN 固定/可变长度序列分类 | UCR SyntheticControl（600条） | `python download_data/download_06_rnn_synthetic_control.py` |
| GNN 节点分类 | LINQS Cora | `python download_data/download_07_gnn_cora.py` |

所有训练脚本都只使用训练集统计量进行标准化，避免验证集和测试集信息泄漏。时序数据按照时间顺序划分，分类数据则采用固定随机种子进行可复现的划分。

建议练习顺序：先运行并逐段理解；第二天只看模块标题重写；第三天从空白文件重写；第四天修改数据维度、类别数或预测目标。
