# Deep Learning Basic Practice

这个仓库使用“一个 Python 文件对应一个完整流程”的方式训练深度学习编码能力。所有训练流程按编号放在根目录，所有公开数据下载脚本统一放在 `download_data/`。

## 学习阶段

- [第一阶段：基础训练闭环](STAGE_1_BASIC_FLOWS.md)：01–07
- [第二阶段：主流深度学习任务](STAGE_2_MAINSTREAM_FLOWS.md)：08–14
- [第三阶段：强化学习流程](STAGE_3_REINFORCEMENT_LEARNING.md)：未来实现

## 创建 Conda 环境

推荐 Python 3.11：

```bash
conda create -n py311-dl-basic python=3.11 -y
conda activate py311-dl-basic
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

检查环境：

```bash
python --version
python -c "import sys, torch, torchvision; print(sys.executable); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 下载数据与开始训练

所有命令都从仓库根目录运行。示例：

```bash
python download_data/download_01_mlp_wine.py
python 01_mlp_classification.py
```

第二阶段示例：

```bash
python download_data/download_08_oxford_pet.py
python 08_unet_image_segmentation.py

# 第二阶段后续流程示例
python download_data/download_12_penn_fudan.py
python 12_faster_rcnn_object_detection.py
```

## RNN 序列分类：切换长度模式

在 `06_rnn_sequence_classification.py` 文件顶部设置 `VARIABLE_LENGTH = False`（固定长度）或 `VARIABLE_LENGTH = True`（随机截短原本等长的序列，用于练习可变长度）。需要调整训练轮数时直接修改 `EPOCHS`。保存文件后从仓库根目录运行：

```powershell
python .\download_data\download_06_rnn_synthetic_control.py
python .\06_rnn_sequence_classification.py
```

启动时会输出 `mode=fixed` 或 `mode=variable` 及部分训练样本的真实长度；训练结束会打印测试样本的 `lengths`。两个模式分别保存 `best_rnn_fixed.pt` 和 `best_rnn_variable.pt`。

## 当前流程

| 编号 | 流程 | 阶段 |
|---|---|---|
| 01 | MLP 表格分类 | 第一阶段 |
| 02 | CNN 图像分类 | 第一阶段 |
| 03 | LSTM 单步时序预测 | 第一阶段 |
| 04 | Transformer 时序预测 | 第一阶段 |
| 05 | Autoencoder 异常检测 | 第一阶段 |
| 06 | RNN 序列分类 | 第一阶段 |
| 07 | GNN 节点分类 | 第一阶段 |
| 08 | U-Net 图像分割 | 第二阶段 |
| 09 | Embedding + LSTM 文本分类 | 第二阶段 |
| 10 | ResNet18 迁移学习 | 第二阶段 |
| 11 | 多变量多步时序预测 | 第二阶段 |
| 12 | Faster R-CNN 目标检测 | 第二阶段 |
| 13 | DistilBERT 文本微调 | 第二阶段 |
| 14 | DDPM 图像生成 | 第二阶段 |

## 练习方法

每天新建一个空白 Python 文件，只保留：

```python
# imports
# config
# data loading
# dataset and dataloader
# model
# loss and optimizer
# training
# evaluation
# checkpoint
# inference
```

先独立完成，再与编号脚本对照。重点检查 tensor shape、dtype、device、数据划分和 data leakage。数据、checkpoint 和模型权重均由 `.gitignore` 排除。
