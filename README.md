# Deep Learning Basic Practice

这个仓库按照“基础训练闭环 -> 应用型项目 -> 论文复现”分成三个阶段：

| 目录 | 阶段 | 目标 |
|---|---|---|
| [`stage1_basic_flows/`](stage1_basic_flows/) | 第一阶段 | 熟练写出 Dataset、DataLoader、训练、验证、checkpoint 和 inference |
| [`stage2_applied_projects/`](stage2_applied_projects/) | 第二阶段 | 掌握回归、分割、NLP、迁移学习和多步预测 |
| [`stage3_paper_reproduction/`](stage3_paper_reproduction/) | 第三阶段 | 从论文实现模型，完成对比、消融和可视化 |

## 1. 创建 Conda 环境

推荐 Python 3.11：

```bash
conda create -n py311-dl-basic python=3.11 -y
conda activate py311-dl-basic
```

确认当前解释器：

```bash
python --version
python -c "import sys; print(sys.executable)"
```

## 2. 安装依赖

进入仓库根目录：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果希望在 NVIDIA GPU 上训练，应根据 PyTorch 官方安装页面选择与本机驱动匹配的安装命令，然后再安装其余依赖。验证环境：

```bash
python -c "import torch, torchvision, numpy, sklearn; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 3. 开始训练

建议先完成第一阶段，再进入第二阶段。每个阶段的 README 包含数据下载和运行命令：

- [第一阶段说明](stage1_basic_flows/README.md)
- [第二阶段说明](stage2_applied_projects/README.md)
- [第三阶段说明](stage3_paper_reproduction/README.md)

示例：

```bash
python stage1_basic_flows/download_data/download_01_mlp_wine.py
python stage1_basic_flows/01_mlp_classification.py
```

第二阶段示例：

```bash
python stage2_applied_projects/download_data/download_10_sms_spam.py
python stage2_applied_projects/10_lstm_text_classification.py
```

所有数据、checkpoint 和模型权重都被 `.gitignore` 排除，不会提交到 GitHub。

## 推荐练习节奏

```text
运行并理解 -> 只看标题重写 -> 从空白文件重写 -> 修改任务 -> 记录实验
```

不要把“记住所有 API”当作目标。目标是能够独立组织数据、追踪 tensor shape、完成训练闭环，并能定位 shape、dtype、device 和 data leakage 问题。
