# 第二阶段：手写 ResNet 与 Informer

## 运行

从仓库根目录执行：

```powershell
git pull
conda activate py311-dl-basic
python download_data/download_15_resnet_cifar10.py
python 15_resnet_image_classification.py
python download_data/download_16_informer_etth1.py
python 16_informer_time_series_forecast.py
```

复用现有 requirements.txt，无新增依赖。下载脚本只在准备数据时运行，训练脚本不自动下载。两者默认10个epoch；可修改 EPOCHS 或在 PowerShell 设置 `$env:EPOCHS="1"`。每个流程都有独立 build_dataloaders、run_epoch、checkpoint 和推理示例。训练第一批及每100批打印进度。

## 15：ResNet-18，CIFAR-10，从零训练

[代码](../15_resnet_image_classification.py)

手写 BasicBlock 与四组残差阶段，不调用 torchvision.models.resnet18。与11号预训练迁移学习流程分开练习。

| 步骤 | shape |
|---|---|
| images | [B,3,32,32] |
| stem | [B,64,32,32] |
| layer1：2个BasicBlock | [B,64,32,32] |
| layer2：2个BasicBlock | [B,128,16,16] |
| layer3：2个BasicBlock | [B,256,8,8] |
| layer4：2个BasicBlock | [B,512,4,4] |
| pool + flatten | [B,512] |
| head | [B,10] |

每个 BasicBlock：Conv → BN → ReLU → Conv → BN → 加上 shortcut → ReLU。通道或尺寸变化时，shortcut 使用1×1卷积和BN；其余使用Identity。

相加要求两个张量shape相同。与U-Net的cat不同，相加不增加通道数。

这里采用CIFAR输入层：3×3、stride=1、不加输入MaxPool；原ImageNet ResNet-18采用7×7、stride=2和MaxPool。这是保持18层主干结构的CIFAR适配版，不是原论文CIFAR ResNet-20/32等结构。

训练/验证从官方50000张训练图片中固定划分45000/5000。训练有随机裁剪与翻转，验证和测试没有随机增强；官方测试集只用于最后评估。使用CrossEntropyLoss、SGD momentum、余弦学习率调度，按验证loss保存最佳模型。输出accuracy及5张测试图片的预测类别。

权重：`checkpoints/16_resnet18.pt`。10个epoch用于流程练习，不代表训练已收敛或论文精度。

## 16：Informer，多变量多步预测

[代码](../16_informer_time_series_forecast.py)

实现三个主要结构思想：

1. **ProbSparse编码器注意力**：采样少量key估计各query的稀疏性，选top query，只有选中query计算对全部key的注意力；其余query使用value平均作为初始context。
2. **Self-attention distilling**：编码器层之间使用Conv1d → BN → ELU → MaxPool1d，序列长度96变成48。
3. **一次生成整个未来区间**：decoder输入为已知历史48步加未来24步零占位，单次forward输出未来24步；不循环逐步生成，也不使用真实未来值作为decoder输入。

默认输入7个ETTh1变量，输出HUFL与OT；历史96步，label_len=48，horizon=24，d_model=64，4个head，2个encoder层，1个decoder层。

| 步骤 | shape |
|---|---|
| encoder_values | [B,96,7] |
| encoder_time | [B,96,8] |
| encoder_embedding / encoder1 | [B,96,64] |
| distilling / encoder2 | [B,48,64] |
| decoder_values：48步历史+24步零 | [B,72,7] |
| decoder_time | [B,72,8] |
| decoder_embedding / decoder | [B,72,64] |
| head | [B,72,2] |
| 最后24步 | [B,24,2] |

时间特征为月、日、星期、小时的sin/cos，共8维；未来日历时间是已知信息，可以使用，未来观测值不能使用。

### 教学版与论文配置的区别

- 保留真正的ProbSparse编码器计算，不把普通Transformer换个名字。
- decoder自注意力使用完整因果注意力，cross-attention使用完整注意力，以便明确演示mask及encoder-decoder关系。因此不能把整个实现的时间复杂度都称为O(L log L)：该复杂度优势主要属于ProbSparse部分。
- value embedding使用Linear而非官方的卷积token embedding，时间embedding使用周期特征Linear。
- 采用紧凑模型、70%/15%/15%按时间划分、多目标HUFL/OT，未采用论文全部实验配置、官方固定月份划分、InformerStack或超参数搜索。
- 默认96→24便于对比已有基础Transformer；较短序列下top-query比例可能较大，不能据此证明长序列效率优势。之后可增大LOOKBACK/HORIZON探索。
- 训练时随机采样key；评估时使用固定局部随机种子，保持评分和checkpoint比较稳定，不重置外部随机状态。

### 数据泄漏与训练

按时间先划分边界，仅用训练区间计算mean/std。验证和测试借用边界前96步作历史context，但全部预测标签都位于各自区间内。

decoder的前48步来自预测起点以前的历史，后24步始终为零。未来真实值只用于targets和loss。

MSE用于训练，报告标准化尺度的MSE与MAE。保存最佳val_mse；checkpoint同时包含标准化统计、输入输出变量名和窗口配置。推理示例加载保存的统计量，逆标准化并按时间打印未来24步预测和真实值。

权重：`checkpoints/17_informer.pt`。不同变量量纲不同，因此主要使用标准化指标比较；打印值恢复为各目标原单位。若更改窗口或模型配置，需要使用相应配置重新训练。

## 验证范围

开发时检查前向shape、loss反向传播、参数更新、checkpoint重载，以及Informer零占位、窗口标签、因果mask和重复评估一致性。另使用真实CIFAR-10和ETTh1各4个样本运行了1个epoch的main闭环，覆盖训练、验证、最佳checkpoint加载、测试和推理。完整数据集训练精度需在目标设备训练后评估，不能由这些检查推出。

## 原论文与官方实现

- [ResNet论文](https://arxiv.org/abs/1512.03385)
- [Informer论文](https://arxiv.org/abs/2012.07436)
- [Informer官方实现](https://github.com/zhouhaoyi/Informer2020)
- [ETTh1数据](https://github.com/zhouhaoyi/ETDataset)
