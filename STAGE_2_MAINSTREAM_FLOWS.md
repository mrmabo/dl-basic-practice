# 第二阶段：主流深度学习任务（08–15）

## 目标

第二阶段继续保持“一个 Python 文件覆盖一个完整流程”。重点不是多文件工程架构，而是覆盖更复杂、更常见的任务输入、标签、输出头、loss、metric 和推理方式。

## 已实现流程

| 编号 | 项目 | 公开数据 | 新能力 |
|---|---|---|---|
| 08 | CNN 图像回归 | MNIST | 连续输出、MSE、MAE、RMSE |
| 09 | U-Net 图像分割 | Oxford-IIIT Pet | image/mask、skip connection、Dice |
| 10 | Embedding + LSTM 文本分类 | UCI SMS Spam | tokenize、vocabulary、padding、collate_fn |
| 11 | ResNet18 迁移学习 | CIFAR-10 | 预训练权重、冻结、解冻、分组学习率 |
| 12 | 多变量多步时序预测 | ETTh1 | 历史96步预测未来24步、逆标准化 |

下载脚本位于 `download_data/download_08_*.py` 至 `download_data/download_12_*.py`。

## 计划补充的主流流程

| 编号 | 项目 | 重点 |
|---|---|---|
| 13 | Faster R-CNN 目标检测 | bounding box、可变数量标签、NMS、IoU/mAP、预测框可视化 |
| 14 | 预训练 Transformer 文本微调 | tokenizer、attention_mask、fine-tuning、原始文本推理 |
| 15 | 简化 DDPM 图像生成 | noise schedule、timestep embedding、噪声预测、反向采样 |

完成01–15后，将覆盖表格、图像分类/回归/分割/检测、文本序列与预训练模型、单步与多步时序、异常检测、图神经网络、迁移学习和生成模型等主要任务范式。

## 工程能力的加入方式

保持单文件，但让不同流程分别练习少量工程能力：

- custom `collate_fn`
- data augmentation
- scheduler
- early stopping
- gradient clipping
- AMP mixed precision
- checkpoint resume
- 预测结果可视化

不要求每个脚本都重复全部工程代码。

## 期望达到的效果

完成第二阶段后，应当能够：

- 根据任务设计 Dataset、标签、输出头、loss 和 metric
- 使用预训练模型并完成冻结、解冻和 fine-tuning
- 处理 segmentation mask、bounding box 和不等长文本
- 完成多目标、多步预测和逆标准化
- 从原始图片、文本或时序数据开始执行 inference
- 独立修改公开流程以适配新的同类型数据集
