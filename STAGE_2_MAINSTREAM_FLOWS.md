# 第二阶段：主流深度学习任务（08–15）

## 目标

第二阶段继续保持“一个 Python 文件覆盖一个完整流程”。重点不是多文件工程架构，而是掌握更复杂、更常见的任务输入、标签、输出头、loss、metric 和推理方式。

## 已实现流程

| 编号 | 项目 | 公开数据 | 新能力 |
|---|---|---|---|
| 08 | CNN 图像回归 | MNIST | 连续输出、MSE、MAE、RMSE |
| 09 | U-Net 图像分割 | Oxford-IIIT Pet | image/mask、skip connection、Dice |
| 10 | Embedding + LSTM 文本分类 | UCI SMS Spam | tokenize、vocabulary、padding、collate_fn |
| 11 | ResNet18 迁移学习 | CIFAR-10 | 预训练权重、冻结、解冻、分组学习率 |
| 12 | 多变量多步时序预测 | ETTh1 | 历史96步预测未来24步、逆标准化 |
| 13 | Faster R-CNN 目标检测 | Penn-Fudan Pedestrian | bounding box、可变长度标签、IoU、预测框可视化 |
| 14 | DistilBERT 文本微调 | UCI SMS Spam | tokenizer、attention mask、动态padding、预训练模型微调 |
| 15 | DDPM 图像生成 | MNIST | noise schedule、timestep embedding、噪声预测、反向采样 |

完成01–15后，练习范围覆盖表格、图像分类/回归/分割/检测、文本序列与预训练模型、单步与多步时序、异常检测、图神经网络、迁移学习和生成模型等主要任务范式。

## 下载与运行

所有命令从仓库根目录运行：

```bash
python download_data/download_08_mnist.py
python 08_cnn_image_regression.py

python download_data/download_09_oxford_pet.py
python 09_unet_image_segmentation.py

python download_data/download_10_sms_spam.py
python 10_lstm_text_classification.py

python download_data/download_11_cifar10.py
python 11_transfer_learning_cifar10.py

python download_data/download_12_etth1.py
python 12_multistep_time_series_forecast.py

python download_data/download_13_penn_fudan.py
python 13_faster_rcnn_object_detection.py

python download_data/download_14_transformer_sms.py
python 14_transformer_text_finetuning.py

python download_data/download_15_ddpm_mnist.py
python 15_ddpm_image_generation.py
```

## 13：目标检测需要掌握

```text
image + variable number of boxes + labels
-> Faster R-CNN
-> classification/box losses
-> confidence filtering
-> IoU matching
-> precision/recall/F1
-> draw predicted boxes
```

Penn-Fudan 很小，适合流程练习。代码中的指标是便于学习的 IoU@0.5 匹配指标，不是完整 COCO mAP 实现。

## 14：预训练 Transformer 需要掌握

```text
raw text
-> pretrained tokenizer
-> input_ids + attention_mask
-> dynamic padding
-> DistilBERT
-> classification head
-> fine-tuning
-> raw-text inference
```

该流程还使用 class weight 处理 SMS Spam 的类别不平衡，并使用 gradient clipping。

## 15：DDPM 需要掌握

```text
clean image + random timestep + sampled noise
-> forward diffusion
-> model predicts noise
-> noise MSE
-> reverse diffusion sampling
-> generated image grid
```

这是教学用紧凑 DDPM，重点是理解扩散流程，不以生成高质量图片为目标。

## 工程能力的加入方式

保持单文件，但让不同流程分别练习少量工程能力：

- 09：像素级 metric
- 10：custom `collate_fn`
- 11：冻结、解冻、分组 learning rate
- 12：多步输出与逆标准化
- 13：scheduler、gradient clipping、可视化
- 14：动态 padding、类别不平衡、gradient clipping
- 15：训练/验证噪声损失、迭代生成、结果保存

不要求每个脚本重复全部工程代码。

## 期望达到的效果

完成第二阶段后，应当能够：

- 根据任务设计 Dataset、标签、输出头、loss 和 metric
- 使用预训练模型并完成冻结、解冻和 fine-tuning
- 处理 segmentation mask、bounding box 和不等长文本
- 完成多目标、多步预测和逆标准化
- 理解判别模型与生成模型的训练差异
- 从原始图片、文本或时序数据开始执行 inference
- 独立修改公开流程以适配新的同类型数据集
