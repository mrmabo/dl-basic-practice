# 第二阶段：主流深度学习任务（08–17）

## 目标

第二阶段继续保持“一个 Python 文件覆盖一个完整流程”。重点不是多文件工程架构，而是掌握更复杂、更常见的任务输入、标签、输出头、loss、metric 和推理方式。

## 已实现流程

| 编号 | 项目 | 公开数据 | 新能力 |
|---|---|---|---|
| 08 | U-Net 图像分割 | Oxford-IIIT Pet | image/mask、skip connection、Dice |
| 09 | Embedding + LSTM 文本分类 | UCI SMS Spam | tokenize、vocabulary、padding、collate_fn |
| 10 | Mini GPT 英文文本生成 | Tiny Shakespeare（约1.1 MB） | 因果注意力、下一token预测、自回归生成 |
| 11 | ResNet18 迁移学习 | CIFAR-10 | 预训练权重、冻结、解冻、分组学习率 |
| 12 | 多变量多步时序预测 | ETTh1 | 历史96步预测未来24步、逆标准化 |
| 13 | Faster R-CNN 目标检测 | Penn-Fudan Pedestrian | bounding box、可变长度标签、IoU、预测框可视化 |
| 14 | DistilBERT 文本微调 | UCI SMS Spam | tokenizer、attention mask、动态padding、预训练模型微调 |
| 15 | DDPM 图像生成 | MNIST | noise schedule、timestep embedding、噪声预测、反向采样 |
| 16 | 手写 ResNet-18 图像分类 | CIFAR-10 | BasicBlock、相加shortcut、投影shortcut、从零训练 |
| 17 | Informer 多步时序预测 | ETTh1 | ProbSparse、distilling、因果decoder、未来零占位 |

完成01–17后，练习范围覆盖表格、图像分类/分割/检测、文本序列与预训练模型、单步与多步时序、异常检测、图神经网络、迁移学习和生成模型等主要任务范式。

## 下载与运行

所有命令从仓库根目录运行：

```bash
python download_data/download_08_oxford_pet.py
python 08_unet_image_segmentation.py

python download_data/download_09_sms_spam.py
python 09_lstm_text_classification.py

python download_data/download_10_tiny_shakespeare.py
python 10_mini_gpt_text_generation.py

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

python download_data/download_16_resnet_cifar10.py
python 16_resnet_image_classification.py

python download_data/download_17_informer_etth1.py
python 17_informer_time_series_forecast.py
```

## 08：U-Net 结构

08 现在使用四次下采样、底部卷积块、四次上采样和四组对应层的 skip concatenation；通道从 64 增长到 1024 再逐层缩小。输入为 [B, 3, 128, 128]，输出为 [B, 1, 128, 128] 的二分类 logits。

这是针对 Oxford-IIIT Pet 图片的完整对称 U-Net 结构。原论文使用无 padding 卷积，并裁剪编码器特征图后再拼接；本练习采用 padding=1，便于和同尺寸的标签直接计算 BCE loss。原论文的数据增强、边界加权损失及 overlap-tile 推理未在本流程实现。模型更大，默认 batch size 改为 2，权重保存在 `08_unet_full.pt`。

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

- 08：像素级 metric
- 09：custom `collate_fn`
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

## 16–17：模型结构进阶

新增流程均有独立 build_dataloaders、训练、评估、checkpoint 和推理。详细结构、shape、数据边界和论文配置区别见 [ResNet 与 Informer 练习](docs/ResNet_Informer_Practice.md)。16保留11的预训练迁移学习流程，另行手写残差主干；17在编码器实现ProbSparse，在解码器使用完整因果注意力，属于紧凑教学版，不声称复现论文精度。

## 10：从零训练 Mini GPT

这个流程使用小型公开英文语料 Tiny Shakespeare（约1.1 MB）。字符级 tokenizer 无需安装新的依赖；模型约63万参数，3层、4个attention heads、128维embedding、128字符上下文。显存不足时减小 batch size；CPU也能运行，但完整训练较慢。

```bash
python download_data/download_10_tiny_shakespeare.py
python 10_mini_gpt_text_generation.py
# 先用一轮检查流程，正式训练默认10轮
python 10_mini_gpt_text_generation.py --epochs 1 --batch-size 8
# 加载最佳验证权重，交互式输入英文提示
python 10_mini_gpt_text_generation.py --generate-only --interactive
python 10_mini_gpt_text_generation.py --generate-only --prompt "ROMEO:" --temperature 0.8
```

训练文本按连续区间划分90%/5%/5%，然后在各自区间构建窗口，避免窗口跨越划分边界。词表只使用训练文本，验证/测试的未知字符映射为空格。输入和标签为[B,T]，标签右移一位；输出[B,T,V]，展平后计算CrossEntropyLoss。手写Q/K/V、多头拆分、缩放点积和下三角因果遮罩；每个block使用LayerNorm、残差连接和FFN。验证loss选择checkpoint，测试报告loss和perplexity。

生成时只取最后位置的logits，使用temperature和top-k采样，将新字符拼回输入；上下文超过128字符时截取末尾128个字符。checkpoint同时保存词表和模型配置。输入字符必须在词表内。

这是教学用GPT文本续写器。交互界面不代表模型经过聊天训练，它会模仿莎士比亚的语言风格，不保证回答问题，也不保留多轮会话。想进一步做指令聊天，需要对话数据和通常更大的预训练模型。

练习：解释为什么标签右移、因果遮罩如何阻止偷看答案、分类输出[B,C]与语言模型输出[B,T,V]的区别，以及为什么推理要循环采样而训练能并行预测全部位置。

验证：下载真实语料成功；检查默认模型的输出shape、标签右移、因果性与参数更新。用真实语料前20000字符和缩小模型跑通1轮train/val/test、checkpoint重载与生成；未运行默认模型完整10轮训练，生成质量需在本机训练后评估。
