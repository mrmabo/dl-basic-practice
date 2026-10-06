# 第二阶段：主流深度学习任务（08–17）

## 目标

第二阶段继续保持“一个 Python 文件覆盖一个完整流程”。重点不是多文件工程架构，而是掌握更复杂、更常见的任务输入、标签、输出头、loss、metric 和推理方式。

## 已实现流程

| 编号 | 项目 | 公开数据 | 新能力 |
|---|---|---|---|
| 08 | U-Net 图像分割 | Oxford-IIIT Pet | image/mask、skip connection、Dice |
| 09 | Embedding + LSTM 文本分类 | UCI SMS Spam | tokenize、vocabulary、padding、collate_fn |
| 10 | ResNet18 迁移学习 | CIFAR-10 | 预训练权重、冻结、解冻、分组学习率 |
| 11 | 多变量多步时序预测 | ETTh1 | 历史96步预测未来24步、逆标准化 |
| 12 | Faster R-CNN 目标检测 | Penn-Fudan Pedestrian | bounding box、可变长度标签、IoU、预测框可视化 |
| 13 | DistilBERT 文本微调 | UCI SMS Spam | tokenizer、attention mask、动态padding、预训练模型微调 |
| 14 | DDPM 图像生成 | MNIST | noise schedule、timestep embedding、噪声预测、反向采样 |
| 15 | 手写 ResNet-18 图像分类 | CIFAR-10 | BasicBlock、相加shortcut、投影shortcut、从零训练 |
| 16 | Informer 多步时序预测 | ETTh1 | ProbSparse、distilling、因果decoder、未来零占位 |
| 17 | Mini GPT 英文聊天 | DailyDialog（默认5000组训练问答） | 因果注意力、下一token预测、自回归生成 |

完成01–17后，练习范围覆盖表格、图像分类/分割/检测、文本序列与预训练模型、单步与多步时序、异常检测、图神经网络、迁移学习和生成模型等主要任务范式。

## 下载与运行

所有命令从仓库根目录运行：

```bash
python download_data/download_08_oxford_pet.py
python 08_unet_image_segmentation.py

python download_data/download_09_sms_spam.py
python 09_lstm_text_classification.py

python download_data/download_10_cifar10.py
python 10_transfer_learning_cifar10.py

python download_data/download_11_etth1.py
python 11_multistep_time_series_forecast.py

python download_data/download_12_penn_fudan.py
python 12_faster_rcnn_object_detection.py

python download_data/download_13_transformer_sms.py
python 13_transformer_text_finetuning.py

python download_data/download_14_ddpm_mnist.py
python 14_ddpm_image_generation.py

python download_data/download_15_resnet_cifar10.py
python 15_resnet_image_classification.py

python download_data/download_16_informer_etth1.py
python 16_informer_time_series_forecast.py

python download_data/download_17_dailydialog.py
python 17_mini_gpt_text_generation.py
```

## 08：U-Net 结构

08 现在使用四次下采样、底部卷积块、四次上采样和四组对应层的 skip concatenation；通道从 64 增长到 1024 再逐层缩小。输入为 [B, 3, 128, 128]，输出为 [B, 1, 128, 128] 的二分类 logits。

这是针对 Oxford-IIIT Pet 图片的完整对称 U-Net 结构。原论文使用无 padding 卷积，并裁剪编码器特征图后再拼接；本练习采用 padding=1，便于和同尺寸的标签直接计算 BCE loss。原论文的数据增强、边界加权损失及 overlap-tile 推理未在本流程实现。模型更大，默认 batch size 改为 2，权重保存在 `08_unet_full.pt`。

## 12：目标检测需要掌握

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

## 13：预训练 Transformer 需要掌握

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

## 14：DDPM 需要掌握

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
- 10：冻结、解冻、分组 learning rate
- 11：多步输出与逆标准化
- 12：scheduler、gradient clipping、可视化
- 13：动态 padding、类别不平衡、gradient clipping
- 14：训练/验证噪声损失、迭代生成、结果保存

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

## 15–16：模型结构进阶

新增流程均有独立 build_dataloaders、训练、评估、checkpoint 和推理。详细结构、shape、数据边界和论文配置区别见 [ResNet 与 Informer 练习](docs/ResNet_Informer_Practice.md)。15保留10的预训练迁移学习流程，另行手写残差主干；16在编码器实现ProbSparse，在解码器使用完整因果注意力，属于紧凑教学版，不声称复现论文精度。

## 17：DailyDialog Mini GPT英文聊天

数据来源：[DailyDialog论文](https://aclanthology.org/I17-1099/)、[ConvLab镜像](https://huggingface.co/datasets/ConvLab/dailydialog)。镜像标注CC BY-NC-SA 4.0，供非商业学习使用。下载压缩包约3.7 MB，仅处理JSON，不运行远程代码。

保留官方train/validation/test对话划分，再提取相邻发言作为问题/回答。仅保留问题不超过96字符、回答不超过150字符的完整短对话。固定种子抽样，默认5000/500/500组问答，同一对话不会跨split。DailyDialog是人与人日常对话，不是知识助手指令数据。

```bash
python download_data/download_17_dailydialog.py
python 17_mini_gpt_text_generation.py
# 快速验证，或调整数据规模
python 17_mini_gpt_text_generation.py --epochs 1 --batch-size 8
python download_data/download_17_dailydialog.py --train-pairs 2000 --eval-pairs 200
# 加载最佳权重交互
python 17_mini_gpt_text_generation.py --generate-only --interactive
python 17_mini_gpt_text_generation.py --generate-only --prompt "Hello, how are you?"
```

字符词表只使用训练集。PAD/UNK/BOS/SEP/EOS是五个独立token。训练序列为 `[BOS] question [SEP] answer [EOS]`，输入和标签错开一位；问题部分与padding标签为-100，CrossEntropyLoss只计算回答字符及EOS，平均loss按有效标签数统计。

输入/标签[B,T]，logits[B,T,V]；256字符上下文、128维embedding、4个heads、3层，约65万参数。Q/K/V、缩放点积、下三角因果遮罩、LayerNorm、残差和FFN均在单文件实现。右侧padding无需额外注意力遮罩：因果注意力使有效位置看不到后面的PAD，padding标签也不参与loss。

推理构建[BOS]+问题+[SEP]，使用temperature和top-k逐字符采样。禁止生成PAD/UNK/BOS/SEP，遇EOS结束；只显示回答。过长问题截取前96字符，未知字符映射UNK。每次提问独立。模型未预训练，不能保证理解任意问题，也不保证事实正确。

checkpoint保存模型、字符词表、配置与格式标识，文件为 `checkpoints/10_mini_gpt_dailydialog.pt`。旧莎士比亚权重不能复用。普通训练命令从零开始，不会续训已有权重。

测试提示：`Hello, how are you?`、`What do you do on weekends?`、`Would you like some coffee?`。关注拼写、句子通顺程度及与问题的相关性，结合验证loss观察进步。

验证：真实DailyDialog下载及5000/500/500抽样通过；检查split隔离、回答标签mask、因果性、参数更新和EOS停止。用各32组真实问答及缩小模型跑通一轮训练、测试、checkpoint重载和生成；未进行默认模型完整10轮训练。
