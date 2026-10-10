# Neural Network Building Blocks

这个目录保存手动实现的经典神经网络核心模块。每个模块都应当：

- 可以独立运行并检查 tensor shape
- 包含关键步骤和 shape 注释
- 尽量只依赖基础 PyTorch API
- 重点展示内部原理，而不是替代 PyTorch 的生产级实现

完整的数据处理、训练、验证、checkpoint 和测试流程继续放在仓库根目录。

## 已实现模块

| 文件 | 模块 | 核心练习内容 |
|---|---|---|
| `multi_head_self_attention.py` | Multi-Head Self-Attention | Q/K/V projection、scaled dot-product attention、拆分与拼接多个 head、attention mask |
| `UNet_Blocks.py` | U-Net DoubleConv / Down / Up | 两次卷积、池化下采样、转置卷积上采样与 skip 拼接；Up 要求空间尺寸匹配 |

运行示例：

```bash
python building_blocks/multi_head_self_attention.py
```

预期 shape：

```text
input:              [B, S, D]
Q / K / V:          [B, H, S, D/H]
attention weights:  [B, H, S, S]
output:             [B, S, D]
```

学习笔记：[多头自注意力：20 个问题与答案](doc/multi_head_self_attention_notes.md)。

## 按功能分类的 Blocks 与论文索引

- `[x]` 表示已实现，`[ ]` 表示待实现。所有原有条目均保留。
- 分类按模块的主要功能组织；一个模块可能同时承担多种功能，每项只列一次。
- 链接优先提供原始论文；通用组件、工程组合或家族名称使用基础论文或代表性实现，并注明关系，不能将参考论文理解为该名称的独立首创论文。
- 同一论文可以对应多个 block；论文设计与本仓库的简化实现可能存在差异。
- DOI 链接可能需要出版商访问权限；arXiv 页面通常可直接下载 PDF。

英文标题说明每组的功能；分组文件名采用 `snake_case`，可用于将同组 blocks 放在一个 `.py` 文件中。如果每个 block 单独建文件，则使用模块名称，例如 `batch_normalization.py` 或 `resnet_basic_block.py`。

建议练习顺序：先读结构图和对应公式 → 标注输入输出 shape → 空白页实现 → 验证行为与梯度 → 回看论文核对。无需每次先读完整篇论文。

### 特征变换与表示 / Feature Transformation and Representation

建议分组文件名：`feature_transformation.py`。

- [ ] MLP / Dense Block — [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0)（基础参考，非独立 block 论文）
- [ ] Conv-Norm-Activation Block — [Batch Normalization](https://arxiv.org/abs/1502.03167)（通用组合参考）
- [ ] Embedding / Embedding Projection — [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)
- [ ] Gated Linear Unit（GLU） — [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)
- [ ] GEGLU Feed-Forward Block — [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [ ] SwiGLU Feed-Forward Block — [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [ ] Position-wise Feed-Forward Network（FFN） — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] MLP-Mixer Block — [MLP-Mixer](https://arxiv.org/abs/2105.01601)

### 归一化与训练尺度控制 / Normalization and Training Scale Control

建议分组文件名：`normalization.py`。

- [ ] Batch Normalization — [Batch Normalization](https://arxiv.org/abs/1502.03167)
- [ ] Layer Normalization — [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [ ] Group Normalization — [Group Normalization](https://arxiv.org/abs/1803.08494)
- [ ] Instance Normalization — [Instance Normalization](https://arxiv.org/abs/1607.08022)
- [ ] RMSNorm — [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [ ] Reversible Instance Normalization（RevIN） — [Reversible Instance Normalization](https://openreview.net/forum?id=cGDAkQo1C0p)

### 残差、密集连接与信息通路 / Residual and Dense Connections

建议分组文件名：`residual_and_dense_connections.py`。

- [ ] Residual / Add & Norm Block — [ResNet](https://arxiv.org/abs/1512.03385)（残差参考；Add & Norm 另见 Transformer）；[Transformer 的 Add & Norm](https://arxiv.org/abs/1706.03762)
- [ ] Highway Block — [Highway Networks](https://arxiv.org/abs/1505.00387)
- [ ] ResNet Basic Block — [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [ ] ResNet Bottleneck Block — [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [ ] Pre-Activation Residual Block — [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- [ ] Wide Residual Block — [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
- [ ] ResNeXt Block — [Aggregated Residual Transformations](https://arxiv.org/abs/1611.05431)
- [ ] DenseNet Dense Layer / Dense Block — [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [ ] DenseNet Transition Block — [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [ ] Inception-ResNet Block — [Inception-v4, Inception-ResNet](https://arxiv.org/abs/1602.07261)
- [ ] CSP Block（Cross Stage Partial） — [CSPNet](https://arxiv.org/abs/1911.11929)

### 局部特征提取与高效卷积 / Local Feature Extraction and Efficient Convolution

建议分组文件名：`efficient_convolution.py`。

- [ ] Inception Block — [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- [ ] Depthwise Separable Convolution — [MobileNets](https://arxiv.org/abs/1704.04861)
- [ ] Grouped Convolution Block — [ResNeXt](https://arxiv.org/abs/1611.05431)（代表性应用，非该算子的首创）
- [ ] Dilated Convolution Block — [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
- [ ] MobileNetV2 Inverted Residual / Linear Bottleneck — [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [ ] MobileNetV3 Block — [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [ ] MBConv Block — [EfficientNet](https://arxiv.org/abs/1905.11946)（代表性应用；基础设计另见 MobileNetV2）；[MobileNetV2](https://arxiv.org/abs/1801.04381)
- [ ] Fused-MBConv Block — [EfficientNetV2](https://arxiv.org/abs/2104.00298)
- [ ] ShuffleNet Unit / Channel Shuffle — [ShuffleNet](https://arxiv.org/abs/1707.01083)
- [ ] Fire Module（SqueezeNet） — [SqueezeNet](https://arxiv.org/abs/1602.07360)
- [ ] Ghost Module / Ghost Bottleneck — [GhostNet](https://arxiv.org/abs/1911.11907)
- [ ] ConvNeXt Block — [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [ ] C3 / C2f Block（YOLO） — [CSPNet](https://arxiv.org/abs/1911.11929)（设计基础；精确实现见官方源码，非独立原始论文）；[官方模块源码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py)
- [ ] RepVGG Block（结构重参数化） — [RepVGG](https://arxiv.org/abs/2101.03697)
- [ ] Deformable Convolution Block — [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)

### 通道、空间注意力与特征调制 / Channel and Spatial Attention and Feature Modulation

建议分组文件名：`attention_and_feature_modulation.py`。

- [ ] Squeeze-and-Excitation（SE）Block — [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [ ] CBAM Block — [CBAM](https://arxiv.org/abs/1807.06521)
- [ ] Efficient Channel Attention（ECA）Block — [ECA-Net](https://arxiv.org/abs/1910.03151)
- [ ] Coordinate Attention Block — [Coordinate Attention](https://arxiv.org/abs/2103.02907)
- [ ] Non-Local Block — [Non-local Neural Networks](https://arxiv.org/abs/1711.07971)
- [ ] Spatial Self-Attention Block — [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)（代表性空间注意力实现）
- [ ] Attention Gate（Attention U-Net） — [Attention U-Net](https://arxiv.org/abs/1804.03999)
- [ ] FiLM / Feature-wise Affine Modulation — [FiLM](https://arxiv.org/abs/1709.07871)
- [ ] Adaptive Instance Normalization（AdaIN） — [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
- [ ] StyleGAN Modulated / Demodulated Convolution — [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)

### 空间分辨率变换与编码器—解码器融合 / Spatial Resampling and Encoder–Decoder Fusion

建议分组文件名：`resampling_and_skip_fusion.py`。

- [x] U-Net DoubleConv Block — [U-Net](https://arxiv.org/abs/1505.04597)（论文使用无 padding 卷积；当前实现用 padding=1）
- [x] U-Net Down Block — [U-Net](https://arxiv.org/abs/1505.04597)
- [x] U-Net Up / Skip Fusion Block — [U-Net](https://arxiv.org/abs/1505.04597)（论文含裁剪 skip；当前实现要求空间尺寸一致）
- [ ] Residual U-Net Block — [Road Extraction by Deep Residual U-Net](https://arxiv.org/abs/1711.10684)
- [ ] Transposed Convolution Upsampling Block — [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)（通用算子的代表性应用）
- [ ] Interpolation + Convolution Upsampling Block — [Distill: Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)（技术文章，非独立 block 论文）
- [ ] PixelShuffle Upsampling Block — [Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)
- [ ] Autoencoder Encoder / Decoder Block — [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647)（基础参考，具体结构依任务）

### 多尺度聚合与特征金字塔 / Multi-Scale Aggregation and Feature Pyramids

建议分组文件名：`multi_scale_feature_aggregation.py`。

- [ ] Feature Pyramid Network（FPN） — [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [ ] Path Aggregation Network（PAN / PANet） — [Path Aggregation Network](https://arxiv.org/abs/1803.01534)
- [ ] Bidirectional Feature Pyramid Network（BiFPN） — [EfficientDet](https://arxiv.org/abs/1911.09070)
- [ ] Spatial Pyramid Pooling（SPP） — [Spatial Pyramid Pooling in Deep Convolutional Networks](https://arxiv.org/abs/1406.4729)
- [ ] Spatial Pyramid Pooling Fast（SPPF） — [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)（背景论文；SPPF 精确结构见官方源码）；[官方模块源码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py)
- [ ] Atrous Spatial Pyramid Pooling（ASPP） — [DeepLab](https://arxiv.org/abs/1606.00915)
- [ ] Pyramid Pooling Module（PPM） — [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
- [ ] Multi-scale Feature Fusion Block — [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)（通用类别的代表性参考）

### 序列匹配与注意力聚合 / Sequence Matching and Attention Aggregation

建议分组文件名：`sequence_attention.py`。

- [x] Multi-Head Self-Attention — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] Scaled Dot-Product Attention（独立基础组件） — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] Causal / Masked Multi-Head Self-Attention — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] Multi-Head Cross-Attention — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] Additive / Bahdanau Attention — [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [ ] Multiplicative / Luong Attention — [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [ ] Multi-Query Attention（MQA） — [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150)
- [ ] Grouped-Query Attention（GQA） — [GQA](https://arxiv.org/abs/2305.13245)
- [ ] Sliding-Window / Local Attention — [Longformer](https://arxiv.org/abs/2004.05150)（代表性局部注意力）
- [ ] Sparse Attention Block — [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
- [ ] Linear Attention Block — [Transformers are RNNs](https://arxiv.org/abs/2006.16236)
- [ ] KV Cache（自回归注意力的缓存组件） — [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150)（推理缓存参考，非独立网络 block）
- [ ] Temporal Attention Pooling — [Attentive Statistics Pooling for Deep Speaker Embedding](https://arxiv.org/abs/1803.10963)（代表性应用；含加权均值与标准差）

### 位置与图像 token 表示 / Positional and Image Token Representations

建议分组文件名：`position_and_token_embeddings.py`。

- [ ] Sinusoidal Positional Encoding — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] Learned Positional Embedding — [BERT](https://arxiv.org/abs/1810.04805)（代表性应用）
- [ ] Relative Position Bias / Encoding — [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)（相对位置表示；bias 变体另见 Swin）；[Swin 的相对位置 bias](https://arxiv.org/abs/2103.14030)
- [ ] Rotary Position Embedding（RoPE） — [RoFormer](https://arxiv.org/abs/2104.09864)
- [ ] ALiBi Attention Bias — [Train Short, Test Long](https://arxiv.org/abs/2108.12409)
- [ ] Patch Embedding — [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [ ] Patch Merging — [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [ ] Time-series Patch Embedding（PatchTST 风格） — [A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730)

### Transformer 层与卷积—注意力组合 / Transformer Layers and Convolution–Attention Hybrids

建议分组文件名：`transformer_and_hybrid_blocks.py`。

- [ ] Transformer Encoder Block — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] Transformer Decoder Block（包含 Cross-Attention） — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] GPT / Decoder-only Transformer Block — [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [ ] Vision Transformer（ViT）Block — [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [ ] Window Multi-Head Self-Attention — [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [ ] Shifted Window Attention（Swin） — [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [ ] Swin Transformer Block — [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [ ] Conv-Attention Hybrid Block — [CoAtNet](https://arxiv.org/abs/2106.04803)（通用类别的代表性实现）
- [ ] Conformer Block（FFN + Attention + Convolution） — [Conformer](https://arxiv.org/abs/2005.08100)

### 循环状态、因果时序与长序列建模 / Recurrent, Causal, and Long-Sequence Modeling

建议分组文件名：`recurrent_and_sequence_modeling.py`。

- [ ] Vanilla RNN Cell — [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0)（基础训练参考；非 RNN Cell 独立首创论文）
- [ ] LSTM Cell — [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)（原始版本；现代 forget gate 扩展另见下方）；[Learning to Forget：forget gate 扩展](https://doi.org/10.1162/089976600300015015)
- [ ] GRU Cell — [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078)
- [ ] Bidirectional Recurrent Block — [Bidirectional Recurrent Neural Networks](https://doi.org/10.1109/78.650093)
- [ ] ConvLSTM Cell — [Convolutional LSTM Network](https://arxiv.org/abs/1506.04214)
- [ ] Causal Conv1d Block — [WaveNet](https://arxiv.org/abs/1609.03499)
- [ ] Dilated Temporal Residual Block（TCN） — [An Empirical Evaluation of Generic Convolutional and Recurrent Networks](https://arxiv.org/abs/1803.01271)
- [ ] Gated Temporal Convolution Block — [WaveNet](https://arxiv.org/abs/1609.03499)
- [ ] Trend / Seasonal Decomposition Block — [Autoformer](https://arxiv.org/abs/2106.13008)
- [ ] Multi-scale Temporal Convolution Block — [InceptionTime](https://arxiv.org/abs/1909.04939)（代表性多尺度时序卷积）
- [ ] State Space Model（SSM）Block — [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)（S4，通用类别的代表性实现）
- [ ] Selective State Space / Mamba Block — [Mamba](https://arxiv.org/abs/2312.00752)

### 图邻居聚合与图级表示 / Graph Neighborhood Aggregation and Graph Representations

建议分组文件名：`graph_aggregation.py`。

- [ ] Graph Convolution（GCN）Block — [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [ ] GraphSAGE Aggregation Block — [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
- [ ] Graph Attention（GAT）Block — [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [ ] Graph Isomorphism Network（GIN）Block — [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)
- [ ] Message Passing Block（MPNN） — [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)
- [ ] Graph Transformer Block — [A Generalization of Transformer Networks to Graphs](https://arxiv.org/abs/2012.09699)（代表性实现）
- [ ] Graph Readout / Global Pooling — [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)（通用 readout 参考）

### 潜变量、生成与概率变换 / Latent Variables, Generation, and Probabilistic Transformations

建议分组文件名：`generative_and_probabilistic_blocks.py`。

- [ ] Variational Autoencoder Reparameterization — [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [ ] Vector Quantization Block（VQ-VAE） — [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- [ ] GAN Generator Upsampling Block — [DCGAN](https://arxiv.org/abs/1511.06434)（代表性结构）
- [ ] GAN Discriminator Downsampling Block — [DCGAN](https://arxiv.org/abs/1511.06434)（代表性结构）
- [ ] Normalizing Flow Affine Coupling Block — [Real NVP](https://arxiv.org/abs/1605.08803)

### 扩散时间条件与去噪特征处理 / Diffusion Time Conditioning and Denoising

建议分组文件名：`diffusion_blocks.py`。

- [ ] Diffusion Timestep Embedding — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [ ] Time-conditioned Residual Block — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [ ] Diffusion U-Net Attention / Cross-Attention Block — [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [ ] Adaptive LayerNorm / AdaLN-Zero（DiT） — [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [ ] Diffusion Transformer（DiT）Block — [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

### 参数高效适配与稀疏专家路由 / Parameter-Efficient Adaptation and Sparse Expert Routing

建议分组文件名：`adaptation_and_expert_routing.py`。

- [ ] Adapter Block — [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [ ] LoRA Linear — [LoRA](https://arxiv.org/abs/2106.09685)
- [ ] Mixture-of-Experts（MoE）Feed-Forward Block — [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)
- [ ] Top-k Router / Expert Gating — [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)

未勾选项为未来计划；模块实现文件见上方“已实现模块”表格。
