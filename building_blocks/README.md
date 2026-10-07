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

## 未来待实现的常见 Blocks

以下为待实现清单，按模型家族分类；包含可复用 block 和常见基础组件。不同论文的命名与组合方式可能重叠，深度学习也没有固定、穷尽的 block 目录。本清单覆盖常见模型家族，后续可继续扩展。

### 基础组件

- [ ] MLP / Dense Block
- [ ] Conv-Norm-Activation Block
- [ ] Embedding / Embedding Projection
- [ ] Batch Normalization
- [ ] Layer Normalization
- [ ] Group Normalization
- [ ] Instance Normalization
- [ ] RMSNorm
- [ ] Residual / Add & Norm Block
- [ ] Gated Linear Unit（GLU）
- [ ] Highway Block

### CNN：卷积与多分支模块

- [ ] ResNet Basic Block
- [ ] ResNet Bottleneck Block
- [ ] Pre-Activation Residual Block
- [ ] Wide Residual Block
- [ ] ResNeXt Block
- [ ] DenseNet Dense Layer / Dense Block
- [ ] DenseNet Transition Block
- [ ] Inception Block
- [ ] Inception-ResNet Block
- [ ] Depthwise Separable Convolution
- [ ] Grouped Convolution Block
- [ ] Dilated Convolution Block
- [ ] MobileNetV2 Inverted Residual / Linear Bottleneck
- [ ] MobileNetV3 Block
- [ ] MBConv Block
- [ ] Fused-MBConv Block
- [ ] ShuffleNet Unit / Channel Shuffle
- [ ] Fire Module（SqueezeNet）
- [ ] Ghost Module / Ghost Bottleneck
- [ ] ConvNeXt Block
- [ ] CSP Block（Cross Stage Partial）
- [ ] C3 / C2f Block（YOLO）
- [ ] RepVGG Block（结构重参数化）
- [ ] Deformable Convolution Block

### 视觉注意力与特征调制

- [ ] Squeeze-and-Excitation（SE）Block
- [ ] CBAM Block
- [ ] Efficient Channel Attention（ECA）Block
- [ ] Coordinate Attention Block
- [ ] Non-Local Block
- [ ] Spatial Self-Attention Block
- [ ] Attention Gate（Attention U-Net）
- [ ] FiLM / Feature-wise Affine Modulation

### 分割、检测与多尺度特征

- [ ] U-Net DoubleConv Block
- [ ] U-Net Down Block
- [ ] U-Net Up / Skip Fusion Block
- [ ] Residual U-Net Block
- [ ] Transposed Convolution Upsampling Block
- [ ] Interpolation + Convolution Upsampling Block
- [ ] PixelShuffle Upsampling Block
- [ ] Feature Pyramid Network（FPN）
- [ ] Path Aggregation Network（PAN / PANet）
- [ ] Bidirectional Feature Pyramid Network（BiFPN）
- [ ] Spatial Pyramid Pooling（SPP）
- [ ] Spatial Pyramid Pooling Fast（SPPF）
- [ ] Atrous Spatial Pyramid Pooling（ASPP）
- [ ] Pyramid Pooling Module（PPM）
- [ ] Multi-scale Feature Fusion Block

### Attention 与 Transformer

- [ ] Scaled Dot-Product Attention（独立基础组件）
- [ ] Causal / Masked Multi-Head Self-Attention
- [ ] Multi-Head Cross-Attention
- [ ] Additive / Bahdanau Attention
- [ ] Multiplicative / Luong Attention
- [ ] Position-wise Feed-Forward Network（FFN）
- [ ] GEGLU Feed-Forward Block
- [ ] SwiGLU Feed-Forward Block
- [ ] Transformer Encoder Block
- [ ] Transformer Decoder Block（包含 Cross-Attention）
- [ ] GPT / Decoder-only Transformer Block
- [ ] Sinusoidal Positional Encoding
- [ ] Learned Positional Embedding
- [ ] Relative Position Bias / Encoding
- [ ] Rotary Position Embedding（RoPE）
- [ ] ALiBi Attention Bias
- [ ] Multi-Query Attention（MQA）
- [ ] Grouped-Query Attention（GQA）
- [ ] Sliding-Window / Local Attention
- [ ] Sparse Attention Block
- [ ] Linear Attention Block
- [ ] KV Cache（自回归注意力的缓存组件）

### Vision Transformer 与混合架构

- [ ] Patch Embedding
- [ ] Vision Transformer（ViT）Block
- [ ] Window Multi-Head Self-Attention
- [ ] Shifted Window Attention（Swin）
- [ ] Swin Transformer Block
- [ ] Patch Merging
- [ ] MLP-Mixer Block
- [ ] Conv-Attention Hybrid Block
- [ ] Conformer Block（FFN + Attention + Convolution）

### 循环网络、时序与状态空间

- [ ] Vanilla RNN Cell
- [ ] LSTM Cell
- [ ] GRU Cell
- [ ] Bidirectional Recurrent Block
- [ ] ConvLSTM Cell
- [ ] Causal Conv1d Block
- [ ] Dilated Temporal Residual Block（TCN）
- [ ] Gated Temporal Convolution Block
- [ ] Temporal Attention Pooling
- [ ] Time-series Patch Embedding（PatchTST 风格）
- [ ] Trend / Seasonal Decomposition Block
- [ ] Reversible Instance Normalization（RevIN）
- [ ] Multi-scale Temporal Convolution Block
- [ ] State Space Model（SSM）Block
- [ ] Selective State Space / Mamba Block

### 图神经网络

- [ ] Graph Convolution（GCN）Block
- [ ] GraphSAGE Aggregation Block
- [ ] Graph Attention（GAT）Block
- [ ] Graph Isomorphism Network（GIN）Block
- [ ] Message Passing Block（MPNN）
- [ ] Graph Transformer Block
- [ ] Graph Readout / Global Pooling

### 生成模型与扩散模型

- [ ] Autoencoder Encoder / Decoder Block
- [ ] Variational Autoencoder Reparameterization
- [ ] Vector Quantization Block（VQ-VAE）
- [ ] GAN Generator Upsampling Block
- [ ] GAN Discriminator Downsampling Block
- [ ] Adaptive Instance Normalization（AdaIN）
- [ ] StyleGAN Modulated / Demodulated Convolution
- [ ] Diffusion Timestep Embedding
- [ ] Time-conditioned Residual Block
- [ ] Diffusion U-Net Attention / Cross-Attention Block
- [ ] Adaptive LayerNorm / AdaLN-Zero（DiT）
- [ ] Diffusion Transformer（DiT）Block
- [ ] Normalizing Flow Affine Coupling Block

### 参数高效微调与专家模块

- [ ] Adapter Block
- [ ] LoRA Linear
- [ ] Mixture-of-Experts（MoE）Feed-Forward Block
- [ ] Top-k Router / Expert Gating

以上均为未来计划；当前可运行的实现仍以“已实现模块”列表为准。
