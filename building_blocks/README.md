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

## 常见模块与练习路线

下面是建议实现的路线，不代表这些文件已经存在。当前已实现的独立模块以“已实现模块”表格为准。根目录完整模型中的组件可以逐步提取到这里，但需要补充独立运行示例和验证。

符号约定：`B` 为 batch size，`S` 为序列长度，`D` 为 embedding 维度，`Nh` 为注意力头数，`dh = D / Nh`；图像使用 `[B, C, H, W]`，时序卷积使用 `[B, C, T]`。以下 shape 均以各模块明确的 stride、padding 和通道配置为准。

### 第一阶段：基础组件与 CNN blocks

先掌握特征变换、残差连接、通道变换和多分支融合。

| 模块 / 建议文件名（待实现） | 作用与典型应用 | 关键 shape / 练习重点 |
|---|---|---|
| MLP Block / `mlp_block.py` | 全连接特征变换；分类头、表格模型、Transformer FFN 的基础 | `[..., Din] → [..., Dout]`；Linear、激活、Dropout，区分 logits 与概率 |
| Conv-Norm-Activation / `conv_norm_act.py` | CNN 常用的卷积、归一化、激活组合 | `[B, Cin, H, W] → [B, Cout, Hout, Wout]`；计算 kernel、stride、padding 对尺寸的影响 |
| Normalization / `normalization.py` | BatchNorm、LayerNorm、GroupNorm；理解不同归一化轴 | 保持 shape；手写均值、方差、可学习缩放和平移，比较 train/eval；BatchNorm 还要处理 running statistics |
| Residual Basic Block / `residual_block.py` | ResNet-18/34 的基本模块；学习增量特征 | `F(x) + shortcut(x)`；两层 3×3 卷积，尺寸或通道变化时用投影 shortcut |
| Bottleneck Block / `bottleneck_block.py` | ResNet-50 等的瓶颈残差模块 | 1×1 降维 → 3×3 → 1×1 升维；理解中间通道与输出通道，以及 shortcut 的匹配 |
| Inception Block / `inception_block.py` | GoogLeNet 的多分支特征提取 | 各分支空间尺寸一致，沿通道 concat；输出通道是所有分支输出通道之和；1×1 卷积控制计算量 |
| Depthwise Separable Conv / `depthwise_separable_conv.py` | MobileNet 风格的高效卷积 | Depthwise 使用 `groups=Cin`，再以 1×1 Pointwise 混合通道；比较普通卷积的参数量 |
| Inverted Residual Block / `inverted_residual_block.py` | MobileNetV2：先扩展通道，再做深度卷积和线性投影 | 1×1 expand → depthwise → 1×1 project；通常仅 stride=1 且输入输出通道相同时直接相加 |
| Squeeze-and-Excitation / `squeeze_excitation_block.py` | 通道注意力，为不同通道学习权重 | `[B,C,H,W] → [B,C,1,1]` 的权重，再广播乘回输入；全局平均池化、瓶颈 MLP、Sigmoid |
| CBAM / `cbam_block.py` | 先通道注意力，再空间注意力 | 保持输入 shape；区分通道权重 `[B,C,1,1]` 与空间权重 `[B,1,H,W]` |
| U-Net Blocks / `unet_blocks.py` | 分割模型中的 DoubleConv、Down、Up 模块 | Down 降低空间尺寸；Up 上采样后与对应 skip 沿通道 concat；concat 后通道相加，处理奇数尺寸不匹配 |

注意两种 skip connection 的差别：

- **ResNet 相加**：`F(x) + shortcut(x)`，两侧 shape 必须一致；相加后通道数不增加。
- **U-Net 拼接**：`torch.cat([upsampled, skip], dim=1)`，batch 和空间尺寸必须一致；通道数相加。

GoogLeNet 的核心模块通常称为 `InceptionBlock`，对应文件建议命名为 `inception_block.py`。

### 第二阶段：Attention 与 Transformer blocks

在已有 Multi-Head Self-Attention 的基础上，逐步组装完整 Transformer 层。位置编码、归一化和 FFN 也是需要单独练习的组件。

| 模块 / 建议文件名 | 作用与典型应用 | 关键 shape / 练习重点 | 状态 |
|---|---|---|---|
| Multi-Head Self-Attention / `multi_head_self_attention.py` | 同一序列内部交互 | `[B,S,D] → [B,Nh,S,dh]`；QKᵀ、缩放、mask、softmax、加权 V、合并 heads | 已实现 |
| Positional Encoding / `positional_encoding.py` | 为序列提供位置信息 | 正弦位置编码或可学习位置 embedding 加到 `[B,S,D]`；广播、长度范围、buffer 与 parameter 的区别 | 待实现 |
| Position-wise FFN / `feed_forward_block.py` | 对每个 token 独立做非线性特征变换 | `[B,S,D] → [B,S,Dff] → [B,S,D]`；Linear 作用于最后一维，FFN 本身不混合不同 token | 待实现 |
| Transformer Encoder Block / `transformer_encoder_block.py` | 编码整条序列；文本、时序、ViT | Self-Attention + FFN，各自带残差与归一化；保持 `[B,S,D]`；比较 Pre-LN / Post-LN | 待实现 |
| Causal Self-Attention / `causal_self_attention.py` | 自回归模型只能访问当前位置及之前 | 下三角 mask；修改未来 token 后，之前位置的输出应保持不变（eval 模式） | 待实现 |
| Cross-Attention / `cross_attention.py` | 一条序列查询另一条序列；encoder-decoder、多模态 | Q 来自查询序列，K/V 来自 memory；权重 `[B,Nh,Sq,Sk]`，输出长度为 Sq | 待实现 |
| Transformer Decoder Block / `transformer_decoder_block.py` | 经典 encoder-decoder Transformer 解码层 | Causal Self-Attention + Cross-Attention + FFN；分别处理目标 mask 和 memory padding mask | 待实现 |
| GPT Block / `gpt_block.py` | Decoder-only Transformer 核心层 | Causal Self-Attention + FFN + 残差/归一化；通常没有 encoder cross-attention | 待实现 |
| Patch Embedding / `patch_embedding.py` | 把图像变为 token 序列；ViT | patch 大小 P，图像尺寸可整除时 `[B,C,H,W] → [B,(H/P)(W/P),D]`；用 stride=P、kernel=P 的 Conv2d 实现 | 待实现 |
| RMSNorm / `rms_norm.py` | 现代 Transformer 常见归一化组件 | 按最后一维的均方根归一化，通常不减均值；epsilon 与可学习缩放 | 待实现 |
| SwiGLU FFN / `swiglu_block.py` | 带门控的 FFN 变体 | `SiLU(Wg x) * (Wu x)` 后再投影；两条分支维度一致，保持最终输出 D | 待实现 |
| Rotary Position Embedding / `rotary_position_embedding.py` | 在 Q/K 上应用位置相关旋转；RoPE | 在每个头内成对旋转特征；练习旋转维度为偶数、位置索引和广播；通常不直接加到 V | 待实现 |

建议先独立实现 FFN、位置编码和 Encoder Block，再实现 causal attention、GPT Block、Cross-Attention 和经典 Decoder Block。RMSNorm、SwiGLU、RoPE 属于后续扩展。

#### 已有 attention 的 shape 与 mask 约定

拆头过程为：

```text
[B,S,D] → reshape → [B,S,Nh,dh] → transpose(1,2) → [B,Nh,S,dh]
```

拆分的是每个 token 的特征维度，每个头仍然拥有全部 S 个 token。矩阵乘法在最后两个维度进行，所以每个 batch、每个 head 独立计算 `[S,dh] @ [dh,S]`。

当前实现中 `mask=True` 表示允许关注，`False` 表示屏蔽。mask 必须能广播到 `[B,Nh,S,S]`：

- Causal mask 可以使用 `[S,S]` 的布尔下三角矩阵。
- 如果有效 token 标记为 `[B,S]`，作为 key padding mask 时应扩展为 `[B,1,1,S]`，不能直接假定 `[B,S]` 会按 batch 维广播。
- 当前实现要求每个 query 至少有一个未屏蔽的 key；整行都屏蔽时，全部分数变成负无穷，softmax 会产生 NaN。若要支持该情况，需要明确输出策略并修改实现。
- 返回的 `attention_weights` 是 Dropout 之前的权重；计算 context 使用 Dropout 之后的权重。
- 不同 PyTorch attention 接口的布尔 mask 语义可能不同，对照验证时必须先核对约定。

### 第三阶段：时序、循环网络与图网络 blocks

| 模块 / 建议文件名（待实现） | 作用与典型应用 | 关键 shape / 练习重点 |
|---|---|---|
| LSTM Cell / `lstm_cell.py` | 理解循环模型的状态更新 | 单步 `x_t: [B,Din]`，`h_t,c_t: [B,Dh]`；手写 input/forget/output gate 和 candidate，再沿时间循环 |
| GRU Cell / `gru_cell.py` | 更紧凑的门控循环单元 | reset/update gate；只有 hidden state；明确采用的公式版本，尤其 reset gate 放在 hidden 投影之前还是之后 |
| Temporal Residual Block / `temporal_residual_block.py` | TCN 的时序建模组件 | `[B,Cin,T] → [B,Cout,T]`；dilated causal Conv1d、残差投影，保证不访问未来 |
| Graph Convolution Block / `graph_convolution_block.py` | GCN 的邻居特征聚合 | 单图 `X:[N,Fin]`、`A:[N,N]`；添加自环、度归一化、聚合和线性变换，输出 `[N,Fout]` |
| Graph Attention Block / `graph_attention_block.py` | GAT 对邻居学习不同权重 | 只在邻居集合内归一化；处理自环、孤立节点，多头 concat 与平均的输出维度不同 |

TCN 的因果卷积可以通过仅左侧 padding 实现；如果使用对称 padding，则必须正确裁掉右侧多出的输出。重点验证改变未来输入不会影响过去输出。

### 第四阶段：按研究方向选择的扩展

这些模块常见于特定模型族，无需在完成基础路线前全部实现。

| 模块 / 建议文件名（待实现） | 适用方向 | 练习重点 |
|---|---|---|
| Feature Pyramid Network / `feature_pyramid_network.py` | 目标检测、多尺度视觉任务 | 多层特征、1×1 lateral projection、自顶向下上采样相加，再用 3×3 卷积处理 |
| Spatial Attention / `spatial_attention.py` | 图像生成、扩散模型 | 将空间位置展平为 token；区分全空间 self-attention 与 CBAM 的空间权重 |
| Time-conditioned Residual Block / `time_conditioned_residual_block.py` | Diffusion U-Net | 时间 embedding 投影后广播加到特征，或生成 scale/shift；不要把时间 embedding 当成图像宽高 |
| LoRA Linear / `lora_linear.py` | 参数高效微调 | 冻结基础权重，用低秩 A/B 学习增量；`W_eff = W + (alpha/r) BA`，验证只有适配器参数更新 |
| Mixture-of-Experts FFN / `moe_block.py` | 稀疏专家模型 | router、top-k 选择、专家分发与加权合并；理解负载均衡和容量限制，作为进阶练习 |

## 推荐开始顺序

不必一次写完所有模块。先完成下面这组核心练习：

1. Conv-Norm-Activation：掌握图像通道和空间尺寸。
2. Residual Basic Block：掌握相加与投影 shortcut。
3. U-Net DoubleConv / Down / Up：掌握上采样和 concat。
4. Depthwise Separable Conv：掌握 groups 和通道混合。
5. Squeeze-and-Excitation：掌握 pooling、门控和广播。
6. Multi-Head Self-Attention（已有）：从空文件重写，解释拆头与 mask。
7. Position-wise FFN：掌握逐 token 的特征变换。
8. Positional Encoding：掌握序列位置表示。
9. Transformer Encoder Block：组合 attention、FFN、残差和 LayerNorm。
10. Causal Self-Attention + GPT Block：掌握自回归约束。
11. Cross-Attention + Transformer Decoder Block：掌握不同来源的 Q 与 K/V。
12. Temporal Residual Block：掌握 dilation、感受野和因果性。

随后按兴趣补 Inception、Bottleneck、Inverted Residual、CBAM、LSTM/GRU Cell、GCN/GAT，以及现代 Transformer 扩展。

## 每个模块的实现要求

每个文件顶部应写清楚：

- **做什么**：模块解决的问题、典型模型和应用场景。
- **数学原理**：核心公式及变量含义。
- **怎么做**：主要步骤、输入输出约定和可配置参数。
- **维度流**：在 forward 的关键操作旁标注 tensor shape。
- **达到什么效果**：能够解释机制、独立重写、验证输出和梯度，并接入完整模型。
- **常见错误**：例如残差 shape 不匹配、concat 维度错误、mask 广播错误、因果卷积泄露未来信息。

实现内部可使用 `nn.Linear`、`nn.Conv2d` 等基础算子；练习 attention 时不要直接调用 `nn.MultiheadAttention`，练习 Transformer 层时不要用 `nn.TransformerEncoderLayer` 代替内部实现。官方实现可以作为对照。

### 最小验证标准

无需下载数据或训练完整模型，先用小型随机 tensor 做有意义的检查：

1. 验证输出 shape，既覆盖默认参数，也覆盖 stride、输入输出通道不一致等关键分支。
2. 检查输出是否包含 NaN/Inf；对于 attention，验证被屏蔽位置权重为零，Dropout 前有效行权重和接近 1。
3. 对输出构造标量 loss 并 backward，检查预期可训练参数获得有限梯度。
4. 验证机制本身：causal 模块不受未来输入影响；残差投影能够匹配尺寸；concat 通道数正确。
5. 有对应官方实现时，在相同权重、相同约定、关闭 Dropout 后做数值对照，不能只比较 shape。

注意：`model.eval()` 控制 Dropout 和 BatchNorm 的行为，`torch.no_grad()` 控制梯度记录，两者用途不同。进行确定性的数值对照时需要同时考虑这两点。

## 参考资料

以论文和 PyTorch 官方文档为原理与接口参考：

- [PyTorch nn 文档](https://docs.pytorch.org/docs/stable/nn.html)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [GoogLeNet / Inception](https://arxiv.org/abs/1409.4842)
- [MobileNet](https://arxiv.org/abs/1704.04861) 与 [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Squeeze-and-Excitation](https://arxiv.org/abs/1709.01507) 与 [CBAM](https://arxiv.org/abs/1807.06521)
- [U-Net](https://arxiv.org/abs/1505.04597)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 与 [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [TCN](https://arxiv.org/abs/1803.01271)
- [GCN](https://arxiv.org/abs/1609.02907) 与 [GAT](https://arxiv.org/abs/1710.10903)
- [RMSNorm](https://arxiv.org/abs/1910.07467)、[GLU Variants](https://arxiv.org/abs/2002.05202)、[RoPE](https://arxiv.org/abs/2104.09864)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)、[DDPM](https://arxiv.org/abs/2006.11239)、[LoRA](https://arxiv.org/abs/2106.09685)、[Switch Transformers](https://arxiv.org/abs/2101.03961)
