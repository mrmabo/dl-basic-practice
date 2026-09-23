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

## 后续可以加入

```text
building_blocks/
├── multi_head_self_attention.py
├── residual_block.py
├── inception_block.py
├── depthwise_separable_conv.py
├── squeeze_excitation_block.py
└── transformer_encoder_block.py
```

其中 GoogLeNet 的核心模块通常称为 `InceptionBlock`，因此建议对应文件命名为 `inception_block.py`。
