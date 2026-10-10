# 多头自注意力：20 个理解问题与答案

对应实现：[multi_head_self_attention.py](../multi_head_self_attention.py)。

## 符号与完整计算

| 符号 | 含义 | 示例 |
|---|---|---|
| B | batch 中的序列数量 | 2 |
| S | 每条序列的 token 数量 | 10 |
| D | d_model，每个 token 的输入/输出特征数 | 32 |
| H | num_heads，注意力头数；不是图像高度 | 4 |
| dh | head_dim = D // H | 8 |

采用行向量表示，忽略 bias 时：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

每个头分别计算：

$$
A_h=\operatorname{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}+M\right),
\qquad C_h=A_hV_h
$$

M 在允许的位置为 0，在屏蔽的位置为负无穷。训练时，代码在 A 上施加 Dropout 后才计算 C。

$$
Y=\operatorname{Concat}(C_1,\ldots,C_H)W_O+b_O
$$

| 步骤 | Shape |
|---|---|
| 输入和 Q/K/V 投影 | [B,S,D] |
| 拆分特征 | [B,S,H,dh] |
| 调整维度顺序 | [B,H,S,dh] |
| QKᵀ、缩放、mask、softmax、Dropout | [B,H,S,S] |
| 注意力权重乘 V | [B,H,S,dh] |
| 转置、合并头 | [B,S,H,dh] → [B,S,D] |
| 输出投影 | [B,S,D] |

本笔记讨论代码中的标准等宽多头自注意力；它本身不包含位置编码、残差连接、LayerNorm 或 FFN。

## 1. 为什么同一个输入要分别投影成 Q、K、V？共享权重会限制什么？

Q 表示当前位置用什么特征进行查询，K 表示每个位置提供什么匹配特征，V 表示匹配后要汇总的内容。它们是帮助理解的角色比喻，不是人工预先指定的语义。

三个独立的 Linear 让模型分别学习匹配方式和内容表达。每个 token 都使用同一套 WQ，但 WQ、WK、WV 三套参数彼此独立。

如果 Q/K 共享投影，点积分数被限制为同一表示空间中的相似度，未加 mask 的 QKᵀ 会对称，表达的关系更受约束。再让 V 共享，就把用于匹配的表示与传递内容的表示也绑定起来。共享并非不能工作，但会减少可独立学习的自由度。

## 2. Linear 会混合不同 token 吗？作用于哪个维度？

不会。Linear(D,D) 作用于最后一个特征维度，分别处理每个 token：

$$
q_{b,i}=x_{b,i}W_Q+b_Q
$$

q[b,i] 只依赖当前层输入 x[b,i]，不会直接读取 x[b,j]。输入输出都是 [B,S,D]。所有 token 共享参数，但共享参数不等于混合 token。

不同 token 的信息在后面的注意力加权 V 时汇合。若 x 来自前一层，它本身可能已经包含上下文信息，这不改变本层 Linear 的逐 token 性质。

## 3. 为什么 D 必须能被 H 整除？

这个实现把 D 维向量平均拆成 H 个头，每个头 dh=D/H 维，因此必须整除：

~~~python
if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
    raise ValueError("Require positive dimensions and d_model % num_heads == 0")
~~~

例如 D=32、H=4，则 dh=8。D=32、H=5 无法按当前 reshape 分成等宽的 5 个头。

这是当前等宽实现的要求；更一般的架构可以先投影到其他总维度，但需要相应修改投影和输出层。

## 4. 拆头是在拆 token，还是拆特征？

拆的是每个 token 投影后的特征，不是序列中的 token：

~~~text
一个 token 的 32 维 → 4 个头，每个头 8 维
~~~

每个头仍然拥有全部 S 个 token。所以 [B,4,10,8] 表示每条序列有 4 个头，每个头包含全部 10 个 token。

## 5. 每个头能利用原始输入的全部特征吗？

能。因为先进行完整的 Linear 投影，再拆头。

Head 0 得到投影后的第 0～7 个输出特征，但每个输出特征都可以是原始 32 个输入特征的加权组合。它不等于只读取原始输入的前 8 维。

按输出列拆分大的 WQ，相当于各头拥有自己的 D×dh 投影矩阵。因此一次大投影再拆头，与各头分别投影后组织结果，在数学上可以等价。

## 6. 为什么先 reshape，再 transpose？

~~~python
x = x.reshape(B, S, H, dh)
x = x.transpose(1, 2)
~~~

reshape 把 D 拆成 H×dh，transpose 把每个头的整条序列组织成矩阵：

~~~text
[B,S,D] → [B,S,H,dh] → [B,H,S,dh]
~~~

此时 q[b,h] 为 [S,dh]，适合计算 token 之间的匹配。

若直接用 [B,S,H,dh] 进行同样的 @ 运算，最后两个维度会变成 [H,dh] @ [dh,H]，得到 [B,S,H,H]，计算对象就错了。

## 7. QKᵀ 为什么得到 [B,H,S,S]？两个 S 是什么？

矩阵乘法规则为 (m,n)@(n,p)→(m,p)。令 dh = d_model // num_heads：

~~~text
Q:   [B,H,S,dh]
Kᵀ:  [B,H,dh,S]
结果: [B,H,S,S]
~~~

只看一个样本、一个 head，最后两个维度进行 [S,dh]@[dh,S]→[S,S]。中间的 dh 维通过点积求和，保留两边的 S。前面的 B、H 是独立矩阵组的索引，不会把不同样本或头混在一起。

两个 S 的大小相同，但含义不同：

| 维度 | 含义 | 对应问题 |
|---|---|---|
| 第一个 S：行 | query 的位置 | 哪个 token 正在查询？ |
| 第二个 S：列 | key 的位置 | 正在与哪个 token 匹配？ |

例如序列为“我 喜欢 苹果”，一个 head 的分数表为：

| Query / Key | 我 | 喜欢 | 苹果 |
|---|---|---|---|
| 我 | 分数₀₀ | 分数₀₁ | 分数₀₂ |
| 喜欢 | 分数₁₀ | 分数₁₁ | 分数₁₂ |
| 苹果 | 分数₂₀ | 分数₂₁ | 分数₂₂ |

每个 query 都要与全部 key 匹配，因此共有 S×S 个分数。“喜欢”这一行表示它的 query 与三个位置的 key 的匹配分数。

代码完成缩放后：

$$
\text{scores}[b,h,i,j]
=\frac{Q[b,h,i,:]\cdot K[b,h,j,:]}{\sqrt{d_h}}
$$

具体 shape 示例：[2,4,10,8]@[2,4,8,10]→[2,4,10,10]。

自注意力中 Q 和 K 来自同一条序列，所以两个长度都是 S。交叉注意力中，query 和 key 序列长度可以不同，分数 shape 为 [B,H,S_query,S_key]。

## 8. scores[b,h,i,j] 如何计算？

它是第 i 个 query 与第 j 个 key 的点积：

$$
\text{scores}_{b,h,i,j}
=\sum_{r=0}^{d_h-1}Q_{b,h,i,r}K_{b,h,j,r}
$$

例如 q_i=[1,2]、k_j=[3,4]，原始分数是 1×3+2×4=11，随后除以 sqrt(2)。

分数是可学习特征中的匹配结果，不是概率。softmax 后才成为非负归一化权重。

## 9. 为什么除以 sqrt(dh)，不是 sqrt(D)？

单个头的点积累加 dh 项。在 Q/K 各分量独立、均值为 0、方差为 1 的简化假设下，点积方差约为 dh；除以 sqrt(dh) 后方差约为 1。

未缩放时，大维度下分数可能幅度过大，softmax 过度集中，进入饱和区域，使梯度变小。实际训练的 Q/K 不一定满足上述假设，但该分析解释了缩放动机。

每个头实际相乘的是 dh 维向量，因此使用 sqrt(dh)。使用 sqrt(D) 会比标准缩放多缩小 sqrt(H) 倍，通常使分布更平缓。

## 10. 为什么 softmax 用 dim=-1？dim=-2 会怎样？

scores 是 [B,H,S_query,S_key]。最后一维是 key 的位置，不是 head_dim。

我们要回答：**对于一个固定的 query，它应该把多少注意力分配给各个 key？** 因此对同一行的 key 分数做 softmax：

~~~python
attention_weights = torch.softmax(scores, dim=-1)
# [B,H,S_query,S_key]，沿 S_key 归一化，shape 不变
~~~

$$
A_{i,j}=\frac{\exp(z_{i,j})}{\sum_k\exp(z_{i,k})}
$$

每个样本、每个 head、每个 query 都独立进行归一化；在至少有一个允许的 key、且数值有效的情况下，每行权重之和为 1。这个性质针对 Dropout 前的权重。

例如，“喜欢”这个 query 对“我、喜欢、苹果”的缩放后分数为：

~~~python
scores_row = [1.0, 2.0, 3.0]
weights_row = [0.090, 0.245, 0.665]  # softmax 后的近似值
~~~

它分配给三个 key 的注意力约为 9%、24.5%、66.5%。接下来用这行权重加权对应的 V：

$$
C_{\text{喜欢}}
\approx 0.090V_{\text{我}}
+0.245V_{\text{喜欢}}
+0.665V_{\text{苹果}}
$$

每个 query 都得到自己的一个 dh 维内容向量，这正是 [S,S]@[S,dh]→[S,dh] 的含义。

如果改用 dim=-2，则沿 query 维归一化，每列之和为 1：固定一个 key，比较不同 query 的分数。这不符合当前“为每个 query 分配 key 权重，再汇总 V”的标准注意力定义。

**记忆句：行是“谁在查”，列是“查谁”；每行沿着列做 softmax，再按这些权重汇总 V。**

## 11. Self-attention 的注意力矩阵一定对称吗？

不一定，有两个原因：

1. Q/K 投影不同，通常 q_i·k_j 不等于 q_j·k_i，所以分数本身不对称。
2. 即使 Q=K、分数对称，逐行 softmax 的分母也可能不同，归一化后的矩阵仍不对称。

例如对称分数 [[1,0],[0,2]]，softmax 后 A[0,1]约为0.269，A[1,0]约为0.119。Causal mask 也会造成方向性。

“自注意力”表示 Q/K/V 来自同一条输入序列，不表示关注关系必须对称。

## 12. 为什么屏蔽位置填 -inf，而不是 0？

softmax 中 exp(-inf)=0，使屏蔽位置权重为 0。填 0 时 exp(0)=1，该位置仍参与分母并获得权重；当其他分数为负时，它甚至可能得到较大权重。

当前代码约定：

~~~python
# mask=True 允许，False 屏蔽
scores = scores.masked_fill(~mask, float("-inf"))
~~~

mask 语义取决于具体接口，不能假定所有 PyTorch attention 接口的 True 含义都相同。

## 13. Causal mask 与 padding mask 分别屏蔽什么？Shape 是什么？

Causal mask 屏蔽未来 key，允许 j≤i：

~~~python
causal_mask = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
# [S,S] 可广播到 [B,H,S,S]
~~~

Padding mask 屏蔽补齐用的 key。假设 valid_tokens 是 [B,S]，True 表示真实 token：

~~~python
key_mask = valid_tokens[:, None, None, :]  # [B,1,1,S]
combined_mask = causal_mask[None, None, :, :] & key_mask
# [B,1,S,S]，沿 heads 广播
~~~

两者可以同时使用。mask 应与 scores 在同一 device。

屏蔽 padding key 不会自动把 padding query 的输出清零；训练 loss 或后续聚合仍应忽略无效 query。不要简单屏蔽整行 query 后直接使用普通 softmax，否则会触发下一题的问题。

## 14. 一整行都被屏蔽会怎样？

这一行全部为 -inf，普通 softmax 会产生 NaN。稳定 softmax 通常先减去行最大值，此时出现 -inf-(-inf)。

当前手写实现要求每个 query 至少有一个允许的 key。对于有效 query，正确构造 causal/padding mask 通常能保证这一点。

若需要支持全屏蔽行，应显式定义策略，例如让该行注意力权重为零，并在 softmax 前使用安全值，避免先产生 NaN 再依赖事后替换：

~~~python
allowed = torch.broadcast_to(mask, scores.shape)
has_key = allowed.any(dim=-1, keepdim=True)
masked_scores = scores.masked_fill(~allowed, float("-inf"))
safe_scores = torch.where(has_key, masked_scores, torch.zeros_like(scores))
weights = torch.softmax(safe_scores, dim=-1)
weights = weights.masked_fill(~allowed, 0.0)
~~~

这是替代当前 mask/softmax 步骤的处理示例。全屏蔽行的 context 为零，但有 bias 的输出投影仍可能输出 bias；若要求最终输出为零，还需显式处理。

## 15. 权重乘 V 的 shape 为什么这样变化？加权求和是什么？

~~~text
[B,H,S,S] @ [B,H,S,dh] → [B,H,S,dh]
~~~

每个 query 的一行权重，对所有 key 对应的 V 向量做加权求和：

$$
C_i=\sum_j A_{i,j}V_j
$$

例如 A_i=[0.1,0.2,0.7]，V_0=[1,0]、V_1=[0,2]、V_2=[3,4]：

$$
C_i=0.1[1,0]+0.2[0,2]+0.7[3,4]=[2.2,3.2]
$$

每个 V 是 dh 维，求和后仍是 dh 维；所有 S 个 query 各产生一个输出。未施加 Dropout 时，有效行是 V 的凸组合；训练时 Dropout 会改变这一性质。

## 16. 为什么在权重上使用 Dropout？每行还和为 1 吗？

Dropout 随机丢弃部分注意力连接，作为正则化，减少对固定连接的依赖。丢弃概率 p<1 时，保留下来的权重乘 1/(1-p)，保持每个权重的期望值。

它不是丢弃整个 token 或整个 head。训练时每行权重之和不保证为 1，也没有自动重新归一化。eval 模式下 Dropout 是恒等操作。

当前实现返回的是 Dropout 前的 attention_weights，context 使用的是 Dropout 后的 dropped_weights。这两个张量的 shape 相同，数值在训练时可能不同。

## 17. 合并头前为什么要 transpose？contiguous 必须吗？

注意力结果为 [B,H,S,dh]。要把同一个 token 的不同头拼接，应先变为 [B,S,H,dh]：

~~~python
context = context.transpose(1, 2).contiguous()
context = context.reshape(B, S, D)
~~~

直接从 [B,H,S,dh] reshape 到 [B,S,D]，通常会把不同 token 的片段放到同一输出 token 中，因为逻辑元素顺序不对。

小例子：H=2、S=2、dh=1，Head 0 的两个 token 为 [a,b]，Head 1 为 [c,d]。正确结果是 token 0=[a,c]、token 1=[b,d]；直接 reshape 会得到 [a,b]、[c,d]。

contiguous 不改变 shape 或数值，只整理连续内存布局。使用 reshape 时可省略，它会在需要时复制数据：

~~~python
context = context.transpose(1, 2).reshape(B, S, D)
~~~

使用 view 时，转置后的布局可能不兼容，应先 contiguous。transpose 负责语义顺序，contiguous 负责内存布局，两者不能互相替代。

## 18. 拼接后为什么还有 output_projection？

拼接只把各头的结果放在一起，没有学习如何融合：

$$
y_i=[C_i^{(1)};\ldots;C_i^{(H)}]W_O+b_O
$$

output_projection 的每个输出特征都可以利用全部头的输入特征，所以它学习跨头的线性组合，输出 shape 为 [B,S,D]。

它仍然逐 token 工作，不直接混合不同位置。跨 token 汇总已经在权重乘 V 时发生。省略输出投影仍能得到多头 context，但缺少标准 MHA 中可学习的最终融合步骤。

## 19. 没有位置编码和 causal mask，能区分顺序吗？重排输入会怎样？

假设输入仅包含 token 内容，没有位置相关信息，也没有位置相关 mask，且在 eval 模式关闭 Dropout，则该模块具有置换等变性：

$$
f(PX)=Pf(X)
$$

P 是重排序列位置的置换矩阵。重排输入会对应重排输出，而不是输出完全不变。

原因是投影逐 token 共享参数，分数随位置同时重排行和列，softmax 和 V 聚合也随之对应变化。模块不能仅凭这种结构知道“哪个词先出现”。

如果输入已包含位置编码，上述假设不成立。Causal mask 也引入顺序约束，但不等价于显式位置编码。训练时独立随机 Dropout 的两次结果不能直接要求严格数值相等。

## 20. 固定 D，增加头数有什么影响？S 翻倍呢？

固定 D 且保持可整除，H 增大时 dh=D/H 减小。

| 项目 | 固定 D，H 增加 | 固定其他参数，S 翻倍 |
|---|---|---|
| 每头维度 dh | 减小 | 不变 |
| Q/K/V 总元素数 | 不变，均为 BSD | 约 2 倍 |
| 四个 Linear 参数量 | 不变 | 不变 |
| 显式权重张量 BHSS | 随 H 线性增长 | 4 倍 |
| QKᵀ 与 AV 的理论计算量 | 约 O(BS²D)，不因拆头改变阶数 | 约 4 倍 |
| 投影计算量 O(BSD²) | 不变 | 约 2 倍 |

当前四个 Linear 都带 bias，因此参数总量是：

$$
4D^2+4D
$$

D=32 时为 4224 个参数，与头数无关。

H 个头，每头计算量约 S²dh，合计 HS²dh=S²D。但更多头可能增加 softmax、调度等开销，实际速度不保证相同。

示例：B=2、S=10、D=32。H=4 时权重有800个元素；H=8 时有1600个元素。H=4、S=20 时则有3200个元素。

这里的显存讨论针对代码中显式构建 scores/weights 的实现。融合 attention 内核可以避免完整保存这些中间矩阵，但不会让标准全注意力的数学计算自动变为线性。头数更多也不保证效果更好。

## 复习方式

先遮住答案解释每题，再从空文件重写实现。优先复习第5、9、11、18、19、20题，检查自己能否解释“为什么”，而不只是背出 shape。

可用三个小实验检验理解：

1. eval 模式下对无位置编码、无 mask 的输入重排 token，验证输出对应重排。
2. 使用 causal mask，修改未来 token，验证之前位置的输出不变。
3. 复制官方实现的对应参数，在相同 mask 语义、关闭 Dropout 后比较输出数值；仅比较 shape 不足以证明实现正确。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：缩放点积注意力、多头投影与位置编码。
- [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [PyTorch matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
- [PyTorch Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch Tensor reshape](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html)
