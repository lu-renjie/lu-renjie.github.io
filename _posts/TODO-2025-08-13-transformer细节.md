---
title: 【深度学习】transformer实现细节
tags: 深度学习 笔记 脚本
---

<!--
* 20250812: 创建文件，写了非大模型的部分
* 202508: 完善 MOE 和 MLA 的部分
-->

<!--more-->

## 位置编码

### 绝对位置编码

Transformer原论文的位置编码设计为：

$$
p_m = \begin{bmatrix}
\sin (w_1m)\\
\cos (w_1m)\\
\vdots\\
\sin (w_im)\\
\cos (w_im)\\
\vdots\\
\sin (w_{\frac{d}{2}}m)\\
\cos (w_{\frac{d}{2}}m)
\end{bmatrix}
$$

其中 $w_i=(10000^{-\frac{2}{d}})^{i}, i=1,2,\cdots,\frac{d}{2}$。位置编码这样设计的原因是能让相似度根据距离衰减，$p_m\cdot p_n=\sum_{i=1}^{d/2}[\sin(w_im)\sin(w_in)+\cos(w_im)\cos(w_in)]
=\sum_{i=1}^{d/2}\cos(w_i(m-n)) $，是相对位置距离 $|m-n|$ 的递减函数。

不过这样的位置编码是由三角函数组成，是周期性的，距离太大超过周期就不是递减的了，相似度也会重新上升，它的最大周期是多少呢？当 $i=\frac{d}{2}$ 时，也就是三角函数的频率达到最低时，周期最大，此时 $w_{\frac{d}{2}}=\frac1{10000}$，周期为 $5000\pi$。因此位置编码能正常生效的最大输入长度是15000多。

这种位置编码的代码实现：
```python
def position_embedding(pos, dim=768, max_len=10000):
    """
    Args:
        pos: tensor or int
        dim: dimension of the token, normally it is 768.
        device: create the tensor on cpu or gpu.
    Returns:
        position embedding, a (B, L, D) tensor
    """
    x = torch.arange(1, dim + 1, device=pos.device)
    phi = (x % 2 == 0) * (torch.pi / 2)

    x[x % 2 == 1] += 1
    if isinstance(pos, torch.Tensor):
        for i in range(len(pos.shape)):
            x.unsqueeze(0)
        x = pos.unsqueeze(-1) / (max_len ** (x / dim))
    else:
        x = pos / (max_len ** (x / dim))

    pe = torch.sin(x + phi)
    return pe
```

把位置编码可视化出来长这样，两两之间的内积会随距离增大显著衰减：
<div align=center>
<img src="../../../assets/images/posts/2025-08-13/pos_emb.png" width="50%" />
</div>

对于二维的位置编码，例如ViT，通常会把两个维度各用一个长度减半的位置编码，然后concat成一个位置编码。除了这种正弦余弦的绝对位置编码，一个更简单的实现是使用可学习的Embedding作为位置编码，效果没差别。


### 相对位置编码

相对位置编码不在输入的部分加，而是在attention的部分加，直接在attention矩阵上根据相对位置加上一个bias，一个典型的代表是[T5](https://arxiv.org/abs/1910.10683)：

```python
class RelativePositionAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 可学习的相对位置偏置
        self.rel_pos_bias = nn.Embedding(2 * max_len - 1, num_heads)
        
    def forward(self, q, k, v):
        ...  # attention前置计算

        # 添加相对位置偏置（忽略mask）
        rel_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)  # (L, L)
        rel_pos = rel_pos.clamp(-self.max_len + 1, self.max_len - 1)
        rel_pos = rel_pos + self.max_len - 1  # 映射到有效索引
        rel_pos_bias = self.rel_pos_bias(rel_pos).permute(2, 0, 1)  # (H, L, L)
        
        attn_scores = attn_scores + rel_pos_bias.unsqueeze(0)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        ... # attention后续计算
```

根据[HSTU](https://arxiv.org/abs/2402.17152)的结果，这种相对位置编码在序列推荐中的效果远远好于绝对位置编码和ROPE。

### 旋转位置编码

旋转位置编码可以以绝对位置编码的形式（加上position embedding）实现相对位置编码的功能，在NLP任务上很solid地比其它位置编码要好，后续更是发展出了长度外推，称为大模型的首选位置编码。

#### RoPE

[RoPE](https://arxiv.org/abs/2104.09864)。

#### 长度外推

对ROPE做一些修改可以实现长度外推，即在测试时可以使用比训练时更长的长度。两个比较常用的是NTK-ROPE和YaRN。


## Padding

训练和推理是需要将多个长度不一的序列组成batch，为此需要把短的序列添加一些padding token到最长。

### Right Padding
BERT系列模型进行right padding，这样有两个好处，一是位置编码更好加，而是放在开头的CLS token方便取出来。

### Left Padding
GPT系列为了方便生成下一个token，会近left padding，保证序列在右侧对齐以方便取出末尾的token。这会导致一个问题是attention的时候前几个token由于causal mask的存在无法计算attention。


## Attention

$$
O = {\rm softmax}(QK^T)V
$$

### Multi-head的重要性

多个head的作用是让模型可以在不同的子空间内计算注意力。例如一些head专门在safety相关的子空间上做注意力，是模型的safety能力的主要来源；一些head可以专门负责copy其它token的信息，即induction head：
<div align=center>
<img src="../../../assets/images/posts/2025-08-13/induction_head.png" width="60%" />
</div>
这种功能性的head负责一定的功能，而不是只对特定pattern生效，保证了一定的泛化性。在使用multi-head的情况下，transformer应该被理解成下图的形式：

<div align=center>
<img src="../../../assets/images/posts/2025-08-13/multi_head.png" width="40%" />
</div>

信息可以在不同的层经过不同的head处理，也可以跳过head，不同功能的head共同促成了transformer的强大拟合能力。

### mask的实现

transformer存在两个mask，一个是人为加上的attention_mask，例如causal mask，另一个是padding mask。二者在计算attention的时候需要合并：

```python
maxlen = seqs.shape[1]
ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.device)
causal_mask = torch.tril(ones_matrix)
padding_mask = (mask == 0).to(self.device)
attention_mask = causal_mask.unsqueeze(0) & (~padding_mask.unsqueeze(1))  # (B, L, L)
```

在计算attention时需要在softmax之前把需要mask的部分置为负无穷：
```python
scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

if attn_mask is not None:
    scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

attn_weights = F.softmax(scores, dim=-1)
```

不过目前高效的attention实现都被写进cuda了，在python层面直接把mask传进去即可，例如pytorch 2.2以上版本自带的Flash Attention：
```python
attn_output = F.scaled_dot_product_attention(
    Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
)
```

在对padding token的处理上，不同的attention实现有不同的处理方式，有的会置为0，有的不会管，导致padding token的输出非0，需要注意。


## FFN

### FFN的可解释性

FFN起到记忆的作用，用于记忆训练中见过的常见pattern。一些研究使用稀疏自编码器对FFN进行研究。

### SwiGLU

FFN的著名变体是[SwiGLU](https://arxiv.org/pdf/2002.05202)：

```python
mid = self.linear1(inputs) * F.silu(self.linear2(inputs))
outputs = self.linear3(mid)
return outputs
```

目前大模型都用这个，不过从个人经验来看不一定总是比FFN好。

### Mixture-of-Expert

MoE是稀疏版本的FFN，利用FFN激活值的稀疏性，在不大幅增加计算量的情况下扩大模型，在大模型、AIGC中被广泛使用，是一种有效的scaling路径。


## Normalization

transformer的训练十分困难，Normalization必不可少。

### PreNorm

<div align=center>
<img src="../../../assets/images/posts/2025-08-13/pre-norm.svg" width="40%" />
</div>

PreNorm可以显著降低transformer的训练难度，原因在于有短路连接的情况下，从输入可以“直达”输出，梯度传播更加顺畅。不过在相同的层数下，PreNorm的结果相比PostNorm要略微差一点，可能是因为PostNorm将归一化层放在主干上增加了模型的深度。

### RMSNorm

$$
\text{RMSNorm}(x) = g\odot \frac{x}{\sqrt{\frac1n\sum_{i=1}^{d} x_i^2}}
$$

RMSNorm是LayerNorm的替代归一化方式，相较于LayerNorm不会减去均值，训练参数上也不会加上一个bias。从个人经验上看，效果上几乎没有差异，但是RMSNorm在大模型中被更广泛的使用，尚不清楚原因。


## 大模型相关

### KVCache

大模型在生成时限制了每个token只和之前的token交互，因此在生成后面的token的时候，前面的token是不变的，无需重复计算，于是把这部分存起来就可以大幅降低生成的计算量。但是保存之前的token会占非常多的显存，因此和KVCache相关的技术是如何压缩KVCache显存占用，同时保证性能不要下降太多。

#### GQA

减少K和V的head数量，例如缩小k倍，但是Q的数量不变，在计算的时候把K和V重复k次实现attention。通过控制k的大小可以实现performance和内存的折中。

#### MLA

MLA是DeepSeek V2提出来的减少KVCache的方法，MLA神奇的地方是它不仅压缩了KVCache的大小，而且效果比原始的Multi-Head Attention还要好。

### Multi-Token Prediction

通过一次预测多个token而不是1个来加速生成。有许多论文都做MTP，这里介绍比较有名的DeepSeek V3的做法。
