---
title: 【深度学习】对比学习
tag: 深度学习 笔记
published: false
---

## 对比学习

### InfoNCE Loss

负样本数量的影响，温度系数的影响。

projection层。


#### 理解温度系数

在基于上述对softmax loss的理解基础上，再来看看给softmax加入常见的温度系数 $T$ 看看它会有什么影响。可以推导出梯度是：

$$
\begin{aligned}
\mathcal L &= \text{softmax}(\frac{z}{T}) \\
\frac{\partial \mathcal L}{\partial z} &= \color{red}{\frac1T}\color{black}(p^T-y^T )
\end{aligned}
$$

这里温度系数引入了对softmax平滑的先验，$T$ 越小（小于1），logits会被放得越大，经过softmax后越接近one-hot。但是这样有什么意义？注意softmax是对one-hot近似，它永远无法接近one-hot，loss永远不会为0。即使模型分类准确率100%了，模型总是能让正确的类分数最高，由于loss不为0，它还是会让分数进一步变大，这意味着特征更加靠近类中心了，类似于CenterLoss那样间接的给类之间加了margin。而 $T$ 控制了对one-hot的近似程度，它也就控制了这个间接margin的大小。

首先从优化角度理解温度系数的作用。根据[知乎文章](https://zhuanlan.zhihu.com/p/52108088)，小的温度系数会导致loss边界非常陡峭，进而导致优化过程中，特征不会更倾向于往类内聚拢，使得margin变小。

除此之外，温度系数也会影响loss对**难样本**的关注程度。在[这篇论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Understanding_the_Behaviour_of_Contrastive_Loss_CVPR_2021_paper.pdf)中，作者从单个负样本在所有负样本中的梯度占比进行分析：

$$
r(z_{i})
$$

随着 $T$ 减小，难样本的梯度占比指数级减小而总梯度不变，随着 $T$ 增大，简单和难负样本梯度会越来越接近。过小的 $T$ 会导致整个loss几乎只关注最难的一两个样本，当 $T\rightarrow 0$ 时，梯度为：

$$
-\frac1T\max(z_{max} - z_i, 0)
$$

负类梯度被唯一的样本主导，退化成了triplet loss，而且margin为0。当 $T\rightarrow +\infty$ 时，所有负样本有相同梯度。