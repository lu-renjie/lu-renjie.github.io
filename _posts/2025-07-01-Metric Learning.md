---
title: 【深度学习】Metric Learning
tags: 深度学习 笔记
published: false
---

看到知乎上的[这篇](https://zhuanlan.zhihu.com/p/45368976)文章对metric learning有了比较大的兴趣，于是整理学习了相关的内容，如有错误请指出。
<!--more-->

## Metric Learning

Metric Learning旨在给特定的任务学习合适的距离函数，使得该距离函数能够较好的衡量诸如图片这样的高维数据的相似度，以便进行最近邻检索之类的操作。在深度学习的场景下，Metric Learning很大程度上退化成了表征学习，如果能给两张图片 $I_1, I_2$ 提取足够好的特征 $f_1, f_2$，那么这两个特征之间的欧式距离就能用来衡量原来图片之间的距离，距离度量简单的变为 $$\text d(I_1, I_2)=\|f_1-f_2\|^2=\|\text{NN}(I_1)-\text{NN}(I_2)\|^2$$。只要能提取足够好的特征，Metric Learning似乎就被解决了。不过事情没有这么简单，即使我们把神经网络当成黑盒，并且把提取的特征当做线性表征，如何训练模型使得它提取的特征满足我们期望的性质，即适合计算距离，也不是容易的事情。

Metric Learning的方法可以分为两类，一类是Pair-based方法，通过拉近离正样本的距离，拉远和负样本之间的距离来学习合适的表征；另一类方法则是classification-based，通过分类任务实现metric learning，通常是魔改softmax函数。无论是哪种方法，都是通过约束特征之间的距离来学习距离度量。

### Pair-based

Pair-based方法的代表是[FaceNet](https://arxiv.org/pdf/1503.03832)提出的Triplet Loss。给定一个人脸的特征$f$， 取一张相同人脸的图片$f_+$，再取一张不同人脸的图片$f_-$，我们希望：

$$
\text d(f, f_+) < \text d(f, f_-)
$$

这里 $\text d$ 表示距离函数。该式子等价于：

$$
\text d(f, f_+) - \text d(f, f_-) \le 0
$$

为了使特征满足该式，只需要最小化 $$\text d(f, f_+) - \text d(f, f_-)$$即可，并且如果这个式子小于0，loss也为0，于是损失函数为：

$$
\begin{aligned}
\mathcal L &= 
\begin{cases}
\text d(f, f_+) - \text d(f, f_-), & \text d(f, f_+) > \text d(f, f_-)\\
0 ,& \text d(f, f_+) \le \text d(f, f_-)
\end{cases}

\\
&= \max\{\text d(f, f_+) - \text d(f, f_-), 0\}
\end{aligned}
$$

但是这样其实只要求了训练时 $\text d(f, f_+) \le \text d(f, f_-)$，即使模型在训练集上能把loss降低到0，我们也无法预期模型在遇到新的人脸时也能做到。为了使模型学到的特征更有判别力，Triplet Loss要求：

$$
\text d(f, f_+) - \text d(f, f_-) \le -m
$$

其中 $m>0$ 是一个超参数，表示margin，即不仅要求相同人脸特征距离更近，还要近不少。这样提高了训练难度，也给模型在泛化到新人脸留了容错。最后loss变为：

$$
\mathcal L = \max\{ \text d(f, f_+) - \text d(f, f_-) + m, 0 \}
$$

这就得到了Triplet Loss，或许是因为对每个图片样本（称为anchor），还要搭配一个正样本和负样本组成三元组 $(f, f_+, f_-)$ 进行计算，所以起名叫三元组损失。三元组损失非常简单，直观理解就是让anchor离正样本距离比负样本的距离小一个margin，属于比较早期的对比学习方法。


### Classification-based

Classification-based的方法没有pair-based方法那么直接，需要首先理解softmax函数的一些性质才能理解。所以下面首先介绍softmax分类以及相关性质，然后介绍代表性的Classification-based方法。

#### 理解softmax分类

分类的通用做法是用深度学习模型提取特征，后面接一个Linear层映射到类别维度，之后使用softmax+交叉熵损失进行训练，写成公式就是：

$$
\begin{aligned}
x &= \text{NN}(I)\\
y &= {\rm softmax}(z)={\rm softmax}(Wx+b)\\
\mathcal L &= \sum\limits_{i=1}^c -y_i\ln p_i=-y\ln p^T
\end{aligned}
$$

这里分成了两部分：第一部分是非线性的部分，神经网络提取图片的特征；第二部分是线性的部分，基于特征进行线性分类。第二部分是一个线性分类模型，容易推公式分析，同时对神经网络提取特征的学习有重要影响，所以值得研究。

这里首先对损失函数求梯度，softmax函数的雅克比矩阵是：

$$
{\rm diag}(y)-yy^T
$$

对角线是$y_i-y_i^2$，注意$y_i$在0到1之间，所以$y_i-y_i^2>0$，其余的元素是$-y_iy_j<0$。然后就可以计算得到**_softmax loss_**（softmax+交叉熵损失）的梯度：

$$
\frac{\partial \mathcal L}{\partial z}
=\frac{\partial \mathcal L}{\partial y}\frac{\partial y}{\partial z}
=-y^T {\rm diag}(\frac1p)[{\rm diag}(p)-pp^T]
=-y^T(I-1p)=y^T1p^T-y^T=p^T-y^T 
$$

这里有个反直觉的地方：对$z_i$的梯度只和$y_i$有关，但计算$y_i$是和所有$z$有关的，所以反向传播似乎应该和所有的$y$有关，但实际上它们消掉了。
{:.success}

对于one-hot标签就是（假设第$i$个类别是正类）：

$$
\frac{\partial{\mathcal L}}{\partial{z_i}} = \begin{cases}
p_i - 1,   &i=j\\
p_i  & i \ne j
\end{cases} 
$$

正类对应的logits的梯度为$y_i-1$，其它所有负类的logits的梯度为$\sum_{i≠j}y_i$。由于$\sum y_i=1$，softmax+交叉熵损失对正类的梯度和负类的梯度之和为0，并且二者大小相等，方向相反。正类的梯度小于0，它的值会上升，父类的梯度大于0，它的值会下降。进一步，如果把对正类的梯度拆分为$c-1$份，每份对应一个负类梯度：

<div align=center>
<img src="../../../assets/images/posts/2025-07-01/softmax_grad.png" width="50%" />
</div>

那这样就可以把softmax分类理解为训练了 $c$ 个分类器，每个分类器都做到了正类和负类的平衡，且会根据预测的难度大小自适应的调整每个负类的权重。难度大的类别 $p_i - y_i$ 整体会更大，所以梯度更大，难度小的则梯度更小。

softmax分类除了有良好的梯度性质，其训练得到的权重 $W$ 也有很直观的几何含义，不考虑bias的话，第 $i$ 个类别对应的参数 $w_i$ 从原点指向类别中心。

对于第 $i$ 和第 $j$ 个类别，当 $w_ix_i+b_i=w_jx_j+b_j$ 时说明模型认为样本属于两个类别的概率相同，对于线性模型而言这意味着刚好落在决策边界上，因此 $(w_i-w_j)^Tx+b_i-b_j=0$ 对应的超平面就是这两个类别的分界面，与 $w_i-w_j$ 垂直的超平面就是决策边界。

#### CenterLoss

首先需要明确使用分类任务训练的模型为什么不能提取适用于检索的特征？把每个人当做一个类别，让模型对人脸进行分类，模型应该也能实现对人脸区分？实际是不行的，[CenterLoss](https://kpzhang93.github.io/papers/eccv2016.pdf)论文给了一个很好的图进行解释：

分类只需要保证提取的特征落在对应类别的区域内即可，在决策边界处，不同类别的特征距离可能比相同类别的还近，即特征是Sepaerable的。但Seperable的特征对检索来说是不能接受的，检索需要的是discriminative feature。

CenterLoss的想法很直接，就是让类内的特征更紧凑一些，从而间接拉大类之间的距离，只是确定类中心着实有些麻烦。后续的方法避开了这个问题，直接去掉了bias部分，于是 $w_i$ 就是类别中心。并且由于去掉了bias，特征就一定是放射状的，此时在角度上约束特征更近就成了一个更简单直接的方法，于是有了在角度上加margin而不是在欧式距离上加margin的方法。

#### Angular Margin

L-Softmax

A-Softmax

[ArcFace](https://arxiv.org/abs/1801.07698)是Classification-based方法的代表，实现简单，不需要构建三元组，而且效果很好。ArcFace加margin的方式很直观：

<div align=center>
<img src="../../../assets/images/posts/2025-07-01/softmax_arcface.png" width="50%" />
</div>

上图可视化了二维的特征，使用softmax训练得到的特征边界模糊，而使用ArcFace训练得到的特征边界清晰，不同人脸特征之间的角度远远大于相同人脸的角度。

ArcFace的做法是把softmax进行如下替换：

$$
\frac{
    e^{w_i^Tf_i}
}
{
    \sum_{j=1}^n e^{w_j^Tf_j}
}

\rightarrow

\frac{
    e^{s\cos (\theta_i+m)}
}
{
    \sum_{j=1}^n e^{s\cos (\theta_j+m)}
}
$$

这里$s>1$是超参数（通常取10到40），$$\cos\theta_i=\frac{w_i^Tf}{\|w_i\|\|f\|}$$是特征$f$和类别中心$w_i$的夹角余弦值，$m$则是margin。softmax替换后可以选择交叉熵或者Focal Loss进行分类。这里的替换涉及两个要点，一是为什么把 $w_i^Tf_i$ 替换成 $s\cos (\theta_i)$，二是margin为什么这么加。

$w_i^Tf_i$替换成$\cos (\theta_i)$可以参考[这里](https://zhuanlan.zhihu.com/p/49939159).

为什么加margin可以参考[这里](https://zhuanlan.zhihu.com/p/52108088).

ArcFace的angular margin对应着弧距(arc margin)（也叫geodesic测地距离）

把margin加在$\cos$里是ArcFace的创新点，这样加的几何意义如下图所示：

<div align=center>
<img src="../../../assets/images/posts/2025-07-01/arcface.png" width="70%" />
</div>

### 统一视角

https://www.zhihu.com/question/440729199/answer/1704992808

#### Circle Loss

Circle Loss不仅给两类metric learnning方法提供了统一的视角，还可以用来实现多标签分类。
