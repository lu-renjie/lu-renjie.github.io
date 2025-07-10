---
title: 【深度学习】Metric Learning
tags: 深度学习 笔记
published: true
key: 2025-07-10-comment-1
---

看到知乎上的[这篇](https://zhuanlan.zhihu.com/p/45368976)王峰大佬的文章对metric learning有了比较大的兴趣，于是整理学习了相关的内容，如有错误请指出。
<!--more-->


Metric Learning旨在给特定的任务学习合适的距离函数，使得该距离函数能够较好的衡量诸如图片这样的高维稀疏数据的语义相似度，以便进行最近邻检索之类的操作。在深度学习的背景中，Metric Learning很大程度上退化成了表征学习，如果能给两张图片 $I_1, I_2$ 提取足够好的特征 $f_1, f_2$，那么这两个特征之间的欧式距离就能用来衡量原来图片之间的距离，距离度量简单的变为 $$\text d(I_1, I_2)=\|f_1-f_2\|^2=\|\text{NN}(I_1)-\text{NN}(I_2)\|^2$$。只要能提取足够好的特征，Metric Learning似乎就被解决了。不过事情没有这么简单，即使我们把神经网络当成黑盒，并且把提取的特征当做线性表征，如何训练模型使得它提取的特征满足我们期望的性质，即适合计算距离，也不是容易的事情。

Metric Learning的方法可以分为两类，一类是Pair-based方法，通过拉近离正样本的距离，拉远和负样本之间的距离来学习合适的表征；另一类方法则是classification-based，通过分类任务实现metric learning，通常是魔改softmax函数。无论是哪种方法，都是通过约束特征之间的距离来学习距离度量。

## Pair-based

Pair-based方法似乎不常用了，所以这里简单记录两个经典的方法作为了解。

### DeepID2

[DeepID2(2014)](https://arxiv.org/pdf/1406.4773)基于一对人脸的特征 $<f_i, f_j>$ 计算如下的loss：

$$
\mathcal L_{deepID2} = \begin{cases}
\frac12\|f_i - f_j\|^2 & \text{if}\ i, j\ \text{are same face}\\
\max(0, m - \frac12\|f_i - f_j\|^2) & \text{if}\ i, j\ \text{are different face}
\end{cases}
$$

其中 $m$ 表示margin。这个loss的含义非常直观，让相同人脸的特征接近，让不同人脸的特征远离到大于等于 $m$ 为止。

### Triplet Loss

DeepID2提出的loss是很直接的metric learnning思路，是一种基于二元组的对比损失。[FaceNet(2015)](https://arxiv.org/pdf/1503.03832)提出了一种三元组Triplet Loss（看论文里说的，Triplet Loss其实来源于一篇05年的[论文](https://papers.nips.cc/paper_files/paper/2005/hash/a7f592cef8b130a6967a90617db5681b-Abstract.html)），基于三元组计算loss，用一个loss实现同时拉近相同人脸特征，拉远不同人脸特征。给定一个人脸的特征$f$， 取一张相同人脸的图片$f_+$，再取一张不同人脸的图片$f_-$，组成一个三元组 $<f, f_+, f_->$，Triplet Loss为：

$$
\mathcal L_{tri} = \max\{ \text d(f, f_+) - \text d(f, f_-) + m, 0 \}
$$

三元组损失也非常简单，可以认为就是把DeepID2的loss合并成一个，肯定要比二元组的方便一些，也少了构造数据上的类别均衡问题。但是这种pair-based方法存在一个比较大的缺陷是需要构造难的样本才能得到好的效果，因为这个loss太容易达成了，对于大部分的三元组，模型都能保证不同人脸的特征差异大于 $m$。为了解决这个问题，Triplet Loss通常都要搭配难样本挖掘的方法，在FaceNet中，会随着训练过程逐渐加大难度，实现课程学习。最难的样本显然是在整个数据集里找一个里 $f$ 最远的样本特征作为 $f_+$ 和 最近的作为 $f_-$ 构成三元组，即：

$$
\mathcal L_{hard,tri} = \max_{i,j}\{ \text d(f, f_+^i) - \text d(f, f_-^j) + m, 0 \}
$$

但是这种做法需要在每一步迭代都遍历数据集给每个样本提特征，计算量太大，折中的方法是只在batch内寻找，或者放宽寻找的条件，这里就不细说了，具体可以参阅原论文。

## Classification-based

尽管Pair-based方法非常简单直接，效果也挺好，但是构造元组样本还是很麻烦。后续的Classification-based的方法更简单，只需要魔改softmax，而且效果更好。但是要理解这些方法需要先理解softmax函数的一些性质，所以下面首先介绍softmax分类以及相关性质，然后介绍代表性的Classification-based方法。

### 理解softmax分类

分类的通用做法是用深度学习模型提取特征，这里记为 $x$，后面接一个Linear层映射到类别维度得到logits $z$，之后使用softmax+交叉熵损失进行训练，写成公式就是：

$$
\begin{aligned}
x &= \text{NN}(I)\\
p &= {\rm softmax}(z)={\rm softmax}(Wx+b)\\
\mathcal L &= \sum\limits_{i=1}^c -y_i\ln p_i=-y^T\ln p
\end{aligned}
$$

这里分成了两部分：第一部分是非线性的部分，神经网络提取图片的特征；第二部分是线性的部分，基于特征进行线性分类。第二部分是一个线性分类模型，容易推公式分析，同时对神经网络提取特征的学习有重要影响，所以值得研究。

#### 梯度性质
这里首先对损失函数求梯度，softmax函数的雅克比矩阵是：

$$
{\rm diag}(p)-pp^T
$$

对角线是$p_i-p_i^2$，注意$p_i$在0到1之间，所以$p_i-p_i^2>0$，其余的元素是$-p_ip_j<0$。然后就可以计算得到**_softmax loss_**（指softmax+交叉熵损失）的梯度（雅克比矩阵相乘）：

$$
\frac{\partial \mathcal L}{\partial z}
=\frac{\partial \mathcal L}{\partial p}\frac{\partial p}{\partial z}
=\left[ -y^T {\rm diag}(\frac1p)\right] \left[{\rm diag}(p)-pp^T \right]
=-y^T(I-\textbf{1} p)=y^T\textbf {1}p^T-y^T=\color{red}{p^T-y^T} 
$$

这里 $\textbf{1}$ 表示全为1的列向量。有个反直觉的地方是对$z_i$的梯度只和$y_i$有关，但计算$y_i$是和所有$z$有关的，所以反向传播似乎应该和所有的$y$有关，但实际上它们消掉了。

对于分类来说，$y$ 是one-hot向量，那么梯度就是（假设第$i$个类别是正类）：

$$
\frac{\partial{\mathcal L}}{\partial{z_i}} = \begin{cases}
p_i - 1,   &i=j\\
p_i  & i \ne j
\end{cases} 
$$

正类对应的logits的梯度为$y_i-1$，其它所有负类的logits的梯度为$\sum_{i≠j}y_i$。由于$\sum y_i=1$，softmax+交叉熵损失对正类的梯度和负类的梯度之和为0，并且二者大小相等，方向相反。**正类的梯度小于0，梯度下降后它的值（logits）会增大，负类的梯度大于0，它的值会减小**，也就是让模型输出的正确的类分数更大，错误的类分数更小。除此之外，softmax loss还有个很大的好处是对 $z_i$ 的绝对值之和只有2，其中正类是1，负类也是1。这种恒定的梯度大小会使优化更加简单，也让我们可以很方便的调整学习率。

进一步，如果把对正类的梯度拆分为$c-1$份，每份就可以对应一个负类梯度：

<div align=center>
<img src="../../../assets/images/posts/2025-07-12/softmax_gradient.svg" width="60%" />
</div>

这样就可以把softmax分类理解成训练了 $c$ 个二分类器，**每个二分类器把一个类作为正类，其余的所有类都作为负类进行判断**。这样的每个分类器都做到了正类和负类梯度的平衡，而且会根据预测的难度大小自适应的调整每个负类的权重，难度大的负类 $p_i - y_i$ 会更大，梯度更大，难度小的则梯度更小。

Softmax loss除了有上述良好的梯度性质，其训练得到的权重 $W$ 也有很直观的几何含义，不考虑bias的话，第 $i$ 个类别对应的参数 $w_i$ 从原点指向这个类别比较中心的位置。因为经过训练后需要保证 $w_i^Tx$ 比较大才能正确分类。

另外，从 $W$ 和 $b$ 我们也能得到分类超平面。对于第 $i$ 和第 $j$ 个类别，当 $w_ix_i+b_i=w_jx_j+b_j$ 时说明模型认为样本属于两个类别的概率相同，对于线性模型而言这意味着刚好落在决策边界上，因此 $(w_i-w_j)^Tx+b_i-b_j=0$ 对应的超平面就是这两个类别的分界面，与 $w_i-w_j$ 垂直的超平面就是决策边界。

<!-- 
#### 理解温度系数（WIP）

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

负类梯度被唯一的样本主导，退化成了triplet loss，而且margin为0。当 $T\rightarrow +\infty$ 时，所有负样本有相同梯度。 -->



### CenterLoss

既然softmax loss有这么良好的性质，那么可以用分类任务进行度量学习吗？例如把每个人当做一个类别，让模型对人脸进行分类，模型应该也能实现对人脸区分？这是不行的，[CenterLoss](https://kpzhang93.github.io/papers/eccv2016.pdf)论文给了一个很好的图进行解释：

<div align=center>
<img src="../../../assets/images/posts/2025-07-12/seperable.png" width="40%">
</div>

Serperable feature需要保证的是特征距离正类中心比负类中心更近，但是不保证类内特征之间的距离比类之间特征的距离更近，Discriminative feature是保证后者的。分类得到的是Seperable feature，所以无法进行度量学习。

要改进softmax分类进行度量学习，CenterLoss的想法很直接，就是让类内的特征更紧凑一些，从而间接拉大类之间的距离：

$$
\mathcal L = \sum_{i=1}^c \text{CELoss}(p, y) + \frac{\lambda}{2} \|x - c_y\|^2
$$

这里 $x$ 表示特征， $c_y$ 表示 $x$ 所属的类别的中心特征。这里的问题是确定类中心有些麻烦，CenterLoss用了个不太优雅的方式：把 $c$ 也作为可学习的参数，同时人为定义它的更新梯度为：

$$
\Delta c_j = \alpha\frac{\sum_{i=1}^m \mathbb I(y_i=j)(c_j - x_i)}{1 + \sum_{i=1}^m\mathbb I(y_i=j)}
$$

其中 $c_j$ 表示第 $j$ 类的中心；$m$ 是batch的大小；$y_i$ 表示batch内第 $i$ 个样本所属的类别；$\mathbb I$ 是示性函数，当条件满足时为1，否则为0；$\alpha$ 是给 $c_j$ 专门设置的学习率，论文中设为0.5。这个式子也好理解：$c_j - x_i$ 是 $$\frac12\|c_j - x_i\|^2$$ 的梯度，让类中心靠近样本，加上一堆示性函数的作用是只让第 $j$ 类的样本更新 $c_j$。整个梯度相当于让 $c$ 逐步朝类中心的方向移动。

尽管CenterLoss的方法直接有效，但CenerLoss只约束了类内距离更小，没有约束类之间的距离更大，而且维护类中心的方法也挺麻烦。后续的方法直接去掉了bias部分，于是 $w_i$ 就相当于类中心。并且由于去掉了bias，特征就一定是放射状的，此时在角度上约束特征更近就成了一个更简单直接的方法，于是有了在角度上加margin而不是在欧式距离上加margin的方法。


### Angular Margin

Softmax的每个logit值是 $w_i^Tf+b_i$，[L-Softmax(2016)](https://arxiv.org/abs/1612.02295)提出在角度上加margin，把 $w^Tf$ 拆解成 $$\|w_i\|\|f\|\cos\theta$$，并通过 $\theta$ 控制角度上的margin：

$$
\mathcal L_{l-softmax} = -\ln\frac{
    e^{\|w_i\|\|f\|\cos(\color{red}{m}\theta_i)}
}
{
    e^{\|w_i\|\|f\|\cos(\color{red}{m}\theta_i)} + \sum_{j\ne i}^n e^{\|w_i\|\|f\|\cos\theta_j}
}
$$

$$\cos\theta_i=\frac{w_i^Tf}{\|w_i\|\|f\|}$$ 的含义是特征 $f$ 和类别中心 $w_i$ 的夹角余弦值，$m$用来控制margin，通常取4。因为 $\cos$ 函数是个递减函数，所以**乘以一个大于1的倍数会减小logit值，进而让loss变大，因此优化到和不加margin相同的loss大小特征之间的margin更大**。

这个做法非常简单，实质上只修改了最后的softmax。softmax替换后可以选择交叉熵或者Focal Loss进行分类。当然实际上会比这复杂，例如夹角的范围应该在$0~\pi$之间，乘了 $m$ 会超出这个范围，所以需要一些额外的处理。但是这里为了说明核心思想就忽略这一点。另外，为了保证角度的含义，这里也去掉了bias，这样不会影响效果，也可以简化理论上的分析。




### L2 Normalization

[SphereFace(2017)](https://arxiv.org/abs/1704.08063)在L-softmax的基础上进一步强化了角度的学习。考虑到softmax的决策边界是 $(w_1^T-w_2^T)f+b_1-b_2=0$，如果令 $\|w_1\|=\|w_2\|=1$ 且 $b_1=b_2=0$，那么边界就变为 $\|f\|(\cos\theta_1 - \cos\theta_2)=0$，边界开始纯粹与角度相关，和参数无关，和L-softmax相比无疑是更纯粹的角度margin方法，实验也表明这样效果更好。因此，SphereFace提出把参数进行归一化，也就是令 $w_i=1$，于是loss变为:

$$
\mathcal L_{sphereface} = -\ln\frac{
    e^{\|f\|\cos(m\theta_i)}
}
{
    e^{\|f\|\cos(m\theta_i)} + \sum_{j\ne i} e^{\|f\|\cos\theta_j}
}
$$

既然归一化了参数 $w$，那为什么不把 $f$ 也归一化？实际上当时的方法都会在测试的时候对 $f$ 归一化，那为什么不在训练的时候归一化？因为这会不收敛。[NormFace(2017)](https://dl.acm.org/doi/pdf/10.1145/3123266.3123359)深入研究了这个问题，并给出了解决方案：把 $w_i$ 和特征 $f$ 归一化之后，再乘以 $s=\frac1T > 1$： 

$$
\mathcal L_{normface} = -\ln
\frac{
    e^{\color{blue}{s} \color{black}\cos\theta_i}
}
{
    e^{\color{blue}{s} \color{black}\cos\theta_i}
    +
    \sum_{j\ne i} e^{\color{red}{s} \color{black}\cos(\theta_j)}
}
$$

$s$ 是可学习参数，通常取值是几十，作用类似于温度系数。关于为什么要把特征归一化，以及为什么要乘以 $s$，下面后面会进行分析。
在这之后，[CosFace(2018)](https://arxiv.org/pdf/1801.09414)和[AM-Softmax(2018)](https://arxiv.org/abs/1801.05599)在NormFace的基础上改变了加margin的方式（两篇论文撞车了），变成了加性的margin：

$$
\mathcal L_{am-softmax} = -\ln
\frac{
    e^{\color{blue}{s} \color{black}[\cos\theta_i \color{red}{-m} ]}
}
{
    e^{\color{blue}{s} \color{black}[\cos\theta_i \color{red}{-m} ]}
    +
    \sum_{j\ne i} e^{\color{red}{s} \color{black}\cos\theta_j}
}
$$

加性的margin会更好收敛。到这里，用classification-based方法做人脸识别效果已经非常好了。

#### 为什么要归一化

<div align=center>
<img src="../../../assets/images/posts/2025-07-12/feature_visualization.png" width="60%" />
</div>

1. 上图是CenterLoss提供的MNIST训练集和测试集的特征可视化结果，可以看出训练后的特征呈现放射状（结合前面分析的softmax loss梯度性质很容易理解这是因为梯度喜欢正类特征变长）。这样的放射状特征本身不满足Metric Learnning的要求，但是归一化之后能显著增大不同类别之间的margin，显然更好。实验结果也表明，测试时先归一化特征，再进行最近邻检索能有提升。

3. 归一化之后，L2距离和余弦距离等价，因为由 $$\|x\|=\|y\|=1$$ 可以得到 $$\|x - y\|^2 = 2 - 2x^Ty$$，再也不用纠结用哪个了~

3. 固定长度的特征相当于固定了单位，margin含义相对固定，$m$ 只需要在一个量级内调参。

#### 为什么乘以$s$

1. 如果仅仅使用余弦相似度，由于$\cos\theta$的范围是$[-1, 1]$，经过softmax后难以近似one-hot，softmax loss会一直很高，带来优化问题。根据前面的分析，softmax loss的梯度会均匀分配给正类和负类，大的softmax loss会导致梯度一直很大，即使前面的神经网络已经提取了很好的特征，这种大的loss也会破坏模型的学习。

2. 前面提到softmax的温度系数可以控制loss对难样本的关注程度以及类之间的margin，由于 $s = \frac1T$，$s$ 是温度系数的倒数，$s$ 越大，loss就越关注难样本，但同时也会导致类之间的margin变小。所以一个合适的 $s$ 既能在一定程度上关注难样本，也保证一定的margin大小。

综上，引入 $s$ 在特征归一化的情况下能解决优化问题，同时也使得loss在更加关注难样本的同时容许了类之间的margin，使得classification-based方法能以一种简单的方式解决Triplet Loss做起来比较麻烦的难三元组构造。

#### 梯度性质（WIP）

这部分内容来源于NormFace。

### ArcFace（WIP）

[ArcFace(2019)](https://arxiv.org/abs/1801.07698)是classification-based方法的集大成者（或者说是margin softmax这类方法），效果非常好。做法是固定了 $s=64$，并把margin从$cos$外面移动到里面：

$$
\mathcal L_{arcface} = -\ln
\frac{
    e^{s[\cos(\theta_i\color{red} +m \color{black})]}
}
{
    e^{s[\cos(\theta_i\color{red} +m \color{black})]}
    +
    \sum_{j\ne i} e^{\color{black}\cos\theta_j}
}
$$

并且对比了不同加margin方式的区别：

<div align=center>
<img src="../../../assets/images/posts/2025-07-12/arcface.png" width="70%" />
</div>

无论是哪种加margin的方式，都是让logit变小，使得相同margin下的loss变大，迫使模型学到更大的margin。ArcFace的angular margin对应着弧距(arc margin)，所以叫ArcFace。


## 统一视角

<!-- https://www.zhihu.com/question/440729199/answer/1704992808 -->

### Circle Loss

[Circle Loss(2015)](https://arxiv.org/pdf/2002.10857)不仅给两类metric learnning方法提供了统一的视角，还可以用来实现多标签分类。基于AM-Softmax进行推导：

$$
\begin{aligned}
\mathcal L_{am-softmax} &= -\ln
\frac{
    e^{s(\cos\theta_i-m)}
}
{
    e^{s(\cos\theta_i-m)}
    +
    \sum_{j\ne i} e^{s\cos\theta_j}
}\\

&=

\ln\left\{
\frac
{
    e^{s(\cos\theta_i-m)}
    +
    \sum_{j\ne i} e^{s\cos\theta_j}
}
{
    e^{s(\cos\theta_i-m)}
}
\right\}\\

&=

\ln\left\{
1 + \sum_{j\ne i} e^{\color{red}  s\cos\theta_j - s\cos\theta_i + sm}

\right\}

\end{aligned}
$$

这里已经可以看出最后得到的式子指数部分和triplet loss很像。Circle Loss考虑到了这一点，提出了下面这个loss：

$$
\mathcal L_{circle}
= \ln\left\{
1 + \sum_{i=1}^K e^{ - \gamma s_p^i}\sum_{j=1}^L e^{\gamma(s_n^j + m)}
\right\}
$$

其中 $s_n^j$ 为anchor $f$ 和第 $j$ 个负样本的相似度，$s_p$ 为 $f$ 和第 $i$ 个正样本的相似度，$K$ 表示正例数量，$L$ 表示负例数量，$m$ 是margin，$\gamma$ 是 scale factor，对应原来的 $s$。这个loss很有意思，做了多个改进：

1. 把余弦相似度 $cos\theta$ 推广为相似度 $s_n$，于是 $s_n$ 可以由不同的相似度方法替代。

2. 以一个**合适的方式**将单个正样本扩展到了多个，使之适用于多标签分类。

先说第一点。如果把正例负例取类别的“中心” $w_i$，那么正例只有一个，再采用余弦相似度，可以直接得到AM-Softmax；如果把正例负例取其它样本的特征 $f_+^i$ 和 $ f_-^j $，也让正例只有一个，并且把相似度定义为L2距离的相反数，对 $\gamma$ 取极限可以得到考虑难样本挖掘的Triplet loss：

$$
\begin{aligned}

\lim\limits_{\gamma\rightarrow\infty} \frac1\gamma\mathcal L_{uni}
&= \max_{i,j} \{s_n^j - s_p^i + m, 0\}\\
&= \max_{i,j} \{-\|f-f_{-}^j\| + \|f-f_{+}^i\| + m, 0\}\\
&= \mathcal L_{hard,tri}
\end{aligned}
$$

这里需要用到LogSumExp（LSE）函数的性质，即LSE函数是max函数的平滑版本，极限情况下二者相等：

$$
\max\{x_1, \cdots, x_n\}
= \lim_{\gamma\rightarrow\infty}\frac1\gamma\text{LSE}_{\gamma}(x_1, \cdots, x_n)
= \lim_{\gamma\rightarrow\infty}\frac1\gamma\ln\left(\sum_{i=1}^n e^{\gamma x_i}\right)
$$

综上就可以理解为什么说这个loss统一了前面提到的pair-based和classification-based的损失函数。无论是哪种loss，本质上都在最大化 $s_p$和最小化 $s_n$，只是正负样本选取的对象和方式不同。

关于第二点，如何处理多标签分类一直是个麻烦的问题，Circle Loss给出了一个优雅的解决方案，正类和负类天然是均衡的。具体而言，Circle Loss对 $\max$、$\min$ 以及 $\max(x, 0)$ 函数都进行了平滑化：

$$
\begin{aligned}
\min_i\{s_p\} &\approx \ln\sum_i e^{ - \gamma s_p^i}\\
\max_j\{s_n\} &\approx \ln\sum_{j=1}^L e^{\gamma(s_n^j + m)}\\
\max(x, 0) &\approx \ln(1+e^x)
\end{aligned}
$$

进而可以知道Circle Loss其实是在优化下面的函数（为了公式间接用ReLU替换了$$\max\{x, 0\}$$）：

$$
\mathcal L_{circle} \approx \text{ReLU}\left(  \max_j\{s_n^j\} - \min_i\{s_p^j\}+m  \right)
$$

这意味着Circle Loss会自动找难样本进行对比学习的优化，并给正类和负类分配大小相等、方向相反的梯度，且梯度绝对值都是1。而且因为这些近似的替换是这个式子的smooth版本，会类似于softmax**把总和为1的负类梯度分配给每个负类一样，把总和为1的正类梯度分配给每个正类，也把总和为1的负类梯度分配每个负类**，实现巧妙的类别均衡，如下图所示。

<div align=center>
<img src="../../../assets/images/posts/2025-07-12/circleloss_gradient.svg" width="40%" />
</div>

### 再看Softmax

在Circle Loss的视角下，令 $K=1$, $\gamma=1$, $m=0$ 就能得到softmax loss：

$$
\ln\left\{
1 + e^{-s_p}  \sum_{j}^L e^{s_n^j}
\right\}

=

-\ln\frac{e^{s_p}}
{e^{s_p} + \sum_{j=1}^L e^{s_n}}

=\text{softmax loss}
$$

结合前面Circle loss是公式 $(23)$ 的smooth版本的理解，softmax就是 $$ \text{ReLU}(\max_j\{s_n^j\} - s_p) $$ 的smooth版本。得益于LSE函数的梯度是softmax函数，使用LSE替换 $\max$ 函数能保持梯度总和为1，真是一个非常好的性质。

<!-- ### 为什么叫Circle（WIP）

实际上原论文中的Circle loss的motivation是给 $s_n$ 和 $s_p$ 加权。

## 总结（WIP）

还有[L2 Softmax](https://arxiv.org/pdf/1703.09507)看看。
{:.info} -->
