---
title: 【推荐系统】推荐系统中关注的话题
tags: 推荐系统 深度学习 笔记
---

<!-- 
updates:
* 202508:一边面试，一边学习，花了很久写的
* 202509: 开始泛读论文去写
-->


推荐系统模型很直接，实质上还是在学特征之间比较基础的联系，比如线性回归的加权，LR的baseline本身结果就很好，不需要学习CV或NLP里面那些很high-level的抽象表征。

### 用户体验与业务目标

#### Feedback Loop问题

推荐系统决定用户看到的内容，用户的行为会以训练数据或序列特征的形式影响推荐系统，形成一个闭环。这样会导致马太效应：item本身由于内容差异以及冷启动问题会在热度上形成一个长尾分布，而由于推荐系统feedback loop的特征，被推荐得越多的item用户看到的也越多，导致长尾加剧。热门内容越来越热门，冷启动问题需要专门的设计进行解决。

Feedback Loop and Bias Amplification in Recommender Systems

Filter Bubble。

#### 内容分层

公司内部会根据内容标签，比如游戏、生活等内容会由不同的团队设计模型，针对这样更专门的场景进行优化。在AB测试时，一种常出现的情况是在一些内容上负向，但是在其它内容上正向，这个要根据运营目标抉择。

内容可以分为UGC和PGC，内容是流量平台的基石，运营好平台内容非常重要。PGC是平台内的专业内容，可以吸引专业用户或者带有学习目的的用户；UGC是平台内的非专业内容，可以保证平台内容多样性，增强用户互动性。

内容可以按热门度划分，按类别划分等。热门的内容可能是整活，短期热门，也可能是专业度很高的内容，长期热门。

#### 用户分层

不同用户的兴趣差异非常大，用户使用APP的目的也各不相同，对用户分层的理解有助于推荐系统的宏观决策。

用户可以按年龄划分，按使用APP时长划分（高价值用户），按职业划分，按性别划分。在跨业务的时候可以迁移其它业务的高价值用户。

APP的用户可以分为深度用户和新回用户（新用户、回流用户、未注册用户），深度用户推荐算法不太会影响他们的留存，更多看平均使用时长，而新用户体验不好可能就不会使用这个APP了，看重DAU。

* User Journey
* User冷启动（用户增长）
* Item冷启动
* 推荐多样性
* 安全性和鲁棒性：如何避免刷榜、套利等软件中存在的问题


### 推荐系统的各种Bias

[推荐系统中的各种bias](https://zhuanlan.zhihu.com/p/428037218)

#### Sample Selection Bias

全量样本>召回样本>曝光样本>点击样本>转化样本。

对于CVR模型，模型用于训练的样本都是click的样本，但实际推理的时候会对所有send的样本进行CVR预测，包括click和非click的样本，因此存在bias。

CTR模型不存在严重的SSB问题，但召粗模型和CVR模型存在SSB问题。

实际上所有模型都有一定的SSB问题，模型在线上总会遇到新的item、新的用户，都是训练集所没见过的。

#### Popularity Bias

* 热门样本作为正样本的机会太多，在采样负样本的时候采样更多热门样本作为负样本。
* 避免热门样本占据太多训练机会。通过自监督学习增强表征，随机采样样本进行自监督学习保证了每个item都能得到平等的训练机会。

#### Position bias

指由于推送内容在app中的位置导致的bias，比如双栏推荐，用户比较喜欢看第一个，用户没有点击的item并不意味着他不感兴趣。

[https://zhuanlan.zhihu.com/p/439499892](https://zhuanlan.zhihu.com/p/439499892)

简单的解决办法是加特征，把位置也作为特征输入进去。

PAL。

#### 其它bias
* extreme bias：只有用户非常喜欢或不喜欢的时候才会进行点赞或收藏等行为
* conformity bias：大多数人的评价会影响用户判断


### 模型改进

#### 高估低估优化

https://zhuanlan.zhihu.com/p/16484895233

广告系统高估低估，平滑校准算法。

* PCOC

#### 特征建模

* 连续特征：
    * 保持连续值输入：直接输入，或者经过归一化、取对数、标准化等操作后输入
    * 转化为离散特征：各种分桶方法
    * Soft离散化：根据离散化的思路设计输入神经网络处理连续值，例如[AutoDis](https://arxiv.org/pdf/2012.08986)。
* 离散特征：过Embedding层，Embedding层有很多相关研究
    * Embedding压缩（假设原本embedding大小为n）：
        * 引入淘汰机制：例如只给出现频率高于阈值的添加embedding、LossAwareEvicit、去掉30天没出现的值的embedding
        * 量化压缩：把embedding层降低到半精度
        * 神奇的[Deep Hash Embedding](https://arxiv.org/pdf/2010.10784)
        * 基于哈希的方法：[Double Hash](https://arxiv.org/abs/1709.03933)、[Hybrid Hash](https://arxiv.org/pdf/2007.14523)
    * Embedding层的结构：残差embedding、
* 多模态特征建模：RQVAE、RQKMeans

#### 行为序列建模

推荐系统希望建模用户兴趣或意图，判断依据主要有两类：一是基于用户画像，二是用户行为。前者可以说是一些刻板印象，后者则更能反映用户的个性化偏好，一般各种行为特征占了总共特征数量的90%，剩下的才是用户画像、item特征、context特征。

* 长序列建模
    * SIM
    * ETA
    * TWIN
* 序列推荐的范式
    * SASRec
    * PinnerFormer
    * HSTU

#### 交叉模型

* gated DCN，DHEN，RankMixer

#### LTV建模

建模用户长期价值。

* 预测用户长期价值
* 基于强化学习的推荐

#### 新的召回范式

* 生成式召回
* 多兴趣建模：TDM、Deep Retreieval、Trinity

#### 新的范式

* 生成式推荐：TIGER、OneRec、OneRecV2

### 线上线下不一致

离线数据训练有提升，但是线上不一定有提升，原因在于：
- 除此之外，在线和离线的实现会有很多差异。
- SSB问题。粗排送过来的数据和训练用的曝光过的数据不同
- 线上存在大量新样本，与离线不一致
- 特征差异：
    - match特征存在穿越问题
    - example age的穿越问题

[https://zhuanlan.zhihu.com/p/42521586](https://zhuanlan.zhihu.com/p/42521586)
[https://blog.csdn.net/legendavid/article/details/80653433](https://blog.csdn.net/legendavid/article/details/80653433)
