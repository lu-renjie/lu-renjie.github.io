---
title: 【大模型】入门学习
tags: 深度学习 大模型 笔记
published: false
---

入门学习一下大模型。

<!--more-->

## 大模型介绍

## Pretrain

## Post-Training

### 偏好对齐

在标注的偏好数据上训练。

#### SFT

#### RLHF

Reward模型使用BPR Loss进行训练。
PPO是强化学习的经典算法，实现简单，效果好，但是trick非常多，有[博客](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)专门讲PPO实现的37个trick。

#### DPO
DPO直接在偏好数据上进行训练，把RL和Reward模型训练合并成一个环节。



## 推理模型

PPO前面介绍了，这里介绍一些针对大模型场景改进的强化学习算法。

### DeepSeek R1

1. 使用强化学习训练DeepSeek-R1-Zero做数学题，训练推理能力。
2. 使用DeepSeek-R1-Zero生成的推理样本给DeepSeek-R1训练作为冷启动
3. 使用推理数据+偏好数据用SFT训练DeepSeek-R1
4. 使用推理reward和偏好reward训练DeepSeek-R1

#### GRPO

GRPO在DeepSeek-Math中提出，是一个比PPO更简单的算法，可以理解为针对大模型优化的REINFORCE算法。

#### GSPO

按QWen团队的说法，用GRPO在MOE架构的模型上根本不work（说GRPO是DeepSeek团队的烟雾弹），原因在于MOE的稀疏性，上一步训练的几个expert在下一次迭代是可能完全没用到，给RL训练带来困难，为此QWen团队提出了GSPO。

#### GAPO



从GPT3开始，模型变得非常大，并学习到了非常多的知识。而随着prompt learning的发展，大模型的知识被发掘并逐渐变得可用，于是引发的大模型的热潮。



## 大模型预训练

### 数据

互联网数据、维基百科、代码数据、书籍数据、等等。

如何清洗？[https://www.cnblogs.com/theseventhson/p/18293145](https://www.cnblogs.com/theseventhson/p/18293145)

- 先收集高质量数，如维基百科、书籍、代码等
- 爬取数据
- 个人信息数据去除
- 去重
    - 使用minHash和LSH去重
        - minHash用于对集合计算哈希，对于爬取的文本，会先分词再计算它的minHash
        - LSH集合minHash可以用来快速查找相似的文本

        python有个叫datasketches的库用来做这个。
    - 按行去重
    - 规则过滤，人为总结pattern
- 质量过滤
    - 使用质量打分模型
    - 语种分类
    - 使用训练好的语言模型过滤高困惑度文本
    - 人工筛选
- 数据混合
    - 通用知识类
    - 代码和推理类
    - 多语言

不同的数据是采样训练的，从不同质量、语种、类型的数据随机选择，每种类型的数据会事先concat到一起，并划分成chunk，随机选一个进行训练。

### 训练

- 初始预训练，通常长度为8192

    简单的Next Token Prediction。
- loss spike问题
- 逐步增加上下文长度，进行长上下文训练

## Post training

模型通过post training学习post training data的“人格”，如果这些数据总是自信的进行一些回答等等，那么大模型就几乎总是会自信的回答问题，进而出现问题，即使模型有能力知道这个东西它并不熟悉。 

### Post training——SFT

instructGPT

这里讲得很详细：[https://zhuanlan.zhihu.com/p/677607581](https://zhuanlan.zhihu.com/p/677607581)

1. SFT

    OpenAI让40名labeler想的一些prompt，包括简单的任务；Few-shot任务；根据GPT3用户的数据让labeler写相应的指令。

    这些数据先用来微调GPT3，得到一个reference模型。

### Post training——RLHF

SFT可以理解为看书，而RL则是在做练习题，并根据答案的反馈进行训练。

1. 训练reward模型
    - 收集一些prompt，让大模型生成很多数据
    - 让人类打分或排序
    - 训练一个reward模型

    这里$x$是prompt，$y_{win}$是分数更高的回答，$y_{lose}$是分数更低的回答。

$$
\max \sigma(\sigma(r(x,y_{win})-r(x,y_{lose})))
$$
2. RLHF
    - 收集一些prompt，这些prompt将作为环境

        数据也来源于GPT3的用户，包括生成任务、问答任务、头脑风暴、对话等等。
    - 基于reward模型，使用PPO算法多模型进行训练

    ![](https://secure2.wostatic.cn/static/7BKU69oKMUpfFyWmaBdVcF/image.png?auth_key=1759401135-tmauKX1gFai6Sf6R8uKgSd-0-404c1e53c54d318a8aff30146659faa0)

    总的来说还是很复杂的，需要很多Reference model作为约束，需要Reward Model预测分数，还需要Critic预测残缺句子的return。每个模型都是一个transformer。

#### DPO算法

把Reward的训练和RLHF合并到一起了。缺陷是，reward model可以泛化到更多数据上，但直接训练模型就不行了。与RLHF相比，模型可以自己生成更多数据让reward model去打分，从而泛化性更好。

这篇文章讲得很好：[https://zhuanlan.zhihu.com/p/11913305485](https://zhuanlan.zhihu.com/p/11913305485)

#### GRPO算法

DeepSeek提出来的简化PPO的算法，让大模型根据prompt生成一组输出来计算loss。

## 模型

- RoPE
- SwiGLU
- preNorm
- MoE
    - DeepSeek v3：共享专家，一个简单的MoE负载均衡策略

除了RoPE和MoE能明显提升模型能力，SwiGLU和preNorm感觉都是为了训练稳定。

## 推理加速

- Prefilling
    - 需要高算力
- Decoding
    - 需要高显存
    - KVCache
        - GQA
        - MLA：
            - 原本KV的shape都是(B, H, N, C)，拼到一起是(B,H,N,2C)，把它们reshap成(B,N,2HC)，然后用一个线性层降维成(B,N,d)，其中d是降维后的维度。由于采用了RoPE，这里的K是旋转过的
            - query也会压缩

PD分离。

- MoE部署

## 减轻幻觉

大模型在预训练阶段构建对互联网数据的模糊回忆。对于一些重要的数据，比如wiki，可以重复很多次让模型更清楚。

### 数据层面

- 构建post training的数据，提供上下文，问一些问题，如果这个答案在上下文中就说知道，不在就说不知道
- 让另一个大模型获取上下文，check要训练的大模型是否知道答案。通过这种方式构建大模型不知道的内容的数据集。

一个感受是，transformer真的足够强大到可以表征各种内容，只是我们还没挖掘出来。

### 引入工具

让大模型联网。

![](https://secure2.wostatic.cn/static/9ArHWBoobTQn7VooKe2gkX/image.png?auth_key=1759401135-v61NuAdAqYRjfDSM5FRrVG-0-fc3c877656819bcc98a39b79ccb3391b)

还是通过构建数据集，让大模型学会使用工具。

### 解决计算问题

一个解决计算问题的方法是，一是使用更多的token，让模型学习计算的过程；二是让模型编写代码，并且这个代码是可以执行的，这样结果几乎总是对的。

### 数数问题

模型并不善于数数。

## 推理模型

- RFT
- DeepSeek R1 Zero：纯强化学习训练
- DeepSeek R1：
    - 用几千条推理数据进行微调
    - 然后用DeepSeek R1 Zero相同的方法进行训练
    - 语言混杂的问题，在强化学习训练期间我们引入了**语言一致性奖励**，其计算方式为 CoT 中目标语言词汇所占的比例。

[https://github.com/sail-sg/understand-r1-zero/tree/main](https://github.com/sail-sg/understand-r1-zero/tree/main)。

## 大模型评估

- AGI Eval
- Chatbot Arena
- 推理benchmark
    - 数学：
        - AIME

## 大模型安全

- 防御方法
    - Safety Prompt
- 攻击方法

LLM本身可以判断输入的句子是否有害，体现在有害和无害的特征差异比较大。加上safety prompt之后，模型会倾向于拒绝回答很多问题。

## 多模态大模型

一些研究表明大模型的训练语料里加上代码、弱智吧的内容会提高模型的表现。这个很有意思，说明模型有能力学习到复杂的推理逻辑，只是大模型没法主动思考，导致无法自身去进行这种练习。

