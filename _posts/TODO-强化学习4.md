---
title: 【强化学习】深度强化学习
tags: 强化学习 深度学习 笔记
published: true
---

## Exploration v.s. Exploitation

策略影响收集到的数据，数据又会影响策略。

## 稀疏奖励

## AlphaZero

Model based RL.


## 确定性策略梯度

### DDPG

之前介绍的方法全部都只能适用于离散动作空间的问题，[DDPG(2015)](https://arxiv.org/abs/1509.02971)提出了一个适用于**确定性连续动作空间**的强化学习算法，确定性指策略没有随机性，也就是连续值动作没有随机性。例如我们如果想训练一个模型控制赛车游戏的转弯，确定性策略直接根据当前状态预测一个转角。DDPG是[DPG](https://inria.hal.science/file/index/docid/938992/filename/dpg-icml2014.pdf)的Deep版本，结合了DPG和DQN。它的目标函数为：

$$
\mathcal L(\theta)=E_{s\sim d^{\pi_{\theta}}}[Q^{\pi_{\theta}}(s,\pi_{\theta}(s))]
$$

$\pi_{\theta}(s)$ 就是确定性策略。相应的也有确定性策略梯度定理：

基于这个定理，结合前面介绍的DQN，很容易理解DDPG算法：

- 初始化 $\pi_{\theta},Q_{w}$，拷贝 $\pi_{\theta},Q_{w}$ 得到目标网络 $\pi_{\theta'},Q_{w'}$
- 迭代 $N$ 次：
    - 基于当前策略走一步得到$s, a, r, s'$，
    - 使用 $s, a, r, s'$ 更新replay buffer
    - 从replay buffer中采样一个batch的样本
        - 基于DQN损失函数对 $Q_w$ 进行训练，其中 $target=r+\gamma Q_{w'}(s',\pi_{\theta'}(s'))$

        - 使用下面的损失函数对 $\pi_{\theta}$ 进行训练（梯度穿过 $Q$ 反向传播到 $a$，进一步到 $\pi_{\theta}$）

            $$
            \mathcal L(\theta) = -\frac1B\sum_{i=1}^B Q_{w}(s_i, a)
            $$


    - 使用移动加权平均更新目标网络的参数（通常$\tau=0.999$）：

        $$
        \begin{aligned}
        \theta'=\tau\theta'+(1-\tau)\theta\\
        w'=\tau w+(1-\tau)w 
        \end{aligned}
        \notag$$

由于DDPG采用确定性策略，没有探索环境的能力。为了能支持探索，需要给行为加上高斯噪声，即 $a=\pi_{\theta}(s) + \epsilon$，其中 $\epsilon$ 是高斯噪声，噪声的方差需要随着迭代次数的增加而减小。DDPG的一个重要改进版本是[TD3(2018)](https://arxiv.org/pdf/1802.09477)。

## Safe Exploration

Safe Exploration in Continuous Action Spaces
