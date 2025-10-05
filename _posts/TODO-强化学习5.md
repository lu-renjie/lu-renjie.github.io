---
title: 【强化学习】SAC、Distributional RL
tags: 强化学习 深度学习 笔记
---

## Soft Actor-Critic

强化学习几个实用的算法是PPO、TD3和SAC，其中PPO和TD3前面已经提及，这里来介绍[Soft Actor-Critic（2018，SAC）](https://arxiv.org/pdf/1801.01290)。

### Entropy-regularized RL

SAC的基本思路是在目标函数里面加一个关于策略的熵 $\mathcal H(\pi(\cdot\|s))$，不仅仅最大化reward，也最大化策略的熵。因为熵衡量一个概率分布的不确定性，越不确定就越接近均匀分布，所以最大化熵能够让策略更加“随机”，进而加强探索。引入策略的熵后，我们可可以把熵也当做reward和原本的reward加起来，这种情况Q函数和V函数的关系变为：

$$
\begin{aligned}
V(s) &= E_{a\sim\pi(\cdot|s)}\left[  Q(s,a) \right] + \alpha\mathcal H(\pi(\cdot|s))
\\
&= E_{a\sim\pi(\cdot|s)}\left[  Q(s,a)-\alpha\ln \pi(a|s)  \right]
\end{aligned}
$$

其中 $\alpha$ 是超参数，用来控制熵的权重。进一步可以得到引入了熵的Bellman最优方程：

$$
Q(s,a) = E_{s'\sim\pi(\cdot|s)}\left[
    r + \gamma\ \left(\max_{a'} Q(s', a')-\alpha\ln\pi(a'|s')\right)
\right]
$$

### SAC算法

SAC整体和TD3比较类似，主要是用来解决连续动作空间问题的，在TD3的基础上引入了最大化熵的目标，并去掉了Target Policy Smoothing的操作，因为引入熵本身就能使策略更加随机，此时Target Policy Smoothing就没有必要了。

SAC因为引入了随机性，不再是确定性策略，所以还需要重新参数化一遍，一般使用网络预测均值和方差。

SAC 算法流程：
1. 初始化 $\pi_{\theta},Q_{w_1},Q_{w_2}$，并复制一份得到目标网络  $Q_{w_1'},Q_{w_2'}$，<span style="color:red;">注意策略网络不需要复制</span>
2. 迭代 $N$ 次：
    - 基于当前策略走一步得到$s, a, r, s'$
    - 使用 $s, a, r, s'$ 更新replay buffer
    - 迭代 $d$ 次
        - 随机采样一个batch训练 $Q_{w_1}$ 和 $Q_{w_2}$，训练目标是：

            $$
            \text{target} = r + \gamma\min_{i=1,2}\{Q_{w_i'}(s',a')-\alpha\ln\pi_{\theta}(a'|s')\}
            \notag$$

            注意这里用的是新采样的动作 $a'\sim\pi_{\theta}(\cdot\|s')$，不是replay buffer里的动作。
    
        - 随机采样一个batch训练策略：

            $$
            -\frac1B\sum_{i=1}^B \left[\min_{i=1,2} Q_{w_i}(s, a') - \alpha\ln\pi_{\theta}(a'|s')\right]
            $$

        - 更新目标网络

            $$
            \begin{aligned}
            w_1' &\leftarrow \tau w_1+(1-\tau)w_1
            \\
            w_2' &\leftarrow \tau w_2+(1-\tau)w_2
            \end{aligned}
            \notag$$

SAC对reward的要求更高。

### 自动调alpha

### SAC对比TD3

PPO比较好调参，SAC和TD3对参数很敏感，不好调参。


## TODO
1. TD3和PPO到底是sample多少步更新一次？
1. 没理解为什么Q-Learning会高估


## Distributional RL

考虑值函数的不确定性。

### DSAC

### DSACv2（DSAC-T）



## 单步决策问题（utility theory）


## safe exploration
