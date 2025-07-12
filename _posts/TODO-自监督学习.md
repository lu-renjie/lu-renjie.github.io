---
title: 【深度学习】自监督学习
tags: 深度学习 笔记
published: false
---


## Self-Supervised Learning

### Language
#### BERT
#### WhiteningBERT
#### SimCSE

### Image

#### Contrastive Learning
SimCLR
MoCov1,MoCov2

#### SwAV：聚类

#### BYOL：同一个图片的不同特征相互预测
    - 后来有人发现prediction hea\text d需要加BN，不加就会得到平凡解
    - 原因在于BN泄露了一个Batch的样本的信息，BYOL本质上是让模型学习样本与batch平均图片的差异
    - 后来BYOL的作者研究发现去掉所有BN，BYOL和SimCLR都不行了，但是加上合适的初始化之后，模型还是能很好的训练，所以合适的初始化就能很好的防止模型学到平凡解
#### SimSiam
    - SimSiam在前面论文的基础上简化了模型，去掉了负样本、大Batch、动量编码器，用类似于BYOL的方法还是可以训好
    - SimSiam认为BYOL有效的原因在于STOP-Gradient，就是在相互预测特征的时候作为target的那边要detach
    - 并给出了EM算法的解释
    - 最后的结果还是比BYOL差一些
#### Barlow Twins
#### BYOL
#### iBOT

#### SSL by Reconstruction
BEIT
MAE

#### DINOv1,DINOv2
DINOv1把MoCo的方法迁移到ViT，发现ViT的Attention很有可解释性，可以直接做分割。

DINOv2希望能够得到一个不需要微调的预训练模型，能够为各种视觉任务，无论是图片级别，还是像素级别都提供足够好的特征，也就是训练得到一个基础模型。

- \text dINOv2主要贡献在于：
    - 提出了一个筛选数据的方法，得到了高质量的142M图片数据集（LV\text d-142M）
        - 基于现有的一些公开数据集，对互联网上爬的1.2B张图片做筛选
    - 在iBOT的基础上做了些细节和实现上的改进，这些改进还是不错的

### 视觉语言自监督
#### CLIP
对比学习。

#### SigLIP
