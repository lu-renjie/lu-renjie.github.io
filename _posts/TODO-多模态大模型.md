## 多模态大模型

一些研究表明大模型的训练语料里加上代码、弱智吧的内容会提高模型的表现。这个很有意思，说明模型有能力学习到复杂的推理逻辑，只是大模型没法主动思考，导致无法自身去进行这种练习。


- LXMERT，UNITER，早期论文，还在用检测网络，各种花里胡哨的task
- 2021 VILT：经典论文，图片和文本concat一起训，效果就很好，就是收敛慢。
- 2021 ALBEF：经典论文，2个模态的encoder+1个cross attention融合模块，核心是引入CLS的预对齐
- 2021.02 CLIP：简单粗暴的对比学习
- 2022 VLMo：指出FFN层可以替换
- 2022 Coca：和ALBEF结构类似，但是训练任务从MLM改成了文本生成以支持更多任务
- 2022.01 BLIP：3个loss，caption+contrastive+itm，外加bootstrap的数据
- 2022 BEiTv3：统一的完形填空训练任务

    BEiTv1就是引入dVAE进行MIM的训练，BEiTv2引入了teacher的监督，BEiT引入了多模态，统一用完形填空训练（还包括MultiWay Transformer）。

    - BEIT v3除了BERT外还有别的特殊的设计吗
    - V3和V2的 embbeding有什么不同
- 2023.03 SigLIP：用sigmoid计算loss的CLIP

后面是大模型[https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)

- 2022.04 Flamingo：多模态领域最早期的尝试做类似于GPT的few-shot的论文。弄了一个类似于QFormer的东西（感知重采样层），以及门控机制。以如今的视角来看没有必要。
- 2023 BLIP2：QFormer+大模型，感觉从这里开始就明显的是Adapter的时代了。
- LLaVA
    - 模型没啥特点，图片用ViT提特征之后加linear层输入到LLM，两阶段训练。
    - LLaVa Next：图片分块+Resize整体获取局部整体特征，OCR改进模型
- Intern-VL：
- QWen-VL：
- DeepSeek-VL
- 多模态大模型总结
    - 模态的编码，LLM的backbone，以及多模态的生成
    - Input Projector
