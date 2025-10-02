---
title: 【大模型】Agent
tags: 深度学习 大模型 笔记
---


可以用于智能客服、完成一些自动化工作、智能化生活、辅助搜广推系统、帮助写代码等等。

### Prompt

- CoT技巧，输出rationale
- guided_decoding

### Agent

Agent就是让大模型使用工具，使用工具的方式是让大模型输出一些json文件，说明如何使用工具。

- ReAct：Reason-Act-Observe的循环来解决问题，可以达到比CoT更好的效果。现在用来构建Agent
- Plan-and-Execute：针对长期任务，分解出plan和execute2个模块。Plan可以进行high-level的规划，因此可以支持更长的任务，但是效果也难说。
- Multi-Agent：由不同的agent负责任务创建、优先级排序、长短期记忆管理

### RAG

- Naive RAG：引入检索
    - 构建知识库，类似于搜索引擎，可以引入embedding查询实现模糊检索
    - 检索出来的文本以一定的格式（模板）拼接在一起，输入到大模型

    检索这个东西可以细分啊，比如：

    - 相关内容检索（类似于指环王的电影）
    - 反向理解（不要犯罪主题的电影），完全是两个东西。
- Advanced RAG
    - 分为pre-retrieval和post-retrieval
    - pre-retrieval类似于召回，用来获取足够多的相关内容
        - 比如会通过“query改写”等操作得到更多结果
    - post-retrieval
        - 排序
        - 上下文压缩
- Modular RAG，把RAG链路各个节点之间定义协议，实现模块化
- Graph RAG，结合知识图谱和LLM
    - 从检索词中分析出实体和关系来进行检索
    - 向量库+图数据库
- Agentic RAG，让Agent负责检索而不是LLM，使得可以主动规划检索策略、自我验证等
    - 将问题拆解成task
    - 执行task
        - 执行过程中LLM可以检索、使用工具，产出结果
        - 另一个LLM用来判断task是否完成
- workflow，以节点编辑器的形式自定义agent

这里RAG作为一个系统，提取embedding等操作同时通过远程API的形式实现的，langChain提供了文本的分块，访问一些常见的比如OpenAI的embedding API，还有对向量数据库如faiss的访问。