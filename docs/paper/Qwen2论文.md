# Qwen2 paper



huggingface里的源代码。 https://github.com/huggingface/transformers/tree/main/src/transformers/models



## intro 简介

Qwen2 使用了transformer架构，用next-token prediction进行预测。
Qwen2发行版包括0.5b 、1.5b 、7b 、72b的dense model，以及57b的MOE model
pre-training阶段，用了7万亿的token
post-training阶段，使用了 supervised fine-tuning 和 direct preference optimization

## Tokenizer 分词器

使用了byte-level byte-pair encoding

所有的模型都使用了相同的词汇库，词汇库包含151643个常规词和3个control token。

## architecture 架构

基于transformer架构，
具体而言，包括
1) dense language mode of 4 scales
2) a mixture of experts model


### Dense model
Dense model即多层transformer层，每一层都是由注意力机制和前馈全神经网络FNNs组成。

注意：
**Grouped Query Attention**
**Dual Chunk Attention with YARN**

1. 使用Grouped Query Attention  代替了 multi-head attention（MHA）。
    GQA 优化推理过程中 KV 缓存的使用，显着提高吞吐量。

2. Dual Chunk Attention将长序列分割成可管理长度的块。

   如果输入可以在一个块中处理，DCA 会产生与原始注意力相同的结果。否则，**DCA 有助于有效捕获块内和块间标记之间的相对位置信息**，从而提高长上下文性能。此外，我们还使用 **YARN** 来重新调整注意力权重，以获得更好的长度外推。

此外，我们遵循 Qwen，使用 **SwiGLU** 进行激活，使用旋转位置嵌入 (**RoPE**) 进行位置嵌入，**QKV 偏差** 进行注意力，**RMSNorm** 和 为了训练稳定性的**预归一化**。

### MOE model

> MoE model和Dense model的区别在于： 
>
> 与原始的前馈全连接网络FFN不同在于，**MoE FFN由n个独立的FFNs组成，每一个FFN充当一个专家。**

在moe中，

对于每一个输入的token而言，

先由 一个**门控网络** 输出token与各个专家的适配概率 p，
![image-20241030155013648](../picture.asset/image-20241030155013648.png)

接着，再由**指定的专家**进行计算。  

![image-20241030155021945](../picture.asset/image-20241030155021945.png)

> 将Dense model转换为 MoE model的简单直接的策略是：
>
> 将每个专家的参数设置为等于原始密集模型中单个 FFN 的参数。

**专家粒度：**本模型使用**细粒度的专家**，创建较小规模的专家，同时激活更多数量的专家。给定相同数量的专家参数和激活参数，细粒度专家提供更丰富的专家组合集。

**专家路由：** **在 MoE 层内集成共享专家和特定于路由的专家**已成为一个显着趋势。我们采用这种方法，因为它有助于共享专家在各种任务中的应用，同时保留其他专家在特定路由场景中选择性使用。

**专家初始化：**我们利用密集模型的权重，以与升级改造类似的方式初始化专家。

给定指定的专家中间大小 hE、专家数量 n 和原始 FFN 中间大小 hFFN，FFN 被复制 ⌈n×hE/hFFN⌉ 次。为了促进每个 FFN 副本内的多样性，参数沿着中间维度进行shuffle改组。这保证了每个细粒度专家即使在不同的 FFN 副本中也能展现出独特的特征。随后，从 FFN 副本中提取这些专家，并丢弃剩余的维度。**对于每个细粒度专家，其 50% 的参数会被随机重新初始化。**



![image-20241030155504229](../picture.asset/image-20241030155504229.png)

## Pre-Training 预训练

在 Qwen2 的预训练中，我们的工作重点是**完善数据集**和**研究有效处理扩展上下文长度的方法**。



### pre-training data 预训练数据



预训练中开发了一个新的、**高质量的、大规模的、多语言的**数据集。关键点如下：

**数据集质量**：**数据过滤**使用了额外的**启发式算法**和基于**模型**的方法，例如使用了Qwen来过滤低质量数据。**数据合成**也利用了Qwen。

**数据拓展：**我们收集了大量的高质量代码、数学和多语言数据，增强了模型在各自领域的能力。这个新数据集支持大约 30 种语言。

**数据分布：**为了确保模型学习类似于人类学习的分布，我们在缩小模型上进行实验，以**优化来自不同来源和领域的数据的混合**。



注意：Qwen2 预训练过程中集成了**高质量的多任务指令数据**，以增强**情境学习(in-context)**和**指令跟踪(instruction-follow)**能力。



### long context



为了增强上下文处理能力，预训练的最后阶段将上下文长度**从 4,096 个token增加到 32,768 个token**。这种扩展还伴随着**大量高质量、长数据**的引入。结合这些增强功能，我们将 **RoPE 的基频从 10,000 修改为 1,000,000**，以优化长上下文场景中的性能

为了充分利用模型的长度外推潜力，我们采用了 **YARN 机制**和 **Dual Chunk Attention 机制**。这些策略使模型能够处理多达 131,072 个标记的序列，同时保持高性能。



## Post-Training 后训练

与严重依赖广泛的人类监督的传统方法不同，我们的方法侧重于以最少的人类注释进行可扩展的对齐

具体来说，我们**研究了获取监督微调（SFT）和在人类反馈中强化学习（RLHF）的高质量演示和偏好数据的方法**，旨在最大限度地减少人工标记的需求，同时最大限度地提高数据的质量和可靠性。



### post-training data

后训练数据主要由**两部分组成**：

​	演示数据 D = {(xi, yi)} 

​	偏好数据 P = {(xi, yi+ , yi- )}，

​	其中 xi 代表指令，yi 代表满意的响应，yi+ 和 yi- 是对 xi 的两个响应，其中 yi+ 是比 yi- 更受偏好的选择。

​	集合D用于SFT，而P用于RLHF。



训练数据的构建需要两个步骤：协作数据注释和自动数据合成。

首先，我们从大规模指令语料库中提取数据本体，从而产生广泛且多样化的高质量指令。这些指令经过系统增强，变得更加复杂。通过人工注释，我们获得目标响应 yi 及其正负对应项 (yi+ , yi- )。

随后，采用各种**自动对齐策略**来合成大量跨代码、数学、指令遵循、创建、角色扮演和安全领域的人工注释数据。



关于协作数据注释：

​	自动本体提取：该过程从应用 **InsTag**（一种开放集细粒度标记器）开始，从大规模指令数据集中提取底层本体。随后的**人工精炼**保证了提取本体的准确性。

​	指令选择：每个带有标签注释的指令都会评估**标签多样性、语义丰富性、复杂性和意图完整性**。根据这些标准，我们选择了一组有代表性的指令

​	指令进化：为了丰富指令数据集，采用了自进化策略，**prompt Qwen 模型向现有指令添加约束或要求**，从而增加其复杂性并确保内部的各种难度级别。

​	人工注释： 使用**不同的生成策略和不同尺度的 Qwen 模型可以获得对指令的多个响应**。**注释者根据自己的偏好对这些响应进行排名**，确保最佳响应符合既定标准，**从而生成演示数据和偏好数据**。



关于自动数据合成：

​	拒绝采样：对于**具有明确最终答案的数学或类似任务**，应用拒绝采样来提高解决方案的质量。大型语言模型 (LLM) 的任务是为每条指令生成多个响应，即**推理路径**。**得出准确结论并被模型认为合理的路径被保留，作为示范数据。偏好数据是通过对比正确和不正确的路径来生成的。**

​	执行反馈：对于**编码任务**，llm用于**生成解决方案和相关的测试用例**。这些解决方案的功效是通过针对测试用例进行编译和执行来评估的，从而创建演示和偏好数据。

​			该方法也适用于评估**指令遵循**情况。对于每条具有约束（例如长度限制）的指令，llm的任务是生成一个 Python 验证函数，以确保响应符合指令要求。

​	数据再利用：对于未经专门培训的注释者来说，在**文学写作任务**中创建熟练的响应是一项挑战。为了解决这个问题，我们汇总了公共领域的高质量文学作品，并聘请llm来制定不同详细程度的说明。这些说明与原始作品配对，作为演示数据。例如，为了具有生动且引人入胜的角色扮演数据，我们从维基百科等知识库中获取详细的角色档案，并指导llm生成相应的指令和响应）。这个过程类似于阅读理解任务，可确保保持角色个人资料的完整性。

​	宪法反馈：宪法人工智能（Constitutional AI）指的是引导大型语言模型（LLMs）基于预定义的原则集生成回应的过程。为了确保遵守安全和价值观等指导方针，编制了一个宪法数据集。该数据集详细说明了应遵循的原则和应避免的原则。它被用来指导LLMs生成要么与这些指导方针一致、要么偏离这些指导方针的回应，作为演示和偏好数据的参考。

​	

### SFT

我们收集了一个广泛的教学数据集，包含超过 500,000 个示例，涵盖指令遵循、编码、数学、逻辑推理、角色扮演、多语言和安全等技能。

我们的模型**2个epoch**进行了微调，**序列长度为 32,768 个token**。为了优化学习，**学习率从 7 × 10−6 逐渐降低到 7 × 10−7**。

为了解决过度拟合问题，我们应用了 **0.1 的权重衰减**，并**将梯度限制为最大值 1.0**。



### RLHF

我们的 RLHF 训练制度包括两个连续阶段：offline and online training。

在离线训练阶段，我们使用预处理的偏好数据集 P 通过直接偏好优化（**DPO）来最大化 yi+ 和 yi- 之间的可能性差异**。

在在线训练阶段，模型利用奖励模型进行即时反馈，实时迭代地完善其性能。具体来说，**我们从当前策略模型中采样多个响应，奖励模型选择最喜欢和最不喜欢的响应，形成用于每个情节中的 DPO 的偏好对**。此外，我们采用在线合并优化器（Online Merging Optimizer）来减轻对齐税，即与将模型生成与人类偏好对齐相关的性能下降。



## Evaluation评估

......





----

