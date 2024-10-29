
# Qwen2

## intro

Qwen2 使用了transformer架构，用next-token prediction进行预测。
Qwen2发行版包括0.5b 、1.5b 、7b 、72b的dense model，以及57b的MOE model
pre-training阶段，用了7万亿的token
post-training阶段，使用了 supervised fine-tuning 和 direct preference optimization

## Tokenizer分词器

使用了byte-level byte-pair encoding

所有的模型都使用了相同的词汇库，词汇库包含151643个常规词和3个control token。

## architecture架构

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

### MOE

与原始的前馈全连接网络FFN不同在于，**MoE FFN由n个独立的FFNs组成，每一个FFN充当一个专家。**

<font color='red'>123</font>