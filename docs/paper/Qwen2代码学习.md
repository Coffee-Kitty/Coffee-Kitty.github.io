# Qwen2 代码简学

[重要参考文献 Datawhale](https://github.com/datawhalechina/tiny-universe/tree/main/content/Qwen-blog)



Qwen2架构图

<img src="../Qwen2%E4%BB%A3%E7%A0%81%E5%AD%A6%E4%B9%A0/image-20241029203411240.png" alt="image-20241029203411240" style="zoom: 67%;" />

如图，对于输入的一段文本，比如”hello“， 

​	首先经过 Tokenizer 变成 词表里对应的数值。

​	然后经过 Embedding 得到高维向量，

​	接着经过 多个 Decoder layer，

​	最后 RMSNorm，线性层 然后得到最终输出。







## RMSNorm



