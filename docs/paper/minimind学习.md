基本大模型知识参考：https://llmbook-zh.github.io/LLMBook.pdf

基本代码库知识请查看https://transformers.run/

实践项目：minimind  https://github.com/jingyaogong/minimind

# minimind学习

## minimind简介：

​	1）一个 从0开始，训练（最小型号仅为26.88M即0.02B大小）的llm 的项目

​	2）包括数据处理、预训练、指令微调、人类对齐的完整流程

​	3）在 Ceval上进行模型评测

​	4）实现chat接口，便于集成到第三方chatUI使用



> 基本环境搭建：
>
> 
>
> fork后使用git下载到本地
>
> git clone git@github.com:Coffee-Kitty/minimind.git
>
> conda create -n minimind 
>
> conda activate minimind
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
>
> 
>
> 再测试torch是否可用cuda
>
> import torch
> print(torch.cuda.is_available())



## Tokenizer

tokenizer是数据预处理的一个关键步骤，

旨在`将原始文本转换为模型可以识别和建模的词元序列`

基于transformer的语言模型主要使用子词分词器算法，包括**BPE分词**、**WordPiece分词**，**Unigram分词**三种常见方法。

### BPE

**BPE算法从一组基本符号（例如字母）开始，迭代的寻找语料库中两个相邻词元，并将它们替换为新的词源，这一过程称为合并。合并的选择标准是：每次迭代中，最频繁出现的一对词元会被合并为新的词元。迭代过程将在达到预定义的词表大小之后停止。**

例子：

>**(0)** 假设语料中包含了五个英文单词:  “loop”,“pool”,“loot”,“tool”,“loots”  ,在这种情况下,BPE 假设的初始词汇表即为:  [“l”, “o”, “p”, “t”, “s”]  
>
>在实践中,基础词汇表可以包含所有 ASCII 字符,也可能包含一些 Unicode 字符  (比如中文的汉字)。如果正在进行分词的文本中包含了训练语料库中没有的字符,则该字符将被转换为未知词元(如 “<UNK>”)。 
>
>**(1)** 假设单词在语料库中的频率如下:  (“loop”,15),(“pool”,10),(“loot”,10),(“tool”,5),(“loots”,8)  其中,出现频率最高的是 “oo”,出现了 48 次,因此,学习到的第一条合并规则  是(“o”, “o”)→ “oo”,这意味着“oo”将被添加到词汇表中,并且应用这一  合并规则到语料库的所有词汇。
>
>在这一阶段结束时,词汇和语料库如下所示:  词汇:[“l”, “o”, “p”, “t”, “s”, “oo”]  语料库:(“l”“oo”“p”,15),(“p”“oo”“l”,10),(“l”“oo”“t”,  10),(“t”“oo”“l”,5),(“l”“oo”“t”“s”,8)  
>
>**(2)** 此时,出现频率最高的配对是(“l”,“oo”),在语料库中出现了 33 次,因此学习  到的第二条合并规则是(“l”,“oo”)→ “loo”。将其添加到词汇表中并应用到所  有现有的单词,可以得到:  词汇:[“l”, “o”, “p”, “t”, “s”, “oo”, “loo”]  语料库:(“loo”“p”,15),(“p”“oo”“l”,10),(“loo”“t”,10),(“t”“oo”“l”,  5),(“loo”“t”“s”,8)
>
>



**字节级别的 BPE(Byte-level BPE, B-BPE)**是 BPE 算法的一种拓展。它将字节  视为合并操作的基本符号,从而可以实现更细粒度的分割,且解决了未登录词问题。

具体  来说,如果将所有 Unicode 字符都视为基本字符,那么包含所有可能基本字符的基本词表会非常庞大(例如将中文的每个汉字当作一个基本字符)。

而将字节作为基本词表可以设置基本词库的大小为 256,同时确保每个基本字符都包含在词汇中。  

**例如,GPT-2 的词表大小为 50,257 ,包括 256 个字节的基本词元、一个特殊的文  末词元以及通过 50,000 次合并学习到的词元。**

### tokenizer数据集

tokenizer的构建方式有两种，一个是自己构造词表训练一个，一个是使用开源模型训练好的分词器。这里选择自己训练。下载数据集到./dataset目录下。

 [tokenizer数据集：HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 

copy link如下图所示。

![](../picture.asset/image-20241112161411212.png)

```bash
wget https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/tokenizer_train.jsonl
```



总共有600000条数据，数据内容大致为:

![](../picture.asset/image-20241114135232916.png)



下面是tokenizer的训练代码

```python
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os

random.seed(42)

def train_tokenizer():
    # 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    data_path = './dataset/tokenizer_train.jsonl'

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token  unknown、begin、end
    special_tokens = ["<unk>", "<s>", "</s>"]

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=6400, #需要的词表大小为6400
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() #字节级别、256 个字节的基本词元
    )

    # 读取文本数据  获得一个生成器函数
    texts = read_texts_from_jsonl(data_path)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # 保存tokenizer
    tokenizer_dir = "./model/minimind_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./model/minimind_tokenizer")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")

```

![image-20241114140110093](../picture.asset/image-20241114140110093.png)

最终获得了一些文件，

![image-20241114140340898](../picture.asset/image-20241114140340898.png)

vocab.json为词典文件，存放的就是词表。

![vocab.json](../picture.asset/image-20241114140241620.png)

merges.txt则存放了合并规则。应该是按顺序依次合并。

<img src="../picture.asset/image-20241114140417756.png" alt="image-20241114140417756" style="zoom:50%;" />

这两个文件的内容，在tokenizer.json文件中都有包含。此外就是tokenizer_config.json记录了配置文件。

接下来简单测试一下训练的tokenizer。

已经保存了分词器为 `tokenizer.json` 文件，接下来可以使用 `AutoTokenizer` 加载它。

```python
def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/minimind_tokenizer")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids)
    print('decoder和原始文本是否一致：', response == new_prompt)
```

![image-20241114141018895](../picture.asset/image-20241114141018895.png)

## pretrain

### pretrain数据







```bash
wget https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_data.csv
```

```python
import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os

# 设置环境变量以避免在多线程环境下tokenizer可能遇到的问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        super().__init__()  # 调用父类的初始化方法
        self.df = df  # 存储传入的DataFrame对象，包含文本数据
        self.tokenizer = tokenizer  # 存储传入的tokenizer对象，用于文本编码
        self.max_length = max_length  # 设置最大序列长度，默认为512
        self.padding = 0  # 设置padding的值，这里假设为0，但实际应为tokenizer的padding标记ID 对应于token <unk>

    def __len__(self):
        # 返回数据集中样本的数量，即DataFrame的行数
        return self.df.shape[0]

    def __getitem__(self, index: int):
        # 根据索引index获取单个样本
        sample = self.df.iloc[index]  # 从DataFrame中获取第index行的数据
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"  # 将文本前后分别加上开始和结束标记
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]  # 使用tokenizer编码文本，并截断到最大长度
        text_len = len(input_id)  # 计算编码后文本的实际长度
        padding_len = self.max_length - text_len  # 计算需要padding的长度
        input_id = input_id + [self.padding] * padding_len  # 对编码后的文本进行padding
        # 创建loss_mask，用于在训练时忽略padding部分的损失计算
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)  # 将input_id转换为NumPy数组
        X = np.array(input_id[:-1]).astype(np.int64)  # 提取除了最后一个标记之外的所有标记作为输入X
        Y = np.array(input_id[1:]).astype(np.int64)  # 提取从第二个标记到最后一个标记作为目标Y
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)  # 将loss_mask转换为NumPy数组，并去掉第一个元素
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)  # 将NumPy数组转换为PyTorch张量并返回
```

需要注意的是，这里是并行输入的，所以X是 0-len-2  而作为预测的Y是 1-len-1

<font color='green'>创建loss_mask，用于在训练时忽略padding部分的损失计算</font>



#### 例子

这里抽出来几条数据来看，

这里的text为简单的给text加上了\<s> \</s>

![image-20241215181649283](../picture.asset/image-20241215181649283.png)

接下来做了被tokenize然后，根据做大长度做padding,

并且给出loss_mask，即padding的部分不去计算loss,

如下图，此例子就需要padding

![image-20241215181847421](../picture.asset/image-20241215181847421.png)

然后X,Y取得很神奇，

具体来看，如下图

![image-20241215182019355](../picture.asset/image-20241215182019355.png)

![image-20241215182031259](../picture.asset/image-20241215182031259.png)



然后我们进入train_epoch查看具体是怎么计算的loss，

```python
   for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
```



首先这里DataLoader批量加载了5条数据，

![image-20241215182254736](../picture.asset/image-20241215182254736.png)

这里511的长度与上面相符合，

然后进入model中去，loss在模型中被计算

```python
    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None,
                kv_cache=False, **keyargs):
        current_idx = 0
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']
        if 'current_idx' in keyargs:
            current_idx = int(keyargs['current_idx'])

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        pos_cis = self.pos_cis[current_idx:current_idx + seqlen]
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis, kv_cache)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                             ignore_index=0, reduction='none')
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT
```



![image-20241215183010479](../picture.asset/image-20241215183010479.png)



>  https://blog.csdn.net/wuliBob/article/details/104119616
>
> 对PyTorch中F.cross_entropy()的理解
>
> PyTorch提供了求[交叉熵](https://so.csdn.net/so/search?q=交叉熵&spm=1001.2101.3001.7020)的两个常用[函数](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782&utm_source=glcblog&spm=1001.2101.3001.7020)，一个是`F.cross_entropy()`，另一个是`F.nll_entropy()`，
>
> ![image-20241215194830960](../picture.asset/image-20241215194830960.png)
> ![image-20241215194922879](../picture.asset/image-20241215194922879.png)
>
> ![image-20241215194941378](../picture.asset/image-20241215194941378.png)
>
> 
>
> ![image-20241215195032018](../picture.asset/image-20241215195032018.png)
>
> 
>
> ![image-20241215195231883](../picture.asset/image-20241215195231883.png)
>
>
> ![image-20241215195249199](../picture.asset/image-20241215195249199.png)





此时，就可串联起来模型所有的输入输出了

我们给输入一句话，然后attention里做的mask！！！

```python
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
```

![image-20241215183503046](../picture.asset/image-20241215183503046.png)

也就是说这一句话，模型会经过atten_mask，自回归的一个一个的生成下一个token，

最终生成512个token，

然后这512个再跟  target（即target跟这个input就差了开头这一个字符）比较，计算交叉熵loss

也就是是符合预测下一个词元的，如下图

![image-20241215183830075](../picture.asset/image-20241215183830075.png)

但是输入真的就是一句话，对应的也真的就是 这一句话从第二个字开始。。。



### 模型

#### 配置

```python
from transformers import PretrainedConfig
from typing import List

# 定义一个继承自PretrainedConfig的配置类LMConfig
class LMConfig(PretrainedConfig):
    # 设置模型类型为"minimind"
    model_type = "minimind"

    # 初始化方法，定义模型的各种配置参数
    def __init__(
            self,
            dim: int = 512,  # 模型的隐藏维度，默认值为512
            n_layers: int = 8,  # 模型的层数，默认值为8
            n_heads: int = 16,  # 注意力头的数量，默认值为16
            n_kv_heads: int = 8,  # 键值注意力头的数量，默认值为8
            vocab_size: int = 6400,  # 词汇表大小，默认值为6400
            hidden_dim: int = None,  # 隐藏层维度，默认值为None
            multiple_of: int = 64,  # 隐藏层维度的倍数，默认值为64
            norm_eps: float = 1e-5,  # 归一化层的epsilon值，默认值为1e-5
            max_seq_len: int = 512,  # 最大序列长度，默认值为512
            dropout: float = 0.0,  # Dropout比率，默认值为0.0
            flash_attn: bool = True,  # 是否使用Flash Attention，默认值为True
            ####################################################
            # 下面是MOE（混合专家）模型的特定配置
            # 当use_moe为False时，以下配置无效
            ####################################################
            use_moe: bool = False,  # 是否使用MOE，默认值为False
            num_experts_per_tok: int = 2,  # 每个token选择的专家数量，默认值为2
            n_routed_experts: int = 4,  # 总的专家数量，默认值为4
            n_shared_experts: bool = True,  # 是否共享专家，默认值为True
            scoring_func: str = 'softmax',  # 评分函数，默认值为'softmax'
            aux_loss_alpha: float = 0.01,  # 辅助损失的alpha参数，默认值为0.01
            seq_aux: bool = True,  # 是否在序列级别上计算辅助损失，默认值为True
            norm_topk_prob: bool = True,  # 是否标准化top-k概率，默认值为True
            **kwargs,  # 其他关键字参数
    ):
        # 设置模型的隐藏维度
        self.dim = dim
        # 设置模型的层数
        self.n_layers = n_layers
        # 设置注意力头的数量
        self.n_heads = n_heads
        # 设置键值注意力头的数量
        self.n_kv_heads = n_kv_heads
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置隐藏层维度
        self.hidden_dim = hidden_dim
        # 设置隐藏层维度的倍数
        self.multiple_of = multiple_of
        # 设置归一化层的epsilon值
        self.norm_eps = norm_eps
        # 设置最大序列长度
        self.max_seq_len = max_seq_len
        # 设置Dropout比率
        self.dropout = dropout
        # 设置是否使用Flash Attention
        self.flash_attn = flash_attn
        ####################################################
        # 设置MOE（混合专家）模型的特定配置
        # 当use_moe为False时，以下配置无效
        ####################################################
        # 设置是否使用MOE
        self.use_moe = use_moe
        # 设置每个token选择的专家数量
        self.num_experts_per_tok = num_experts_per_tok
        # 设置总的专家数量
        self.n_routed_experts = n_routed_experts
        # 设置是否共享专家
        self.n_shared_experts = n_shared_experts
        # 设置评分函数
        self.scoring_func = scoring_func
        # 设置辅助损失的alpha参数
        self.aux_loss_alpha = aux_loss_alpha
        # 设置是否在序列级别上计算辅助损失
        self.seq_aux = seq_aux
        # 设置是否标准化top-k概率
        self.norm_topk_prob = norm_topk_prob
        # 调用父类的初始化方法，传递其他关键字参数
        super().__init__(**kwargs)
```



#### 模型:

```python
def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转位置嵌入（Positional Codes in Complex Space）。

    参数:
        dim (int): 嵌入维度。
        end (int): 序列的最大长度。
        theta (float): 缩放因子，默认为10000.0。

    返回:
        torch.Tensor: 形状为 (end, dim) 的复数张量，包含预计算的位置嵌入。
    """
    # 计算频率向量
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 创建时间步长向量
    t = torch.arange(end, device=freqs.device)
    
    # 计算外积以得到每个时间步长对应的所有频率
    freqs = torch.outer(t, freqs).float()
    
    # 使用极坐标形式创建复数张量
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    """
    将旋转位置嵌入应用到查询和键张量上。
    
    参数:
        xq (torch.Tensor): 查询张量，形状为 (batch_size, seq_len, n_heads, head_dim)。
        xk (torch.Tensor): 键张量，形状为 (batch_size, seq_len, n_heads, head_dim)。
        pos_cis (torch.Tensor): 位置码，形状为 (seq_len, head_dim)。

    返回:
        tuple: 应用旋转嵌入后修改的查询和键张量。
    """
    def unite_shape(pos_cis, x):
        """
        调整位置码的形状以匹配输入张量的维度。

        参数:
            pos_cis (torch.Tensor): 位置码，形状为 (seq_len, head_dim)。
            x (torch.Tensor): 输入张量，其形状用于调整 pos_cis。

        返回:
            torch.Tensor: 形状调整后的位置码。
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # 将查询和键的实部和虚部转换为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 匹配位置码的形状与查询和键的形状
    pos_cis = unite_shape(pos_cis, xq_)

    # 通过乘法应用旋转嵌入
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)

    # 返回修改后的查询和键，并恢复原始数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    沿头部维度重复键值对。

    参数:
        x (torch.Tensor): 形状为 (batch_size, seq_len, n_kv_heads, head_dim) 的张量。
        n_rep (int): 每个键值对的重复次数。

    返回:
        torch.Tensor: 形状为 (batch_size, seq_len, n_local_heads, head_dim) 的张量，
                      其中 n_local_heads = n_kv_heads * n_rep。
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
```



##### Transformer

```python
import torch
from torch import nn, Tensor
from typing import Optional
from transformers import PreTrainedModel
from .config import LMConfig  # 假设LMConfig定义在config模块中
from .layers import TransformerBlock, RMSNorm  # 假设TransformerBlock和RMSNorm定义在layers模块中
from .utils import precompute_pos_cis  # 假设precompute_pos_cis定义在utils模块中
from transformers.modeling_outputs import CausalLMOutputWithPast
import math

class Transformer(PreTrainedModel):
    config_class = LMConfig
    last_loss: Optional[torch.Tensor]  # 存储最近一次前向传播的损失值

    def __init__(self, params: LMConfig = None):
        """
        初始化Transformer模型。

        参数:
            params (LMConfig): 模型配置参数，默认为None。如果为None，则使用默认配置。
        """
        super().__init__(params)
        if not params:
            params = LMConfig()  # 如果没有提供配置，则使用默认配置
        self.params = params
        self.vocab_size = params.vocab_size  # 词汇表大小
        self.n_layers = params.n_layers  # 层数量

        # 创建词嵌入层
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        # 创建Dropout层
        self.dropout = nn.Dropout(params.dropout)
        # 创建多个TransformerBlock层
        self.layers = torch.nn.ModuleList([TransformerBlock(layer_id, params) for layer_id in range(self.n_layers)])
        # 创建归一化层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # 创建输出层，映射到词汇表大小
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        # 共享词嵌入层与输出层权重
        self.tok_embeddings.weight = self.output.weight
        # 预计算位置编码
        pos_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        # 注册位置编码作为非持久性缓冲区
        self.register_buffer("pos_cis", pos_cis, persistent=False)

        # 初始化模型权重
        self.apply(self._init_weights)

        # 对特定层的权重进行特殊初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        self.last_loss = None
        # 创建一个用于存储输出的对象
        self.OUT = CausalLMOutputWithPast()
        # 设置不拆分的模块名称列表
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        """
        初始化模型中的线性层和嵌入层的权重。

        参数:
            module: 要初始化的模块。
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None,
                kv_cache=False, **keyargs):
        """
        定义模型的前向传播过程。

        参数:
            tokens (torch.Tensor): 输入的token序列。
            targets (torch.Tensor): 目标token序列，用于计算损失。
            kv_cache (bool): 是否使用缓存的键值对来加速推理。
            **keyargs: 其他关键字参数。
        """
        # 处理输入参数
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']
        if 'current_idx' in keyargs:
            current_idx = int(keyargs['current_idx'])

        # 获取批次大小和序列长度
        _bsz, seqlen = tokens.shape
        # 将token转换为词嵌入
        h = self.tok_embeddings(tokens)
        # 应用Dropout
        h = self.dropout(h)
        # 获取当前位置编码
        pos_cis = self.pos_cis[current_idx:current_idx + seqlen]
        # 通过多层TransformerBlock传递数据
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis, kv_cache)

        # 归一化处理
        h = self.norm(h)

        # 计算损失或输出最终结果
        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            logits = self.output(h[:, [-1], :])  # 只输出最后一个token的结果
            self.last_loss = None

        # 存储输出
        self.OUT['logits'] = logits
        self.OUT['last_loss'] = self.last_loss
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, eos, max_new_tokens, temperature=0.7, top_k=8, stream=True, rp=1., kv_cache=True):
        """
        生成新的token序列。

        参数:
            idx (torch.Tensor): 当前token序列。
            eos (int): 结束符号的索引。
            max_new_tokens (int): 最大生成新token的数量。
            temperature (float): 温度参数，控制生成的多样性。
            top_k (int): 选择前top_k个最高概率的token。
            stream (bool): 是否以流式方式返回生成的token。
            rp (float): 重复惩罚因子。
            kv_cache (bool): 是否使用缓存。
        """
        index = idx.shape[1]
        init_inference = True
        while idx.shape[1] < max_new_tokens - 1:
            # 执行前向传播并获取结果
            if init_inference or not kv_cache:
                inference_res, init_inference = self(idx, kv_cache=kv_cache), False
            else:
                inference_res = self(idx[:, -1:], kv_cache=kv_cache, current_idx=idx.shape[1] - 1)

            logits = inference_res.logits
            logits = logits[:, -1, :]  # 取最后一个token的输出

            # 应用重复惩罚
            for token in set(idx.tolist()[0]):
                logits[:, token] /= rp

            # 根据温度参数调整logits，并可能应用top-k过滤
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)

            # 如果下一个token是结束符号，则停止生成
            if idx_next.item() == eos:
                break

            # 追加新生成的token到序列中
            idx = torch.cat((idx, idx_next), dim=1)
            if stream:
                yield idx[:, index:]  # 流式返回新生成的部分

        if not stream:
            yield idx[:, index:]  # 返回所有新生成的部分

    @torch.inference_mode()
    def eval_answer(self, idx):
        """
        在给定的上下文中评估答案。

        参数:
            idx (torch.Tensor): 输入的token序列。
        """
        # 如果输入序列超过最大长度，则截断
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
        inference_res = self(idx_cond)  # 执行前向传播
        logits = inference_res.logits
        logits = logits[:, -1, :]  # 取最后一个token的输出
        return logits
```

##### TransformerBlock

上面提及的transformerBlock类如下：

```python
import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LMConfig):
        """
        初始化一个TransformerBlock层。

        参数:
            layer_id (int): 当前层的ID。
            args (LMConfig): 模型配置参数。
        """
        super().__init__()
        self.n_heads = args.n_heads  # 注意头的数量
        self.dim = args.dim  # 模型的隐藏维度
        self.head_dim = args.dim // args.n_heads  # 每个注意头的维度
        self.attention = Attention(args)  # 初始化注意力机制

        self.layer_id = layer_id  # 当前层的ID
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 初始化注意力层的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 初始化前馈网络层的归一化层

        if args.use_moe:
            # 如果配置中启用了混合专家（MoE），则使用MOE前馈网络
            self.feed_forward = MOEFeedForward(args)
        else:
            # 否则，使用普通的前馈网络
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )

    def forward(self, x, pos_cis, kv_cache=False):
        """
        定义TransformerBlock层的前向传播过程。

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, dim]。
            pos_cis (torch.Tensor): 位置编码，形状为 [seq_len, head_dim]。
            kv_cache (bool): 是否使用缓存的键值对来加速推理。
        """
        # 注意力机制
        h = x + self.attention(self.attention_norm(x), pos_cis, kv_cache)
        # 前馈网络
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

##### Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    """
    	分组自注意力       n_heads / n_kv_heads 个组
    """
    def __init__(self, args: LMConfig):
        super().__init__()
        # 初始化关键参数
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        # 定义线性变换层
    	# 分组自注意力
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # 初始化缓存
        self.k_cache, self.v_cache = None, None
        
        # 定义dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        # 检查是否支持Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        # 创建掩码以防止未来信息泄露
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache=False):
        """
        前向传播函数，计算自注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。
            pos_cis (torch.Tensor): 预计算的位置嵌入，形状为 (seq_len, head_dim)。
            kv_cache (bool): 是否使用键值缓存，默认为False。

        返回:
            torch.Tensor: 自注意力机制的输出张量，形状为 (batch_size, seq_len, dim)。
        """
        bsz, seqlen, _ = x.shape

        # 计算查询、键和值
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 调整形状以便进行多头注意力计算
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 如果启用kv缓存，则在评估模式下使用缓存
        if kv_cache and self.eval():
            if seqlen == 1 and all(cache is not None for cache in (self.k_cache, self.v_cache)):
                xk = torch.cat((self.k_cache, xk), dim=1)
                xv = torch.cat((self.v_cache, xv), dim=1)
            self.k_cache, self.v_cache = xk, xv

        # 重复键和值以匹配本地头部数量
        xk = repeat_kv(xk, self.n_rep)  # 形状变为 (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # 形状变为 (bs, seqlen, n_local_heads, head_dim)

        # 转置以便进行点积注意力计算
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据条件选择使用Flash Attention或标准注意力机制
        if self.flash and seqlen != 1:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # 添加掩码以防止未来信息泄露
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # 调整输出形状并恢复原始顺序
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终线性变换和残差dropout
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

```



##### FeedForward

```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```



##### MOEFeedForward

```python
class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        # 初始化配置参数
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token分配的expert数量
        self.n_routed_experts = config.n_routed_experts  # 总共的expert数量
        
        self.scoring_func = config.scoring_func  # 得分函数类型
        self.alpha = config.aux_loss_alpha  # 辅助损失权重
        self.seq_aux = config.seq_aux  # 是否使用序列辅助损失
        
        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化top-k概率
        self.gating_dim = config.dim  # 输入维度
        
        # 定义权重矩阵
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        # 使用Kaiming均匀分布初始化权重
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        前向传播函数，计算每个token对应的expert得分并选择top-k expert。

        参数:
            hidden_states (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。

        返回:
            tuple: 包含三个元素的元组：
                   - topk_idx (torch.Tensor): 形状为 (batch_size * seq_len, top_k)，表示选中的expert索引。
                   - topk_weight (torch.Tensor): 形状为 (batch_size * seq_len, top_k)，表示选中的expert权重。
                   - aux_loss (torch.Tensor 或 None): 辅助损失，如果alpha > 0则返回，否则返回None。
        """
        bsz, seq_len, h = hidden_states.shape

        # 将输入张量展平
        hidden_states = hidden_states.view(-1, h)
        
        # 计算每个token到每个expert的得分
        logits = F.linear(hidden_states, self.weight, None)
        
        # 根据得分函数计算得分
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'不支持的评分函数 {self.scoring_func} 用于MoE门控')
        
        # 获取top-k的得分和对应的expert索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        # 如果需要归一化top-k的概率
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        # 计算辅助损失（仅在训练阶段且alpha大于0时）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # 使用序列辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                               torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 使用标准辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        
        return topk_idx, topk_weight, aux_loss
```



##### RMSNorm

![image-20241114154439329](../picture.asset/image-20241114154439329.png)

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```



### 预训练



 `python 1-pretrain.py` 执行预训练，得到 `pretrain_*.pth` 作为预训练的输出权重

什么参数也没改，使用2张A800 80G直接开始train，最终训练时间为24小时左右

```
torchrun --nproc_per_node 2 1-pretrain.py --use_wandb
```

![image-20241121212830766](../picture.asset/image-20241121212830766.png)

https://wandb.ai/coffeecat-university/MiniMind-Pretrain?nw=nwusercoffeecat

![image-20241121144557952](../picture.asset/image-20241121144557952.png)

> ```python
>  CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node 2 1-pretrain.py 
> ```
>
> 再次看下显存占用和 显存利用率
>
> ![image-20241224142755558](../picture.asset/image-20241224142755558.png)



预训练代码如下:

```python
import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(it, all):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

    model = Transformer(lm_config).to(args.device)
    # moe_path = '_moe' if lm_config.use_moe else ''

    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_data.csv", help="Path to training data")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')

    args = parser.parse_args()

    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model()
    df = pd.read_csv(args.data_path)
    df = df.sample(frac=1.0)
    train_ds = PretrainDataset(df, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if False and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
```



然后接下来是测试预训练的效果的代码:

```python
import random
import time

import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model_from = 1  # 1从权重，2用transformers

    if model_from == 1:
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'

        model = Transformer(lm_config)
        state_dict = torch.load(ckp, map_location=device)

        # 处理不需要的前缀
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        for k, v in list(state_dict.items()):
            if 'mask' in k:
                del state_dict[k]

        # 加载到模型中
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained('minimind', trust_remote_code=True)
    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')
    return model, tokenizer


def setup_seed(seed):
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置随机种子（如果有）
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机种子（如果有）
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动调优，避免不确定性


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    out_dir = 'out'
    start = ""
    temperature = 0.7
    top_k = 8
    setup_seed(1337)
    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    max_seq_len = 512
    lm_config = LMConfig()
    lm_config.max_seq_len = max_seq_len
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model(lm_config)
    model = model.eval()
    # int(input('输入0自动测试，输入1问题测试：'))
    answer_way = 0
    stream = True

    prompt_datas = [
        '椭圆和圆的区别',
        '中国关于马克思主义基本原理',
        '人类大脑的主要功能是',
        '万有引力是',
        '世界上人口最多的国家是',
        'DNA的全称是',
        '数学中π的值大约是',
        '世界上最高的山峰是',
        '太阳系中最大的行星是',
        '二氧化碳的化学分子式是',
        '地球上最大的动物是',
        '地球自转一圈大约需要',
        '杭州市的美食有',
        '江苏省的最好的大学',
    ]

    qa_index = 0
    while True:
        start = time.time()
        if answer_way == 1:
            # run generation
            prompt = input('用户：')
        else:
            if qa_index >= len(prompt_datas):
                break
            prompt = prompt_datas[qa_index]
            print('问题：', prompt)
            qa_index += 1

        prompt = tokenizer.bos_token + prompt
        x = tokenizer(prompt).data['input_ids']
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

        with torch.no_grad():
            res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                   top_k=top_k, stream=stream)
            print('回答：', end='')
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")
                continue

            history_idx = 0
            while y != None:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue
                # print(answer)
                if not len(answer):
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                print(answer[history_idx:], end='', flush=True)
                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

            print('\n')

        end = time.time()
        print(end - start, 's')
```

- `python 0-eval_pretrain.py`测试预训练模型的接龙效果

测试效果图如下:

![image-20241121213501823](../picture.asset/image-20241121213501823.png)

```text
模型参数: 26.878464 百万 = 0.026878464 B (Billion)
问题： 椭圆和圆的区别
回答：
椭圆和圆的区别：椭圆的外形比较大，圆的内壁比较平整，圆的中间没有明显的凹凸感，所以椭圆和圆的外壁比较平整，圆的内壁比较光滑，圆的内壁比较平滑，圆的内壁比较光滑。
椭圆和圆的区别：椭圆和圆的内壁比较明显，圆的外壁比较光滑，圆的内壁比较平整，圆的内壁比较光滑，圆的内壁比较光滑。
椭圆和圆的区别：椭圆的外壁比较光滑，圆的内壁比较光滑，圆的内壁比较光滑，圆的外壁比较光滑，圆的圆的内壁比较光滑，圆的内壁比较光滑。
椭圆和圆的外壁比较光滑，圆的内壁比较光滑，圆的内壁比较光滑，圆的内壁比较光滑，圆的内壁比较光滑。

2.048009157180786 s
问题： 中国关于马克思主义基本原理
回答：概论和中国特色社会主义理论体系概论的论文，是马克思主义理论研究的重要组成部分。在马克思主义基本理论体系中，中国的哲学、经济学和政治学等一系列学科的理论研究，都在马克思主义中国化的理论研究中存在着较大的分量和深度。
中国共产党马克思主义基本原理概论的论文，是马克思主义基本原理的集成体系，是马克思主义基本理论体系的理论体系，是马克思主义基本原理的理论组成部分。在马克思主义基本原理中，中国共产党在马克思主义基本原理中具有重要的理论价值和重要的现实应用价值，在马克思主义基本原理中，以马克思主义为主导的中国共产党，具有重要的理论和实践基础。在马克思主义基本原理中，中国共产党是马克思主义基本原理的理论体系，是马克思主义理论学科体系中的重要组成部分。
马克思主义基本原理与中国传统文化相比，马克思主义基本原理的理论体系是“理论与实践”结合，在理论与实践的结合中，理论与实践的结合是“理论与实践”相结合，而理论与实践相结合则是“理论与实践相结合”。

1.9480769634246826 s
问题： 人类大脑的主要功能是
回答：记忆。人类大脑中的神经元在人体内分泌出一种叫“脑电波”的物质，称为脑电波。脑电波在人体的脑组织中具有频率变化的特性，称为频率变化。
1、脑电波可以分为两类：一类是脑电波，另一类是脑电波，如脑电波对人的记忆，脑电波对人体的记忆，脑电波对人的记忆，脑电波对人体的记忆，脑电波对人体的记忆，脑电波对人体的记忆，脑电波对人的记忆，脑电波对人的记忆，脑电波对人体的记忆，脑电波对人的记忆，脑电波对人的记忆，脑电波对人体的记忆，脑电波对人的记忆，脑电波对人的行为，脑电波对人体的记忆，脑电波对人的行为，脑电波对于人的行为，脑电波对人的行为，脑电波对人的行为，脑电波对人体的行为。
2、脑电波对人的行为分为两类，一类是人在工作的时候，对人的行为，它是人在工作的时候，对人的行为，它是人在工作的时候，对人的行为。人在工作的时候，对人的行为，它是人在工作的时候，对人的行为，它是人在工作的时候，对人的行为，它是人在工作的时候，对人的行为。

2.1810011863708496 s
问题： 万有引力是
回答：全球最大的电动汽车集团之一，其全球总销量超过6亿辆。万有引力是全球最大的电动汽车集团之一，其全球总销量超过6亿辆。
2014年3月，万有引力发布了一份新一代电动汽车的最新报告，该报告显示，该报告还发布了2015年全球最大的电动汽车集团之一，其中包括万有引力在内的300家汽车制造商。
万有引力表示，其全球最大的电动汽车集团之一，其最新数据显示，2015年全球销量达到5.5亿辆。

0.8550448417663574 s
问题： 世界上人口最多的国家是
回答：墨西哥，最大的国家是墨西哥。世界上有70余万人口，其中少数民族人口超过1亿，是世界上人口最多的国家之一。
墨西哥的国家有：墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城等。
墨西哥人口最多的国家有：墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城。
墨西哥国家的国家有：墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨西哥城、墨

3.2232298851013184 s
问题： DNA的全称是
回答：基因序列，它是由基因的组成部分。基因序列可以分为基因序列和基因序列，基因序列可分为基因序列的单位、基因序列的单位和基因序列的单位。
基因序列的结构是：
1、序列：
基因序列是基因中一条线，其内在结构是：
1、基因序列的内在结构是：
基因序列是基因序列的内在结构。
2、基因序列是：
基因序列是基因的内在结构，其内在结构是：
1、基因序列是：
基因序列是：
1、基因序列是：
基因序列的内在结构是：
1、基因序列是：
基因序列是：
1、基因序列是：
基因序列是：
1、基因序列是：
2、基因序列是：
基因序列是：
1、基因序列是：
基因序列是：
1、基因序列的内在结构是：
DNA序列是：
1、基因序列的内在结构是：
DNA序列是：
1、基因序列是：
DNA序列的内在结构是：
1、基因序列是：
DNA序列是：
1、基因序列是：
DNA序列的内在结构是：
DNA序列的内在结构是：
1、基因序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
DNA序列的内在结构是：
D

3.228148937225342 s
问题： 数学中π的值大约是
回答：0，如果在1，π的值为0，则π的值为0，如果在1，π的值为0，则π的值为0，如果在1，π的值为0，则π的值为0，如果在1，π的值为0，则π的值为2，因此π的值为0，如果在10，π的值为2，因为π的值为1，因此π的值为2。
在2中，π与π的值是0，如果在1，π的值是0，那么π的值为1，因此π的值为1，由于π的值是0，π的值为2，因此π的值为1。
在2中，π的值为0，如果在2中，π的值为3，那么π的值为2。
如果在3，π的值为1，那么π的值为2。
如果在4，π的值为3，那么π的值为4。
如果在4，π的值为0，那么π的值为1。
如果在4，π的值为1，那么π的值为10。
如果在6，π的值为2，那么π的值为2。
如果在8，π的值为3，那么π的值为0。
如果在8，π的值为5，那么π的值为1。
如果在8，π的值为0，那么π的值为1。
如果在9，π的值为2，那么π的值为1。
如果在10，π的值为2，那么π的值为0。
如果在10，π的值为5，那么π的值为2。
如果在10，π的值为1，那么π的值为0。

2.795119285583496 s
问题： 世界上最高的山峰是
回答：“西部山脉”，“西南山脉”和“西北山脉”之间有“西部山脉”。
“西南山脉”是西北山脉中最著名的山峰。它是西南山脉的最高峰，也是西南山脉最高的山峰。
西南山脉在北纬20.5度的山峰中，位于南纬14.2度的东南方向，是西南山脉中的第一个山峰，是我国西北地区唯一的山峰，也是我国西南山脉中唯一的山峰。
在西南山脉中，“西南山脉”是西南山脉最重要的山脉之一，也是我国西南山脉中唯一的山脉，也是西南山脉中唯一的极高峰，是我国西南山脉中唯一的山峰，也是我国西南山脉中唯一的山脉。
在西南山脉中，“西南山脉”也是我国西南山脉中唯一的山峰，也是我国西南山脉中最高的山峰，也是我国西南山脉中唯一的山峰。
在西南山脉中，“西南山脉”是西北山脉中唯一的山峰，也是我国西南山脉中唯一的山脉。
在西南山脉中唯一的山峰，位于南纬14.2度的东南山脉中唯一的山脉中唯一的山脉。在西南山脉中唯一的山峰，位于南纬14.2度的东南山脉中唯一的山脉中唯一的山峰。

2.84102725982666 s
问题： 太阳系中最大的行星是
回答：白垩纪星系，在地球上，地球上行星的星系比地球上行星大，而白垩纪星系比地球上的行星更大，因此白垩纪星系的面积是地球上最大的。
白垩纪星系的面积大概是地球上面积的3倍，而白垩纪星系是面积的4倍。白垩纪星系的面积大概是地球上最大，而白垩纪星系的面积是地球上最小的。

0.6513104438781738 s
问题： 二氧化碳的化学分子式是
回答：CO2。CO2的化学性质与二氧化碳的化学性质不同。
二氧化碳是无色、无味的气体，具有较强的腐蚀性，所以在使用时要避免与其他化学物品接触，以免对皮肤造成严重的刺激。二氧化碳在空气中会吸收二氧化碳，因此在使用时要避免与其它物质接触。二氧化碳在空气中会变成粉末状，如果在空气中吸收二氧化碳，二氧化碳会吸收二氧化碳，所以在使用时要避免与其它物质接触，以免对身体造成严重危害。另外二氧化碳在储存时应避免与空气中的氧发生反应，以免氧化。
二氧化碳在储存时要避免与氧气接触，避免与氧气发生反应，以免对皮肤造成严重的刺激。二氧化碳在储存时要避免与氧气接触，以免对皮肤造成严重的刺激。二氧化碳在储存时要避免与氧气接触，以免对眼睛造成损伤。

1.5668373107910156 s
问题： 地球上最大的动物是
回答：雄性，雌性的动物有4只。
地球上的生物有3种，分别是雄性和雌性：雌性为雄动物，雌性为雌性，雌性为雌性，雄性为雌性。雌性雌性为雄性，雌性为雌性，雌性为雄性。
地球上有很多的大雄蛛，它们在地球上的生活和活动是很有可能的。
世界上有3种雌性动物是：雄性动物，雌性是雌性动物。
雌性动物是：雄性动物是雌性动物，雌性是雌性动物。
雌性动物是：雌性动物是雄性动物，雌性雌性动物是雄性动物。
雌性动物是雌性动物
雌性动物
雄性动物是指雌性动物，雄性动物雌性动物。
雄性动物是雄性动物。
地球上的动物是雄性动物，雌性动物是雌性动物。
地球上的动物是雌性动物，雄性动物是雌性动物。
地球上还有一种雌性动物，雌性动物是雄性动物，雌性动物是雌性动物。
地球上有雄性动物，雄性动物是雌性动物。
地球上的动物是雌性动物，雄性动物是雄性动物。

2.1399803161621094 s
问题： 地球自转一圈大约需要
回答：1.5个小时。
地球自转一圈大约需要1.5个小时。
地球自转一圈大约需要2.5个小时。
但地球自转一圈大约需要1.5个小时。
地球自转一圈大约需要1.5个小时。
地球自转一圈大约需要1.5个小时。
地球自转一圈大约需要2.5个小时。
地球自转一圈大约需要1.5个小时。
地球自转一圈大约需要1.5个小时。
地球自转一圈大约需要2.5个小时。
地球自转一圈大约需要1.5个小时。
地球自转一圈大约需要1.5个小时。
地球自转一圈大约需要3.5个小时。
地球自转一圈大约需要0.5个小时。
地球自转一圈大约需要0.5个小时。
地球自转一圈大约需要1.5个小时。
地球自转一圈大约需要5小时。
地球自转一圈大约需要6.5个小时。
地球自转一圈大约需要2.5个小时。
地球自转一圈大约需要4.5个小时。
地球自转一圈大约需要6.5小时。
地球自转一圈大约需要3.5个小时。
地球自转一圈大约需要5-6小时。
地球自转一圈大约需要2.5个小时。
地球自转一圈大约需要1小时。

2.3083384037017822 s
问题： 杭州市的美食有
回答：哪些
美食有哪些
美食有哪些
美容、美容、美发、保健、美甲等。美容、美发、护肤、清洁、美白、美体、美白、美白、美白、美白、美白、保健。美容、美发、保健、美甲等。
美容、美发、美甲等。美容、美发、护肤、清洁、美容、美甲等。美容、保健、美甲等。美容、美发、保健、美甲等。美容、美发、保健、美甲等。
美容、美容、美发、保健、美甲等。美容、保健、美甲等。美容、美甲等。美容、美发、保健、美甲等。

1.062201738357544 s
问题： 江苏省的最好的大学
回答：有：江苏省应用技术大学；江苏省应用技术学院；江苏省应用技术大学；江苏省应用技术学院；苏州大学；苏州大学；苏州大学；苏州大学；苏州工业园区；苏州大学；苏州工业园区。
江苏大学的前身是1979年创办的江苏高等专科学校，1998年更名为江苏大学。2014年4月，由江苏省人民政府批准成立江苏大学。2017年，江苏大学与苏州工业园区共建高校，并于2020年7月正式合并。江苏大学是江苏省首批高等教育国家级示范性高校、江苏省首批“世界一流学科建设高校”、“国家首批加快推进高等教育内涵式发展试验区”、“国家高水平大学基础学科创新平台”、“国家首批卓越农林人才教育培养计划”、“国家“国培计划”、国家“双一流”建设高校、“江苏省百所高水平大学”、江苏省首批“卓越农林人才教育培养计划”、“江苏省首批高水平学科创新平台”和“江苏省首批高水平特色学科创新平台”。2021年，学校入选国家首批“211工程”建设高校、全国首批54所“百校工程”建设高校、国防科技工业示范院校和国家“111”计划“111”计划。

2.908013343811035 s
```



>
>

## 指令微调

### sft

这里sft先用 sft_data_single.csv数据集

下面是对应Dataset

```python
class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024, prompt_max_len=512, answer_max_len=256):
        super().__init__()
        self.df = df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        #
        self.tokenizer = tokenizer
        self.padding = 0
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']

    def __len__(self):
        return self.df.shape[0]

    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res

    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        history = self.safe_eval(sample['history'])
        q = str(sample['q'])
        a = str(sample['a'])

        messages = []
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append(
                {"role": 'user', "content": str(history_message[0])[:self.max_length // 2]}
            )
            messages.append(
                {"role": 'assistant', "content": str(history_message[1])[:self.max_length // 2]}
            )

        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]

        # 实际长度
        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = len(input_id) - question_length - padding_len
        # 0表示不计算损失
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        return X_tensor, Y_tensor, loss_mask_tensor

```











`python 3-full_sft.py` 执行指令微调，得到 `full_sft_*.pth` 作为指令微调的输出权重

接着单机双卡做sft

注意接着用--use_wandb

```bash
torchrun --nproc_per_node 2 3-full_sft.py --use_wandb
```

> ```python
>  CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node 2 3-full_sft.py
> ```
>
> 同样注意：使用两张卡，分别是cuda:0和cuda:3
>
> ![image-20241224143405201](../picture.asset/image-20241224143405201.png)



跑了15个epoch， 大概

![image-20241122142422814](../picture.asset/image-20241122142422814.png)

![image-20241122142540664](../picture.asset/image-20241122142540664.png)

下面是sft的代码

```python
import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')

def Logger(content):
    """
    日志记录函数，仅在主进程中打印日志。

    参数:
        content (str): 要打印的日志内容。
    """
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(it, all):
    """
    计算当前迭代的学习率。学习率先线性增长然后余弦衰减。

    参数:
        it (int): 当前迭代次数。
        all (int): 总迭代次数。

    返回:
        float: 当前学习率。
    """
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch, wandb):
    """
    训练一个epoch。

    参数:
        epoch (int): 当前epoch编号。
        wandb (wandb.WandbRun 或 None): Weights & Biases运行对象，用于日志记录。
    """
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 获取当前迭代的学习率并更新优化器的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用自动混合精度上下文管理器（如果适用）
        with ctx:
            logits = model(X, Y).logits
            # 计算损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduction='none')
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # 梯度缩放以支持混合精度训练
        scaler.scale(loss).backward()

        # 梯度累积步骤
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 将日志发送到Weights & Biases
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item(),
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 保存模型检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()

def init_model():
    """
    初始化模型和分词器。

    返回:
        tuple: 包含模型和分词器的元组。
    """
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model_from = 1  # 1从权重文件加载，2使用transformers库中的预训练模型

    def count_parameters(model):
        """
        计算模型中可训练参数的数量。

        参数:
            model (nn.Module): 模型实例。

        返回:
            int: 可训练参数的数量。
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if model_from == 1:
        model = Transformer(lm_config)
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained('./minimind-v1-small', trust_remote_code=True)

    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(args.device)

    return model, tokenizer

def init_distributed_mode():
    """
    初始化分布式训练模式。
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=19, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')

    args = parser.parse_args()

    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 自动混合精度上下文管理器
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否为分布式数据并行运行？
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化Weights & Biases项目
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model()

    # 加载数据集并初始化数据加载器
    df = pd.read_csv('./dataset/sft_data_single.csv')
    df = df.sample(frac=1.0)
    train_ds = SFTDataset(df, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 初始化梯度缩放器和优化器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 编译模型（目前禁用）
    if False and not lm_config.use_moe and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    # 设置DistributedDataParallel
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 开始训练循环
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)

```



`python 2-eval.py`测试模型的对话效果，可见有了明显的指令遵循能力

```text
模型参数: 26.878464 百万 = 0.026878464 B (Billion)
[Q]: 你叫什么名字
[A]: 我是一台计算机程序，不会有任何个人名字。

[Q]: 你是谁
[A]: 我的名字是AI助手，是由一系列算法和算法组成的，可以为人们提供各种服务。我的设计使我能够不断学习和进化，以提供最优质的服务。

[Q]: 中国有哪些比较好的大学？
[A]: 中国有很多著名的大学，比如清华大学、北京大学、上海交通大学、中国农业大学等等。每个大学的学科和校园都有其独特的特色和优势。如果您感兴趣，可以去看看各大学院级别的综合性大学，如哈尔滨学士院、上海交通大学、北京大学等。

[Q]: 全世界最好的大学是什么？
[A]:  这个问题很主观，每个人对于“最好的大学”的定义都不同。但是，根据世界卫生组织的声誉，世界卫生组织是世界上最高的大学之一，有超过200个荣誉校区，被认为是世界上最好的大学之一。此外，还有一些著名的大学，如哈佛大学、斯坦福大学、麻省理工学院等，它们在全球范围内拥有极高的声誉和影响力。

[Q]: 你知道光速是多少吗？
[A]: 光速是17小时。光速是宇宙中最基本的常量之一，它不仅仅是光速。它是所有物质都可以达到的速度之一。光速是宇宙中最大的物理常数之一，也是最基本的速度单位之一。

[Q]: 你知道长江吗？
[A]: 长江是中国最长的河流，总长约6300千米。它位于中国的中部地区，流经中国境内，是中国最重要的河流之一。长江流域是中国最大的经济体，也是中国最重要的水利资源之一。

[Q]: 人类的血液主要由哪些成分组成？
[A]: 人类的血液主要由氨基酸、磷酸和硅酸盐组成。氨基酸是人体细胞中负责酶的组成部分，是构成人体组织和细胞的主要组分之一。磷酸和硫酸酯是负责人体内部的物质合成和传递的信息和蛋白质。

[Q]: 第一颗人造卫星是哪个国家发射的？
[A]: 第一颗人造卫星是中国的“天宫二号”，它于2007年运营。

[Q]: 你知道杭州有什么美食吗？
[A]:  杭州以其美味的美食而闻名，比如西湖醋鱼、龙井虾仁、东坡肉等等。此外，还有许多特色小吃，如龙井虾仁，杭州小笼包等。

[Q]: 你知道泰山在哪里吗？
[A]: 泰山位于中国的山东省，是中国的五岳之一。它位于山东省泰安市，是中国文化和历史的象征之一。泰山有千年历史，是中国传统文化的重要象征之一，也是世界著名的旅游胜地之一。

[Q]: 地球上最大的动物是什么？
[A]: 地球上最大的动物是蓝鲸，它可以长达30米，重达173吨。

[Q]: 地球自转一圈大约需要多少时间？
[A]: 地球自转一周需要大约24小时。因为地球是球形的，它的自转轴与地球轴的夹角是固定的，所以我们每天会经历大约24小时地球自转一圈。

[Q]: 人类最早使用的金属是什么？
[A]:  人类最早使用的金属是锁。在古代，人们将锁用于财产或财产的保管、财产管理等方面。随着时间的推移，人们逐渐使用锁来保护财产或其他财产。现在，金属是最早出现的物质，被用于工业、商业和其他领域。

[Q]: 水的化学分子式是什么？
[A]: 水的化学式是H2O，表示由两个氢原子和一个氧原子组成的分子。

[Q]: 大气层中含量最多的气体是什么？
[A]:  大气层中含量最多的气体是氮气，它占据了大约50％的体积。

[Q]: 世界上最高的山峰是什么？
[A]: 世界上最高的山峰是珠穆朗玛峰，位于中国和尼泊尔的边境处，海拔8848米。

[Q]: 你知道世界上最深的海沟是什么吗？
[A]: 当然，世界上最深的海沟是马里亚纳海沟，它位于西太平洋，深度达到了10911米。

[Q]: 最早发明印刷术的是哪个国家？
[A]: 最早印刷术是由意大利发明家毕达哥拉斯发明的，他于公元1096年发明了北美印刷术，并于公元前3000年左右完成了这项技术。

[Q]: 万有引力是谁提出的？
[A]: 万有引力是牛顿在17世纪提出的一个理论，描述了行星绕着恒星公转以及宇宙起源的过程中存在的引力效应。该理论被广泛应用于天文学、物理学、工程学和技术等领域。

[Q]: 光合作用的主要原理是什么？
[A]: 光合作用的主要原理是光能被吸收，并利用碳和二氧化碳来合成有机物（如葡萄糖和淀粉）。这个过程产生氧气和葡萄糖，这些有机物质在光的作用下会被二氧化碳和水分子分解成氧气和葡萄糖。这个过程需要能量和二氧化碳，所以光合作用是非常重要的。

[Q]: 你知道大熊猫的主要食物是什么吗？
[A]: 大熊猫的主要食物是竹子，但也会吃些其他植物、水果、坚果和小型动物。它们还会吃少量的植物果实、坚果和种子等。

[Q]: 海水为什么是咸的？
[A]: 海水是咸的主要原因是其中含有大量的盐分。盐分主要来自于海洋中的盐分溶液，当这些溶液中氯离子浓度低时，就会形成盐水。当这些氯离子与水分子接触时，它们被分散在海水中，形成了海水中的盐度较高的盐水。所以，海水中含有很少的盐分，而盐分也是海水中含有很少盐分的原因。

[Q]: 我们平时喝的牛奶主要含有什么营养成分？
[A]: 牛奶是富含蛋白质、钙、维生素D等营养成分的重要来源。适量的牛奶可以提供我们所需的蛋白质、钙、维生素D等营养成分，同时还可以提高人体免疫力。但是要注意的是，不要将牛奶煮成过多的脂肪和盐分，否则会导致摄入过多的脂肪和钙，影响健康。

[Q]: 一星期有多少天？
[A]:  一星期有七天。
```

下面是sft的训练代码

```python
import random
import time

import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model_from = 1  # 1从权重，2用transformers

    if model_from == 1:
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'

        model = Transformer(lm_config)
        state_dict = torch.load(ckp, map_location=device)

        # 处理不需要的前缀
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        for k, v in list(state_dict.items()):
            if 'mask' in k:
                del state_dict[k]

        # 加载到模型中
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained('./minimind-v1-small', trust_remote_code=True)
    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')
    return model, tokenizer


def setup_seed(seed):
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置随机种子（如果有）
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机种子（如果有）
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动调优，避免不确定性


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    out_dir = 'out'
    start = ""
    temperature = 0.7
    top_k = 16
    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    max_seq_len = 1 * 1024
    lm_config = LMConfig()
    lm_config.max_seq_len = max_seq_len
    # 对话是否携带历史对话（当前模型没有在连续对话数据集上训练，增大历史上文基本不会有新的问答能力）
    contain_history_chat = False
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model(lm_config)

    model = model.eval()
    # 推送到huggingface
    # model.push_to_hub("minimind")
    # tokenizer.push_to_hub("minimind")

    # answer_way = int(input('输入0自动测试，输入1问题测试：'))
    answer_way = 0
    stream = True

    prompt_datas = [
        '你叫什么名字',
        '你是谁',
        '中国有哪些比较好的大学？',
        '全世界最好的大学是什么？',
        '你知道光速是多少吗？',
        '你知道长江吗？',
        '人类的血液主要由哪些成分组成？',
        '第一颗人造卫星是哪个国家发射的？',
        '你知道杭州有什么美食吗？',
        '你知道泰山在哪里吗？',
        '地球上最大的动物是什么？',
        '地球自转一圈大约需要多少时间？',
        '人类最早使用的金属是什么？',
        '水的化学分子式是什么？',
        '大气层中含量最多的气体是什么？',
        '世界上最高的山峰是什么？',
        '你知道世界上最深的海沟是什么吗？',
        '最早发明印刷术的是哪个国家？',
        '万有引力是谁提出的？',
        '光合作用的主要原理是什么？',
        '你知道大熊猫的主要食物是什么吗？',
        '海水为什么是咸的？',
        '我们平时喝的牛奶主要含有什么营养成分？',
        '一星期有多少天？'
    ]

    messages_origin = []
    messages = messages_origin

    i = 0
    while i < len(prompt_datas):
        # Generate a random seed
        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)
        if not contain_history_chat:
            messages = messages_origin.copy()

        if answer_way == 1:
            prompt = input('[Q]: ')
        else:
            prompt = prompt_datas[i]
            print(f'[Q]: {prompt}')
            i += 1

        prompt = '请问，' + prompt
        messages.append({"role": "user", "content": prompt})

        # print(messages)
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-(max_seq_len - 1):]

        x = tokenizer(new_prompt).data['input_ids']
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

        answer = new_prompt

        with torch.no_grad():
            res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                   top_k=top_k, stream=stream)
            print('[A]: ', end='')
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")
                continue

            history_idx = 0
            while y != None:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue
                # print(answer)
                if not len(answer):
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                print(answer[history_idx:], end='', flush=True)
                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

            print('\n')

        if contain_history_chat:
            assistant_answer = answer.replace(new_prompt, "")
            messages.append({"role": "assistant", "content": assistant_answer})

```





### lora

这里lora用的peft库

```python
def find_linear_with_keys(model, keys=["wq", "wk"]):
    cls = torch.nn.Linear
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, cls):
            for key in keys:
                if key in name:
                    linear_names.append(name)
                    break
    return linear_names


def init_model():
    model_name_or_path = "./minimind-v1-small"
    tokenizer_name_or_path = "./minimind-v1-small"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(args.device)

    target_modules = find_linear_with_keys(model)
    peft_config = LoraConfig(
        r=8,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.to(args.device)
    return model, tokenizer

```

train_epoch函数还是那个样子，主要差别在于Dataset里加载的X,Y，loss

> ![image-20241224143605178](../picture.asset/image-20241224143605178.png)
>
> 这里只用了一张卡，做lora
> ![image-20241224143623406](../picture.asset/image-20241224143623406.png)

跑10个epoch试试

总共跑了18个小时左右

接着合并权重并保存：

```python
moe_path = '_moe' if lm_config.use_moe else ''
ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
state_dict = torch.load(ckp, map_location=args.device)
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

 
model = PeftModel.from_pretrained(model, args.save_dir)
# 合并并卸载 LoRA 权重
model = model.merge_and_unload()

ckp = f'{args.save_dir}/lora_sft{lm_config.dim}{moe_path}.pth'

if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    state_dict = model.module.state_dict()
else:
    state_dict = model.state_dict()

torch.save(state_dict, ckp)

```





## 人类对齐

### DPO

关于PPO，可以通过训练五子棋ai相似的方式来理解，而DPO，不太好理解，

一般而言，就记住了个loss

![image-20241218201923939](../picture.asset/image-20241218201923939.png)

然后这里主要调用trl的DPOTrainer来实现。

【DPO数据】：大约合并后共8万条dpo数据，人工标注的偏好数据，均来自[活字模型](https://github.com/HIT-SCIR/huozi) ，可以用于训练奖励模型，优化模型回复质量，使其更加符合人类偏好。

```python

def init_model():
    device = 'cuda:0'
    # Do model patching and add fast LoRA weights
    
    tokenizer_name_or_path = "./model/minimind_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    lm_config = LMConfig()
    model = Transformer(lm_config)
    # ckpt = "./minimind_dpo/dpo_512.pth"   
    ckpt = "./out/full_sft_512.pth"   
    state_dict = torch.load(ckpt,map_location=device)

    model.load_state_dict(state_dict, strict=False)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(device)

    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = init_model()
    training_config = DPOConfig(
        output_dir="./minimind_dpo",
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        report_to="none",
        save_steps=2000,
        learning_rate=4e-5,
        save_safetensors=False, # 设置为False，改为保存为pytorch格式的模型  
    )

    dataset_path = './dataset/dpo/dpo_demo.json'
    train_dataset = load_dataset('json', data_files=dataset_path)
    
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_config,
        beta=0.1,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=512
    )
    dpo_trainer.train()
    dpo_trainer.save_model("./minimind_dpo")

    # 7. save
    model = dpo_trainer.model
    ckp = f'./minimind_dpo/dpo_512.pth'

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, ckp)
```







## 评测

> C-EVAL[279] 是一个旨在评估基于<font color='red'>中文语境</font>的基础模型在<font color='red'>知识和推理能力</font>方面的能力的评估  工具。它<font color='red'>类似于 MMLU 基准评测</font>,<font color='green'>包含了四个难度级别的多项选择题:初中、高中、大学和专业。</font> 除了英语科目外,C-EVAL 还包括了初中和高中的标准科目。在大学级别,C-EVAL 选择了我国教  育部列出的所有 13 个官方本科专业类别中的 25 个代表性科目,每个类别至少选择一个科目,以  确保领域覆盖的全面性。在专业层面上,C-EVAL 参考了中国官方的国家职业资格目录,并选择了  12 个有代表性的科目,例如医生、法律和公务员等。这些科目按照主题被分为四类:STEM(科  学、技术、工程和数学)、社会科学、人文学科和其他领域。C-EVAL 共包含 52 个科目,并按照其  所属类别进行了划分,具体信息可参见图8.17。<font color='green'>C-EVAL 还附带有 C-EVAL HARD,这是 C-EVAL 中非常具有挑战性的一部分主题(子集),需要高级推理能力才能解决。</font>
>
> 
>
> ![image-20241218161124534](../picture.asset/image-20241218161124534.png)
>
> 
>
> C-Eval包含三份数据分别是<font color='purple'>dev，val和test</font>，其中dev数据有答案并且带有答案解释，目的是用来构建CoT思维链的few-shot提示语，val数据集有答案，而test数据集没有答案，<font color='red'>一般的，利用dev的few-shot在val数据做离线测试获得C-Eval评分，而在test数据集上提交答案给C-Eval官网获得最终得分。</font>
>
> ![image-20241218161214641](../picture.asset/image-20241218161214641.png)



CEval评测 经过full-sft的模型，

> 请注意： 这里只是简单的 zero-shot的评测了一下val数据集

结果如下：

总题数: 1346

总正确数: 317

总正确率: 23.55%

```python
import random
import time
import os

import pandas as pd
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig
import torch.nn.functional as F

warnings.filterwarnings('ignore')


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer',
                                              trust_remote_code=True, use_fast=False)
    model_from = 1  # 1从权重，2用transformers

    if model_from == 1:
        moe_path = '_moe' if lm_config.use_moe else ''
        # ckp = f'./out/single_chat/full_sft_{lm_config.dim}{moe_path}.pth'
        ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'

        model = Transformer(lm_config)
        state_dict = torch.load(ckp, map_location=device)

        # 处理不需要的前缀
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        # 加载到模型中
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained('minimind', trust_remote_code=True)
    model = model.to(device)

    return model, tokenizer


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    seed = random.randint(1, 2000)
    # device = 'cuda:0'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    lm_config = LMConfig()
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model(lm_config)
    model = model.eval()

    # 消息模板，具体实现根据你的tokenizer进行调整
    messages_origin = [{"role": "system", "content": "开始回答问题"}]

    # 定义文件目录
    File_Dir = "ceval/ceval-exam/val"
    results_dir = "ceval/ceval_result"

    # 确保结果目录存在
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 用于记录所有文件的总正确数和总题数
    total_correct = 0
    total_questions = 0

    # 遍历目录下的所有CSV文件
    for filename in os.listdir(File_Dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(File_Dir, filename)
            test_df = pd.read_csv(file_path)

            # 存储结果的DataFrame
            results_df = pd.DataFrame(columns=['question', 'A', 'B', 'C', 'D', 'answer', 'llm_answer', 'is_right'])
            total_correct_in_file = 0  # 用于记录当前文件的正确数

            for row in test_df.itertuples(index=True, name='Pandas'):
                id = getattr(row, 'id')
                question = getattr(row, 'question')
                A = getattr(row, 'A')
                B = getattr(row, 'B')
                C = getattr(row, 'C')
                D = getattr(row, 'D')
                right_answer = getattr(row, 'answer')

                prompt = f'{question}。选择 A: {A}, B: {B}, C: {C}, D: {D}'

                messages = messages_origin.copy()
                messages.append({"role": "user", "content": prompt})

                # print(messages)
                new_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                x = tokenizer(new_prompt).data['input_ids']
                x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
                res_ids = model.eval_answer(x)

                # 假设 res_ids 是模型的 logits 输出，我们使用 softmax 转换为概率分布
                probabilities = F.softmax(res_ids, dim=-1)

                # 定义每个选项的 token id
                A_id = tokenizer('A').data['input_ids']
                B_id = tokenizer('B').data['input_ids']
                C_id = tokenizer('C').data['input_ids']
                D_id = tokenizer('D').data['input_ids']

                # 获取每个选项的概率
                A_prob = probabilities[0, A_id].item()
                B_prob = probabilities[0, B_id].item()
                C_prob = probabilities[0, C_id].item()
                D_prob = probabilities[0, D_id].item()

                # 将每个选项的概率放入字典中便于处理
                options_prob = {
                    'A': A_prob,
                    'B': B_prob,
                    'C': C_prob,
                    'D': D_prob
                }

                # 找到具有最大概率的选项
                max_option_answer = max(options_prob, key=options_prob.get)

                # 比较答案并记录
                is_right = 1 if max_option_answer == right_answer else 0
                results_df = results_df.append({
                    'question': question,
                    'A': A,
                    'B': B,
                    'C': C,
                    'D': D,
                    'answer': right_answer,
                    'llm_answer': max_option_answer,
                    'is_right': is_right
                }, ignore_index=True)
                # print(f'id: {id} 问题: {question[:10]}... 是否正确: {is_right}')

                if is_right:
                    total_correct_in_file += 1

            total_correct += total_correct_in_file
            total_questions += len(test_df)

            # 计算当前文件的正确率并添加到结果DataFrame的最后一行
            accuracy = total_correct_in_file / len(test_df)
            results_df = results_df.append({
                'question': '-',
                'A': '-',
                'B': '-',
                'C': '-',
                'D': '-',
                'answer': f'文件 {filename} 的正确率: {accuracy:.2%}',
                'llm_answer': '-',
                'is_right': '-'
            }, ignore_index=True)

            print(f'{filename.split(".")[0]} ，{total_correct_in_file}/{len(test_df)}，正确率: {accuracy:.2%}')

            # 保存结果到CSV
            results_path = os.path.join(results_dir, f"{filename.split('.')[0]}_result.csv")
            results_df.to_csv(results_path, index=False)

    # 计算总正确率
    total_accuracy = total_correct / total_questions if total_questions > 0 else 0

    # 将各个文件的正确率以及总正确率写入到 "ceval/ceval_result/test.log"
    log_path = os.path.join(results_dir, "test.log")
    with open(log_path, 'w') as log_file:
        result = f"总题数: {total_questions}\n总正确数: {total_correct}\n总正确率: {total_accuracy:.2%}"
        log_file.write(result)

        print(result)

        for filename in os.listdir(File_Dir):
            if filename.endswith('.csv'):
                accuracy_file = pd.read_csv(os.path.join(results_dir, f"{filename.split('.')[0]}_result.csv"))
                last_row = accuracy_file.iloc[-1]['answer']
                log_file.write(f"{filename}: {last_row}\n")

```



CEval评测 经过lora-sft的模型，

![image-20241218190549564](../picture.asset/image-20241218190549564.png)







## 部署

#todo gradio

### 将训练好的模型推送到huggingface

* push_to_hub api接口

首先登录


```python
from huggingface_hub import notebook_login

notebook_login()
```

或者在终端输入

```shell
huggingface-cli login
```

在这两种情况下，系统都会提示您输入用户名和密码，这与您用于登录 Hub 的用户名和密码相同。

这之后就可以简单的push_to_hub()了

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model")
```

<font color='red'>这将创建新的存储库 **dummy-model** 在您的个人资料中，并用您的模型文件填充它。</font> 对标记器执行相同的操作，以便所有文件现在都可以在此存储库中使用。

> 如果您属于一个组织，
> 如果您希望使用特定的 Hugging Face 令牌
>
> ```python
> tokenizer.push_to_hub("dummy-model", organization="huggingface", use_auth_token="<TOKEN>")
> ```



> 简单使用
>
> ```shell
> huggingface-cli login
> ```
>
> ![image-20241224144445896](../picture.asset/image-20241224144445896.png)
>
> 在hugginface官网注册一个access token
>
> ![image-20241224144736054](../picture.asset/image-20241224144736054.png)
>
>
> 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
>
> ![](../picture.asset/image-20241224181511627.png)
> 运行如下代码将模型push到huggingface。
>
> ```python
> import torch
> import warnings
> from transformers import AutoTokenizer, AutoModelForCausalLM
> from model.LMConfig import LMConfig
> from model.model import Transformer
> 
> warnings.filterwarnings('ignore', category=UserWarning)
> 
> 
> def count_parameters(model):
>  return sum(p.numel() for p in model.parameters() if p.requires_grad)
> 
> 
> def export_transformers_model():
>  LMConfig.register_for_auto_class()
>  Transformer.register_for_auto_class("AutoModelForCausalLM")
> 
>  lm_config = LMConfig()
>  lm_model = Transformer(lm_config)
>  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
>  moe_path = '_moe' if lm_config.use_moe else ''
>  ckpt_path = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'
> 
>  state_dict = torch.load(ckpt_path, map_location=device)
>  unwanted_prefix = '_orig_mod.'
>  for k, v in list(state_dict.items()):
>      if k.startswith(unwanted_prefix):
>          state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
>  lm_model.load_state_dict(state_dict, strict=False)
>  print(f'模型参数: {count_parameters(lm_model) / 1e6} 百万 = {count_parameters(lm_model) / 1e9} B (Billion)')
> 
>  lm_model.save_pretrained("minimind-v1-small", safe_serialization=True)
> 
> 
> def export_tokenizer():
>  tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer',
>                                            trust_remote_code=True, use_fast=False)
>  tokenizer.save_pretrained("minimind-v1-small",safe_serialization=True)
> 
> 
> def push_to_hf():
>  def init_model():
>      tokenizer = AutoTokenizer.from_pretrained('./minimind-v1-small',
>                                                trust_remote_code=True, use_fast=False)
>      model = AutoModelForCausalLM.from_pretrained('./minimind-v1-small', trust_remote_code=True)
>      return model, tokenizer
> 
>  model, tokenizer = init_model()
>  # 推送到huggingface
>  model.push_to_hub("minimind-v1-small",safe_serialization=True)
>  tokenizer.push_to_hub("minimind-v1-small", safe_serialization=True)
> 
> 
> def test():
>  def init_model():
>      model_id = "./minimind-v1-small"
>      tokenizer = AutoTokenizer.from_pretrained(model_id,
>                                                trust_remote_code=True, use_fast=False)
>      model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
>      return model, tokenizer
> 
>  model, tokenizer = init_model()
> 
> if __name__ == '__main__':
>  # 1
>  # export_transformers_model()
>  # # 2
>  # export_tokenizer()
>  # # 3
>  # push_to_hf()
>  test()
> 
> ```
>
> 最后成功推送，并测试成功
>
> ![image-20241224171623749](../picture.asset/image-20241224171623749.png)
>
> 当传入了safe_serialization=False时，push_to_hub会报下列错误，
>
> RuntimeError: The weights trying to be saved contained shared tensors [{'tok_embeddings.weight', 'output.weight'}] that are mismatching the transformers base configuration. Try saving using `safe_serialization=False` or remove this tensor sharing.
> 然而这里由于传入了safe_serialization=True，导致忽略这些shared tensors,
> 但是忽略这些后最终 部署的minimind生成的token基本是 杂乱无用的信息	
> 因此，<font color='red'>需要指定safe_serialization=False</font>
>
> 然后重新操作，
> 但是发现如果指定safe_serialization=False后，难以加载模型，
> 将报错![image-20241224174629640](../picture.asset/image-20241224174629640.png)
>
> 诡异的是streamlit可以运行，
>
> ***<font color='red'>这里很玄学，我指定safe_serialization=False并且重新运行一遍后，又全都好了，有点懵</font>***
>
> <font color='red'>然后streamlit又崩溃了，然后我又什么都没变做了一遍，又都好了</font>
>
> > 最后的最后，所有的都设置为False是对的，
> > <font color='red'>有一个问题是push_to_hub应该就只能push一次</font>，
> > 这么多波折还有个问题时中途加载时可能错了一次模型





* huggingface_hub python库

**huggingface_hub** Python 库是一个包，它为模型和数据集中心提供了一组工具。

<font color='red'>提供了在 git 之上工作的简单 API 来管理这些存储库的内容并集成 Hub 在您的项目和库中。</font>

```python
from huggingface_hub import (
    # User management
    login,
    logout,
    whoami,

    # Repository creation and management
    create_repo,
    delete_repo,
    update_repo_visibility,

    # And some methods to retrieve/change information about the content
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
)
```









 ### 使用streamlit部署

这里选择托管到huggingface的spaces中

https://juejin.cn/post/7044757186064416798

https://blog.csdn.net/qiaotl/article/details/130528192

安装streamlit

```python
 pip install streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple  
```



首先在huggingface中spaces中创建一个space，然后git clone下来，并且仅仅修改app.py即可，

然后git push回去就行

下面是app.py的代码

> 本地测试app.py 
> streamlit run app.py 

```python
import json
import random
import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

# 设置页面标题
st.set_page_config(page_title="MiniMind-V1")
st.title("MiniMind-V1")

# 模型路径
model_id = "./minimind-v1"

# 使用 @st.cache_resource 装饰器来缓存加载的模型和分词器，避免每次请求都重新加载
@st.cache_resource
def load_model_tokenizer():
    # 加载预训练的语言模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True  # 如果模型有自定义代码，则信任并使用它
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,  # 不使用快速（Rust）版本的分词器
        trust_remote_code=True
    )
    # 将模型设置为评估模式
    model = model.eval()
    # 加载生成配置
    generation_config = GenerationConfig.from_pretrained(model_id)
    return model, tokenizer, generation_config

# 清除聊天记录
def clear_chat_messages():
    if 'messages' in st.session_state:
        del st.session_state.messages
    if 'chat_messages' in st.session_state:
        del st.session_state.chat_messages

# 初始化聊天消息
def init_chat_messages():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("我是由JingyaoGong创造的MiniMind，很高兴为您服务😄  \n"
                    "注：所有AI生成内容的准确性和立场无法保证，不代表我们的态度或观点。")

    # 如果会话状态中存在消息，则显示它们
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🫡" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        # 否则初始化会话状态中的消息列表
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

# 在侧边栏添加一些设定调整选项
st.sidebar.title("设定调整")
st.session_state.history_chat_num = st.sidebar.slider("携带历史对话条数", 0, 6, 0, step=2)  # 历史对话条数
st.session_state.max_new_tokens = st.sidebar.slider("最大输入/生成长度", 256, 768, 512, step=1)  # 最大生成长度
st.session_state.top_k = st.sidebar.slider("top_k", 0, 16, 14, step=1)  # top-k sampling 参数
st.session_state.temperature = st.sidebar.slider("temperature", 0.3, 1.3, 0.5, step=0.01)  # 温度参数

# 设置随机种子以确保结果可复现
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 主函数
def main():
    # 加载模型和分词器
    model, tokenizer, generation_config = load_model_tokenizer()
    messages = init_chat_messages()

    # 当用户通过 chat_input 输入时触发
    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
            messages.append({"role": "user", "content": prompt})
            st.session_state.chat_messages.append({"role": "user", "content": '请问，' + prompt + '？'})
        
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()  # 创建一个占位符用于实时更新助手的回答
            # 生成一个随机种子
            random_seed = random.randint(0, 2 ** 32 - 1)
            setup_seed(random_seed)

            # 准备提示文本，包括历史对话和当前问题
            new_prompt = tokenizer.apply_chat_template(
                st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):],
                tokenize=False,
                add_generation_prompt=True
            )[-(st.session_state.max_new_tokens - 1):]

            # 对提示文本进行编码
            x = tokenizer(new_prompt).data['input_ids']
            x = (torch.tensor(x, dtype=torch.long)[None, ...])

            # 开始无梯度推理，生成回答
            with torch.no_grad():
                res_y = model.generate(
                    x,
                    do_sample=True,  # 是否启用采样
                    max_new_tokens=st.session_state.max_new_tokens,  # 最大新生成的token数量
                    temperature=st.session_state.temperature,  # 控制生成的多样性
                    top_k=st.session_state.top_k,  # top-k sampling 参数
                    stream=True  # 流式生成
                )
                
                try:
                    y = next(res_y)  # 获取第一个生成的token
                except StopIteration:
                    return

                while y is not None:
                    answer = tokenizer.decode(y[0].tolist())  # 解码生成的token为字符串
                    if answer and answer[-1] == '�':  # 如果解码结果包含无效字符，则继续获取下一个token
                        try:
                            y = next(res_y)
                        except StopIteration:
                            break
                        continue
                    if not len(answer):  # 如果解码结果为空，则继续获取下一个token
                        try:
                            y = next(res_y)
                        except StopIteration:
                            break
                        continue
                    placeholder.markdown(answer)  # 更新占位符中的回答
                    try:
                        y = next(res_y)
                    except StopIteration:
                        break

            # 移除原始提示，只保留生成的回答部分
            assistant_answer = answer.replace(new_prompt, "")
            messages.append({"role": "assistant", "content": assistant_answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_answer})

    # 添加一个按钮用于清空对话
    st.button("清空对话", on_click=clear_chat_messages)

if __name__ == "__main__":
    main()
```

测试结果如下图所示

![image-20241224173647631](../picture.asset/image-20241224173647631.png)

使用git psuh后报下列错误

![image-20241224181716261](../picture.asset/image-20241224181716261.png)

> 这个问题也是 死活解决不了
>
> 最后干脆，直接在网站上提交app.py和requirements.txt，并且编辑
> 最终成功部署，如下图
> ![image-20241224211640061](../picture.asset/image-20241224211640061.png)



## 推理增强

#todo mctsr



# 分析

### 数据集

预训练用了seq-Monkey通用文本数据集，

该数据集是由多种公开来源的数据（如网页、百科、博客、开源代码、书籍等）汇总清洗而成，

格式统一为jsonl，经过严格筛选、去重、总量大概在**10Btoken**，适合中文大语言模型的训练



sft用的是匠数大模型sft数据集。

是一个经过数据清洗的包含10M条中文数据、2M条英文数据的数据集。

总量大概在**3Btoken**。



DPO数据，来自活字模型，大约8万条dpo数据，人工标注的偏好数据。



## 梯度累加与裁剪

[梯度累加](https://blog.csdn.net/hxxjxw/article/details/123407867)

在训练神经网络时，batch_size的大小会对最终模型效果影响很大。

一定情况下，batch_size设置的越大，模型就会越稳定。

然而，当我们需要做一些计算量很大的任务，batch_size太大会遇到"OUT OF MEMORY"的不可抗力报错。

<font color='red'>怎么办呢？如何在有限的计算资源的条件下，训练时采用更大的batch size呢？这就是梯度累加(Gradient Accumulation)技术了。</font>

**常规训练过程**

```python
for i, (inputs, labels) in enumerate(trainloader):
    optimizer.zero_grad()                   # 梯度清零
    outputs = net(inputs)                   # 正向传播
    loss = criterion(outputs, labels)       # 计算损失
    loss.backward()                         # 反向传播，计算梯度
    optimizer.step()                        # 更新参数
    if (i+1) % evaluation_steps == 0:
        evaluate_model()
```



**使用梯度累加的训练过程**

```python
for i, (inputs, labels) in enumerate(trainloader):
    outputs = net(inputs)                   # 正向传播
    loss = criterion(outputs, labels)       # 计算损失函数
    loss = loss / accumulation_steps        # 损失标准化
    loss.backward()                         # 反向传播，计算梯度
    if (i+1) % accumulation_steps == 0:
        optimizer.step()                    # 更新参数
        optimizer.zero_grad()               # 梯度清零
        if (i+1) % evaluation_steps == 0:
            evaluate_model()
```

> 注意这里的loss每次要除以accumulation_step，
>
> <font color='red'>当累加到一定步数optimizer更新的值其实是这些累加值的均值</font>。其实就相当于扩大了batch_size，让更新的loss更加平稳。一个batch的数据最后算的loss也是这个batch的数据个数的均值。
>
>   ***总结来讲，梯度累加就是每计算一个batch的梯度，不进行清零，而是做梯度的累加(平均)，当累加到一定的次数之后，再更新网络参数，然后将梯度清零。***
>
>   ***通过这种参数延迟更新的手段，可以实现与采用大batch_size相近的效果。***知乎上有人说
>
> "在平时的实验过程中，我一般会采用梯度累加技术，大多数情况下，采用梯度累加训练的模型效果，要比采用小batch_size训练的模型效果要好很多"。
>



[梯度裁剪](https://blog.csdn.net/ZacharyGz/article/details/135410610)

梯度剪裁，一种避免梯度爆炸的方式。

梯度裁剪（Gradient Clipping）是一种在训练神经网络时常用的技术，它用于防止梯度爆炸问题。\

梯度爆炸是指在训练过程中，梯度的大小急剧增加，导致权重更新过大，从而使得模型无法收敛或者性能急剧下降的现象.为了避免这个问题，梯度裁剪通过设定一个阈值来限制梯度的大小。如果梯度超过这个阈值，它们将被缩放至阈值以内，从而避免了大的权重更新。这样做有助于稳定训练过程，尤其是在训练复杂的深度学习模型时。

在PyTorch中，可以使用`torch.nn.utils.clip_grad_norm_`或`torch.nn.utils.clip_grad_value_`[函数](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782&utm_source=glcblog&spm=1001.2101.3001.7020)来实现梯度裁剪。

![image-20241215152733215](../picture.asset/image-20241215152733215.png)



梯度裁剪的两种常见形式是：

1. **梯度范数裁剪（Gradient Norm Clipping）**:
   - 这种方法涉及计算所有参数梯度的范数（例如L2范数），如果这个范数超过了设定的阈值，就将梯度缩放到这个阈值以内。在PyTorch中，这可以通过`torch.nn.utils.clip_grad_norm_`函数实现。
2. **梯度值裁剪（Gradient Value Clipping）**:
   - 这种方法对每个参数的梯度值进行独立裁剪，确保它们不会超过一个设定的最大值或最小值。在PyTorch中，这可以通过`torch.nn.utils.clip_grad_value_`函数实现。

> 梯度范数裁剪（Gradient Norm Clipping）
> 梯度范数裁剪通过调整整个参数梯度向量来保持其总体范数不超过特定阈值。它不关注单个梯度的值，而是关注所有梯度构成的整体范数。如果梯度的范数超过了指定的阈值，则会按比例缩小梯度向量的每个分量，使得整体范数等于或小于该阈值。
> 这种方法的数学表达为：
> 如果∥g∥>c，则更新梯度g为 g × c /∥g∥其中∥g∥是梯度向量的范数，c是预设的阈值
> 这种方法的优点是它能够保持梯度向量的方向不变，同时缩小其长度

> 梯度值裁剪（Gradient Value Clipping）
> 梯度值裁剪对梯度向量中的每个元素独立地进行裁剪，确保它们的值不会超出指定的范围。对于每个梯度 gᵢ ，如果 gᵢ > c，那么 gᵢ 就被设置为c；如果 gᵢ < -c，那么 gᵢ 就被设置为-c。
> 这种方法不考虑梯度向量的整体范数，只是单独地限制每个梯度值的大小。这可能会改变梯度向量的方向，但确保了没有任何一个梯度的绝对值太大。

总结
梯度范数裁剪和梯度值裁剪都可以防止梯度过大，但它们的方法和影响各不相同。梯度范数裁剪更多地用于保持梯度方向的稳定性，而梯度值裁剪则是限制梯度的大小。实际应用中选择哪一种取决于具体问题和模型的需求。<font color='green'>通常，梯度范数裁剪更受青睐，因为它能够在不改变梯度方向的前提下，有效地控制梯度的大小。</font>

梯度裁剪可能会影响学习过程，因为它人为地限制了梯度的大小，这可能会防止模型探索参数空间的某些部分。

示例：

```python
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

# 假设 model 是您的模型实例，optimizer 是您的优化器实例

max_grad_norm = 1.0  # 设定梯度的最大范数

for epoch in range(num_epochs):
    for inputs, targets in dataloader:  # dataloader 是您的数据加载器
        optimizer.zero_grad()  # 清除旧的梯度
        outputs = model(inputs)  # 获取模型预测结果
        loss = loss_function(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        
        # 在执行优化步骤之前裁剪梯度
        clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()  # 更新模型参数

```

在这段代码中，`clip_grad_norm_`函数将模型参数的梯度范数限制在`max_grad_norm`指定的范围内。如果梯度的范数超过了这个值，那么会按比例缩小梯度以使范数等于`max_grad_norm`。



```python
from torch.nn.utils import clip_grad_value_

clip_value = 0.5  # 设定梯度的最大绝对值

# 在训练循环中...
clip_grad_value_(model.parameters(), clip_value)

```

在这个例子中，任何大于`clip_value`的梯度值将被设置为`clip_value`，任何小于`-clip_value`的梯度值将被设置为`-clip_value`。



梯度裁剪通常在反向传播之后、参数更新之前进行。这有助于控制梯度的大小，从而防止训练过程中出现数值问题。



## 混合精度训练

检查 CUDA 是否可用：使用 torch.cuda.is_available() 检查是否有可用的 CUDA 设备。
获取 CUDA 设备名称：使用 torch.cuda.get_device_name(device) 获取 CUDA 设备的名称。
获取 CUDA 设备能力：使用 torch.cuda.get_device_capability(device) 获取 CUDA 设备的能力值。这些值通常表示为 (major, minor) 对，其中 major 表示主要版本号，而 minor 表示次要版本号。
根据能力值判断支持的精度：
如果 major 大于等于 5，则支持半精度（FP16）。
如果 major 大于等于 7，则支持张量核心（TF32）、混合精度（FP16 + FP32）、全精度（FP32）和双精度（FP64）

```python
import torch

# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
    # 获取第一个可用的 CUDA 设备
    device = torch.device("cuda:0")
    
    # 获取 CUDA 设备的一些基本信息
    print("CUDA Device Name:", torch.cuda.get_device_name(device))
    print("CUDA Device Capability:", torch.cuda.get_device_capability(device))
    
    # CUDA 设备的能力值可以用来确定支持的精度
    major, minor = torch.cuda.get_device_capability(device)
    
    # 根据 CUDA 设备的能力值判断支持的精度
    if major >= 5:
        print("Supports half precision (FP16)")
    if major >= 7:
        print("Supports tensor cores (TF32)")
    if major >= 7:
        print("Supports mixed precision (FP16 + FP32)")
    if major >= 7:
        print("Supports full precision (FP32)")
    if major >= 7:
        print("Supports double precision (FP64)")
else:
    print("No CUDA-capable device found.")

```



[float32、float16、bfloat16之间的差异](https://blog.csdn.net/a61022706/article/details/135190055)

简单来说就是

1. **Float32** ：又称FP32(float point 32)，是具有32位的浮点数，其中有1个符号位，8位表示指数，23位表示尾数。是标准的数据类型。

2. **Float16**：又称FP16，其位数只有FP32的一半，<font color='red'>指数保留5位</font>，尾数保留10位。FP16所能表示的最大的数字是64k，存在上溢和下溢的风险。
3. **Bfloat16**：又称BF16，<font color='red'>其指数位数与FP32相同，都是8位</font>，<font color='green'>因此表示的数据范围更广，但是精度比FP16要差。</font>



[混合精度训练（Mixed Precision Training）](https://www.cnblogs.com/s-tyou/p/16558976.html)

[混合精度训练（Mixed Precision Training）](https://blog.csdn.net/magic_ll/article/details/124689395)

混合精度训练（Mixed Precision Training）是一种在深度学习中提高训练速度和减少内存占用的技术。

在PyTorch中，通过使用半精度浮点数（16位浮点数，FP16）和单精度浮点数（32位浮点数，FP32）的组合。

> Python中的浮点数类型（float）： 在Python中，浮点数类型被称为float。float是一种内置数据类型，用于表示有小数部分的数值。它提供了双精度浮点数的精度，通常在大多数情况下，float的精度已经足够满足计算要求。<font color='green'>由于Python的设计哲学之一是简洁明了，因此没有引入明确的double类型，而是直接使用float来表示双精度浮点数。</font>



在深度学习中，使用FP16进行训练可以显著减少模型的内存占用，加快数据传输和计算速度，尤其是在配备有Tensor Core的NVIDIA GPU上。<font color='green'>然而，由于FP16的数值范围较小，可能会导致数值下溢（underflow）或精度损失</font>，因此在训练过程中可能需要一些特殊的技术（<font color='red'>如梯度缩放和混合精度训练</font>）来确保模型的数值稳定性和最终精度。



下面是一个使用PyTorch进行混合精度训练的例子：

1. 首先，确保你的硬件和PyTorch版本支持FP16运算。然后，导入必要的库

```python
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.cuda.amp import autocast, GradScaler
```



2. 创建一个简单的神经网络模型，例如一个多层感知机（MLP）：

```python
   class SimpleMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(10, 5)
           self.fc2 = nn.Linear(5, 2)

       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = self.fc2(x)
           return x

```

3. 启用混合精度训练

***<font color='red'>使用`autocast()`上下文管理器来指定哪些操作应该使用FP16执行：</font>***

```python
   model = SimpleMLP().cuda()
   model.train()
   scaler = GradScaler()

   for epoch in range(num_epochs):
       for batch in data_loader:
           x, y = batch
           x, y = x.cuda(), y.cuda()

           with autocast():
               outputs = model(x)
               loss = criterion(outputs, y)

           # 反向传播和权重更新
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()
```

在这个例子中，***`autocast()`将模型的前向传播和损失计算转换为FP16格式***。然而，***反向传播仍然是在FP32精度下进行的，这是为了保持数值稳定性。***

4. **使用`GradScaler`**：

   由于FP16的数值范围较小，可能会导致梯度下溢（underflow）。`GradScaler`在反向传播之前将梯度的值放大，然后在权重更新之后将其缩放回来

GradScaler 是 PyTorch 中 torch.cuda.amp 模块提供的一个工具，它用于帮助进行混合精度训练。在混合精度训练中，我们通常使用 FP16 来存储模型的权重和进行前向计算，以减少内存占用和加速计算。

然而，FP16 的数值范围比 FP32 小，这可能导致在梯度计算和权重更新时出现数值下溢（underflow），<font color='green'>***即梯度的数值变得非常小，以至于在 FP16 格式下无法有效表示。***</font>

<font color='blue'>GradScaler 通过在反向传播之前自动放大（scale up）梯度的值来解决这个问题。然后，在执行权重更新之后，GradScaler 会将放大的梯度缩放（scale down）回原来的大小。这个过程确保了即使在 FP16 格式下，梯度的数值也能保持在可表示的范围内，从而避免了数值下溢的问题</font>

```python
scaler = torch.cuda.amp.GradScaler()
for inputs, targets in dataloader:
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    scaler.scale(loss).backward()  # 放大梯度
    scaler.step(optimizer)  # 应用缩放后的梯度进行权重更新
    scaler.update()  # 更新缩放因子
```



5. 保存和加载模型

```python
   torch.save(model.state_dict(), 'model.pth')
   model.load_state_dict(torch.load('model.pth'))
```

在混合精度训练中，虽然模型的权重在训练过程中可能会被转换为 FP16 格式以节省内存和加速计算，但在保存模型时，我们通常会将权重转换回 FP32 格式。这是因为 FP32 提供了更高的数值精度和更广泛的硬件支持，这使得模型在不同环境中的兼容性和可靠性更好。

在 PyTorch 中，当你调用 model.state_dict() 方法时，默认情况下它会返回一个包含 FP32 权重的字典。即使你在训练时使用了 FP16，这个字典也会包含 FP32 权重，因为 PyTorch 会先转换为 FP32 再保存。同样，当你使用 torch.load() 加载模型时，如果模型权重是 FP16 格式，PyTorch 会自动将它们转换为 FP32。

注意，如果你的模型是在 GPU 上训练的，加载模型时应该使用 map_location 参数来指定加载到 CPU，然后再将模型转换为 FP32 并移回 GPU。

>**Gradient clipping**
>
>```python
>scaler = GradScaler()
>
>for epoch in epochs:
>    for input, target in data:
>        optimizer.zero_grad()
>        with autocast():
>            output = model(input)
>            loss = loss_fn(output, target)
>        scaler.scale(loss).backward()
>
>        # 先进行unscale 梯度，此时的clip threshold才能正确对梯度使用
>        scaler.unscale_(optimizer)
>        # clip梯度
>        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
>
>        # unscale_() 已经被显式调用了，scaler执行step时就不再unscalse更新参数，有nan/inf也会跳过
>        scaler.step(optimizer)
>        scaler.update()
>
>```



## DDP

https://www.cnblogs.com/liyier/p/18135209



https://www.cnblogs.com/liyier/p/18136458


具体参见上面两篇博客，非常详细



在PyTorch中，<font color='green'>分布式数据并行（Distributed Data Parallel</font>，简称DDP）是一种常见的分布式训练方法.

![image-20241215175603691](../picture.asset/image-20241215175603691.png)



>  数据并行是一种提高训练吞吐量的方法，它将模型参数和优化器状态复制到多个GPU上，然后将
>
> 训练数据平均分配到这些GPU上，
>
> 这样每个GPU只需要处理分配给它的数据，然后执行前向传播、反向传播获取梯度
>
> 当所有GPU执行完毕后，该策略会将不同GPU的梯度进行平均，得到整体的梯度来一次性同一更新所有GPU上的模型参数

如图所示,四条数据被分成两份,  由两张卡进行分别计算,然后我们会将两张卡的梯度进行平均后再更新模型,这  样便等效于执行了批次为 4 的梯度更新。



DDP的优势
使用DDP进行分布式训练有以下几个优势：

a. 加速训练：通过数据并行，DDP能够在多个设备或节点上同时处理不同批次的数据，从而加快训练速度。

b. 内存效率：DDP在每个设备上只保存模型的局部副本和相应的梯度，而不是整个模型的副本，这样可以节省内存。

c. 不需要额外的代码：在PyTorch中，使用DDP进行分布式训练几乎不需要修改您的原始模型和训练代码。


分布式数据并行（DistributedDataParallel，DDP）。***DDP采用Ring-All-Reduce架构***，其训练过程是多进程的。如果要用DDP来进行训练，我们通常需要修改三个地方的代码：数据读取器dataloader，日志输出print，指标评估evaluate。其代码实现略微复杂，不过我们只需要始终牢记一点即可：**每一块GPU都对应一个进程，除非我们手动实现相应代码，不然各个进程的数据都是不互通的。Pytorch只为我们实现了同步梯度和参数更新的代码，其余的需要我们自己实现。**



***DDP的训练过程可以总结为如下步骤：***

***1）在训练开始时，整个数据集被均等分配到每个GPU上。每个GPU独立地对其分配到的数据进行前向传播（计算预测输出）和反向传播（计算梯度）。***

***2）同步各个GPU上的梯度，以确保模型更新的一致性，该过程通过Ring-All-Reduce算法实现。***

***3）一旦所有的GPU上的梯度都同步完成，每个GPU就会使用这些聚合后的梯度来更新其维护的模型副本的参数。因为每个GPU都使用相同的更新梯度，所以所有的模型副本在任何时间点上都是相同的。***



###### Ring-All-Reduce算法

Ring-All-Reduce架构是一个环形架构，所有GPU的位置都是对等的。每个GPU上都会维持一个模型的副本，并且只需要和它相连接的两个GPU通信。

<font color='red'>对于第k个GPU而言，只需要接收来自于第k-1个GPU的数据，并将数据汇总后发送给第k+1个GPU。这个过程在环中持续进行，每个GPU轮流接收、聚合并发送梯度。经过 N 次的迭代循环后（N是GPU的数量），每个GPU将累积得到所有其他GPU的梯度数据的总和</font>此时，每个GPU上的梯度数据都是完全同步的。

DDP的通信开销与GPU的数量无关，因而比DP更为高效。如果你的训练数据达到了十万这个量级，并且需要使用4卡及以上的设备来进行训练，DDP将会是你的最佳选择。



 



 

