

## GPU显存估计


相较于预训练减少了模型梯度，优化器状态，同时激活值开销也降低
deepseek
部署需要考虑 模型参数， 激活值、还有部分函数库的开销

> 没想明白
> 只知道要算三个部分
> 1.模型参数，  使用bf16激活， 则 1B模型将需要  1B * 16/8 = 2G参数
> 2.激活值  batchsize * seq * hidden * 1 (训练时技术优化为2， 推理部署时貌似变成*1)
> 3.函数库等的加载开销

> 这里使用vllm， vllm将分页管理显存，还需vllm框架的开销

## 下载DeepSeek-R1-Distill-Llama-70B
使用git lfs拉取
```bash

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh 
apt-get install git-lfs
git lfs install

#GIT_LFS_SKIP_SMUDGE=1：这是一个环境变量设置。在 Git LFS 中，smudge 操作是指在检出文件时将 LFS 指针文件替换为实际的大文件。当设置 GIT_LFS_SKIP_SMUDGE=1 时，Git 在克隆或检出操作过程中会跳过这个 smudge 过程，也就是不会下载实际的大文件，而是只下载 LFS 指针文件。这些指针文件是小文本文件，它们指向存储在 LFS 服务器上的实际大文件。
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B

cd DeepSeek-R1-Distill-Llama-70B

# 查看 LFS 文件指针（未下载时显示指针哈希）
git lfs ls-files

git lfs pull #全部文件



```
![alt text](assets/vllm部署deepseek/image.png)

## 下载DeepSeek-R1-Distill-Qwen-1.5B
上面下载太慢，先下个1.5B的试下    
```bash
# hf网址
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

```

```bash
# 先别全下载
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

cd DeepSeek-R1-Distill-Qwen-1.5B

# 查看 LFS 文件指针（未下载时显示指针哈希）
git lfs ls-files

git lfs pull #全部文件

```


## vllm部署DeepSeek-R1-Distill-Qwen-1.5B
先用4卡， 2TP2PP
```bash
pip install vllm
# 命令中每行结尾的反斜杠 \ 是用于续行的，但它后面不能有空格，有空格会导致语法错误，需要去掉空格。修正后的命令如下：
vllm serve ./DeepSeek-R1-Distill-Qwen-1.5B  \
    --host 0.0.0.0 --port 7888 \

    --dtype float32 \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 2

```

## 调用
下面调用，
使用curl也行，
使用openai也行，因为与openai协议兼容

    --api-key "token123456" \ 咋curl没学会,先不加，使用openai函数时再加

### curl
```bash 
# Call the server using curl:
curl -X POST "http://localhost:7888/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "DeepSeek-R1-Distill-Qwen-1.5B",
		"messages": [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
	}'
```
成功
![alt text](assets/vllm部署deepseek/image-1.png)

使用nvidia-smi查看显存，占用很大，应该是vllm做了显存的页式管理？
![alt text](assets/vllm部署deepseek/image-2.png)

下面是单卡启动的显存图示
![alt text](assets/vllm部署deepseek/image-3.png)


### openai
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://127.0.0.1:7888/v1",
    api_key="token123456", #启动时加上key
)


completion = client.chat.completions.create(
  model="DeepSeek-R1-Distill-Qwen-1.5B",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)


print(completion.choices[0].message)
```
同样成功调用
![alt text](assets/vllm部署deepseek/image-4.png)



> 最后有一个问题，这开了个vllm把7888端口给占用了， jupyter-notebook无法连接了

> 在容器内部换个端口，然后通过7888远程访问容器jupyter，然后jupyter内部又能够访问vllm server 即可

更换端口为 8888？即可

![alt text](assets/vllm部署deepseek/image-5.png)

路径很奇怪，但确实成功启动了，这里的模型选择应该就是对服务器说选择那个模型，不涉及模型的参数文件的具体路径



### langchain
```py
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model='DeepSeek-R1-Distill-Qwen-1.5B', 
    openai_api_key='token123456', 
    openai_api_base='http://127.0.0.1:8888/v1',
    max_tokens=128
)

response = llm.invoke("给我一个很土但是听起来很好养活的男孩小名", temperature=1)
print(response.content)
```
![alt text](assets/vllm部署deepseek/image-6.png)


下面使用langchain的集成 deepseek库调用
https://python.langchain.com/api_reference/deepseek/chat_models/langchain_deepseek.chat_models.ChatDeepSeek.html
```python

from langchain_deepseek import ChatDeepSeek 
llm = ChatDeepSeek(
    api_key='token123456', 
    api_base='http://127.0.0.1:8888/v1',
    model="DeepSeek-R1-Distill-Qwen-1.5B", 
    temperature=0, 
    max_tokens=None,
    timeout=None, 
    max_retries=2,
    # other params... 
    )

messages = [ 
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."), 
] 
llm.invoke(messages)
```

也成功调用
![alt text](assets/vllm部署deepseek/image-7.png)



