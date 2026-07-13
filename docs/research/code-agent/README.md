# Code Agent

这个目录用于整理 code 相关的 swe agent 系列论文

## 1. **Benchmark**：code agent 评测任务、数据集、排行榜、验证方式。

SWE-bench团队自己的更新入口，页面里直接挂了SWE-bench、Verified、Multilingual、Multimodal、Lite 等 benchmark家族入口，也会发 mini-SWE-agent、SWE-smith 这类相关更新。
https://www.swebench.com/blog.html

### **SWE-bench-(lite)**: 
ICLR24, 普林斯顿
基于github issue及其对应pr的任务
![task format](assets/README/image.png)
![repo分布](assets/README/image-1.png)
![统计数据](assets/README/image-2.png)
![task example](assets/README/image-3.png)

### **SWE-bench（verified）**:
openai
500 个 human-validated samples，不是独立论文

openai官方解释报告：https://openai.com/index/introducing-swe-bench-verified/

![数据集上的表现](assets/README/image-4.png)

### Multi-SWE-bench/ Multi-SWE-RL: 
nips25,字节
**多语言版的现实 issue 修复基准，补足 SWE-bench 偏 Python 的问题。**

* 包含 7 种广泛使用编程语言中的 1,632 个 issue：Java、TypeScript、JavaScript、Go、Rust、C 和 C++，进行严格的人工验证，与 SWE-bench Verified 的标准保持一致。
* 作为 Multi-SWE-RL 社区的初始贡献，作者发布了一个包含 4,723 个容器化 issue 修复实例的数据集，覆盖 7 种编程语言。每个实例都配备了可复现的执行环境，使其可以直接用于真实软件场景中的强化学习 agent 训练。

MagentLess：多语言改造版 Agentless；
MSWE-agent：多语言改造版 SWE-agent；
MopenHands：多语言改造版 OpenHands
![简单结果图](assets/README/image-5.png)


### **SWE-bench（Pro）**:
测试当前 AI agent 是否真的具备长程工程能力

SWE-bench Multimodal 

Visual SWE-bench 

### **SWE-Lancer**
关注 JavaScript 和 TypeScript，包含来自 Upwork 的 1,400 多个自由职业任务，其中既包括技术任务，也包括管理任务。

SWE-Skills-Bench

## 2. **Method**：具体 code agent 系统、训练方法、工具接口和工程框架。

agentless

**SWE-agent** : 基于简单react loop的 通用repo-level agent

SWE-smith

SWE-Gym
SWE-RL
SWE-Master
SWE-CI
Live-SWE-agent
SWE-World

agentic rl:
AgentFlow: Trainable Flow Engineering for Agentic Reinforcement Learning  
AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning
