# Code Agent

这个目录用于整理 code 相关的 agent 论文，只关注能处理代码仓库、软件工程任务、bug 修复、测试验证、代码生成与修改的工作。

第一阶段先按两类维护：

1. **Benchmark**：code agent 评测任务、数据集、排行榜、验证方式。
2. **Method**：具体 code agent 系统、训练方法、工具接口和工程框架。

## 第一阶段目标

| 优先级 | 目标 | 论文 |
| --- | --- | --- |
| P0 | 先理解任务定义和基本 agent scaffold | SWE-bench, SWE-agent |
| P1 | 看更难、更接近真实价值的评测 | SWE-Bench Pro, SWE-Lancer, Multi-SWE-bench |
| P1 | 看训练数据、环境和 post-training 方向 | SWE-smith, SWE-Gym, SWE-RL, SWE-Master |
| P2 | 看 CI、skill、部署和自进化等扩展场景 | SWE-CI, SWE-Skills-Bench, Live-SWE-agent, SWE-World |

agentic rl:

AgentFlow: Trainable Flow Engineering for Agentic Reinforcement Learning  

AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning

## Benchmark

| 论文 | 简介 | 特点 | 状态 |
| --- | --- | --- | --- |
| SWE-bench: Can Language Models Resolve Real-World GitHub Issues? | 最核心的 code agent 评测基准，来自真实 GitHub issue | 2,294 个真实 issue，Python 为主，Verified 子集 500 个高质量样本 | 待读 |
| SWE-bench Lite | SWE-bench 的 300 实例子集 | 聚焦 self-contained 功能性 bug，适合作为入门评测集 | 待读 |
| Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving | 多语言 issue resolving benchmark | 覆盖 Java、TypeScript 等非 Python 语言，用来看跨语言泛化 | 待读 |
| SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? | 更难的 long-horizon SWE benchmark | 企业级、跨文件、复杂依赖，对顶级模型区分度更高 | 待读 |
| SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering? | 基于 Upwork 真实外包任务的评测 | 总价值约 $1M，包含独立工程任务和 EM 管理任务，用经济价值量化能力 | 待读 |
| SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration | 持续集成场景下的代码库维护评测 | 关注 CI pipeline 中 agent 维护代码库的能力 | 待读 |
| SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering? | 评估 agent skill 对真实软件工程任务的帮助 | 关注 skill / 知识注入是否真的提升 code agent 表现 | 待读 |

## Method

### 训练与数据

| 论文 | 简介 | 亮点 | 状态 |
| --- | --- | --- | --- |
| SWE-smith: Scaling Data for Software Engineering Agents | 大规模 SWE 训练数据构建 | 将 SWE 训练数据扩展到数万级别，适合看数据规模化路线 | 待读 |
| Training Software Engineering Agents and Verifiers with SWE-Gym | SWE agent 训练环境 | 2,438 个真实任务实例，支持 RL 和 SFT，也训练 verifier | 待读 |
| SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution | 用 RL 训练 SWE agent | 使用开源 GitHub 演化数据做强化学习，关注 reasoning 和 repair 能力 | 待读 |
| SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training | SWE agent post-training 框架 | 系统探索 SFT、RL 等训练策略组合 | 待读 |

### Agent 架构与工程

| 论文 | 简介 | 亮点 | 状态 |
| --- | --- | --- | --- |
| SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering | ACI 框架和经典 code agent scaffold | 强调 Agent-Computer Interface 对真实性能的影响，是入门必读 | 待读 |
| OpenHands | 开源 code agent 平台 | 关注软件开发任务执行、工具调用、环境交互和平台化工程 | 待读 |
| Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly? | 自进化 code agent | 运行时动态更新自身策略，适合看 agent 是否能在线改进 | 待读 |
| SWE-World: Building Software Engineering Agents in Docker-Free Environments | Docker-free SWE agent 环境 | 解决 Docker 依赖和部署负担，适合看轻量化运行环境 | 待读 |

## 阅读顺序

1. **SWE-bench**：先明确 code agent 的标准任务形状：给定 issue 和仓库，产出能通过测试的 patch。
2. **SWE-agent**：理解 ACI、shell、编辑器、测试反馈如何组成 agent 执行闭环。
3. **SWE-bench Lite / Verified**：用小规模、高质量子集理解 benchmark 如何被实际使用。
4. **OpenHands**：看 code agent 从论文 scaffold 走向开源平台时需要哪些工程抽象。
5. **SWE-Gym / SWE-smith / SWE-RL / SWE-Master**：进入训练与数据路线，理解如何用轨迹、任务环境和反馈提升 agent。
6. **SWE-Bench Pro / SWE-Lancer / Multi-SWE-bench**：看 benchmark 如何向长任务、经济价值、多语言泛化扩展。
7. **SWE-CI / SWE-Skills-Bench / Live-SWE-agent / SWE-World**：最后看 CI、skill、自进化和部署环境等细分问题。

## 单篇笔记模板

后续每篇论文可以按这个结构整理：

```text
论文：
类别：Benchmark / Method

问题：
方法：
实验或证据：
对下一步的影响：
```

如果是 Benchmark，额外记录：

```text
任务来源：
输入/输出：
验证方式：
指标：
局限：
```

如果是 Method，额外记录：

```text
Agent 可用工具：
上下文构建方式：
行动循环：
失败恢复：
成本：
```

## 暂不放这里

通用 LLM agent、RAG、模型训练、非代码任务 agent 放到 [LLM / Agent](/research/llm-agent/README.md)；程序分析和代码智能但不强调 agent 执行闭环的论文放到 [SE / LLM4SE](/research/se-llm4se/README.md)。
