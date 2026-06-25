# 内容分类

这个博客建议按“公开展示”和“个人沉淀”分层维护。公开层负责给别人看，也方便自己回顾；个人层保留原始记录、草稿和素材，不进入导航和时间线。

## 公开展示

| 分类 | 放什么 | 当前目录 |
| --- | --- | --- |
| Research / LLM Agent | 模型、数据、评测、RAG、Agent 框架和相关论文 | `/research/llm-agent/` |
| Research / Code Agent | SWE agent、真实仓库修复、工具调用、补丁验证和相关论文 | `/research/code-agent/` |
| Research / SE LLM4SE | 代码智能、程序分析、bug reproduction、program repair | `/research/se-llm4se/` |
| Tech / AI Engineering | vLLM、Ollama、显存估计、深度学习基础 | `/tech/ai-engineering/` |
| Tech / Programming | Python、CPython、爬虫、设计模式、编程语言 | `/tech/programming/` |
| Tech / Dev Tools | VS Code、Typora、zsh、代理、Playwright、Docker | `/tech/dev-tools/` |
| Tech / Algorithms | Hot 100、灵茶题单和算法专题 | `/tech/algorithms/` |
| Tech / CS Courses | 程序分析、分布式系统、计算机导论等课程笔记 | `/tech/cs-courses/` |

## 个人沉淀

| 分类 | 处理方式 | 原因 |
| --- | --- | --- |
| 日记 | 放入 `private/diary/` | 太私人，适合自查，不适合公开索引 |
| 研究日志和组会流水账 | 放入 `private/research-logs/` | 适合沉淀，不适合作为公开入口 |
| 阶段总结、找实习、申请材料 | 放入 `private/reviews/` 或 `private/applications/` | 默认不公开 |
| 简历、密钥 | 放入 `private/resume/` 和 `private/secrets/` | 不应进入公开站点 |
| 原始素材 | 放入 `private/raw-assets/` 或迁到对应 `docs/assets/` | 降低公开站点噪声 |
| 临时文件 | 加入 `.gitignore` | 避免 `.DS_Store`、Office 临时文件污染历史 |

## 建议的长期目录

```text
docs/
  README.md              # 首页
  content-map.md         # 内容分类
  learning-map.md        # 学习主线
  timeline.md            # 公开学习时间线，自动生成
  research/              # 研究主线
  tech/                  # 技术学习
  assets/                # 公开文章使用的图片和附件

private/                 # 不提交，放真正私人的日记、简历、密钥和草稿
```

## 新内容放哪里

| 新内容类型 | 建议位置 |
| --- | --- |
| LLM、RAG、评测、Agent 论文 | `/research/llm-agent/` |
| Code Agent、SWE-bench、自动修复、仓库级编程 Agent 论文 | `/research/code-agent/` |
| 代码智能、软件工程、程序分析研究 | `/research/se-llm4se/` |
| 部署、推理服务、深度学习基础 | `/tech/ai-engineering/` |
| Python、CPython、编程语言、设计模式 | `/tech/programming/` |
| 编辑器、Shell、代理、Docker、Playwright | `/tech/dev-tools/` |
| 算法专题总结 | `/tech/algorithms/` |
| 课程笔记 | `/tech/cs-courses/` |
| 私人日记 | `private/diary/`，不要放进 `docs/` |
