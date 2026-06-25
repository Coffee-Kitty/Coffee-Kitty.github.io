# coffeecat 的小屋

<div class="home-hero">
  <p class="eyebrow">建立于 2024 年 10 月 17 日</p>
  <h1>把研究问题和技术学习整理成能持续跟进的系统。</h1>
  <p>公开站点现在只保留 Research 和 Tech 两个主体；日记、阶段总结、申请材料、简历和密钥都迁到了 private。</p>
</div>

<div class="home-grid">
  <a class="home-card" href="#/content-map.md">
    <strong>内容分类</strong>
    <span>哪些内容公开展示，哪些内容只做个人沉淀。</span>
  </a>
  <a class="home-card" href="#/learning-map.md">
    <strong>学习地图</strong>
    <span>当前主线、长期主题和维护节奏。</span>
  </a>
  <a class="home-card" href="#/timeline.md">
    <strong>公开时间线</strong>
    <span>按日期查看公开研究和技术笔记。</span>
  </a>
  <a class="home-card" href="#/research/README.md">
    <strong>Research</strong>
    <span>LLM / Agent、Code Agent 与 SE / LLM4SE 研究主线。</span>
  </a>
  <a class="home-card" href="#/tech/README.md">
    <strong>Tech</strong>
    <span>AI 工程、编程、工具、算法和课程笔记。</span>
  </a>
</div>

## 内容入口

| 栏目 | 主要内容 | 入口 |
| --- | --- | --- |
| Research | LLM / Agent、Code Agent、SE / LLM4SE、论文和实验结论 | [进入](/research/README.md) |
| Tech | AI 工程、编程、工具、算法和课程学习 | [进入](/tech/README.md) |
| LLM / Agent | 模型、数据、评测、RAG、Agent 框架 | [进入](/research/llm-agent/README.md) |
| Code Agent | SWE agent、真实仓库修复、工具调用和补丁验证 | [进入](/research/code-agent/README.md) |
| SE / LLM4SE | 代码智能、程序分析、bug reproduction、program repair | [进入](/research/se-llm4se/README.md) |
| AI Engineering | vLLM、Ollama、显存估计、深度学习基础 | [进入](/tech/ai-engineering/README.md) |

## 维护方式

这个博客现在按“入口页 + 自动索引 + 主题笔记”的方式维护：

1. 研究文章放进 `docs/research/`，技术文章放进 `docs/tech/`。
2. 公开学习笔记尽量在文件名或文章元信息里保留日期。
3. 运行 `npm run index` 生成公开时间线。
4. 运行 `npm run check:links` 检查内部链接。
5. 日记、阶段总结、申请材料、简历、密钥和草稿放进 `private/`，不要放进公开导航。
