# coffeecat 的小屋

<div class="home-hero">
  <p class="eyebrow">建立于 2024 年 10 月 17 日</p>
  <h1>把生活、学习和研究放到同一条可追踪的线上。</h1>
  <p>这里主要展示论文阅读、LLM/Agent 实践、代码工具、算法训练，以及适合公开的阶段复盘。</p>
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
    <span>按日期查看学习笔记和公开复盘。</span>
  </a>
  <a class="home-card" href="#/paper/README.md">
    <strong>论文学习</strong>
    <span>模型、RAG、代码智能体和研究笔记。</span>
  </a>
</div>

## 内容入口

| 栏目 | 主要内容 | 入口 |
| --- | --- | --- |
| 论文学习 | LLM、RAG、代码智能体、研究计划 | [进入](/paper/README.md) |
| LLM / Agent | vLLM、SFT、OpenManus、部署和工具链 | [进入](/llm/README.md) |
| 代码工具 | Python、Linux、VS Code、Playwright、设计模式 | [进入](/code/README.md) |
| 传统算法 | Hot 100、灵茶题单、数据结构和专题总结 | [进入](/传统算法学习/README.md) |
| 复盘记录 | 年末回顾、实习、公开生活和学习总结 | [进入](/life/README.md) |

## 维护方式

这个博客现在按“入口页 + 自动索引 + 主题笔记”的方式维护：

1. 新文章放进对应公开目录，私人内容放到 `private/`。
2. 公开学习笔记尽量在文件名或文章元信息里保留日期。
3. 运行 `npm run index` 生成公开时间线。
4. 运行 `npm run check:links` 检查内部链接。
5. 栏目 README 只保留精选入口和主题结构，避免手动维护所有文章链接。
