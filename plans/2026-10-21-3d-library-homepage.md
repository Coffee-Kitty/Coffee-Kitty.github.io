# 计划：3D 浮空图书馆首页

> 状态：执行中（Phase 1–2 完成，Phase 3 打磨中，待用户验收后提交）
> 建立：2026-10-21 · 维护人：coffeecat + pi
> 参考：小红书《手把手教你，用GPT6搓一个3D小岛个人主页！》（Rico有三猫）——落地页 = 左侧衬线文案 + 右侧可交互 3D 低模浮空小岛，"场景即导航"。参考帧存档：`/tmp/xhs-g*.png`（临时）。

## 1. 目标 / 非目标

**目标**

1. 首页（`docs/README.md`）顶部变成一屏落地页：左文案 + 右 **three.js 3D 低模浮空图书馆**。
2. 场景即导航：8 排书架 = 8 个主题分类；每本书脊 = 一篇公开笔记；点书架进分类 README，点书脊进文章。
3. 数据全自动：书架和书由 `tools/build_blog_index.py` 生成的 `docs/assets/site-graph.json` 驱动，新文章入仓即上书架。
4. 不破坏现有 docsify 能力：sidebar / search / pagination / latex / 移动端 / 无 WebGL 回退。
5. 仓库纪律：所有依赖 vendored（不用 CDN），场景为程序化几何（不引入 GLB 二进制）。

**非目标（本期不做）**

* 力导向知识星图页（原方案 C，留作 Phase 3 候选）。
* 138MB PDF 清理、`.ipynb_checkpoints` 清理（独立 Phase 4，单独提交）。
* AI 生成 3D 模型 / Blender 管线（不可文本维护，明确放弃）。

## 2. 现状审计摘要（2026-10-21）

| 项 | 事实 |
| --- | --- |
| 站点 | docsify 静态站，`docs/` 部署 GitHub Pages（user site，根路径 `/`），线上正常 |
| 内容 | 公开文章 79 篇（去 README）+ 8 分类；仅 16 篇带日期，`timeline.md` 覆盖低 |
| 体积 | `docs/` 277MB：`tech/cs-courses/program-analysis/nju-static-analysis/` 6 个 PDF 共 138MB；`.git` 687MB |
| 垃圾 | 4 处 `.ipynb_checkpoints`、空 `vendor/.tmp`、空 `vendor/mathjax/`、未跟踪 `docs/tech/infra/*.ipynb` |
| 习惯 | 已有 `build_blog_index.py` / `check_blog_links.py` 等脚本与 `npm run index / check:links / serve` |

## 3. 技术方案

### 3.1 数据管道（Phase 1，已完成）

* `tools/build_blog_index.py` 新增 `CATEGORY_META`（8 分类：id / 显示名 / 组 / 主题色）与 `collect_category_posts()` / `write_site_graph()`。
* 产物 `docs/assets/site-graph.json`：`{generated, totalPosts, categories:[{id,name,group,color,url,posts:[{title,url,date,sub}]}]}`；posts 按日期倒序（无日期排后）。
* 与 timeline 同一次 `npm run index` 生成，永不脱节。

### 3.2 渲染（Phase 2）

* 依赖：`docs/vendor/three/three.module.min.js`（three r170 module build，691KB，gzip ≈165KB），`<script type="importmap">` 映射 `"three"`，场景为 ES module `docs/scripts/library-scene.js`。**不引 OrbitControls**：手写球坐标轨道（拖拽旋转 / 滚轮与双指缩放 / 阻尼 / 闲置自动旋转），少 vendored 一个文件。
* 场景内容（全程序化几何，flatShading 低模风）：
  * 浮空岛：木甲板圆柱 + 岩石锥底 + 3 块悬浮碎石；整岛缓慢上下浮动。
  * 8 排书架环形排列、面向中心；架框 + 2 层书；书脊 = 文章（宽/高微随机、分类色 HSL 抖动、个别倾斜）；空分类（Code Agent）显示空架 + "待写" 牌。
  * 中心：书桌 + 摊开的书 + 台灯（自发光）+ 低模猫（呼吸缩放 + 尾巴摆动）。
  * 氛围：3 本绕岛缓转的飞书、120 粒浮尘 Points、暖色半球光 + 方向光软阴影；renderer alpha 透明，背景用站点纸色 CSS 渐变承接。
* 交互：Raycaster hover → 书架/书脊高亮 + tooltip（分类名+篇数 / 文章标题）；click（位移<5px 且 <400ms 判定为点击）→ `location.hash` 路由到分类或文章；底部 HTML 图例 chips（色点+分类名+篇数）可点击，兼作无 WebGL/读屏回退。
* 性能与回退：
  * `pixelRatio ≤ 2`（窄屏 ≤1.5）；窄屏阴影贴图降 512、浮尘减半。
  * `document.hidden` 或容器离屏（IntersectionObserver）暂停 rAF；路由离开首页 dispose（几何/材质/renderer/监听全清）。
  * `prefers-reduced-motion`：关自动旋转与浮动，时间冻结、按需渲染。
  * WebGL 不可用 / `site-graph.json` 拉取失败：隐藏 canvas，显示 `.scene-fallback` 文案 + 图例 chips 仍可导航；首页其余卡片不受影响。
* 与 docsify 集成：不用 docsify plugin hook（module 加载晚于 docsify 初始化），改用 **MutationObserver 监听 `#library-scene` 出现 + hashchange 判定首页路由**（`''`、`#/`、`#/README.md`）来 init/dispose，与 docsify 内部解耦。
* 版式：`docs/styles/home-scene.css`；hero 双栏 grid（文案 5fr / 场景 7fr），`.markdown-section:has(.home-hero)` 放宽到 1180px；<900px 单列、场景高 320px。
* `coverpage` 关闭（`index.html` 中 `coverpage: false`），首页即落地页；`_coverpage.md` 文件暂留不删，便于回滚。

### 3.3 文件变更清单

| 文件 | 动作 |
| --- | --- |
| `tools/build_blog_index.py` | 改：新增 graph 产出（已完成） |
| `docs/assets/site-graph.json` | 新增：生成物（已完成） |
| `docs/vendor/three/three.module.min.js` | 新增：vendored（已完成） |
| `docs/scripts/library-scene.js` | 新增：场景 module |
| `docs/styles/home-scene.css` | 新增：hero/tooltip/图例样式 |
| `docs/README.md` | 改：hero 区块替换旧 `.home-hero` |
| `docs/index.html` | 改：关 coverpage、加 importmap + module script + css |
| `plans/2026-10-21-3d-library-homepage.md` | 本文档 |

## 4. 分期与验收

| Phase | 内容 | 验收 | 状态 |
| --- | --- | --- | --- |
| 1 | 数据管道 + site-graph.json | `npm run index` 后 json 与目录一致；LSP 无 error | ✅ |
| 2 | 3D 场景 + hero 版式 + 路由集成 | 本地 `npm run serve` 截图：场景渲染、hover tooltip、点击跳转、移动端单列、无 console error | ✅ |
| 3 | 打磨：功能补完 + 美化；`npm run check:links` 通过 | 桌面+移动截图对比参考气质 | ✅ |

### Phase 3 明细（2026-10-21）

* 功能补完：
  * **分类书目面板**：点书架板/背板/名牌/合集书脊 → 镜头飞向该书架 + 弹出书目面板（分类色点+篇数+进入总览+全量文章列表带日期）；×/Esc/点空白关闭并镜头归位；移动端为底部抽屉；空分类显示“书架待写”。
  * **猫交互**：点猫 → 尾巴快速摇摆 + “喵～” tooltip。
  * **开场推镜**：初始半径 +9 阻尼推入。
  * 指针进入 hero 区域即暂停自动旋转（瞄准名牌/书脊时不漂移）；名牌锚点不再跟随浮岛 bob（解决点击移动目标问题）。
* 美化：甲板木纹与地毯环纹（程序化 canvas 纹理）、书脊共享烫金线纹理、书架暖光带、三花猫补丁+领圈、纸缘晕染（`.hero-scene::after`）、衬线大标题与 eyebrow 字距。
* 验收：面板开（AI Engineering 10 条）→ 点条目跳转 `显存估计.md` ✓；Esc 关闭 ✓；移动端底部抽屉 ✓；freeze 钩子点击名牌 ✓。

| 4 | 独立清理提交：PDF 处置（外链或移 private/）、checkpoints/infra 归置、空目录删除 | 仓库 `docs/` < 150MB；链接检查通过 | ⬜ |
| 5 | 上线：提交 + push，线上截图复核 | coffee-kitty.github.io 首页 = 新 hero | ⬜ |

## 5. 风险与对策

| 风险 | 对策 |
| --- | --- |
| three r170 光照强度语义（r155+ 物理光）导致过曝/过暗 | 本地截图迭代调参，半球光 ≈1.1、方向光 ≈2.2 起步 |
| 79 本书 + 8 架 ≈200 draw call，低端机掉帧 | 已控 pixelRatio 与阴影尺寸；必要时书脊改 InstancedMesh（预留 userData 映射方案） |
| GitHub Pages 强缓存旧 json | fetch 带 `cache:'no-cache'`；json 体积小，304 可接受 |
| docsify 路由切换残留 canvas / 监听 | hashchange + dispose 全清；MutationObserver 重建 |
| 中文书脊 tooltip 字体 | tooltip 为 HTML，继承站点字体栈，无 canvas 文字问题 |
| 回滚 | 场景相关全是新增文件 + README/index.html 局部改动，单提交可 revert |

## 6. 实施记录（2026-10-21）

* 验收证据（本地 localhost:3000，Tabbit 自动化）：
  * hover 书脊 → tooltip `SFT · LLM / Agent`；点击 → 路由 `#/research/llm-agent/models-data-evals/sft.md`；
  * 离开首页 canvas 数 = 0（dispose 生效），返回首页 = 1（重建生效）；
  * 桌面 1440×900 与移动 390×844 截图均正常；`npm run check:links` 通过。
* 过程中修复的关键 bug：
  1. MutationObserver 并发触发 sync()，在 `await loadData` 期间守卫失效 → 同场景初始化 4 次、4 张 canvas 叠放；加 `syncing` 并发锁 + 初始化前清理残留 canvas。
  2. 书架朝外导致镜头只看到背板 → 改为朝内环形书房布局，相机抬高俯视。
  3. docsify-toc 对裸 HTML h1 取 `firstChild.innerHTML` 得 `undefined` → toc headings 选择器排除 `#hero-title`，首页隐藏 toc 与冗余 markdown h1（CSS `:has()`，注意 toc 容器在 `.content` 下而非 `.markdown-section` 下）。
  4. `.markdown-section a` 颜色特异性高于 `.hero-btn` → 主按钮白字被主题色吞掉；补 `.markdown-section .hero-btn` 覆盖。
  5. 书脊目标仅数像素，射线易擦过 → pickAt 加 3×3 NDC 邻域容差；另加 `window.__libraryScene.freeze()` 测试钩子（冻结自动旋转/浮动/阻尼）。
* 全页模式（用户反馈“图书馆占据整个页面”）：
  * 首页隐藏 sidebar/字数统计/分页，hero 擑满整屏（`calc(100vh - 64px)`），文案卡（磨砂纸片）叠左上、图例叠底部居中、hint 叠右下；桌面 canvas 右移（width 128% / left -2%）给文案让位。
  * 书架分类名牌改为 HTML 覆盖层（衬线字 pill + 色点 + 篇数 + 连接线），仅在该书架正面朝镜头时淡入；锚点落入标题卡区域时自动让位隐藏；移动端隐藏名牌、图例改横向滚动条。
  * 竖屏按水平视野适配初始半径（38）与俯角（0.95）。
* 修复 docsify 存量坑（关 coverpage 后暴露）：根路由 `/` 下 `getParentPath('/')` 为假值导致 `_navbar.md` 拉取被跳过 → index.html 启动脚本把 `#/` 归一为 `#/README.md`。
* 修复移动端首页导航死路：mergeNavbar 把 navbar 并入 sidebar，而首页隐藏了 sidebar → 桌面才隐藏 sidebar/折叠钮；移动端保留汉堡按钮并隐藏叠在其上的 sidebar-pin 按钮，抽屉可用（内含 navbar）。
* 已知遗留（不阻塞）：three r170 中 `PCFSoftShadowMap` 标记 deprecated（功能正常）；移动端 docsify navbar 换行拥挤为存量问题。

## 7. 待用户拍板（不阻塞 Phase 2）

1. coverpage 已按建议关闭；若想保留全屏封面，改回 `coverpage: true` 即可（文件未删）。
2. Phase 4 的 PDF 处置方式：外链 / 移 `private/` / 保留，三选一。
3. 是否要在 Phase 3 之后加"星图模式"切换（书飞出变节点）。
