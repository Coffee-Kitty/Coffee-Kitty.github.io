# IRIS: LLM-assisted static analysis for detecting security vulnerabilities

ICLR 25, Mayur Naik University of Pennsylvania 宾尼法西亚



## 摘要

​	软件往往容易产生安全漏洞。用于检测漏洞的程序分析工具在实际中效果有限，原因在于它们依赖人工标注的说明specifications）。大型语言模型（LLMs）虽然在代码生成方面表现出色，但它们难以对代码进行复杂推理来发现此类漏洞，尤其是当任务需要对整个代码仓库进行分析时。

​	我们提出 **IRIS** —— 一种**神经符号（neuro-symbolic）方法**，系统地结合 LLM 与静态分析，用于执行**全仓库级别的安全漏洞检测推理**。具体而言，IRIS 利用 LLM 来**推断污点分析规范（taint specifications）并执行 上下文分析（contextual analysis）**，从而减少人工编写规范和手动检查的需求。

​	在评估中，我们构建了一个新的数据集 **CWE-Bench-Java**，包含 120 个经过人工验证的真实世界 Java 项目的安全漏洞。最先进的静态分析工具 **CodeQL** 仅能检测出其中的 27 个，而 **IRIS（结合 GPT-4）** 能检测出 55 个（多出 28 个），并将 CodeQL 的平均误报率（false discovery rate）降低了约 **5 个百分点**。此外，IRIS 还发现了 4 个此前未知、现有工具无法检测出的漏洞。

IRIS 项目已开源，地址为：🔗 [https://github.com/iris-sast/iris](https://github.com/iris-sast/iris?utm_source=chatgpt.com)

## Intro

​	安全漏洞对软件应用及其用户的安全构成重大威胁。仅在 2023 年，就有超过 29,000 条 CVE 被报告——比 2022 年多出近 4,000 条（CVE Trends）**。尽管在揭露漏洞技术方面已有进展，但检测漏洞仍极具挑战性**。一种颇有前景的技术叫做**静态污点分析**（static taint analysis），它被广泛应用于诸如 GitHub CodeQL (Avgustinov et al., 2016)、Facebook Infer (FB Infer)、Checker Framework 和 Snyk Code (Snyk.io) 等流行工具中。然而，这些工具在实践中面临诸多挑战，这些挑战极大地限制了它们的有效性和可用性。







## example

### 🧩 场景例子：Web 后端的文件上传漏洞

假设在一个 Java Web 项目中，有如下代码：

```
public void uploadFile(HttpServletRequest req) {
    String filename = req.getParameter("file");
    saveFile(filename);
}

public void saveFile(String name) {
    File file = new File("/uploads/" + name);
    file.createNewFile();
}
```

------

### 🧠 传统静态分析怎么做

CodeQL / Infer 会尝试看：

- `req.getParameter("file")` → 可能是用户输入（潜在 **source**）
- `file.createNewFile()` → 可能是写入磁盘（潜在 **sink**）

但它需要人告诉它：

> “`getParameter()` 是危险的输入源。”
>  “`createNewFile()` 是敏感操作（sink）。”

如果没有人工写这份 **taint specification**，静态分析器就不会标记这条路径为“从 source → sink”。

🧱 所以它可能完全漏掉这类漏洞。

------

### 🤖 IRIS 怎么做

IRIS 用 LLM（如 GPT-4）读取项目源码、注释、文档，生成类似这样的自动推理结果：

| 函数名            | 猜测类型 | LLM 依据                                                     |
| ----------------- | -------- | ------------------------------------------------------------ |
| `getParameter()`  | source   | “Retrieves input from user HTTP request”                     |
| `createNewFile()` | sink     | “Creates a new file in filesystem; may cause path traversal issues” |

然后，IRIS 把这些推测交给静态分析引擎（例如 CodeQL）去跑**污点传播分析**。

这样一来，它就能自动检测到路径：

```
req.getParameter → filename → saveFile → createNewFile
```

→ 触发“路径穿越漏洞”警告。
 （这是现实中常见的 CWE-22。）

------

### ✨ 创新体现在哪里

> IRIS 不是去“重写静态分析算法”，而是让静态分析**自己长出眼睛去理解API语义**。

也就是：

- LLM 自动推测 API 的 taint 语义；
- 静态分析器保持精准逻辑推理；
- 最终实现“全仓库级”漏洞检测。



> 这里还有一个  验证步骤