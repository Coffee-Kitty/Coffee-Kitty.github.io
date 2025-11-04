# 会议纪要

| 25.11.3 | 生成更优雅的代码的实证研究 |      |
| ------- | -------------------------- | ---- |
|         |                            |      |
|         |                            |      |
|         |                            |      |





# empirial study 

# An Empirical Study on the elegance/graceful  of code via llm , how far are we?

> 灵活，灵活，再灵活！！！

优雅度的衡量：  技术方案是  测试程序的 时间空间复杂度的分布图



Background and Motivation：





技术方案：

* 哪些数据？

> humaneval、mbpp（EvalPlus） HumanEval-X / MBPP-X
>
> APPS 
>
> CodeContests
>
> MultiPL-E
>
> CodeBench







* 需要多语言，



先做python



* 哪些模型？

暂定codellama 7b

以及 qwen 



* 哪些指标？

指标可以分为三类：



1. **功能正确性 (Functional correctness)**
   - `pass@k`：前 k 次生成至少一次通过测试
   - `exact match`：与参考代码完全一致
   - `CodeBLEU`：语法+语义相似度
   - `execution accuracy`：实际运行是否通过测试
2. **优雅性 (Elegance / Gracefulness)**
   - **时间 profiling**：运行时间分布
   - **空间 profiling**：内存占用分布
   - **能耗 profiling**：CPU/GPU 能耗 (可选)
3. **可比性/分析**
   - Edit similarity：LLM vs 人类修改差距
   - Performance distribution：箱线图/直方图展示时间/空间/能耗分布



**时间测量 (Time profiling)**

**空间测量 (Memory profiling)**

**能耗测量 (Energy profiling)**







* 实验：

1. **RQ0 (Prompt 的影响)**
   - 目标：探索不同 prompt 对 LLM 生成代码性能/优雅度的影响。
   - 示例：详细 prompt vs 简单 prompt；是否加上“请写高效/优雅代码”的提示。
2. **RQ1 (LLM vs 人类代码性能分布)**
   - 目标：对比人类编写代码和 LLM 生成代码在时间/空间/能耗上的分布。
3. **RQ2 (不同 LLM 表现的分布)**
   - 目标：不同模型（GPT-4, GPT-4-turbo, Claude, LLaMA 2 等）生成代码的性能分布差异。
4. **RQ3 (性能改进方法)**
   - 目标：探索简单改进性能的方法，如 **执行反馈 (execution feedback)**、**静态分析优化**、**多次生成选择最佳**。







* 结论：





# pipeline



