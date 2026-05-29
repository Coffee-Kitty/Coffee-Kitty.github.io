<!--
 * @Author: coffeecat
 * @Date: 2025-02-26 16:14:36
 * @LastEditors: Do not edit
 * @LastEditTime: 2025-03-10 16:16:21
-->


## agent计划书  v1.0

### 预期
给定一个网址url，
agent将
1. 自动**规划** 基本思路步骤，   
2. 逐步执行各个步骤，这里给其提供 **tools** 使用能力，
3. 获得工具执行的结果后，将其与上一步骤prompt**拼接**作为新的输入，继续模型规划步骤
4. 循环往复直到找到漏洞，或者到达指定步骤时间等限制



### 需要的基础点
1. 模型： 选用目前最强的 Deepseek-R系列
    部署尽可能大的deepseek模型
        (关于推理方面的显存如何预估？)
2.  对于模型的输入，以 **embedding vector**的形式，
    这里需要补充 embedding模型的一些知识
    * prompt模板，prompt方法(few shot)设计


3. $**$可以考虑使用**RAG增强**模型在 目标方面，这里是安全 的知识能力
    * 向量数据库的引入，这里貌似存在各种各样的chunk方法

4.  **规划**能力
    基本的**MapReduce**任务分解尝试
    **COT**等方法的使用尝试，甚至TOT，GOT,CR

5.  **tools使用**及**交互**能力   
    根据目标任务来定，这里就是基本的代码解释器，
    <u>关于*安全工具*，若是命令行形式的，可以考虑bash交互</u>

6. 模型**反馈反思能力**，从而迭代优化
    * 首先尝试经典的**ReACT**反馈框架
    * 关于迭代终止条件


### 重点
首要的，需要学习 ReACT机制
紧接着是，tools的使用与交互机制，
此时已经可以做一个简单的单步骤pipeline

剩余增强，则是 首先考虑prompt，
然后是 embedding能力
然后是 规划能力，
最后是 RAG


### 困惑

上述整个过程是否可以由特定框架支持， 如langchain、llamindex？







## agent计划书 v2.0

接触了pentestAssistant 和 openmanus两个项目后 新的大体规划与疑问


pentestAssistant workflow


openmanus workflow



### 疑问
1. 
funtion calling 对比 基本的对话形式 ， 优点在哪里？
    ds，qwq暂不支持 function calling

为了对接这些模型，则需要以 chat 模式 进行function calling的转写工作


2. 模型本身对于工具的知识是否足够
    使用rag的策略一定是生成更好吗？
    如果做安全工具agent的话，是否考虑 专门的安全大模型

3. 工具
   bash shell之强大
   似乎基本工具用shell交互就行了

    也就是说目前比较好的需要集成的工具有
    shell，python code，
    了解到一个 browse-use的，震惊。



### 新的大致workflow


在 openmanus function calling中加入 rag？




