
# evalplus
[evalplus github](https://github.com/evalplus/evalplus) 

EvalPlus 是 LLM4code的严谨的评估框架
支持的数据集有
1) humaneval+：比起原始humaneval多了80倍测试用例
2) mbpp+：比原始mbpp多了35倍测试用例
3) evalperf：评估生成代码的efficiency  高效性。

## 使用evalplus评测humaneval和mbpp
```bash
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"
# Or `pip install "evalplus[vllm]" --upgrade` for the latest stable release

evalplus.evaluate --model "ise-uiuc/Magicoder-S-DS-6.7B" \
                  --dataset [humaneval|mbpp]             \
                  --backend vllm                         \
                  --greedy
```

注意：然后这里是定制化代码生成与评测

### 代码生成 
>首先请注意problem的格式为：
> task_id
> entry_point   -> 函数名
> **prompt  -> 带着docstring的函数签名**
> canonical_solution  -> 标准答案（重新写的、修复了HumanEval的bug）
> base_input -> 原始humaneval的测试用例输入
> plus_input -> evalplus的增强输入

```python
from evalplus.data import get_[human_eval|mbpp]_plus, write_jsonl

def GEN_SIKUTION(prompt):
    ...

samples = [
    dict(task_id=task_id, solution=GEN_SOLUTION(problem["prompt"]))
    for task_id, problem in get_[human_eval|mbpp]_plus().items()
]

write_jsonl("samples.jsonl", samples)
```

>然后请注意待检测code generation文件的格式
>task_id -> 保持不动
>solution与completion二选一
>> solution包括prompt，如{"task_id": "HumanEval/?", "solution": "def f():\n    return 1"}
>> completion不包括，如{"task_id": "HumanEval/?", "completion": "    return 1"}

### 代码清洗

llm生成的code可能包括自然语言或者额外的不需要的代码，该工具为此而来

```bash
# 💡 If you are storing codes in jsonl:
evalplus.sanitize --samples samples.jsonl
# Sanitized code will be produced to `samples-sanitized.jsonl`

```

然后可以使用 evalplus.syncheck 检测清洗有效性，该命令将打印出代码片段并报告为什么他们错误
```bash
# 💡 If you are storing codes in jsonl:
evalplus.syncheck --samples samples.jsonl --dataset [humaneval|mbpp]
```

### 代码评测

首先是直接执行，但是执行后发现后台会存在无限循环死进程
```bash
evalplus.evaluate --dataset [humaneval|mbpp] --samples samples.jsonl
```

然后是安全的docker执行
```bash
docker run --rm --pull=always -v $(pwd)/evalplus_results:/app ganler/evalplus:latest \
           evalplus.evaluate --dataset humaneval                                     \
           --samples /app/humaneval/ise-uiuc--Magicoder-S-DS-6.7B_vllm_temp_0.0.jsonl
```

上述评估输出应如下：
```bash
Computing expected output...
Expected outputs computed in 15.18s
Reading samples...
164it [00:04, 37.79it/s]
Evaluating samples...
100%|██████████████████████████████████████████| 164/164 [00:03<00:00, 44.75it/s]
Base
{'pass@1': 0.8841463414634146}
Base + Extra
{'pass@1': 0.768}
```
***Base*** is the pass@k for the ***original HumanEval***

***Base + Extra*** is the pass@k for the our ***HumanEval+ (with extra tests)***

***The "k" includes [1, 10, 100] where k values <= the sample size will be used***

**A cache file** named like samples_eval_results.jsonl will be cached. **Remove it to re-run the evaluation**


#### 使用自定义数据集
```bash
HUMANEVAL_OVERRIDE_PATH="/path/to/HumanEvalPlus.jsonl.gz" evalplus.evaluate --dataset humaneval --samples samples.jsonl
```

### code

