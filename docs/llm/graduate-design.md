<!--
 * @Author: coffeecat
 * @Date: 2025-03-22 20:53:56
 * @LastEditors: Do not edit
 * @LastEditTime: 2025-03-22 21:42:13
-->
# graduate design

## 数据集

### HumanEval

论文
Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

```bash
git clone https://github.com/openai/human-eval
pip install -e human-eval


```
#### 注意事项

此程序用于运行不受信任的模型生成代码。
**强烈建议用户不要在强大的安全沙箱之外进行此操作。**
在execution.py中的执行调用被故意注释掉，以确保用户在以潜在不安全的方式运行代码之前阅读此免责声明。
有关更多信息和说明，请参阅execution.py中的注释。

#### 数据格式
按照上述说明启用执行后，以如下 JSON Lines（jsonl）格式生成样本并保存，每个样本格式化为单行，如下所示：
```json
{"task_id": "Corresponding HumanEval task ID", "completion": "Completion only without the prompt"}
```

#### 大模型补全示例

我们在data下提供example_problem.jsonl和example_solutions.jsonl以说明格式并帮助调试。
这里是几乎可用的示例代码（你只需提供generate_one_completion使其工作），该代码将生成的补全内容保存到samples.jsonl。

```python
from human_eval.data import write_jsonl, read_problems

problems = read_problems()

def generate_one_completion(prompt):
    pass
    # complete code here

num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)
```
#### 评估生成样本示例

要评估样本，请运行
```bash
$ evaluate_functional_correctness samples.jsonl
Reading samples...
32800it [00:01, 23787.50it/s]
Running test suites...
100%|...| 32800/32800 [16:11<00:00, 33.76it/s]
Writing results to samples.jsonl_results.jsonl...
100%|...| 32800/32800 [00:00<00:00, 42876.84it/s]
{'pass@1': ..., 'pass@10': ..., 'pass@100': ...}
```

虽然评估使用的内存非常少，但当系统内存不足时，你可能会看到以下错误消息。由于这可能会导致一些正确的程序失败，因此我们建议你释放一些内存并再次尝试。
malloc: can't allocate region



此脚本在以_results.jsonl结尾的新文件中提供更细粒度的信息。现在，每一行都包含补全passed以及执行result，执行结果为 “通过”“超时” 或 “失败” 之一。作为快速的完整性检查，示例样本应产生 0.5 的 pass@1。

```bash
$ evaluate_functional_correctness data/example_samples.jsonl --problem_file=data/example_problem.jsonl
Reading samples...
6it [00:00, 3397.11it/s]
Running example suites...
100%|...| 6/6 [00:03<00:00,  1.96it/s]
Writing results to data/example_samples.jsonl_results.jsonl...
100%|...| 6/6 [00:00<00:00, 6148.50it/s]
{'pass@1': 0.4999999999999999}
```
因为当样本数量少于 k 时，没有无偏的方法来估计 pass@k，所以在这些情况下，脚本不会评估 pass@k。要使用其他 k 值进行评估，请传入 --k=<此处为逗号分隔的值>。有关其他选项，请参阅$ evaluate_functional_correctness --help

然而，我们建议您对其余部分使用默认值。




