# EDAPIBench-Construction

A streamlined pipeline for constructing **EDAPIBench**

## Step 1: Prepare LLMs
Download the required code models from huggingface. You can download them manually or via the huggingface `transformers` library.

### Required Models
| Model Name | Link |
|------------|------------------------|
| DeepSeek Coder 1.3B Base | [deepseek-ai/deepseek-coder-1.3b-base](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) |
| StarCoder2 3B | [bigcode/starcoder2-3b](https://huggingface.co/bigcode/starcoder2-3b) |
| Qwen2.5 Coder 3B | [Qwen/Qwen2.5-Coder-3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) |

## Step 2: Code Search
Retrieve relevant code snippets from open-source repositories via the automated search pipeline.
```bash
sh code-search.sh
```

## Step 3: Deprecated API Sample Collection
Collect code snippets that are completed as deprecated APIs by LLMs.
```bash
sh dapi-collection.sh
```

## Step 4: Generate Evaluation Data
Construct three key evaluation data dimensions (Portability, Generalization, Specificity) for benchmarking:
```bash
sh eval-data-construction.sh
```

> You can either execute the bash script, or directly run the .py scripts following the workflow outlined in the bash script. Some tokens or keys (e.g., OpenAI key) in the scripts need to be provided by yourself.