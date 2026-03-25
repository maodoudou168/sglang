# Mix-Spec: Mixed Benchmark Dataset

A mixed benchmark dataset combining **GSM8K**, **HumanEval**, **MT-Bench**, and **ShareGPT** for comprehensive serving performance evaluation.

Data order: GSM8K -> HumanEval -> MT-Bench -> ShareGPT (data is **not** shuffled).

## Quick Start

### 1. Generate the mixed dataset (auto-download)

```bash
cd benchmark/mix_spec
python prepare_mix_dataset.py --output mix_spec_dataset.jsonl
```

- specify cached data path and number of data to be integrated

```bash
python prepare_mix_dataset.py --output ../../../../dataset/mix_spec/mix_spec_dataset.jsonl --sharegpt-path /home/local/workspace/code/dataset/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json --num-gsm8k 80 --num-humaneval 80 --num-mtbench 80 --num-sharegpt 80
```

### 2. Run benchmark with mix-spec

```bash
# Start the server first
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000

# Run benchmark with mix-spec dataset
python3 -m sglang.bench_serving --backend sglang --dataset-name mix-spec --mix-spec /home/local/workspace/code/dataset/mix_spec/mix_spec_dataset.jsonl --num-prompts 320 --port 62222 --request-rate 1.0 --max-concurrency 64

# Run benchmark with mix-spec dataset, control number of each kind of data
python3 -m sglang.bench_serving --backend sglang --dataset-name mix-spec --mix-spec /home/local/workspace/code/dataset/mix_spec/mix_spec_dataset.jsonl --num-prompts-each 32 --port 62222 --max-concurrency $BATCH_SIZE
```

### 3. Control per-dataset sample count

```bash
python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
    --num-gsm8k 200 --num-humaneval 164 --num-mtbench 80 --num-sharegpt 500
```

## Manual Download Instructions

If automatic download fails (e.g., network restrictions), you can manually download each dataset and specify local paths.

### GSM8K

- **URL**: https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
- **Format**: JSONL, each line: `{"question": "...", "answer": "..."}`
- **Size**: 1319 test samples

```bash
wget -O data_cache/gsm8k_test.jsonl \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```

### HumanEval

- **URL**: https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
- **Format**: Gzipped JSONL, each line: `{"task_id": "...", "prompt": "...", "entry_point": "...", "canonical_solution": "...", "test": "..."}`
- **Size**: 164 programming tasks

```bash
# Download and decompress
wget -O data_cache/HumanEval.jsonl.gz \
    https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
gunzip -k data_cache/HumanEval.jsonl.gz
# Result: data_cache/HumanEval.jsonl
```

### MT-Bench

- **URL**: https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
- **Format**: JSONL, each line: `{"question_id": <int>, "turns": ["<turn1>", "<turn2>"], ...}`
- **Size**: 80 multi-turn questions

```bash
wget -O data_cache/mt_bench_question.jsonl \
    https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
```

### ShareGPT

- **URL**: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
- **Format**: JSON array, each element: `{"conversations": [{"value": "..."}, {"value": "..."}, ...]}`
- **Size**: ~90K conversations

```bash
wget -O data_cache/ShareGPT_V3_unfiltered_cleaned_split.json \
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Using locally downloaded files

After manual download, run the prepare script with local paths:

```bash
python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
    --gsm8k-path data_cache/gsm8k_test.jsonl \
    --humaneval-path data_cache/HumanEval.jsonl \
    --mtbench-path data_cache/mt_bench_question.jsonl \
    --sharegpt-path data_cache/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Output Format

The generated `mix_spec_dataset.jsonl` is a JSONL file where each line is:

```json
{"prompt": "...", "expected_output_len": 512, "source": "gsm8k"}
```

Fields:
- `prompt` (str): The input prompt text.
- `expected_output_len` (int): Expected output token length for benchmarking.
  - GSM8K: 512
  - HumanEval: 512
  - MT-Bench: 256
  - ShareGPT: estimated from original completion length
- `source` (str): Dataset origin, one of `gsm8k`, `human_eval`, `mtbench`, `sharegpt`.

## File Structure

```
benchmark/mix_spec/
├── README.md                    # This file
├── prepare_mix_dataset.py       # Dataset download and preparation script
├── mix_spec_dataset.jsonl       # Generated mixed dataset (after running the script)
└── data_cache/                  # Cached raw datasets (auto-created)
    ├── gsm8k_test.jsonl
    ├── HumanEval.jsonl.gz
    ├── HumanEval.jsonl
    ├── mt_bench_question.jsonl
    └── ShareGPT_V3_unfiltered_cleaned_split.json
```
