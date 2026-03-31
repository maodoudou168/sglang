# Mix-Spec: Mixed Benchmark Dataset

A mixed benchmark dataset combining **GSM8K**, **HumanEval**, **MT-Bench**, **ShareGPT**, and medium-length datasets for comprehensive serving performance evaluation.

Data order: GSM8K -> HumanEval -> MT-Bench -> ShareGPT -> HotpotQA -> SQuAD -> DROP -> MBPP (data is **not** shuffled).

## Features

- **Multiple task types**: Math (GSM8K), Code (HumanEval, MBPP), Reasoning (HotpotQA), Reading Comprehension (SQuAD), Numerical Reasoning (DROP), General chat (MT-Bench, ShareGPT)
- **Flexible input length control**: Pad/truncate prompts to fixed length like the random dataset
- **Source filtering**: Benchmark only specific datasets
- **Medium-length support**: Datasets with 1k-10k token contexts for testing medium-length scenarios

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

```bash
python prepare_mix_dataset.py --output /home/local/workspace/code/dataset/mix_spec/mix_spec_dataset_long.jsonl --num-gsm8k 80 --num-humaneval 80 --num-hotpotqa 80 --num-mtbench 80 --num-sharegpt 80 --num-squad 80 --num-drop 80 --num-mbpp 80 --sharegpt-path /home/local/workspace/code/dataset/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
```

### 2. Run benchmark with mix-spec

```bash
# Start the server first
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000

# Run benchmark with mix-spec dataset
python3 -m sglang.bench_serving --backend sglang --dataset-name mix-spec --mix-spec /home/local/workspace/code/dataset/mix_spec/mix_spec_dataset.jsonl --num-prompts 320 --port 62222 --request-rate 1.0 --max-concurrency 64

# Run benchmark with mix-spec dataset, control number of each kind of data
python3 -m sglang.bench_serving --backend sglang --dataset-name mix-spec --mix-spec /home/local/workspace/code/dataset/mix_spec/mix_spec_dataset.jsonl --num-prompts-each 32 --port 62222 --max-concurrency $BATCH_SIZE

# Run benchmark with fixed input length (pad/truncate to 2048 tokens)
python3 -m sglang.bench_serving --backend sglang --dataset-name mix-spec --mix-spec /path/to/mix_spec_dataset.jsonl --mix-spec-input-len 2048 --mix-spec-range-ratio 0.5

# Run benchmark only on specific source datasets
python3 -m sglang.bench_serving --backend sglang --dataset-name mix-spec --mix-spec /path/to/mix_spec_dataset.jsonl --mix-spec-source hotpotqa,squad --num-prompts-each 80
# This will sample 80 prompts from hotpotqa and 80 from squad (160 total)
```

### 3. Control per-dataset sample count

```bash
python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
    --num-gsm8k 200 --num-humaneval 164 --num-mtbench 80 --num-sharegpt 500
```

### 4. Include medium-length datasets (1k-10k tokens)

```bash
# Include all medium-length datasets
python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
    --include-medium-length

# Or selectively include specific medium-length datasets
python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
    --num-hotpotqa 100 --num-squad 100 --num-drop 50 --num-mbpp 50

# Only include specific datasets (e.g., only reasoning and code tasks)
python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
    --datasets gsm8k,humaneval,hotpotqa,mbpp
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

### Medium-Length Datasets (1k-10k tokens)

These datasets provide longer context for testing medium-length scenarios:

| Dataset | Type | Length Range | Coverage | Download Source |
|---------|------|--------------|----------|-----------------|
| **HotpotQA** | Multi-hop reasoning QA | 1k-5k tokens | Reasoning | Direct URL (CMU) |
| **SQuAD** | Reading comprehension | 1k-3k tokens | QA | Direct URL (Stanford) |
| **DROP** | Numerical reasoning | 1k-4k tokens | Math/Reasoning | Direct URL (AWS S3) |
| **MBPP** | Python code generation | 500-2k tokens | Code | Direct URL (GitHub) |

**Note**: All medium-length datasets can be downloaded directly without HuggingFace.

### Using locally downloaded files

After manual download, run the prepare script with local paths:

```bash
# Basic datasets with local files
python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
    --gsm8k-path data_cache/gsm8k_test.jsonl \
    --humaneval-path data_cache/HumanEval.jsonl \
    --mtbench-path data_cache/mt_bench_question.jsonl \
    --sharegpt-path data_cache/ShareGPT_V3_unfiltered_cleaned_split.json

# Include medium-length datasets with local files
python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
    --include-medium-length \
    --hotpotqa-path data_cache/hotpot_dev_fullwiki_v1.json \
    --squad-path data_cache/dev-v2.0.json \
    --drop-path data_cache/drop_dataset/ \
    --mbpp-path data_cache/mbpp.jsonl
```

**Supported local path parameters:**

| Parameter | Description | File Format |
|-----------|-------------|-------------|
| `--gsm8k-path` | GSM8K test set | JSONL file |
| `--humaneval-path` | HumanEval dataset | JSONL file (decompressed) |
| `--mtbench-path` | MT-Bench questions | JSONL file |
| `--sharegpt-path` | ShareGPT dataset | JSON file |
| `--hotpotqa-path` | HotpotQA dataset | JSON file |
| `--squad-path` | SQuAD dataset | JSON file (dev-v2.0.json) |
| `--drop-path` | DROP dataset | Directory with JSON files |
| `--mbpp-path` | MBPP dataset | JSONL file |

**Manual download instructions:**

If automatic download fails, you can manually download datasets from these URLs:

```bash
# HotpotQA
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json

# SQuAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# DROP (requires unzip)
wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip
unzip drop_dataset.zip

# MBPP
wget https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
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
  - HotpotQA: estimated from answer length
  - SQuAD: estimated from answer length
  - DROP: estimated from answer length
  - MBPP: estimated from reference code length
- `source` (str): Dataset origin, one of `gsm8k`, `human_eval`, `mtbench`, `sharegpt`, `hotpotqa`, `squad`, `drop`, `mbpp`.

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
    ├── ShareGPT_V3_unfiltered_cleaned_split.json
    ├── hotpot_dev_fullwiki_v1.json
    ├── dev-v2.0.json
    ├── drop_dataset/
    └── mbpp.jsonl
```

## Benchmark Parameters

### Input Length Control

Similar to the `random` dataset, mix-spec supports fixed input length:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mix-spec-input-len` | Fixed input length (pad/truncate prompts to this length) | None (use original prompt length) |
| `--mix-spec-range-ratio` | Range ratio for input length variation | 0.0 |
| `--mix-spec-source` | Comma-separated list of source datasets to sample from. Available: `gsm8k`, `human_eval`, `mtbench`, `sharegpt`, `hotpotqa`, `squad`, `drop`, `mbpp` | None (all sources) |

### Output Length Control

Similar to the `random` dataset, mix-spec supports fixed output length:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mix-spec-output-len` | Fixed output length (overrides dataset's expected_output_len) | None (use dataset's expected_output_len) |
| `--mix-spec-output-range-ratio` | Range ratio for output length variation | 0.0 |

**Note**: `--mix-spec-output-len` takes precedence over `--sharegpt-output-len`. If neither is set, the dataset's original `expected_output_len` is used.

### Examples

```bash
# Fixed input length with 50% variation
python3 -m sglang.bench_serving \
    --dataset-name mix-spec \
    --mix-spec /path/to/mix_spec_dataset.jsonl \
    --mix-spec-input-len 4096 \
    --mix-spec-range-ratio 0.5

# Fixed output length (all requests generate 1024 tokens)
python3 -m sglang.bench_serving \
    --dataset-name mix-spec \
    --mix-spec /path/to/mix_spec_dataset.jsonl \
    --mix-spec-output-len 1024

# Fixed output length with 30% variation (e.g., 700-1300 tokens)
python3 -m sglang.bench_serving \
    --dataset-name mix-spec \
    --mix-spec /path/to/mix_spec_dataset.jsonl \
    --mix-spec-output-len 1024 \
    --mix-spec-output-range-ratio 0.3

# Combined input and output length control
python3 -m sglang.bench_serving \
    --dataset-name mix-spec \
    --mix-spec /path/to/mix_spec_dataset.jsonl \
    --mix-spec-input-len 2048 \
    --mix-spec-range-ratio 0.5 \
    --mix-spec-output-len 1024 \
    --mix-spec-output-range-ratio 0.3

# Only benchmark on reasoning and code tasks
python3 -m sglang.bench_serving \
    --dataset-name mix-spec \
    --mix-spec /path/to/mix_spec_dataset.jsonl \
    --mix-spec-source hotpotqa,drop,mbpp \
    --num-prompts-each 100

# Combine with other bench_serving options
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name mix-spec \
    --mix-spec /path/to/mix_spec_dataset.jsonl \
    --mix-spec-input-len 2048 \
    --mix-spec-source hotpotqa,squad \
    --num-prompts-each 50 \
    --request-rate 2.0 \
    --max-concurrency 32
```
