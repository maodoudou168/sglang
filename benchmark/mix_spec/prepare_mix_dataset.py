"""
Prepare a mixed benchmark dataset by downloading and integrating
GSM8K, HumanEval, MT-Bench, and ShareGPT datasets into a single JSONL file.

The data order is: GSM8K -> HumanEval -> MT-Bench -> ShareGPT (not shuffled).

Usage:
    python prepare_mix_dataset.py --output mix_spec_dataset.jsonl

    # Control per-dataset sample count
    python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
        --num-gsm8k 200 --num-humaneval 164 --num-mtbench 80 --num-sharegpt 500

    # Use locally downloaded files (when remote URLs are not accessible)
    python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
        --gsm8k-path /path/to/test.jsonl \
        --humaneval-path /path/to/HumanEval.jsonl \
        --mtbench-path /path/to/question.jsonl \
        --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json
"""

import argparse
import gzip
import json
import os
import sys

import requests
from tqdm import tqdm

# Dataset download URLs
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
MTBENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Default cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


def download_file(url, filename, desc=None):
    """Download a file from a URL with progress bar. Returns the local path."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        print(f"  [Cached] {filename}")
        return filename

    print(f"  Downloading from {url}")
    print(f"  Saving to {filename}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except (requests.RequestException, ConnectionError) as e:
        print(f"\n  ERROR: Failed to download from {url}")
        print(f"  Error details: {e}")
        print(f"  Please manually download the file and place it at: {filename}")
        print(f"  See README.md for manual download instructions.")
        sys.exit(1)

    total_size = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as f, tqdm(
        desc=desc or os.path.basename(filename),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def load_gsm8k(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load GSM8K dataset and convert to unified format."""
    print("\n[1/4] Loading GSM8K dataset...")
    if not path or not os.path.exists(path):
        path = download_file(
            GSM8K_URL,
            os.path.join(cache_dir, "gsm8k_test.jsonl"),
            desc="GSM8K",
        )

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            prompt = "Question: " + data["question"] + "\nAnswer:"
            records.append(
                {
                    "prompt": prompt,
                    "expected_output_len": 512,
                    "source": "gsm8k",
                }
            )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} GSM8K samples")
    return records


def load_humaneval(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load HumanEval dataset and convert to unified format."""
    print("\n[2/4] Loading HumanEval dataset...")
    if not path or not os.path.exists(path):
        gz_path = os.path.join(cache_dir, "HumanEval.jsonl.gz")
        jsonl_path = os.path.join(cache_dir, "HumanEval.jsonl")

        # Check if already decompressed
        if os.path.exists(jsonl_path):
            path = jsonl_path
        else:
            download_file(HUMANEVAL_URL, gz_path, desc="HumanEval")
            # Decompress .gz file
            print("  Decompressing HumanEval.jsonl.gz...")
            with gzip.open(gz_path, "rt", encoding="utf-8") as gz_f:
                with open(jsonl_path, "w", encoding="utf-8") as out_f:
                    out_f.write(gz_f.read())
            path = jsonl_path

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            # Use the prompt field (function signature + docstring)
            prompt = (
                "Read the following function signature and docstring, "
                "and fully implement the function described. "
                "Your response should only contain the code for this function.\n\n"
                + data["prompt"]
            )
            records.append(
                {
                    "prompt": prompt,
                    "expected_output_len": 512,
                    "source": "human_eval",
                }
            )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} HumanEval samples")
    return records


def load_mtbench(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load MT-Bench dataset and convert to unified format."""
    print("\n[3/4] Loading MT-Bench dataset...")
    if not path or not os.path.exists(path):
        path = download_file(
            MTBENCH_URL,
            os.path.join(cache_dir, "mt_bench_question.jsonl"),
            desc="MT-Bench",
        )

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            # Use the first turn question as prompt
            prompt = data["turns"][0]
            records.append(
                {
                    "prompt": prompt,
                    "expected_output_len": 256,
                    "source": "mtbench",
                }
            )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} MT-Bench samples")
    return records


def load_sharegpt(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load ShareGPT dataset and convert to unified format."""
    print("\n[4/4] Loading ShareGPT dataset...")
    if not path or not os.path.exists(path):
        path = download_file(
            SHAREGPT_URL,
            os.path.join(cache_dir, "ShareGPT_V3_unfiltered_cleaned_split.json"),
            desc="ShareGPT",
        )

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Filter out conversations with less than 2 turns
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]

    records = []
    for data in dataset:
        convos = data.get("conversations", data.get("conversation", []))
        prompt = convos[0]["value"]
        completion = convos[1]["value"]

        if not prompt or not completion:
            continue

        # Estimate output length based on completion text
        # Use character count / 4 as a rough token estimate
        estimated_output_len = max(len(completion) // 4, 16)

        records.append(
            {
                "prompt": prompt,
                "expected_output_len": estimated_output_len,
                "source": "sharegpt",
            }
        )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} ShareGPT samples")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mixed benchmark dataset (GSM8K + HumanEval + MT-Bench + ShareGPT)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mix_spec_dataset.jsonl"
        ),
        help="Output JSONL file path. Default: ./mix_spec_dataset.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=CACHE_DIR,
        help="Directory to cache downloaded datasets.",
    )

    # Per-dataset sample count
    parser.add_argument(
        "--num-gsm8k",
        type=int,
        default=None,
        help="Number of GSM8K samples to include. Default: all (1319 test samples).",
    )
    parser.add_argument(
        "--num-humaneval",
        type=int,
        default=None,
        help="Number of HumanEval samples to include. Default: all (164 samples).",
    )
    parser.add_argument(
        "--num-mtbench",
        type=int,
        default=None,
        help="Number of MT-Bench samples to include. Default: all (80 samples).",
    )
    parser.add_argument(
        "--num-sharegpt",
        type=int,
        default=None,
        help="Number of ShareGPT samples to include. Default: all.",
    )

    # Per-dataset local file paths (for manual download)
    parser.add_argument(
        "--gsm8k-path",
        type=str,
        default="",
        help="Path to locally downloaded GSM8K test.jsonl.",
    )
    parser.add_argument(
        "--humaneval-path",
        type=str,
        default="",
        help="Path to locally downloaded HumanEval.jsonl (decompressed).",
    )
    parser.add_argument(
        "--mtbench-path",
        type=str,
        default="",
        help="Path to locally downloaded MT-Bench question.jsonl.",
    )
    parser.add_argument(
        "--sharegpt-path",
        type=str,
        default="",
        help="Path to locally downloaded ShareGPT JSON file.",
    )

    args = parser.parse_args()

    cache_dir = args.cache_dir

    print("=" * 60)
    print("Preparing mixed benchmark dataset (mix-spec)")
    print("Order: GSM8K -> HumanEval -> MT-Bench -> ShareGPT")
    print("=" * 60)

    # Load each dataset
    gsm8k_records = load_gsm8k(args.gsm8k_path, args.num_gsm8k, cache_dir)
    humaneval_records = load_humaneval(args.humaneval_path, args.num_humaneval, cache_dir)
    mtbench_records = load_mtbench(args.mtbench_path, args.num_mtbench, cache_dir)
    sharegpt_records = load_sharegpt(args.sharegpt_path, args.num_sharegpt, cache_dir)

    # Concatenate in order (no shuffling)
    all_records = gsm8k_records + humaneval_records + mtbench_records + sharegpt_records

    # Write to output JSONL
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Mixed dataset written to: {args.output}")
    print(f"Total samples: {len(all_records)}")
    print(f"  - GSM8K:     {len(gsm8k_records)}")
    print(f"  - HumanEval: {len(humaneval_records)}")
    print(f"  - MT-Bench:  {len(mtbench_records)}")
    print(f"  - ShareGPT:  {len(sharegpt_records)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
