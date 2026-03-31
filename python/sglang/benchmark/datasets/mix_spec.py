import json
import os
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_prompt,
)


@dataclass
class MixSpecDataset(BaseDataset):
    """Mixed benchmark dataset (GSM8K + HumanEval + MT-Bench + ShareGPT + Medium-length datasets).

    Loads a pre-built JSONL file prepared by benchmark/mix_spec/prepare_mix_dataset.py.
    Each line has: {"prompt": "...", "expected_output_len": <int>, "source": "gsm8k|human_eval|mtbench|sharegpt|hotpotqa|squad|drop|mbpp"}

    Supports padding/truncating prompts to a fixed length like the random dataset.
    """

    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    fixed_output_range_ratio: float
    context_len: Optional[int]
    num_prompts_each: Optional[int]
    random_input_len: Optional[int]
    random_range_ratio: float
    return_text: bool
    source_filter: Optional[List[str]]

    @classmethod
    def from_args(cls, args: Namespace) -> "MixSpecDataset":
        source_filter = None
        if getattr(args, "mix_spec_source", None):
            source_filter = [s.strip() for s in args.mix_spec_source.split(",")]
        # Use --mix-spec-output-len if set, otherwise fall back to --sharegpt-output-len
        fixed_output_len = getattr(args, "mix_spec_output_len", None)
        if fixed_output_len is None:
            fixed_output_len = args.sharegpt_output_len
        return cls(
            dataset_path=args.mix_spec,
            num_requests=args.num_prompts,
            fixed_output_len=fixed_output_len,
            fixed_output_range_ratio=getattr(args, "mix_spec_output_range_ratio", 0.0),
            context_len=args.sharegpt_context_len,
            num_prompts_each=getattr(args, "num_prompts_each", None),
            random_input_len=getattr(args, "mix_spec_input_len", None),
            random_range_ratio=getattr(args, "mix_spec_range_ratio", 0.0),
            return_text=not getattr(args, "tokenize_prompt", False),
            source_filter=source_filter,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id: Optional[str] = None
    ) -> List[DatasetRow]:
        return sample_mix_spec_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            fixed_output_range_ratio=self.fixed_output_range_ratio,
            context_len=self.context_len,
            num_prompts_each=self.num_prompts_each,
            random_input_len=self.random_input_len,
            random_range_ratio=self.random_range_ratio,
            return_text=self.return_text,
            source_filter=self.source_filter,
        )


def sample_mix_spec_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    fixed_output_range_ratio: float = 0.0,
    context_len: Optional[int] = None,
    num_prompts_each: Optional[int] = None,
    random_input_len: Optional[int] = None,
    random_range_ratio: float = 0.0,
    return_text: bool = True,
    source_filter: Optional[List[str]] = None,
) -> List[DatasetRow]:
    """Load a pre-built mix-spec dataset from a JSONL file.

    The dataset should be prepared using benchmark/mix_spec/prepare_mix_dataset.py.

    Args:
        dataset_path: Path to the JSONL file.
        num_requests: Total number of prompts to sample (used when num_prompts_each is None).
        tokenizer: Tokenizer for computing prompt lengths.
        fixed_output_len: If set, override the expected_output_len from the dataset.
        fixed_output_range_ratio: Range ratio for output length variation when fixed_output_len is set.
        context_len: If set, skip sequences where prompt_len + output_len exceeds this.
        num_prompts_each: If set, sample this many prompts from each source dataset
            (gsm8k, human_eval, mtbench, sharegpt) in order. Overrides num_requests.
        random_input_len: If set, pad/truncate prompts to this length (similar to random dataset).
        random_range_ratio: Range ratio for input length variation when random_input_len is set.
        return_text: If True, return text prompts; otherwise return token IDs.
        source_filter: If set, only sample from the specified source datasets.
    """
    if not dataset_path or not os.path.isfile(dataset_path):
        raise ValueError(
            f"mix-spec dataset path is invalid: '{dataset_path}'. "
            "Please prepare the dataset first using: "
            "python benchmark/mix_spec/prepare_mix_dataset.py --output mix_spec_dataset.jsonl "
            "and then pass the path via --mix-spec."
        )

    print(f"Loading mix-spec dataset from {dataset_path}")

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            # Filter by source if specified
            if source_filter is not None:
                source = record.get("source", "unknown")
                # Normalize source names for matching
                source_normalized = source.lower().replace("-", "_").replace(" ", "_")
                filter_normalized = [
                    s.lower().replace("-", "_").replace(" ", "_") for s in source_filter
                ]
                if source_normalized not in filter_normalized:
                    continue
            records.append(record)

    print(f"Total records in mix-spec dataset: {len(records)}")
    if source_filter:
        print(f"  Filtered by source: {source_filter}")

    if num_prompts_each is not None:
        # Group records by source, preserving order within each group
        source_records: Dict[str, List[dict]] = {}
        for record in records:
            source = record.get("source", "unknown")
            if source not in source_records:
                source_records[source] = []
            source_records[source].append(record)

        print(f"Sampling {num_prompts_each} prompts from each source dataset")
        for source, recs in source_records.items():
            print(f"  - {source}: {len(recs)} available")

        # Take num_prompts_each from each source in order, then concatenate
        selected_records = []
        for source in source_records:
            selected_records.extend(source_records[source][:num_prompts_each])
        records = selected_records
        num_requests = len(records)

    # Process records into DatasetRow list
    filtered_dataset: List[DatasetRow] = []
    source_counts: Dict[str, int] = {}

    # Compute input lengths if random_input_len is specified
    if random_input_len is not None:
        input_lens = compute_random_lens(
            full_len=random_input_len,
            range_ratio=random_range_ratio,
            num=num_requests,
        )
        if return_text:
            # Need to truncate input_len as server encode will add special token.
            num_special_tokens = int(tokenizer.num_special_tokens_to_add())
            for i in range(num_requests):
                input_lens[i] = max(1, input_lens[i] - num_special_tokens)

    # Compute output lengths if fixed_output_len is specified with range_ratio
    if fixed_output_len is not None and fixed_output_range_ratio > 0:
        output_lens = compute_random_lens(
            full_len=fixed_output_len,
            range_ratio=fixed_output_range_ratio,
            num=num_requests,
        )
    else:
        output_lens = None

    for i, record in enumerate(records):
        if len(filtered_dataset) >= num_requests:
            break

        prompt = record["prompt"]
        source = record.get("source", "unknown")

        # Determine output length
        if fixed_output_len is not None:
            if output_lens is not None:
                # Use randomly sampled output length with range ratio
                output_len = output_lens[len(filtered_dataset)]
            else:
                # Use fixed output length
                output_len = fixed_output_len
        else:
            # Use dataset's expected_output_len
            output_len = record.get("expected_output_len", 512)

        if random_input_len is not None:
            # Pad/truncate prompt to target length like random dataset
            target_len = input_lens[len(filtered_dataset)]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            if prompt_len > target_len:
                # Truncate
                input_ids = prompt_token_ids[:target_len]
            else:
                # Repeat to reach target length
                if prompt_len == 0:
                    # Generate random tokens if prompt is empty
                    input_ids = [
                        int(x)
                        for x in np.random.randint(
                            0, tokenizer.vocab_size, size=target_len
                        )
                    ]
                else:
                    ratio = (target_len + prompt_len - 1) // prompt_len
                    input_ids = (prompt_token_ids * ratio)[:target_len]

            prompt_len = len(input_ids)
            if return_text:
                prompt = tokenizer.decode(input_ids)
            else:
                prompt = input_ids
        else:
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

        if prompt_len < 2 or output_len < 2:
            continue

        if context_len and prompt_len + output_len > context_len:
            continue

        filtered_dataset.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"Filtered to {len(filtered_dataset)} samples")
    for source, count in source_counts.items():
        print(f"  - {source}: {count}")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")

    return filtered_dataset
