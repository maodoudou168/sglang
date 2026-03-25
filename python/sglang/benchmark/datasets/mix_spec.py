import json
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class MixSpecDataset(BaseDataset):
    """Mixed benchmark dataset (GSM8K + HumanEval + MT-Bench + ShareGPT).

    Loads a pre-built JSONL file prepared by benchmark/mix_spec/prepare_mix_dataset.py.
    Each line has: {"prompt": "...", "expected_output_len": <int>, "source": "gsm8k|human_eval|mtbench|sharegpt"}
    """

    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    context_len: Optional[int]
    num_prompts_each: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "MixSpecDataset":
        return cls(
            dataset_path=args.mix_spec,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            num_prompts_each=getattr(args, "num_prompts_each", None),
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id: Optional[str] = None
    ) -> List[DatasetRow]:
        return sample_mix_spec_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            context_len=self.context_len,
            num_prompts_each=self.num_prompts_each,
        )


def sample_mix_spec_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    num_prompts_each: Optional[int] = None,
) -> List[DatasetRow]:
    """Load a pre-built mix-spec dataset from a JSONL file.

    The dataset should be prepared using benchmark/mix_spec/prepare_mix_dataset.py.

    Args:
        dataset_path: Path to the JSONL file.
        num_requests: Total number of prompts to sample (used when num_prompts_each is None).
        tokenizer: Tokenizer for computing prompt lengths.
        fixed_output_len: If set, override the expected_output_len from the dataset.
        context_len: If set, skip sequences where prompt_len + output_len exceeds this.
        num_prompts_each: If set, sample this many prompts from each source dataset
            (gsm8k, human_eval, mtbench, sharegpt) in order. Overrides num_requests.
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
            records.append(json.loads(line))

    print(f"Total records in mix-spec dataset: {len(records)}")

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

    for record in records:
        if len(filtered_dataset) >= num_requests:
            break

        prompt = record["prompt"]
        source = record.get("source", "unknown")
        output_len = (
            fixed_output_len
            if fixed_output_len is not None
            else record.get("expected_output_len", 512)
        )

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
