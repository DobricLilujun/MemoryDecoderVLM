#!/usr/bin/env python
"""
Prepare text-only dataset for Memory Decoder training.

This script reads the metadata produced by vlm_save_embed.py and creates a
text-only HuggingFace dataset with:
  - input_ids, labels, attention_mask  (text-only, no image tokens)
  - dstore_range                       (pointer into the KNN datastore)

The dstore_range column aligns each example to its entries in the KNN
datastore (which was built from the VLM's hidden states).

Usage:
    python vlm_prepare_memdec_data.py \
        --metadata_path ./vlm_dstore/metadata_train.json \
        --output_dir    ./vlm_memdec_data
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from loguru import logger

from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare text-only dataset for MemDec training")
    parser.add_argument('--vlm_model', type=str,
                        default='Qwen/Qwen2-VL-2B-Instruct',
                        help='VLM model (used for tokenizer consistency)')
    parser.add_argument('--dataset', type=str,
                        default='jmhessel/newyorker_caption_contest')
    parser.add_argument('--dataset_config', type=str, default='explanation')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata JSON from vlm_save_embed.py')
    parser.add_argument('--output_dir', type=str,
                        default='./vlm_memdec_data')
    parser.add_argument('--max_samples', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load metadata from VLM embedding step -------------------------
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)
    per_example_counts = metadata['per_example_counts']
    logger.info(f"Loaded metadata: {metadata['total_tokens']} total tokens "
                f"from {metadata['num_examples']} examples")

    # ---- Load tokenizer (same family as VLM for token consistency) -----
    processor = AutoProcessor.from_pretrained(args.vlm_model)
    tokenizer = (processor.tokenizer
                 if hasattr(processor, 'tokenizer')
                 else AutoTokenizer.from_pretrained(args.vlm_model))

    # ---- Load raw dataset ----------------------------------------------
    dataset = load_dataset(
        args.dataset, args.dataset_config, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    assert len(dataset) == len(per_example_counts), (
        f"Dataset size ({len(dataset)}) != metadata examples "
        f"({len(per_example_counts)})")

    # ---- Process each example ------------------------------------------
    all_input_ids = []
    all_labels = []
    all_attention_mask = []
    all_dstore_range = []

    dstore_idx = 0
    skipped = 0
    mismatched = 0

    for idx in tqdm(range(len(dataset)), desc="Preparing MemDec data"):
        count = per_example_counts[idx]

        if count == 0:
            # Example was skipped during embedding saving
            continue

        example = dataset[idx]
        desc = example.get('image_description', '')

        # ---- text-only conversation (same prompt, no image) ----
        messages = [
            {"role": "user", "content": "Describe what you see in this cartoon."},
            {"role": "assistant", "content": desc},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

        # ---- prompt-only (to find response boundary) ----
        prompt_messages = [
            {"role": "user", "content": "Describe what you see in this cartoon."},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)

        # ---- tokenise ----
        full_enc = tokenizer(full_text, add_special_tokens=False)
        prompt_enc = tokenizer(prompt_text, add_special_tokens=False)

        full_ids = full_enc['input_ids']
        prompt_len = len(prompt_enc['input_ids'])

        # ---- create labels: -100 for prompt, real IDs for response ----
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        # ---- verify response-token count matches VLM ----
        # After shift: shifted_labels = labels[1:]
        # Non-pad count = number of entries where shifted_labels != -100
        shifted_labels = labels[1:]
        num_response_tokens = sum(1 for x in shifted_labels if x != -100)

        if num_response_tokens != count:
            logger.debug(
                f"Example {idx}: VLM count={count}, "
                f"text count={num_response_tokens} — skipping")
            mismatched += 1
            dstore_idx += count   # advance dstore pointer regardless
            continue

        all_input_ids.append(full_ids)
        all_labels.append(labels)
        all_attention_mask.append([1] * len(full_ids))
        all_dstore_range.append([dstore_idx, dstore_idx + count])

        dstore_idx += count

    logger.info(
        f"Created {len(all_input_ids)} examples "
        f"(skipped {skipped} empty, {mismatched} mismatched)")

    # ---- Save as HuggingFace dataset -----------------------------------
    ds = Dataset.from_dict({
        'input_ids': all_input_ids,
        'labels': all_labels,
        'attention_mask': all_attention_mask,
        'dstore_range': all_dstore_range,
    })
    ds.save_to_disk(args.output_dir)

    # Save a summary for reference
    summary = {
        'num_examples': len(ds),
        'total_dstore_tokens': dstore_idx,
        'mismatched': mismatched,
    }
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved dataset ({len(ds)} examples) to {args.output_dir}")
    logger.info(f"Total dstore tokens: {dstore_idx}")


if __name__ == '__main__':
    main()
