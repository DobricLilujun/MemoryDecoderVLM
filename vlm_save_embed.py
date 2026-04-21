#!/usr/bin/env python
"""
Save hidden-state embeddings from Qwen2-VL for building a KNN datastore.

Pipeline: Raw Dataset → VLM Forward → Capture last-FFN-input activations → Save (keys, vals)

The keys are the last-FFN-input activations at each response-token position.
The vals are the next-token IDs at each response-token position.

Output:
  - Arrow file: (keys, vals) for all response tokens across the dataset
  - JSON metadata: per-example token counts (for dstore_range computation later)

Usage:
    python vlm_save_embed.py --split train --output_dir ./vlm_dstore
    python vlm_save_embed.py --split train --output_dir ./vlm_dstore --max_samples 100
"""

import os
import json
import argparse
import torch
import numpy as np
import pyarrow as pa
from tqdm import tqdm
from loguru import logger

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Save VLM embeddings for KNN datastore")
    parser.add_argument('--vlm_model', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--dataset', type=str, default='jmhessel/newyorker_caption_contest')
    parser.add_argument('--dataset_config', type=str, default='explanation')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_dir', type=str, default='./vlm_dstore')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of examples (for debugging)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load VLM model ------------------------------------------------
    logger.info(f"Loading VLM: {args.vlm_model}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.vlm_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.vlm_model)
    model.eval()

    hidden_dim = model.config.text_config.hidden_size
    logger.info(f"Hidden dimension: {hidden_dim}")

    # ---- Hook into last FFN layer (capture input to MLP) ---------------
    captured_activation = {}

    def hook_fn(module, input, output):
        """Forward hook: capture the input to the last MLP layer."""
        captured_activation['value'] = input[0].detach()

    hook = model.model.language_model.layers[-1].mlp.register_forward_hook(hook_fn)

    # ---- Load dataset --------------------------------------------------
    logger.info(f"Loading dataset: {args.dataset} ({args.dataset_config}, split={args.split})")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    logger.info(f"Dataset size: {len(dataset)} examples")

    # ---- Set up Arrow writer -------------------------------------------
    arrow_path = os.path.join(
        args.output_dir, f'dstore_qwen2_vl_{args.split}_{hidden_dim}.arrow')
    fields = [
        pa.field('keys', pa.list_(pa.float16(), hidden_dim)),
        pa.field('vals', pa.int32()),
    ]
    schema = pa.schema(fields)
    arrow_file = pa.OSFile(arrow_path, 'wb')
    arrow_writer = pa.ipc.new_stream(arrow_file, schema)

    total_saved = 0
    per_example_counts = []

    # ---- Process each example ------------------------------------------
    for idx in tqdm(range(len(dataset)), desc="Saving embeddings"):
        example = dataset[idx]
        image = example.get('image')
        desc = example.get('image_description', '')

        if not image or not desc or len(desc.strip()) < 5:
            per_example_counts.append(0)
            continue

        image_rgb = image.convert("RGB")

        # --- format VLM conversation (matches eval_vlm in run_real_eval.py) ---
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe what you see in this cartoon."},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": desc},
            ]},
        ]

        # --- tokenise full conversation (with image) ---
        try:
            full_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            full_inputs = processor(
                text=[full_text], images=[image_rgb],
                return_tensors="pt", padding=True)
        except Exception as e:
            logger.warning(f"Example {idx}: failed to process full conv: {e}")
            per_example_counts.append(0)
            continue

        # --- tokenise prompt-only (to find response boundary) ---
        prompt_messages = messages[:1]  # keep only user turn
        try:
            prompt_text = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_inputs = processor(
                text=[prompt_text], images=[image_rgb],
                return_tensors="pt", padding=True)
        except Exception as e:
            logger.warning(f"Example {idx}: failed to process prompt: {e}")
            per_example_counts.append(0)
            continue

        prompt_len = prompt_inputs['input_ids'].shape[1]

        # --- move to device ---
        full_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in full_inputs.items()
        }

        # --- create labels: -100 for prompt, real IDs for response ---
        input_ids = full_inputs['input_ids']
        seq_len = input_ids.shape[1]
        labels = torch.full_like(input_ids, -100)
        if prompt_len < seq_len:
            labels[0, prompt_len:] = input_ids[0, prompt_len:]

        # --- forward pass ---
        with torch.no_grad():
            _ = model(**full_inputs)

        captured = captured_activation['value']  # (1, seq_len, hidden_dim)

        # --- shift: key at position t predicts token at position t+1 ---
        keys = captured[0, :-1]    # (seq_len-1, hidden_dim)
        vals = labels[0, 1:]       # (seq_len-1,)

        nonpad_mask = vals != -100
        keys = keys[nonpad_mask]   # (num_response, hidden_dim)
        vals = vals[nonpad_mask]   # (num_response,)

        num_tokens = keys.shape[0]
        per_example_counts.append(num_tokens)

        if num_tokens == 0:
            continue

        # --- convert and write to Arrow ---
        keys_np = keys.cpu().to(torch.float16).numpy()
        vals_np = vals.cpu().numpy().astype(np.int32)

        keys_list = [keys_np[i] for i in range(num_tokens)]
        keys_array = pa.array(keys_list, type=pa.list_(pa.float16(), hidden_dim))
        vals_array = pa.array(vals_np, type=pa.int32())

        batch = pa.RecordBatch.from_arrays(
            [keys_array, vals_array], ['keys', 'vals'])
        arrow_writer.write_batch(batch)

        total_saved += num_tokens

        if (idx + 1) % 200 == 0:
            logger.info(
                f"Progress: {idx+1}/{len(dataset)}, "
                f"Total tokens saved: {total_saved}")

    # ---- finalise ------------------------------------------------------
    arrow_writer.close()
    arrow_file.close()
    hook.remove()

    # Save metadata (needed by vlm_prepare_memdec_data.py for dstore_range)
    metadata = {
        'total_tokens': total_saved,
        'num_examples': len(dataset),
        'per_example_counts': per_example_counts,
        'hidden_dim': hidden_dim,
        'vlm_model': args.vlm_model,
        'split': args.split,
    }
    metadata_path = os.path.join(args.output_dir, f'metadata_{args.split}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done! Saved {total_saved} token embeddings from {len(dataset)} examples")
    logger.info(f"Arrow file: {arrow_path}")
    logger.info(f"Metadata:   {metadata_path}")


if __name__ == '__main__':
    main()
