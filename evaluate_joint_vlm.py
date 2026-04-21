"""
Evaluate a VLM + Memory Decoder on image-captioning perplexity.

Usage:
    python evaluate_joint_vlm.py \
        --vlm_path  Qwen/Qwen2-VL-2B-Instruct \
        --memdec_path  Qwen/Qwen2-0.5B \
        --dataset_name  HuggingFaceM4/COCO \
        --split  validation \
        --max_samples  200 \
        --lmbda 0.25 \
        --batch_size 1
"""

import argparse
import math
import sys
import os

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from datasets import load_dataset

# ── add project root so `demo.memDec` is importable ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo.memDec import MemoryDecoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm_path", type=str, required=True,
                   help="HF model id or local path for the VLM")
    p.add_argument("--memdec_path", type=str, required=True,
                   help="HF model id or local path for the memory decoder (text-only LM)")
    p.add_argument("--dataset_name", type=str, default="HuggingFaceM4/COCO",
                   help="HF dataset for image-captioning evaluation")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--lmbda", type=float, default=0.25)
    p.add_argument("--knn_temp", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Only batch_size=1 is supported for VLM evaluation")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def build_vlm_prompt(processor, text: str):
    """Build a simple image-captioning prompt for Qwen2-VL style models."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Please describe this image in detail."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": text},
            ],
        },
    ]
    return messages


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 1. Load VLM ──────────────────────────────────────────────────
    logger.info(f"Loading VLM from {args.vlm_path} …")
    try:
        from transformers import Qwen2VLForConditionalGeneration
        vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            args.vlm_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
    except Exception:
        vlm = AutoModelForCausalLM.from_pretrained(
            args.vlm_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    vlm.eval()

    processor = AutoProcessor.from_pretrained(args.vlm_path)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else AutoTokenizer.from_pretrained(args.vlm_path)

    # ── 2. Load Memory Decoder (text-only LM) ────────────────────────
    logger.info(f"Loading Memory Decoder from {args.memdec_path} …")
    memdec_lm = AutoModelForCausalLM.from_pretrained(
        args.memdec_path, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # align vocab sizes
    vocab_size = len(tokenizer)
    vlm.resize_token_embeddings(vocab_size)
    memdec_lm.resize_token_embeddings(vocab_size)

    # ── 3. Build joint model ─────────────────────────────────────────
    joint = MemoryDecoder(vlm, memdec_lm, lmbda=args.lmbda, knn_temp=args.knn_temp)
    logger.info(f"MemoryDecoder created  (λ={args.lmbda}, T={args.knn_temp})")

    # ── 4. Load dataset ──────────────────────────────────────────────
    logger.info(f"Loading dataset {args.dataset_name} [{args.split}] …")
    if args.dataset_config:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    else:
        ds = load_dataset(args.dataset_name, split=args.split)
    if args.max_samples and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
    logger.info(f"  {len(ds)} samples loaded")

    # detect column names
    image_col = "image" if "image" in ds.column_names else ds.column_names[0]
    text_col = None
    for col in ["caption", "text", "sentences", "captions"]:
        if col in ds.column_names:
            text_col = col
            break
    if text_col is None:
        # try nested
        text_col = "sentences_raw" if "sentences_raw" in ds.column_names else ds.column_names[1]
    logger.info(f"  image_col={image_col}, text_col={text_col}")

    # ── 5. Evaluate PPL ──────────────────────────────────────────────
    total_nll_joint = 0.0
    total_nll_base = 0.0
    total_tokens = 0

    for idx in tqdm(range(len(ds)), desc="VLM+MemDec eval"):
        sample = ds[idx]
        image = sample[image_col]
        caption = sample[text_col]
        # handle list-of-captions (COCO style)
        if isinstance(caption, list):
            caption = caption[0] if len(caption) > 0 else ""
        if isinstance(caption, dict):
            caption = caption.get("raw", caption.get("text", str(caption)))
        if not caption or len(caption.strip()) == 0:
            continue

        try:
            messages = build_vlm_prompt(processor, caption)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(
                text=[text], images=[image], return_tensors="pt", padding=True,
            )
        except Exception as e:
            logger.debug(f"Skipping sample {idx}: {e}")
            continue

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        # we only compute loss on the assistant (caption) tokens at the end
        # simple heuristic: mask everything before the last occurrence of a newline-like token

        with torch.no_grad():
            # --- base VLM only ---
            base_out = vlm(**inputs)
            shift_logits = base_out.logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # --- joint (VLM + MemDec) ---
            joint_out = joint(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **{k: v for k, v in inputs.items()
                   if k not in ("input_ids", "attention_mask")},
            )
            joint_logits = joint_out.logits[:, :-1].contiguous()

        # align vocab for base
        v = min(shift_logits.shape[-1], joint_logits.shape[-1])
        shift_logits = shift_logits[..., :v]
        joint_logits = joint_logits[..., :v]

        lm_lp = F.log_softmax(shift_logits.float(), dim=-1)
        joint_lp = joint_logits.float()   # already log-probs

        n_tok = shift_labels.numel()
        nll_base = F.nll_loss(
            lm_lp.view(-1, v), shift_labels.view(-1), reduction="sum",
        ).item()
        nll_joint = F.nll_loss(
            joint_lp.view(-1, v), shift_labels.view(-1), reduction="sum",
        ).item()

        total_nll_base += nll_base
        total_nll_joint += nll_joint
        total_tokens += n_tok

    # ── 6. Report ────────────────────────────────────────────────────
    if total_tokens == 0:
        logger.error("No valid tokens – cannot compute PPL.")
        return

    ppl_base = math.exp(total_nll_base / total_tokens)
    ppl_joint = math.exp(total_nll_joint / total_tokens)
    logger.info(f"Tokens evaluated: {total_tokens}")
    logger.info(f"VLM-only PPL:      {ppl_base:.2f}")
    logger.info(f"VLM+MemDec PPL:    {ppl_joint:.2f}")
    logger.info(f"PPL reduction:     {ppl_base - ppl_joint:+.2f}")


if __name__ == "__main__":
    main()
