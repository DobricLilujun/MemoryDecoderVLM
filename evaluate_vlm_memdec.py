#!/usr/bin/env python
"""
Evaluate trained VLM Memory Decoder vs untrained baseline.

Compares:
  1. VLM-only (Qwen2-VL-2B)
  2. VLM + Untrained MemDec (Qwen2-0.5B pretrained)
  3. VLM + Trained MemDec (KNN-distilled)

on the New Yorker Caption Contest validation set.

Usage:
    python evaluate_vlm_memdec.py --memdec_trained ./vlm_memdec_checkpoints/final
"""

import argparse
import math
import sys
import os

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo.memDec import MemoryDecoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--memdec_untrained", type=str, default="Qwen/Qwen2-0.5B")
    p.add_argument("--memdec_trained", type=str,
                   default="./vlm_memdec_checkpoints/final")
    p.add_argument("--max_samples", type=int, default=130)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def evaluate_with_memdec(vlm, processor, tokenizer, memdec_model, device,
                         ds, max_samples, lmbda, label):
    """Run PPL evaluation for VLM + MemDec combination."""
    joint = MemoryDecoder(vlm, memdec_model, lmbda=lmbda, knn_temp=1.0)

    total_nll_base = 0.0
    total_nll_joint = 0.0
    total_tokens = 0
    count = 0
    wins_joint = 0

    for sample in tqdm(ds, desc=label, total=max_samples):
        image = sample.get("image")
        desc = sample.get("image_description", "")
        if not image or not desc or len(desc.strip()) < 10:
            continue

        image_rgb = image.convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe what you see in this cartoon."},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": desc},
            ]},
        ]

        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[text], images=[image_rgb],
                               return_tensors="pt", padding=True)
        except Exception:
            continue

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Get prompt length for label masking
        prompt_messages = messages[:1]
        prompt_text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_inputs = processor(text=[prompt_text], images=[image_rgb],
                                   return_tensors="pt", padding=True)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.no_grad():
            base_out = vlm(**inputs)
            joint_out = joint(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **{k: v for k, v in inputs.items()
                   if k not in ("input_ids", "attention_mask")},
            )

        # Labels: -100 for prompt, real ids for response
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        labels = labels[:, 1:]  # shift

        vv = min(base_out.logits.shape[-1], joint_out.logits.shape[-1])
        shift_base = base_out.logits[:, :-1, :vv]
        shift_joint = joint_out.logits[:, :-1, :vv]

        nonpad = labels != -100
        if nonpad.sum() == 0:
            continue

        base_lp = F.log_softmax(shift_base.float(), dim=-1)
        nll_b = F.nll_loss(base_lp[nonpad].reshape(-1, vv),
                           labels[nonpad].reshape(-1), reduction="sum").item()
        nll_j = F.nll_loss(shift_joint[nonpad].float().reshape(-1, vv),
                           labels[nonpad].reshape(-1), reduction="sum").item()

        if nll_j < nll_b:
            wins_joint += 1

        n = nonpad.sum().item()
        total_nll_base += nll_b
        total_nll_joint += nll_j
        total_tokens += n
        count += 1
        if count >= max_samples:
            break

    if total_tokens == 0:
        logger.error("No valid tokens evaluated.")
        return None, None

    ppl_b = math.exp(total_nll_base / total_tokens)
    ppl_j = math.exp(total_nll_joint / total_tokens)
    logger.info(f"  Samples: {count},  Tokens: {total_tokens}")
    logger.info(f"  VLM-only PPL = {ppl_b:.4f}")
    logger.info(f"  Joint PPL    = {ppl_j:.4f}  (λ={lmbda})")
    logger.info(f"  Δ PPL        = {ppl_b - ppl_j:+.4f}")
    logger.info(f"  Joint wins   = {wins_joint}/{count} "
                f"({100*wins_joint/count:.1f}%)")
    return ppl_b, ppl_j


def main():
    args = parse_args()
    device = args.device

    from transformers import (AutoModelForCausalLM, AutoProcessor,
                              AutoTokenizer, Qwen2VLForConditionalGeneration)
    from datasets import load_dataset

    # ---- Load VLM -------------------------------------------------------
    logger.info(f"Loading VLM: {args.vlm_model}")
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        args.vlm_model, torch_dtype=torch.bfloat16, device_map="auto")
    vlm.eval()
    processor = AutoProcessor.from_pretrained(args.vlm_model)
    tokenizer = (processor.tokenizer if hasattr(processor, "tokenizer")
                 else AutoTokenizer.from_pretrained(args.vlm_model))

    # ---- Load Untrained MemDec ------------------------------------------
    logger.info(f"Loading untrained MemDec: {args.memdec_untrained}")
    mem_untrained = AutoModelForCausalLM.from_pretrained(
        args.memdec_untrained, torch_dtype=torch.bfloat16).to(device).eval()

    # ---- Load Trained MemDec --------------------------------------------
    logger.info(f"Loading trained MemDec: {args.memdec_trained}")
    mem_trained = AutoModelForCausalLM.from_pretrained(
        args.memdec_trained, torch_dtype=torch.bfloat16).to(device).eval()

    # Align vocab sizes
    v = len(tokenizer)
    vlm.resize_token_embeddings(v)
    mem_untrained.resize_token_embeddings(v)
    mem_trained.resize_token_embeddings(v)

    # ---- Load validation dataset ----------------------------------------
    ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation",
                      split="validation")

    # ---- Evaluate: VLM + Untrained MemDec --------------------------------
    logger.info("=" * 60)
    logger.info("  Evaluation: VLM + Untrained Qwen2-0.5B")
    logger.info("=" * 60)
    ppl_b1, ppl_j1 = evaluate_with_memdec(
        vlm, processor, tokenizer, mem_untrained, device,
        ds, args.max_samples, lmbda=0.25, label="Untrained MemDec")

    # ---- Evaluate: VLM + Trained MemDec ----------------------------------
    logger.info("=" * 60)
    logger.info("  Evaluation: VLM + Trained MemDec (KNN-distilled)")
    logger.info("=" * 60)
    ppl_b2, ppl_j2 = evaluate_with_memdec(
        vlm, processor, tokenizer, mem_trained, device,
        ds, args.max_samples, lmbda=0.25, label="Trained MemDec")

    # ---- Summary --------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  VLM-only PPL:                   {ppl_b1:.4f}")
    logger.info(f"  VLM + Untrained MemDec PPL:     {ppl_j1:.4f}")
    logger.info(f"  VLM + Trained MemDec PPL:       {ppl_j2:.4f}")
    logger.info(f"  Improvement (untrained):        {ppl_b1 - ppl_j1:+.4f}")
    logger.info(f"  Improvement (trained):          {ppl_b1 - ppl_j2:+.4f}")
    if ppl_j1 and ppl_j2:
        logger.info(f"  Training gain over untrained:   {ppl_j1 - ppl_j2:+.4f}")


if __name__ == "__main__":
    main()
