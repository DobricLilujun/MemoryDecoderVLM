#!/usr/bin/env python
"""
run_real_eval.py  –  Quick real-data evaluation on New Yorker Caption Contest.

Tests Memory Decoder on both:
  1. LLM mode: Qwen2-0.5B (base) + Qwen2-0.5B (memory) on caption text
  2. VLM mode: Qwen2-VL-2B (base) + Qwen2-0.5B (memory) on image+caption

Usage:
    python run_real_eval.py               # both
    python run_real_eval.py --llm_only
    python run_real_eval.py --vlm_only
    python run_real_eval.py --max_samples 50
"""

import argparse, math, sys, os
import torch, torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo.memDec import MemoryDecoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm_only", action="store_true")
    p.add_argument("--vlm_only", action="store_true")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--lmbda", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def eval_llm(device, max_samples, lmbda):
    """Evaluate Memory Decoder in LLM-only mode on real text."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    logger.info("=" * 60)
    logger.info("  LLM Evaluation: Qwen2-0.5B + Qwen2-0.5B (memory)")
    logger.info("=" * 60)

    model_id = "Qwen/Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_lm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).to(device).eval()
    mem_lm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).to(device).eval()

    v = len(tokenizer)
    base_lm.resize_token_embeddings(v)
    mem_lm.resize_token_embeddings(v)

    joint = MemoryDecoder(base_lm, mem_lm, lmbda=lmbda, knn_temp=1.0)

    # Use captions from New Yorker as text corpus
    ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation",
                      split="validation", streaming=True)

    total_nll_base = 0.0
    total_nll_joint = 0.0
    total_tokens = 0
    count = 0

    for sample in tqdm(ds, desc="LLM eval", total=max_samples):
        text = sample.get("label", "")
        if not text or len(text.strip()) < 20:
            continue

        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512).to(device)
        if enc["input_ids"].shape[1] < 5:
            continue

        with torch.no_grad():
            base_out = base_lm(**enc)
            joint_out = joint(**enc)

        labels = enc["input_ids"][:, 1:]
        vv = min(base_out.logits.shape[-1], joint_out.logits.shape[-1])

        base_lp = F.log_softmax(base_out.logits[:, :-1, :vv].float(), dim=-1)
        joint_lp = joint_out.logits[:, :-1, :vv].float()

        nll_b = F.nll_loss(base_lp.reshape(-1, vv),
                           labels.reshape(-1), reduction="sum").item()
        nll_j = F.nll_loss(joint_lp.reshape(-1, vv),
                           labels.reshape(-1), reduction="sum").item()

        n = labels.numel()
        total_nll_base += nll_b
        total_nll_joint += nll_j
        total_tokens += n
        count += 1
        if count >= max_samples:
            break

    ppl_b = math.exp(total_nll_base / total_tokens)
    ppl_j = math.exp(total_nll_joint / total_tokens)
    logger.info(f"Samples: {count},  Tokens: {total_tokens}")
    logger.info(f"Base  PPL = {ppl_b:.2f}")
    logger.info(f"Joint PPL = {ppl_j:.2f}  (λ={lmbda})")
    logger.info(f"Δ PPL     = {ppl_b - ppl_j:+.2f}")

    # Generation comparison
    prompt = "The cartoon shows a funny scene where"
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_base = base_lm.generate(**inp, max_new_tokens=40, do_sample=False)
        gen_joint = joint.generate(**inp, max_new_tokens=40, do_sample=False)
    logger.info(f"Prompt:  '{prompt}'")
    logger.info(f"Base:    {tokenizer.decode(gen_base[0], skip_special_tokens=True)}")
    logger.info(f"Joint:   {tokenizer.decode(gen_joint[0], skip_special_tokens=True)}")
    return ppl_b, ppl_j


def eval_vlm(device, max_samples, lmbda):
    """Evaluate Memory Decoder in VLM mode on real image+text data."""
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from datasets import load_dataset

    logger.info("=" * 60)
    logger.info("  VLM Evaluation: Qwen2-VL-2B + Qwen2-0.5B (memory)")
    logger.info("=" * 60)

    vlm_id = "Qwen/Qwen2-VL-2B-Instruct"
    mem_id = "Qwen/Qwen2-0.5B"

    try:
        from transformers import Qwen2VLForConditionalGeneration
        vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_id, torch_dtype=torch.bfloat16, device_map="auto")
    except ImportError:
        vlm = AutoModelForCausalLM.from_pretrained(
            vlm_id, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    vlm.eval()

    processor = AutoProcessor.from_pretrained(vlm_id)
    tokenizer = (processor.tokenizer if hasattr(processor, "tokenizer")
                 else AutoTokenizer.from_pretrained(vlm_id))

    mem_lm = AutoModelForCausalLM.from_pretrained(
        mem_id, torch_dtype=torch.bfloat16).to(device).eval()

    v = len(tokenizer)
    vlm.resize_token_embeddings(v)
    mem_lm.resize_token_embeddings(v)

    joint = MemoryDecoder(vlm, mem_lm, lmbda=lmbda, knn_temp=1.0)

    ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation",
                      split="validation", streaming=True)

    total_nll_base = 0.0
    total_nll_joint = 0.0
    total_tokens = 0
    count = 0

    for sample in tqdm(ds, desc="VLM eval", total=max_samples):
        image = sample.get("image")
        desc = sample.get("image_description", "")
        if not image or not desc or len(desc.strip()) < 10:
            continue

        # Build chat with image + description as assistant answer
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
            text = processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=False)
            inputs = processor(text=[text], images=[image.convert("RGB")],
                               return_tensors="pt", padding=True)
        except Exception as e:
            logger.debug(f"Skip sample {count}: {e}")
            continue

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            base_out = vlm(**inputs)
            joint_out = joint(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **{k: v for k, v in inputs.items()
                   if k not in ("input_ids", "attention_mask")},
            )

        # Only compute loss on assistant (description) tokens.
        # Find where the assistant response starts by encoding just
        # the prompt (without the assistant answer).
        prompt_messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe what you see in this cartoon."},
            ]},
        ]
        prompt_text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_inputs = processor(text=[prompt_text], images=[image.convert("RGB")],
                                   return_tensors="pt", padding=True)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Build labels: -100 for prompt, real ids for assistant response
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        labels = labels[:, 1:]  # shift for next-token prediction

        vv = min(base_out.logits.shape[-1], joint_out.logits.shape[-1])
        shift_base = base_out.logits[:, :-1, :vv]
        shift_joint = joint_out.logits[:, :-1, :vv]

        # Mask out prompt positions
        nonpad = labels != -100
        if nonpad.sum() == 0:
            continue

        base_lp = F.log_softmax(shift_base.float(), dim=-1)
        nll_b = F.nll_loss(base_lp[nonpad].reshape(-1, vv),
                           labels[nonpad].reshape(-1), reduction="sum").item()
        nll_j = F.nll_loss(shift_joint[nonpad].float().reshape(-1, vv),
                           labels[nonpad].reshape(-1), reduction="sum").item()

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
    logger.info(f"Samples: {count},  Tokens: {total_tokens}")
    logger.info(f"VLM-only PPL = {ppl_b:.2f}")
    logger.info(f"VLM+Mem  PPL = {ppl_j:.2f}  (λ={lmbda})")
    logger.info(f"Δ PPL        = {ppl_b - ppl_j:+.2f}")

    # Generation on last image
    gen_messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe what you see in this cartoon."},
        ]},
    ]
    gen_text = processor.apply_chat_template(gen_messages, tokenize=False,
                                              add_generation_prompt=True)
    gen_inputs = processor(text=[gen_text], images=[image.convert("RGB")],
                           return_tensors="pt", padding=True)
    gen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in gen_inputs.items()}
    with torch.no_grad():
        gen_base = vlm.generate(**gen_inputs, max_new_tokens=60, do_sample=False)
        gen_joint = joint.generate(
            input_ids=gen_inputs["input_ids"],
            attention_mask=gen_inputs.get("attention_mask"),
            max_new_tokens=60, do_sample=False,
            **{k: v for k, v in gen_inputs.items()
               if k not in ("input_ids", "attention_mask")},
        )
    logger.info("Last-image generation:")
    logger.info(f"  VLM:   {tokenizer.decode(gen_base[0], skip_special_tokens=True)[-200:]}")
    logger.info(f"  Joint: {tokenizer.decode(gen_joint[0], skip_special_tokens=True)[-200:]}")

    return ppl_b, ppl_j


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | <level>{message}</level>")
    dev = args.device if torch.cuda.is_available() else "cpu"

    results = {}
    if not args.vlm_only:
        ppl_b, ppl_j = eval_llm(dev, args.max_samples, args.lmbda)
        results["LLM"] = (ppl_b, ppl_j)
    if not args.llm_only:
        ppl_b, ppl_j = eval_vlm(dev, args.max_samples, args.lmbda)
        results["VLM"] = (ppl_b, ppl_j)

    logger.info("=" * 60)
    logger.info("  Final Results")
    logger.info("=" * 60)
    for mode, (pb, pj) in results.items():
        if pb and pj:
            logger.info(f"  {mode:6s}  Base={pb:.2f}  Joint={pj:.2f}  Δ={pb-pj:+.2f}")


if __name__ == "__main__":
    main()
