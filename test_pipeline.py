#!/usr/bin/env python
"""
test_pipeline.py  –  Smoke-test for Memory Decoder on LLM and VLM.

1. LLM test : GPT-2  +  GPT-2  (same small model as both base & memory)
2. VLM test : Qwen2-VL-2B-Instruct  +  Qwen2-0.5B

Run:
    python test_pipeline.py              # both tests
    python test_pipeline.py --llm_only   # LLM only  (no GPU needed)
    python test_pipeline.py --vlm_only   # VLM only
"""

import argparse
import sys
import os
import math
import textwrap

import torch
import torch.nn.functional as F
from loguru import logger

# make project importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo.memDec import MemoryDecoder


# ====================================================================
# helpers
# ====================================================================
def separator(title: str):
    w = 70
    logger.info("=" * w)
    logger.info(f"  {title}")
    logger.info("=" * w)


def check_output(logits, input_ids, label: str):
    """Quick sanity check on shapes and values."""
    B, T = input_ids.shape
    assert logits.shape[0] == B, f"[{label}] batch mismatch"
    assert logits.shape[1] == T, f"[{label}] seq-len mismatch"
    nan_count = torch.isnan(logits).sum().item()
    inf_count = torch.isinf(logits).sum().item()
    if nan_count > 0:
        logger.warning(f"  [{label}] {nan_count} NaN values detected")
    if inf_count > 0:
        logger.warning(f"  [{label}] {inf_count} Inf values detected")
    logger.info(f"  [{label}] logits shape = {tuple(logits.shape)}  ✓")


# ====================================================================
# 1.  LLM test
# ====================================================================
def test_llm(device: str = "cpu"):
    separator("LLM test  (GPT-2 + GPT-2)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "gpt2"
    logger.info(f"Loading tokenizer & model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_lm = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    mem_lm  = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

    vocab_size = len(tokenizer)
    base_lm.resize_token_embeddings(vocab_size)
    mem_lm.resize_token_embeddings(vocab_size)

    joint = MemoryDecoder(base_lm, mem_lm, lmbda=0.25, knn_temp=1.0)

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ---- forward check ------------------------------------------------
    logger.info("Forward pass …")
    with torch.no_grad():
        out = joint(**inputs)
    check_output(out.logits, inputs["input_ids"], "forward")

    # verify the output is log-probabilities (all ≤ 0)
    assert (out.logits <= 1e-5).all(), "logits should be log-probs (≤ 0)"
    logger.info("  log-prob range: [{:.4f}, {:.4f}]  ✓".format(
        out.logits.min().item(), out.logits.max().item()))

    # ---- generate check -----------------------------------------------
    logger.info("Greedy generation …")
    gen_ids = joint.generate(**inputs, max_new_tokens=30, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    logger.info(f"  Generated: {gen_text}")

    # ---- base-only generation for comparison --------------------------
    base_ids = base_lm.generate(**inputs, max_new_tokens=30, do_sample=False)
    base_text = tokenizer.decode(base_ids[0], skip_special_tokens=True)
    logger.info(f"  Base only: {base_text}")

    # ---- PPL on a small sample ----------------------------------------
    logger.info("PPL computation on small text …")
    sample_text = (
        "Artificial intelligence has transformed many industries. "
        "Machine learning models can now understand natural language, "
        "generate images, and even write code."
    )
    enc = tokenizer(sample_text, return_tensors="pt").to(device)
    with torch.no_grad():
        base_out = base_lm(**enc)
        joint_out = joint(**enc)

    labels = enc["input_ids"][:, 1:]
    # base PPL
    base_lp = F.log_softmax(base_out.logits[:, :-1].float(), dim=-1)
    nll_base = F.nll_loss(base_lp.view(-1, base_lp.size(-1)),
                          labels.reshape(-1), reduction="mean").item()
    # joint PPL
    joint_lp = joint_out.logits[:, :-1].float()
    nll_joint = F.nll_loss(joint_lp.view(-1, joint_lp.size(-1)),
                           labels.reshape(-1), reduction="mean").item()

    logger.info(f"  Base  PPL = {math.exp(nll_base):.2f}")
    logger.info(f"  Joint PPL = {math.exp(nll_joint):.2f}")

    logger.info("LLM test PASSED ✓\n")
    return True


# ====================================================================
# 2.  VLM test
# ====================================================================
def test_vlm(device: str = "cuda"):
    separator("VLM test  (Qwen2-VL-2B + Qwen2-0.5B)")

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    vlm_id = "Qwen/Qwen2-VL-2B-Instruct"
    mem_id = "Qwen/Qwen2-0.5B"

    # ---- load VLM ----------------------------------------------------
    logger.info(f"Loading VLM: {vlm_id}")
    try:
        from transformers import Qwen2VLForConditionalGeneration
        vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
    except ImportError:
        vlm = AutoModelForCausalLM.from_pretrained(
            vlm_id, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    vlm.eval()

    processor = AutoProcessor.from_pretrained(vlm_id)
    tokenizer = (processor.tokenizer
                 if hasattr(processor, "tokenizer")
                 else AutoTokenizer.from_pretrained(vlm_id))

    # ---- load memory decoder -----------------------------------------
    logger.info(f"Loading Memory Decoder LM: {mem_id}")
    mem_lm = AutoModelForCausalLM.from_pretrained(
        mem_id, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vocab_size = len(tokenizer)
    vlm.resize_token_embeddings(vocab_size)
    mem_lm.resize_token_embeddings(vocab_size)

    joint = MemoryDecoder(vlm, mem_lm, lmbda=0.25, knn_temp=1.0)
    logger.info("MemoryDecoder (VLM mode) created  ✓")

    # ---- prepare a dummy / real image --------------------------------
    from PIL import Image
    import requests
    from io import BytesIO

    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    logger.info(f"Downloading test image …")
    try:
        resp = requests.get(img_url, timeout=15)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        logger.warning("Cannot download image, creating a synthetic one (448x448).")
        import random
        image = Image.new("RGB", (448, 448))
        pixels = image.load()
        for i in range(448):
            for j in range(448):
                pixels[i, j] = (random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255))
    logger.info(f"  Image size: {image.size}")

    # ---- forward check ------------------------------------------------
    logger.info("Forward pass (VLM + MemDec) …")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image],
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        out = joint(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **{k: v for k, v in inputs.items()
               if k not in ("input_ids", "attention_mask")},
        )
    check_output(out.logits, inputs["input_ids"], "VLM forward")

    # ---- generate check -----------------------------------------------
    logger.info("Greedy generation (VLM + MemDec) …")
    gen_ids = joint.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=40,
        do_sample=False,
        **{k: v for k, v in inputs.items()
           if k not in ("input_ids", "attention_mask")},
    )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    logger.info(f"  Joint output: {gen_text}")

    # ---- base-only generation -----------------------------------------
    with torch.no_grad():
        base_ids = vlm.generate(**inputs, max_new_tokens=40, do_sample=False)
    base_text = tokenizer.decode(base_ids[0], skip_special_tokens=True)
    logger.info(f"  VLM only:     {base_text}")

    # ---- PPL on the caption -------------------------------------------
    logger.info("PPL comparison on a caption …")
    caption = "A colorful image with transparent areas showing a checkerboard pattern."
    cap_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": caption}],
        },
    ]
    cap_text = processor.apply_chat_template(cap_messages, tokenize=False,
                                              add_generation_prompt=False)
    cap_inputs = processor(text=[cap_text], images=[image],
                           return_tensors="pt", padding=True)
    cap_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in cap_inputs.items()}

    with torch.no_grad():
        base_out = vlm(**cap_inputs)
        joint_out = joint(
            input_ids=cap_inputs["input_ids"],
            attention_mask=cap_inputs.get("attention_mask"),
            **{k: v for k, v in cap_inputs.items()
               if k not in ("input_ids", "attention_mask")},
        )

    labels = cap_inputs["input_ids"][:, 1:]
    v_base = base_out.logits.shape[-1]
    v_joint = joint_out.logits.shape[-1]
    v = min(v_base, v_joint)

    base_lp = F.log_softmax(base_out.logits[:, :-1, :v].float(), dim=-1)
    nll_base = F.nll_loss(base_lp.reshape(-1, v),
                          labels.reshape(-1), reduction="mean").item()
    joint_lp = joint_out.logits[:, :-1, :v].float()
    nll_joint = F.nll_loss(joint_lp.reshape(-1, v),
                           labels.reshape(-1), reduction="mean").item()

    logger.info(f"  VLM  PPL = {math.exp(nll_base):.2f}")
    logger.info(f"  Joint PPL = {math.exp(nll_joint):.2f}")

    logger.info("VLM test PASSED ✓\n")
    return True


# ====================================================================
# main
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_only", action="store_true")
    parser.add_argument("--vlm_only", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu / cuda)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | <level>{message}</level>")

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    if not args.vlm_only:
        results["LLM"] = test_llm(device=dev)
    if not args.llm_only:
        if dev == "cpu":
            logger.warning("VLM test skipped on CPU (needs GPU).")
            results["VLM"] = False
        else:
            results["VLM"] = test_vlm(device=dev)

    separator("Summary")
    for k, v in results.items():
        status = "PASS ✓" if v else "FAIL ✗"
        logger.info(f"  {k:6s}  {status}")


if __name__ == "__main__":
    main()
