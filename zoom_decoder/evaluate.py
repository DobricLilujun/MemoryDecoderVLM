"""
Evaluate accuracy on FineSightBench, broken down by difficulty and task.

Supports several configurations:
    • base       — Qwen2-VL-2B only
    • zoom       — Qwen2-VL-2B + trained Zoom Decoder (joint decoding)
    • lora       — LoRA-fine-tuned Qwen2-VL-2B

For `zoom` mode we step-wise generate by fusing the VLM's and the
decoder's next-token log-probabilities:

    logprob = logaddexp(log(1-λ) + logp_base, log(λ) + logp_zoom)

Because the ZoomDecoder is text-only, we feed it the concatenation of
the decoder template and the already-generated answer prefix.

Usage:
    python -m zoom_decoder.evaluate \
        --mode base|zoom|lora \
        --splits_file ./zoom_decoder/dstore/splits.json \
        --zoom_ckpt ./zoom_decoder/ckpt/sz_full/final \
        --lora_ckpt ./zoom_decoder/ckpt/lora/final \
        --lmbda 0.35 \
        --max_samples 600 \
        --out_file ./zoom_decoder/eval/zoom_full.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from zoom_decoder.data_utils import parse_targets, min_target_size, size_to_bucket_idx, SIZE_BUCKETS
from zoom_decoder.model import ZoomDecoder


# --------------------------------------------------------------------------- #
# Answer parsing & correctness                                                #
# --------------------------------------------------------------------------- #

def _normalise(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def parse_json_safe(s: str) -> Optional[dict]:
    """Best-effort parse of model's JSON output."""
    s = s.strip()
    # Model may generate code fences
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def is_correct(pred_str: str, gt_str: str) -> bool:
    pred = parse_json_safe(pred_str)
    gt = parse_json_safe(gt_str) or {}
    if pred is None or not gt:
        return False
    if set(pred.keys()) != set(gt.keys()):
        return False
    for k, vgt in gt.items():
        vp = pred.get(k)
        if isinstance(vgt, (str, int, float)):
            if _normalise(vp) != _normalise(vgt):
                return False
        elif isinstance(vgt, dict):
            if not isinstance(vp, dict): return False
            # Normalise both sides' str-ified key/val
            gtn = {_normalise(k): _normalise(v) for k, v in vgt.items()}
            pn = {_normalise(k): _normalise(v) for k, v in vp.items()}
            if gtn != pn: return False
        elif isinstance(vgt, list):
            if not isinstance(vp, list): return False
            if [_normalise(x) for x in vgt] != [_normalise(x) for x in vp]:
                return False
        else:
            if str(vp) != str(vgt):
                return False
    return True


# --------------------------------------------------------------------------- #
# Generation                                                                   #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate_base(vlm, processor, image, question, max_new_tokens=48):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(vlm.device)
    ids = vlm.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    prompt_len = int(inputs["input_ids"].size(1))
    gen = ids[0, prompt_len:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def generate_zoom_joint(
    vlm, processor, zoom_decoder, tokenizer, image, question, size_bucket: int,
    lmbda: float = 0.35, max_new_tokens: int = 48,
):
    """Greedy step-wise fusion with KV caching on both models."""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    v_inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(vlm.device)

    # Decoder text-only prompt
    dec_prompt = f"Question: {question}\nAnswer: "
    d_ids = torch.tensor(
        [tokenizer(dec_prompt, add_special_tokens=False)["input_ids"]],
        device=zoom_decoder.base.device,
    )
    d_attn = torch.ones_like(d_ids)
    size_t = torch.tensor([size_bucket], device=zoom_decoder.base.device, dtype=torch.long)

    eos_id = processor.tokenizer.eos_token_id
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = {eos_id, im_end_id}

    log1_lm = math.log(max(1e-6, 1 - lmbda))
    log1_zd = math.log(max(1e-6, lmbda))

    # ---- prefill VLM (with image) ----
    v_kwargs = dict(v_inputs)
    v_kwargs["use_cache"] = True
    v_out = vlm(**v_kwargs, return_dict=True)
    v_past = v_out.past_key_values
    logits_v_last = v_out.logits[0, -1, :]

    # ---- prefill decoder ----
    d_out = zoom_decoder(input_ids=d_ids, attention_mask=d_attn, size_bucket=size_t)
    logits_d_last = d_out.logits[0, -1, :]
    # We don't expose past_kv from ZoomDecoder wrapper; just re-run on appended id each step
    # using the *full* sequence (still cheaper than VLM).  d_ids will grow.

    gen_tokens = []
    for step in range(max_new_tokens):
        V = min(logits_v_last.size(0), logits_d_last.size(0))
        lp_v = F.log_softmax(logits_v_last[:V].float(), dim=-1)
        lp_d = F.log_softmax(logits_d_last[:V].float(), dim=-1)
        joint = torch.logaddexp(lp_v + log1_lm, lp_d + log1_zd)
        nxt = int(joint.argmax().item())
        gen_tokens.append(nxt)
        if nxt in stop_ids:
            break
        # Step VLM with one-token input + past
        nxt_t = torch.tensor([[nxt]], device=vlm.device)
        v_out = vlm(
            input_ids=nxt_t,
            past_key_values=v_past,
            use_cache=True,
            return_dict=True,
        )
        v_past = v_out.past_key_values
        logits_v_last = v_out.logits[0, -1, :]
        # Append to decoder
        d_ids = torch.cat([d_ids, torch.tensor([[nxt]], device=d_ids.device)], dim=1)
        d_attn = torch.cat([d_attn, torch.ones((1, 1), dtype=d_attn.dtype, device=d_attn.device)], dim=1)
        d_out = zoom_decoder(input_ids=d_ids, attention_mask=d_attn, size_bucket=size_t)
        logits_d_last = d_out.logits[0, -1, :]

    return processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base", "zoom", "lora"], required=True)
    p.add_argument("--split", default="perception")
    p.add_argument("--splits_file", required=True)
    p.add_argument("--eval_split", default="test", choices=["val", "test"])
    p.add_argument("--vlm_model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--zoom_ckpt", default=None)
    p.add_argument("--lora_ckpt", default=None)
    p.add_argument("--lmbda", type=float, default=0.35)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--out_file", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    splits = json.loads(Path(args.splits_file).read_text())
    indices = splits[args.eval_split]
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    logger.info("Eval on {} samples", len(indices))

    ds = load_dataset("Volavion/FineSightBench")[args.split]
    processor = AutoProcessor.from_pretrained(args.vlm_model)

    vlm = Qwen2VLForConditionalGeneration.from_pretrained(args.vlm_model, dtype=torch.bfloat16)
    if args.mode == "lora":
        assert args.lora_ckpt, "Need --lora_ckpt for lora mode"
        vlm = PeftModel.from_pretrained(vlm, args.lora_ckpt)
    vlm.to(device).eval()

    zoom_dec = None
    tokenizer = None
    if args.mode == "zoom":
        assert args.zoom_ckpt, "Need --zoom_ckpt"
        zoom_dec = ZoomDecoder.load_from_dir(args.zoom_ckpt, dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.vlm_model)

    # Aggregate metrics by difficulty and task_type
    correct_by_diff = defaultdict(lambda: [0, 0])  # [correct, total]
    correct_by_task = defaultdict(lambda: [0, 0])
    correct_by_size = defaultdict(lambda: [0, 0])
    records = []

    for idx in tqdm(indices, desc=args.mode):
        s = ds[int(idx)]
        image = s["image"].convert("RGB")
        question = s["question"]
        gt = s["answer"]
        try:
            targets = parse_targets(s["metadata"])
        except Exception:
            targets = []
        px = min_target_size(targets) if targets else 48
        size_b = size_to_bucket_idx(px)

        if args.mode in ("base", "lora"):
            pred = generate_base(vlm, processor, image, question, max_new_tokens=args.max_new_tokens)
        else:
            pred = generate_zoom_joint(
                vlm, processor, zoom_dec, tokenizer, image, question,
                size_bucket=size_b, lmbda=args.lmbda,
                max_new_tokens=args.max_new_tokens,
            )
        ok = is_correct(pred, gt)

        diff = s["difficulty"]
        tt = s["task_type"]
        correct_by_diff[diff][1] += 1
        correct_by_diff[diff][0] += int(ok)
        correct_by_task[tt][1] += 1
        correct_by_task[tt][0] += int(ok)
        correct_by_size[px][1] += 1
        correct_by_size[px][0] += int(ok)
        records.append({
            "idx": int(idx), "task": tt, "difficulty": diff, "pixel_size": px,
            "gt": gt, "pred": pred, "correct": bool(ok),
        })

    total_correct = sum(c for c, _ in correct_by_diff.values())
    total_all = sum(t for _, t in correct_by_diff.values())

    def pct(d):
        return {k: {"acc": round(100 * v[0] / max(1, v[1]), 2), "n": v[1]}
                for k, v in sorted(d.items())}

    summary = {
        "mode": args.mode,
        "vlm": args.vlm_model,
        "zoom_ckpt": args.zoom_ckpt,
        "lora_ckpt": args.lora_ckpt,
        "lmbda": args.lmbda if args.mode == "zoom" else None,
        "overall_acc": round(100 * total_correct / max(1, total_all), 2),
        "n": total_all,
        "by_difficulty": pct(correct_by_diff),
        "by_task": pct(correct_by_task),
        "by_pixel_size": pct(correct_by_size),
    }

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps({
        "summary": summary, "records": records,
    }, indent=2))
    logger.success("Overall acc: {:.2f}%", summary["overall_acc"])
    for k in ["extreme", "hard", "medium", "easy"]:
        if k in summary["by_difficulty"]:
            logger.info("  {}: {:.2f}%  (n={})",
                        k, summary["by_difficulty"][k]["acc"],
                        summary["by_difficulty"][k]["n"])
    logger.info("Saved → {}", args.out_file)


if __name__ == "__main__":
    main()
