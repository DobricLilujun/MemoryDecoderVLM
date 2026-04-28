"""
Zoom-Teacher distillation data preparation.

For each FineSightBench sample, we run the base VLM on a *zoomed* crop around
the target bbox and save the top-k output distribution at every answer-token
position.  The Zoom Decoder will later be trained to reproduce these
"magnified-view" distributions from *text only* input.

Outputs two Arrow datasets (same layout as the existing VLM MemDec pipeline):
    • <out_dir>/dataset/        — per-sample rows (input_ids, labels, dstore_range, metadata)
    • <out_dir>/dstore.arrow    — per-answer-token rows (label, token_id, prob)

Usage:
    python -m zoom_decoder.prepare_teacher \
        --vlm_model Qwen/Qwen2-VL-2B-Instruct \
        --split perception \
        --out_dir ./zoom_decoder/dstore \
        --topk 64 \
        --max_samples 3600 \
        --pad_ratio 2.5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration


def _load_vlm_auto(model_id: str, dtype):
    """Load the correct VLM class based on model_type in config."""
    model_type = AutoConfig.from_pretrained(model_id).model_type
    if model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration as _Cls
    elif model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as _Cls
    else:
        raise ValueError(f"Unsupported VLM model_type '{model_type}'. Add it to _load_vlm_auto.")
    return _Cls.from_pretrained(model_id, dtype=dtype)

from zoom_decoder.data_utils import (
    SIZE_BUCKETS,
    min_target_size,
    parse_targets,
    size_to_bucket_idx,
    stratified_split,
    stratified_train_val_split,
    union_bbox,
    zoom_crop,
)

DECODER_TEMPLATE = "Question: {q}\nAnswer: {a}"


def build_priv_prompt(question: str, targets, canvas_wh, mode: str = "bbox") -> str:
    """
    Inject bbox (and optionally value) of each target into the teacher's text
    prompt as *privileged information* (LUPI).  The image is NOT modified.
    The student is trained on the ORIGINAL question only, so it never sees
    these coordinates at inference time.

    mode:
      "bbox" — include bbox coords + pixel size (recommended)
      "full" — also include the target's value (degenerates teacher to
               near-deterministic; useful for ceiling analysis only)
    """
    W, H = canvas_wh
    lines = [f"The image is {W}x{H}. It contains {len(targets)} target region(s):"]
    for i, t in enumerate(targets):
        x1, y1, x2, y2 = t.bbox
        if mode == "bbox":
            lines.append(
                f"  Target {i+1}: bbox=({int(x1)},{int(y1)})-({int(x2)},{int(y2)}), size={t.size}px"
            )
        elif mode == "full":
            lines.append(
                f"  Target {i+1}: bbox=({int(x1)},{int(y1)})-({int(x2)},{int(y2)}), "
                f"size={t.size}px, value={t.value!r}"
            )
        else:
            raise ValueError(f"unknown priv mode: {mode}")
    lines.append("")
    lines.append(question)
    return "\n".join(lines)


def build_chat_inputs(processor, image, question: str, answer: str):
    """Build VLM chat inputs with the answer teacher-forced."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    # Separate prompt (no answer) to find answer-start offset
    messages_prompt = messages[:1]
    prompt_text = processor.apply_chat_template(
        messages_prompt, add_generation_prompt=True, tokenize=False
    )
    return text, prompt_text


def find_answer_span(processor, full_text: str, prompt_text: str) -> int:
    """
    Return the token-index where the assistant's answer content starts in the
    tokenization of `full_text`.  Answer ends at len(full_tokens) - 1 (the
    trailing <|im_end|>/eos token is excluded from supervision).
    """
    full_ids = processor.tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return len(prompt_ids), len(full_ids)


@torch.no_grad()
def extract_teacher_topk(
    vlm,
    processor,
    image,
    question: str,
    answer: str,
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Dict]:
    """
    Teacher-force the answer through the VLM and return top-k logits at each
    answer-token position.  Returns None if alignment fails.
    """
    full_text, prompt_text = build_chat_inputs(processor, image, question, answer)
    inputs = processor(
        text=[full_text], images=[image], return_tensors="pt", padding=False
    )
    # Move to device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    # Determine answer span.  Processor expands image placeholders in input_ids,
    # but the SUFFIX after the assistant header is identical between the prompt-only
    # and full-answer tokenizations.  So we compute answer length in the text-only
    # tokenization and work backwards from the end of input_ids.
    ans_start_text, ans_end_text = find_answer_span(processor, full_text, prompt_text)
    # ans_end_text - ans_start_text = answer_tokens + <|im_end|> (1 trailing special).
    answer_tok_count = (ans_end_text - ans_start_text) - 1
    input_ids = inputs["input_ids"][0]
    # ans_end_idx points at <|im_end|>; answer tokens occupy the preceding span.
    ans_end_idx = int(input_ids.shape[0]) - 1
    ans_start_idx = ans_end_idx - answer_tok_count
    if ans_start_idx < 1 or answer_tok_count <= 0:
        return None

    # Forward pass
    outputs = vlm(**inputs)
    logits = outputs.logits[0]  # (L, V)
    # logits at position t predict input_ids[t+1]; so to predict answer token
    # at index i (in input_ids), use logits[i-1].
    pred_positions = list(range(ans_start_idx - 1, ans_end_idx - 1))
    answer_ids = input_ids[ans_start_idx:ans_end_idx].cpu().tolist()

    if len(pred_positions) == 0:
        return None

    sel_logits = logits[pred_positions].float()  # (A, V)
    probs = F.softmax(sel_logits, dim=-1)
    top_probs, top_ids = probs.topk(topk, dim=-1)  # (A, K)

    return {
        "answer_ids": answer_ids,
        "top_ids": top_ids.cpu().numpy().astype(np.int32),
        "top_probs": top_probs.cpu().numpy().astype(np.float32),
    }


def build_decoder_inputs(tokenizer, question: str, answer: str) -> Dict:
    """Build text-only input for the decoder; labels = -100 except on answer tokens."""
    prompt = f"Question: {question}\nAnswer: "
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    full_ids = prompt_ids + ans_ids + [eos_id]
    # Supervise only the answer tokens; EOS has no matching teacher distribution.
    labels = [-100] * len(prompt_ids) + ans_ids + [-100]
    attn = [1] * len(full_ids)
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": attn,
        "answer_start": len(prompt_ids),
        "answer_end": len(prompt_ids) + len(ans_ids),  # exclude EOS from teacher positions
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm_model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument(
        "--splits", nargs="+", default=["perception"],
        choices=["perception", "reasoning"],
        help="Which FineSightBench split(s) to use for TEACHER prep. "
             "V4 with-zoom group: perception only. V4 without-zoom group: both.",
    )
    p.add_argument(
        "--splits_file", default=None,
        help="Optional V4 splits.json (output of zoom_decoder.make_splits). "
             "If given, train indices are loaded from it instead of regenerating.",
    )
    p.add_argument("--out_dir", default="./zoom_decoder/dstore")
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--pad_ratio", type=float, default=2.5)
    p.add_argument(
        "--teacher_mode",
        choices=["zoom", "priv", "priv_full", "zoom_priv"],
        default="zoom",
        help=(
            "zoom      : original - crop bbox and resize to 448 (image modified). "
            "priv      : no image modification; bbox coords injected into teacher prompt. "
            "priv_full : also inject target values (ceiling, for analysis only). "
            "zoom_priv : combine — zoomed image + priv bbox prompt."
        ),
    )
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_per_cell", type=int, default=70)
    p.add_argument("--val_per_cell", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build / load V4 splits.json                                        #
    # ------------------------------------------------------------------ #
    if args.splits_file is not None and Path(args.splits_file).exists():
        logger.info("Loading existing splits from {}", args.splits_file)
        splits_json = json.loads(Path(args.splits_file).read_text())
        if splits_json.get("version") != "v4":
            raise ValueError(f"{args.splits_file} is not a V4 splits.json")
        for sp in args.splits:
            if sp not in splits_json:
                raise ValueError(f"split '{sp}' not present in {args.splits_file}")
    else:
        splits_json = {
            "version": "v4", "seed": args.seed,
            "train_per_cell": args.train_per_cell,
            "val_per_cell": args.val_per_cell,
        }

    # ------------------------------------------------------------------ #
    # Load datasets and assemble combined training plan                  #
    # ------------------------------------------------------------------ #
    ds_by_split: Dict[str, "Dataset"] = {}
    train_plan: List[tuple] = []  # list of (split_name, sample_idx)
    for split_name in args.splits:
        logger.info("Loading FineSightBench split={}", split_name)
        ds_split = load_dataset("Volavion/FineSightBench")[split_name]
        ds_by_split[split_name] = ds_split
        if split_name in splits_json and "train" in splits_json[split_name]:
            train_idx = list(splits_json[split_name]["train"])
            val_idx = list(splits_json[split_name]["val"])
        else:
            train_idx, val_idx = stratified_train_val_split(
                ds_split, seed=args.seed,
                train_per_cell=args.train_per_cell,
                val_per_cell=args.val_per_cell,
            )
            splits_json[split_name] = {"train": train_idx, "val": val_idx}
        logger.info("  {}: train={} val={} (total={})",
                    split_name, len(train_idx), len(val_idx), len(ds_split))
        for s_idx in train_idx:
            train_plan.append((split_name, int(s_idx)))

    # Persist splits.json into out_dir for reproducibility
    (out_dir / "splits.json").write_text(json.dumps(splits_json, indent=2))

    # Shuffle the combined train plan with the same seed (deterministic)
    import random as _r
    _r.Random(args.seed).shuffle(train_plan)
    if args.max_samples > 0:
        train_plan = train_plan[: args.max_samples]
        logger.info("Truncated combined train pool → {}", len(train_plan))
    else:
        logger.info("Combined train pool size = {}", len(train_plan))

    logger.info("Loading VLM: {}", args.vlm_model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    vlm = _load_vlm_auto(args.vlm_model, dtype).to(device)
    vlm.eval()
    processor = AutoProcessor.from_pretrained(args.vlm_model)
    # Use VLM's tokenizer for the decoder so IDs match
    tokenizer = AutoTokenizer.from_pretrained(args.vlm_model)

    dataset_rows = []
    dstore_rows = []
    running_offset = 0
    skipped = 0

    for split_name, sample_idx in tqdm(train_plan, desc=f"teacher[{args.teacher_mode}]"):
        ds = ds_by_split[split_name]
        s = ds[int(sample_idx)]
        image = s["image"].convert("RGB")
        question = s["question"]
        answer = s["answer"]  # already JSON-encoded string
        try:
            targets = parse_targets(s["metadata"])
        except Exception:
            skipped += 1
            continue
        if not targets:
            skipped += 1
            continue
        bbox = union_bbox(targets, image.size)

        # --- Teacher-side image & prompt selection ---
        if args.teacher_mode == "zoom":
            teacher_img = zoom_crop(image, bbox, out_size=image.size[0], pad_ratio=args.pad_ratio)
            teacher_q = question
        elif args.teacher_mode == "priv":
            teacher_img = image
            teacher_q = build_priv_prompt(question, targets, image.size, mode="bbox")
        elif args.teacher_mode == "priv_full":
            teacher_img = image
            teacher_q = build_priv_prompt(question, targets, image.size, mode="full")
        elif args.teacher_mode == "zoom_priv":
            teacher_img = zoom_crop(image, bbox, out_size=image.size[0], pad_ratio=args.pad_ratio)
            teacher_q = build_priv_prompt(question, targets, image.size, mode="bbox")
        else:
            raise ValueError(args.teacher_mode)

        try:
            td = extract_teacher_topk(
                vlm, processor, teacher_img, teacher_q, answer,
                topk=args.topk, device=device, dtype=dtype,
            )
        except Exception as e:
            logger.warning("teacher failed idx={}: {}", sample_idx, e)
            skipped += 1
            continue
        if td is None:
            skipped += 1
            continue

        # NOTE: decoder inputs ALWAYS use the original question (student sees no privilege)
        dec = build_decoder_inputs(tokenizer, question, answer)
        ans_start, ans_end = dec["answer_start"], dec["answer_end"]
        dec_answer_ids = dec["input_ids"][ans_start:ans_end]

        # Align teacher token span to decoder answer span.  If mismatched
        # (tokenizer idiosyncrasies around newlines), trim to min length.
        teacher_n = len(td["answer_ids"])
        dec_n = len(dec_answer_ids)
        n = min(teacher_n, dec_n)
        if n == 0:
            skipped += 1
            continue
        top_ids = td["top_ids"][:n]
        top_probs = td["top_probs"][:n]
        # Trim decoder side to match (drop trailing tokens; they still get EOS)
        if dec_n > n:
            # reduce answer_end to answer_start+n; invalidate extra labels
            for i in range(ans_start + n, ans_end):
                dec["labels"][i] = -100
            dec["answer_end"] = ans_start + n

        pixel_size = min_target_size(targets)
        for i in range(n):
            dstore_rows.append(
                {
                    "label": int(dec["input_ids"][ans_start + i]),
                    "token_id": top_ids[i].tolist(),
                    "prob": top_probs[i].tolist(),
                }
            )

        dataset_rows.append(
            {
                "input_ids": dec["input_ids"],
                "labels": dec["labels"],
                "attention_mask": dec["attention_mask"],
                "dstore_range": [running_offset, running_offset + n],
                "pixel_size": pixel_size,
                "size_bucket": size_to_bucket_idx(pixel_size),
                "task_type": s["task_type"],
                "difficulty": s["difficulty"],
                "sample_idx": int(sample_idx),
                "source_split": split_name,
            }
        )
        running_offset += n

    logger.info("Built {} rows ({} dstore positions); skipped {}",
                len(dataset_rows), len(dstore_rows), skipped)

    # Save decoder dataset
    text_ds = Dataset.from_list(dataset_rows)
    text_ds.save_to_disk(str(out_dir / "dataset"))

    # Save dstore as arrow (single file)
    dstore_ds = Dataset.from_list(dstore_rows)
    dstore_ds.save_to_disk(str(out_dir / "dstore"))

    # Summary
    summary = {
        "num_samples": len(dataset_rows),
        "num_dstore_rows": len(dstore_rows),
        "skipped": skipped,
        "topk": args.topk,
        "vlm_model": args.vlm_model,
        "splits": args.splits,
        "pad_ratio": args.pad_ratio,
        "teacher_mode": args.teacher_mode,
        "size_buckets": SIZE_BUCKETS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.success("Saved to {}", out_dir)


if __name__ == "__main__":
    main()
