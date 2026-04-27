"""
Classical kNN-LM teacher for a *fair baseline* against our zoom / priv
teachers.  The student architecture, loss, split, and training recipe are
identical to zoom_decoder.prepare_teacher; only the teacher distribution
source is different:

    * zoom     : teacher = VLM forward on zoomed crop
    * priv     : teacher = VLM forward with bbox injected into prompt
    * knn      : teacher = FAISS kNN over (ctx-embedding, next-token) dstore
                 collected on the ORIGINAL image+question pairs.

Step 1. Run VLM on (image, question, answer) — capture last-FFN-input at
        each answer-predicting position as the key, and record the gold
        next-token as the val.
Step 2. Build a single IndexFlatL2 over all keys.
Step 3. For each key, retrieve top-(K+1) neighbors, drop self, and compute
        softmax(-d / knn_temp); aggregate by val to form a per-position
        distribution over the vocabulary.  Take top-`topk` as the teacher.
Step 4. Save in the exact schema that zoom_decoder.train expects.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from zoom_decoder.data_utils import (
    SIZE_BUCKETS,
    min_target_size,
    parse_targets,
    size_to_bucket_idx,
    stratified_split,
    union_bbox,
)
from zoom_decoder.prepare_teacher import build_chat_inputs, build_decoder_inputs, find_answer_span


@torch.no_grad()
def extract_keys_and_vals(vlm, processor, image, question, answer, device, dtype, hook_holder):
    """
    Teacher-force the answer through the VLM, capture the last-FFN-input
    activations, and return (keys, vals) at each answer-predicting position
    along with the aligned decoder answer ids.
    """
    full_text, prompt_text = build_chat_inputs(processor, image, question, answer)
    inputs = processor(text=[full_text], images=[image], return_tensors="pt", padding=False)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    ans_start_text, ans_end_text = find_answer_span(processor, full_text, prompt_text)
    answer_tok_count = (ans_end_text - ans_start_text) - 1
    input_ids = inputs["input_ids"][0]
    ans_end_idx = int(input_ids.shape[0]) - 1
    ans_start_idx = ans_end_idx - answer_tok_count
    if ans_start_idx < 1 or answer_tok_count <= 0:
        return None

    hook_holder.clear()
    vlm(**inputs)
    captured = hook_holder["value"][0]  # (L, H)

    # position t predicts token t+1; to get predictor of answer token at
    # index `i`, use key at position i-1.
    pred_positions = list(range(ans_start_idx - 1, ans_end_idx - 1))
    if not pred_positions:
        return None
    keys = captured[pred_positions].to(torch.float32).cpu().numpy()  # (A, H)
    answer_ids = input_ids[ans_start_idx:ans_end_idx].cpu().tolist()
    return keys, answer_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm_model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--split", default="perception")
    p.add_argument("--out_dir", default="./zoom_decoder/dstore_knn")
    p.add_argument("--topk", type=int, default=32,
                   help="# vocab entries kept per position in the teacher distribution")
    p.add_argument("--knn_k", type=int, default=64,
                   help="# nearest neighbours fetched per query (self excluded)")
    p.add_argument("--knn_temp", type=float, default=100.0,
                   help="temperature for softmax(-d / T) over L2 distances")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_per_cell", type=int, default=70)
    p.add_argument("--val_per_cell", type=int, default=15)
    p.add_argument("--test_per_cell", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FineSightBench split={}", args.split)
    ds = load_dataset("Volavion/FineSightBench")[args.split]
    train_idx, val_idx, test_idx = stratified_split(
        ds, seed=args.seed,
        train_per_cell=args.train_per_cell,
        val_per_cell=args.val_per_cell,
        test_per_cell=args.test_per_cell,
    )
    (out_dir / "splits.json").write_text(
        json.dumps({"train": train_idx, "val": val_idx, "test": test_idx}))
    logger.info("Splits → train={}, val={}, test={}", len(train_idx), len(val_idx), len(test_idx))

    logger.info("Loading VLM: {}", args.vlm_model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(args.vlm_model, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.vlm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.vlm_model)

    # --- Hook last-layer MLP input as the kNN key ---
    hook_holder: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, out):
        hook_holder["value"] = inp[0].detach()

    hook = vlm.model.language_model.layers[-1].mlp.register_forward_hook(hook_fn)

    # ============================================================
    # Pass 1 — collect (keys, vals) and decoder inputs per sample.
    # ============================================================
    per_sample: List[Dict] = []
    all_keys: List[np.ndarray] = []
    all_vals: List[int] = []
    skipped = 0
    for sample_idx in tqdm(train_idx, desc="vlm-keys"):
        s = ds[int(sample_idx)]
        image = s["image"].convert("RGB")
        question = s["question"]
        answer = s["answer"]
        try:
            targets = parse_targets(s["metadata"])
        except Exception:
            skipped += 1
            continue
        if not targets:
            skipped += 1
            continue

        out = extract_keys_and_vals(
            vlm, processor, image, question, answer, device, dtype, hook_holder)
        if out is None:
            skipped += 1
            continue
        keys, teacher_answer_ids = out

        dec = build_decoder_inputs(tokenizer, question, answer)
        ans_start, ans_end = dec["answer_start"], dec["answer_end"]
        dec_answer_ids = dec["input_ids"][ans_start:ans_end]
        n = min(len(teacher_answer_ids), len(dec_answer_ids))
        if n == 0:
            skipped += 1
            continue
        if len(dec_answer_ids) > n:
            for i in range(ans_start + n, ans_end):
                dec["labels"][i] = -100
            dec["answer_end"] = ans_start + n
            dec_answer_ids = dec_answer_ids[:n]
        keys = keys[:n]

        # gold next-token as val (== the decoder-side answer id, same tokenizer)
        vals = dec_answer_ids
        base = len(all_keys)
        for k_vec, v in zip(keys, vals):
            all_keys.append(k_vec)
            all_vals.append(int(v))

        per_sample.append({
            "dec": dec,
            "n": n,
            "key_range": (base, base + n),
            "pixel_size": min_target_size(targets),
            "task_type": s["task_type"],
            "difficulty": s["difficulty"],
            "sample_idx": int(sample_idx),
        })

    hook.remove()
    logger.info("Collected {} keys from {} samples (skipped {})",
                len(all_keys), len(per_sample), skipped)

    # Free VLM — we only need FAISS from here on
    del vlm
    torch.cuda.empty_cache()

    # ============================================================
    # Pass 2 — build FAISS index, query kNN, aggregate distributions.
    # ============================================================
    keys_np = np.ascontiguousarray(np.stack(all_keys).astype("float32"))
    vals_np = np.asarray(all_vals, dtype=np.int64)
    H = keys_np.shape[1]
    logger.info("Building IndexFlatL2 over {} x {}", keys_np.shape[0], H)
    index = faiss.IndexFlatL2(H)
    index.add(keys_np)

    kq = min(args.knn_k + 1, keys_np.shape[0])  # +1 to absorb self hit
    logger.info("FAISS search K={} (self excluded), T={}", args.knn_k, args.knn_temp)
    D, I = index.search(keys_np, kq)  # (N, kq)

    dataset_rows = []
    dstore_rows = []
    running_offset = 0

    # drop self (nearest neighbour with distance 0 is the query itself)
    for rec in tqdm(per_sample, desc="knn-teacher"):
        ans_start = rec["dec"]["answer_start"]
        lo, hi = rec["key_range"]
        n = rec["n"]
        assert hi - lo == n

        sample_top_ids: List[List[int]] = []
        sample_top_probs: List[List[float]] = []
        for pos in range(n):
            row_idx = lo + pos
            nbr_idx = I[row_idx]
            nbr_d = D[row_idx].astype(np.float64)
            # remove self
            self_mask = nbr_idx != row_idx
            nbr_idx = nbr_idx[self_mask][: args.knn_k]
            nbr_d = nbr_d[self_mask][: args.knn_k]
            if nbr_idx.size == 0:
                # degenerate: only self in dstore — fall back to gold one-hot
                gold = int(vals_np[row_idx])
                ids_ = [gold] + [0] * (args.topk - 1)
                ps_ = [1.0] + [0.0] * (args.topk - 1)
                sample_top_ids.append(ids_)
                sample_top_probs.append(ps_)
                continue
            # numerically-stable softmax(-d / T)
            logits = -nbr_d / args.knn_temp
            logits = logits - logits.max()
            w = np.exp(logits)
            w = w / w.sum()
            agg: Dict[int, float] = defaultdict(float)
            for wi, ni in zip(w, nbr_idx):
                agg[int(vals_np[ni])] += float(wi)
            items = sorted(agg.items(), key=lambda kv: -kv[1])[: args.topk]
            ids_ = [tid for tid, _ in items]
            ps_ = [p for _, p in items]
            # renormalise over the kept topk
            s = sum(ps_)
            if s > 0:
                ps_ = [p / s for p in ps_]
            # pad to fixed width `args.topk` (collate_fn needs uniform shape)
            while len(ids_) < args.topk:
                ids_.append(0)
                ps_.append(0.0)
            sample_top_ids.append(ids_)
            sample_top_probs.append(ps_)

        # write dstore rows
        dec = rec["dec"]
        for i in range(n):
            dstore_rows.append({
                "label": int(dec["input_ids"][ans_start + i]),
                "token_id": sample_top_ids[i],
                "prob": sample_top_probs[i],
            })

        dataset_rows.append({
            "input_ids": dec["input_ids"],
            "labels": dec["labels"],
            "attention_mask": dec["attention_mask"],
            "dstore_range": [running_offset, running_offset + n],
            "pixel_size": rec["pixel_size"],
            "size_bucket": size_to_bucket_idx(rec["pixel_size"]),
            "task_type": rec["task_type"],
            "difficulty": rec["difficulty"],
            "sample_idx": rec["sample_idx"],
        })
        running_offset += n

    logger.info("Built {} rows ({} dstore positions)",
                len(dataset_rows), len(dstore_rows))

    Dataset.from_list(dataset_rows).save_to_disk(str(out_dir / "dataset"))
    Dataset.from_list(dstore_rows).save_to_disk(str(out_dir / "dstore"))

    summary = {
        "num_samples": len(dataset_rows),
        "num_dstore_rows": len(dstore_rows),
        "skipped": skipped,
        "topk": args.topk,
        "knn_k": args.knn_k,
        "knn_temp": args.knn_temp,
        "vlm_model": args.vlm_model,
        "split": args.split,
        "teacher_mode": "knn",
        "size_buckets": SIZE_BUCKETS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.success("Saved to {}", out_dir)


if __name__ == "__main__":
    main()
