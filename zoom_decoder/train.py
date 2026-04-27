"""
Train the Zoom Decoder with scale-weighted focal KL distillation.

Reads the dataset + dstore produced by `prepare_teacher.py`.

Usage:
    python -m zoom_decoder.train \
        --data_dir ./zoom_decoder/dstore \
        --decoder_model Qwen/Qwen2-0.5B \
        --num_layers 24  \               # ← choose 6/12/24 for the size sweep
        --out_dir ./zoom_decoder/ckpt/sz_full \
        --epochs 3 --batch_size 8 --lr 5e-4 \
        --scale_gamma 1.0 --alpha_ce 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from zoom_decoder.data_utils import SIZE_BUCKETS, scale_weight, size_to_bucket_idx
from zoom_decoder.losses import scale_weighted_focal_kl, scale_weights_from_pixel
from zoom_decoder.model import ZoomDecoder


def collate_fn(batch, dstore, pad_id: int, scale_gamma: float, ref_size: int = 48):
    """Pad sequences, gather teacher distributions, compute per-token weights."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    size_buckets = []
    pixel_sizes = []
    dstore_ranges = []

    for b in batch:
        L = len(b["input_ids"])
        pad = max_len - L
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
        size_buckets.append(int(b["size_bucket"]))
        pixel_sizes.append(int(b["pixel_size"]))
        dstore_ranges.append(b["dstore_range"])

    # Gather teacher top-k across all answer positions in the batch
    top_ids_list, top_probs_list, per_tok_w = [], [], []
    for rng, px in zip(dstore_ranges, pixel_sizes):
        s, e = int(rng[0]), int(rng[1])
        slc = dstore.select(range(s, e))
        d = slc[:]
        top_ids_list.append(torch.tensor(d["token_id"], dtype=torch.long))
        top_probs_list.append(torch.tensor(d["prob"], dtype=torch.float32))
        w = scale_weight(px, ref=ref_size, gamma=scale_gamma)
        per_tok_w.extend([w] * (e - s))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "size_bucket": torch.tensor(size_buckets, dtype=torch.long),
        "pixel_size": torch.tensor(pixel_sizes, dtype=torch.long),
        "teacher_top_ids": torch.cat(top_ids_list, dim=0),
        "teacher_top_probs": torch.cat(top_probs_list, dim=0),
        "per_token_weights": torch.tensor(per_tok_w, dtype=torch.float32),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="contains dataset/ and dstore/")
    p.add_argument("--decoder_model", default="Qwen/Qwen2-0.5B")
    p.add_argument("--tokenizer_model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_layers", type=int, default=None,
                   help="Truncate to this many layers (omit = full)")
    p.add_argument("--disable_aperture", action="store_true")
    p.add_argument("--disable_scale_weight", action="store_true",
                   help="Use uniform KL (ablation)")
    p.add_argument("--scale_gamma", type=float, default=1.0)
    p.add_argument("--alpha_ce", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info("Loading data from {}", args.data_dir)
    ds = load_from_disk(f"{args.data_dir}/dataset")
    dstore = load_from_disk(f"{args.data_dir}/dstore")
    logger.info("Train samples: {}, dstore rows: {}", len(ds), len(dstore))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    logger.info("Loading decoder: {}  layers={}", args.decoder_model, args.num_layers)
    model = ZoomDecoder.from_pretrained(
        args.decoder_model,
        num_layers=args.num_layers,
        use_aperture=(not args.disable_aperture),
        dtype=torch.bfloat16,
    ).to(device)
    n_params = model.num_parameters()
    logger.info("Decoder params: {:,}  ({:.1f}M)", n_params, n_params / 1e6)

    vocab_size = model.config.vocab_size

    # Dataloader
    def _collate(batch):
        return collate_fn(
            batch, dstore, pad_id=pad_id,
            scale_gamma=(0.0 if args.disable_scale_weight else args.scale_gamma),
        )

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate, num_workers=0,
    )

    # Optimizer
    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"}
    decay_params, nodecay_params = [], []
    for n, pa in model.named_parameters():
        if not pa.requires_grad: continue
        if any(nd in n for nd in no_decay):
            nodecay_params.append(pa)
        else:
            decay_params.append(pa)
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    total_steps = args.epochs * len(loader)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    step = 0
    history = []
    for ep in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch{ep}", leave=False)
        ep_kl = ep_ce = ep_n = 0.0
        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                size_bucket=(None if args.disable_aperture else batch["size_bucket"]),
            )
            logits = out.logits  # (B, T, V)
            loss_dict = scale_weighted_focal_kl(
                logits=logits.float(),
                labels=batch["labels"],
                teacher_top_ids=batch["teacher_top_ids"],
                teacher_top_probs=batch["teacher_top_probs"],
                per_token_weights=batch["per_token_weights"],
                vocab_size=vocab_size,
                alpha_ce=args.alpha_ce,
            )
            loss = loss_dict["loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            ep_kl += float(loss_dict["kl"]) * float(loss_dict["n_tokens"])
            ep_ce += float(loss_dict["ce"]) * float(loss_dict["n_tokens"])
            ep_n += float(loss_dict["n_tokens"])

            step += 1
            pbar.set_postfix(
                loss=float(loss),
                kl=float(loss_dict["kl"]),
                ce=float(loss_dict["ce"]),
                w=float(loss_dict["mean_weight"]),
            )
            if step % args.log_every == 0:
                history.append(
                    {"step": step, "loss": float(loss),
                     "kl": float(loss_dict["kl"]), "ce": float(loss_dict["ce"])}
                )

        logger.info("epoch {}  avg_kl={:.4f}  avg_ce={:.4f}", ep,
                    ep_kl / max(1, ep_n), ep_ce / max(1, ep_n))

    model.save_pretrained(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    meta = {
        "args": vars(args),
        "n_params": n_params,
        "vocab_size": vocab_size,
        "history": history,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.success("Saved → {}", out_dir / "final")


if __name__ == "__main__":
    main()
