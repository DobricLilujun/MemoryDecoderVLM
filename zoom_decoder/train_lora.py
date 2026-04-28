"""
LoRA fine-tuning baseline on FineSightBench for Qwen2-VL-2B.

Trained on the SAME train split used by the Zoom Decoder (from
splits.json) so the comparison is fair.  Visual input preserved at full
448×448 (no zoom).

Usage:
    python -m zoom_decoder.train_lora \
        --vlm_model Qwen/Qwen2-VL-2B-Instruct \
        --splits_file ./zoom_decoder/dstore/splits.json \
        --out_dir ./zoom_decoder/ckpt/lora \
        --epochs 2 --batch_size 2 --lr 1e-4 --lora_rank 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    get_linear_schedule_with_warmup,
)


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


class FSBLoRAItem(TorchDataset):
    """Indexes a list of (split_name, sample_idx) pairs across multiple HF subsets."""
    def __init__(self, ds_by_split, plan):
        self.ds_by_split = ds_by_split
        self.plan = list(plan)

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, i):
        split_name, s_idx = self.plan[i]
        return self.ds_by_split[split_name][int(s_idx)]


def make_collate(processor, tokenizer):
    def _c(batch):
        prompts, imgs, full_texts = [], [], []
        prompt_lens = []
        for s in batch:
            img = s["image"].convert("RGB")
            imgs.append(img)
            msgs_full = [
                {"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text", "text": s["question"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": s["answer"]}]},
            ]
            msgs_prompt = msgs_full[:1]
            full_t = processor.apply_chat_template(msgs_full, add_generation_prompt=False, tokenize=False)
            prompt_t = processor.apply_chat_template(msgs_prompt, add_generation_prompt=True, tokenize=False)
            full_texts.append(full_t)
            prompts.append(prompt_t)
            # tokenizer-level prompt length to compute answer-start shift later
            prompt_lens.append(len(tokenizer(prompt_t, add_special_tokens=False)["input_ids"]))

        enc = processor(text=full_texts, images=imgs, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        # Build labels: -100 on padding and on the prompt part; supervise only the
        # answer tokens (all tokens after the prompt, including <|im_end|>).
        labels = input_ids.clone()
        for i in range(input_ids.size(0)):
            seq = input_ids[i]
            seq_len = int(attn[i].sum().item())
            # Total non-pad; answer starts at seq_len - answer_len where
            # answer_len = total_text_toks - prompt_toks (includes im_end).
            # We compute answer_len from tokenizer of full vs prompt text.
            full_ids = tokenizer(full_texts[i], add_special_tokens=False)["input_ids"]
            ans_len = len(full_ids) - prompt_lens[i]
            ans_start = seq_len - ans_len
            labels[i, :ans_start] = -100
            labels[i, seq_len:] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }
        for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if k in enc:
                out[k] = enc[k]
        return out
    return _c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm_model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument(
        "--train_splits", nargs="+", default=["perception", "reasoning"],
        choices=["perception", "reasoning"],
    )
    p.add_argument("--splits_file", required=True,
                   help="V4 splits.json (output of zoom_decoder.make_splits).")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_train", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    splits = json.loads(Path(args.splits_file).read_text())
    if splits.get("version") != "v4":
        raise ValueError(f"{args.splits_file} is not a V4 splits.json")

    ds_by_split = {}
    plan = []
    for split_name in args.train_splits:
        if split_name not in splits:
            raise ValueError(f"split '{split_name}' not in splits_file")
        logger.info("Loading FineSightBench {}", split_name)
        ds_by_split[split_name] = load_dataset("Volavion/FineSightBench")[split_name]
        for s_idx in splits[split_name]["train"]:
            plan.append((split_name, int(s_idx)))
    import random as _r
    _r.Random(args.seed).shuffle(plan)
    if args.max_train > 0:
        plan = plan[: args.max_train]
    logger.info("Combined LoRA train pool size = {}", len(plan))

    processor = AutoProcessor.from_pretrained(args.vlm_model)
    tokenizer = processor.tokenizer
    model = _load_vlm_auto(args.vlm_model, torch.bfloat16)
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    torch_ds = FSBLoRAItem(ds_by_split, plan)
    loader = DataLoader(
        torch_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate(processor, tokenizer),
        num_workers=0,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(loader) // args.grad_accum
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    step = 0
    history = []
    for ep in range(args.epochs):
        pbar = tqdm(loader, desc=f"lora-ep{ep}", leave=False)
        running = 0.0
        n = 0
        for micro, batch in enumerate(pbar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out_ = model(**batch)
            loss = out_.loss / args.grad_accum
            loss.backward()
            running += float(loss) * args.grad_accum
            n += 1
            if (micro + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                pbar.set_postfix(loss=running / max(1, n))
                if step % 20 == 0:
                    history.append({"step": step, "loss": running / max(1, n)})
        logger.info("epoch {} avg_loss={:.4f}", ep, running / max(1, n))

    model.save_pretrained(str(out / "final"))
    processor.save_pretrained(str(out / "final"))
    (out / "train_meta.json").write_text(json.dumps({
        "args": vars(args), "history": history,
    }, indent=2))
    logger.success("LoRA saved to {}", out / "final")


if __name__ == "__main__":
    main()
