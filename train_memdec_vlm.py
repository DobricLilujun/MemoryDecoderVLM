#!/usr/bin/env python
"""
Train Memory Decoder (Qwen2-0.5B) on KNN distributions distilled from VLM.

This script trains a small text-only model to predict the KNN probability
distributions computed from the VLM's hidden states.  The loss is:

    L = α · KL(p_knn ‖ p_mem) + (1 − α) · CE(p_mem, y)

Usage:
    python train_memdec_vlm.py \
        --model_name_or_path Qwen/Qwen2-0.5B \
        --dataset_name ./vlm_memdec_data \
        --knn_save_path ./vlm_dstore/knn_qwen2_vl_train_1536.arrow \
        --output_dir ./vlm_memdec_checkpoints \
        --num_train_epochs 30 \
        --per_device_train_batch_size 8 \
        --learning_rate 1e-3

    # With accelerate for multi-GPU:
    accelerate launch train_memdec_vlm.py ...
"""

import sys
import os
import inspect
import argparse
import logging
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from loguru import logger

import datasets
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)


# ---------------------------------------------------------------------------
# Loss functions (same as utils/cal_loss.py but inlined for clarity)
# ---------------------------------------------------------------------------

def kl_loss_token(logits, batch, vocab_size, alpha=0.5):
    """
    Compute MemDec training loss:
        L = α · KL(p_knn ‖ p_mem) + (1 − α) · CE(p_mem, y)

    Args:
        logits:     (B, T, V) model output
        batch:      dict with 'labels', 'knn_label', 'knn_probs'
        vocab_size: vocabulary size
        alpha:      interpolation weight
    """
    knn_label = batch['knn_label']
    label_probs = batch['knn_probs']

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = batch['labels'][:, 1:].contiguous()

    nonpad_mask = shift_labels != -100
    shift_logits = shift_logits[nonpad_mask]
    shift_labels = shift_labels[nonpad_mask]

    # Normalise KNN probs (they are already mostly normalised)
    label_probs = label_probs / label_probs.sum(dim=-1, keepdim=True)

    # Truncate logits to match KNN vocab if necessary
    if shift_logits.shape[-1] > label_probs.shape[-1]:
        shift_logits = shift_logits[..., :label_probs.shape[-1]]
    elif shift_logits.shape[-1] < label_probs.shape[-1]:
        label_probs = label_probs[..., :shift_logits.shape[-1]]

    assert shift_logits.shape == label_probs.shape, (
        f"logits {shift_logits.shape} vs probs {label_probs.shape}")

    kl_loss = F.kl_div(
        F.log_softmax(shift_logits, dim=-1),
        label_probs, reduction='batchmean')
    lm_loss = F.cross_entropy(shift_logits, shift_labels)

    total_loss = alpha * kl_loss + (1 - alpha) * lm_loss
    return total_loss, kl_loss, lm_loss


def kl_loss_evaluate(logits, batch, vocab_size, lmbda=0.25):
    """Evaluate with interpolated log-probs."""
    knn_label = batch['knn_label']
    label_probs = batch['knn_probs']

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = batch['labels'][:, 1:].contiguous()

    nonpad_mask = shift_labels != -100
    shift_logits = shift_logits[nonpad_mask]
    shift_labels = shift_labels[nonpad_mask]
    label_probs = label_probs / label_probs.sum(dim=-1, keepdim=True)

    if shift_logits.shape[-1] > label_probs.shape[-1]:
        shift_logits = shift_logits[..., :label_probs.shape[-1]]
    elif shift_logits.shape[-1] < label_probs.shape[-1]:
        label_probs = label_probs[..., :shift_logits.shape[-1]]

    lm_log_probs = F.log_softmax(shift_logits, dim=-1)
    knn_log_probs = label_probs.log()
    knn_log_probs = torch.nan_to_num(knn_log_probs, neginf=-10000.0)

    joint_log_probs = torch.logaddexp(
        lm_log_probs + np.log(1 - lmbda),
        knn_log_probs + np.log(lmbda))

    nll_loss = F.nll_loss(joint_log_probs, shift_labels, reduction='sum')
    lm_loss = F.nll_loss(lm_log_probs, shift_labels, reduction='sum')
    token_num = shift_labels.shape[0]

    return nll_loss, lm_loss, token_num


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def knn_collate_fn(batch, knn_dstore, vocab_size, pad_token_id):
    """
    Custom collate that:
      1. Dynamically pads variable-length sequences
      2. Loads KNN distributions for the corresponding dstore_range
    """
    max_len = max(len(b['input_ids']) for b in batch)

    input_ids = []
    labels = []
    attention_mask = []
    dstore_ranges = []

    for b in batch:
        seq_len = len(b['input_ids'])
        pad_len = max_len - seq_len
        input_ids.append(b['input_ids'] + [pad_token_id] * pad_len)
        labels.append(b['labels'] + [-100] * pad_len)
        attention_mask.append(b['attention_mask'] + [0] * pad_len)
        dstore_ranges.append(b['dstore_range'])

    collated = {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    }

    # --- load KNN distributions for each example in the batch -----------
    knn_labels_list = []
    knn_probs_list = []

    for cur_range in dstore_ranges:
        start, end = int(cur_range[0]), int(cur_range[1])
        knn_slice = knn_dstore.select(range(start, end))
        slice_data = knn_slice[:]

        cur_knn_label = slice_data['label']
        cur_token_id = slice_data['token_id']
        cur_prob = slice_data['prob']

        cur_knn_prob = torch.zeros(end - start, vocab_size)
        for i in range(end - start):
            ids = cur_token_id[i]
            probs = cur_prob[i]
            # Guard against out-of-vocab IDs
            valid = ids < vocab_size
            if not isinstance(valid, torch.Tensor):
                valid = torch.tensor(valid)
            cur_knn_prob[i].scatter_(0, ids[valid].long(), probs[valid].float())

        knn_labels_list.append(cur_knn_label)
        knn_probs_list.append(cur_knn_prob)

    collated['knn_label'] = torch.cat(knn_labels_list, dim=0)
    collated['knn_probs'] = torch.cat(knn_probs_list, dim=0)

    return collated


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Memory Decoder with VLM KNN distributions")
    p.add_argument('--model_name_or_path', type=str,
                   default='Qwen/Qwen2-0.5B')
    p.add_argument('--dataset_name', type=str,
                   default='./vlm_memdec_data',
                   help='Path to HF dataset from vlm_prepare_memdec_data.py')
    p.add_argument('--knn_save_path', type=str, required=True,
                   help='Path to KNN distribution Arrow file')
    p.add_argument('--output_dir', type=str,
                   default='./vlm_memdec_checkpoints')
    # Training
    p.add_argument('--per_device_train_batch_size', type=int, default=8)
    p.add_argument('--per_device_eval_batch_size', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--num_train_epochs', type=int, default=30)
    p.add_argument('--max_train_steps', type=int, default=None)
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--lr_scheduler_type', type=str, default='linear',
                   choices=['linear', 'cosine', 'constant',
                            'constant_with_warmup'])
    p.add_argument('--num_warmup_steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    # MemDec
    p.add_argument('--alpha', type=float, default=0.5,
                   help='Weight for KL loss vs CE loss')
    p.add_argument('--lmbda', type=float, default=0.25,
                   help='Lambda for interpolation during evaluation')
    # Checkpointing
    p.add_argument('--checkpointing_steps', type=str, default='epoch')
    p.add_argument('--resume_from_checkpoint', type=str, default=None)
    p.add_argument('--logging_steps', type=int, default=1)
    # Eval
    p.add_argument('--do_test', action='store_true',
                   help='Run evaluation only (no training)')
    p.add_argument('--report_to', type=str, default='none')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps)

    # ---- logging -------------------------------------------------------
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        log_level = logging.INFO
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        log_level = logging.ERROR

    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> "
               "| <cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=log_level)

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            level = logger.level(record.levelname).name \
                if record.levelname in logger._core.levels else record.levelno
            frame, depth = inspect.currentframe(), 0
            while frame and (depth == 0
                             or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()],
                        level=log_level, force=True)

    if args.seed is not None:
        set_seed(args.seed)

    # ---- load model ----------------------------------------------------
    logger.info(f"Loading model: {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    model.resize_token_embeddings(vocab_size)
    logger.info(f"Vocab size: {vocab_size}")

    # ---- load dataset --------------------------------------------------
    logger.info(f"Loading dataset from {args.dataset_name}")
    lm_dataset = load_from_disk(args.dataset_name)

    # ---- load KNN distributions ----------------------------------------
    logger.info(f"Loading KNN dstore from {args.knn_save_path}")
    knn_dstore = Dataset.from_file(args.knn_save_path)
    knn_dstore.set_format(type='torch',
                          columns=['id_cnt', 'token_id', 'prob', 'label'])

    # ---- create collate fn ---------------------------------------------
    collate_fn = partial(
        knn_collate_fn,
        knn_dstore=knn_dstore,
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    # ---- evaluation only -----------------------------------------------
    if args.do_test:
        eval_loader = DataLoader(
            lm_dataset, collate_fn=collate_fn,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False, num_workers=2, pin_memory=True)
        model, eval_loader = accelerator.prepare(model, eval_loader)
        model.eval()

        total_joint = total_lm = total_tokens = 0
        for batch in tqdm(eval_loader, desc="Evaluating"):
            with torch.no_grad():
                out = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'])
                nll, lm, n = kl_loss_evaluate(
                    out.logits, batch, vocab_size, lmbda=args.lmbda)
                total_joint += nll.item()
                total_lm += lm.item()
                total_tokens += n

        total_joint = accelerator.gather(
            torch.tensor(total_joint, device=accelerator.device)).sum().item()
        total_lm = accelerator.gather(
            torch.tensor(total_lm, device=accelerator.device)).sum().item()
        total_tokens = accelerator.gather(
            torch.tensor(total_tokens, device=accelerator.device)).sum().item()

        logger.info(f"Joint PPL: {math.exp(total_joint / total_tokens):.4f}")
        logger.info(f"LM PPL:    {math.exp(total_lm / total_tokens):.4f}")
        return

    # ---- training setup ------------------------------------------------
    train_loader = DataLoader(
        lm_dataset, collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
        shuffle=False, num_workers=2, pin_memory=True)

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters()
                     if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                     if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # ---- training loop -------------------------------------------------
    total_batch_size = (args.per_device_train_batch_size
                        * accelerator.num_processes
                        * args.gradient_accumulation_steps)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples      = {len(lm_dataset)}")
    logger.info(f"  Num epochs        = {args.num_train_epochs}")
    logger.info(f"  Batch size/device = {args.per_device_train_batch_size}")
    logger.info(f"  Total batch size  = {total_batch_size}")
    logger.info(f"  Gradient accum    = {args.gradient_accumulation_steps}")
    logger.info(f"  Total opt steps   = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # ---- resume from checkpoint ----------------------------------------
    if args.resume_from_checkpoint:
        accelerator.print(
            f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
        training_diff = os.path.splitext(path)[0]
        if "epoch" in training_diff:
            starting_epoch = int(
                training_diff.replace("epoch_", "")) + 1
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(
                training_diff.replace("step_", "")) \
                * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_loader)
            completed_steps = resume_step // args.gradient_accumulation_steps

    progress_bar.update(completed_steps)

    log_loss = log_kl = log_lm = 0.0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        active_loader = train_loader

        for step, batch in enumerate(active_loader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=None,
                )
                loss, kl_loss, lm_loss = kl_loss_token(
                    outputs.logits, batch, vocab_size, alpha=args.alpha)

                log_loss += loss.detach().float()
                log_kl += kl_loss.detach().float()
                log_lm += lm_loss.detach().float()

                accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1

                if (args.logging_steps
                        and completed_steps % args.logging_steps == 0):
                    actual = (args.gradient_accumulation_steps
                              if not accelerator.gradient_state.end_of_dataloader
                              else len(active_loader)
                                   % args.gradient_accumulation_steps
                                   or args.gradient_accumulation_steps)
                    denom = actual * args.logging_steps
                    avg_loss = accelerator.gather(log_loss).mean().item() / denom
                    avg_kl = accelerator.gather(log_kl).mean().item() / denom
                    avg_lm = accelerator.gather(log_lm).mean().item() / denom

                    if accelerator.is_main_process:
                        logger.info(
                            f"epoch {epoch} step {completed_steps}  "
                            f"loss={avg_loss:.4f}  "
                            f"kl={avg_kl:.4f}  "
                            f"lm={avg_lm:.4f}  "
                            f"lr={lr_scheduler.get_last_lr()[0]:.2e}")
                    log_loss = log_kl = log_lm = 0.0

                if (isinstance(checkpointing_steps, int)
                        and completed_steps % checkpointing_steps == 0
                        and accelerator.sync_gradients):
                    out_dir = os.path.join(
                        args.output_dir, f"step_{completed_steps}")
                    accelerator.save_state(out_dir)
                    _save_model(accelerator, model, tokenizer, out_dir)

            if completed_steps >= args.max_train_steps:
                break

        # ---- epoch checkpoint ------------------------------------------
        if args.checkpointing_steps == "epoch":
            out_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            accelerator.save_state(out_dir)
            _save_model(accelerator, model, tokenizer, out_dir)

    # ---- final save ----------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    _save_model(accelerator, model, tokenizer, final_dir)
    logger.info(f"Training complete. Final model saved to {final_dir}")


def _save_model(accelerator, model, tokenizer, output_dir):
    """Unwrap and save model + tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    if accelerator.is_main_process:
        unwrapped.config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    main()
