"""
V4 split generator.

Produces ONE canonical splits.json shared by every V4 configuration, so that
all teacher-prep / training / eval scripts see byte-identical train/val sets.

Schema:
{
  "version": "v4",
  "seed": 0,
  "train_per_cell": 70,
  "val_per_cell": 30,
  "perception": {"train": [int, ...], "val": [int, ...]},
  "reasoning":  {"train": [int, ...], "val": [int, ...]}
}

Indices are local to each FineSightBench split.

Usage:
    python -m zoom_decoder.make_splits \
        --out_file ./zoom_decoder/splits_v4.json \
        --splits perception reasoning \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger

from zoom_decoder.data_utils import stratified_train_val_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_file", default="./zoom_decoder/splits_v4.json")
    p.add_argument("--splits", nargs="+", default=["perception", "reasoning"],
                   choices=["perception", "reasoning"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_per_cell", type=int, default=70)
    p.add_argument("--val_per_cell", type=int, default=30)
    args = p.parse_args()

    out = {
        "version": "v4",
        "seed": args.seed,
        "train_per_cell": args.train_per_cell,
        "val_per_cell": args.val_per_cell,
    }
    for split in args.splits:
        logger.info("Loading FineSightBench {}", split)
        ds = load_dataset("Volavion/FineSightBench")[split]
        train_idx, val_idx = stratified_train_val_split(
            ds, seed=args.seed,
            train_per_cell=args.train_per_cell,
            val_per_cell=args.val_per_cell,
        )
        logger.info("  {}: total={}  train={}  val={}",
                    split, len(ds), len(train_idx), len(val_idx))
        out[split] = {"train": train_idx, "val": val_idx}

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(out, indent=2))
    logger.success("Saved → {}", args.out_file)


if __name__ == "__main__":
    main()
