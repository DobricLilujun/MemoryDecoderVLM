"""Shared utilities for FineSightBench."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image

SIZE_BUCKETS = [4, 8, 12, 16, 24, 32, 48]
DIFFICULTY_BUCKETS = ["extreme", "hard", "medium", "easy"]


@dataclass
class Target:
    value: str
    size: int
    cx: float
    cy: float

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        s = self.size
        return (self.cx - s / 2, self.cy - s / 2, self.cx + s / 2, self.cy + s / 2)


def parse_targets(metadata_json: str) -> List[Target]:
    md = json.loads(metadata_json)
    out: List[Target] = []
    for t in md.get("targets", []):
        pos = t.get("position", [0, 0])
        out.append(
            Target(
                value=str(t.get("value", "")),
                size=int(t.get("size", 0)),
                cx=float(pos[0]),
                cy=float(pos[1]),
            )
        )
    return out


def union_bbox(targets: List[Target], canvas: Tuple[int, int]) -> Tuple[float, float, float, float]:
    if not targets:
        return (0, 0, canvas[0], canvas[1])
    xs1, ys1, xs2, ys2 = [], [], [], []
    for t in targets:
        x1, y1, x2, y2 = t.bbox
        xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
    return (min(xs1), min(ys1), max(xs2), max(ys2))


def zoom_crop(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    out_size: int = 448,
    pad_ratio: float = 2.5,
    min_crop: int = 64,
) -> Image.Image:
    W, H = image.size
    x1, y1, x2, y2 = bbox
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(float(min_crop), pad_ratio * max(bw, bh))
    half = side / 2.0
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), H - half)
    left, top = int(round(cx - half)), int(round(cy - half))
    right, bottom = int(round(cx + half)), int(round(cy + half))
    left = max(0, left); top = max(0, top)
    right = min(W, right); bottom = min(H, bottom)
    crop = image.crop((left, top, right, bottom))
    if crop.size[0] <= 1 or crop.size[1] <= 1:
        crop = image
    return crop.resize((out_size, out_size), Image.BICUBIC)


def min_target_size(targets: List[Target]) -> int:
    if not targets:
        return 48
    return int(min(t.size for t in targets))


def size_to_bucket_idx(size: int) -> int:
    best = 0
    best_d = 1e9
    for i, s in enumerate(SIZE_BUCKETS):
        d = abs(s - size)
        if d < best_d:
            best_d = d
            best = i
    return best


def scale_weight(size: int, ref: int = 48, gamma: float = 1.0, cap: float = 8.0) -> float:
    w = (ref / max(1, size)) ** gamma
    return float(min(cap, w))


def stratified_split(
    dataset,
    seed: int = 0,
    train_per_cell: int = 70,
    val_per_cell: int = 15,
    test_per_cell: int = 15,
) -> Tuple[List[int], List[int], List[int]]:
    """V3 three-way split (kept for backward compatibility)."""
    rng = random.Random(seed)
    by_cell: dict = {}
    for i, s in enumerate(dataset):
        key = (s["task_type"], s["difficulty"])
        by_cell.setdefault(key, []).append(i)
    train, val, test = [], [], []
    for key, idxs in by_cell.items():
        rng.shuffle(idxs)
        tr = idxs[:train_per_cell]
        v = idxs[train_per_cell: train_per_cell + val_per_cell]
        te = idxs[train_per_cell + val_per_cell: train_per_cell + val_per_cell + test_per_cell]
        train.extend(tr); val.extend(v); test.extend(te)
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def stratified_train_val_split(
    dataset,
    seed: int = 0,
    train_per_cell: int = 70,
    val_per_cell: int = 30,
) -> Tuple[List[int], List[int]]:
    """V4 two-way split: stratified by (task_type, difficulty), 70/30 by default."""
    rng = random.Random(seed)
    by_cell: dict = {}
    for i, s in enumerate(dataset):
        key = (s["task_type"], s["difficulty"])
        by_cell.setdefault(key, []).append(i)
    train, val = [], []
    for key, idxs in by_cell.items():
        rng.shuffle(idxs)
        tr = idxs[:train_per_cell]
        v = idxs[train_per_cell: train_per_cell + val_per_cell]
        train.extend(tr); val.extend(v)
    rng.shuffle(train); rng.shuffle(val)
    return train, val
