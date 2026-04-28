"""
Aggregate all V4 eval JSONs into a single Markdown report.

Reads zoom_decoder/eval_v4/*.json (output by zoom_decoder.evaluate in V4 mode,
each containing {"version":"v4", "summary": {split_name: {...}}, "records": ...}})
and emits a Markdown document with per-group, per-split tables matching the
shape used in PRIV_RESULTS.md.

Usage:
    python -m zoom_decoder.summarize_v4 \
        --eval_dir ./zoom_decoder/eval_v4 \
        --out_file ./zoom_decoder/V4_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# Configuration order for the report
WITH_ZOOM_GROUP: List[str] = [
    "base",
    "zd_24L_perc",
    "zd_priv_24L_perc",
    "zd_zoompriv_24L_perc",
    "zd_24L_noap_perc",
    "zd_24L_nosw_perc",
]
WITHOUT_ZOOM_GROUP: List[str] = [
    "base",
    "zd_priv_24L_all",
    "zd_knn_6L_all",
    "zd_knn_6L_noap_all",
    "zd_knn_6L_nosw_all",
    "zd_knn_6L_plain_all",
    "zd_knn_24L_all",
    "lora_r16_all",
]

DIFF_KEYS = ["easy", "medium", "hard", "extreme"]
TASK_ORDER_PERC = ["animal", "block_recognition", "color_block", "letter", "shape", "text"]


def load_summary(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    return d.get("summary", {})


def fmt(x):
    if x is None:
        return "—"
    return f"{x:.2f}"


def diff_row(name: str, summ: dict, baseline: Optional[dict]) -> str:
    overall = summ.get("overall_acc")
    n = summ.get("n", 0)
    parts = [name, fmt(overall)]
    by_d = summ.get("by_difficulty", {})
    for k in DIFF_KEYS:
        v = by_d.get(k, {}).get("acc")
        parts.append(fmt(v))
    if baseline is None:
        delta = "—"
    else:
        b_over = baseline.get("overall_acc")
        if b_over is None or overall is None:
            delta = "—"
        else:
            d = overall - b_over
            delta = ("+" if d >= 0 else "") + f"{d:.2f}"
    parts.append(delta)
    parts.append(str(n))
    return "| " + " | ".join(parts) + " |"


def task_row(name: str, summ: dict, task_keys: List[str]) -> str:
    by_t = summ.get("by_task", {})
    parts = [name]
    for k in task_keys:
        v = by_t.get(k, {}).get("acc")
        parts.append(fmt(v))
    return "| " + " | ".join(parts) + " |"


def render_split_block(title: str, eval_split: str, group: List[str],
                       summaries: Dict[str, Dict[str, dict]]) -> str:
    out = [f"### {title} (eval = {eval_split})", ""]
    out.append("| 配置 | Overall | easy | medium | hard | extreme | Δ vs base | N |")
    out.append("| :-- | --: | --: | --: | --: | --: | --: | --: |")
    base_summ = summaries.get("base", {}).get(eval_split)
    for cfg in group:
        s = summaries.get(cfg, {}).get(eval_split)
        if s is None:
            out.append(f"| {cfg} | _missing_ |  |  |  |  |  |  |")
            continue
        out.append(diff_row(cfg, s, base_summ if cfg != "base" else None))
    out.append("")

    # Task breakdown — discover task keys from any non-empty summary
    task_keys: List[str] = []
    for cfg in group:
        s = summaries.get(cfg, {}).get(eval_split)
        if s and s.get("by_task"):
            task_keys = sorted(s["by_task"].keys())
            break
    if task_keys:
        out.append("**按任务分解**")
        out.append("")
        out.append("| 配置 | " + " | ".join(task_keys) + " |")
        out.append("| :-- | " + " | ".join(["--:"] * len(task_keys)) + " |")
        for cfg in group:
            s = summaries.get(cfg, {}).get(eval_split)
            if s is None:
                continue
            out.append(task_row(cfg, s, task_keys))
        out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="./zoom_decoder/eval_v4")
    p.add_argument("--out_file", default="./zoom_decoder/V4_RESULTS.md")
    p.add_argument("--splits_file", default="./zoom_decoder/splits_v4.json")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    summaries: Dict[str, Dict[str, dict]] = {}
    for j in sorted(eval_dir.glob("*.json")):
        s = load_summary(j)
        if s is None:
            continue
        summaries[j.stem] = s

    splits_meta = {}
    sp = Path(args.splits_file)
    if sp.exists():
        meta = json.loads(sp.read_text())
        for k in ("perception", "reasoning"):
            if k in meta:
                splits_meta[k] = {
                    "train": len(meta[k].get("train", [])),
                    "val": len(meta[k].get("val", [])),
                }

    md: List[str] = []
    md.append("# V4 实验结果\n")
    md.append("> 数据划分：每个 (task_type × difficulty) cell 70 条 train / 30 条 val（seed=0）。")
    md.append("> 不再保留 V3 的 test split——所有评估都在 val 集上。\n")
    if splits_meta:
        md.append("**Split 规模**")
        md.append("")
        md.append("| split | train | val |")
        md.append("| :-- | --: | --: |")
        for k, v in splits_meta.items():
            md.append(f"| {k} | {v['train']} | {v['val']} |")
        md.append("")

    md.append("## 1. 有 Zoom 组（仅 perception 训练）\n")
    md.append("> **训练数据**：FineSightBench/perception 的 train split。")
    md.append("> **评估**：perception-val（同分布）+ reasoning-val（分布外泛化）。\n")
    md.append(render_split_block(
        "1.1 perception-val（同分布）", "perception", WITH_ZOOM_GROUP, summaries))
    md.append(render_split_block(
        "1.2 reasoning-val（分布外泛化）", "reasoning", WITH_ZOOM_GROUP, summaries))

    md.append("## 2. 无 Zoom 组（perception + reasoning 联合训练）\n")
    md.append("> **训练数据**：perception.train ∪ reasoning.train，shuffle 后混合 batch。")
    md.append("> **评估**：perception-val + reasoning-val 分别报告。\n")
    md.append(render_split_block(
        "2.1 perception-val", "perception", WITHOUT_ZOOM_GROUP, summaries))
    md.append(render_split_block(
        "2.2 reasoning-val", "reasoning", WITHOUT_ZOOM_GROUP, summaries))

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text("\n".join(md))
    print(f"Wrote {args.out_file}  ({len(summaries)} configs)")


if __name__ == "__main__":
    main()
