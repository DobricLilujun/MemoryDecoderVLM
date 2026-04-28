"""
Aggregate V4 + large-model experiment eval JSONs and summarize model upper bounds.

Usage:
  python -m zoom_decoder.summarize_all_upper_bounds \
    --root ./zoom_decoder \
    --out_file ./zoom_decoder/ALL_MODEL_UPPER_BOUNDS.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SplitBest:
    config: str
    overall: float


@dataclass
class ExpSummary:
    key: str
    title: str
    eval_dir: Path
    report_file: Path
    base_perc: Optional[float]
    base_reas: Optional[float]
    best_perc: Optional[SplitBest]
    best_reas: Optional[SplitBest]
    n_json: int


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _extract_overall(summ: dict, split: str) -> Optional[float]:
    if not isinstance(summ, dict):
        return None
    sp = summ.get(split)
    if not isinstance(sp, dict):
        return None
    v = sp.get("overall_acc")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _scan_eval_dir(eval_dir: Path) -> Tuple[int, Optional[float], Optional[float], Optional[SplitBest], Optional[SplitBest]]:
    if not eval_dir.exists():
        return 0, None, None, None, None

    files = sorted(p for p in eval_dir.glob("*.json") if p.is_file())
    base_perc = None
    base_reas = None
    best_perc: Optional[SplitBest] = None
    best_reas: Optional[SplitBest] = None

    for p in files:
        d = _load_json(p)
        if d is None:
            continue
        summary = d.get("summary", {})
        cfg = p.stem

        perc = _extract_overall(summary, "perception")
        reas = _extract_overall(summary, "reasoning")

        if cfg == "base":
            base_perc = perc
            base_reas = reas

        if perc is not None and (best_perc is None or perc > best_perc.overall):
            best_perc = SplitBest(config=cfg, overall=perc)

        if reas is not None and (best_reas is None or reas > best_reas.overall):
            best_reas = SplitBest(config=cfg, overall=reas)

    return len(files), base_perc, base_reas, best_perc, best_reas


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def _fmt_gain(best: Optional[SplitBest], base: Optional[float]) -> str:
    if best is None or base is None:
        return "—"
    g = best.overall - base
    sign = "+" if g >= 0 else ""
    return f"{sign}{g:.2f}"


def _exp_templates(root: Path) -> List[Tuple[str, str, str, str]]:
    return [
        ("v4", "V4 Baseline (Qwen2-VL-2B + Qwen2-0.5B)", "eval_v4", "V4_RESULTS.md"),
        ("v4_large", "V4 Large (Qwen3-VL-4B + Qwen3-1.7B)", "eval_v4_large", "V4_LARGE_RESULTS.md"),
        ("v4_large1", "V4 Large+1 (Qwen3-VL-4B + Qwen2-0.5B)", "eval_v4_large1", "V4_LARGE1_RESULTS.md"),
        ("v4_large2", "V4 Large+2 (Qwen3-VL-8B + Qwen3-4B)", "eval_v4_large2", "V4_LARGE2_RESULTS.md"),
    ]


def build_summary(root: Path) -> List[ExpSummary]:
    out: List[ExpSummary] = []
    for key, title, edir, rfile in _exp_templates(root):
        eval_dir = root / edir
        report_file = root / rfile
        n_json, bp, br, bestp, bestr = _scan_eval_dir(eval_dir)
        out.append(
            ExpSummary(
                key=key,
                title=title,
                eval_dir=eval_dir,
                report_file=report_file,
                base_perc=bp,
                base_reas=br,
                best_perc=bestp,
                best_reas=bestr,
                n_json=n_json,
            )
        )
    return out


def render_markdown(summaries: List[ExpSummary]) -> str:
    lines: List[str] = []
    lines.append("# 所有实验模型上限汇总")
    lines.append("")
    lines.append("> 该文件从各实验 eval JSON 自动汇总。若某实验尚未跑完，会显示为 _pending_。")
    lines.append("")

    lines.append("## 1) 单实验上限")
    lines.append("")
    lines.append("| 实验 | eval JSON 数 | perception base | perception best(config) | Δ | reasoning base | reasoning best(config) | Δ | 报告 |")
    lines.append("| :-- | --: | --: | :-- | --: | --: | :-- | --: | :-- |")

    for s in summaries:
        pbest = (
            f"{s.best_perc.config} ({s.best_perc.overall:.2f})"
            if s.best_perc is not None
            else "_pending_"
        )
        rbest = (
            f"{s.best_reas.config} ({s.best_reas.overall:.2f})"
            if s.best_reas is not None
            else "_pending_"
        )
        report_cell = f"存在 ({s.report_file.name})" if s.report_file.exists() else "_pending_"
        lines.append(
            "| "
            + " | ".join(
                [
                    s.title,
                    str(s.n_json),
                    _fmt(s.base_perc),
                    pbest,
                    _fmt_gain(s.best_perc, s.base_perc),
                    _fmt(s.base_reas),
                    rbest,
                    _fmt_gain(s.best_reas, s.base_reas),
                    report_cell,
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## 2) 跨实验全局上限")
    lines.append("")

    best_perc: Optional[Tuple[ExpSummary, SplitBest]] = None
    best_reas: Optional[Tuple[ExpSummary, SplitBest]] = None

    for s in summaries:
        if s.best_perc is not None:
            if best_perc is None or s.best_perc.overall > best_perc[1].overall:
                best_perc = (s, s.best_perc)
        if s.best_reas is not None:
            if best_reas is None or s.best_reas.overall > best_reas[1].overall:
                best_reas = (s, s.best_reas)

    if best_perc is None:
        lines.append("- perception 全局上限：_pending_")
    else:
        lines.append(
            f"- perception 全局上限：{best_perc[1].overall:.2f} "
            f"（{best_perc[0].title} / {best_perc[1].config}）"
        )

    if best_reas is None:
        lines.append("- reasoning 全局上限：_pending_")
    else:
        lines.append(
            f"- reasoning 全局上限：{best_reas[1].overall:.2f} "
            f"（{best_reas[0].title} / {best_reas[1].config}）"
        )

    lines.append("")
    lines.append("## 3) 结论模板（自动更新）")
    lines.append("")
    lines.append("- 若 Large/XLarge 组的全局上限超过 V4 baseline，可归因为更强 VLM 主干和/或更大 DEC 容量。")
    lines.append("- 若同一 VLM 下 `Qwen3-1.7B` vs `Qwen2-0.5B` 差异显著，可直接归因于解码器容量差。")
    lines.append("- 若 8B VLM 在 reasoning 提升更明显，说明跨任务组合推理更依赖视觉主干容量。")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./zoom_decoder")
    p.add_argument("--out_file", default="./zoom_decoder/ALL_MODEL_UPPER_BOUNDS.md")
    args = p.parse_args()

    root = Path(args.root)
    out_file = Path(args.out_file)

    summaries = build_summary(root)
    md = render_markdown(summaries)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md)
    print(f"Wrote {out_file} ({len(summaries)} experiments)")


if __name__ == "__main__":
    main()
