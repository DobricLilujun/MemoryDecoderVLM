#!/usr/bin/env python3
"""Parse pipeline logs under output/ and generate a training-loss comparison figure."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

LOSS_RE = re.compile(r"(?:\bloss\b\s*[:=]\s*|\bloss\s+)([0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)")


def parse_losses(log_file: Path) -> List[float]:
    losses: List[float] = []
    for line in log_file.read_text(errors="ignore").splitlines():
        m = LOSS_RE.search(line)
        if not m:
            continue
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if v >= 0:
            losses.append(v)
    return losses


def latest_pipeline_log(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("pipeline_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    return logs[-1]


def collect(output_root: Path) -> Dict[str, List[float]]:
    series: Dict[str, List[float]] = {}
    if not output_root.exists():
        return series

    for exp_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        log_dir = exp_dir / "logs"
        if not log_dir.exists():
            continue
        log_file = latest_pipeline_log(log_dir)
        if log_file is None:
            continue
        losses = parse_losses(log_file)
        if losses:
            series[exp_dir.name] = losses
    return series


def write_csv(csv_path: Path, series: Dict[str, List[float]]) -> None:
    max_len = max((len(v) for v in series.values()), default=0)
    headers = ["step"] + list(series.keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(max_len):
            row: List[str | float | int] = [i + 1]
            for k in series.keys():
                vals = series[k]
                row.append(vals[i] if i < len(vals) else "")
            w.writerow(row)


def plot_png(png_path: Path, series: Dict[str, List[float]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(11, 6))
    for name, vals in series.items():
        x = list(range(1, len(vals) + 1))
        plt.plot(x, vals, linewidth=1.3, label=name)

    plt.title("Training Loss Comparison Across V4 Experiments")
    plt.xlabel("Logged Step Index")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=180)
    plt.close()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default="./output")
    ap.add_argument("--out_file", default="./output/loss_compare.png")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    out_png = Path(args.out_file)
    out_csv = out_png.with_suffix(".csv")

    series = collect(output_root)
    if not series:
        print(f"No loss series found under {output_root}")
        return

    write_csv(out_csv, series)
    ok = plot_png(out_png, series)
    if ok:
        print(f"Wrote figure: {out_png}")
    else:
        print("matplotlib unavailable, skipped PNG")
    print(f"Wrote table: {out_csv}")


if __name__ == "__main__":
    main()
