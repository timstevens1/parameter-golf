"""
Compare in-loop runs: train/val loss vs wallclock and step.
Reads ./grad_interpolation/results/inloop_<mode>.jsonl for each mode passed
on the command line, default {adam, spectral}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main(modes: list[str], results_dir: Path, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    colors = {"adam": "tab:blue", "spectral": "tab:orange", "armijo": "tab:green"}
    for mode in modes:
        records = load_jsonl(results_dir / f"inloop_{mode}.jsonl")
        if not records:
            print(f"WARN: no records for mode={mode}")
            continue
        steps = [r["step"] for r in records]
        wall = [r["wallclock_s"] for r in records]
        loss = [r["loss_w"] for r in records]
        val = [(r["step"], r["val_loss"]) for r in records if "val_loss" in r]
        eta = [(r["step"], r["eta"]) for r in records if "eta" in r]

        c = colors.get(mode, "k")
        axes[0, 0].plot(steps, loss, "-", color=c, label=mode, alpha=0.8)
        axes[0, 1].plot(wall, loss, "-", color=c, label=mode, alpha=0.8)
        if val:
            vs, vL = zip(*val)
            axes[1, 0].plot(vs, vL, "-o", color=c, label=mode, ms=3)
        if eta:
            es, eL = zip(*eta)
            axes[1, 1].plot(es, eL, "-", color=c, label=mode, alpha=0.8)

    axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("train loss")
    axes[0, 0].set_title("train loss vs step")
    axes[0, 1].set_xlabel("wallclock (s)"); axes[0, 1].set_ylabel("train loss")
    axes[0, 1].set_title("train loss vs wallclock")
    axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("val loss (fixed batch)")
    axes[1, 0].set_title("val loss vs step")
    axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("chosen eta")
    axes[1, 1].set_title("step size history")
    axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["adam", "spectral"])
    parser.add_argument("--results-dir", default="grad_interpolation/results")
    parser.add_argument("--out", default="grad_interpolation/results/inloop_compare.png")
    args = parser.parse_args()
    main(args.modes, Path(args.results_dir), Path(args.out))
