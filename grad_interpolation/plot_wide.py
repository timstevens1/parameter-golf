"""Plot wide-interval fits: ground truth + per-N predictions + errors."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_one(json_path: Path, out_path: Path):
    with json_path.open() as f:
        d = json.load(f)
    eta_far = d["eta_far"]
    natural = d["natural_step_norm"]
    vis_e = np.array(d["vis_etas"])
    vis_L = np.array(d["vis_loss"])
    queries = np.array(d["queries"])
    L_q = np.array(d["L_queries"])
    fits = d["fits"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(vis_e, vis_L, "k-", lw=2, label="ground truth")
    cmap = plt.cm.viridis
    for i, fit in enumerate(fits):
        c = cmap(i / max(len(fits) - 1, 1))
        n = fit["n_nodes"]
        ax.plot(vis_e, fit["vis_pred_plain"], "-", color=c, alpha=0.7,
                label=f"plain N={n}")
        ax.plot(vis_e, fit["vis_pred_hermite"], "--", color=c, alpha=0.5,
                label=f"hermite N={n}")
        nodes = np.array(fit["nodes"])
        L_n = np.array(fit["L_nodes"])
        ax.plot(nodes, L_n, "o", color=c, ms=3, alpha=0.8)
    gt = d["gt_min"]
    ax.axvline(gt["eta"], color="red", ls=":", alpha=0.5,
                label=f"GT min eta={gt['eta']:.2g} ({gt['adam_steps']:.1f} Adam-steps)")
    ax.set_ylabel("loss")
    ax.set_title(
        f"step={d['config']['target_step']}  Adam direction  "
        f"eta_far={eta_far:.3g} ({d['config']['eta_far_mult']:.0f} Adam-steps)  "
        f"L range: {vis_L.min():.2f}..{vis_L.max():.2f}"
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, fit in enumerate(fits):
        c = cmap(i / max(len(fits) - 1, 1))
        n = fit["n_nodes"]
        err_p = np.array(fit["err_plain"])
        err_h = np.array(fit["err_hermite"])
        ax.semilogy(queries, np.abs(err_p) + 1e-12, "o", color=c, ms=4,
                    label=f"plain N={n}")
        ax.semilogy(queries, np.abs(err_h) + 1e-12, "x", color=c, ms=6,
                    mew=1.2, alpha=0.6, label=f"hermite N={n}")
    ax.set_xlabel("eta")
    ax.set_ylabel("|err| at query")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_summary(json_paths: list[Path], out_path: Path):
    """RMSE vs N across multiple wide-interval runs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for jp in json_paths:
        with jp.open() as f:
            d = json.load(f)
        ns = [fit["n_nodes"] for fit in d["fits"]]
        rmse_p = [fit["rmse"]["plain"] for fit in d["fits"]]
        rmse_h = [fit["rmse"]["hermite"] for fit in d["fits"]]
        label = f"x{int(d['config']['eta_far_mult'])}"
        axes[0].loglog(ns, rmse_p, "-o", label=f"plain {label}", alpha=0.8)
        axes[1].loglog(ns, rmse_h, "-s", label=f"hermite {label}", alpha=0.8)
    for ax, title in zip(axes, ["plain", "hermite"]):
        ax.set_xlabel("N nodes")
        ax.set_ylabel("RMSE on in-interval queries")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(json_pattern: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(p) for p in glob.glob(json_pattern))
    if not paths:
        raise FileNotFoundError(f"no JSONs match {json_pattern}")
    for p in paths:
        out_path = out_dir / f"wide_{p.stem}.png"
        plot_one(p, out_path)
        print(f"wrote {out_path}")
    summary_path = out_dir / "wide_summary.png"
    plot_summary(paths, summary_path)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="grad_interpolation/results/wide_step*.json")
    parser.add_argument("--out-dir", default="grad_interpolation/results")
    args = parser.parse_args()
    main(args.pattern, Path(args.out_dir))
