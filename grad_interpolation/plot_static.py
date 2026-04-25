"""
Plot the fit-then-query static probe results.

For each (stage, direction) we render a panel:
  - ground-truth curve over the wide window [eta_lo, eta_hi]
  - plain and Hermite predicted curves (extrapolating outside [0, eta_max])
  - fit nodes as filled circles
  - query points colored by region (below / inside / above) and shape-coded
    by ground-truth value vs predicted value
  - bottom strip: per-query absolute error on a log scale, colored by region

A summary panel aggregates per-region RMSE across all (stage, direction) cells,
plus a final coefficient-decay panel for the full-interval fits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REGION_COLORS = {"below": "tab:red", "inside": "tab:blue", "above": "tab:purple"}


def plot_stage_direction(stage_step: int, direction_kind: str, data: dict, out_path: Path):
    eta_lo = data["eta_lo"]; eta_hi = data["eta_hi"]; eta_max = data["eta_max"]
    fit_nodes = np.array(data["fit_nodes"])
    fit_loss = np.array(data["fit_loss"])
    queries = np.array(data["queries"])
    region = np.array(data["query_region"])
    L_truth = np.array(data["loss_truth"])
    Lhat_p = np.array(data["loss_plain"])
    Lhat_h = np.array(data["loss_hermite"])
    err_p = np.array(data["err_plain"])
    err_h = np.array(data["err_hermite"])
    vis_e = np.array(data["vis_etas"])
    vis_L = np.array(data["vis_loss"])
    vis_pp = np.array(data["vis_pred_plain"])
    vis_ph = np.array(data["vis_pred_hermite"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(vis_e, vis_L, "k-", lw=2, label="ground truth")
    ax.plot(vis_e, vis_pp, "-", color="tab:green", alpha=0.7, label="plain")
    ax.plot(vis_e, vis_ph, "--", color="tab:orange", alpha=0.7, label="hermite")
    ax.plot(fit_nodes, fit_loss, "o", color="black", ms=6, label="fit nodes")
    ax.axvspan(0.0, eta_max, color="tab:gray", alpha=0.08, label="fit interval")
    # Query markers: ground truth as squares, predicted as x's, lines connecting
    for i, q in enumerate(queries):
        c = REGION_COLORS[region[i]]
        ax.plot([q, q], [Lhat_p[i], L_truth[i]], "-", color=c, alpha=0.3, lw=0.5)
        ax.plot(q, L_truth[i], "s", color=c, ms=4, alpha=0.8)
    ax.set_ylabel("loss")
    span = vis_L.max() - vis_L.min()
    ax.set_ylim(vis_L.min() - 0.15 * span, vis_L.max() + 0.5 * span)
    ax.set_title(f"step={stage_step}  direction={direction_kind}  "
                 f"eta_max={eta_max:.3g}  fit-N={len(fit_nodes)}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Error panel
    ax = axes[1]
    for r in ("below", "inside", "above"):
        mask = region == r
        if not mask.any():
            continue
        ax.semilogy(queries[mask], np.abs(err_p[mask]) + 1e-12, "o",
                    color=REGION_COLORS[r], ms=5, label=f"plain {r}")
        ax.semilogy(queries[mask], np.abs(err_h[mask]) + 1e-12, "x",
                    color=REGION_COLORS[r], ms=8, mew=1.5, label=f"hermite {r}")
    ax.axvspan(0.0, eta_max, color="tab:gray", alpha=0.08)
    ax.axvline(0.0, color="k", lw=0.5, ls=":")
    ax.axvline(eta_max, color="k", lw=0.5, ls=":")
    ax.set_xlabel("eta")
    ax.set_ylabel("|err| at query")
    ax.legend(fontsize=7, ncol=3, loc="upper center")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_summary(stages: list[dict], out_path: Path):
    """Bar chart of per-region RMSE for plain vs hermite, grouped by (stage, direction)."""
    rows = []
    for stage in stages:
        for kind, data in stage["directions"].items():
            for region, r in data["rmse"].items():
                if r is None or region == "all":
                    continue
                rows.append({
                    "stage": stage["step"], "kind": kind, "region": region,
                    "plain": r["plain"], "hermite": r["hermite"], "n": r["n"],
                })

    stages_set = sorted({r["stage"] for r in rows})
    kinds = ["adam", "grad", "random"]
    regions = ["below", "inside", "above"]

    fig, axes = plt.subplots(len(stages_set), len(kinds),
                              figsize=(4 * len(kinds), 3.0 * len(stages_set)),
                              squeeze=False)
    for i, st in enumerate(stages_set):
        for j, kd in enumerate(kinds):
            ax = axes[i, j]
            xs = np.arange(len(regions))
            width = 0.35
            plain_y = [next((r["plain"] for r in rows
                            if r["stage"] == st and r["kind"] == kd and r["region"] == rg),
                            np.nan) for rg in regions]
            herm_y = [next((r["hermite"] for r in rows
                           if r["stage"] == st and r["kind"] == kd and r["region"] == rg),
                           np.nan) for rg in regions]
            ax.bar(xs - width/2, plain_y, width, label="plain",
                    color="tab:green", alpha=0.7)
            ax.bar(xs + width/2, herm_y, width, label="hermite",
                    color="tab:orange", alpha=0.7)
            ax.set_yscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels(regions)
            ax.set_title(f"step={st}  {kd}")
            if j == 0:
                ax.set_ylabel("RMSE")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_coefficient_decay(stages: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4),
                              squeeze=False)
    axes = axes[0]
    kinds = ["adam", "grad", "random"]
    colors = {"adam": "tab:blue", "grad": "tab:orange", "random": "tab:green"}
    for j, stage in enumerate(stages):
        ax = axes[j]
        for kind in kinds:
            data = stage["directions"][kind]
            cp = np.array(data["decay_plain"]["abs_coeffs"])
            ch = np.array(data["decay_hermite"]["abs_coeffs"])
            ax.semilogy(np.arange(len(cp)), cp + 1e-15, "-o",
                        color=colors[kind], alpha=0.8, label=f"{kind} plain")
            ax.semilogy(np.arange(len(ch)), ch + 1e-15, "--s",
                        color=colors[kind], alpha=0.5, label=f"{kind} hermite")
        ax.set_xlabel("Chebyshev coefficient index k")
        ax.set_ylabel("|c_k|")
        ax.set_title(f"step={stage['step']}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(json_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open() as f:
        results = json.load(f)
    stages = results["stages"]
    for stage in stages:
        for kind, data in stage["directions"].items():
            out_path = out_dir / f"static_step{stage['step']}_{kind}.png"
            plot_stage_direction(stage["step"], kind, data, out_path)
            print(f"wrote {out_path}")
    summary_path = out_dir / "static_summary.png"
    plot_summary(stages, summary_path)
    print(f"wrote {summary_path}")
    decay_path = out_dir / "static_coefficient_decay.png"
    plot_coefficient_decay(stages, decay_path)
    print(f"wrote {decay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="grad_interpolation/results/static.json")
    parser.add_argument("--out-dir", default="grad_interpolation/results")
    args = parser.parse_args()
    main(Path(args.json), Path(args.out_dir))
