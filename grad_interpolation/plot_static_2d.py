"""Plot 2D static probe results: ground truth + predicted contours + minima."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_one(d: dict, r: dict, out_path: Path):
    fine_e1 = np.array(r["fine_e1"])
    fine_e2 = np.array(r["fine_e2"])
    Z_dense = np.array(r["Z_dense"]).T   # transpose so rows=eta2, cols=eta1 for imshow
    Z_pred = np.array(r["Z_pred"]).T
    nodes_e1 = np.array(r["etas1_nodes"])
    nodes_e2 = np.array(r["etas2_nodes"])
    L0 = d["L0"]
    m1d = r["min_1d_at_e2_0"]
    m2d = r["min_2d"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vmin = min(Z_dense.min(), Z_pred.min())
    vmax = max(Z_dense.max(), Z_pred.max())

    extent = [fine_e1.min(), fine_e1.max(), fine_e2.min(), fine_e2.max()]

    # Ground truth contour
    ax = axes[0]
    im = ax.imshow(Z_dense, origin="lower", extent=extent, aspect="auto",
                    vmin=vmin, vmax=vmax, cmap="viridis")
    cs = ax.contour(fine_e1, fine_e2, Z_dense, levels=18, colors="white",
                     linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    plt.colorbar(im, ax=ax)
    # Sample nodes
    XX, YY = np.meshgrid(nodes_e1, nodes_e2, indexing="xy")
    ax.scatter(XX.ravel(), YY.ravel(), c="red", s=12, marker="o",
                edgecolors="white", linewidths=0.5, label="fit nodes")
    ax.scatter([m1d["eta1"]], [m1d["eta2"]], c="cyan", s=70, marker="*",
                edgecolors="black", linewidths=0.7, label=f"1D min L={m1d['loss']:.4f}")
    ax.scatter([m2d["eta1"]], [m2d["eta2"]], c="yellow", s=120, marker="*",
                edgecolors="black", linewidths=0.7, label=f"2D min L={m2d['loss']:.4f}")
    ax.axhline(0, color="white", lw=0.5, alpha=0.5)
    ax.set_xlabel("eta1 (along d1)")
    ax.set_ylabel("eta2 (along d2)")
    ax.set_title(f"Ground truth  d2={r['d2_kind']}  L0={L0:.4f}")
    ax.legend(fontsize=7, loc="upper right")

    # Predicted contour
    ax = axes[1]
    im = ax.imshow(Z_pred, origin="lower", extent=extent, aspect="auto",
                    vmin=vmin, vmax=vmax, cmap="viridis")
    cs = ax.contour(fine_e1, fine_e2, Z_pred, levels=18, colors="white",
                     linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    plt.colorbar(im, ax=ax)
    ax.scatter(XX.ravel(), YY.ravel(), c="red", s=12, marker="o",
                edgecolors="white", linewidths=0.5)
    ax.scatter([m2d["eta1"]], [m2d["eta2"]], c="yellow", s=120, marker="*",
                edgecolors="black", linewidths=0.7)
    ax.axhline(0, color="white", lw=0.5, alpha=0.5)
    ax.set_xlabel("eta1 (along d1)")
    ax.set_ylabel("eta2 (along d2)")
    ax.set_title("Chebyshev fit prediction")

    # Error
    ax = axes[2]
    err = Z_pred - Z_dense
    vmax_err = float(np.max(np.abs(err)))
    im = ax.imshow(err, origin="lower", extent=extent, aspect="auto",
                    vmin=-vmax_err, vmax=vmax_err, cmap="RdBu")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("eta1")
    ax.set_ylabel("eta2")
    ax.set_title(f"Pred - Truth  (max |err|={vmax_err:.4f})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(json_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open() as f:
        d = json.load(f)
    for r in d["results"]:
        out_path = out_dir / f"static_2d_step{d['config']['target_step']}_{r['d2_kind']}.png"
        plot_one(d, r, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="grad_interpolation/results/static_2d_step1000.json")
    parser.add_argument("--out-dir", default="grad_interpolation/results")
    args = parser.parse_args()
    main(Path(args.json), Path(args.out_dir))
