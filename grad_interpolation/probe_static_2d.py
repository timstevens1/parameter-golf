"""
Experiment Phase-2 (static): does an off-axis direction unlock structure
that a 1D Adam ray cannot see?

At a checkpoint (default step=1000) on a fixed probe batch:

  1. Compute g(w), the Adam direction d1.
  2. Sample (loss, grad) at M Lobatto nodes on the d1 ray over [0, eta1_max].
  3. Construct three candidate d2 directions:
       a) SVD of orthogonal-to-d1 gradient components
       b) Random gaussian, projected orthogonal to d1
       c) Fresh random gradient batch's gradient minus g(w), orthogonalized
          (Krylov-style; uses one extra batch)
  4. For each d2, sample a 2D MxK Lobatto-Lobatto grid on
     [0, eta1_max] x [-eta2_max, eta2_max]. (The eta1 axis row at eta2 = 0 is
     reused from the d1 ray's loss values to save K-1 evaluations.)
  5. Fit a 2D Chebyshev polynomial; find the minimum on the rectangle.
  6. Report:
       - 1D-only minimum on the d1 ray (a slice through the 2D grid).
       - 2D minimum (eta1*, eta2*, L*).
       - Loss reduction beyond 1D = L_1D_min - L_2D_min.

Output: ./grad_interpolation/results/static_2d.json plus per-d2 contour
plots in ./grad_interpolation/results/static_2d_<kind>.png.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

import grad_interpolation.chebyshev as cheb1
import grad_interpolation.chebyshev_2d as cheb2
from grad_interpolation.data import (
    DEFAULT_TRAIN_PATTERN, DEFAULT_VAL_PATTERN, TokenLoader,
    load_validation_tokens, make_fixed_batch,
)
from grad_interpolation.model import TinyGPT
from grad_interpolation.perturb import (
    adam_update_direction, adam_update_unnormalized, evaluate_along_ray,
    evaluate_value_and_grad_along_ray, gradient_direction, random_direction,
    tree_axpy, tree_clone, tree_dot, tree_norm, zero_state_like,
)
from grad_interpolation.perturb_2d import (
    d2_from_gradient_difference, d2_from_orthogonal_grads,
    d2_gradient_at_min, d2_random_orthogonal,
    evaluate_loss_2d_grid,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEQ_LEN = 128
TRAIN_BATCH = 32
PROBE_BATCH = 128                # bigger probe batch (4x train) per Phase-2 advice
ADAM_LR = 3e-3
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8

DEFAULT_TARGET_STEP = 1000
M_NODES = 6                      # nodes along d1 (eta1 axis)
K_NODES = 5                      # nodes along d2 (eta2 axis)
ETA1_K = 8.0                     # eta1_max in natural-Adam-step units
ETA2_K = 4.0                     # eta2_max in natural-Adam-step units (symmetric: [-, +])
DENSE_GRID = 41                  # for ground-truth contour plot


def adam_step_inplace(model, params, m, v, grads, lr: float, step: int):
    flat_g = dict(tree_flatten(grads))
    flat_p = dict(tree_flatten(params))
    flat_m = dict(tree_flatten(m))
    flat_v = dict(tree_flatten(v))
    bc1 = 1.0 - ADAM_BETA1 ** step
    bc2 = 1.0 - ADAM_BETA2 ** step
    new_p, new_m, new_v = {}, {}, {}
    for k in flat_g:
        new_m[k] = ADAM_BETA1 * flat_m[k] + (1.0 - ADAM_BETA1) * flat_g[k]
        new_v[k] = ADAM_BETA2 * flat_v[k] + (1.0 - ADAM_BETA2) * (flat_g[k] * flat_g[k])
        m_hat = new_m[k] / bc1
        v_hat = new_v[k] / bc2
        new_p[k] = flat_p[k] - lr * m_hat / (mx.sqrt(v_hat) + ADAM_EPS)
    p = tree_unflatten(list(new_p.items()))
    m_out = tree_unflatten(list(new_m.items()))
    v_out = tree_unflatten(list(new_v.items()))
    model.update(p)
    return p, m_out, v_out


def probe_2d(model, params, d1, d2, probe_batch, eta1_max, eta2_max,
             m_nodes: int, k_nodes: int, label: str) -> dict:
    """Sample 2D MxK Lobatto grid; fit; minimize. Return dict of results."""
    x, y = probe_batch
    loss_fn = lambda: model.loss(x, y)

    etas1 = cheb1.chebyshev_lobatto_nodes(m_nodes, 0.0, eta1_max)
    etas2 = cheb1.chebyshev_lobatto_nodes(k_nodes, -eta2_max, eta2_max)

    t0 = time.perf_counter()
    Z = evaluate_loss_2d_grid(model, params, d1, d2,
                               etas1.tolist(), etas2.tolist(), loss_fn)
    grid_time = time.perf_counter() - t0

    C = cheb2.fit_chebyshev_2d(etas1, etas2, Z,
                                domain_x=(0.0, eta1_max),
                                domain_y=(-eta2_max, eta2_max))

    # 2D minimum
    e1_2d, e2_2d, L_2d = cheb2.minimize_on_rect(
        C, (0.0, eta1_max), (-eta2_max, eta2_max), n_grid=51,
    )

    # 1D minimum on the d1 axis (eta2 = 0): use fitted polynomial sliced at e2 = 0
    # Build a 1D polynomial in eta1 by collapsing C with T_j(0) for the eta2 basis.
    # Equivalent: evaluate C along eta2=0 on a fine grid and find min.
    e1_grid = np.linspace(0.0, eta1_max, 401)
    L_1d_grid = cheb2.evaluate_2d(C, e1_grid, np.zeros_like(e1_grid),
                                   (0.0, eta1_max), (-eta2_max, eta2_max))
    idx = int(np.argmin(L_1d_grid))
    e1_1d = float(e1_grid[idx])
    L_1d = float(L_1d_grid[idx])

    # Ground-truth dense contour for the plot
    fine_e1 = np.linspace(0.0, eta1_max, DENSE_GRID)
    fine_e2 = np.linspace(-eta2_max, eta2_max, DENSE_GRID)
    Z_dense = evaluate_loss_2d_grid(model, params, d1, d2,
                                     fine_e1.tolist(), fine_e2.tolist(), loss_fn)
    Z_pred = cheb2.evaluate_grid_2d(C, fine_e1, fine_e2,
                                     (0.0, eta1_max), (-eta2_max, eta2_max))

    return {
        "label": label,
        "eta1_max": float(eta1_max),
        "eta2_max": float(eta2_max),
        "etas1_nodes": etas1.tolist(),
        "etas2_nodes": etas2.tolist(),
        "Z_nodes": Z.tolist(),
        "C_shape": list(C.shape),
        "min_2d": {"eta1": float(e1_2d), "eta2": float(e2_2d), "loss": float(L_2d)},
        "min_1d_at_e2_0": {"eta1": float(e1_1d), "eta2": 0.0, "loss": float(L_1d)},
        "fine_e1": fine_e1.tolist(),
        "fine_e2": fine_e2.tolist(),
        "Z_dense": Z_dense.tolist(),
        "Z_pred": Z_pred.tolist(),
        "grid_eval_seconds": float(grid_time),
    }


def main(target_step: int, m_nodes: int, k_nodes: int, eta1_k: float,
         eta2_k: float, out_dir: Path, train_pattern: str, val_pattern: str,
         seed: int, d1_source: str = "adam"):
    mx.random.seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] training to step={target_step}")
    model = TinyGPT()
    params = tree_clone(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(params))
    print(f"[setup] params: {n_params:,}")

    val_tokens = load_validation_tokens(val_pattern)
    probe_batch = make_fixed_batch(val_tokens, batch_size=PROBE_BATCH,
                                   seq_len=SEQ_LEN, seed=seed)
    loader = TokenLoader(train_pattern, seq_len=SEQ_LEN, batch_size=TRAIN_BATCH, seed=seed)

    m_state = zero_state_like(params)
    v_state = zero_state_like(params)
    vag_train = nn.value_and_grad(model, lambda x, y: model.loss(x, y))

    grad_prev = None  # for Krylov-style d2
    step = 0
    while step < target_step:
        xb, yb = loader.next_batch()
        loss, grads = vag_train(xb, yb)
        mx.eval(loss, grads)
        # Save the previous-batch gradient (just before applying step) for
        # the Krylov d2 candidate later.
        if step + 1 == target_step:
            grad_prev = grads
        step += 1
        params, m_state, v_state = adam_step_inplace(
            model, params, m_state, v_state, grads, lr=ADAM_LR, step=step,
        )
        if step % 200 == 0 or step == 1:
            print(f"[train] step={step}  loss={float(loss):.4f}")

    print(f"\n[probe] computing d1 + reference quantities at step={step}")
    x, y = probe_batch
    L0_arr, grad_w = nn.value_and_grad(model, lambda: model.loss(x, y))()
    mx.eval(L0_arr, grad_w)
    L0 = float(L0_arr.item())

    # Natural Adam step norm is always reported (units for eta_max)
    adam_unnorm = adam_update_unnormalized(grad_w, m_state, v_state,
                                            beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                                            eps=ADAM_EPS, step=max(step, 1))
    natural_step_norm = ADAM_LR * float(tree_norm(adam_unnorm).item())

    # d1 source: "adam" -> Adam-preconditioned direction (with momentum);
    #            "grad" -> raw unit-normalized gradient
    if d1_source == "adam":
        d1 = adam_update_direction(grad_w, m_state, v_state,
                                    beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                                    eps=ADAM_EPS, step=max(step, 1))
    elif d1_source == "grad":
        d1 = gradient_direction(grad_w)
    else:
        raise ValueError(f"unknown d1_source={d1_source}")

    # eta1_max / eta2_max in actual Adam-step units (no auto-scaling).
    eta1_max = eta1_k * natural_step_norm
    eta2_max = eta2_k * natural_step_norm
    print(f"[probe] L0={L0:.4f}  natural_step_norm={natural_step_norm:.4g}  d1={d1_source}")
    print(f"[probe] eta1_max={eta1_max:.3g} ({eta1_k:.1f} adam-steps)  "
          f"eta2_max={eta2_max:.3g} ({eta2_k:.1f} adam-steps)")

    # Sample (loss, grad) along d1 ray for d2 extraction
    print(f"[probe] sampling {m_nodes} (loss, grad) nodes on d1 ray for d2 extraction")
    nodes_d1 = cheb1.chebyshev_lobatto_nodes(m_nodes, 0.0, eta1_max)
    vag = nn.value_and_grad(model, lambda: model.loss(x, y))
    L_d1, dphi_d1 = evaluate_value_and_grad_along_ray(
        model, params, d1, nodes_d1.tolist(), vag,
    )
    # Re-evaluate to also get the gradient TREES (not just dphi). This costs
    # m_nodes more (loss, grad) evaluations -- the prior call only kept dphi.
    grads_at_nodes = []
    for eta in nodes_d1.tolist():
        try:
            shifted = tree_axpy(-float(eta), d1, params)
            model.update(shifted)
            _, g = vag()
            mx.eval(g)
            grads_at_nodes.append(g)
        finally:
            model.update(params)

    # Build the four d2 candidates
    print("[probe] building d2 candidates")
    d2_svd, sing = d2_from_orthogonal_grads(grads_at_nodes, d1)
    print(f"[probe]   svd singular values (top 4): {sing[:4]}")
    d2_rand = d2_random_orthogonal(d1)
    if grad_prev is not None:
        d2_kry = d2_from_gradient_difference(grad_w, grad_prev, d1)
    else:
        d2_kry = None

    # Cheap variant: gradient at the 1D-along-d1 minimum.
    # We find the minimum from the M loss/dphi samples we already have, by
    # fitting a 1D Hermite-Chebyshev and minimizing it.
    L_arr = np.array(L_d1)
    dphi_arr = np.array(dphi_d1)
    coeffs_1d = cheb1.fit_hermite_chebyshev(nodes_d1, L_arr, dphi_arr,
                                             domain=(0.0, eta1_max))
    eta_1d_min, _ = cheb1.minimize_on_interval(coeffs_1d, (0.0, eta1_max))
    print(f"[probe]   1D min along d1 at eta={eta_1d_min:.3g}")
    try:
        shifted = tree_axpy(-float(eta_1d_min), d1, params)
        model.update(shifted)
        _, grad_at_min = vag()
        mx.eval(grad_at_min)
    finally:
        model.update(params)
    d2_atmin = d2_gradient_at_min(grad_at_min, d1)

    # Sanity: verify orthogonality
    for kind, d2 in [("svd", d2_svd), ("atmin", d2_atmin), ("rand", d2_rand),
                       ("krylov", d2_kry)]:
        if d2 is None:
            continue
        ip = float(tree_dot(d1, d2).item())
        nm = float(tree_norm(d2).item())
        print(f"[probe]   d2[{kind}]: <d1, d2>={ip:.2e}  ||d2||={nm:.4f}")

    candidates = [("svd", d2_svd), ("atmin", d2_atmin), ("rand", d2_rand)]
    if d2_kry is not None:
        candidates.append(("krylov", d2_kry))

    results = []
    for kind, d2 in candidates:
        print(f"\n[probe] === d2={kind} ===")
        t0 = time.perf_counter()
        r = probe_2d(model, params, d1, d2, probe_batch, eta1_max, eta2_max,
                     m_nodes=m_nodes, k_nodes=k_nodes, label=f"step{step}_{kind}")
        r["d2_kind"] = kind
        r["wallclock_s"] = time.perf_counter() - t0
        m1d = r["min_1d_at_e2_0"]
        m2d = r["min_2d"]
        delta = m1d["loss"] - m2d["loss"]
        print(f"[probe]   1D min (eta2=0):  eta1={m1d['eta1']:.3g}  L={m1d['loss']:.4f}")
        print(f"[probe]   2D min:           eta1={m2d['eta1']:.3g}  eta2={m2d['eta2']:.3g}  L={m2d['loss']:.4f}")
        print(f"[probe]   2D extra reduction over 1D: {delta:+.4f}")
        print(f"[probe]   ({r['wallclock_s']:.1f}s)")
        results.append(r)

    out = {
        "config": {
            "target_step": target_step, "m_nodes": m_nodes, "k_nodes": k_nodes,
            "eta1_k": eta1_k, "eta2_k": eta2_k,
            "seq_len": SEQ_LEN, "train_batch": TRAIN_BATCH,
            "probe_batch": PROBE_BATCH, "adam_lr": ADAM_LR, "seed": seed,
        },
        "L0": L0,
        "natural_step_norm": natural_step_norm,
        "eta1_max": eta1_max, "eta2_max": eta2_max,
        "svd_singular_values": sing.tolist(),
        "results": results,
    }
    out_path = out_dir / f"static_2d_step{target_step}_d1{d1_source}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-step", type=int, default=DEFAULT_TARGET_STEP)
    parser.add_argument("--m-nodes", type=int, default=M_NODES)
    parser.add_argument("--k-nodes", type=int, default=K_NODES)
    parser.add_argument("--eta1-k", type=float, default=ETA1_K)
    parser.add_argument("--eta2-k", type=float, default=ETA2_K)
    parser.add_argument("--out-dir", default="grad_interpolation/results")
    parser.add_argument("--train-pattern", default=DEFAULT_TRAIN_PATTERN)
    parser.add_argument("--val-pattern", default=DEFAULT_VAL_PATTERN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d1-source", default="adam", choices=["adam", "grad"],
                        help="What direction to use for d1: Adam-preconditioned "
                             "or raw unit-normalized gradient.")
    args = parser.parse_args()
    main(args.target_step, args.m_nodes, args.k_nodes, args.eta1_k,
         args.eta2_k, Path(args.out_dir), args.train_pattern, args.val_pattern,
         args.seed, d1_source=args.d1_source)
