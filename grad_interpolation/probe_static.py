"""
Experiment 1A (fit-then-query): how accurately does a Chebyshev fit on a
descent ray predict loss at points NOT used to fit it?

Per (stage, direction) we run one experimental cycle:

  1. Compute unit-normalized direction d on a fixed batch (Adam-update,
     raw-gradient, or random).
  2. Choose eta_max via a doubling search.
  3. Evaluate (loss, dphi/deta) at N Chebyshev-Lobatto nodes on [0, eta_max].
     Fit Hermite-Chebyshev (degree 2N-1) and plain-Chebyshev (degree N-1).
  4. Sample K random query points uniformly on
        [-QUERY_BACK_FRAC * eta_max, (1 + QUERY_FWD_FRAC) * eta_max]
     so the queries are split into:
        below  : eta < 0           (uphill extrapolation)
        inside : 0 <= eta <= eta_max
        above  : eta > eta_max     (forward extrapolation)
  5. At each query point, evaluate ground-truth L(w - eta*d) on the SAME
     training batch; compare to plain and Hermite predictions.

Outputs JSON at ./grad_interpolation/results/static.json plus a plotted
summary via plot_static.py.
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

import grad_interpolation.chebyshev as cheb
from grad_interpolation.data import (
    DEFAULT_TRAIN_PATTERN, DEFAULT_VAL_PATTERN, TokenLoader,
    load_validation_tokens, make_fixed_batch,
)
from grad_interpolation.model import TinyGPT
from grad_interpolation.perturb import (
    adam_update_direction, adam_update_unnormalized, evaluate_along_ray,
    evaluate_value_and_grad_along_ray, gradient_direction, random_direction,
    tree_axpy, tree_clone, tree_norm, zero_state_like,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STAGE_STEPS = (0, 200, 1000)
DIRECTION_KINDS = ("adam", "grad", "random")
N_FIT_NODES = 8                     # Chebyshev nodes per fit
N_QUERY = 25                        # random query points per (stage, direction)
QUERY_BACK_FRAC = 0.5               # window goes to eta = -0.5 * eta_max
QUERY_FWD_FRAC = 0.5                # window goes to eta = 1.5 * eta_max
N_VIS_DENSE = 80                    # visualization-only dense sweep on the wide window

DOUBLE_LOSS_FACTOR = 1.5            # eta_max chosen so loss(eta_max) >= factor * loss(0)
DOUBLE_INIT_ETA = 1e-3
DOUBLE_MAX_ETA = 50.0

SEQ_LEN = 128
TRAIN_BATCH = 32
PROBE_BATCH = 64
ADAM_LR = 3e-3
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def value_and_grad_pair(model: TinyGPT, x: mx.array, y: mx.array):
    """Return (loss_fn, value_and_grad_fn) closures over a fixed (x, y) batch."""
    loss_fn = lambda: model.loss(x, y)
    vag = nn.value_and_grad(model, lambda: model.loss(x, y))
    return loss_fn, vag


def find_eta_max(model, params, direction, loss_fn, factor=DOUBLE_LOSS_FACTOR,
                 init_eta=DOUBLE_INIT_ETA, max_eta=DOUBLE_MAX_ETA) -> float:
    L0 = float(loss_fn().item())
    eta = init_eta
    while eta <= max_eta:
        try:
            model.update(tree_axpy(-eta, direction, params))
            L = float(loss_fn().item())
        finally:
            model.update(params)
        if not np.isfinite(L) or L >= factor * L0 or L <= L0 / factor:
            return eta
        eta *= 2.0
    return max_eta


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


# ---------------------------------------------------------------------------
# Per-direction probe: fit then query
# ---------------------------------------------------------------------------

def probe_direction(model, base_params, direction, batch, label: str, rng,
                     natural_step_norm: float | None = None) -> dict:
    """`natural_step_norm` is the flat-vec L2 norm of one Adam step at lr=ADAM_LR.
    Used to express eta values in units of equivalent constant-LR Adam steps."""
    x, y = batch
    loss_fn, vag = value_and_grad_pair(model, x, y)

    # 1. Calibrate eta_max with the descent direction
    eta_max = find_eta_max(model, base_params, direction, loss_fn)

    # 2. Fit at N Chebyshev-Lobatto nodes on [0, eta_max]
    nodes = cheb.chebyshev_lobatto_nodes(N_FIT_NODES, 0.0, eta_max)
    L_nodes, dphi_nodes = evaluate_value_and_grad_along_ray(
        model, base_params, direction, nodes.tolist(), vag,
    )
    L_nodes = np.array(L_nodes)
    dphi_nodes = np.array(dphi_nodes)

    coeffs_plain = cheb.fit_chebyshev(nodes, L_nodes, deg=N_FIT_NODES - 1,
                                       domain=(0.0, eta_max))
    coeffs_herm = cheb.fit_hermite_chebyshev(nodes, L_nodes, dphi_nodes,
                                              domain=(0.0, eta_max))

    # 3. Random query points on the wide window
    eta_lo = -QUERY_BACK_FRAC * eta_max
    eta_hi = (1.0 + QUERY_FWD_FRAC) * eta_max
    queries = np.sort(rng.uniform(eta_lo, eta_hi, size=N_QUERY))

    # 4. Ground truth at each query point on the same batch
    L_queries = np.array(evaluate_along_ray(
        model, base_params, direction, queries.tolist(), loss_fn,
    ))

    # 5. Predicted values (extrapolating outside [0, eta_max] when applicable)
    Lhat_plain = cheb.evaluate(coeffs_plain, queries, (0.0, eta_max))
    Lhat_herm = cheb.evaluate(coeffs_herm, queries, (0.0, eta_max))

    region = np.where(queries < 0.0, "below",
              np.where(queries > eta_max, "above", "inside"))

    err_plain = Lhat_plain - L_queries
    err_herm = Lhat_herm - L_queries

    def rmse(mask):
        if not mask.any():
            return None
        return {
            "plain": float(np.sqrt(np.mean(err_plain[mask] ** 2))),
            "hermite": float(np.sqrt(np.mean(err_herm[mask] ** 2))),
            "n": int(mask.sum()),
        }

    rmse_summary = {
        "below": rmse(region == "below"),
        "inside": rmse(region == "inside"),
        "above": rmse(region == "above"),
        "all": rmse(np.ones_like(region, dtype=bool)),
    }

    # Visualization-only dense sweep (forward only, cheap)
    vis_etas = np.linspace(eta_lo, eta_hi, N_VIS_DENSE)
    vis_loss = np.array(evaluate_along_ray(
        model, base_params, direction, vis_etas.tolist(), loss_fn,
    ))
    vis_pred_plain = cheb.evaluate(coeffs_plain, vis_etas, (0.0, eta_max))
    vis_pred_herm = cheb.evaluate(coeffs_herm, vis_etas, (0.0, eta_max))

    decay_plain = cheb.coefficient_decay(coeffs_plain)
    decay_herm = cheb.coefficient_decay(coeffs_herm)
    decay_plain["abs_coeffs"] = decay_plain["abs_coeffs"].tolist()
    decay_herm["abs_coeffs"] = decay_herm["abs_coeffs"].tolist()

    # Loss minima along the ray
    # (a) on the fit interval [0, eta_max] from the dense vis sweep
    # (b) on the wide window [eta_lo, eta_hi]
    in_mask = (vis_etas >= 0.0) & (vis_etas <= eta_max)
    idx_in = int(np.argmin(vis_loss[in_mask]))
    eta_min_in = float(vis_etas[in_mask][idx_in])
    L_min_in = float(vis_loss[in_mask][idx_in])
    idx_wide = int(np.argmin(vis_loss))
    eta_min_wide = float(vis_etas[idx_wide])
    L_min_wide = float(vis_loss[idx_wide])
    # Polynomial-predicted minimum on the fit interval (Hermite)
    eta_min_pred_h, L_min_pred_h = cheb.minimize_on_interval(coeffs_herm,
                                                             (0.0, eta_max))
    eta_min_pred_p, L_min_pred_p = cheb.minimize_on_interval(coeffs_plain,
                                                             (0.0, eta_max))

    minima = {
        "fit_interval": {"eta": eta_min_in, "loss": L_min_in},
        "wide_window": {"eta": eta_min_wide, "loss": L_min_wide},
        "predicted_plain": {"eta": eta_min_pred_p, "loss": L_min_pred_p},
        "predicted_hermite": {"eta": eta_min_pred_h, "loss": L_min_pred_h},
    }
    if natural_step_norm is not None and natural_step_norm > 0:
        # Convert eta (distance along unit-direction d) to "equivalent number
        # of constant-LR Adam steps" by dividing by ||lr * adam_update||.
        for k in minima:
            minima[k]["adam_steps_equiv"] = minima[k]["eta"] / natural_step_norm

    return {
        "label": label,
        "eta_max": float(eta_max),
        "eta_lo": float(eta_lo),
        "eta_hi": float(eta_hi),
        "n_fit_nodes": N_FIT_NODES,
        "fit_nodes": nodes.tolist(),
        "fit_loss": L_nodes.tolist(),
        "fit_dphi": dphi_nodes.tolist(),
        "coeffs_plain": coeffs_plain.tolist(),
        "coeffs_hermite": coeffs_herm.tolist(),
        "queries": queries.tolist(),
        "query_region": region.tolist(),
        "loss_truth": L_queries.tolist(),
        "loss_plain": Lhat_plain.tolist(),
        "loss_hermite": Lhat_herm.tolist(),
        "err_plain": err_plain.tolist(),
        "err_hermite": err_herm.tolist(),
        "rmse": rmse_summary,
        "vis_etas": vis_etas.tolist(),
        "vis_loss": vis_loss.tolist(),
        "vis_pred_plain": vis_pred_plain.tolist(),
        "vis_pred_hermite": vis_pred_herm.tolist(),
        "decay_plain": decay_plain,
        "decay_hermite": decay_herm,
        "minima": minima,
        "natural_step_norm": natural_step_norm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir: Path, train_pattern: str, val_pattern: str, seed: int):
    mx.random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] building model")
    model = TinyGPT()
    params = tree_clone(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(params))
    print(f"[setup] params: {n_params:,}")

    val_tokens = load_validation_tokens(val_pattern)
    probe_batch = make_fixed_batch(val_tokens, batch_size=PROBE_BATCH,
                                   seq_len=SEQ_LEN, seed=seed)
    print(f"[setup] probe batch: {probe_batch[0].shape}")

    loader = TokenLoader(train_pattern, seq_len=SEQ_LEN, batch_size=TRAIN_BATCH, seed=seed)

    m_state = zero_state_like(params)
    v_state = zero_state_like(params)
    step = 0
    target_steps = sorted(STAGE_STEPS)
    stages: list[dict] = []

    vag_fn_train = nn.value_and_grad(model, lambda x, y: model.loss(x, y))

    for target in target_steps:
        while step < target:
            xb, yb = loader.next_batch()
            loss, grads = vag_fn_train(xb, yb)
            mx.eval(loss, grads)
            step += 1
            params, m_state, v_state = adam_step_inplace(
                model, params, m_state, v_state, grads, lr=ADAM_LR, step=step,
            )
            if step % 100 == 0 or step == 1:
                print(f"[train] step={step}  loss={float(loss):.4f}")

        print(f"\n[probe] === stage step={step} ===")
        x, y = probe_batch
        loss_at_w, grad_at_w = nn.value_and_grad(model, lambda: model.loss(x, y))()
        mx.eval(loss_at_w, grad_at_w)
        print(f"[probe] loss at w (probe batch): {float(loss_at_w):.4f}")

        # Natural Adam-step magnitude in flat-vec L2: ||lr * adam_update||
        adam_unnorm = adam_update_unnormalized(grad_at_w, m_state, v_state,
                                                beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                                                eps=ADAM_EPS, step=max(step, 1))
        natural_step_norm = ADAM_LR * float(tree_norm(adam_unnorm).item())
        print(f"[probe] natural Adam step norm (lr * ||u||): {natural_step_norm:.4g}")

        stage_record = {"step": step, "loss_at_w": float(loss_at_w),
                        "natural_adam_step_norm": natural_step_norm,
                        "directions": {}}

        for kind in DIRECTION_KINDS:
            t0 = time.perf_counter()
            if kind == "adam":
                d = adam_update_direction(grad_at_w, m_state, v_state,
                                           beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                                           eps=ADAM_EPS, step=max(step, 1))
            elif kind == "grad":
                d = gradient_direction(grad_at_w)
            elif kind == "random":
                d = random_direction(params)
            else:
                raise ValueError(kind)

            print(f"[probe] direction={kind}  ||d||={float(tree_norm(d)):.4f}")
            res = probe_direction(model, params, d, probe_batch,
                                  label=f"step{step}_{kind}", rng=rng,
                                  natural_step_norm=natural_step_norm)
            res["wallclock_s"] = time.perf_counter() - t0
            stage_record["directions"][kind] = res

            r = res["rmse"]["all"]
            mn = res["minima"]
            in_eq = mn["fit_interval"].get("adam_steps_equiv")
            in_eq_str = f" ({in_eq:.1f} adam-steps)" if in_eq is not None else ""
            print(f"[probe]   eta_max={res['eta_max']:.4g}  "
                  f"min on fit-int: eta={mn['fit_interval']['eta']:.3g}, "
                  f"L={mn['fit_interval']['loss']:.4f}{in_eq_str}")
            print(f"[probe]   plain RMSE (all)={r['plain']:.4f}  "
                  f"hermite RMSE (all)={r['hermite']:.4f}  "
                  f"({res['wallclock_s']:.1f}s)")
            for region in ("below", "inside", "above"):
                rg = res["rmse"][region]
                if rg is None:
                    continue
                print(f"[probe]     {region:>6} (n={rg['n']}): "
                      f"plain={rg['plain']:.4f}  hermite={rg['hermite']:.4f}")

        stages.append(stage_record)

    out = {
        "config": {
            "stage_steps": list(STAGE_STEPS),
            "n_fit_nodes": N_FIT_NODES,
            "n_query": N_QUERY,
            "query_back_frac": QUERY_BACK_FRAC,
            "query_fwd_frac": QUERY_FWD_FRAC,
            "directions": list(DIRECTION_KINDS),
            "double_loss_factor": DOUBLE_LOSS_FACTOR,
            "seq_len": SEQ_LEN,
            "train_batch": TRAIN_BATCH,
            "probe_batch": PROBE_BATCH,
            "adam_lr": ADAM_LR,
            "n_params": n_params,
            "seed": seed,
        },
        "stages": stages,
    }
    out_path = out_dir / "static.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="grad_interpolation/results")
    parser.add_argument("--train-pattern", default=DEFAULT_TRAIN_PATTERN)
    parser.add_argument("--val-pattern", default=DEFAULT_VAL_PATTERN)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(Path(args.out_dir), args.train_pattern, args.val_pattern, args.seed)
