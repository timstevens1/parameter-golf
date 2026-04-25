"""
Wide-interval Chebyshev fit experiment.

At a target training stage along the Adam direction, fit Hermite + plain
Chebyshev polynomials at several N values on a wide fit interval
[0, eta_far], where eta_far is many natural-Adam-step lengths.

Then evaluate accuracy *inside* the fit interval at random query points and
on a dense ground-truth sweep. The question: how many nodes are needed to
capture the loss curve over hundreds-of-Adam-steps distances along the ray?

Run:
  python -m grad_interpolation.probe_wide --target-step 1000 --eta-far-mult 100
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
    evaluate_value_and_grad_along_ray, tree_axpy, tree_clone, tree_norm,
    zero_state_like,
)


SEQ_LEN = 128
TRAIN_BATCH = 32
PROBE_BATCH = 64
ADAM_LR = 3e-3
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8

DEFAULT_N_VALUES = (8, 16, 32, 64)
N_DENSE = 200
N_QUERY = 50


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


def run_one_fit(model, base_params, direction, batch, vag_fn, loss_fn,
                eta_far: float, n_nodes: int) -> dict:
    """Fit at N Lobatto nodes on [0, eta_far]. Return coeffs, fits, decay."""
    nodes = cheb.chebyshev_lobatto_nodes(n_nodes, 0.0, eta_far)
    L_nodes, dphi_nodes = evaluate_value_and_grad_along_ray(
        model, base_params, direction, nodes.tolist(), vag_fn,
    )
    L_nodes = np.array(L_nodes)
    dphi_nodes = np.array(dphi_nodes)

    coeffs_plain = cheb.fit_chebyshev(nodes, L_nodes, deg=n_nodes - 1,
                                      domain=(0.0, eta_far))
    coeffs_herm = cheb.fit_hermite_chebyshev(nodes, L_nodes, dphi_nodes,
                                              domain=(0.0, eta_far))
    decay_p = cheb.coefficient_decay(coeffs_plain)
    decay_h = cheb.coefficient_decay(coeffs_herm)
    decay_p["abs_coeffs"] = decay_p["abs_coeffs"].tolist()
    decay_h["abs_coeffs"] = decay_h["abs_coeffs"].tolist()
    return {
        "n_nodes": n_nodes,
        "nodes": nodes.tolist(),
        "L_nodes": L_nodes.tolist(),
        "dphi_nodes": dphi_nodes.tolist(),
        "coeffs_plain": coeffs_plain.tolist(),
        "coeffs_hermite": coeffs_herm.tolist(),
        "decay_plain": decay_p,
        "decay_hermite": decay_h,
    }


def main(target_step: int, eta_far_mult: float, n_values: list[int],
         out_dir: Path, train_pattern: str, val_pattern: str, seed: int):
    mx.random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] building model, training to step={target_step}")
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
    vag_fn_train = nn.value_and_grad(model, lambda x, y: model.loss(x, y))

    step = 0
    while step < target_step:
        xb, yb = loader.next_batch()
        loss, grads = vag_fn_train(xb, yb)
        mx.eval(loss, grads)
        step += 1
        params, m_state, v_state = adam_step_inplace(
            model, params, m_state, v_state, grads, lr=ADAM_LR, step=step,
        )
        if step % 200 == 0 or step == 1:
            print(f"[train] step={step}  loss={float(loss):.4f}")

    print(f"\n[probe] target reached. computing Adam direction.")
    x, y = probe_batch
    L0_arr, grad_w = nn.value_and_grad(model, lambda: model.loss(x, y))()
    mx.eval(L0_arr, grad_w)
    L0 = float(L0_arr.item())

    adam_unnorm = adam_update_unnormalized(grad_w, m_state, v_state,
                                            beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                                            eps=ADAM_EPS, step=max(step, 1))
    natural_step_norm = ADAM_LR * float(tree_norm(adam_unnorm).item())
    direction = adam_update_direction(grad_w, m_state, v_state,
                                       beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                                       eps=ADAM_EPS, step=max(step, 1))

    eta_far = eta_far_mult * natural_step_norm
    print(f"[probe] L0={L0:.4f}  natural_step_norm={natural_step_norm:.4g}  "
          f"eta_far={eta_far:.3g} ({eta_far_mult:.0f} adam-steps)")

    loss_fn = lambda: model.loss(x, y)
    vag = nn.value_and_grad(model, lambda: model.loss(x, y))

    # 1. Dense ground truth across the wide interval
    print(f"[probe] dense ground-truth sweep ({N_DENSE} pts)")
    t0 = time.perf_counter()
    vis_etas = np.linspace(0.0, eta_far, N_DENSE)
    vis_loss = np.array(evaluate_along_ray(model, params, direction,
                                            vis_etas.tolist(), loss_fn))
    print(f"[probe]   done in {time.perf_counter() - t0:.1f}s. "
          f"L range: {vis_loss.min():.4f} .. {vis_loss.max():.4f}")

    # 2. Random query points within the fit interval
    queries = np.sort(rng.uniform(0.0, eta_far, size=N_QUERY))
    L_queries = np.array(evaluate_along_ray(model, params, direction,
                                             queries.tolist(), loss_fn))

    # 3. Run fits at each N
    fits = []
    for n_nodes in n_values:
        print(f"[fit] N={n_nodes}")
        t0 = time.perf_counter()
        f = run_one_fit(model, params, direction, probe_batch, vag, loss_fn,
                        eta_far=eta_far, n_nodes=n_nodes)
        coeffs_p = np.array(f["coeffs_plain"])
        coeffs_h = np.array(f["coeffs_hermite"])

        # Predict on viz sweep + queries
        vis_pred_p = cheb.evaluate(coeffs_p, vis_etas, (0.0, eta_far))
        vis_pred_h = cheb.evaluate(coeffs_h, vis_etas, (0.0, eta_far))
        q_pred_p = cheb.evaluate(coeffs_p, queries, (0.0, eta_far))
        q_pred_h = cheb.evaluate(coeffs_h, queries, (0.0, eta_far))

        err_p = q_pred_p - L_queries
        err_h = q_pred_h - L_queries
        rmse_p = float(np.sqrt(np.mean(err_p ** 2)))
        rmse_h = float(np.sqrt(np.mean(err_h ** 2)))
        max_p = float(np.max(np.abs(err_p)))
        max_h = float(np.max(np.abs(err_h)))

        # Polynomial-predicted minimum on fit interval
        emin_p, lmin_p = cheb.minimize_on_interval(coeffs_p, (0.0, eta_far))
        emin_h, lmin_h = cheb.minimize_on_interval(coeffs_h, (0.0, eta_far))

        f["vis_pred_plain"] = vis_pred_p.tolist()
        f["vis_pred_hermite"] = vis_pred_h.tolist()
        f["q_pred_plain"] = q_pred_p.tolist()
        f["q_pred_hermite"] = q_pred_h.tolist()
        f["err_plain"] = err_p.tolist()
        f["err_hermite"] = err_h.tolist()
        f["rmse"] = {"plain": rmse_p, "hermite": rmse_h,
                      "plain_max": max_p, "hermite_max": max_h}
        f["pred_min_plain"] = {"eta": float(emin_p), "loss": float(lmin_p),
                                "adam_steps": float(emin_p) / natural_step_norm}
        f["pred_min_hermite"] = {"eta": float(emin_h), "loss": float(lmin_h),
                                  "adam_steps": float(emin_h) / natural_step_norm}
        f["wallclock_s"] = time.perf_counter() - t0
        fits.append(f)

        print(f"[fit]   N={n_nodes}  RMSE plain={rmse_p:.4f} (max {max_p:.3f})  "
              f"hermite={rmse_h:.4f} (max {max_h:.3f})  "
              f"({f['wallclock_s']:.1f}s)")

    # 4. Ground-truth minimum on the wide interval (dense argmin)
    idx_min = int(np.argmin(vis_loss))
    gt_min = {
        "eta": float(vis_etas[idx_min]),
        "loss": float(vis_loss[idx_min]),
        "adam_steps": float(vis_etas[idx_min]) / natural_step_norm,
    }
    print(f"[probe] ground-truth min on [0, eta_far]: "
          f"eta={gt_min['eta']:.3g}, L={gt_min['loss']:.4f}, "
          f"{gt_min['adam_steps']:.1f} adam-steps")

    out = {
        "config": {
            "target_step": target_step, "eta_far_mult": eta_far_mult,
            "n_values": list(n_values), "n_dense": N_DENSE, "n_query": N_QUERY,
            "seq_len": SEQ_LEN, "train_batch": TRAIN_BATCH,
            "probe_batch": PROBE_BATCH, "adam_lr": ADAM_LR, "seed": seed,
        },
        "L0": L0, "natural_step_norm": natural_step_norm, "eta_far": eta_far,
        "vis_etas": vis_etas.tolist(), "vis_loss": vis_loss.tolist(),
        "queries": queries.tolist(), "L_queries": L_queries.tolist(),
        "fits": fits, "gt_min": gt_min,
    }
    out_path = out_dir / f"wide_step{target_step}_x{int(eta_far_mult)}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-step", type=int, default=1000)
    parser.add_argument("--eta-far-mult", type=float, default=100.0,
                        help="eta_far in units of natural Adam step norm")
    parser.add_argument("--n-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--out-dir", default="grad_interpolation/results")
    parser.add_argument("--train-pattern", default=DEFAULT_TRAIN_PATTERN)
    parser.add_argument("--val-pattern", default=DEFAULT_VAL_PATTERN)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.target_step, args.eta_far_mult, args.n_values,
         Path(args.out_dir), args.train_pattern, args.val_pattern, args.seed)
