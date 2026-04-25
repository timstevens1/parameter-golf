"""
Experiment 1B: in-loop spectral line search vs Adam baseline.

Each spectral training step:
  1. Compute (loss, grad) at w on the training batch  -- cost: 1 fwd + 1 bwd
  2. Update Adam (m, v) state and form unit-normalized direction d
  3. Reuse the eta=0 sample as the first Lobatto node
     (loss, dphi) = (loss_at_w, -g . d).
  4. Evaluate (loss, dphi) at the remaining (N-1) Lobatto nodes on
     [0, eta_max]            -- cost: (N-1) fwd + (N-1) bwd
  5. Fit Hermite-Chebyshev (degree 2N-1) and find its minimum on [0, eta_max].
  6. Apply step w := w - eta_min * d.
  7. Adapt eta_max for next step:
        - if eta_min near right edge: grow by GROW
        - else if tail-energy ratio above THRESH (poor fit): shrink
        - else: shrink toward 2 * eta_min (keep the minimum near the middle)

Total per-step cost (spectral, with origin reuse): N fwd + N bwd
Cost (adam baseline): 1 fwd + 1 bwd

Three modes via --mode:
  spectral  -- the method described above (default)
  armijo    -- Armijo-backtracking line search along the Adam direction.
               Implemented but DISABLED unless USE_ARMIJO=1; not used in the
               apples-to-apples comparison yet.
  adam      -- vanilla Adam at fixed lr, the simple baseline.

Outputs ./grad_interpolation/results/inloop_<mode>.jsonl with one record per
log step containing step, loss, val_loss (periodic), wallclock, and NFE
counters {fwd, bwd}.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

import grad_interpolation.chebyshev as cheb
import grad_interpolation.chebyshev_2d as cheb2
from grad_interpolation.data import (
    DEFAULT_TRAIN_PATTERN, DEFAULT_VAL_PATTERN, TokenLoader,
    load_validation_tokens, make_fixed_batch,
)
from grad_interpolation.model import TinyGPT
from grad_interpolation.perturb import (
    adam_state_update, adam_update_direction, adam_update_unnormalized,
    evaluate_along_ray, evaluate_value_and_grad_along_ray,
    evaluate_value_grad_and_trees_along_ray, gradient_direction,
    make_macro_loss_fn, make_macro_vag_fn,
    tree_axpy, tree_clone, tree_dot, tree_norm, zero_state_like,
)
from grad_interpolation.perturb_2d import (
    d2_from_orthogonal_grads, d2_gradient_at_min, evaluate_loss_2d_grid,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SEQ_LEN = 128
TRAIN_BATCH = 32                  # batch for the gradient computation
SEARCH_BATCH = 128                # bigger batch for the line search nodes
                                  # (different sample than gradient batch, to
                                  #  decorrelate noise and prevent the line
                                  #  search from overfitting to the gradient
                                  #  batch's local descent direction)
VAL_BATCH = 64
ADAM_LR = 3e-3
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8

# Spectral defaults (informed by static probe)
SPECTRAL_N_DEFAULT = 5            # static probe showed N=4 already very smooth
SPECTRAL_ETA_K_INIT = 5.0         # initial eta_max in natural-Adam-step units
SPECTRAL_TAIL_THRESH = 0.3        # tail/head energy above which we shrink
SPECTRAL_GROW = 1.5
SPECTRAL_SHRINK_TAIL = 0.6
SPECTRAL_MIN_K = 1.0              # eta_max never smaller than 1 natural step
SPECTRAL_MAX_K = 10.0             # never bigger than 10 natural steps
                                  # (static probe: real min is at 2-5 natural steps;
                                  #  larger caps let mini-batch noise drag the
                                  #  polynomial's minimum out into spurious regions)

# 2D spectral defaults (informed by 2D static probe at step=1000)
SPECTRAL2D_M = 6                  # nodes along d1
SPECTRAL2D_K = 5                  # nodes along d2
SPECTRAL2D_ETA1_K_INIT = 8.0      # initial eta1_max in natural-Adam-step units
SPECTRAL2D_ETA2_K_INIT = 4.0      # initial eta2_max in natural-Adam-step units
SPECTRAL2D_MAX_K1 = 12.0
SPECTRAL2D_MAX_K2 = 8.0
SPECTRAL2D_MIN_K1 = 1.0
SPECTRAL2D_MIN_K2 = 0.5
SPECTRAL2D_SVD_FLOOR = 1e-3       # if top SV < floor, fall back to 1D step
SPECTRAL2D_GROW = 1.4
SPECTRAL2D_SHRINK = 0.7

# Cheap-variant trust region (raw gradient direction has a much steeper
# landscape than Adam-preconditioned direction; static probe found
# eta_max ~8.2 along grad vs ~32.8 along Adam at step=1000).
SPECTRAL2D_CHEAP_ETA1_K_INIT = 2.0
SPECTRAL2D_CHEAP_ETA2_K_INIT = 1.0
SPECTRAL2D_CHEAP_MAX_K1 = 4.0
SPECTRAL2D_CHEAP_MAX_K2 = 2.0
SPECTRAL2D_CHEAP_MIN_K1 = 0.25
SPECTRAL2D_CHEAP_MIN_K2 = 0.1

# Armijo defaults (off by default)
ARMIJO_C = 1e-4
ARMIJO_INIT_ETA = 1.0
ARMIJO_SHRINK = 0.5
ARMIJO_MAX_BACKTRACKS = 20


# ---------------------------------------------------------------------------
# Adam baseline step
# ---------------------------------------------------------------------------

def adam_step(model, params, m, v, grads, lr: float, step: int):
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
# Spectral line search
# ---------------------------------------------------------------------------

def spectral_step(model, params, direction, eta_max: float, n_nodes: int,
                  L_origin: float, dphi_origin: float,
                  vag_fn, loss_fn) -> tuple[float, float, float, dict]:
    """
    Run one spectral line search and apply the chosen step.

    The eta=0 Lobatto node is reused: (L_origin, dphi_origin) = (loss(w), -g.d).
    Only the remaining (n_nodes - 1) nodes need actual evaluations.

    Returns (chosen_eta, next_eta_max, tail_ratio, info_dict).
    info_dict["nfe_fwd"], info_dict["nfe_bwd"] count THIS step's extra evals.
    """
    nodes = cheb.chebyshev_lobatto_nodes(n_nodes, 0.0, eta_max)
    # Reuse origin: nodes[0] = 0 by Lobatto construction
    extra_nodes = nodes[1:]
    L_extra, dphi_extra = evaluate_value_and_grad_along_ray(
        model, params, direction, extra_nodes.tolist(), vag_fn,
    )
    L_arr = np.concatenate([[L_origin], np.array(L_extra)])
    dphi_arr = np.concatenate([[dphi_origin], np.array(dphi_extra)])

    coeffs = cheb.fit_hermite_chebyshev(nodes, L_arr, dphi_arr,
                                         domain=(0.0, eta_max))
    eta_min, _val_min = cheb.minimize_on_interval(coeffs, (0.0, eta_max))
    decay = cheb.coefficient_decay(coeffs)
    tail_ratio = decay["tail_ratio"]

    # Apply step
    new_params = tree_axpy(-eta_min, direction, params)
    model.update(new_params)

    # Adapt eta_max
    pinned_right = eta_min >= 0.95 * eta_max
    poor_fit = tail_ratio > SPECTRAL_TAIL_THRESH
    if pinned_right:
        next_eta_max = eta_max * SPECTRAL_GROW
    elif poor_fit:
        next_eta_max = eta_max * SPECTRAL_SHRINK_TAIL
    else:
        # Center next interval somewhere reasonable: 2 * eta_min, but never
        # less than half of the current
        next_eta_max = max(2.0 * eta_min, 0.5 * eta_max)

    info = {
        "nfe_fwd": n_nodes - 1,
        "nfe_bwd": n_nodes - 1,
        "tail_ratio": float(tail_ratio),
        "L_arr": L_arr.tolist(),
        "dphi_arr": dphi_arr.tolist(),
    }
    return float(eta_min), float(next_eta_max), float(tail_ratio), info


# ---------------------------------------------------------------------------
# 2D spectral line search
# ---------------------------------------------------------------------------

def spectral2d_step(model, params, d1, eta1_max: float, eta2_max: float,
                     m_nodes: int, k_nodes: int,
                     L_origin: float, dphi_origin: float,
                     vag_fn, loss_fn,
                     natural_step: float = 0.0,
                     svd_floor: float = SPECTRAL2D_SVD_FLOOR,
                     grad_origin=None,
                     ) -> tuple[float, float, float, float, dict]:
    """Run one 2D spectral line search and apply the chosen step.

    Process:
      1. Sample (loss, grad_tree) at M Lobatto nodes on d1 ray over [0, eta1_max].
         Reuse origin: nodes[0] = 0 (loss = L_origin, grad_tree from caller).
         (Note: caller must pass `vag_fn` such that the *origin's* gradient
         is the search-batch gradient, otherwise the first node's gradient
         tree is from the wrong batch.)
      2. SVD-extract d2 from orthogonal-grad components.
      3. Sample (M-1) x K supplementary 2D grid (loss only) -- M*K total
         entries, the eta2=0 row already filled from step 1.
      4. Fit 2D Chebyshev tensor.
      5. Find minimum on rectangle.
      6. Apply step; adapt eta_max for next call.

    Returns (eta1*, eta2*, next_eta1_max, next_eta2_max, info).
    """
    nodes_e1 = cheb.chebyshev_lobatto_nodes(m_nodes, 0.0, eta1_max)
    nodes_e2 = cheb.chebyshev_lobatto_nodes(k_nodes, -eta2_max, eta2_max)

    # 1. Sample d1 ray. Need (loss, grad_tree) at each node.
    extra_e1 = nodes_e1[1:]
    L_extra, _dphi_extra, grads_extra = evaluate_value_grad_and_trees_along_ray(
        model, params, d1, extra_e1.tolist(), vag_fn,
    )
    L_d1 = np.concatenate([[L_origin], np.array(L_extra)])

    # Origin gradient tree. If caller supplied one (already computed for
    # other purposes), reuse it; otherwise evaluate at eta=0.
    if grad_origin is None:
        try:
            model.update(params)
            _, g_origin = vag_fn()
            mx.eval(g_origin)
        finally:
            model.update(params)
        origin_extra_fwd = 1
        origin_extra_bwd = 1
    else:
        g_origin = grad_origin
        origin_extra_fwd = 0
        origin_extra_bwd = 0
    grads_d1 = [g_origin] + grads_extra

    # 2. SVD-extract d2
    d2, sing = d2_from_orthogonal_grads(grads_d1, d1)
    sing0 = float(sing[0]) if len(sing) else 0.0
    if sing0 < svd_floor:
        # No useful off-axis signal -- take a 1D Hermite step instead.
        # Build the 1D Hermite poly from the d1 ray's (loss, dphi) data.
        dphi_d1 = []
        try:
            model.update(params)
            _, g0 = vag_fn()
            mx.eval(g0)
            dphi_d1.append(-float(tree_dot(g0, d1).item()))
        finally:
            model.update(params)
        # Compute dphi at remaining nodes via grads we already have
        for g in grads_extra:
            dphi_d1.append(-float(tree_dot(g, d1).item()))
        dphi_d1 = np.array(dphi_d1)
        coeffs = cheb.fit_hermite_chebyshev(nodes_e1, L_d1, dphi_d1,
                                              domain=(0.0, eta1_max))
        e1_min, _v = cheb.minimize_on_interval(coeffs, (0.0, eta1_max))
        new_params = tree_axpy(-float(e1_min), d1, params)
        model.update(new_params)
        info = {
            "fallback_1d": True,
            "svd_top": sing0,
            "nfe_fwd": (m_nodes - 1) + origin_extra_fwd,
            "nfe_bwd": (m_nodes - 1) + origin_extra_bwd,
        }
        return float(e1_min), 0.0, eta1_max, eta2_max, info

    # 3. Build 2D grid Z[i, j] = L(w - eta1_i*d1 - eta2_j*d2)
    # Find the eta2=0 column index (Lobatto with even k won't have exact 0;
    # check ascending nodes_e2 and find closest to 0)
    j_zero_candidates = np.where(np.abs(nodes_e2) < 1e-12)[0]
    have_zero_row = len(j_zero_candidates) > 0
    Z = np.zeros((m_nodes, k_nodes))
    if have_zero_row:
        j0 = int(j_zero_candidates[0])
        Z[:, j0] = L_d1
    # Off-axis: evaluate L at all (eta1_i, eta2_j) where j != j_zero
    extra_2d_count = 0
    try:
        for j, e2 in enumerate(nodes_e2.tolist()):
            if have_zero_row and j == j0:
                continue
            for i, e1 in enumerate(nodes_e1.tolist()):
                shifted = tree_axpy(-float(e1), d1, params)
                shifted = tree_axpy(-float(e2), d2, shifted)
                model.update(shifted)
                L = loss_fn()
                mx.eval(L)
                Z[i, j] = float(L.item())
                extra_2d_count += 1
    finally:
        model.update(params)

    # 4 & 5. Fit + minimize
    C = cheb2.fit_chebyshev_2d(nodes_e1, nodes_e2, Z,
                                 domain_x=(0.0, eta1_max),
                                 domain_y=(-eta2_max, eta2_max))
    # Constrained minimization: keep e1 >= natural_step (one Adam step floor).
    # Without this constraint cross-batch noise collapses e1 to ~0 and the
    # optimizer fails to make d1 progress.
    e1_floor = natural_step if 0.0 < natural_step < eta1_max else 0.0
    e1_min, e2_min, L_pred = cheb2.minimize_on_rect(
        C, (e1_floor, eta1_max), (-eta2_max, eta2_max), n_grid=51,
    )

    # 6. Apply step
    new_params = tree_axpy(-float(e1_min), d1, params)
    new_params = tree_axpy(-float(e2_min), d2, new_params)
    model.update(new_params)

    # Adapt eta1_max
    if e1_min >= 0.95 * eta1_max:
        next_eta1_max = eta1_max * SPECTRAL2D_GROW
    else:
        next_eta1_max = max(2.0 * e1_min, 0.5 * eta1_max)
    # Adapt eta2_max (symmetric: |eta2|)
    abs_e2 = abs(e2_min)
    if abs_e2 >= 0.95 * eta2_max:
        next_eta2_max = eta2_max * SPECTRAL2D_GROW
    else:
        next_eta2_max = max(2.0 * abs_e2, 0.5 * eta2_max)

    info = {
        "fallback_1d": False,
        "svd_top": sing0,
        "svd_top2": float(sing[1]) if len(sing) > 1 else 0.0,
        "nfe_fwd": (m_nodes - 1) + origin_extra_fwd + extra_2d_count,
        "nfe_bwd": (m_nodes - 1) + origin_extra_bwd,    # backward only on d1 ray
        "L_pred_min": float(L_pred),
    }
    return float(e1_min), float(e2_min), next_eta1_max, next_eta2_max, info


# ---------------------------------------------------------------------------
# 2D spectral CHEAP: d1=raw-grad, d2=grad-at-1D-min, plain Chebyshev
# ---------------------------------------------------------------------------

def spectral2d_cheap_step(model, params, eta1_max: float, eta2_max: float,
                           m_nodes: int, k_nodes: int,
                           L_origin: float, grad_w,
                           vag_fn, loss_fn,
                           natural_step: float = 0.0,
                           d2_floor: float = 1e-6,
                           ) -> tuple[float, float, float, float, dict]:
    """One cheap 2D spectral step.

    Process (no Adam state, no Hermite, no SVD):
      1. d1 = unit_normalize(grad_w).
      2. Sample loss at M Lobatto nodes on d1 ray (forward only; reuse origin).
      3. Fit 1D plain Chebyshev along d1, find eta1_1d_min.
      4. Compute one gradient at (w - eta1_1d_min * d1).
      5. d2 = orthogonalize(grad_at_min, d1).
      6. Sample (M-1)*(K-1) supplementary off-axis loss values.
      7. Fit 2D plain Chebyshev tensor product, minimize on rectangle.
      8. Apply step.

    NFE: M (fwd) + 1 (fwd+bwd at min) + (M-1)*(K-1) (fwd) per step.
    """
    # 1. d1 = unit raw gradient (no Adam state)
    d1 = gradient_direction(grad_w)

    # 2. Sample d1 ray (forward only; eta=0 reused from L_origin)
    nodes_e1 = cheb.chebyshev_lobatto_nodes(m_nodes, 0.0, eta1_max)
    extra_e1 = nodes_e1[1:]
    L_extra = evaluate_along_ray(model, params, d1, extra_e1.tolist(), loss_fn)
    L_d1 = np.concatenate([[L_origin], np.array(L_extra)])

    # 3. Fit 1D plain Chebyshev along d1, find min
    coeffs_1d = cheb.fit_chebyshev(nodes_e1, L_d1, deg=m_nodes - 1,
                                    domain=(0.0, eta1_max))
    eta_1d_min, _ = cheb.minimize_on_interval(coeffs_1d, (0.0, eta1_max))

    # 4. Compute gradient at the 1D min
    try:
        shifted = tree_axpy(-float(eta_1d_min), d1, params)
        model.update(shifted)
        _, grad_at_min = vag_fn()
        mx.eval(grad_at_min)
    finally:
        model.update(params)

    # 5. d2 = orthogonalize(grad_at_min, d1). If too small, fall back to 1D.
    from grad_interpolation.perturb_2d import remove_component
    d2_unnorm = remove_component(grad_at_min, d1)
    d2_raw_norm = float(tree_norm(d2_unnorm).item())
    if d2_raw_norm < d2_floor:
        new_params = tree_axpy(-float(eta_1d_min), d1, params)
        model.update(new_params)
        return float(eta_1d_min), 0.0, eta1_max, eta2_max, {
            "fallback_1d": True,
            "d2_norm": d2_raw_norm,
            "eta_1d_min": float(eta_1d_min),
            "nfe_fwd": (m_nodes - 1) + 1,
            "nfe_bwd": 1,
        }
    d2 = d2_gradient_at_min(grad_at_min, d1)

    # 6. Sample 2D off-axis supplementary grid
    nodes_e2 = cheb.chebyshev_lobatto_nodes(k_nodes, -eta2_max, eta2_max)
    j_zero_candidates = np.where(np.abs(nodes_e2) < 1e-12)[0]
    have_zero_row = len(j_zero_candidates) > 0
    Z = np.zeros((m_nodes, k_nodes))
    if have_zero_row:
        j0 = int(j_zero_candidates[0])
        Z[:, j0] = L_d1
    extra_2d_count = 0
    try:
        for j, e2 in enumerate(nodes_e2.tolist()):
            if have_zero_row and j == j0:
                continue
            for i, e1 in enumerate(nodes_e1.tolist()):
                shifted = tree_axpy(-float(e1), d1, params)
                shifted = tree_axpy(-float(e2), d2, shifted)
                model.update(shifted)
                L = loss_fn()
                mx.eval(L)
                Z[i, j] = float(L.item())
                extra_2d_count += 1
    finally:
        model.update(params)

    # 7. Fit + minimize
    C = cheb2.fit_chebyshev_2d(nodes_e1, nodes_e2, Z,
                                domain_x=(0.0, eta1_max),
                                domain_y=(-eta2_max, eta2_max))
    e1_floor = natural_step if 0.0 < natural_step < eta1_max else 0.0
    e1_min, e2_min, L_pred = cheb2.minimize_on_rect(
        C, (e1_floor, eta1_max), (-eta2_max, eta2_max), n_grid=51,
    )

    # 8. Apply step
    new_params = tree_axpy(-float(e1_min), d1, params)
    new_params = tree_axpy(-float(e2_min), d2, new_params)
    model.update(new_params)

    # Adapt eta_max
    if e1_min >= 0.95 * eta1_max:
        next_eta1_max = eta1_max * SPECTRAL2D_GROW
    else:
        next_eta1_max = max(2.0 * e1_min, 0.5 * eta1_max)
    abs_e2 = abs(e2_min)
    if abs_e2 >= 0.95 * eta2_max:
        next_eta2_max = eta2_max * SPECTRAL2D_GROW
    else:
        next_eta2_max = max(2.0 * abs_e2, 0.5 * eta2_max)

    return float(e1_min), float(e2_min), next_eta1_max, next_eta2_max, {
        "fallback_1d": False,
        "d2_norm": d2_raw_norm,
        "eta_1d_min": float(eta_1d_min),
        "L_pred_min": float(L_pred),
        "nfe_fwd": (m_nodes - 1) + 1 + extra_2d_count,
        "nfe_bwd": 1,
    }


# ---------------------------------------------------------------------------
# Armijo backtracking (scaffolding, off by default)
# ---------------------------------------------------------------------------

def armijo_step(model, params, direction, dphi0: float, L0: float, loss_fn,
                init_eta: float = ARMIJO_INIT_ETA) -> tuple[float, int]:
    eta = init_eta
    for k in range(ARMIJO_MAX_BACKTRACKS):
        try:
            model.update(tree_axpy(-eta, direction, params))
            L = float(loss_fn().item())
        finally:
            model.update(params)
        if L <= L0 + ARMIJO_C * eta * dphi0:
            new_params = tree_axpy(-eta, direction, params)
            model.update(new_params)
            return eta, k
        eta *= ARMIJO_SHRINK
    return 0.0, ARMIJO_MAX_BACKTRACKS


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def fixed_val_loss(model, val_batch) -> float:
    x, y = val_batch
    L = model.loss(x, y)
    mx.eval(L)
    return float(L.item())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(mode: str, n_steps: int, log_every: int, val_every: int,
         spectral_n: int, spectral_eta_k: float,
         out_path: Path, train_pattern: str, val_pattern: str, seed: int,
         train_batch: int = TRAIN_BATCH, search_batch: int = SEARCH_BATCH,
         same_batch: bool = False,
         macro_batch_tokens: int = 0, inner_steps: int = 1,
         warmup_adam_steps: int = 0, warmup_batch: int = 32):
    if mode not in {"spectral", "spectral2d", "spectral2d_cheap", "adam", "armijo"}:
        raise ValueError(f"unknown MODE={mode}")
    if mode == "armijo" and not int(os.environ.get("USE_ARMIJO", "0")):
        raise RuntimeError(
            "Armijo path is implemented but disabled. "
            "Pass USE_ARMIJO=1 in the environment to enable explicitly."
        )

    mx.random.seed(seed)
    np.random.seed(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    microbatch_tokens = train_batch * SEQ_LEN
    if macro_batch_tokens > 0:
        num_micro = max(1, macro_batch_tokens // microbatch_tokens)
        macro_active = num_micro > 1
    else:
        num_micro = 1
        macro_active = False
    if inner_steps > 1 and not macro_active:
        raise ValueError("inner_steps > 1 only valid with macro_batch_tokens > 0")

    print(f"[setup] mode={mode}  outer_steps={n_steps}  spectral_N={spectral_n}  "
          f"eta_k_init={spectral_eta_k}  train_batch={train_batch}  "
          f"search_batch={search_batch}  same_batch={same_batch}")
    if macro_active:
        print(f"[setup] macro_batch_tokens={macro_batch_tokens}  "
              f"micro_tokens={microbatch_tokens}  num_micro={num_micro}  "
              f"inner_steps={inner_steps}")
    model = TinyGPT()
    params = tree_clone(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(params))
    print(f"[setup] params: {n_params:,}")

    val_tokens = load_validation_tokens(val_pattern)
    val_batch = make_fixed_batch(val_tokens, batch_size=VAL_BATCH,
                                 seq_len=SEQ_LEN, seed=seed)
    print(f"[setup] val batch: {val_batch[0].shape}")

    loader = TokenLoader(train_pattern, seq_len=SEQ_LEN, batch_size=train_batch, seed=seed)
    if same_batch:
        # Single loader: gradient + line search use the same batch (mirrors
        # static-probe regime; tests whether cross-batch noise was the problem).
        search_loader = loader
    else:
        # Separate, larger loader for the line-search batch.
        search_loader = TokenLoader(train_pattern, seq_len=SEQ_LEN,
                                    batch_size=search_batch, seed=seed + 12345)

    m_state = zero_state_like(params)
    v_state = zero_state_like(params)

    # Cumulative NFE counters
    nfe_fwd = 0
    nfe_bwd = 0

    # Spectral state
    eta_max = None    # set lazily after first natural-step calibration
    eta1_max_2d = None
    eta2_max_2d = None

    t0 = time.perf_counter()
    log_path = out_path
    if log_path.exists():
        log_path.unlink()
    log_f = log_path.open("a")

    # Warmup phase: plain Adam at small batch (decorrelated noise helps escape
    # the random-init plateau before switching to the requested mode).
    if warmup_adam_steps > 0:
        print(f"[warmup] {warmup_adam_steps} Adam steps at batch={warmup_batch}")
        warmup_loader = TokenLoader(train_pattern, seq_len=SEQ_LEN,
                                     batch_size=warmup_batch, seed=seed + 99)
        warmup_vag = nn.value_and_grad(model, lambda x, y: model.loss(x, y))
        for ws in range(1, warmup_adam_steps + 1):
            xb, yb = warmup_loader.next_batch()
            L_arr, grad = warmup_vag(xb, yb)
            mx.eval(L_arr, grad)
            params, m_state, v_state = adam_step(
                model, params, m_state, v_state, grad, lr=ADAM_LR, step=ws,
            )
            nfe_fwd += 1
            nfe_bwd += 1
            if ws % 50 == 0 or ws == warmup_adam_steps:
                vL = fixed_val_loss(model, val_batch)
                nfe_fwd += 1
                wall = time.perf_counter() - t0
                print(f"[warmup] step={ws}/{warmup_adam_steps}  L={float(L_arr):.4f}  "
                      f"val={vL:.4f}  wall={wall:.1f}s  nfe={nfe_fwd}f+{nfe_bwd}b")
                # Log warmup steps too so the JSONL has a continuous record
                log_f.write(json.dumps({
                    "step": ws, "phase": "warmup", "mode": "adam",
                    "loss_w": float(L_arr), "val_loss": vL,
                    "nfe_fwd": nfe_fwd, "nfe_bwd": nfe_bwd,
                    "wallclock_s": wall,
                }) + "\n")
                log_f.flush()
        # Reset Adam step counter for the main phase (so bias-correction
        # restarts cleanly for the new mode if it's not adam).
        # NOTE: keeps m, v values so momentum carries over.
        warmup_step_offset = warmup_adam_steps
    else:
        warmup_step_offset = 0

    total_step = warmup_step_offset
    for outer_step in range(1, n_steps + 1):
        # Sample one macro-batch (list of micro-batches). For non-macro mode,
        # this degenerates to a single (xb, yb).
        microbatches = [loader.next_batch() for _ in range(num_micro)]
        if macro_active:
            loss_fn = make_macro_loss_fn(model, microbatches)
            vag_fn_pair = make_macro_vag_fn(model, microbatches)
        else:
            xb, yb = microbatches[0]
            loss_fn = lambda: model.loss(xb, yb)
            vag_fn_pair = nn.value_and_grad(model, lambda: model.loss(xb, yb))

        for inner in range(inner_steps):
            total_step += 1
            step = total_step  # backwards-compat with downstream code

            # value+grad at w on (macro) batch — 1 macro NFE = num_micro micro NFEs
            L_w_arr, grad_w = vag_fn_pair()
            mx.eval(L_w_arr, grad_w)
            L_w = float(L_w_arr.item())
            nfe_fwd += num_micro
            nfe_bwd += num_micro

            record = {"step": step, "outer_step": outer_step, "inner": inner,
                      "mode": mode, "loss_w": L_w, "num_micro": num_micro}

            if mode == "adam":
                params, m_state, v_state = adam_step(
                    model, params, m_state, v_state, grad_w, lr=ADAM_LR, step=step,
                )

            elif mode == "spectral":
                # Update Adam state and form preconditioned direction
                m_state, v_state = adam_state_update(grad_w, m_state, v_state,
                                                      ADAM_BETA1, ADAM_BETA2)
                d = adam_update_direction(grad_w, m_state, v_state,
                                           beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                                           eps=ADAM_EPS, step=step)
                adam_unnorm = adam_update_unnormalized(grad_w, m_state, v_state,
                                                        beta1=ADAM_BETA1,
                                                        beta2=ADAM_BETA2,
                                                        eps=ADAM_EPS, step=step)
                natural_step = ADAM_LR * float(tree_norm(adam_unnorm).item())

                if eta_max is None:
                    eta_max = spectral_eta_k * natural_step
                eta_max = float(np.clip(eta_max,
                                         SPECTRAL_MIN_K * natural_step,
                                         SPECTRAL_MAX_K * natural_step))

                if macro_active:
                    # Reuse macro batch for line search
                    search_loss_fn = loss_fn
                    search_vag = vag_fn_pair
                    L_origin = L_w
                    dphi_origin = -float(tree_dot(grad_w, d).item())
                    extra_origin_fwd = 0
                    extra_origin_bwd = 0
                else:
                    sxb, syb = search_loader.next_batch()
                    search_loss_fn = lambda: model.loss(sxb, syb)
                    search_vag = nn.value_and_grad(model, lambda: model.loss(sxb, syb))
                    L_origin_arr, grad_origin = search_vag()
                    mx.eval(L_origin_arr, grad_origin)
                    L_origin = float(L_origin_arr.item())
                    dphi_origin = -float(tree_dot(grad_origin, d).item())
                    extra_origin_fwd = 1
                    extra_origin_bwd = 1
                nfe_fwd += extra_origin_fwd * num_micro
                nfe_bwd += extra_origin_bwd * num_micro

                chosen_eta, eta_max, tail, info = spectral_step(
                    model, params, d, eta_max=eta_max, n_nodes=spectral_n,
                    L_origin=L_origin, dphi_origin=dphi_origin,
                    vag_fn=search_vag, loss_fn=search_loss_fn,
                )
                params = tree_clone(model.parameters())
                nfe_fwd += info["nfe_fwd"] * num_micro
                nfe_bwd += info["nfe_bwd"] * num_micro
                record.update({
                    "eta": chosen_eta,
                    "eta_max_next": eta_max,
                    "tail_ratio": tail,
                    "natural_step": natural_step,
                    "adam_steps_equiv": chosen_eta / natural_step if natural_step > 0 else None,
                })

            elif mode == "spectral2d":
                m_state, v_state = adam_state_update(grad_w, m_state, v_state,
                                                      ADAM_BETA1, ADAM_BETA2)
                d1 = adam_update_direction(grad_w, m_state, v_state,
                                            beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                                            eps=ADAM_EPS, step=step)
                adam_unnorm = adam_update_unnormalized(grad_w, m_state, v_state,
                                                        beta1=ADAM_BETA1,
                                                        beta2=ADAM_BETA2,
                                                        eps=ADAM_EPS, step=step)
                natural_step = ADAM_LR * float(tree_norm(adam_unnorm).item())

                if eta1_max_2d is None:
                    eta1_max_2d = SPECTRAL2D_ETA1_K_INIT * natural_step
                    eta2_max_2d = SPECTRAL2D_ETA2_K_INIT * natural_step
                eta1_max_2d = float(np.clip(eta1_max_2d,
                                             SPECTRAL2D_MIN_K1 * natural_step,
                                             SPECTRAL2D_MAX_K1 * natural_step))
                eta2_max_2d = float(np.clip(eta2_max_2d,
                                             SPECTRAL2D_MIN_K2 * natural_step,
                                             SPECTRAL2D_MAX_K2 * natural_step))

                if same_batch or macro_active:
                    search_loss_fn = loss_fn
                    search_vag = vag_fn_pair
                    L_origin = L_w
                    dphi_origin = -float(tree_dot(grad_w, d1).item())
                    extra_origin_fwd = 0
                    extra_origin_bwd = 0
                else:
                    sxb, syb = search_loader.next_batch()
                    search_loss_fn = lambda: model.loss(sxb, syb)
                    search_vag = nn.value_and_grad(model, lambda: model.loss(sxb, syb))
                    L_origin_arr, grad_origin = search_vag()
                    mx.eval(L_origin_arr, grad_origin)
                    L_origin = float(L_origin_arr.item())
                    dphi_origin = -float(tree_dot(grad_origin, d1).item())
                    extra_origin_fwd = 1
                    extra_origin_bwd = 1
                nfe_fwd += extra_origin_fwd * num_micro
                nfe_bwd += extra_origin_bwd * num_micro

                e1_chosen, e2_chosen, eta1_max_2d, eta2_max_2d, info = spectral2d_step(
                    model, params, d1,
                    eta1_max=eta1_max_2d, eta2_max=eta2_max_2d,
                    m_nodes=SPECTRAL2D_M, k_nodes=SPECTRAL2D_K,
                    L_origin=L_origin, dphi_origin=dphi_origin,
                    vag_fn=search_vag, loss_fn=search_loss_fn,
                    natural_step=natural_step,
                    grad_origin=(grad_w if (same_batch or macro_active) else None),
                )
                params = tree_clone(model.parameters())
                nfe_fwd += info["nfe_fwd"] * num_micro
                nfe_bwd += info["nfe_bwd"] * num_micro
                record.update({
                    "eta1": e1_chosen,
                    "eta2": e2_chosen,
                    "eta1_max_next": eta1_max_2d,
                    "eta2_max_next": eta2_max_2d,
                    "natural_step": natural_step,
                    "adam_steps_eta1": e1_chosen / natural_step if natural_step > 0 else None,
                    "adam_steps_eta2": e2_chosen / natural_step if natural_step > 0 else None,
                    "svd_top": info["svd_top"],
                    "fallback_1d": info["fallback_1d"],
                })

            elif mode == "spectral2d_cheap":
                # Compute Adam-state natural step (for unit reporting only;
                # cheap variant does not actually use Adam direction).
                m_state, v_state = adam_state_update(grad_w, m_state, v_state,
                                                      ADAM_BETA1, ADAM_BETA2)
                adam_unnorm = adam_update_unnormalized(grad_w, m_state, v_state,
                                                        beta1=ADAM_BETA1,
                                                        beta2=ADAM_BETA2,
                                                        eps=ADAM_EPS, step=step)
                natural_step = ADAM_LR * float(tree_norm(adam_unnorm).item())

                if eta1_max_2d is None:
                    eta1_max_2d = SPECTRAL2D_CHEAP_ETA1_K_INIT * natural_step
                    eta2_max_2d = SPECTRAL2D_CHEAP_ETA2_K_INIT * natural_step
                eta1_max_2d = float(np.clip(eta1_max_2d,
                                             SPECTRAL2D_CHEAP_MIN_K1 * natural_step,
                                             SPECTRAL2D_CHEAP_MAX_K1 * natural_step))
                eta2_max_2d = float(np.clip(eta2_max_2d,
                                             SPECTRAL2D_CHEAP_MIN_K2 * natural_step,
                                             SPECTRAL2D_CHEAP_MAX_K2 * natural_step))

                # Always reuse origin grad (we use the same batch for d1 grad
                # and search; the cheap variant requires this for L_origin
                # to match the search-batch loss landscape).
                if same_batch or macro_active:
                    search_loss_fn = loss_fn
                    search_vag = vag_fn_pair
                    L_origin = L_w
                else:
                    sxb, syb = search_loader.next_batch()
                    search_loss_fn = lambda: model.loss(sxb, syb)
                    search_vag = nn.value_and_grad(model, lambda: model.loss(sxb, syb))
                    L_origin_arr, grad_w_search = search_vag()
                    mx.eval(L_origin_arr, grad_w_search)
                    L_origin = float(L_origin_arr.item())
                    grad_w = grad_w_search   # use search-batch grad for d1
                    nfe_fwd += num_micro
                    nfe_bwd += num_micro

                e1_chosen, e2_chosen, eta1_max_2d, eta2_max_2d, info = spectral2d_cheap_step(
                    model, params,
                    eta1_max=eta1_max_2d, eta2_max=eta2_max_2d,
                    m_nodes=SPECTRAL2D_M, k_nodes=SPECTRAL2D_K,
                    L_origin=L_origin, grad_w=grad_w,
                    vag_fn=search_vag, loss_fn=search_loss_fn,
                    natural_step=natural_step,
                )
                params = tree_clone(model.parameters())
                nfe_fwd += info["nfe_fwd"] * num_micro
                nfe_bwd += info["nfe_bwd"] * num_micro
                record.update({
                    "eta1": e1_chosen,
                    "eta2": e2_chosen,
                    "eta1_max_next": eta1_max_2d,
                    "eta2_max_next": eta2_max_2d,
                    "natural_step": natural_step,
                    "adam_steps_eta1": e1_chosen / natural_step if natural_step > 0 else None,
                    "adam_steps_eta2": e2_chosen / natural_step if natural_step > 0 else None,
                    "d2_norm": info.get("d2_norm"),
                    "fallback_1d": info["fallback_1d"],
                })

            elif mode == "armijo":
                m_state, v_state = adam_state_update(grad_w, m_state, v_state,
                                                      ADAM_BETA1, ADAM_BETA2)
                d = adam_update_direction(grad_w, m_state, v_state,
                                           beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                                           eps=ADAM_EPS, step=step)
                dphi0 = -float(tree_dot(grad_w, d).item())
                eta, nb = armijo_step(model, params, d, dphi0=dphi0, L0=L_w,
                                       loss_fn=loss_fn)
                nfe_fwd += (nb + 1) * num_micro
                params = tree_clone(model.parameters())
                record.update({"eta": eta, "backtracks": nb})

            # Periodic val eval (val batch is fixed and small; not scaled)
            if val_every > 0 and (step % val_every == 0 or step == n_steps * inner_steps or step == 1):
                val_loss = fixed_val_loss(model, val_batch)
                record["val_loss"] = val_loss
                nfe_fwd += 1

            record["nfe_fwd"] = nfe_fwd
            record["nfe_bwd"] = nfe_bwd
            wallclock = time.perf_counter() - t0
            record["wallclock_s"] = wallclock

            if step % log_every == 0 or step == 1 or step == n_steps * inner_steps:
                extra = ""
                if "eta" in record:
                    extra += f"  eta={record['eta']:.3g}"
                if "adam_steps_equiv" in record and record["adam_steps_equiv"] is not None:
                    extra += f"={record['adam_steps_equiv']:.1f}as"
                if "eta1" in record:
                    e1 = record["eta1"]; e2 = record["eta2"]
                    a1 = record.get("adam_steps_eta1")
                    a2 = record.get("adam_steps_eta2")
                    extra += f"  e1={e1:.3g}({a1:.1f}as)  e2={e2:+.3g}({a2:+.1f}as)"
                    extra += f"  svd={record.get('svd_top', 0):.2g}"
                    if record.get("fallback_1d"):
                        extra += "  [1D fallback]"
                if "tail_ratio" in record:
                    extra += f"  tail={record['tail_ratio']:.2g}"
                if "val_loss" in record:
                    extra += f"  val={record['val_loss']:.4f}"
                tag = f"o{outer_step}.i{inner}" if macro_active else f"s{step}"
                print(f"[{mode}] {tag}  L={L_w:.4f}  wall={wallclock:.1f}s  "
                      f"nfe={nfe_fwd}f+{nfe_bwd}b{extra}")
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()

    log_f.close()
    print(f"[done] {mode}  final nfe={nfe_fwd}f + {nfe_bwd}b  log -> {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=os.environ.get("MODE", "spectral"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--spectral-n", type=int, default=SPECTRAL_N_DEFAULT)
    parser.add_argument("--spectral-eta-k", type=float, default=SPECTRAL_ETA_K_INIT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--train-pattern", default=DEFAULT_TRAIN_PATTERN)
    parser.add_argument("--val-pattern", default=DEFAULT_VAL_PATTERN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch", type=int, default=TRAIN_BATCH)
    parser.add_argument("--search-batch", type=int, default=SEARCH_BATCH)
    parser.add_argument("--same-batch", action="store_true",
                        help="Use train batch for line search (skip search loader)")
    parser.add_argument("--macro-batch-tokens", type=int, default=0,
                        help="Total tokens per macro-batch (0 = disabled). "
                             "Sampled as ceil(macro/microtokens) micro-batches "
                             "with gradient accumulation.")
    parser.add_argument("--inner-steps", type=int, default=1,
                        help="Spectral steps to take per macro-batch (only "
                             "valid with --macro-batch-tokens > 0)")
    parser.add_argument("--warmup-adam-steps", type=int, default=0,
                        help="Steps of plain Adam at small batch to run BEFORE "
                             "switching to the chosen mode. Helps escape the "
                             "random-init plateau.")
    parser.add_argument("--warmup-batch", type=int, default=32,
                        help="Batch size used during warmup Adam phase.")
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(f"grad_interpolation/results/inloop_{args.mode}.jsonl")
    main(args.mode, args.steps, args.log_every, args.val_every,
         args.spectral_n, args.spectral_eta_k,
         out, args.train_pattern, args.val_pattern, args.seed,
         train_batch=args.train_batch, search_batch=args.search_batch,
         same_batch=args.same_batch,
         macro_batch_tokens=args.macro_batch_tokens,
         inner_steps=args.inner_steps,
         warmup_adam_steps=args.warmup_adam_steps,
         warmup_batch=args.warmup_batch)
