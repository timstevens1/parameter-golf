"""
Flat parameter ops, direction computation, and directional derivatives for
the spectral line-search experiments.

A "direction" is a parameter-tree-shaped object with the same structure as
model.parameters(). Inner products are computed as sum over leaves of
elementwise products, giving the natural Euclidean inner product on the
flattened parameter vector.

Directions are unit-normalized (||d|| = 1 in flat-vector L2) so that eta
has consistent units across the three direction kinds (Adam-update,
raw-grad, random gaussian).
"""
from __future__ import annotations

from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten


# ---------------------------------------------------------------------------
# Tree arithmetic
# ---------------------------------------------------------------------------

def tree_axpy(alpha: float, x_tree, y_tree):
    """Return alpha*x + y, leafwise."""
    flat_x = dict(tree_flatten(x_tree))
    flat_y = dict(tree_flatten(y_tree))
    out = {k: alpha * flat_x[k] + flat_y[k] for k in flat_y}
    return tree_unflatten(list(out.items()))


def tree_dot(a_tree, b_tree) -> mx.array:
    """Sum of elementwise products over all leaves. Shape: scalar."""
    flat_a = dict(tree_flatten(a_tree))
    flat_b = dict(tree_flatten(b_tree))
    s = mx.array(0.0, dtype=mx.float32)
    for k in flat_a:
        s = s + mx.sum(flat_a[k].astype(mx.float32) * flat_b[k].astype(mx.float32))
    return s


def tree_norm(x_tree) -> mx.array:
    return mx.sqrt(tree_dot(x_tree, x_tree))


def tree_scale(x_tree, alpha: float):
    return tree_map(lambda v: alpha * v, x_tree)


def tree_normalize(x_tree):
    """Return x_tree / ||x_tree|| in flat-vector L2."""
    n = float(tree_norm(x_tree).item())
    if n < 1e-20:
        raise ValueError(f"Cannot normalize zero direction (norm={n})")
    return tree_scale(x_tree, 1.0 / n)


def tree_clone(x_tree):
    """Deep copy of a parameter tree (each leaf becomes a new mx.array)."""
    return tree_map(lambda v: mx.array(v), x_tree)


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

def gradient_direction(grads_tree):
    """Unit-normalized raw gradient. Returns a direction tree of same shape."""
    return tree_normalize(grads_tree)


def adam_update_unnormalized(grads_tree, m_tree, v_tree, beta1: float = 0.9,
                             beta2: float = 0.999, eps: float = 1e-8,
                             step: int = 1):
    """Compute Adam's would-be update direction (before lr), un-normalized.

    Returns a tree representing m_hat / (sqrt(v_hat) + eps). Multiplying this
    by lr gives the actual step Adam would take. Does NOT mutate m or v.
    """
    flat_g = dict(tree_flatten(grads_tree))
    flat_m = dict(tree_flatten(m_tree))
    flat_v = dict(tree_flatten(v_tree))
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    out = {}
    for k in flat_g:
        m_new = beta1 * flat_m[k] + (1.0 - beta1) * flat_g[k]
        v_new = beta2 * flat_v[k] + (1.0 - beta2) * (flat_g[k] * flat_g[k])
        m_hat = m_new / bc1
        v_hat = v_new / bc2
        out[k] = m_hat / (mx.sqrt(v_hat) + eps)
    return tree_unflatten(list(out.items()))


def adam_update_direction(grads_tree, m_tree, v_tree, beta1: float = 0.9,
                          beta2: float = 0.999, eps: float = 1e-8,
                          step: int = 1):
    """Compute Adam's would-be update direction (before lr) and return unit-normalized.

    Given current first/second moment trees (m, v) and the new gradient, returns
    the direction Adam would step in: m_hat / (sqrt(v_hat) + eps), normalized.
    Does NOT mutate m or v.
    """
    direction = adam_update_unnormalized(grads_tree, m_tree, v_tree,
                                         beta1=beta1, beta2=beta2, eps=eps, step=step)
    return tree_normalize(direction)


def adam_state_update(grads_tree, m_tree, v_tree, beta1: float = 0.9,
                      beta2: float = 0.999):
    """Return updated (m, v) trees given a new gradient. Pure function."""
    flat_g = dict(tree_flatten(grads_tree))
    flat_m = dict(tree_flatten(m_tree))
    flat_v = dict(tree_flatten(v_tree))
    new_m = {k: beta1 * flat_m[k] + (1.0 - beta1) * flat_g[k] for k in flat_g}
    new_v = {k: beta2 * flat_v[k] + (1.0 - beta2) * (flat_g[k] * flat_g[k]) for k in flat_g}
    return tree_unflatten(list(new_m.items())), tree_unflatten(list(new_v.items()))


def zero_state_like(params_tree):
    return tree_map(lambda v: mx.zeros_like(v), params_tree)


def random_direction(params_tree, key: mx.array | None = None):
    """Unit-normalized gaussian direction with the same tree shape as params."""
    flat = dict(tree_flatten(params_tree))
    out = {}
    for k, v in flat.items():
        out[k] = mx.random.normal(v.shape, dtype=mx.float32 if v.dtype == mx.float32 else v.dtype)
    direction = tree_unflatten(list(out.items()))
    return tree_normalize(direction)


# ---------------------------------------------------------------------------
# Line probe
# ---------------------------------------------------------------------------

def directional_derivative(grads_tree, direction_tree) -> float:
    """Compute g(w) . d as a scalar, used for Hermite line-search data.

    Note: this is dL/d(w + alpha*d) at alpha=0; for L(w - eta*d) you want
    -directional_derivative wrt eta.
    """
    return float(tree_dot(grads_tree, direction_tree).item())


def evaluate_along_ray(model: nn.Module, base_params, direction, etas: Iterable[float],
                       loss_fn) -> list[float]:
    """For each eta, set params to (base_params - eta*direction), evaluate loss_fn(),
    then restore base_params at the end.

    `loss_fn` is a zero-arg callable returning an mx.array scalar (closes over
    a fixed batch).
    """
    losses = []
    try:
        for eta in etas:
            model.update(tree_axpy(-float(eta), direction, base_params))
            loss = loss_fn()
            mx.eval(loss)
            losses.append(float(loss.item()))
    finally:
        model.update(base_params)
    return losses


def evaluate_value_and_grad_along_ray(model: nn.Module, base_params, direction,
                                      etas: Iterable[float], value_and_grad_fn):
    """For each eta, set params to (base_params - eta*direction), evaluate
    (loss, grads), record (loss, g . d / step-units).

    Returns (losses, dphi_detas) where phi(eta) = L(w - eta*d) and
    dphi/deta = -g(w - eta*d) . d.

    `value_and_grad_fn` is a zero-arg callable returning (loss, grads_tree).
    """
    losses = []
    dphi = []
    try:
        for eta in etas:
            model.update(tree_axpy(-float(eta), direction, base_params))
            loss, grads = value_and_grad_fn()
            mx.eval(loss, grads)
            losses.append(float(loss.item()))
            g_dot_d = float(tree_dot(grads, direction).item())
            dphi.append(-g_dot_d)
    finally:
        model.update(base_params)
    return losses, dphi


def make_macro_loss_fn(model: nn.Module, microbatches: list):
    """Return a zero-arg loss closure that averages model.loss over a list of
    (xb, yb) micro-batches (gradient accumulation pattern, loss-only).

    Each call evaluates `len(microbatches)` micro-forward passes; that's the
    micro-NFE caller should track.
    """
    n = len(microbatches)
    def fn():
        total = 0.0
        for xb, yb in microbatches:
            L = model.loss(xb, yb)
            mx.eval(L)
            total += float(L.item())
        return mx.array(total / n, dtype=mx.float32)
    return fn


def make_macro_vag_fn(model: nn.Module, microbatches: list):
    """Return a zero-arg (loss, grad_tree) closure that averages over a list
    of (xb, yb) micro-batches (proper gradient accumulation).

    Each call performs `len(microbatches)` micro-forward+backward passes.
    """
    n = len(microbatches)
    base_vag = nn.value_and_grad(model, lambda x, y: model.loss(x, y))
    def fn():
        accum_loss = 0.0
        accum_grad = None
        for xb, yb in microbatches:
            L, g = base_vag(xb, yb)
            mx.eval(L, g)
            accum_loss += float(L.item())
            if accum_grad is None:
                # Initialize accumulator with first micro's grad scaled by 1/n
                flat = dict(tree_flatten(g))
                accum_grad = tree_unflatten([(k, v / n) for k, v in flat.items()])
            else:
                # Accumulate scaled gradient: accum += g/n
                flat_a = dict(tree_flatten(accum_grad))
                flat_g = dict(tree_flatten(g))
                accum_grad = tree_unflatten(
                    [(k, flat_a[k] + flat_g[k] / n) for k in flat_a]
                )
                mx.eval(accum_grad)  # keep graph from growing per-iter
        return mx.array(accum_loss / n, dtype=mx.float32), accum_grad
    return fn


def evaluate_value_grad_and_trees_along_ray(model: nn.Module, base_params, direction,
                                            etas: Iterable[float], value_and_grad_fn):
    """Like evaluate_value_and_grad_along_ray but ALSO returns the gradient TREES.

    Returns (losses, dphis, grad_trees), where grad_trees[i] is the full
    gradient at (base_params - etas[i] * direction). Used by 2D spectral
    extraction (SVD of orthogonal-grad components).
    """
    losses = []
    dphis = []
    grad_trees = []
    try:
        for eta in etas:
            model.update(tree_axpy(-float(eta), direction, base_params))
            loss, grads = value_and_grad_fn()
            mx.eval(loss, grads)
            losses.append(float(loss.item()))
            dphis.append(-float(tree_dot(grads, direction).item()))
            grad_trees.append(grads)
    finally:
        model.update(base_params)
    return losses, dphis, grad_trees


if __name__ == "__main__":
    # Smoke test: random params, random direction, sweep eta
    import math
    from grad_interpolation.model import TinyGPT

    mx.random.seed(0)
    model = TinyGPT()
    params = tree_clone(model.parameters())
    d = random_direction(params)

    print(f"||d|| = {float(tree_norm(d).item()):.6f}  (should be 1)")

    x = mx.random.randint(0, 1024, (2, 128))
    y = mx.random.randint(0, 1024, (2, 128))
    loss_fn = lambda: model.loss(x, y)
    etas = [0.0, 0.5, 1.0, 2.0, 5.0]
    losses = evaluate_along_ray(model, params, d, etas, loss_fn)
    print("losses along random ray:")
    for e, L in zip(etas, losses):
        print(f"  eta={e:.2f}  loss={L:.4f}")
    # Verify params restored
    p_now = dict(tree_flatten(model.parameters()))
    p_orig = dict(tree_flatten(params))
    max_diff = max(float(mx.max(mx.abs(p_now[k] - p_orig[k])).item()) for k in p_now)
    print(f"max param-restoration diff: {max_diff:.2e}")
