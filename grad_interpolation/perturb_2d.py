"""
2D direction extraction and 2D ray evaluation, complementing perturb.py.

A direction tree is a leaf-by-leaf gaussian-shaped object aligned with
model.parameters(). For 2D operations we maintain TWO direction trees
(d1, d2) which we keep mutually orthogonal in the flat-vec L2 inner product
and each unit-normalized.

The proposal's d2-extraction (Phase 2) takes the gradients evaluated at the
N Lobatto nodes along d1, removes the d1 component, and computes the top
left singular vector of the resulting flattened matrix. We do this here in
flat-vec coordinates because the SVD of an N x D matrix (with D huge) is
the same as the eigendecomposition of an N x N Gram matrix --- which is
cheap.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from grad_interpolation.perturb import (
    tree_axpy, tree_dot, tree_norm, tree_normalize, tree_scale,
)


# ---------------------------------------------------------------------------
# Tree-level vector ops (extending perturb.py's set)
# ---------------------------------------------------------------------------

def tree_sub(a_tree, b_tree):
    """a - b, leafwise."""
    flat_a = dict(tree_flatten(a_tree))
    flat_b = dict(tree_flatten(b_tree))
    return tree_unflatten([(k, flat_a[k] - flat_b[k]) for k in flat_a])


def project_onto(v_tree, unit_d_tree):
    """Return (v . d) * d, where d is assumed already unit-normalized."""
    coef = float(tree_dot(v_tree, unit_d_tree).item())
    return tree_scale(unit_d_tree, coef)


def remove_component(v_tree, unit_d_tree):
    """v - (v . d) * d. Returns the orthogonal-to-d component."""
    return tree_sub(v_tree, project_onto(v_tree, unit_d_tree))


def orthogonalize_against(v_tree, unit_d_tree):
    """Remove component then re-normalize. Useful when constructing a d2 from
    a candidate that may already have d1-content."""
    return tree_normalize(remove_component(v_tree, unit_d_tree))


# ---------------------------------------------------------------------------
# d2 candidates
# ---------------------------------------------------------------------------

def d2_from_orthogonal_grads(grads_at_nodes: list, unit_d1) -> tuple[object, np.ndarray]:
    """Phase-2-style: SVD of the orthogonal-to-d1 components of the gradients
    sampled along the d1 ray.

    Inputs:
        grads_at_nodes: list of N gradient trees, one per Lobatto node on d1
        unit_d1: unit direction tree

    Returns (d2_unit_tree, singular_values).

    Implementation: we form an N x N Gram matrix of the orthogonal-component
    inner products, eigendecompose that, then reconstruct the top left
    singular vector as a linear combination of the orthogonal-component trees.
    This avoids ever materializing a (D, N) matrix where D is the parameter
    count.
    """
    perp = [remove_component(g, unit_d1) for g in grads_at_nodes]
    N = len(perp)
    # Gram matrix: G[i, j] = <perp[i], perp[j]>
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            G[i, j] = float(tree_dot(perp[i], perp[j]).item())
            G[j, i] = G[i, j]
    # If perp = U S V^T (D x N matrix), then G = N V S^2 V^T (just from columns).
    # Actually G[i, j] = <col_i, col_j> = (V S^2 V^T)[i, j], i.e. G = V S^2 V^T.
    # Top left singular vector u_1 = perp @ v_1 / s_1.
    eigvals, eigvecs = np.linalg.eigh(G)
    # eigh returns ascending; take the top
    idx = int(np.argmax(eigvals))
    s1_sq = max(eigvals[idx], 0.0)
    s1 = float(np.sqrt(s1_sq))
    v1 = eigvecs[:, idx]
    if s1 < 1e-30:
        # Degenerate -- orthogonal components have negligible energy.
        # Fall back to a zero direction (caller can detect via singular values).
        zero = tree_map(lambda v: mx.zeros_like(v), unit_d1)
        return zero, np.array([0.0] * N)
    # Build d2 = sum_i v1[i] * perp[i] / s1
    flat_acc = None
    for i, perp_tree in enumerate(perp):
        scaled = tree_scale(perp_tree, float(v1[i]) / s1)
        if flat_acc is None:
            flat_acc = scaled
        else:
            flat_acc = tree_axpy(1.0, scaled, flat_acc)
    # Numerical hygiene: re-orthogonalize against d1 and renormalize.
    d2 = orthogonalize_against(flat_acc, unit_d1)
    sing = np.sqrt(np.maximum(eigvals, 0.0))[::-1]  # descending
    return d2, sing


def d2_gradient_at_min(grad_at_min, unit_d1) -> object:
    """Cheap d2: take the raw gradient at the 1D-along-d1 minimum, project
    orthogonal to d1, normalize.

    At a 1D minimum on the d1 ray, g . d1 ≈ 0 by definition, so the projection
    barely changes the direction; we still do it for numerical hygiene.

    Cost vs SVD-d2: only 1 backward pass (at the min) instead of M backward
    passes along the ray.
    """
    return orthogonalize_against(grad_at_min, unit_d1)


def d2_random_orthogonal(unit_d1) -> object:
    """Null baseline: random gaussian, projected orthogonal to d1, normalized."""
    flat = dict(tree_flatten(unit_d1))
    rand = {k: mx.random.normal(v.shape, dtype=mx.float32) for k, v in flat.items()}
    rand_tree = tree_unflatten(list(rand.items()))
    return orthogonalize_against(rand_tree, unit_d1)


def d2_from_gradient_difference(grad_w, grad_prev, unit_d1) -> object:
    """Krylov-style: (grad_w - grad_prev), orthogonalized + normalized.

    Captures the curvature direction along which the gradient has been
    changing recently.
    """
    diff = tree_sub(grad_w, grad_prev)
    return orthogonalize_against(diff, unit_d1)


# ---------------------------------------------------------------------------
# 2D ray evaluation
# ---------------------------------------------------------------------------

def evaluate_loss_2d_grid(model: nn.Module, base_params, d1, d2,
                          etas1: Iterable[float], etas2: Iterable[float],
                          loss_fn) -> np.ndarray:
    """Evaluate L(w - eta1*d1 - eta2*d2) at every (eta1, eta2) in the
    rectangular grid. Returns a (len(etas1), len(etas2)) numpy array.

    Restores base_params at the end.
    """
    etas1 = list(etas1)
    etas2 = list(etas2)
    Z = np.zeros((len(etas1), len(etas2)))
    try:
        for i, e1 in enumerate(etas1):
            for j, e2 in enumerate(etas2):
                shifted = tree_axpy(-float(e1), d1, base_params)
                shifted = tree_axpy(-float(e2), d2, shifted)
                model.update(shifted)
                L = loss_fn()
                mx.eval(L)
                Z[i, j] = float(L.item())
    finally:
        model.update(base_params)
    return Z


def apply_2d_step(model, base_params, d1, d2, eta1: float, eta2: float):
    """Set model params to w - eta1*d1 - eta2*d2 and return the new tree."""
    shifted = tree_axpy(-float(eta1), d1, base_params)
    shifted = tree_axpy(-float(eta2), d2, shifted)
    model.update(shifted)
    return shifted


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from grad_interpolation.model import TinyGPT
    from grad_interpolation.perturb import (
        gradient_direction, random_direction, tree_clone,
    )

    mx.random.seed(0)
    model = TinyGPT()
    params = tree_clone(model.parameters())
    x = mx.random.randint(0, 1024, (4, 128))
    y = mx.random.randint(0, 1024, (4, 128))
    loss_fn = lambda: model.loss(x, y)
    vag = nn.value_and_grad(model, lambda: model.loss(x, y))

    L0, g0 = vag()
    mx.eval(L0, g0)
    d1 = gradient_direction(g0)

    # Check orthogonalization
    perp = remove_component(g0, d1)
    print(f"<g, d1> after remove: {float(tree_dot(perp, d1).item()):.2e}  (should be ~0)")

    # Generate fake N=4 gradients by perturbing g0; verify d2 extraction
    grads_n = [g0, tree_axpy(0.5, random_direction(params), g0),
               tree_axpy(-0.3, random_direction(params), g0),
               tree_axpy(1.2, random_direction(params), g0)]
    d2, sing = d2_from_orthogonal_grads(grads_n, d1)
    print(f"||d2|| = {float(tree_norm(d2).item()):.4f}  singulars: {sing}")
    print(f"<d1, d2> = {float(tree_dot(d1, d2).item()):.2e}  (should be ~0)")

    # 2D loss grid smoke test
    Z = evaluate_loss_2d_grid(model, params, d1, d2,
                              etas1=[0.0, 1.0, 2.0],
                              etas2=[-0.5, 0.0, 0.5], loss_fn=loss_fn)
    print(f"loss grid (3x3):\n{Z}")
