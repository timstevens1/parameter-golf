"""
2D tensor-product Chebyshev primitives for the Phase-2 spectral optimizer.

A 2D polynomial in (x, y) is represented by a coefficient matrix C of shape
(deg_x + 1, deg_y + 1):

    p(x, y) = sum_{i, j} C[i, j] * T_i(t_x) * T_j(t_y)

where t_x = (2x - (a_x + b_x)) / (b_x - a_x) maps the x-domain to [-1, 1] and
likewise for t_y. We use np.polynomial.chebyshev.chebval2d for evaluation and
np.polynomial.chebyshev.chebgrid2d for grid evaluation.

The fit interface takes a tensor-grid of M x K Lobatto-Lobatto nodes and
produces an exact-interpolating coefficient matrix of shape (M, K). For
plain (no Hermite) fitting only.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as cheb

from grad_interpolation.chebyshev import chebyshev_lobatto_nodes


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def lobatto_grid_2d(m: int, k: int,
                    domain_x: tuple[float, float],
                    domain_y: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) -- M Lobatto nodes on x-domain, K on y-domain."""
    xs = chebyshev_lobatto_nodes(m, *domain_x)
    ys = chebyshev_lobatto_nodes(k, *domain_y)
    return xs, ys


# ---------------------------------------------------------------------------
# Fit (interpolation, exact)
# ---------------------------------------------------------------------------

def _vandermonde_1d(t: np.ndarray, n: int) -> np.ndarray:
    """V[i, k] = T_k(t_i), shape (len(t), n)."""
    V = np.zeros((len(t), n))
    for k in range(n):
        ck = np.zeros(n)
        ck[k] = 1.0
        V[:, k] = cheb.chebval(t, ck)
    return V


def fit_chebyshev_2d(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray,
                     domain_x: tuple[float, float],
                     domain_y: tuple[float, float]) -> np.ndarray:
    """
    Exact 2D tensor-product Chebyshev fit.

    Inputs:
        xs: 1D array of M x-nodes (in original coords)
        ys: 1D array of K y-nodes
        Z:  shape (M, K) loss values at the grid (xs[i], ys[j])
    Returns:
        C of shape (M, K) -- coefficients in the standard basis on [-1, 1]^2.

    Solves Z = V_x @ C @ V_y.T  by  C = inv(V_x) @ Z @ inv(V_y).T.
    """
    a_x, b_x = domain_x
    a_y, b_y = domain_y
    t_x = (2 * xs - (a_x + b_x)) / (b_x - a_x)
    t_y = (2 * ys - (a_y + b_y)) / (b_y - a_y)

    M, K = Z.shape
    V_x = _vandermonde_1d(t_x, M)
    V_y = _vandermonde_1d(t_y, K)
    # C = V_x^{-1} Z V_y^{-T}
    tmp = np.linalg.solve(V_x, Z)
    # tmp = V_x^{-1} Z, shape (M, K)
    # Now solve C @ V_y.T = tmp -> V_y @ C.T = tmp.T -> C.T = V_y^{-1} tmp.T
    C_T = np.linalg.solve(V_y, tmp.T)
    C = C_T.T
    return C


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_2d(C: np.ndarray, x: np.ndarray | float, y: np.ndarray | float,
                domain_x: tuple[float, float], domain_y: tuple[float, float]):
    """Evaluate p(x, y) elementwise (scalar or matched arrays)."""
    a_x, b_x = domain_x
    a_y, b_y = domain_y
    t_x = (2 * np.asarray(x, dtype=float) - (a_x + b_x)) / (b_x - a_x)
    t_y = (2 * np.asarray(y, dtype=float) - (a_y + b_y)) / (b_y - a_y)
    return cheb.chebval2d(t_x, t_y, C)


def evaluate_grid_2d(C: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                     domain_x: tuple[float, float], domain_y: tuple[float, float]):
    """Evaluate p on an outer-product grid xs x ys, returns shape (len(xs), len(ys))."""
    a_x, b_x = domain_x
    a_y, b_y = domain_y
    t_x = (2 * xs - (a_x + b_x)) / (b_x - a_x)
    t_y = (2 * ys - (a_y + b_y)) / (b_y - a_y)
    return cheb.chebgrid2d(t_x, t_y, C)


# ---------------------------------------------------------------------------
# Minimize on the rectangle
# ---------------------------------------------------------------------------

def minimize_on_rect(C: np.ndarray,
                     domain_x: tuple[float, float],
                     domain_y: tuple[float, float],
                     n_grid: int = 51) -> tuple[float, float, float]:
    """Find argmin of p(x, y) on [a_x, b_x] x [a_y, b_y].

    Uses a fine grid search to bracket the minimum, then a local Newton
    refinement (closed-form: solve grad = 0 by gradient + Hessian from cheb
    derivatives). Returns (x*, y*, p(x*, y*)).
    """
    a_x, b_x = domain_x
    a_y, b_y = domain_y

    # 1. Coarse grid search to bracket
    xs = np.linspace(a_x, b_x, n_grid)
    ys = np.linspace(a_y, b_y, n_grid)
    G = evaluate_grid_2d(C, xs, ys, domain_x, domain_y)
    i, j = np.unravel_index(int(np.argmin(G)), G.shape)
    x0, y0 = float(xs[i]), float(ys[j])

    # 2. Newton refinement using analytic derivatives in t-space
    Cx = cheb.chebder(C, axis=0)
    Cy = cheb.chebder(C, axis=1)
    Cxx = cheb.chebder(Cx, axis=0)
    Cxy = cheb.chebder(Cx, axis=1)
    Cyy = cheb.chebder(Cy, axis=1)
    half_x = (b_x - a_x) / 2.0
    half_y = (b_y - a_y) / 2.0

    def to_t(x, y):
        return ((2 * x - (a_x + b_x)) / (b_x - a_x),
                (2 * y - (a_y + b_y)) / (b_y - a_y))

    x, y = x0, y0
    for _ in range(30):
        t_x, t_y = to_t(x, y)
        # gradients in original-x: dp/dx = (1/half_x) * dp/dt_x
        gx = cheb.chebval2d(t_x, t_y, Cx) / half_x
        gy = cheb.chebval2d(t_x, t_y, Cy) / half_y
        Hxx = cheb.chebval2d(t_x, t_y, Cxx) / (half_x * half_x)
        Hyy = cheb.chebval2d(t_x, t_y, Cyy) / (half_y * half_y)
        Hxy = cheb.chebval2d(t_x, t_y, Cxy) / (half_x * half_y)
        H = np.array([[Hxx, Hxy], [Hxy, Hyy]])
        g = np.array([gx, gy])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        x_new = x - step[0]
        y_new = y - step[1]
        # If Newton steps out of the rectangle, fall back to projected step
        x_new = float(np.clip(x_new, a_x, b_x))
        y_new = float(np.clip(y_new, a_y, b_y))
        if abs(x_new - x) + abs(y_new - y) < 1e-9:
            x, y = x_new, y_new
            break
        x, y = x_new, y_new

    p_min = float(evaluate_2d(C, x, y, domain_x, domain_y))
    # Compare with grid-search winner; keep whichever is smaller (Newton may
    # have converged to a saddle / local max in non-convex p).
    if G[i, j] < p_min:
        return x0, y0, float(G[i, j])
    return float(x), float(y), p_min


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test: bilinear-quadratic surface (x-1)^2 + 2*(y+0.5)^2 + 0.3*x*y on
    # [0, 3] x [-2, 1]; minimum is somewhere in the interior, easily checkable.
    truth = lambda x, y: (x - 1.0) ** 2 + 2.0 * (y + 0.5) ** 2 + 0.3 * x * y
    a_x, b_x = 0.0, 3.0
    a_y, b_y = -2.0, 1.0
    xs, ys = lobatto_grid_2d(4, 4, (a_x, b_x), (a_y, b_y))
    Z = np.array([[truth(x, y) for y in ys] for x in xs])
    C = fit_chebyshev_2d(xs, ys, Z, (a_x, b_x), (a_y, b_y))

    # Spot-check evaluation
    for xt, yt in [(0.5, 0.0), (1.5, -1.0), (2.7, 0.8)]:
        pred = float(evaluate_2d(C, xt, yt, (a_x, b_x), (a_y, b_y)))
        gt = truth(xt, yt)
        print(f"({xt:.1f}, {yt:.1f}): pred={pred:.6f}  truth={gt:.6f}  err={abs(pred-gt):.2e}")

    # Closed-form minimum: solve gradient = 0
    # dL/dx = 2(x-1) + 0.3y = 0 ; dL/dy = 4(y+0.5) + 0.3x = 0
    # x = 1 - 0.15y; substitute -> 4y + 2 + 0.3 - 0.045y = 0 -> 3.955y = -2.3 -> y = -0.5816
    # x = 1 - 0.15 * (-0.5816) = 1.0872
    x_star, y_star, v_star = minimize_on_rect(C, (a_x, b_x), (a_y, b_y))
    print(f"min: ({x_star:.4f}, {y_star:.4f}) -> {v_star:.6f}")
    print(f"truth min: (1.0872, -0.5816) -> {truth(1.0872, -0.5816):.6f}")
