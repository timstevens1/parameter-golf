"""
Chebyshev interpolation primitives for the spectral line-search experiments.

All polynomials are represented as numpy arrays of Chebyshev coefficients in the
*standard* (T_0, T_1, ..., T_n) basis on [-1, 1]. Helpers convert to/from an
arbitrary interval [a, b] via the affine map x = (b-a)/2 * t + (a+b)/2.

Two fitting modes are exposed:

  fit_chebyshev(x, y, deg)        -- least-squares fit to N (x, y) pairs
  fit_hermite_chebyshev(x, y, dy) -- exact Hermite fit (degree 2N-1 from N nodes
                                     by matching values AND first derivatives)

`np.polynomial.chebyshev` is used for evaluation, root-finding, and basis
operations to avoid hand-rolling numerically delicate code.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as cheb


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def chebyshev_lobatto_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """N Chebyshev-Lobatto nodes (extrema of T_{n-1}) on [a, b], including endpoints.

    Good when you want endpoint samples — useful for line search where you
    typically want eta=0 (current loss) included.
    """
    if n < 2:
        raise ValueError("Need at least 2 Lobatto nodes")
    k = np.arange(n)
    t = -np.cos(np.pi * k / (n - 1))  # ascending in [-1, 1]
    return 0.5 * (b - a) * t + 0.5 * (a + b)


def chebyshev_gauss_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """N Chebyshev-Gauss nodes (roots of T_n) on [a, b], strictly interior.

    Optimal in the minimax-error sense for plain interpolation.
    """
    k = np.arange(1, n + 1)
    t = np.cos((2 * k - 1) * np.pi / (2 * n))
    t = t[::-1]  # ascending
    return 0.5 * (b - a) * t + 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_chebyshev(x: np.ndarray, y: np.ndarray, deg: int,
                  domain: tuple[float, float] | None = None) -> np.ndarray:
    """Least-squares Chebyshev fit. Returns coefficients in standard basis on [-1, 1].

    The returned coefficients are in [-1, 1] coordinates. Use evaluate(coeffs, x, domain)
    to evaluate on the original interval.
    """
    if domain is None:
        domain = (float(x.min()), float(x.max()))
    a, b = domain
    t = (2 * x - (a + b)) / (b - a)
    # numpy returns coeffs in the natural basis on [-1, 1] when given t
    coeffs = cheb.chebfit(t, y, deg)
    return coeffs


def fit_hermite_chebyshev(x: np.ndarray, y: np.ndarray, dy: np.ndarray,
                          domain: tuple[float, float] | None = None) -> np.ndarray:
    """Exact Hermite fit: match values AND derivatives at N nodes.

    From N (x, y, dy) triples, solve a 2N x 2N linear system in the Chebyshev
    basis to obtain a polynomial of degree 2N-1. Returns coefficients in
    standard basis on [-1, 1].

    The derivatives dy must be with respect to the *original* x variable; they
    are rescaled internally by the chain rule for the affine map to [-1, 1].
    """
    n = len(x)
    if len(y) != n or len(dy) != n:
        raise ValueError("x, y, dy must all have length N")
    if domain is None:
        domain = (float(x.min()), float(x.max()))
    a, b = domain
    half_width = (b - a) / 2.0
    t = (2 * x - (a + b)) / (b - a)            # nodes in [-1, 1]
    dy_t = dy * half_width                      # dL/dt = dL/dx * dx/dt

    deg = 2 * n - 1                             # polynomial degree
    n_coef = deg + 1                            # number of unknowns

    # Build 2N x 2N linear system. Top half: T_k(t_i) for value match.
    # Bottom half: T_k'(t_i) for derivative match.
    A = np.zeros((2 * n, n_coef))
    rhs = np.zeros(2 * n)
    for i in range(n):
        # Value row
        e_val = np.zeros(n_coef)
        for k in range(n_coef):
            ck = np.zeros(n_coef)
            ck[k] = 1.0
            e_val[k] = cheb.chebval(t[i], ck)
        A[i] = e_val
        rhs[i] = y[i]

        # Derivative row: T_k'(t_i)
        e_der = np.zeros(n_coef)
        for k in range(n_coef):
            ck = np.zeros(n_coef)
            ck[k] = 1.0
            dck = cheb.chebder(ck)
            e_der[k] = cheb.chebval(t[i], dck)
        A[n + i] = e_der
        rhs[n + i] = dy_t[i]

    coeffs, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return coeffs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(coeffs: np.ndarray, x: np.ndarray | float,
             domain: tuple[float, float]) -> np.ndarray:
    """Evaluate a Chebyshev polynomial on the original-x interval."""
    a, b = domain
    t = (2 * np.asarray(x, dtype=float) - (a + b)) / (b - a)
    return cheb.chebval(t, coeffs)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def coefficient_decay(coeffs: np.ndarray) -> dict:
    """Summarize Chebyshev coefficient magnitudes.

    Returns absolute values of coefficients plus a couple of summary metrics
    used by the phase-transition logic in the proposal:
      - tail_ratio: |c_n+1..N-1| / |c_0..n| for n = floor(N/2)
      - decay_rate: slope of log|c_k| vs k from a linear fit (negative = decaying)
    """
    abs_c = np.abs(coeffs)
    n = len(abs_c)
    half = max(1, n // 2)
    head = abs_c[:half].sum() + 1e-30
    tail = abs_c[half:].sum()
    tail_ratio = tail / head

    # log-linear fit on nonzero coefficients only
    mask = abs_c > 0
    if mask.sum() >= 2:
        ks = np.arange(n)[mask]
        ls = np.log(abs_c[mask])
        slope, _intercept = np.polyfit(ks, ls, 1)
    else:
        slope = 0.0

    return {
        "abs_coeffs": abs_c,
        "tail_ratio": float(tail_ratio),
        "decay_rate": float(slope),
    }


# ---------------------------------------------------------------------------
# Minimization on an interval
# ---------------------------------------------------------------------------

def minimize_on_interval(coeffs: np.ndarray, domain: tuple[float, float]) -> tuple[float, float]:
    """Find argmin_{x in [a,b]} p(x) for a Chebyshev polynomial p.

    Roots of p' on [-1, 1] are found via numpy's chebroots (companion matrix);
    candidate minima are these roots plus the endpoints. Returns (x_min, p(x_min))
    in original-x coordinates.
    """
    a, b = domain
    deriv = cheb.chebder(coeffs)
    roots = cheb.chebroots(deriv)
    roots = np.real(roots[np.isreal(roots)])
    candidates_t = roots[(roots >= -1.0) & (roots <= 1.0)]
    candidates_t = np.concatenate([candidates_t, [-1.0, 1.0]])

    candidates_x = 0.5 * (b - a) * candidates_t + 0.5 * (a + b)
    values = cheb.chebval(candidates_t, coeffs)
    best = int(np.argmin(values))
    return float(candidates_x[best]), float(values[best])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Test 1: plain fit recovers a cubic on [0, 4] from 8 nodes
    x_nodes = chebyshev_lobatto_nodes(8, 0.0, 4.0)
    truth = lambda x: 0.3 * x ** 3 - 1.5 * x ** 2 + 0.7 * x + 0.1
    y_nodes = truth(x_nodes)
    coeffs = fit_chebyshev(x_nodes, y_nodes, deg=7, domain=(0.0, 4.0))
    x_test = np.linspace(0, 4, 50)
    err = np.max(np.abs(evaluate(coeffs, x_test, (0.0, 4.0)) - truth(x_test)))
    print(f"plain fit cubic max-err: {err:.2e}")

    # Test 2: Hermite fit with 4 nodes recovers a degree-7 polynomial exactly
    deg7 = lambda x: np.polyval([1.0, -2.0, 0.5, 3.0, -1.0, 0.4, 0.7, 0.2], x - 2.0)
    deg7_p = lambda x: np.polyval(np.polyder([1.0, -2.0, 0.5, 3.0, -1.0, 0.4, 0.7, 0.2]), x - 2.0)
    x_nodes = chebyshev_lobatto_nodes(4, 0.0, 4.0)
    y_nodes = deg7(x_nodes)
    dy_nodes = deg7_p(x_nodes)
    coeffs = fit_hermite_chebyshev(x_nodes, y_nodes, dy_nodes, domain=(0.0, 4.0))
    x_test = np.linspace(0, 4, 50)
    err = np.max(np.abs(evaluate(coeffs, x_test, (0.0, 4.0)) - deg7(x_test)))
    print(f"hermite fit deg-7 max-err: {err:.2e}")

    # Test 3: minimization on a quadratic with known minimum at x=2
    coeffs = fit_chebyshev(x_nodes, (x_nodes - 2.0) ** 2 + 5.0, deg=3, domain=(0.0, 4.0))
    x_min, v_min = minimize_on_interval(coeffs, (0.0, 4.0))
    print(f"min of (x-2)^2 + 5 found at x={x_min:.4f}, v={v_min:.4f}")

    # Test 4: coefficient decay on a smooth function
    f = lambda x: np.exp(-(x - 2.0) ** 2 / 0.5)
    x_nodes = chebyshev_lobatto_nodes(16, 0.0, 4.0)
    y_nodes = f(x_nodes)
    coeffs = fit_chebyshev(x_nodes, y_nodes, deg=15, domain=(0.0, 4.0))
    diag = coefficient_decay(coeffs)
    print(f"gauss bump tail_ratio: {diag['tail_ratio']:.2e}, decay_rate: {diag['decay_rate']:.3f}")
