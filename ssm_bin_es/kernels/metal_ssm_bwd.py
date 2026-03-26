"""
Metal backward kernel for Mamba selective scan with mx.custom_function integration.

This module provides:
1. A Metal GPU kernel for the backward pass (BPTT) of the selective scan
2. A forward+backward wrapper using mx.custom_function for automatic differentiation
3. The forward kernel re-exported for convenience

Integration into train_mamba_mlx.py:
    In MambaBlock.__call__, replace the selective scan call with:

        from metal_ssm_bwd import selective_scan_with_grad
        y = selective_scan_with_grad(x_path, dt, B, C, self.A_log)

    This gives you Metal-accelerated forward AND backward passes with proper
    gradient flow through mx.custom_function's VJP mechanism.

Architecture:
    - Forward: saves h_checkpoints every CHUNK=32 steps (recomputation strategy)
    - Backward: processes chunks in reverse, recomputes forward states from
      checkpoints, then does BPTT within each chunk
    - Grid: one thread per (batch, inner_dim) lane
    - dC and dB use atomic adds (multiple D threads reduce to shared (B,L,N) outputs)
    - dA_log uses per-batch partial sums, reduced in Python after kernel
"""
from __future__ import annotations

import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 32  # checkpoint interval (must divide seq_len evenly for simplicity)

# ---------------------------------------------------------------------------
# Forward Metal kernel source (identical to train_mamba_mlx.py but also writes
# h_checkpoints every CHUNK steps).
# ---------------------------------------------------------------------------
_SELECTIVE_SCAN_FWD_CKPT_SOURCE = """
    // Thread handles one (batch, inner_dim) lane
    uint tid = thread_position_in_grid.x;
    uint total_lanes = BATCH_SIZE * D;
    if (tid >= total_lanes) return;

    uint b = tid / D;
    uint d = tid % D;

    // Initialize hidden state h[N] in registers
    T h[N];
    for (int n = 0; n < N; n++) {
        h[n] = T(0);
    }

    // Preload A values: A = -exp(A_log)
    T A_vals[N];
    for (int n = 0; n < N; n++) {
        A_vals[n] = -metal::exp(A_log[d * N + n]);
    }

    int num_chunks = L / CHUNK;

    // Write initial checkpoint (h=0 at t=0)
    // h_checkpoints layout: (num_chunks+1, B, D, N)
    // checkpoint c corresponds to state BEFORE chunk c starts
    uint ckpt_stride = BATCH_SIZE * D * N;  // stride per checkpoint
    for (int n = 0; n < N; n++) {
        h_checkpoints[0 * ckpt_stride + b * D * N + d * N + n] = T(0);
    }

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int t_start = chunk * CHUNK;
        for (int t = t_start; t < t_start + CHUNK; t++) {
            T x_val = x[b * L * D + t * D + d];
            T dt_val = dt[b * L * D + t * D + d];
            uint bc_base = b * L * N + t * N;

            T y_val = T(0);
            for (int n = 0; n < N; n++) {
                T A_bar = metal::exp(A_vals[n] * dt_val);
                T B_val = B_in[bc_base + n];
                T B_bar = dt_val * B_val;
                h[n] = A_bar * h[n] + B_bar * x_val;
                T C_val = C_in[bc_base + n];
                y_val += h[n] * C_val;
            }
            y[b * L * D + t * D + d] = y_val;
        }

        // Save checkpoint AFTER this chunk (= state before next chunk)
        for (int n = 0; n < N; n++) {
            h_checkpoints[(chunk + 1) * ckpt_stride + b * D * N + d * N + n] = h[n];
        }
    }
"""

# ---------------------------------------------------------------------------
# Backward Metal kernel source
# ---------------------------------------------------------------------------
_SELECTIVE_SCAN_BWD_SOURCE = """
    // Thread handles one (batch, inner_dim) lane — same mapping as forward
    uint tid = thread_position_in_grid.x;
    uint total_lanes = BATCH_SIZE * D;
    if (tid >= total_lanes) return;

    uint b = tid / D;
    uint d = tid % D;

    // Preload A = -exp(A_log) and A_log values
    T A_vals[N];
    T A_log_vals[N];
    for (int n = 0; n < N; n++) {
        A_log_vals[n] = A_log[d * N + n];
        A_vals[n] = -metal::exp(A_log_vals[n]);
    }

    int num_chunks = L / CHUNK;
    uint ckpt_stride = BATCH_SIZE * D * N;

    // Accumulate dA_log for this (b, d) lane
    T dA_log_accum[N];
    for (int n = 0; n < N; n++) {
        dA_log_accum[n] = T(0);
    }

    // Backward hidden state gradient (propagated across chunks)
    T dh[N];
    for (int n = 0; n < N; n++) {
        dh[n] = T(0);
    }

    // Process chunks in reverse order
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int t_start = chunk * CHUNK;

        // ---- Phase 1: Recompute forward states within this chunk ----
        // Load checkpoint at start of this chunk
        T h_local[CHUNK + 1][N_PADDED];
        for (int n = 0; n < N; n++) {
            h_local[0][n] = h_checkpoints[chunk * ckpt_stride + b * D * N + d * N + n];
        }

        // Also store A_bar and B_bar for reuse in backward
        T A_bar_local[CHUNK][N_PADDED];
        T B_bar_local[CHUNK][N_PADDED];

        for (int i = 0; i < CHUNK; i++) {
            int t = t_start + i;
            T x_val = x[b * L * D + t * D + d];
            T dt_val = dt[b * L * D + t * D + d];
            uint bc_base = b * L * N + t * N;

            for (int n = 0; n < N; n++) {
                T a_bar = metal::exp(A_vals[n] * dt_val);
                T B_val = B_in[bc_base + n];
                T b_bar = dt_val * B_val;

                A_bar_local[i][n] = a_bar;
                B_bar_local[i][n] = b_bar;

                h_local[i + 1][n] = a_bar * h_local[i][n] + b_bar * x_val;
            }
        }

        // ---- Phase 2: Backward through this chunk ----
        for (int i = CHUNK - 1; i >= 0; i--) {
            int t = t_start + i;
            T x_val = x[b * L * D + t * D + d];
            T dt_val = dt[b * L * D + t * D + d];
            uint bc_base = b * L * N + t * N;

            // dy[b, t, d]
            T dy_val = dy[b * L * D + t * D + d];

            // From y[t] = sum_n(h[t][n] * C[t][n]):
            // dh[t][n] += dy[t] * C[t][n]
            // dC[t][n] += dy[t] * h[t][n]  (needs atomic add across D)
            T dx_val = T(0);
            T ddt_val = T(0);

            for (int n = 0; n < N; n++) {
                T C_val = C_in[bc_base + n];
                T h_t_n = h_local[i + 1][n];

                // Gradient from output
                dh[n] += dy_val * C_val;

                // dC: atomic add because multiple D threads write to same (b,t,n)
                // dC[b*L*N + t*N + n] += dy_val * h_t_n
                atomic_fetch_add_explicit(
                    &dC_ssm[bc_base + n],
                    dy_val * h_t_n,
                    memory_order_relaxed
                );
            }

            // From h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]:
            for (int n = 0; n < N; n++) {
                T a_bar = A_bar_local[i][n];
                T b_bar = B_bar_local[i][n];
                T h_prev_n = h_local[i][n];
                T B_val = B_in[bc_base + n];

                // dA_bar[t][n] = dh[t][n] * h[t-1][n]
                T dA_bar_n = dh[n] * h_prev_n;

                // dB_bar[t][n] = dh[t][n] * x[t]
                T dB_bar_n = dh[n] * x_val;

                // dx[t] += dh[t][n] * B_bar[t][n]
                dx_val += dh[n] * b_bar;

                // Chain rule through A_bar = exp(A * dt):
                // ddt += dA_bar * A_bar * A  (gradient of exp wrt dt)
                ddt_val += dA_bar_n * a_bar * A_vals[n];

                // ddt += dB_bar * B[t][n]  (gradient through B_bar = dt * B)
                ddt_val += dB_bar_n * B_val;

                // dB[t][n] += dB_bar * dt[t]  (atomic across D)
                atomic_fetch_add_explicit(
                    &dB_ssm[bc_base + n],
                    dB_bar_n * dt_val,
                    memory_order_relaxed
                );

                // dA_log[d][n] += dA_bar * A_bar * dt * A * (-1)
                // Since A = -exp(A_log), dA/dA_log = -exp(A_log) = A
                // So dA_log += dA_bar * A_bar * dt * A
                // = dA_bar * A_bar * dt * A_vals[n]
                dA_log_accum[n] += dA_bar_n * a_bar * dt_val * A_vals[n];

                // Propagate dh backward: dh[t-1] += dh[t] * A_bar[t]
                dh[n] = dh[n] * a_bar;
            }

            // Write dx[b, t, d] — atomic store (only one thread writes per location)
            atomic_store_explicit(&dx[b * L * D + t * D + d], dx_val, memory_order_relaxed);
            // Write ddt[b, t, d]
            atomic_store_explicit(&ddt[b * L * D + t * D + d], ddt_val, memory_order_relaxed);
        }
    }

    // Write dA_log_partial: shape (B, D, N) — each thread writes its own (b, d) slice
    for (int n = 0; n < N; n++) {
        atomic_store_explicit(&dA_log_partial[b * D * N + d * N + n], dA_log_accum[n], memory_order_relaxed);
    }
"""

# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def _build_fwd_kernel(batch: int, seq_len: int, inner_dim: int, state_dim: int):
    """Build the forward Metal kernel with checkpointing."""
    num_chunks = seq_len // CHUNK_SIZE
    assert seq_len % CHUNK_SIZE == 0, (
        f"seq_len ({seq_len}) must be divisible by CHUNK_SIZE ({CHUNK_SIZE})"
    )

    header = (
        f"#define BATCH_SIZE {batch}\n"
        f"#define L {seq_len}\n"
        f"#define D {inner_dim}\n"
        f"#define N {state_dim}\n"
        f"#define CHUNK {CHUNK_SIZE}\n"
    )

    kernel = mx.fast.metal_kernel(
        name="selective_scan_fwd_ckpt",
        input_names=["x", "dt", "B_in", "C_in", "A_log"],
        output_names=["y", "h_checkpoints"],
        source=_SELECTIVE_SCAN_FWD_CKPT_SOURCE,
        header=header,
    )
    return kernel, num_chunks


def _build_bwd_kernel(batch: int, seq_len: int, inner_dim: int, state_dim: int):
    """Build the backward Metal kernel."""
    # N_PADDED must be >= N to allow static array sizing; use next power-of-2 or N itself
    n_padded = state_dim if state_dim >= 16 else 16

    header = (
        f"#define BATCH_SIZE {batch}\n"
        f"#define L {seq_len}\n"
        f"#define D {inner_dim}\n"
        f"#define N {state_dim}\n"
        f"#define N_PADDED {n_padded}\n"
        f"#define CHUNK {CHUNK_SIZE}\n"
    )

    kernel = mx.fast.metal_kernel(
        name="selective_scan_bwd",
        input_names=["dy", "x", "dt", "B_in", "C_in", "A_log", "h_checkpoints"],
        output_names=["dx", "ddt", "dB_ssm", "dC_ssm", "dA_log_partial"],
        source=_SELECTIVE_SCAN_BWD_SOURCE,
        header=header,
        atomic_outputs=True,
    )
    return kernel


# ---------------------------------------------------------------------------
# Python-level kernel runners
# ---------------------------------------------------------------------------

def _run_fwd_with_checkpoints(
    x: mx.array, dt: mx.array, B_ssm: mx.array, C_ssm: mx.array, A_log: mx.array,
) -> tuple[mx.array, mx.array]:
    """
    Run forward selective scan with checkpoint saving.

    Returns:
        y: (B, L, D) output
        h_checkpoints: (num_chunks+1, B, D, N) hidden state checkpoints
    """
    batch, seq_len, inner_dim = x.shape
    state_dim = A_log.shape[1]

    x_f = x.astype(mx.float32)
    dt_f = dt.astype(mx.float32)
    B_f = B_ssm.astype(mx.float32)
    C_f = C_ssm.astype(mx.float32)
    A_log_f = A_log.astype(mx.float32)

    kernel, num_chunks = _build_fwd_kernel(batch, seq_len, inner_dim, state_dim)

    total_threads = batch * inner_dim
    tg_size = min(256, total_threads)
    grid_x = ((total_threads + tg_size - 1) // tg_size) * tg_size

    outputs = kernel(
        inputs=[x_f, dt_f, B_f, C_f, A_log_f],
        template=[("T", mx.float32)],
        grid=(grid_x, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[
            (batch, seq_len, inner_dim),                    # y
            (num_chunks + 1, batch, inner_dim, state_dim),  # h_checkpoints
        ],
        output_dtypes=[mx.float32, mx.float32],
    )

    return outputs[0], outputs[1]


def _run_bwd(
    dy: mx.array,
    x: mx.array, dt: mx.array, B_ssm: mx.array, C_ssm: mx.array,
    A_log: mx.array, h_checkpoints: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """
    Run backward selective scan kernel.

    Returns:
        dx: (B, L, D)
        ddt: (B, L, D)
        dB_ssm: (B, L, N)
        dC_ssm: (B, L, N)
        dA_log: (D, N)  — already reduced across batch
    """
    batch, seq_len, inner_dim = x.shape
    state_dim = A_log.shape[1]

    dy_f = dy.astype(mx.float32)
    x_f = x.astype(mx.float32)
    dt_f = dt.astype(mx.float32)
    B_f = B_ssm.astype(mx.float32)
    C_f = C_ssm.astype(mx.float32)
    A_log_f = A_log.astype(mx.float32)
    h_ckpt_f = h_checkpoints.astype(mx.float32)

    kernel = _build_bwd_kernel(batch, seq_len, inner_dim, state_dim)

    total_threads = batch * inner_dim
    tg_size = min(256, total_threads)
    grid_x = ((total_threads + tg_size - 1) // tg_size) * tg_size

    outputs = kernel(
        inputs=[dy_f, x_f, dt_f, B_f, C_f, A_log_f, h_ckpt_f],
        template=[("T", mx.float32)],
        grid=(grid_x, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[
            (batch, seq_len, inner_dim),        # dx
            (batch, seq_len, inner_dim),        # ddt
            (batch, seq_len, state_dim),        # dB_ssm (atomic)
            (batch, seq_len, state_dim),        # dC_ssm (atomic)
            (batch, inner_dim, state_dim),      # dA_log_partial (per-batch)
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
        init_value=0.0,
    )

    dx, ddt, dB_ssm, dC_ssm, dA_log_partial = outputs

    # Reduce dA_log across batch dimension: (B, D, N) -> (D, N)
    dA_log = dA_log_partial.sum(axis=0)

    return dx, ddt, dB_ssm, dC_ssm, dA_log


# ---------------------------------------------------------------------------
# mx.custom_function wrapper
# ---------------------------------------------------------------------------

@mx.custom_function
def selective_scan_with_grad(
    x: mx.array,
    dt: mx.array,
    B_ssm: mx.array,
    C_ssm: mx.array,
    A_log: mx.array,
) -> mx.array:
    """
    Metal-accelerated selective scan with automatic differentiation support.

    Forward pass uses a Metal kernel that also saves hidden state checkpoints.
    Backward pass uses a separate Metal kernel that recomputes states from
    checkpoints (memory-efficient BPTT).

    Args:
        x:     (B, L, D) — input after conv+SiLU
        dt:    (B, L, D) — discretization timestep (after softplus)
        B_ssm: (B, L, N) — input-dependent B
        C_ssm: (B, L, N) — input-dependent C
        A_log: (D, N)    — log-space A parameters (learned)

    Returns:
        y:     (B, L, D) — scan output
    """
    y, _h_checkpoints = _run_fwd_with_checkpoints(x, dt, B_ssm, C_ssm, A_log)
    return y


@selective_scan_with_grad.vjp
def selective_scan_vjp(primals, cotangents, outputs):
    """
    VJP (backward pass) for the selective scan.

    Uses the backward Metal kernel with checkpoint-based state recomputation.
    """
    x, dt, B_ssm, C_ssm, A_log = primals
    dy = cotangents if not isinstance(cotangents, (list, tuple)) else cotangents[0]

    # Re-run forward with checkpoints to get h_checkpoints
    # (mx.custom_function does not provide a way to stash intermediates,
    # so we recompute the forward pass here to get the checkpoints)
    _y, h_checkpoints = _run_fwd_with_checkpoints(x, dt, B_ssm, C_ssm, A_log)

    dx, ddt, dB_ssm, dC_ssm, dA_log = _run_bwd(
        dy, x, dt, B_ssm, C_ssm, A_log, h_checkpoints
    )

    # Cast gradients back to the input dtypes
    dx = dx.astype(x.dtype)
    ddt = ddt.astype(dt.dtype)
    dB_ssm = dB_ssm.astype(B_ssm.dtype)
    dC_ssm = dC_ssm.astype(C_ssm.dtype)
    dA_log = dA_log.astype(A_log.dtype)

    return (dx, ddt, dB_ssm, dC_ssm, dA_log)
