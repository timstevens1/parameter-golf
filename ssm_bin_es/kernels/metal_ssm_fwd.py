"""
Metal forward-only kernel for Mamba selective scan.

Each thread handles one (batch, inner_dim) lane, keeping h[N] in registers.
Sequential over timesteps, parallel over batch * inner_dim.

No gradient flow — use for inference/eval only. For training with gradients,
use selective_scan_with_grad from metal_ssm_bwd.py.
"""
from __future__ import annotations

import mlx.core as mx

_SELECTIVE_SCAN_FWD_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint total_lanes = BATCH_SIZE * D;
    if (tid >= total_lanes) return;

    uint b = tid / D;
    uint d = tid % D;

    T h[N];
    for (int n = 0; n < N; n++) {
        h[n] = T(0);
    }

    T A_vals[N];
    for (int n = 0; n < N; n++) {
        A_vals[n] = -metal::exp(A_log[d * N + n]);
    }

    for (int t = 0; t < L; t++) {
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
"""


def run_selective_scan_fwd(
    x: mx.array, dt: mx.array, B: mx.array, C: mx.array, A_log: mx.array,
) -> mx.array:
    """
    Metal-accelerated selective scan (forward pass only).

    Args:
        x:     (batch, seq_len, inner_dim)
        dt:    (batch, seq_len, inner_dim)
        B:     (batch, seq_len, state_dim)
        C:     (batch, seq_len, state_dim)
        A_log: (inner_dim, state_dim)

    Returns:
        y:     (batch, seq_len, inner_dim)
    """
    batch, seq_len, inner_dim = x.shape
    state_dim = A_log.shape[1]

    x_f = x.astype(mx.float32)
    dt_f = dt.astype(mx.float32)
    B_f = B.astype(mx.float32)
    C_f = C.astype(mx.float32)
    A_log_f = A_log.astype(mx.float32)

    kernel = mx.fast.metal_kernel(
        name="selective_scan_fwd",
        input_names=["x", "dt", "B_in", "C_in", "A_log"],
        output_names=["y"],
        source=_SELECTIVE_SCAN_FWD_SOURCE,
        header=(
            f"#define BATCH_SIZE {batch}\n"
            f"#define L {seq_len}\n"
            f"#define D {inner_dim}\n"
            f"#define N {state_dim}\n"
        ),
    )

    total_threads = batch * inner_dim
    threadgroup_size = min(256, total_threads)
    grid_x = ((total_threads + threadgroup_size - 1) // threadgroup_size) * threadgroup_size

    outputs = kernel(
        inputs=[x_f, dt_f, B_f, C_f, A_log_f],
        template=[("T", mx.float32)],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[(batch, seq_len, inner_dim)],
        output_dtypes=[mx.float32],
    )

    return outputs[0]
