"""
Fused Metal kernel for binary matrix multiplication on Apple Silicon via MLX.

Computes y = x @ (unpack(binary_packed) * scale_broadcast).T entirely on-GPU,
avoiding the materialization of the full FP weight matrix.

binary_packed: (out_dim, in_dim_packed) uint8, 8 binary {-1,+1} weights per byte
scale:         (out_dim, num_groups)     float16, one scale per GROUP_SIZE weights
x:             (M, in_dim)              float32/float16
y:             (M, out_dim)             float32

Bit layout within each packed byte (LSB-first):
  bit 0 -> weight index k*8+0, bit 7 -> weight index k*8+7
  bit=0 -> weight = -1, bit=1 -> weight = +1
"""
from __future__ import annotations

import mlx.core as mx


# ---------------------------------------------------------------------------
# Packing / unpacking utilities
# ---------------------------------------------------------------------------

def pack_binary_weights(binary_int8: mx.array) -> mx.array:
    """Pack a {-1, +1} int8 tensor into uint8 with 8 weights per byte (LSB-first).

    Args:
        binary_int8: (out_dim, in_dim) int8 tensor with values in {-1, +1}.
                     in_dim is padded to the next multiple of 8 if needed.

    Returns:
        packed: (out_dim, ceil(in_dim / 8)) uint8 tensor.

    Edge case: if in_dim % 8 != 0, the input is zero-padded on the right
    (pad values map to bit=0 -> weight=-1, which is fine since those positions
    are beyond the real in_dim and get masked by the kernel's IN_DIM bound).
    """
    out_dim, in_dim = binary_int8.shape

    # Pad to multiple of 8
    pad_amount = (8 - in_dim % 8) % 8
    if pad_amount > 0:
        padding = mx.full((out_dim, pad_amount), -1, dtype=mx.int8)
        binary_int8 = mx.concatenate([binary_int8, padding], axis=1)

    padded_in = binary_int8.shape[1]
    # Map {-1, +1} -> {0, 1}
    bits = ((binary_int8 + 1) // 2).astype(mx.uint8)  # (out_dim, padded_in)

    # Reshape to groups of 8 and pack LSB-first
    bits = bits.reshape(out_dim, padded_in // 8, 8)
    shifts = mx.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=mx.uint8)
    packed = mx.sum(bits * (mx.array(1, dtype=mx.uint8) << shifts), axis=2).astype(mx.uint8)
    return packed


def unpack_binary_weights(packed: mx.array, numel: int) -> mx.array:
    """Unpack a uint8 packed tensor back to {-1, +1} int8.

    Args:
        packed: (out_dim, in_dim_packed) uint8 tensor.
        numel:  the original (unpadded) in_dim to slice to.

    Returns:
        binary_int8: (out_dim, numel) int8 tensor with values in {-1, +1}.
    """
    out_dim, in_dim_packed = packed.shape
    shifts = mx.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=mx.uint8)

    # Expand each byte into 8 bits
    expanded = packed[:, :, None]  # (out_dim, in_dim_packed, 1)
    bits = (mx.broadcast_to(expanded, (out_dim, in_dim_packed, 8)) >> shifts) & mx.array(1, dtype=mx.uint8)
    # Reshape to flat
    bits = bits.reshape(out_dim, in_dim_packed * 8)
    # Map {0, 1} -> {-1, +1}
    result = (bits.astype(mx.int8) * 2 - 1)
    return result[:, :numel]


# ---------------------------------------------------------------------------
# Metal kernel source
# ---------------------------------------------------------------------------

# M (batch dim) is passed at runtime via m_param[0] so the kernel can be
# cached per (out_dim, in_dim, group_size) — the weight dimensions that
# are fixed per layer. Only GROUP_SIZE >= 8 variant (covers group_size=64).
_KERNEL_SOURCE_FAST = """
    uint row = thread_position_in_grid.x;
    uint col = thread_position_in_grid.y;

    uint m = static_cast<uint>(m_param[0]);
    if (row >= m || col >= OUT_DIM) return;

    float acc = 0.0f;

    for (uint pb = 0; pb < IN_DIM_PACKED; pb++) {
        uint8_t packed_byte = binary_packed[col * IN_DIM_PACKED + pb];
        uint base_k = pb * 8;
        uint group_idx = base_k / GROUP_SIZE;
        float s = static_cast<float>(scale[col * NUM_GROUPS + group_idx]);

        float local_acc = 0.0f;
        uint remaining = (IN_DIM > base_k + 8) ? 8 : (IN_DIM - base_k);

        for (uint bit = 0; bit < remaining; bit++) {
            float xval = static_cast<float>(x[row * IN_DIM + base_k + bit]);
            float w = ((packed_byte >> bit) & 1) ? 1.0f : -1.0f;
            local_acc += xval * w;
        }
        acc += local_acc * s;
    }

    y[row * OUT_DIM + col] = static_cast<T>(acc);
"""


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

# Kernel cache: keyed by (out_dim, in_dim, in_dim_packed, group_size, num_groups)
_kernel_cache: dict[tuple, object] = {}


def _get_cached_kernel(out_dim: int, in_dim: int, in_dim_packed: int,
                       group_size: int, num_groups: int):
    """Build or retrieve a cached binary_matmul kernel for these weight dims."""
    key = (out_dim, in_dim, in_dim_packed, group_size, num_groups)
    if key not in _kernel_cache:
        header = (
            f"#define OUT_DIM {out_dim}\n"
            f"#define IN_DIM {in_dim}\n"
            f"#define IN_DIM_PACKED {in_dim_packed}\n"
            f"#define GROUP_SIZE {group_size}\n"
            f"#define NUM_GROUPS {num_groups}\n"
        )
        _kernel_cache[key] = mx.fast.metal_kernel(
            name="binary_matmul",
            input_names=["x", "binary_packed", "scale", "m_param"],
            output_names=["y"],
            source=_KERNEL_SOURCE_FAST,
            header=header,
        )
    return _kernel_cache[key]


def binary_matmul(
    x: mx.array,
    binary_packed: mx.array,
    scale: mx.array,
    in_dim: int,
    group_size: int = 64,
) -> mx.array:
    """Fused binary matrix multiplication via a cached Metal kernel.

    Kernels are compiled once per unique (out_dim, in_dim, group_size)
    combination and reused across calls with different batch sizes M.

    Args:
        x:              (M, in_dim) input activations.
        binary_packed:  (out_dim, in_dim_packed) uint8, packed binary weights.
        scale:          (out_dim, num_groups) float16, per-group scales.
        in_dim:         original (unpadded) input dimension.
        group_size:     number of weights per scale group (default 64).

    Returns:
        y: (M, out_dim) float32 output.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    M = x.shape[0]
    out_dim = binary_packed.shape[0]
    in_dim_packed = binary_packed.shape[1]
    num_groups = scale.shape[1]

    x_f = x.astype(mx.float32)
    scale_f16 = scale.astype(mx.float16)
    m_param = mx.array([M], dtype=mx.uint32)

    kernel = _get_cached_kernel(out_dim, in_dim, in_dim_packed, group_size, num_groups)

    tg_x = min(M, 32)
    tg_y = min(out_dim, 32)

    outputs = kernel(
        inputs=[x_f, binary_packed, scale_f16, m_param],
        template=[("T", mx.float32)],
        grid=(M, out_dim, 1),
        threadgroup=(tg_x, tg_y, 1),
        output_shapes=[(M, out_dim)],
        output_dtypes=[mx.float32],
    )

    return outputs[0]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== metal_binary_matmul self-test ===\n")

    def _test(name: str, M: int, in_dim: int, out_dim: int, group_size: int = 64):
        print(f"Test: {name}  (M={M}, in={in_dim}, out={out_dim}, gs={group_size})")

        # Generate random binary weights
        raw = mx.random.normal((out_dim, in_dim))
        binary_int8 = mx.where(raw >= 0, mx.array(1, dtype=mx.int8),
                               mx.array(-1, dtype=mx.int8))

        # Pad in_dim for grouping
        padded_in = in_dim + (group_size - in_dim % group_size) % group_size
        if padded_in > in_dim:
            pad = mx.full((out_dim, padded_in - in_dim), -1, dtype=mx.int8)
            binary_padded = mx.concatenate([binary_int8, pad], axis=1)
        else:
            binary_padded = binary_int8

        num_groups = padded_in // group_size
        scale = mx.random.uniform(shape=(out_dim, num_groups)).astype(mx.float16)

        # Pack
        packed = pack_binary_weights(binary_padded)

        # Verify pack/unpack roundtrip
        unpacked = unpack_binary_weights(packed, padded_in)
        mx.eval(unpacked, binary_padded)
        assert mx.array_equal(unpacked, binary_padded), "Pack/unpack roundtrip FAILED"
        print("  pack/unpack roundtrip: OK")

        # Reference: explicit FP matmul
        x = mx.random.normal((M, in_dim)).astype(mx.float32)
        scale_broadcast = mx.repeat(scale.astype(mx.float32), group_size, axis=1)
        w_eff = (binary_padded.astype(mx.float32) * scale_broadcast)[:, :in_dim]
        y_ref = x @ w_eff.T

        # Metal kernel
        y_metal = binary_matmul(x, packed, scale, in_dim, group_size)
        mx.eval(y_ref, y_metal)

        diff = mx.abs(y_ref - y_metal)
        max_err = mx.max(diff).item()
        mean_err = mx.mean(diff).item()
        print(f"  max_err={max_err:.6e}  mean_err={mean_err:.6e}")
        # Allow small FP tolerance (float16 scale -> float32 accumulation)
        assert max_err < 1e-2, f"Accuracy FAILED: max_err={max_err}"
        print("  accuracy: OK\n")

    # Standard case
    _test("standard", M=4, in_dim=512, out_dim=256, group_size=64)

    # in_dim not divisible by 8
    _test("non_div_8", M=2, in_dim=100, out_dim=64, group_size=64)

    # in_dim not divisible by group_size
    _test("non_div_gs", M=3, in_dim=200, out_dim=128, group_size=64)

    # Small group size
    _test("small_group", M=2, in_dim=128, out_dim=64, group_size=8)

    # Single row
    _test("single_row", M=1, in_dim=256, out_dim=128, group_size=64)

    # Larger
    _test("larger", M=32, in_dim=1024, out_dim=512, group_size=64)

    # Benchmark
    print("--- Benchmark (M=128, in=1024, out=1024, gs=64) ---")
    M_bench, in_bench, out_bench, gs_bench = 128, 1024, 1024, 64
    raw = mx.random.normal((out_bench, in_bench))
    bw = mx.where(raw >= 0, mx.array(1, dtype=mx.int8), mx.array(-1, dtype=mx.int8))
    packed = pack_binary_weights(bw)
    num_g = in_bench // gs_bench
    sc = mx.random.uniform(shape=(out_bench, num_g)).astype(mx.float16)
    x_bench = mx.random.normal((M_bench, in_bench)).astype(mx.float32)
    mx.eval(packed, sc, x_bench)

    # Warm up
    for _ in range(5):
        _ = binary_matmul(x_bench, packed, sc, in_bench, gs_bench)
        mx.eval(_)

    t0 = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        out = binary_matmul(x_bench, packed, sc, in_bench, gs_bench)
        mx.eval(out)
    t1 = time.perf_counter()
    print(f"  {n_iters} iters in {t1-t0:.3f}s -> {(t1-t0)/n_iters*1e3:.2f} ms/iter")

    print("\nAll tests passed.")
