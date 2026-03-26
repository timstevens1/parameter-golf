"""
Metal kernels for evolutionary strategy operations on packed binary weights.

Binary weights are packed as uint8 arrays (8 weights per byte, bit=0 -> -1,
bit=1 -> +1). These kernels operate directly on the packed representation,
avoiding the 8x memory expansion of unpacking to int8.

Kernels:
    mutate_packed  - Flip random bits with a given probability (XOR mutation)
    crossover_packed - Uniform crossover between two packed parents
"""
from __future__ import annotations

import mlx.core as mx


# ---------------------------------------------------------------------------
# Metal kernel sources
# ---------------------------------------------------------------------------

MUTATE_PACKED_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= N) return;

    uint8_t thr = threshold[0];
    uint rand_base = idx * 8;

    uint8_t flip_mask = 0;
    for (int bit = 0; bit < 8; bit++) {
        if (random_vals[rand_base + bit] < thr) {
            flip_mask |= (1 << bit);
        }
    }

    child[idx] = parent[idx] ^ flip_mask;
"""

CROSSOVER_PACKED_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= N) return;

    child[idx] = (parent_a[idx] & ~mask[idx]) | (parent_b[idx] & mask[idx]);
"""


# ---------------------------------------------------------------------------
# Kernel constructors (lazily cached)
# ---------------------------------------------------------------------------

def _build_mutate_kernel(n: int):
    """Build the mutate_packed Metal kernel for a given size."""
    return mx.fast.metal_kernel(
        name="mutate_packed",
        input_names=["parent", "random_vals", "threshold"],
        output_names=["child"],
        source=MUTATE_PACKED_SOURCE,
        header=f"#define N {n}\n",
    )


def _build_crossover_kernel(n: int):
    """Build the crossover_packed Metal kernel for a given size."""
    return mx.fast.metal_kernel(
        name="crossover_packed",
        input_names=["parent_a", "parent_b", "mask"],
        output_names=["child"],
        source=CROSSOVER_PACKED_SOURCE,
        header=f"#define N {n}\n",
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def mutate_packed(parent: mx.array, flip_rate: float) -> mx.array:
    """Mutate packed binary weights by flipping bits with probability flip_rate.

    Each bit in the packed uint8 representation is independently flipped with
    probability ``flip_rate``. Internally this generates 8 random bytes per
    packed byte for fine-grained per-bit control, compares each against a
    threshold, builds a flip mask, and XORs with the parent.

    Args:
        parent: Packed binary weights as a uint8 array (arbitrary shape).
                Each byte encodes 8 binary weights.
        flip_rate: Probability of flipping each bit, in [0.0, 1.0].

    Returns:
        A new uint8 array of the same shape with mutated packed weights.
    """
    original_shape = parent.shape
    flat = parent.reshape(-1)
    n = flat.size

    # Convert flip_rate to uint8 threshold (0-255).
    # A random byte < threshold means "flip". threshold=0 -> never flip,
    # threshold=255 -> flip with probability 254/255.
    threshold_val = int(round(flip_rate * 255.0))
    threshold_val = max(0, min(255, threshold_val))
    threshold = mx.array([threshold_val], dtype=mx.uint8)

    # Generate 8 random bytes per packed byte for per-bit control
    random_vals = mx.random.randint(
        0, 256, shape=(n * 8,), dtype=mx.uint8
    )

    kernel = _build_mutate_kernel(n)
    threads_per_group = min(256, n)
    grid_x = (n + threads_per_group - 1) // threads_per_group * threads_per_group

    outputs = kernel(
        inputs=[flat, random_vals, threshold],
        grid=(grid_x, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.uint8],
    )

    return outputs[0].reshape(original_shape)


def crossover_packed(parent_a: mx.array, parent_b: mx.array) -> mx.array:
    """Uniform crossover between two packed binary weight arrays.

    For each bit position, independently selects from parent_a or parent_b
    with equal probability. Implemented as:
        child = (parent_a & ~mask) | (parent_b & mask)
    where mask is a random uint8 array (each bit independently random).

    Args:
        parent_a: First parent, packed uint8 array (arbitrary shape).
        parent_b: Second parent, packed uint8 array (same shape as parent_a).

    Returns:
        A new uint8 array of the same shape with crossover result.

    Raises:
        ValueError: If parent_a and parent_b have different shapes.
    """
    if parent_a.shape != parent_b.shape:
        raise ValueError(
            f"Parent shapes must match: {parent_a.shape} vs {parent_b.shape}"
        )

    original_shape = parent_a.shape
    flat_a = parent_a.reshape(-1)
    flat_b = parent_b.reshape(-1)
    n = flat_a.size

    # Random bit mask -- each bit independently 0 or 1
    mask = mx.random.randint(0, 256, shape=(n,), dtype=mx.uint8)

    kernel = _build_crossover_kernel(n)
    threads_per_group = min(256, n)
    grid_x = (n + threads_per_group - 1) // threads_per_group * threads_per_group

    outputs = kernel(
        inputs=[flat_a, flat_b, mask],
        grid=(grid_x, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.uint8],
    )

    return outputs[0].reshape(original_shape)
