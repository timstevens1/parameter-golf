"""
Metal scan kernel integration for the binary Mamba model.

Provides monkey-patching to replace the Python sequential selective scan
in BinaryMambaBlock with Metal GPU kernels.

Modes:
    0 - Python sequential scan (default, no patching)
    1 - Metal forward-only kernel (inference/eval only, no gradients)
    2 - Metal forward+backward via mx.custom_function (full gradient support)
"""
from __future__ import annotations

import mlx.core as mx

from .metal_ssm_fwd import run_selective_scan_fwd
from .metal_ssm_bwd import selective_scan_with_grad


def _selective_scan_metal_fwd_only(self, x: mx.array, dt: mx.array,
                                   B: mx.array, C: mx.array) -> mx.array:
    """Metal forward-only selective scan (mode=1)."""
    y_f32 = run_selective_scan_fwd(
        x.astype(mx.float32), dt.astype(mx.float32),
        B.astype(mx.float32), C.astype(mx.float32),
        self.A_log.astype(mx.float32),
    )
    return y_f32.astype(x.dtype)


def _selective_scan_metal_fwd_bwd(self, x: mx.array, dt: mx.array,
                                  B: mx.array, C: mx.array) -> mx.array:
    """Metal forward+backward selective scan (mode=2)."""
    y_f32 = selective_scan_with_grad(
        x.astype(mx.float32), dt.astype(mx.float32),
        B.astype(mx.float32), C.astype(mx.float32),
        self.A_log.astype(mx.float32),
    )
    return y_f32.astype(x.dtype)


def _find_mamba_blocks(model) -> list:
    """Walk the model tree and return all unique BinaryMambaBlock instances."""
    from ssm_bin_es.experiment.model import BinaryMambaBlock

    blocks = []
    seen_ids = set()
    for layer in model.layers:
        block = layer.mamba
        if isinstance(block, BinaryMambaBlock) and id(block) not in seen_ids:
            blocks.append(block)
            seen_ids.add(id(block))
    return blocks


def patch_model_for_metal_scan(model, mode: int = 0) -> None:
    """Monkey-patch all BinaryMambaBlock instances to use a Metal scan kernel.

    Args:
        model: BinaryMambaLM instance.
        mode: 0=Python, 1=Metal fwd-only, 2=Metal fwd+bwd.
    """
    if mode not in (0, 1, 2):
        raise ValueError(f"mode must be 0, 1, or 2, got {mode}")

    from ssm_bin_es.experiment.model import BinaryMambaBlock

    blocks = _find_mamba_blocks(model)

    if mode == 0:
        for block in blocks:
            block._selective_scan = BinaryMambaBlock._selective_scan.__get__(block)
    elif mode == 1:
        for block in blocks:
            block._selective_scan = _selective_scan_metal_fwd_only.__get__(block)
    elif mode == 2:
        for block in blocks:
            block._selective_scan = _selective_scan_metal_fwd_bwd.__get__(block)

    mode_names = {0: "Python sequential", 1: "Metal forward-only", 2: "Metal forward+backward"}
    key = f"_printed_mode_{mode}"
    if not getattr(patch_model_for_metal_scan, key, False):
        print(f"[scan_integration] Patched {len(blocks)} block(s) -> {mode_names[mode]}")
        setattr(patch_model_for_metal_scan, key, True)
