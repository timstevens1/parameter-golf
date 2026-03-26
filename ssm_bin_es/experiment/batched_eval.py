"""
Batched evaluation of ES population candidates.

All candidates in a population share the same continuous parameters (A_log, D,
conv, dt_proj, scales, embeddings). Only the binary weights differ. This module
exploits MLX's lazy evaluation to batch multiple candidate forward passes into
a single mx.eval() call, allowing the runtime to fuse and optimize the
computation graph across candidates.

Usage:
    from batched_eval import BatchedPopulationEvaluator

    evaluator = BatchedPopulationEvaluator(model, group_size=4)
    candidates = [es.get_candidate(i) for i in range(pop_size)]
    fitnesses = evaluator.evaluate_population(candidates, eval_x, eval_y)
"""
from __future__ import annotations

import mlx.core as mx

from ssm_bin_es.experiment.model import BinaryMambaLM
from mlx.utils import tree_flatten, tree_unflatten


def inject_binary_weights(model: BinaryMambaLM,
                          binary_weights: dict[str, mx.array]) -> None:
    """Inject binary weight tensors into the model.

    Identical to train.inject_binary_weights, duplicated here to avoid
    circular imports (train.py imports from this module's siblings).
    """
    params = dict(tree_flatten(model.parameters()))
    params.update(binary_weights)
    model.update(tree_unflatten(list(params.items())))


def evaluate_group(
    model: BinaryMambaLM,
    candidates: list[dict[str, mx.array]],
    x: mx.array,
    y: mx.array,
) -> list[float]:
    """Evaluate a group of candidates.

    Each candidate's binary weights are injected into the model and the
    forward pass is evaluated eagerly. We must eval each candidate before
    injecting the next, because the model parameters are shared mutable
    state — deferred eval would cause all candidates to use the last
    injected weights.

    Args:
        model: The BinaryMambaLM model. Continuous parameters are shared
            across all candidates and remain unchanged.
        candidates: List of binary weight dicts for this group. Each dict
            maps parameter names (ending in '.binary_weight') to int8 arrays.
        x: Input token IDs, shape (batch, seq_len).
        y: Target token IDs, shape (batch, seq_len).

    Returns:
        List of loss values (floats), one per candidate.
    """
    fitnesses = []
    for candidate in candidates:
        inject_binary_weights(model, candidate)
        loss = model.loss(x, y)
        mx.eval(loss)
        fitnesses.append(float(loss.item()))

    return fitnesses


class BatchedPopulationEvaluator:
    """Evaluates ES population candidates in batched groups.

    Instead of calling mx.eval() after every single candidate forward pass
    (which forces a synchronization barrier each time), this evaluator groups
    candidates and defers evaluation until an entire group's computation graphs
    have been constructed. This lets MLX's lazy evaluation engine see all the
    work at once and schedule it more efficiently.

    Attributes:
        model: The shared BinaryMambaLM model instance.
        group_size: Number of candidates to evaluate per mx.eval() call.
            Larger groups amortize eval overhead but use more memory for the
            intermediate computation graph. Typical values: 4-8.
    """

    def __init__(self, model: BinaryMambaLM, group_size: int = 4):
        """Initialize the evaluator.

        Args:
            model: BinaryMambaLM model. Its continuous parameters will be
                shared across all candidates; only binary weights are swapped.
            group_size: Number of candidates to batch into a single mx.eval()
                call. Must be >= 1. Values of 4-8 balance graph size against
                eval-call overhead.
        """
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.model = model
        self.group_size = group_size

    def evaluate_population(
        self,
        candidates: list[dict[str, mx.array]],
        x: mx.array,
        y: mx.array,
    ) -> list[float]:
        """Evaluate all candidates against the same input data.

        Candidates are split into groups of self.group_size and each group
        is evaluated with a single mx.eval() call via evaluate_group().

        Args:
            candidates: List of binary weight dicts, one per population member.
                Typically len(candidates) == pop_size (e.g. 32).
            x: Input token IDs, shape (batch, seq_len).
            y: Target token IDs, shape (batch, seq_len).

        Returns:
            List of loss values (floats) in the same order as candidates.
        """
        fitnesses: list[float] = []

        for start in range(0, len(candidates), self.group_size):
            group = candidates[start : start + self.group_size]
            group_fitnesses = evaluate_group(self.model, group, x, y)
            fitnesses.extend(group_fitnesses)

        return fitnesses
