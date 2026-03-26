"""
Evolutionary strategy for binary weight optimization.

Weights are stored as FP16 tensors where each value is +scale or -scale.
ES mutation = negate random elements (flip sign).
ES crossover = select elements from parent A or B.

No custom kernels needed — all operations are native MLX.
"""
from __future__ import annotations

import random as pyrandom
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class ESConfig:
    pop_size: int = 32
    elite_frac: float = 0.25
    init_flip_rate: float = 0.01
    min_flip_rate: float = 0.0001
    flip_rate_decay: float = 0.999
    flip_rate_explore_mult: float = 2.0
    stagnation_gens: int = 10
    crossover_rate: float = 0.3
    eval_tokens: int = 32768
    gradient_steps_per_gen: int = 4


@dataclass
class ESState:
    generation: int = 0
    flip_rate: float = 0.01
    best_fitness: float = float('inf')
    gens_without_improvement: int = 0
    total_evaluations: int = 0


class BinaryES:
    """Evolutionary strategy operating on FP16 binary-constrained weights.

    Each weight value is +mag or -mag. Mutation negates random elements.
    All operations are native MLX — no custom kernels.
    """

    def __init__(self, config: ESConfig, binary_params: dict[str, mx.array]):
        self.config = config
        self.state = ESState(flip_rate=config.init_flip_rate)
        self.elite_count = max(1, int(config.pop_size * config.elite_frac))

        base = {k: v for k, v in binary_params.items()}
        self.population: list[dict[str, mx.array]] = [base]
        for _ in range(config.pop_size - 1):
            self.population.append(self._mutate(base, config.init_flip_rate))

        self.fitnesses: list[float] = [float('inf')] * config.pop_size

    def _mutate(self, parent: dict[str, mx.array],
                flip_rate: float) -> dict[str, mx.array]:
        """Negate random elements with probability flip_rate."""
        child = {}
        for k, w in parent.items():
            mask = mx.random.bernoulli(flip_rate, w.shape)
            # Where mask=True, negate; else keep. This is just w * (1 - 2*mask).
            flip = mx.where(mask, mx.array(-1.0, dtype=w.dtype),
                            mx.array(1.0, dtype=w.dtype))
            child[k] = w * flip
        return child

    def _crossover(self, parent_a: dict[str, mx.array],
                   parent_b: dict[str, mx.array]) -> dict[str, mx.array]:
        """Uniform crossover: for each element, pick from A or B."""
        child = {}
        for k in parent_a:
            mask = mx.random.bernoulli(0.5, parent_a[k].shape)
            child[k] = mx.where(mask, parent_a[k], parent_b[k])
        return child

    def get_candidate(self, idx: int) -> dict[str, mx.array]:
        return self.population[idx]

    def set_fitnesses(self, fitnesses: list[float]) -> None:
        assert len(fitnesses) == len(self.population)
        self.fitnesses = fitnesses

    def step(self) -> dict[str, mx.array]:
        """Select elites, generate new population. Returns best candidate."""
        ranked = sorted(range(len(self.population)),
                        key=lambda i: self.fitnesses[i])

        elites = [self.population[ranked[i]] for i in range(self.elite_count)]
        best_fitness = self.fitnesses[ranked[0]]

        # Adaptive flip rate
        if best_fitness < self.state.best_fitness - 1e-6:
            self.state.best_fitness = best_fitness
            self.state.gens_without_improvement = 0
            self.state.flip_rate *= self.config.flip_rate_decay
        else:
            self.state.gens_without_improvement += 1
            if self.state.gens_without_improvement >= self.config.stagnation_gens:
                self.state.flip_rate = min(
                    self.state.flip_rate * self.config.flip_rate_explore_mult,
                    self.config.init_flip_rate * 2.0,
                )
                self.state.gens_without_improvement = 0

        self.state.flip_rate = max(self.state.flip_rate, self.config.min_flip_rate)

        # Generate new population
        new_population = list(elites)
        while len(new_population) < self.config.pop_size:
            parent_idx = pyrandom.randint(0, self.elite_count - 1)
            if (pyrandom.random() < self.config.crossover_rate
                    and self.elite_count > 1):
                other_idx = pyrandom.randint(0, self.elite_count - 2)
                if other_idx >= parent_idx:
                    other_idx += 1
                child = self._crossover(elites[parent_idx], elites[other_idx])
                child = self._mutate(child, self.state.flip_rate)
            else:
                child = self._mutate(elites[parent_idx], self.state.flip_rate)
            new_population.append(child)

        self.population = new_population
        self.fitnesses = [float('inf')] * self.config.pop_size
        self.state.generation += 1
        self.state.total_evaluations += self.config.pop_size

        return elites[0]

    def log_state(self) -> str:
        return (
            f"es_gen:{self.state.generation} "
            f"flip_rate:{self.state.flip_rate:.6f} "
            f"best_fitness:{self.state.best_fitness:.4f} "
            f"stagnation:{self.state.gens_without_improvement} "
            f"total_evals:{self.state.total_evaluations}"
        )
