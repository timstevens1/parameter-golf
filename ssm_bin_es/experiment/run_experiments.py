#!/usr/bin/env python3
"""
Experiment runner: launch parallel training runs with different configs.

Usage:
    python -m ssm_bin_es.experiment.run_experiments [phase]

Where [phase] is: phase1, phase2a, phase2b, phase2c, phase2d, phase3, or a custom JSON file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    name: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentPhase:
    name: str
    experiments: list[ExperimentConfig]
    max_parallel: int = 3


# ---- Shared defaults ----
DEFAULTS = {
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "USE_METAL_SCAN": "1",
    "SEED": "1337",
    "OUT_DIR": "logs",
}

# ---- Phase definitions ----

PHASE1_SHARED = {
    "NUM_LAYERS": "4",
    "MODEL_DIM": "256",
    "STATE_DIM": "16",
    "EXPAND_FACTOR": "2",
    "MLP_MULT": "2",
    "TRAIN_SEQ_LEN": "256",
    "ES_GENERATIONS": "10000",
    "MAX_WALLCLOCK_SECONDS": "300",
    "GRADIENT_BATCH_TOKENS": "8192",
    "ES_EVAL_TOKENS": "8192",
    "VAL_LOSS_EVERY": "25",
    "TRAIN_LOG_EVERY": "5",
    "CONTINUOUS_LR": "0.01",
}

PHASE1 = ExperimentPhase(
    name="phase1_is_es_helping",
    max_parallel=1,  # Metal GPU deadlocks with >1 concurrent MLX process
    experiments=[
        ExperimentConfig(
            name="phase1_hybrid",
            env={**PHASE1_SHARED,
                 "MODE": "hybrid", "POP_SIZE": "16",
                 "GRADIENT_STEPS_PER_GEN": "4",
                 "INIT_FLIP_RATE": "0.01"},
        ),
        ExperimentConfig(
            name="phase1_gradient",
            env={**PHASE1_SHARED,
                 "MODE": "gradient_only", "POP_SIZE": "1",
                 "GRADIENT_STEPS_PER_GEN": "16"},
        ),
        ExperimentConfig(
            name="phase1_rebinarize",
            env={**PHASE1_SHARED,
                 "MODE": "gradient_rebinarize", "POP_SIZE": "1",
                 "GRADIENT_STEPS_PER_GEN": "16",
                 "REBINARIZE_EVERY": "4"},
        ),
    ],
)


def build_phase2a(winner_env: dict[str, str]) -> ExperimentPhase:
    """Architecture scale sweep based on Phase 1 winner."""
    base = {**winner_env, "MAX_WALLCLOCK_SECONDS": "300"}
    experiments = []
    for layers, dim in [(4, 256), (8, 384), (12, 512)]:
        experiments.append(ExperimentConfig(
            name=f"phase2a_L{layers}_D{dim}",
            env={**base, "NUM_LAYERS": str(layers), "MODEL_DIM": str(dim)},
        ))
    return ExperimentPhase(name="phase2a_scale", experiments=experiments, max_parallel=3)


def build_phase2b(winner_env: dict[str, str]) -> ExperimentPhase:
    """Learning rate sweep."""
    base = {**winner_env, "MAX_WALLCLOCK_SECONDS": "300"}
    experiments = []
    for lr in [0.005, 0.01, 0.02]:
        experiments.append(ExperimentConfig(
            name=f"phase2b_lr{lr}",
            env={**base, "CONTINUOUS_LR": str(lr)},
        ))
    return ExperimentPhase(name="phase2b_lr", experiments=experiments, max_parallel=3)


def build_phase2c(winner_env: dict[str, str]) -> ExperimentPhase:
    """Sequence length sweep."""
    base = {**winner_env, "MAX_WALLCLOCK_SECONDS": "300"}
    experiments = []
    for seq_len in [256, 512, 1024]:
        experiments.append(ExperimentConfig(
            name=f"phase2c_seq{seq_len}",
            env={**base, "TRAIN_SEQ_LEN": str(seq_len)},
        ))
    return ExperimentPhase(name="phase2c_seq", experiments=experiments, max_parallel=3)


def build_phase2d(winner_env: dict[str, str]) -> ExperimentPhase:
    """ES-specific hyperparameter sweep (only if hybrid won Phase 1)."""
    base = {**winner_env, "MAX_WALLCLOCK_SECONDS": "300"}
    experiments = []
    for pop_size, grad_steps in [(8, 2), (8, 8), (32, 2), (32, 8)]:
        experiments.append(ExperimentConfig(
            name=f"phase2d_pop{pop_size}_gs{grad_steps}",
            env={**base, "POP_SIZE": str(pop_size),
                 "GRADIENT_STEPS_PER_GEN": str(grad_steps)},
        ))
    return ExperimentPhase(name="phase2d_es_params", experiments=experiments, max_parallel=2)


# ---- Process management ----

def run_phase(phase: ExperimentPhase) -> dict[str, Path]:
    """Run all experiments in a phase, max_parallel at a time."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(phase.experiments)
    running: dict[str, tuple[subprocess.Popen, Path, float]] = {}
    completed: dict[str, Path] = {}

    print(f"\n{'='*60}")
    print(f"Phase: {phase.name}")
    print(f"Experiments: {len(phase.experiments)}, max_parallel: {phase.max_parallel}")
    print(f"{'='*60}\n")

    while pending or running:
        # Launch up to max_parallel
        while pending and len(running) < phase.max_parallel:
            config = pending.pop(0)
            env = {**os.environ, **DEFAULTS, **config.env, "RUN_ID": config.name}
            proc = subprocess.Popen(
                [sys.executable, "-m", "ssm_bin_es.experiment.train"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            log_path = log_dir / f"{config.name}.txt"
            running[config.name] = (proc, log_path, time.time())
            print(f"  [LAUNCH] {config.name} (PID {proc.pid})")

        # Poll for completion
        time.sleep(5)
        for name in list(running):
            proc, log_path, start_t = running[name]
            if proc.poll() is not None:
                elapsed = time.time() - start_t
                status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
                if proc.returncode != 0:
                    stderr = proc.stderr.read().decode() if proc.stderr else ""
                    print(f"  [ERROR] {name}: {stderr[-500:]}")
                # Try to extract final BPB from log
                final_bpb = _extract_final_bpb(log_path)
                bpb_str = f" val_bpb={final_bpb:.4f}" if final_bpb else ""
                print(f"  [DONE]   {name} -> {status} ({elapsed:.0f}s){bpb_str}")
                completed[name] = log_path
                del running[name]

        # Status update
        if running:
            summaries = []
            for name, (proc, log_path, start_t) in running.items():
                elapsed = time.time() - start_t
                latest = _extract_latest_metric(log_path)
                summaries.append(f"{name}({elapsed:.0f}s{latest})")
            print(f"  [RUNNING] {', '.join(summaries)}")

    print(f"\nPhase {phase.name} complete. {len(completed)} runs finished.\n")
    return completed


def _extract_final_bpb(log_path: Path) -> float | None:
    """Extract final val_bpb from a log file."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text()
        for line in reversed(text.splitlines()):
            if "FINAL" in line and "val_bpb:" in line:
                for part in line.split():
                    if part.startswith("val_bpb:"):
                        return float(part.split(":")[1])
    except Exception:
        pass
    return None


def _extract_latest_metric(log_path: Path) -> str:
    """Extract the latest training loss from a log file for status display."""
    if not log_path.exists():
        return ""
    try:
        text = log_path.read_text()
        for line in reversed(text.splitlines()):
            if "best_loss:" in line:
                for part in line.split():
                    if part.startswith("best_loss:"):
                        return f",loss={part.split(':')[1]}"
            if "train_loss:" in line:
                for part in line.split():
                    if part.startswith("train_loss:"):
                        return f",loss={part.split(':')[1]}"
    except Exception:
        pass
    return ""


def load_phase_from_json(path: str) -> ExperimentPhase:
    """Load a custom phase from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    experiments = [ExperimentConfig(name=e["name"], env=e.get("env", {}))
                   for e in data["experiments"]]
    return ExperimentPhase(
        name=data.get("name", Path(path).stem),
        experiments=experiments,
        max_parallel=data.get("max_parallel", 3),
    )


PHASES = {
    "phase1": PHASE1,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ssm_bin_es.experiment.run_experiments <phase>")
        print(f"Available phases: {', '.join(PHASES.keys())}")
        print("Or pass a JSON file path for custom phases.")
        sys.exit(1)

    phase_name = sys.argv[1]
    if phase_name in PHASES:
        phase = PHASES[phase_name]
    elif os.path.exists(phase_name):
        phase = load_phase_from_json(phase_name)
    else:
        print(f"Unknown phase: {phase_name}")
        sys.exit(1)

    results = run_phase(phase)

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Run':<30} {'Final BPB':>12} {'Status':>8}")
    print("-" * 52)
    for name, log_path in sorted(results.items()):
        bpb = _extract_final_bpb(log_path)
        bpb_str = f"{bpb:.4f}" if bpb else "N/A"
        print(f"{name:<30} {bpb_str:>12} {'OK':>8}")


if __name__ == "__main__":
    main()
