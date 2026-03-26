#!/usr/bin/env python3
"""
Log parser and monitor for training experiments.

Usage:
    python -m ssm_bin_es.experiment.monitor logs/*.txt
    python -m ssm_bin_es.experiment.monitor logs/phase1_*.txt --curves
    python -m ssm_bin_es.experiment.monitor logs/phase1_*.txt --watch
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenRecord:
    gen: int
    best_loss: float | None = None
    train_loss: float | None = None
    mean_loss: float | None = None
    flip_rate: float | None = None
    gen_time: float | None = None
    elapsed: float | None = None
    grad_steps: int | None = None

    @property
    def loss(self) -> float | None:
        return self.best_loss or self.train_loss


@dataclass
class ValRecord:
    gen: int
    val_loss: float
    val_bpb: float
    elapsed: float | None = None


@dataclass
class RunConfig:
    mode: str | None = None
    num_layers: int | None = None
    model_dim: int | None = None
    model_params: int | None = None
    estimated_mb: float | None = None


@dataclass
class RunLog:
    name: str
    path: Path
    config: RunConfig = field(default_factory=RunConfig)
    gen_records: list[GenRecord] = field(default_factory=list)
    val_records: list[ValRecord] = field(default_factory=list)
    final_val_loss: float | None = None
    final_val_bpb: float | None = None
    total_time: float | None = None
    total_generations: int | None = None
    total_grad_steps: int | None = None


KV_RE = re.compile(r'(\w+):([\d.eE+\-]+)')


def parse_kv(line: str) -> dict[str, str]:
    """Extract key:value pairs from a log line."""
    return dict(KV_RE.findall(line))


def parse_log(path: Path) -> RunLog:
    """Parse a training log file into structured records."""
    name = path.stem
    log = RunLog(name=name, path=path)

    if not path.exists():
        return log

    text = path.read_text(errors="replace")
    last_elapsed = None

    for line in text.splitlines():
        kv = parse_kv(line)

        # Config lines
        if "mode:" in line and "Starting" not in line and "gen:" not in line:
            log.config.mode = kv.get("mode")
        if line.startswith("mode:"):
            log.config.mode = kv.get("mode")
        if "model_params:" in line and "gen:" not in line:
            try:
                log.config.model_params = int(kv["model_params"])
            except (KeyError, ValueError):
                pass
        if "estimated_serialized:" in line:
            try:
                log.config.estimated_mb = float(kv.get("estimated_serialized", "0"))
            except ValueError:
                pass
        if "layers:" in line and "dim:" in line and "gen:" not in line:
            try:
                log.config.num_layers = int(kv.get("layers", "0"))
                log.config.model_dim = int(kv.get("dim", "0"))
            except ValueError:
                pass

        # Generation records (both hybrid and gradient modes)
        if line.startswith("gen:") and "val_loss:" not in line and "FINAL" not in line:
            rec = GenRecord(gen=int(kv.get("gen", "0")))
            if "best_loss" in kv:
                rec.best_loss = float(kv["best_loss"])
            if "train_loss" in kv:
                rec.train_loss = float(kv["train_loss"])
            if "mean_loss" in kv:
                rec.mean_loss = float(kv["mean_loss"])
            if "flip_rate" in kv:
                rec.flip_rate = float(kv["flip_rate"])
            if "gen_time" in kv:
                rec.gen_time = float(kv["gen_time"])
            if "elapsed" in kv:
                rec.elapsed = float(kv["elapsed"])
                last_elapsed = rec.elapsed
            if "grad_steps" in kv:
                rec.grad_steps = int(kv["grad_steps"])
            log.gen_records.append(rec)

        # Validation records
        if line.startswith("val gen:") or (line.startswith("val ") and "val_bpb:" in line):
            try:
                rec = ValRecord(
                    gen=int(kv.get("gen", "0")),
                    val_loss=float(kv.get("val_loss", "0")),
                    val_bpb=float(kv.get("val_bpb", "0")),
                    elapsed=last_elapsed,
                )
                log.val_records.append(rec)
            except (KeyError, ValueError):
                pass

        # Final record
        if "FINAL" in line and "val_bpb:" in line:
            try:
                log.final_val_bpb = float(kv["val_bpb"])
                log.final_val_loss = float(kv["val_loss"])
                log.total_time = float(kv.get("total_time", "0"))
                log.total_generations = int(kv.get("generations", "0"))
                log.total_grad_steps = int(kv.get("grad_steps", "0")) if "grad_steps" in kv else None
            except (KeyError, ValueError):
                pass

    return log


def print_summary(logs: list[RunLog]) -> None:
    """Print a comparison summary table."""
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*80}")

    header = (f"{'Run':<25} {'Mode':<12} {'Params':>8} {'MB':>6} "
              f"{'Final BPB':>10} {'Final Loss':>10} {'Time':>6} {'Gens':>6}")
    print(header)
    print("-" * 80)

    # Sort by final BPB (best first)
    sorted_logs = sorted(logs, key=lambda l: l.final_val_bpb if l.final_val_bpb else 999)

    for log in sorted_logs:
        mode = log.config.mode or "?"
        params = f"{log.config.model_params/1e6:.1f}M" if log.config.model_params else "?"
        mb = f"{log.config.estimated_mb:.1f}" if log.config.estimated_mb else "?"
        bpb = f"{log.final_val_bpb:.4f}" if log.final_val_bpb else "running..."
        loss = f"{log.final_val_loss:.4f}" if log.final_val_loss else "..."
        time_s = f"{log.total_time:.0f}s" if log.total_time else "..."
        gens = str(log.total_generations) if log.total_generations else "..."

        print(f"{log.name:<25} {mode:<12} {params:>8} {mb:>6} "
              f"{bpb:>10} {loss:>10} {time_s:>6} {gens:>6}")

    if any(l.final_val_bpb for l in sorted_logs):
        best = sorted_logs[0]
        print(f"\nBest: {best.name} with val_bpb={best.final_val_bpb:.4f}")


def print_learning_curves(logs: list[RunLog], time_points: list[float] | None = None) -> None:
    """Print val_bpb at matched wall-time intervals."""
    if time_points is None:
        time_points = [30, 60, 120, 180, 240, 300]

    print(f"\n{'='*80}")
    print("LEARNING CURVES (val_bpb at wall-time checkpoints)")
    print(f"{'='*80}")

    header = f"{'Run':<25} " + " ".join(f"{t:>7}s" for t in time_points)
    print(header)
    print("-" * (25 + 8 * len(time_points)))

    for log in logs:
        if not log.val_records:
            continue
        vals = []
        for t in time_points:
            # Find closest val record by elapsed time
            best_rec = None
            for rec in log.val_records:
                if rec.elapsed is not None and rec.elapsed <= t:
                    best_rec = rec
            if best_rec:
                vals.append(f"{best_rec.val_bpb:.4f}")
            else:
                vals.append("   -   ")
        print(f"{log.name:<25} " + " ".join(f"{v:>7}" for v in vals))


def print_training_progress(logs: list[RunLog]) -> None:
    """Print latest training metrics for each run (for --watch mode)."""
    print(f"\n--- {time.strftime('%H:%M:%S')} ---")
    for log in logs:
        if log.gen_records:
            latest = log.gen_records[-1]
            loss_str = f"loss={latest.loss:.4f}" if latest.loss else "?"
            elapsed_str = f"{latest.elapsed:.0f}s" if latest.elapsed else "?"
            gen_str = f"gen={latest.gen}"
        else:
            loss_str = "no data"
            elapsed_str = "?"
            gen_str = "?"

        val_str = ""
        if log.val_records:
            latest_val = log.val_records[-1]
            val_str = f" val_bpb={latest_val.val_bpb:.4f}"

        final_str = ""
        if log.final_val_bpb:
            final_str = f" FINAL={log.final_val_bpb:.4f}"

        print(f"  {log.name:<25} {gen_str} {loss_str} {elapsed_str}{val_str}{final_str}")


def main():
    parser = argparse.ArgumentParser(description="Monitor training experiments")
    parser.add_argument("logs", nargs="+", help="Log file paths (supports globs)")
    parser.add_argument("--curves", action="store_true", help="Show learning curves")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval (seconds)")
    args = parser.parse_args()

    log_paths = []
    for pattern in args.logs:
        p = Path(pattern)
        if p.exists():
            log_paths.append(p)
        else:
            log_paths.extend(Path(".").glob(pattern))

    if not log_paths:
        print("No log files found.")
        sys.exit(1)

    if args.watch:
        try:
            while True:
                logs = [parse_log(p) for p in log_paths]
                print_training_progress(logs)
                all_done = all(l.final_val_bpb is not None for l in logs)
                if all_done:
                    print("\nAll runs complete!")
                    print_summary(logs)
                    if args.curves:
                        print_learning_curves(logs)
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nInterrupted. Final state:")
            logs = [parse_log(p) for p in log_paths]
            print_summary(logs)
    else:
        logs = [parse_log(p) for p in log_paths]
        print_summary(logs)
        if args.curves:
            print_learning_curves(logs)


if __name__ == "__main__":
    main()
