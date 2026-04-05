"""
run_exp1.py  -  Experiment 1: R=50, Gamma=10, 4 policies x 4 DI x 50 seeds = 800 runs

Base re-run with new DI base (sqrt(1.6)) and epsilon=100.
Results saved to ../result/exp1/
Estimated time: ~1 hour (sequential)
"""

import subprocess
import sys
import os
import time
from itertools import product

SEEDS = range(1, 51)
DI_SCENARIOS = ['HD', 'MD', 'LD', 'Mixed']
GAMMA = 10
INSTANCE = 'full'
RESULT_DIR = os.path.join('..', 'result', 'exp1')

def run_one(cmd, label):
    """Run a single experiment and return (label, success, elapsed)."""
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    success = result.returncode == 0
    if not success:
        print(f"  FAILED: {label}")
        print(f"  stderr: {result.stderr[-200:]}")
    return label, success, elapsed

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Build all jobs
    jobs = []

    # Optimal policy
    for seed, di in product(SEEDS, DI_SCENARIOS):
        cmd = [sys.executable, 'main.py', INSTANCE, str(GAMMA),
               '--seed', str(seed), '--di', di, '--result_dir', RESULT_DIR]
        jobs.append((cmd, f"Optimal seed={seed} DI={di}"))

    # Fixed mode policies (FM0, FM1, FM2)
    for mode, seed, di in product([0, 1, 2], SEEDS, DI_SCENARIOS):
        cmd = [sys.executable, 'main_fixed_mode.py', INSTANCE, str(GAMMA),
               '--seed', str(seed), '--di', di, '--mode', str(mode), '--result_dir', RESULT_DIR]
        jobs.append((cmd, f"FM{mode} seed={seed} DI={di}"))

    total = len(jobs)
    print(f"=" * 60)
    print(f"EXPERIMENT 1: {total} runs")
    print(f"  Instance: {INSTANCE}, Gamma: {GAMMA}")
    print(f"  Results: {os.path.abspath(RESULT_DIR)}")
    print(f"=" * 60)

    completed = 0
    failed = 0
    total_time = 0
    start_all = time.time()

    for cmd, label in jobs:
        completed += 1
        print(f"[{completed}/{total}] {label}...", end=" ", flush=True)
        _, success, elapsed = run_one(cmd, label)
        total_time += elapsed
        if success:
            print(f"OK ({elapsed:.1f}s)")
        else:
            failed += 1

        # Progress estimate
        if completed % 50 == 0:
            avg = total_time / completed
            remaining = (total - completed) * avg
            print(f"  --- Progress: {completed}/{total}, "
                  f"avg {avg:.1f}s/run, ETA {remaining/60:.0f}min ---")

    wall_time = time.time() - start_all
    print(f"\n{'=' * 60}")
    print(f"DONE: {completed - failed}/{total} succeeded, {failed} failed")
    print(f"Wall time: {wall_time/60:.1f} min")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
