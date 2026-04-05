"""
run_breakeven.py  -  Breakeven Analysis: TC_2 sensitivity, LD only, Optimal + FM2 = 500 runs

Tests when fast delivery (FM2) becomes unprofitable as transport cost increases.
TC_2 in {0.20, 0.40, 0.60, 0.80, 1.00}, LD scenario, 50 seeds, 2 policies.
Results saved to ../result/breakeven/
Estimated time: ~40 minutes (sequential)
"""

import subprocess
import sys
import os
import time
from itertools import product

SEEDS = range(1, 51)
TC2_VALUES = [0.20, 0.40, 0.60, 0.80, 1.00]
GAMMA = 10
INSTANCE = 'full'
DI = 'LD'
RESULT_DIR = os.path.join('..', 'result', 'breakeven')

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    jobs = []

    for tc2, seed in product(TC2_VALUES, SEEDS):
        # Optimal
        cmd = [sys.executable, 'main.py', INSTANCE, str(GAMMA),
               '--seed', str(seed), '--di', DI, '--tc2', str(tc2), '--result_dir', RESULT_DIR]
        jobs.append((cmd, f"Optimal TC2={tc2} seed={seed}"))

        # FM2
        cmd = [sys.executable, 'main_fixed_mode.py', INSTANCE, str(GAMMA),
               '--seed', str(seed), '--di', DI, '--mode', '2', '--tc2', str(tc2), '--result_dir', RESULT_DIR]
        jobs.append((cmd, f"FM2 TC2={tc2} seed={seed}"))

    total = len(jobs)
    print(f"=" * 60)
    print(f"BREAKEVEN ANALYSIS: {total} runs")
    print(f"  TC_2 values: {TC2_VALUES}")
    print(f"  DI: {DI}, Gamma: {GAMMA}")
    print(f"=" * 60)

    completed = 0
    failed = 0
    start_all = time.time()

    for cmd, label in jobs:
        completed += 1
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        success = result.returncode == 0

        if completed % 50 == 0 or not success:
            wall = time.time() - start_all
            avg = wall / completed
            eta = (total - completed) * avg
            status = "OK" if success else "FAIL"
            print(f"[{completed}/{total}] {status} {label} ({elapsed:.0f}s) | ETA {eta/60:.0f}min")

        if not success:
            failed += 1

    wall_time = time.time() - start_all
    print(f"\n{'=' * 60}")
    print(f"DONE: {completed - failed}/{total} succeeded, {failed} failed")
    print(f"Wall time: {wall_time/60:.1f} min")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
