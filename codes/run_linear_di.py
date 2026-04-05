"""
run_linear_di.py  -  Linear DI Robustness Check: 2 functions x 4 DI x 10 seeds x 4 policies = 320 runs

Compares exponential DI (base sqrt(1.6)) vs linear DI (1 + 0.3*k*m).
Both anchored to same endpoint: DI=1.60 at kappa=1, m=2.
Results saved to ../result/linear_di/
Estimated time: ~30 minutes (sequential)
"""

import subprocess
import sys
import os
import time
from itertools import product

SEEDS = range(1, 11)  # 10 seeds (reduced experiment)
DI_SCENARIOS = ['HD', 'MD', 'LD', 'Mixed']
DI_FUNCS = ['exponential', 'linear']
GAMMA = 10
INSTANCE = 'full'
RESULT_DIR = os.path.join('..', 'result', 'linear_di')

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    jobs = []

    for di_func, seed, di in product(DI_FUNCS, SEEDS, DI_SCENARIOS):
        # Optimal
        cmd = [sys.executable, 'main.py', INSTANCE, str(GAMMA),
               '--seed', str(seed), '--di', di, '--di-func', di_func, '--result_dir', RESULT_DIR]
        jobs.append((cmd, f"Optimal {di_func} seed={seed} DI={di}"))

        # FM0, FM1, FM2
        for mode in [0, 1, 2]:
            cmd = [sys.executable, 'main_fixed_mode.py', INSTANCE, str(GAMMA),
                   '--seed', str(seed), '--di', di, '--mode', str(mode),
                   '--di-func', di_func, '--result_dir', RESULT_DIR]
            jobs.append((cmd, f"FM{mode} {di_func} seed={seed} DI={di}"))

    total = len(jobs)
    print(f"=" * 60)
    print(f"LINEAR DI COMPARISON: {total} runs")
    print(f"  Functions: {DI_FUNCS}")
    print(f"  Seeds: 1-10, Gamma: {GAMMA}")
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

        if completed % 40 == 0 or not success:
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
