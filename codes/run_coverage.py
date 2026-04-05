"""
run_coverage.py  -  Service Coverage SA: 3 coverage x 10 seeds x 4 DI x 4 policies = 480 runs

Tests how distance-based service coverage constraints affect VoF.
Coverage scenarios: tight (25/50th pctl), moderate (50/75th), relaxed (75/90th).
No-limit baseline reuses Exp1 results.
Results saved to ../result/coverage/
Estimated time: ~40 minutes (sequential)
"""

import subprocess
import sys
import os
import time
from itertools import product

SEEDS = range(1, 11)  # 10 seeds (reduced experiment)
DI_SCENARIOS = ['HD', 'MD', 'LD', 'Mixed']
COVERAGES = ['tight', 'moderate', 'relaxed']  # no_limit = Exp1 baseline
GAMMA = 10
INSTANCE = 'full'
RESULT_DIR = os.path.join('..', 'result', 'coverage')

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    jobs = []

    for cov, seed, di in product(COVERAGES, SEEDS, DI_SCENARIOS):
        # Optimal
        cmd = [sys.executable, 'main.py', INSTANCE, str(GAMMA),
               '--seed', str(seed), '--di', di, '--coverage', cov, '--result_dir', RESULT_DIR]
        jobs.append((cmd, f"Optimal {cov} seed={seed} DI={di}"))

        # FM0, FM1, FM2
        for mode in [0, 1, 2]:
            cmd = [sys.executable, 'main_fixed_mode.py', INSTANCE, str(GAMMA),
                   '--seed', str(seed), '--di', di, '--mode', str(mode),
                   '--coverage', cov, '--result_dir', RESULT_DIR]
            jobs.append((cmd, f"FM{mode} {cov} seed={seed} DI={di}"))

    total = len(jobs)
    print(f"=" * 60)
    print(f"COVERAGE SA: {total} runs")
    print(f"  Coverages: {COVERAGES}")
    print(f"  Seeds: 1-10, Gamma: {GAMMA}")
    print(f"=" * 60)

    completed = 0
    failed = 0
    fm2_infeasible = 0
    start_all = time.time()

    for cmd, label in jobs:
        completed += 1
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        success = result.returncode == 0

        # Track FM2 infeasibility (expected for tight coverage)
        if not success and 'FM2' in label and 'tight' in label:
            fm2_infeasible += 1

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
    if fm2_infeasible > 0:
        print(f"  FM2 infeasible (tight coverage): {fm2_infeasible} cases")
    print(f"Wall time: {wall_time/60:.1f} min")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
