"""
run_exp2.py  -  Experiment 2: R=200, Gamma sensitivity, 4 policies x 4 DI x 50 seeds x 5 Gamma = 4,000 runs

Gamma sensitivity at larger scale with new DI base and epsilon=100.
Results saved to ../result/exp2/
Estimated time: ~15 hours (4 parallel workers) or ~59 hours (sequential)

Uses multiprocessing for parallel execution. Adjust WORKERS based on your CPU.
"""

import subprocess
import sys
import os
import time
from itertools import product
from multiprocessing import Pool, cpu_count

SEEDS = range(1, 51)
DI_SCENARIOS = ['HD', 'MD', 'LD', 'Mixed']
GAMMAS = [20, 40, 60, 80, 100]
INSTANCE = 'full200'
RESULT_DIR = os.path.join('..', 'result', 'exp2')
WORKERS = min(4, cpu_count() - 1)  # Leave 1 core free; adjust if needed

def run_one(args):
    """Run a single experiment. Args: (cmd_list, label_string)."""
    cmd, label = args
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    success = result.returncode == 0
    status = "OK" if success else "FAIL"
    return label, success, elapsed, status

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Build all jobs
    jobs = []

    # Optimal policy
    for gamma, seed, di in product(GAMMAS, SEEDS, DI_SCENARIOS):
        cmd = [sys.executable, 'main.py', INSTANCE, str(gamma),
               '--seed', str(seed), '--di', di, '--result_dir', RESULT_DIR]
        jobs.append((cmd, f"Optimal G={gamma} seed={seed} DI={di}"))

    # Fixed mode policies
    for mode, gamma, seed, di in product([0, 1, 2], GAMMAS, SEEDS, DI_SCENARIOS):
        cmd = [sys.executable, 'main_fixed_mode.py', INSTANCE, str(gamma),
               '--seed', str(seed), '--di', di, '--mode', str(mode), '--result_dir', RESULT_DIR]
        jobs.append((cmd, f"FM{mode} G={gamma} seed={seed} DI={di}"))

    total = len(jobs)
    print(f"=" * 60)
    print(f"EXPERIMENT 2: {total} runs ({WORKERS} parallel workers)")
    print(f"  Instance: {INSTANCE}, Gammas: {GAMMAS}")
    print(f"  Results: {os.path.abspath(RESULT_DIR)}")
    print(f"=" * 60)

    start_all = time.time()
    completed = 0
    failed = 0

    with Pool(WORKERS) as pool:
        for label, success, elapsed, status in pool.imap_unordered(run_one, jobs):
            completed += 1
            if not success:
                failed += 1
            if completed % 100 == 0 or not success:
                wall = time.time() - start_all
                avg = wall / completed
                eta = (total - completed) * avg
                print(f"[{completed}/{total}] {status} {label} ({elapsed:.0f}s) "
                      f"| ETA {eta/3600:.1f}h")

    wall_time = time.time() - start_all
    print(f"\n{'=' * 60}")
    print(f"DONE: {completed - failed}/{total} succeeded, {failed} failed")
    print(f"Wall time: {wall_time/3600:.1f} hours")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
