"""
main_fixed_mode.py  -  Main Entry Point (Fixed Mode Policies: FM0/FM1/FM2)

Runs C&CG with a fixed transportation mode for all customers.
Used to compute FM0, FM1, FM2 baselines for Value of Flexibility analysis.

Depends on: config.py, data_gen.py, algo_fixed_mode.py (CCGAlgorithmFixedMode)
Counterpart: main.py (for Optimal policy)

Usage:
  python main_fixed_mode.py full 10 --seed 1 --di HD --mode 0      # FM0 (slow)
  python main_fixed_mode.py full 10 --seed 1 --di HD --mode 1      # FM1 (medium)
  python main_fixed_mode.py full 10 --seed 1 --di HD --mode 2      # FM2 (fast)
  python main_fixed_mode.py full 10 --seed 1 --di LD --mode 2 --tc2 0.60   # Breakeven
  python main_fixed_mode.py full 10 --seed 1 --di HD --mode 2 --coverage moderate  # Coverage
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime

from config import ProblemConfig
from data_gen import SupplyChainData, generate_supply_chain_data
from algo_fixed_mode import CCGAlgorithmFixedMode, print_solution_summary_fixed_mode


def run_single_gamma_fixed_mode(data, config, gamma, fixed_mode, coverage_thresholds=None):
    """
    Run C&CG algorithm for a single Gamma value with fixed transportation mode.

    Args:
        data: SupplyChainData instance
        config: ProblemConfig instance
        gamma: Uncertainty budget value
        fixed_mode: Transportation mode to force (0, 1, or 2)
        coverage_thresholds: dict {m: max_distance} or None for no coverage limit

    Returns:
        dict: Results from C&CG algorithm (with added facility information)
    """
    print("\n" + "=" * 80)
    print(f"RUNNING C&CG FOR Gamma = {gamma} (FIXED MODE = {fixed_mode})")
    print("=" * 80)

    # Set gamma
    config.set_gamma(gamma)

    # Record start time
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create C&CG algorithm with FIXED MODE
    ccg = CCGAlgorithmFixedMode(data, config, fixed_mode)

    if coverage_thresholds is not None:
        ccg.initialize()
        ccg.master.apply_coverage_constraint(coverage_thresholds)
        results = ccg.run(skip_init=True)
    else:
        results = ccg.run()

    # Record end time
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Add timing and mode information
    results['start_time'] = start_time_str
    results['end_time'] = end_time_str
    results['fixed_mode'] = fixed_mode

    # Extract opened facilities information
    if results['optimal_solution'] is not None:
        solution = results['optimal_solution']
        # Product-specific plants
        opened_plants_by_product = {}
        for k in range(data.K):
            opened_plants_k = [i for i in range(data.I) if solution['x'][(k, i)] > 0.5]
            opened_plants_by_product[k] = opened_plants_k

        opened_dcs = [j for j in range(data.J) if solution['y'][j] > 0.5]

        results['opened_plants_by_product'] = opened_plants_by_product
        results['opened_plants'] = opened_plants_by_product
        results['opened_dcs'] = opened_dcs
        results['num_plants_opened'] = sum(len(v) for v in opened_plants_by_product.values())
        results['num_dcs_opened'] = len(opened_dcs)

        print_solution_summary_fixed_mode(results['optimal_solution'], data, fixed_mode)
    else:
        results['opened_plants'] = []
        results['opened_dcs'] = []
        results['num_plants_opened'] = 0
        results['num_dcs_opened'] = 0

    return results


def main():
    """Main execution function."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run C&CG Algorithm with FIXED Transportation Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_fixed_mode.py full 10 --seed 1 --di HD --mode 0   # Slow mode
  python main_fixed_mode.py full 10 --seed 1 --di HD --mode 1   # Medium mode
  python main_fixed_mode.py full 10 --seed 1 --di HD --mode 2   # Fast mode

Mode descriptions:
  Mode 0: Slowest delivery, lowest cost, lowest demand impact (DI)
  Mode 1: Medium speed, medium cost, medium demand impact (DI)
  Mode 2: Fastest delivery, highest cost, highest demand impact (DI)
        """
    )
    parser.add_argument('instance_type', choices=['toy', 'full', 'full200'],
                        help='Instance type: toy, full, or full200')
    parser.add_argument('gamma', type=int,
                        help='Gamma value (uncertainty budget)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Dataset seed number (1-50 for full/full200, 1-5 for toy)')
    parser.add_argument('--di', '--di_scenario', dest='di_scenario', default='HD',
                        choices=['HD', 'MD', 'LD', 'Mixed'],
                        help='DI scenario: HD, MD, LD, or Mixed (default: HD)')
    parser.add_argument('--mode', type=int, required=True, choices=[0, 1, 2],
                        help='REQUIRED: Fixed transportation mode (0, 1, or 2)')
    parser.add_argument('--result_dir', type=str, default='../result',
                        help='Directory to save results (default: ../result)')
    parser.add_argument('--tc2', type=float, default=None,
                        help='Override TC_2 (fast mode transport cost) for breakeven analysis')
    parser.add_argument('--di-func', dest='di_func', default='exponential',
                        choices=['exponential', 'linear'],
                        help='DI function form: exponential (default) or linear')
    parser.add_argument('--coverage', type=str, default='no_limit',
                        choices=['tight', 'moderate', 'relaxed', 'no_limit'],
                        help='Service coverage scenario (default: no_limit)')

    args = parser.parse_args()

    instance_type = args.instance_type
    gamma = args.gamma
    seed = args.seed
    di_scenario = args.di_scenario
    fixed_mode = args.mode
    result_dir = args.result_dir

    print("\n" + "=" * 80)
    print("ROBUST SUPPLY CHAIN OPTIMIZATION - C&CG ALGORITHM (FIXED MODE)")
    print("=" * 80)
    print(f"Instance Type: {instance_type.upper()}")
    print(f"Gamma: {gamma}")
    if seed:
        print(f"Dataset Seed: {seed}")
    print(f"DI Scenario: {di_scenario}")
    print(f"FIXED MODE: {fixed_mode}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Initialize configuration and load data
    config = ProblemConfig(instance_type=instance_type)

    # Load data based on seed
    if seed is not None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', f"DH_data_{instance_type}_seed{seed}.pkl")
        if not os.path.exists(data_file):
            print(f"Error: Data file not found: {data_file}")
            print(f"Available seeds for {instance_type}: ", end="")
            import glob
            available = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', f"DH_data_{instance_type}_seed*.pkl"))
            if available:
                seeds = [int(f.split('seed')[1].split('.pkl')[0]) for f in available]
                print(f"{min(seeds)}-{max(seeds)}")
            else:
                print("None")
            sys.exit(1)
        print(f"Loading data from {data_file}...")
        data = SupplyChainData.load(data_file)
    else:
        # Use default data file
        if os.path.exists(config.data_file):
            print(f"Loading data from {config.data_file}...")
            data = SupplyChainData.load(config.data_file)
        else:
            print(f"Data file not found. Generating new data...")
            data = generate_supply_chain_data(config, seed=42)
            data.save(config.data_file)

    # Apply selected DI scenario
    if hasattr(data, 'DI_scenarios') and di_scenario in data.DI_scenarios:
        print(f"\nApplying DI scenario: {di_scenario}")

        if args.di_func == 'linear':
            from data_gen import generate_DI_vector_linear
            print(f"  Using LINEAR DI function (robustness check)")
            for k_idx in range(config.K):
                k_param = data.DI_k_params[di_scenario][k_idx]
                di_vec = generate_DI_vector_linear(k_param, config.M)
                for m in range(config.M):
                    data.DI[(m, k_idx)] = di_vec[m]
        else:
            DI_matrix = data.DI_scenarios[di_scenario]
            for m in range(config.M):
                for k in range(config.K):
                    data.DI[(m, k)] = DI_matrix[k][m]

        print(f"DI values ({args.di_func}):")
        for k in range(config.K):
            k_param = data.DI_k_params[di_scenario][k]
            print(f"  Product {k}: {[f'{data.DI[(m, k)]:.3f}' for m in range(config.M)]} (k={k_param:.3f})")
    else:
        print(f"\nWarning: DI scenario '{di_scenario}' not found in data. Using default DI values.")

    # Override TC_2 if specified (for breakeven analysis)
    if args.tc2 is not None:
        old_tc2 = data.TC[2]
        data.TC[2] = args.tc2
        print(f"\n[BREAKEVEN] TC_2 overridden: {old_tc2} -> {args.tc2}")

    # Compute coverage thresholds (if specified)
    coverage_thresholds = None
    if args.coverage != 'no_limit':
        from data_gen import compute_coverage_thresholds
        coverage_thresholds = compute_coverage_thresholds(data, args.coverage)
        print(f"\n[COVERAGE] Scenario: {args.coverage}")
        for m in range(config.M):
            d_bar = coverage_thresholds[m]
            d_str = f"{d_bar:.1f}" if d_bar != float('inf') else "unlimited"
            print(f"  Mode {m}: max distance = {d_str}")

    # Show mode information
    print(f"\nTransportation Mode Information:")
    print(f"  Mode costs (TC): {[data.TC[m] for m in range(data.M)]}")
    print(f"  FIXED MODE {fixed_mode}: TC={data.TC[fixed_mode]:.2f}")
    print(f"  DI for fixed mode {fixed_mode}: {[data.DI[(fixed_mode, k)] for k in range(data.K)]}")

    data.summary()

    # Run with fixed mode
    start_time = time.time()

    results = run_single_gamma_fixed_mode(data, config, gamma, fixed_mode, coverage_thresholds)

    # Print results
    print("\n" + "=" * 80)
    print(f"SINGLE RUN RESULTS (FIXED MODE = {fixed_mode})")
    print("=" * 80)
    print(f"Gamma: {gamma}")
    print(f"Fixed Mode: {fixed_mode}")
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")
    print(f"Start Time: {results.get('start_time', 'N/A')}")
    print(f"End Time: {results.get('end_time', 'N/A')}")
    print(f"Total Time: {results['total_time']:.2f}s")

    # Handle None values for non-converged cases
    opt_val_str = f"{results['optimal_value']:.2f}" if results['optimal_value'] is not None else "N/A"
    lb_str = f"{results['LB']:.2f}" if results['LB'] != -float('inf') else "-inf"
    ub_str = f"{results['UB']:.2f}" if results['UB'] != float('inf') else "inf"
    gap_str = f"{results['gap']:.6f}" if results['gap'] != float('inf') else "inf"

    print(f"Optimal Value: {opt_val_str}")
    print(f"Lower Bound: {lb_str}")
    print(f"Upper Bound: {ub_str}")
    print(f"Gap: {gap_str}")
    print(f"Plants Opened: {results.get('opened_plants', [])} ({results.get('num_plants_opened', 0)} plants)")
    print(f"DCs Opened: {results.get('opened_dcs', [])} ({results.get('num_dcs_opened', 0)} DCs)")
    print("=" * 80)

    # Save result
    seed_str = f"_seed{seed}" if seed else ""
    # Build filename with all experiment conditions
    parts = [f"fm{fixed_mode}_{instance_type}{seed_str}_{di_scenario}_gamma{gamma}"]
    if args.di_func != 'exponential':
        parts.append(f"difunc_{args.di_func}")
    if args.tc2 is not None:
        parts.append(f"tc2_{args.tc2:.2f}")
    if args.coverage != 'no_limit':
        parts.append(f"cov_{args.coverage}")
    output_file = os.path.join(result_dir, "_".join(parts) + ".csv")
    os.makedirs(result_dir, exist_ok=True)

    df_result = pd.DataFrame([{
        'Gamma': gamma,
        'Seed': seed if seed else 'default',
        'DI_Scenario': di_scenario,
        'Fixed_Mode': fixed_mode,
        'Converged': results['converged'],
        'Iterations': results['iterations'],
        'Start_Time': results.get('start_time', ''),
        'End_Time': results.get('end_time', ''),
        'Total_Time': results['total_time'],
        'Optimal_Value': results['optimal_value'],
        'LB': results['LB'],
        'UB': results['UB'],
        'Gap': results['gap'],
        'Num_Scenarios': len(results['critical_scenarios']) if results['critical_scenarios'] else 0,
        'Num_Plants_Opened': results.get('num_plants_opened', 0),
        'Num_DCs_Opened': results.get('num_dcs_opened', 0),
        'Opened_Plants': str(results.get('opened_plants', [])),
        'Opened_DCs': str(results.get('opened_dcs', []))
    }])
    df_result.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
