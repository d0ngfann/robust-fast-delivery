"""
main.py  -  Main Entry Point (Optimal Policy)

Runs C&CG for the Optimal policy (free mode selection).
Supports single gamma run or full sensitivity analysis.

Depends on: config.py, data_gen.py, algo.py (CCGAlgorithm)
Counterpart: main_fixed_mode.py (for FM0/FM1/FM2 policies)

Usage:
  python main.py full 10 --seed 1 --di HD                          # Basic run
  python main.py full 10 --seed 1 --di LD --tc2 0.60               # Breakeven analysis
  python main.py full 10 --seed 1 --di HD --di-func linear         # Linear DI robustness check
  python main.py full 10 --seed 1 --di HD --coverage moderate      # Coverage constraint
  python main.py full --seed 5 --di HD                             # Gamma sensitivity analysis
"""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from config import ProblemConfig, SensitivityConfig
from data_gen import SupplyChainData, generate_supply_chain_data
from algo import CCGAlgorithm, print_solution_summary


def run_single_gamma(data, config, gamma, coverage_thresholds=None):
    """
    Run C&CG algorithm for a single Gamma value.

    Args:
        data: SupplyChainData instance
        config: ProblemConfig instance
        gamma: Uncertainty budget value
        coverage_thresholds: dict {m: max_distance} or None for no coverage limit

    Returns:
        dict: Results from C&CG algorithm (with added facility information)
    """
    print("\n" + "=" * 80)
    print(f"RUNNING C&CG FOR Gamma = {gamma}")
    print("=" * 80)

    # Set gamma
    config.set_gamma(gamma)

    # Record start time
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create C&CG algorithm
    ccg = CCGAlgorithm(data, config)

    # Apply coverage constraint: must be after initialize() creates the master problem
    # but before the C&CG loop starts solving
    if coverage_thresholds is not None:
        ccg.initialize()
        ccg.master.apply_coverage_constraint(coverage_thresholds)
        results = ccg.run(skip_init=True)
    else:
        results = ccg.run()

    # Record end time
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Add timing information
    results['start_time'] = start_time_str
    results['end_time'] = end_time_str

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
        results['opened_plants'] = opened_plants_by_product  # For backward compatibility
        results['opened_dcs'] = opened_dcs
        results['num_plants_opened'] = sum(len(v) for v in opened_plants_by_product.values())
        results['num_dcs_opened'] = len(opened_dcs)

        print_solution_summary(results['optimal_solution'], data)
    else:
        results['opened_plants'] = []
        results['opened_dcs'] = []
        results['num_plants_opened'] = 0
        results['num_dcs_opened'] = 0

    return results


def run_sensitivity_analysis(instance_type='toy', gamma_values=None):
    """
    Run sensitivity analysis for different Gamma values.

    Args:
        instance_type: 'toy' or 'full'
        gamma_values: List of gamma values to test (if None, uses default range)

    Returns:
        pd.DataFrame: Results summary
    """
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS - {instance_type.upper()} INSTANCE")
    print("=" * 80)

    # Initialize configuration
    config = ProblemConfig(instance_type=instance_type)

    # Load or generate data
    if os.path.exists(config.data_file):
        print(f"Loading data from {config.data_file}...")
        data = SupplyChainData.load(config.data_file)
        data.summary()
    else:
        print(f"Data file not found. Generating new data...")
        data = generate_supply_chain_data(config, seed=42)
        data.save(config.data_file)
        data.summary()

    # Determine gamma values to test
    if gamma_values is None:
        sens_config = SensitivityConfig(config.R)
        gamma_values = sens_config.gamma_values

    print(f"\nTesting Gamma values: {gamma_values}")

    # Results storage
    results_list = []

    # Run for each gamma
    for gamma in gamma_values:
        try:
            results = run_single_gamma(data, config, gamma)

            # Extract key metrics
            row = {
                'Gamma': gamma,
                'Converged': results['converged'],
                'Iterations': results['iterations'],
                'Start_Time': results.get('start_time', ''),
                'End_Time': results.get('end_time', ''),
                'Total_Time': results['total_time'],
                'Optimal_Value': results['optimal_value'] if results['optimal_value'] is not None else float('nan'),
                'LB': results['LB'],
                'UB': results['UB'],
                'Gap': results['gap'],
                'Num_Scenarios': results['num_scenarios'],
                'Num_Plants_Opened': results.get('num_plants_opened', 0),
                'Num_DCs_Opened': results.get('num_dcs_opened', 0),
                'Opened_Plants': str(results.get('opened_plants', [])),
                'Opened_DCs': str(results.get('opened_dcs', []))
            }

            results_list.append(row)

            # Save intermediate results
            df_temp = pd.DataFrame(results_list)
            os.makedirs(config.results_dir, exist_ok=True)
            temp_file = os.path.join(config.results_dir, f"DH_sensitivity_{instance_type}_temp.csv")
            df_temp.to_csv(temp_file, index=False)
            print(f"\nIntermediate results saved to {temp_file}")

        except Exception as e:
            print(f"\nERROR: Failed to run Gamma = {gamma}")
            print(f"Exception: {str(e)}")
            import traceback
            traceback.print_exc()

            # Log failed run
            row = {
                'Gamma': gamma,
                'Converged': False,
                'Iterations': 0,
                'Start_Time': '',
                'End_Time': '',
                'Total_Time': 0,
                'Optimal_Value': float('nan'),
                'LB': float('nan'),
                'UB': float('nan'),
                'Gap': float('nan'),
                'Num_Scenarios': 0,
                'Num_Plants_Opened': 0,
                'Num_DCs_Opened': 0,
                'Opened_Plants': '[]',
                'Opened_DCs': '[]'
            }
            results_list.append(row)

    # Create results DataFrame
    df_results = pd.DataFrame(results_list)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config.results_dir, f"DH_sensitivity_{instance_type}_{timestamp}.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")

    return df_results


def plot_sensitivity_results(df_results, instance_type='toy'):
    """
    Create plots for sensitivity analysis results.

    Args:
        df_results: DataFrame with results
        instance_type: 'toy' or 'full'
    """
    try:
        import matplotlib.pyplot as plt

        # Filter out failed runs
        df_valid = df_results[df_results['Converged'] == True].copy()

        if len(df_valid) == 0:
            print("No valid results to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Optimal Value vs Gamma
        ax1 = axes[0, 0]
        ax1.plot(df_valid['Gamma'], df_valid['Optimal_Value'], marker='o', linewidth=2)
        ax1.set_xlabel('Uncertainty Budget (Gamma)', fontsize=12)
        ax1.set_ylabel('Optimal Objective Value', fontsize=12)
        ax1.set_title('Robust Profit vs Uncertainty Budget', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Number of Scenarios vs Gamma
        ax2 = axes[0, 1]
        ax2.plot(df_valid['Gamma'], df_valid['Num_Scenarios'], marker='s', linewidth=2, color='green')
        ax2.set_xlabel('Uncertainty Budget (Gamma)', fontsize=12)
        ax2.set_ylabel('Number of Critical Scenarios', fontsize=12)
        ax2.set_title('Critical Scenarios vs Uncertainty Budget', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Iterations vs Gamma
        ax3 = axes[1, 0]
        ax3.plot(df_valid['Gamma'], df_valid['Iterations'], marker='^', linewidth=2, color='orange')
        ax3.set_xlabel('Uncertainty Budget (Gamma)', fontsize=12)
        ax3.set_ylabel('Number of Iterations', fontsize=12)
        ax3.set_title('Convergence Iterations vs Uncertainty Budget', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Total Time vs Gamma
        ax4 = axes[1, 1]
        ax4.plot(df_valid['Gamma'], df_valid['Total_Time'], marker='d', linewidth=2, color='red')
        ax4.set_xlabel('Uncertainty Budget (Gamma)', fontsize=12)
        ax4.set_ylabel('Total Time (seconds)', fontsize=12)
        ax4.set_title('Computation Time vs Uncertainty Budget', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(config.results_dir, f"DH_sensitivity_{instance_type}_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")

        plt.close()

    except ImportError:
        print("Matplotlib not available. Skipping plots.")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")


def main():
    """Main execution function."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run C&CG Algorithm for Robust Supply Chain Optimization')
    parser.add_argument('instance_type', nargs='?', default='toy', choices=['toy', 'full', 'full200'],
                        help='Instance type: toy, full, or full200 (default: toy)')
    parser.add_argument('gamma', nargs='?', type=int, default=None,
                        help='Single Gamma value to run (optional, runs sensitivity analysis if not specified)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Dataset seed number (1-50 for full/full200, 1-5 for toy). If not specified, uses default data.')
    parser.add_argument('--di', '--di_scenario', dest='di_scenario', default='HD',
                        choices=['HD', 'MD', 'LD', 'Mixed'],
                        help='DI scenario to use: HD (High), MD (Medium), LD (Low), or Mixed (default: HD)')
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
    single_gamma = args.gamma
    seed = args.seed
    di_scenario = args.di_scenario
    result_dir = args.result_dir

    print("\n" + "=" * 80)
    print("ROBUST SUPPLY CHAIN OPTIMIZATION - C&CG ALGORITHM")
    print("=" * 80)
    print(f"Instance Type: {instance_type.upper()}")
    if seed:
        print(f"Dataset Seed: {seed}")
    print(f"DI Scenario: {di_scenario}")
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
            # Linear DI: recalculate from stored k_params using linear function
            # DI_m = 1 + 0.3 * kappa * m (anchored to same endpoint as exponential at m=2, kappa=1 -> 1.60)
            from data_gen import generate_DI_vector_linear
            print(f"  Using LINEAR DI function (robustness check)")
            for k_idx in range(config.K):
                k_param = data.DI_k_params[di_scenario][k_idx]
                di_vec = generate_DI_vector_linear(k_param, config.M)
                for m in range(config.M):
                    data.DI[(m, k_idx)] = di_vec[m]
        else:
            # Default: use exponential DI from data (already generated with sqrt(1.6) base)
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

    data.summary()

    # Run either single gamma or sensitivity analysis
    start_time = time.time()

    if single_gamma is not None:
        # Run single gamma value
        print(f"\nTesting Gamma value: {single_gamma}")

        results = run_single_gamma(data, config, single_gamma, coverage_thresholds)

        # Print results
        print("\n" + "=" * 80)
        print("SINGLE RUN RESULTS")
        print("=" * 80)
        print(f"Gamma: {single_gamma}")
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

        # Save single run result
        seed_str = f"_seed{seed}" if seed else ""
        # Build filename with all experiment conditions
        parts = [f"optimal_{instance_type}{seed_str}_{di_scenario}_gamma{single_gamma}"]
        if args.di_func != 'exponential':
            parts.append(f"difunc_{args.di_func}")
        if args.tc2 is not None:
            parts.append(f"tc2_{args.tc2:.2f}")
        if args.coverage != 'no_limit':
            parts.append(f"cov_{args.coverage}")
        output_file = os.path.join(result_dir, "_".join(parts) + ".csv")
        os.makedirs(result_dir, exist_ok=True)
        df_single = pd.DataFrame([{
            'Gamma': single_gamma,
            'Seed': seed if seed else 'default',
            'DI_Scenario': di_scenario,
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
        df_single.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    else:
        # Run sensitivity analysis
        df_results = run_sensitivity_analysis(instance_type=instance_type)

        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(df_results.to_string(index=False))
        print("=" * 80)

        # Create plots
        plot_sensitivity_results(df_results, instance_type=instance_type)

    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
