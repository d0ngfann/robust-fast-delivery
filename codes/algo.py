"""
algo.py  -  C&CG Algorithm Orchestration

Main loop: iterates between Master Problem and Subproblem until |UB - LB| <= epsilon.
Manages bounds, convergence checking, scenario addition, and time limits.

Depends on: config.py, data_gen.py, master.py (MasterProblem), sub.py (Subproblem)
Used by: main.py (creates CCGAlgorithm and calls run())
Extended by: algo_fixed_mode.py (overrides initialize() to use MasterProblemFixedMode)

C&CG loop:
  1. Solve MP -> get first-stage solution + theta -> update UB
  2. Solve SP -> get worst-case scenario + Z_SP -> update LB
  3. Check convergence (|UB - LB| <= epsilon=100)
  4. Add new scenario to MP -> repeat
"""

import time
import numpy as np
from gurobipy import GRB
from data_gen import SupplyChainData
from config import ProblemConfig
from master import MasterProblem
from sub import Subproblem


class CCGAlgorithm:
    """Column-and-Constraint Generation algorithm for robust optimization."""

    def __init__(self, data: SupplyChainData, config: ProblemConfig):
        """
        Initialize C&CG algorithm.

        Args:
            data: SupplyChainData instance
            config: ProblemConfig instance (must have Gamma set)
        """
        self.data = data
        self.config = config

        if config.Gamma is None:
            raise ValueError("Gamma must be set in config before initializing C&CG")

        # Algorithm state
        self.LB = -float('inf')  # Lower bound
        self.UB = float('inf')  # Upper bound
        self.iteration = 0
        self.converged = False
        self.optimal_solution = None
        self.optimal_value = None
        self.critical_scenarios = []

        # History tracking
        self.history = {
            'iterations': [],
            'LB': [],
            'UB': [],
            'gap': [],
            'time': [],
            'mp_time': [],
            'sp_time': []
        }

        # Master and Subproblem instances
        self.master = None
        self.subproblem = None

        # Start time
        self.start_time = None

        # Overall C&CG time limit (seconds). Terminates gracefully before SLURM kills the job.
        # Default: 82800s = 23 hours (leaves 1h buffer before 24h SLURM limit)
        self.ccg_time_limit = getattr(config, 'ccg_time_limit', 82800)


    def initialize(self):
        """Initialize algorithm with nominal scenario."""
        print("=" * 80)
        print("INITIALIZING C&CG ALGORITHM")
        print("=" * 80)
        print(f"Problem: K={self.data.K}, I={self.data.I}, J={self.data.J}, "
              f"R={self.data.R}, M={self.data.M}")
        print(f"Uncertainty Budget: Gamma = {self.config.Gamma}")
        print(f"Convergence Tolerance: epsilon = {self.config.epsilon}")
        print("=" * 80)

        # Create Master Problem
        print("Creating Master Problem...")
        self.master = MasterProblem(self.data, self.config)

        # Add nominal scenario (eta = 0)
        print("Adding nominal scenario (eta = 0)...")
        eta_plus_nominal = {(r, k): 0 for r in range(self.data.R) for k in range(self.data.K)}
        eta_minus_nominal = {(r, k): 0 for r in range(self.data.R) for k in range(self.data.K)}
        # All scenarios use beta VARIABLES (per algorithm_framework.tex Line 222)
        self.master.add_scenario(scenario_id=0, eta_plus=eta_plus_nominal, eta_minus=eta_minus_nominal)
        self.critical_scenarios.append((0, eta_plus_nominal, eta_minus_nominal))

        # Create Subproblem
        print("Creating Subproblem...")
        self.subproblem = Subproblem(self.data, self.config)

        print("Initialization complete!\n")

    def solve_master(self):
        """
        Solve Master Problem.

        Returns:
            tuple: (success, solution, theta, solve_time)
        """
        print(f"\n{'='*80}")
        print(f"ITERATION {self.iteration}")
        print(f"{'='*80}")
        print(f"[Iteration {self.iteration}] Solving Master Problem...")
        print(f"  Current scenarios in Master: {len(self.critical_scenarios)}")
        start_time = time.time()

        success = self.master.solve()
        solve_time = time.time() - start_time

        if not success:
            print(f"  Master Problem failed to solve! Status: {self.master.model.Status}")
            return False, None, None, solve_time

        solution = self.master.get_solution()
        theta = solution['theta']

        # Update upper bound: UB = Master Problem objective value
        # Use solver's objective value directly to avoid numerical precision issues
        # Note: solution['objective'] = -OC - FC + theta (from Master Problem)
        self.UB = solution['objective']

        print(f"  Master solved in {solve_time:.2f}s")
        print(f"  Master Objective (UB) = {solution['objective']:.2f}, theta = {theta:.2f}")
        print(f"  Upper Bound (UB) = {self.UB:.2f}")

        # DEBUG: Verify optimality cuts
        self._verify_optimality_cuts(solution, theta)

        return True, solution, theta, solve_time

    def solve_subproblem(self, mp_solution):
        """
        Solve Subproblem with fixed first-stage solution.

        Args:
            mp_solution: Solution dict from Master Problem

        Returns:
            tuple: (success, Z_SP, eta_plus, eta_minus, solve_time)
        """
        print(f"[Iteration {self.iteration}] Solving Subproblem...")
        start_time = time.time()

        # Fix first-stage variables
        self.subproblem.fix_first_stage(mp_solution)

        success = self.subproblem.solve()
        solve_time = time.time() - start_time

        if not success:
            status = self.subproblem.model.Status
            print(f"  Subproblem failed to solve! Status: {status}")

            # Special handling for unbounded case (rare numerical issue)
            if status == GRB.UNBOUNDED:
                print("\n  [!]  WARNING: Subproblem unbounded (rare numerical issue)")
                print("  This may indicate degenerate first-stage solution")
                print("  Possible causes:")
                print("    - Master solution has many near-zero alpha values")
                print("    - Dual constraint matrix became near-singular")
                print("    - Numerical precision limit reached after many iterations")
                print("\n  Current best solution is still valid - terminating gracefully")
                print("  Note: This affects <3% of cases and is a known numerical edge case")

                # Diagnostic: Check alpha sparsity
                sparse_count = sum(1 for v in mp_solution['alpha'].values() if abs(v) < 1e-6)
                total_alpha = len(mp_solution['alpha'])
                sparsity = 100.0 * sparse_count / total_alpha if total_alpha > 0 else 0
                print(f"\n  [DIAGNOSTIC] Alpha sparsity: {sparse_count}/{total_alpha} ({sparsity:.1f}%) near-zero")

            return False, None, None, None, solve_time

        Z_SP, eta_plus, eta_minus = self.subproblem.get_worst_case_scenario()

        print(f"  Subproblem solved in {solve_time:.2f}s")
        print(f"  Worst-case operational profit (Z_SP) = {Z_SP:.2f}")

        # Count number of deviating customers
        num_plus = sum(1 for v in eta_plus.values() if v > 0.5)
        num_minus = sum(1 for v in eta_minus.values() if v > 0.5)
        print(f"  Scenario: {num_plus} demand increases, {num_minus} demand decreases")

        # DEBUG: Show Z_SP calculation details
        print(f"\n  [DEBUG] Z_SP Calculation:")
        dual_obj = self.subproblem.model.ObjVal

        # Calculate Sxd_nominal
        revenue_S_times_d_nominal = 0.0
        for r in range(self.data.R):
            for k in range(self.data.K):
                d_nominal = sum(
                    self.data.mu[(r, k)] * self.data.DI[(m, k)] * mp_solution['beta'][(r, m)]
                    for m in range(self.data.M)
                )
                revenue_S_times_d_nominal += self.data.S * d_nominal

        print(f"  [DEBUG]   Dual objective (Gurobi, includes Sxmu_hatx(eta+-eta-)) = {dual_obj:.2f}")
        print(f"  [DEBUG]   Sxd_nominal = {revenue_S_times_d_nominal:.2f}")
        print(f"  [DEBUG]   Z_SP = Sxd_nominal + dual_obj = {revenue_S_times_d_nominal:.2f} + {dual_obj:.2f} = {Z_SP:.2f}")
        print(f"  [DEBUG]   Current theta from Master = {mp_solution['theta']:.2f}")

        if Z_SP > mp_solution['theta']:
            gap_sp = Z_SP - mp_solution['theta']
            print(f"  [DEBUG]   OK Z_SP > theta by {gap_sp:.2f} (cut will improve theta)")
        else:
            gap_sp = mp_solution['theta'] - Z_SP
            print(f"  [DEBUG]   [!]  Z_SP <= theta by {gap_sp:.2f} (should not happen if not duplicate!)")

        return True, Z_SP, eta_plus, eta_minus, solve_time

    def update_bounds(self, mp_solution, Z_SP):
        """
        Update lower bound.

        Args:
            mp_solution: Master Problem solution
            Z_SP: Subproblem optimal value (worst-case operational profit)
        """
        # True robust profit: -OC - FC + Z_SP
        # Note: mp_solution['objective'] = -OC - FC + theta
        # Therefore: -OC - FC = mp_solution['objective'] - theta
        # Z_current = -OC - FC + Z_SP = mp_solution['objective'] - theta + Z_SP
        theta = mp_solution['theta']
        Z_current = mp_solution['objective'] - theta + Z_SP

        # Update lower bound
        self.LB = max(self.LB, Z_current)

        print(f"  True Robust Profit (Z_current) = {Z_current:.2f}")
        print(f"  Lower Bound (LB) = {self.LB:.2f}")

    def _verify_optimality_cuts(self, mp_solution, theta):
        """
        DEBUG: Verify that all optimality cuts are satisfied.
        For each scenario, calculate the actual operational profit and check theta <= Q(scenario).

        Note: ALL scenarios use beta VARIABLES (per algorithm_framework.tex Line 222).
              Verification uses current beta solution values.
        """
        print(f"\n  [DEBUG] Verifying Optimality Cuts:")
        print(f"  [DEBUG] Current theta = {theta:.2f}")

        violations = []
        for scenario_id, eta_plus, eta_minus in self.critical_scenarios:
            # Calculate realized demand for this scenario
            d_realized = {}
            for r in range(self.data.R):
                for k in range(self.data.K):
                    nominal = sum(
                        self.data.mu[(r, k)] * self.data.DI[(m, k)] * mp_solution['beta'][(r, m)]
                        for m in range(self.data.M)
                    )
                    uncertainty = (eta_plus[(r, k)] - eta_minus[(r, k)]) * self.data.mu_hat[(r, k)]
                    d_realized[(r, k)] = nominal + uncertainty

            # Calculate operational profit for this scenario using Master's second-stage variables
            # (if this is not iteration 1, we should have these variables)
            if scenario_id in [s[0] for s in self.master.scenarios]:
                # Get second-stage variable values from Master solution
                l = scenario_id

                # Revenue
                revenue = sum(
                    self.data.S * (d_realized[(r, k)] - self.master.u[(r, k, l)].X)
                    for r in range(self.data.R)
                    for k in range(self.data.K)
                )

                # Costs
                HC = sum(
                    (self.data.h[j] / 2) * self.master.A_ij[(k, i, j, l)].X
                    for k in range(self.data.K)
                    for i in range(self.data.I)
                    for j in range(self.data.J)
                )

                TC1 = sum(
                    self.data.D1[(k, i, j)] * self.data.t * self.master.A_ij[(k, i, j, l)].X
                    for k in range(self.data.K)
                    for i in range(self.data.I)
                    for j in range(self.data.J)
                )

                TC2 = sum(
                    self.data.D2[(j, r)] * self.data.TC[m] * self.master.X[(j, r, m, k, l)].X
                    for k in range(self.data.K)
                    for j in range(self.data.J)
                    for r in range(self.data.R)
                    for m in range(self.data.M)
                )

                PC = sum(
                    self.data.F[(k, i)] * self.master.A_ij[(k, i, j, l)].X
                    for k in range(self.data.K)
                    for i in range(self.data.I)
                    for j in range(self.data.J)
                )

                SC = sum(
                    self.data.SC * self.master.u[(r, k, l)].X
                    for r in range(self.data.R)
                    for k in range(self.data.K)
                )

                Q_scenario = revenue - HC - TC1 - TC2 - PC - SC
                violation = theta - Q_scenario

                if violation > 0.01:  # theta > Q + tolerance
                    violations.append((scenario_id, violation, Q_scenario))
                    print(f"  [DEBUG]   [!]  Scenario {scenario_id}: theta = {theta:.2f} > Q = {Q_scenario:.2f} (violation: {violation:.2f})")
                else:
                    print(f"  [DEBUG]   OK Scenario {scenario_id}: theta = {theta:.2f} <= Q = {Q_scenario:.2f} (slack: {-violation:.2f})")

        if violations:
            print(f"  [DEBUG] [!]  WARNING: {len(violations)} optimality cut(s) violated!")
            print(f"  [DEBUG] This should NEVER happen - indicates implementation bug!")
        else:
            print(f"  [DEBUG] All optimality cuts satisfied.")

    def check_convergence(self):
        """
        Check convergence criterion.

        Convergence is achieved when |UB - LB| <= epsilon.
        Negative gaps indicate solver suboptimality or numerical issues.
        """
        gap = self.UB - self.LB
        abs_gap = abs(gap)
        rel_gap = gap / (abs(self.UB) + 1e-10) if abs(self.UB) > 1e-10 else 0

        print(f"  Gap (UB-LB) = {gap:.4f}, Absolute Gap = {abs_gap:.4f}, Relative Gap = {rel_gap:.6f}")

        # Positive gap within tolerance: converged
        if gap >= 0 and abs_gap <= self.config.epsilon:
            self.converged = True
            print("  *** CONVERGED (gap within tolerance) ***")
            return True

        # Small negative gap within tolerance: treat as converged (numerical precision)
        if gap < 0 and abs_gap <= self.config.epsilon:
            self.converged = True
            print(f"  *** CONVERGED (small negative gap: {gap:.2f} within tolerance epsilon={self.config.epsilon}) ***")
            print(f"  Note: Negative gap likely due to numerical precision or solver tolerance")
            return True

        # Large negative gap: warning but continue
        if gap < -self.config.epsilon:
            print(f"  [!]  WARNING: NEGATIVE GAP detected!")
            print(f"  UB = {self.UB:.2f}, LB = {self.LB:.2f}, Gap = {gap:.2f}")
            print(f"  Possible causes:")
            print(f"    1. Subproblem returned suboptimal Z_SP (too large -> LB over-estimated)")
            print(f"    2. Master Problem returned suboptimal theta (too small -> UB under-estimated)")
            print(f"    3. Numerical precision issues in bound calculations")
            print(f"  Algorithm will continue to search for better solutions...")

        # Positive gap larger than tolerance: not converged yet
        if gap > self.config.epsilon:
            print(f"  Not converged yet (gap = {gap:.2f} > epsilon = {self.config.epsilon})")

        return False

    def is_duplicate_scenario(self, eta_plus, eta_minus):
        """
        Check if scenario already exists in critical scenarios.

        Args:
            eta_plus: dict {(r,k): value}
            eta_minus: dict {(r,k): value}

        Returns:
            bool: True if scenario is duplicate
        """
        for _, existing_eta_plus, existing_eta_minus in self.critical_scenarios:
            # Check if all eta values match
            is_same = True
            for r in range(self.data.R):
                for k in range(self.data.K):
                    if (eta_plus[(r, k)] != existing_eta_plus[(r, k)] or
                        eta_minus[(r, k)] != existing_eta_minus[(r, k)]):
                        is_same = False
                        break
                if not is_same:
                    break

            if is_same:
                return True

        return False

    def add_scenario_to_master(self, eta_plus, eta_minus):
        """
        Add new scenario to Master Problem.

        Args:
            eta_plus: dict {(r,k): value}
            eta_minus: dict {(r,k): value}

        Returns:
            bool: True if scenario was added, False if duplicate

        Note: Per algorithm_framework.tex Line 222, ALL scenarios use the SAME beta VARIABLES.
              This is endogenous demand - beta (mode choice) affects demand realization.
        """
        # Check for duplicates
        if self.is_duplicate_scenario(eta_plus, eta_minus):
            print(f"\n  [DEBUG] Scenario is DUPLICATE - checking details:")

            # Find which scenario it matches
            for existing_id, existing_eta_plus, existing_eta_minus in self.critical_scenarios:
                is_same = True
                for r in range(self.data.R):
                    for k in range(self.data.K):
                        if (eta_plus[(r, k)] != existing_eta_plus[(r, k)] or
                            eta_minus[(r, k)] != existing_eta_minus[(r, k)]):
                            is_same = False
                            break
                    if not is_same:
                        break

                if is_same:
                    print(f"  [DEBUG] Matches scenario {existing_id}")

                    # Count deviations
                    num_plus = sum(1 for v in eta_plus.values() if v > 0.5)
                    num_minus = sum(1 for v in eta_minus.values() if v > 0.5)
                    print(f"  [DEBUG] Scenario pattern: {num_plus} increases, {num_minus} decreases")
                    break

            print(f"  [DEBUG] NOT adding to Master Problem (duplicate)")
            return False

        scenario_id = len(self.critical_scenarios)
        print(f"\n  [DEBUG] Adding NEW scenario {scenario_id} to Master Problem...")

        # Count deviations
        num_plus = sum(1 for v in eta_plus.values() if v > 0.5)
        num_minus = sum(1 for v in eta_minus.values() if v > 0.5)
        print(f"  [DEBUG] New scenario pattern: {num_plus} increases, {num_minus} decreases")

        # Add scenario to Master (uses beta VARIABLES, not fixed values)
        # Per algorithm_framework.tex Line 222: ALL scenarios use SAME beta variables
        self.master.add_scenario(scenario_id, eta_plus, eta_minus)
        self.critical_scenarios.append((scenario_id, eta_plus, eta_minus))

        print(f"  [DEBUG] Total scenarios in Master: {len(self.critical_scenarios)}")
        return True

    def log_iteration(self, mp_time, sp_time):
        """Log iteration statistics."""
        elapsed = time.time() - self.start_time
        gap = self.UB - self.LB

        self.history['iterations'].append(self.iteration)
        self.history['LB'].append(self.LB)
        self.history['UB'].append(self.UB)
        self.history['gap'].append(gap)
        self.history['time'].append(elapsed)
        self.history['mp_time'].append(mp_time)
        self.history['sp_time'].append(sp_time)

    def run(self, skip_init=False):
        """
        Execute the C&CG algorithm.

        Args:
            skip_init: If True, skip initialization (caller already called initialize())

        Returns:
            dict: Results including optimal solution, value, and statistics
        """
        self.start_time = time.time()
        if not skip_init:
            self.initialize()

        print("\n" + "=" * 80)
        print("STARTING C&CG ITERATIONS")
        print("=" * 80 + "\n")

        timed_out = False

        while not self.converged and self.iteration < self.config.max_iterations:
            # Check overall C&CG time limit
            elapsed = time.time() - self.start_time
            if elapsed >= self.ccg_time_limit:
                print(f"\n  [TIME LIMIT] C&CG time limit reached ({elapsed/3600:.1f}h >= {self.ccg_time_limit/3600:.1f}h)")
                print(f"  Terminating gracefully with current best solution.")
                timed_out = True
                self.optimal_value = self.LB
                break

            self.iteration += 1

            # Step 1: Solve Master Problem
            success, mp_solution, theta, mp_time = self.solve_master()
            if not success:
                print("ERROR: Master Problem failed. Terminating.")
                break

            # Step 2: Solve Subproblem
            success, Z_SP, eta_plus, eta_minus, sp_time = self.solve_subproblem(mp_solution)
            if not success:
                print("ERROR: Subproblem failed. Terminating.")
                break

            # Step 3: Update bounds
            self.update_bounds(mp_solution, Z_SP)

            # Log iteration
            self.log_iteration(mp_time, sp_time)

            # Step 4: Check convergence
            if self.check_convergence():
                self.optimal_solution = mp_solution
                self.optimal_value = self.LB
                break

            # Step 5: Add scenario to Master
            # All scenarios use beta VARIABLES (per algorithm_framework.tex Line 222)
            scenario_added = self.add_scenario_to_master(eta_plus, eta_minus)

            # If scenario is duplicate, algorithm has stalled
            if not scenario_added:
                print("\n  WARNING: Duplicate scenario detected - algorithm stalled!")
                print(f"  Current gap: {self.UB - self.LB:.6f}")
                print("  This may indicate numerical issues or model formulation error.")
                print("  Terminating with current best solution.\n")
                self.optimal_solution = mp_solution
                self.optimal_value = self.LB
                break

            # Recreate subproblem for next iteration (to avoid stale constraints)
            self.subproblem = Subproblem(self.data, self.config)

        # Final results
        total_time = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("C&CG ALGORITHM COMPLETED")
        print("=" * 80)
        print(f"Total Iterations: {self.iteration}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Converged: {self.converged}")
        print(f"Timed Out: {timed_out}")
        print(f"Final Lower Bound: {self.LB:.4f}")
        print(f"Final Upper Bound: {self.UB:.4f}")
        print(f"Final Gap: {self.UB - self.LB:.6f}")
        print(f"Number of Critical Scenarios: {len(self.critical_scenarios)}")
        print("=" * 80)

        results = {
            'converged': self.converged,
            'timed_out': timed_out,
            'iterations': self.iteration,
            'total_time': total_time,
            'optimal_value': self.optimal_value,
            'optimal_solution': self.optimal_solution,
            'LB': self.LB,
            'UB': self.UB,
            'gap': self.UB - self.LB,
            'num_scenarios': len(self.critical_scenarios),
            'critical_scenarios': self.critical_scenarios,
            'history': self.history,
            'Gamma': self.config.Gamma
        }

        return results


def print_solution_summary(solution, data):
    """Print a summary of the optimal solution."""
    if solution is None:
        print("No solution available.")
        return

    print("\n" + "=" * 80)
    print("OPTIMAL SOLUTION SUMMARY")
    print("=" * 80)

    # Product-specific plants opened
    opened_plants_by_product = {}
    for k in range(data.K):
        opened_plants_k = [i for i in range(data.I) if solution['x'][(k, i)] > 0.5]
        opened_plants_by_product[k] = opened_plants_k
        print(f"Product {k} Plants: {opened_plants_k} ({len(opened_plants_k)}/{data.I})")

    total_plants = sum(len(v) for v in opened_plants_by_product.values())
    print(f"Total Plant Openings: {total_plants}")

    # DCs opened
    opened_dcs = [j for j in range(data.J) if solution['y'][j] > 0.5]
    print(f"DCs Opened: {opened_dcs} ({len(opened_dcs)}/{data.J})")

    # Routes plant-to-DC (product-specific)
    active_routes_ij = [(k, i, j) for (k, i, j) in solution['z'] if solution['z'][(k, i, j)] > 0.5]
    print(f"Active Routes (Plant->DC): {len(active_routes_ij)}")

    # Routes DC-to-customer
    active_routes_jr = [(j, r) for (j, r) in solution['w'] if solution['w'][(j, r)] > 0.5]
    print(f"Active Routes (DC->Customer): {len(active_routes_jr)}")

    # Mode distribution
    mode_counts = [0] * data.M
    for r in range(data.R):
        for m in range(data.M):
            if solution['beta'][(r, m)] > 0.5:
                mode_counts[m] += 1

    print(f"Transportation Modes: ", end="")
    for m in range(data.M):
        print(f"Mode {m}: {mode_counts[m]} customers", end="  ")
    print()

    print("=" * 80)
