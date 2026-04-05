"""
algo_fixed_mode.py  -  C&CG Algorithm with Fixed Mode

Inherits from algo.py (CCGAlgorithm). Overrides initialize() to use
MasterProblemFixedMode instead of MasterProblem. Everything else (loop, SP, bounds) is inherited.

Depends on: algo.py (CCGAlgorithm), master_fixed_mode.py, sub.py, config.py, data_gen.py
Used by: main_fixed_mode.py
"""

from algo import CCGAlgorithm, print_solution_summary
from master_fixed_mode import MasterProblemFixedMode
from sub import Subproblem
from data_gen import SupplyChainData
from config import ProblemConfig


class CCGAlgorithmFixedMode(CCGAlgorithm):
    """C&CG algorithm with a fixed transportation mode."""

    def __init__(self, data: SupplyChainData, config: ProblemConfig, fixed_mode: int):
        """
        Initialize C&CG algorithm with fixed mode.

        Args:
            data: SupplyChainData instance
            config: ProblemConfig instance (must have Gamma set)
            fixed_mode: The transportation mode to force (0, 1, or 2)
        """
        self.fixed_mode = fixed_mode

        # Call parent constructor
        super().__init__(data, config)

    def initialize(self):
        """Initialize algorithm with nominal scenario using fixed-mode Master Problem."""
        print("=" * 80)
        print("INITIALIZING C&CG ALGORITHM (FIXED MODE)")
        print("=" * 80)
        print(f"Problem: K={self.data.K}, I={self.data.I}, J={self.data.J}, "
              f"R={self.data.R}, M={self.data.M}")
        print(f"FIXED MODE: {self.fixed_mode}")
        print(f"Uncertainty Budget: Gamma = {self.config.Gamma}")
        print(f"Convergence Tolerance: epsilon = {self.config.epsilon}")
        print("=" * 80)

        # Create Master Problem with FIXED MODE
        print("Creating Master Problem (FIXED MODE)...")
        self.master = MasterProblemFixedMode(self.data, self.config, self.fixed_mode)

        # Add nominal scenario (eta = 0)
        print("Adding nominal scenario (eta = 0)...")
        eta_plus_nominal = {(r, k): 0 for r in range(self.data.R) for k in range(self.data.K)}
        eta_minus_nominal = {(r, k): 0 for r in range(self.data.R) for k in range(self.data.K)}
        self.master.add_scenario(scenario_id=0, eta_plus=eta_plus_nominal, eta_minus=eta_minus_nominal)
        self.critical_scenarios.append((0, eta_plus_nominal, eta_minus_nominal))

        # Create Subproblem (same as parent - subproblem doesn't need modification)
        print("Creating Subproblem...")
        self.subproblem = Subproblem(self.data, self.config)

        print("Initialization complete!\n")


def print_solution_summary_fixed_mode(solution, data, fixed_mode):
    """Print a summary of the optimal solution with fixed mode info."""
    if solution is None:
        print("No solution available.")
        return

    print("\n" + "=" * 80)
    print(f"OPTIMAL SOLUTION SUMMARY (FIXED MODE: {fixed_mode})")
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

    # Mode distribution (should all be fixed_mode)
    mode_counts = [0] * data.M
    for r in range(data.R):
        for m in range(data.M):
            if solution['beta'][(r, m)] > 0.5:
                mode_counts[m] += 1

    print(f"Transportation Modes (FIXED={fixed_mode}): ", end="")
    for m in range(data.M):
        marker = " *" if m == fixed_mode else ""
        print(f"Mode {m}: {mode_counts[m]} customers{marker}", end="  ")
    print()

    # Verify all customers use fixed mode
    if mode_counts[fixed_mode] == data.R:
        print(f"OK All {data.R} customers correctly use Mode {fixed_mode}")
    else:
        print(f"[!]  WARNING: Expected all {data.R} customers on Mode {fixed_mode}, "
              f"but got {mode_counts[fixed_mode]}")

    print("=" * 80)
