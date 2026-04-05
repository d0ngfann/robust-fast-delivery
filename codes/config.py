"""
config.py  -  Configuration & Parameters

Problem dimensions, solver settings, cost parameters for the robust supply chain model.
All other modules import from this file.

Used by: data_gen.py, master.py, sub.py, algo.py, main.py, main_fixed_mode.py
Key classes:
  - ProblemConfig: instance type (toy/full/full200), epsilon, Gamma, solver tolerances
  - DataParameters: cost ranges, capacity ranges, DI base values, demand parameters
  - SensitivityConfig: Gamma sensitivity analysis settings
"""

import os

class ProblemConfig:
    """Configuration class for problem dimensions and algorithm settings."""

    def __init__(self, instance_type='full'):
        """
        Initialize problem configuration.

        Args:
            instance_type: 'toy' for small test instance, 'full' for complete problem,
                           'full200' for large-scale (R=200, 4x capacity)
        """
        if instance_type == 'toy':
            # Small test instance for verification
            self.K = 1  # Products
            self.I = 2  # Plants
            self.J = 2  # DCs
            self.R = 5  # Customers
            self.M = 2  # Transport modes
            self.grid_size = 50  # Coordinate grid size
        elif instance_type == 'full':
            # Full-scale problem
            self.K = 3   # Products
            self.I = 3   # Plants
            self.J = 5   # DCs
            self.R = 50  # Customers
            self.M = 3   # Transport modes
            self.grid_size = 100  # Coordinate grid size
        elif instance_type == 'full200':
            # Large-scale problem with R=200 customers
            self.K = 3    # Products (same as full)
            self.I = 3    # Plants per product (same as full)
            self.J = 5    # DCs (same as full)
            self.R = 200  # Customers (4x of full)
            self.M = 3    # Transport modes (same as full)
            self.grid_size = 100  # Coordinate grid size (same as full)
        else:
            raise ValueError(f"Unknown instance type: {instance_type}")

        # Capacity scaling factor (for larger instances)
        # full200 uses 4x capacity to match 4x customers; costs stay the same
        if instance_type == 'full200':
            self.capacity_scale = 4.0
        else:
            self.capacity_scale = 1.0

        # Algorithm parameters
        self.epsilon = 100  # Convergence tolerance (absolute gap), fixed for all instances
        self.max_iterations = 100  # Maximum C&CG iterations

        # Instance-specific algorithm tuning
        if instance_type == 'full200':
            self.ccg_time_limit = 82800  # 23 hours time limit

        # Gurobi solver settings
        self.gurobi_time_limit = 3600  # seconds per optimization
        self.gurobi_mip_gap = 1e-4  # 0.01% MIP gap (balanced speed/accuracy)
        self.gurobi_threads = 0  # 0 = use all available cores
        self.gurobi_output_flag = 1  # 1 = show logs, 0 = silent

        # Additional solver tolerances for better convergence
        self.gurobi_feasibility_tol = 1e-9  # Primal feasibility tolerance
        self.gurobi_int_feas_tol = 1e-9  # Integer feasibility tolerance
        self.gurobi_opt_tol = 1e-9  # Dual feasibility tolerance

        # Uncertainty budget (configurable for sensitivity analysis)
        # Will be set externally for each run
        self.Gamma = None

        # Data file paths
        self.data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', f"DH_data_{instance_type}.pkl")
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')

        # Output settings
        self.log_console = True
        self.save_results = True

    def set_gamma(self, gamma_value):
        """Set the uncertainty budget Gamma_k (same for all products)."""
        if gamma_value < 0 or gamma_value > self.R:
            raise ValueError(f"Gamma must be in [0, {self.R}]")
        self.Gamma = gamma_value

    def __repr__(self):
        return (f"ProblemConfig(K={self.K}, I={self.I}, J={self.J}, "
                f"R={self.R}, M={self.M}, Gamma={self.Gamma})")


class DataParameters:
    """
    Default parameter values for data generation.
    These are reasonable values for a supply chain optimization problem.
    """

    # Economic parameters
    S = 150.0  # Unit selling price
    SC = 50.0  # Unit shortage cost

    # Cost parameters (ranges for random generation)
    # Plant fixed costs: C^plant_{ki}
    C_plant_min = 3500
    C_plant_max = 7000

    # DC fixed costs: C^dc_j
    C_dc_min = 2000
    C_dc_max = 4000

    # Ordering costs at DCs: O_j
    O_min = 3500
    O_max = 10000

    # Route fixed costs plant-to-DC: L^1_{kij}
    L1_min = 1000
    L1_max = 5000

    # Route fixed costs DC-to-customer: L^2_{jr}
    L2_min = 500
    L2_max = 2000

    # Unit production cost: F_{ki}
    F_min = 10.0
    F_max = 30.0

    # Unit holding cost at DC: h_j
    h_min = 2.0
    h_max = 8.0

    # Unit transportation cost (plant to DC): t
    t = 0.1  # per unit distance

    # Transportation cost coefficients by mode: TC_m
    # Mode 0: slow/cheap, Mode 1: medium, Mode 2: fast/expensive
    TC_modes = [0.05, 0.10, 0.20]  # per unit distance

    # Demand increase factors by mode: DI_{mk}
    # Faster delivery -> higher demand
    # Shape: (M, K) - mode x product
    # Base = sqrt(1.6) ~ 1.265, calibrated to Marino & Zotteri (2018)
    DI_base = [
        [1.0, 1.0, 1.0],    # Mode 0: no increase
        [1.265, 1.265, 1.265],  # Mode 1: ~27% increase (at kappa=1)
        [1.60, 1.60, 1.60]  # Mode 2: 60% increase (at kappa=1)
    ]

    # Capacity parameters
    # Plant capacity: MP_{ki}
    MP_min = 900
    MP_max = 1800

    # DC capacity: MC_j
    MC_min = 1500
    MC_max = 1800

    # Demand parameters
    # Nominal demand: mu_{rk} - actual generation in data_gen.py uses 10*U[0.8, 3.5] = [8, 35]
    mu_min = 8.0
    mu_max = 35.0

    # Demand deviation (maximum uncertainty): mu_hat_{rk}
    # As a fraction of nominal demand
    mu_hat_factor_min = 0.2  # 20% of nominal
    mu_hat_factor_max = 0.5  # 50% of nominal

    @classmethod
    def get_TC_modes(cls, M):
        """Get transport mode costs, extended if M > 3."""
        if M <= len(cls.TC_modes):
            return cls.TC_modes[:M]
        else:
            # Extend with linear interpolation
            return cls.TC_modes + [cls.TC_modes[-1]] * (M - len(cls.TC_modes))

    @classmethod
    def get_DI_matrix(cls, M, K):
        """Get demand increase factors matrix (M x K)."""
        import numpy as np
        if M <= len(cls.DI_base) and K <= len(cls.DI_base[0]):
            # Extract the submatrix from nested list
            sublist = [cls.DI_base[m][:K] for m in range(M)]
            return np.array(sublist)
        else:
            # Create default matrix
            DI = np.ones((M, K))
            for m in range(M):
                # Higher mode -> higher demand increase
                factor = 1.0 + 0.25 * m
                DI[m, :] = factor
            return DI


# Sensitivity analysis settings
class SensitivityConfig:
    """Configuration for Gamma sensitivity analysis."""

    def __init__(self, R):
        """
        Initialize sensitivity analysis configuration.

        Args:
            R: Number of customers (determines max Gamma)
        """
        self.gamma_values = list(range(0, R + 1, max(1, R // 10)))  # 0, R/10, 2R/10, ..., R
        self.output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result', 'DH_sensitivity_results.csv')
        self.plot_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result', 'DH_sensitivity_plot.png')

    def __repr__(self):
        return f"SensitivityConfig(gamma_values={self.gamma_values})"
