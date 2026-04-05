"""
data_gen.py  -  Data Generation & Utilities

Generates synthetic supply chain data (costs, capacities, demands, coordinates, distances).
Also contains DI functions and coverage threshold computation.

Depends on: config.py (ProblemConfig, DataParameters)
Used by: master.py, sub.py, algo.py, main.py, main_fixed_mode.py, generate_50_seeds.py

Key components:
  - SupplyChainData: data container class (saved/loaded as .pkl)
  - generate_supply_chain_data(): main data generation function
  - generate_DI_vector(): exponential DI with base sqrt(1.6), calibrated to Marino & Zotteri (2018)
  - generate_DI_vector_linear(): linear DI for robustness check (same anchor points)
  - generate_DI_scenarios(): creates HD/MD/LD/Mixed scenarios from kappa ranges
  - compute_coverage_thresholds(): computes D_bar[m] percentiles for coverage constraint
"""

import numpy as np
import pickle
import os
from config import ProblemConfig, DataParameters


class SupplyChainData:
    """Container for all problem data."""

    def __init__(self, config: ProblemConfig):
        """
        Initialize data container.

        Args:
            config: ProblemConfig instance with problem dimensions
        """
        self.config = config
        self.K = config.K
        self.I = config.I
        self.J = config.J
        self.R = config.R
        self.M = config.M

        # Economic parameters (scalars)
        self.S = None
        self.SC = None

        # Cost parameters (dictionaries with appropriate indices)
        self.C_plant = {}  # {(k,i): value}
        self.C_dc = {}  # {j: value}
        self.O = {}  # {j: value}
        self.L1 = {}  # {(k,i,j): value}
        self.L2 = {}  # {(j,r): value}
        self.F = {}  # {(k,i): value}
        self.h = {}  # {j: value}
        self.t = None  # scalar
        self.TC = {}  # {m: value}

        # Capacity parameters
        self.MP = {}  # {(k,i): value}
        self.MC = {}  # {j: value}

        # Demand parameters
        self.s_rk = {}  # {(r,k): 0/1} - binary indicator: does customer r demand product k?
        self.mu = {}  # {(r,k): value}
        self.mu_hat = {}  # {(r,k): value}
        self.DI = {}  # {(m,k): value} - default DI (for backward compatibility)
        self.DI_scenarios = {}  # {'HD': [[...], [...], [...]], 'MD': ..., 'LD': ..., 'Mixed': ...}
        self.DI_k_params = {}  # Store k parameters for each scenario

        # Distance matrices
        self.D1 = {}  # {(k,i,j): value}
        self.D2 = {}  # {(j,r): value}

        # Coordinates (for distance calculation)
        self.plant_coords = {}  # {(k,i): (x,y)} - Product-specific plants
        self.dc_coords = {}  # {j: (x,y)}
        self.customer_coords = {}  # {r: (x,y)}

    def save(self, filepath):
        """Save data to pickle file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Data saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load data from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def summary(self):
        """Print summary statistics of the data."""
        print("=" * 60)
        print("SUPPLY CHAIN DATA SUMMARY")
        print("=" * 60)
        print(f"Dimensions: K={self.K}, I={self.I}, J={self.J}, R={self.R}, M={self.M}")
        print(f"\nEconomic Parameters:")
        print(f"  Selling price (S): {self.S}")
        print(f"  Shortage cost (SC): {self.SC}")
        print(f"\nCapacities:")
        print(f"  Plant capacity (MP): min={min(self.MP.values()):.0f}, "
              f"max={max(self.MP.values()):.0f}")
        print(f"  DC capacity (MC): min={min(self.MC.values()):.0f}, "
              f"max={max(self.MC.values()):.0f}")
        print(f"\nDemand Structure (s_rk matrix):")
        total_demands = sum(self.s_rk.values())
        possible_demands = self.R * self.K
        sparsity = (1 - total_demands / possible_demands) * 100
        print(f"  Total customer-product pairs: {total_demands}/{possible_demands} ({sparsity:.1f}% sparse)")
        # Count how many products each customer demands on average
        demands_per_customer = [sum(self.s_rk[(r,k)] for k in range(self.K)) for r in range(self.R)]
        avg_demands = sum(demands_per_customer) / self.R
        print(f"  Average products per customer: {avg_demands:.2f}")
        print(f"\nDemand (only non-zero entries):")
        nonzero_mu = [v for v in self.mu.values() if v > 0]
        nonzero_mu_hat = [v for v in self.mu_hat.values() if v > 0]
        print(f"  Nominal (mu): min={min(nonzero_mu):.2f}, "
              f"max={max(nonzero_mu):.2f}, count={len(nonzero_mu)}")
        print(f"  Deviation (mu_hat): min={min(nonzero_mu_hat):.2f}, "
              f"max={max(nonzero_mu_hat):.2f}")
        print(f"\nCosts:")
        print(f"  Plant fixed cost: min={min(self.C_plant.values()):.0f}, "
              f"max={max(self.C_plant.values()):.0f}")
        print(f"  DC fixed cost: min={min(self.C_dc.values()):.0f}, "
              f"max={max(self.C_dc.values()):.0f}")
        print(f"  Transport modes (TC): {[self.TC[m] for m in range(self.M)]}")

        # Print DI scenarios if available
        if hasattr(self, 'DI_scenarios') and self.DI_scenarios:
            print(f"\nDemand Increase (DI) Scenarios:")
            for scenario_name in ['HD', 'MD', 'LD', 'Mixed']:
                if scenario_name in self.DI_scenarios:
                    print(f"  {scenario_name}:")
                    for k, di_vec in enumerate(self.DI_scenarios[scenario_name]):
                        k_param = self.DI_k_params[scenario_name][k]
                        print(f"    Product {k}: {[f'{v:.3f}' for v in di_vec]} (k={k_param:.3f})")

        print("=" * 60)


def generate_coordinates(n, grid_size, seed=None, mode='uniform'):
    """
    Generate random coordinates on a 2D grid.

    Args:
        n: Number of points
        grid_size: Size of the grid (points in [0, grid_size] x [0, grid_size])
        seed: Random seed for reproducibility
        mode: 'uniform', 'gaussian', or 'donut'

    Returns:
        dict {i: (x, y)} of coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    coords = {}

    if mode == 'uniform':
        for i in range(n):
            x = np.random.uniform(0, grid_size)
            y = np.random.uniform(0, grid_size)
            coords[i] = (x, y)

    elif mode == 'gaussian':
        # 2D Gaussian centered at grid center
        center = grid_size / 2
        std = grid_size / 5  # Standard deviation

        x_coords = np.random.normal(center, std, n)
        y_coords = np.random.normal(center, std, n)

        # Clip to grid bounds
        x_coords = np.clip(x_coords, 0, grid_size)
        y_coords = np.clip(y_coords, 0, grid_size)

        for i in range(n):
            coords[i] = (float(x_coords[i]), float(y_coords[i]))

    elif mode == 'donut':
        # Exclude central region (donut pattern)
        exclude_min = int(0.2 * grid_size)
        exclude_max = int(0.8 * grid_size)

        # Generate all valid points in donut region
        available_points = [
            (float(x), float(y))
            for x in range(int(grid_size) + 1)
            for y in range(int(grid_size) + 1)
            if not (exclude_min <= x < exclude_max and exclude_min <= y < exclude_max)
        ]

        if n > len(available_points):
            raise ValueError(f"Cannot sample {n} points from donut region with only {len(available_points)} available")

        import random
        random.seed(seed)
        selected = random.sample(available_points, n)

        for i in range(n):
            coords[i] = selected[i]

    return coords


def euclidean_distance(coord1, coord2):
    """Calculate Euclidean distance between two 2D points."""
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)


def compute_coverage_thresholds(data, scenario='no_limit'):
    """
    Compute D_bar[m] based on D2 distance distribution percentiles.
    D_bar[m] is the maximum distance at which mode m can serve a customer.

    Args:
        data: SupplyChainData instance (must have D2 populated)
        scenario: 'tight', 'moderate', 'relaxed', or 'no_limit'

    Returns:
        dict {m: max_distance} for each mode
    """
    all_D2 = list(data.D2.values())
    if scenario == 'tight':
        return {0: float('inf'), 1: np.percentile(all_D2, 50), 2: np.percentile(all_D2, 25)}
    elif scenario == 'moderate':
        return {0: float('inf'), 1: np.percentile(all_D2, 75), 2: np.percentile(all_D2, 50)}
    elif scenario == 'relaxed':
        return {0: float('inf'), 1: np.percentile(all_D2, 90), 2: np.percentile(all_D2, 75)}
    else:  # no_limit
        return {0: float('inf'), 1: float('inf'), 2: float('inf')}


def generate_DI_vector(k, M=3):
    """
    Generate DI vector using the formula: DI_m = base^(k*m)

    Base = sqrt(1.6) ~ 1.265, calibrated to Marino & Zotteri (2018):
    At HD max (k=1, m=2): DI = 1.60 (+60% demand increase),
    matching the empirical upper bound.

    Args:
        k: Sensitivity parameter (kappa)
        M: Number of transportation modes

    Returns:
        list: DI values for each mode [mode_0, mode_1, ..., mode_M-1]
    """
    import math
    base = math.sqrt(1.6)  # ~ 1.2649
    return [base ** (k * m) for m in range(M)]


def generate_DI_vector_linear(k, M=3):
    """
    Generate DI vector using LINEAR function: DI_m = 1 + 0.3 * k * m

    Anchored to the same endpoint as the exponential function:
    At k=1, m=2: DI = 1 + 0.3*1*2 = 1.60 (same as sqrt(1.6)^2 = 1.60)
    At k=1, m=1: DI = 1.30 (vs exponential 1.265)

    Used for robustness check: verifying conclusions are not dependent on function form.

    Args:
        k: Sensitivity parameter (kappa)
        M: Number of transportation modes

    Returns:
        list: DI values for each mode [mode_0, mode_1, ..., mode_M-1]
    """
    return [1.0 + 0.3 * k * m for m in range(M)]


def generate_DI_scenarios(K=3, M=3, seed=None):
    """
    Generate DI matrices for all four scenarios: HD, MD, LD, Mixed.

    Formula: DI_m = sqrt(1.6)^(k*m) for m = 0, 1, 2

    Args:
        K: Number of products
        M: Number of transportation modes
        seed: Random seed for reproducibility

    Returns:
        dict: {
            'HD': [[DI for product 0], [DI for product 1], [DI for product 2]],
            'MD': [...],
            'LD': [...],
            'Mixed': [...]
        }
        dict: k_params used for each scenario
    """
    if seed is not None:
        np.random.seed(seed + 1000)  # Different seed for DI generation

    # k parameter ranges for each scenario
    k_ranges = {
        'HD': (2/3, 1.0),      # High demand sensitivity
        'MD': (1/3, 2/3),      # Medium demand sensitivity
        'LD': (0.0, 1/3)       # Low demand sensitivity
    }

    scenarios = {}
    k_params = {}

    # Generate HD, MD, LD scenarios
    for scenario_type, (k_min, k_max) in k_ranges.items():
        scenario_matrix = []
        scenario_k_params = []

        for k in range(K):
            # Sample k parameter for this product
            k_param = np.random.uniform(k_min, k_max)
            scenario_k_params.append(k_param)

            # Generate DI vector
            DI_vector = generate_DI_vector(k_param, M)
            scenario_matrix.append(DI_vector)

        scenarios[scenario_type] = scenario_matrix
        k_params[scenario_type] = scenario_k_params

    # Generate Mixed scenario: Product 0 from HD, Product 1 from MD, Product 2 from LD
    # Handle cases where K < 3
    if K >= 3:
        scenarios['Mixed'] = [
            scenarios['HD'][0],   # Product 0: High sensitivity
            scenarios['MD'][1],   # Product 1: Medium sensitivity
            scenarios['LD'][2]    # Product 2: Low sensitivity
        ]
        k_params['Mixed'] = [
            k_params['HD'][0],
            k_params['MD'][1],
            k_params['LD'][2]
        ]
    elif K == 2:
        scenarios['Mixed'] = [
            scenarios['HD'][0],   # Product 0: High sensitivity
            scenarios['LD'][1]    # Product 1: Low sensitivity
        ]
        k_params['Mixed'] = [
            k_params['HD'][0],
            k_params['LD'][1]
        ]
    else:  # K == 1
        scenarios['Mixed'] = [
            scenarios['MD'][0]    # Product 0: Medium sensitivity
        ]
        k_params['Mixed'] = [
            k_params['MD'][0]
        ]

    return scenarios, k_params


def generate_supply_chain_data(config: ProblemConfig, seed=42):
    """
    Generate synthetic supply chain data based on configuration.

    Args:
        config: ProblemConfig instance
        seed: Random seed for reproducibility

    Returns:
        SupplyChainData instance
    """
    np.random.seed(seed)
    data = SupplyChainData(config)
    params = DataParameters()

    K, I, J, R, M = config.K, config.I, config.J, config.R, config.M

    # Capacity scaling factor (defaults to 1.0 for backward compatibility)
    cap_scale = getattr(config, 'capacity_scale', 1.0)

    print(f"Generating supply chain data for {config}...")
    if cap_scale != 1.0:
        print(f"  Capacity scaling factor: {cap_scale}x")

    # ========== Scalar Parameters ==========
    data.S = params.S
    data.SC = params.SC
    data.t = params.t

    # ========== Transportation Mode Costs ==========
    TC_list = params.get_TC_modes(M)
    for m in range(M):
        data.TC[m] = TC_list[m]

    # ========== Demand Increase Factors ==========
    # Generate all four DI scenarios: HD, MD, LD, Mixed
    DI_scenarios, k_params = generate_DI_scenarios(K, M, seed)
    data.DI_scenarios = DI_scenarios
    data.DI_k_params = k_params

    # Set default DI to HD scenario (for backward compatibility)
    for m in range(M):
        for k in range(K):
            data.DI[(m, k)] = DI_scenarios['HD'][k][m]

    # ========== Fixed Costs ==========
    # Plant fixed costs C^plant_{ki}
    for k in range(K):
        for i in range(I):
            data.C_plant[(k, i)] = np.random.uniform(params.C_plant_min, params.C_plant_max)

    # DC fixed costs C^dc_j
    for j in range(J):
        data.C_dc[j] = np.random.uniform(params.C_dc_min, params.C_dc_max)

    # Ordering costs O_j
    for j in range(J):
        data.O[j] = np.random.uniform(params.O_min, params.O_max)

    # Production costs F_{ki}
    for k in range(K):
        for i in range(I):
            data.F[(k, i)] = np.random.uniform(params.F_min, params.F_max)

    # Holding costs h_j
    for j in range(J):
        data.h[j] = np.random.uniform(params.h_min, params.h_max)

    # ========== Capacities ==========
    # Plant capacities MP_{ki} (scaled by cap_scale for larger instances)
    for k in range(K):
        for i in range(I):
            data.MP[(k, i)] = np.random.uniform(params.MP_min * cap_scale, params.MP_max * cap_scale)

    # DC capacities MC_j (scaled by cap_scale for larger instances)
    for j in range(J):
        data.MC[j] = np.random.uniform(params.MC_min * cap_scale, params.MC_max * cap_scale)

    # ========== Demand Parameters ==========
    # First generate s_rk matrix (binary indicator: does customer r demand product k?)
    s_matrix = np.random.randint(0, 2, size=(R, K))

    # Ensure each customer demands at least one product
    for r in range(R):
        if np.all(s_matrix[r] == 0):
            s_matrix[r, np.random.randint(0, K)] = 1

    # Store s_rk
    for r in range(R):
        for k in range(K):
            data.s_rk[(r, k)] = int(s_matrix[r, k])

    # Nominal demand mu_{rk} - only non-zero where s_rk = 1
    for r in range(R):
        for k in range(K):
            if data.s_rk[(r, k)] == 1:
                # Nominal demand: 10 * U[0.8, 3.5], range [8, 35]
                data.mu[(r, k)] = 10.0 * np.random.uniform(0.8, 3.5)
            else:
                # No demand for this product
                data.mu[(r, k)] = 0.0

    # Demand deviation mu_hat_{rk} using min(mu, U[4,10]) instead of percentage
    for r in range(R):
        for k in range(K):
            if data.s_rk[(r, k)] == 1:
                # Use min(mu, uniform[4,10]) as in old_data_generation.py
                random_bound = np.random.uniform(4, 10)
                data.mu_hat[(r, k)] = min(data.mu[(r, k)], random_bound)
            else:
                data.mu_hat[(r, k)] = 0.0

    # ========== Coordinates and Distances ==========
    # Generate coordinates with spatial patterns
    # Plants: uniform distribution (PRODUCT-SPECIFIC - each product has its own set of plants)
    for k in range(K):
        # Each product k has I independent plant locations
        plant_coords_k = generate_coordinates(I, config.grid_size, seed=seed+k*100, mode='uniform')
        for i in range(I):
            data.plant_coords[(k, i)] = plant_coords_k[i]

    # DCs: donut pattern (exclude center) for better spatial distribution (SHARED across products)
    data.dc_coords = generate_coordinates(J, config.grid_size, seed=seed+1, mode='donut')
    # Customers: Gaussian clustered around center (more realistic)
    data.customer_coords = generate_coordinates(R, config.grid_size, seed=seed+2, mode='gaussian')

    # Calculate distances D1_{kij} (product k plant i to DC j)
    # Each product has independent plant locations
    for k in range(K):
        for i in range(I):
            for j in range(J):
                dist = euclidean_distance(data.plant_coords[(k, i)], data.dc_coords[j])
                data.D1[(k, i, j)] = dist

    # Calculate distances D2_{jr} (DC j to customer r)
    for j in range(J):
        for r in range(R):
            dist = euclidean_distance(data.dc_coords[j], data.customer_coords[r])
            data.D2[(j, r)] = dist

    # ========== Route Fixed Costs ==========
    # L1_{kij} (route cost plant-to-DC, distance-based component)
    for k in range(K):
        for i in range(I):
            for j in range(J):
                base_cost = np.random.uniform(params.L1_min, params.L1_max)
                # Add distance-dependent component
                dist_component = 0.5 * data.D1[(k, i, j)]
                data.L1[(k, i, j)] = base_cost + dist_component

    # L2_{jr} (route cost DC-to-customer)
    for j in range(J):
        for r in range(R):
            base_cost = np.random.uniform(params.L2_min, params.L2_max)
            dist_component = 0.3 * data.D2[(j, r)]
            data.L2[(j, r)] = base_cost + dist_component

    print("Data generation complete!")
    return data


if __name__ == "__main__":
    """Test data generation."""
    # Generate toy instance
    print("\n" + "="*60)
    print("GENERATING TOY INSTANCE")
    print("="*60)
    config_toy = ProblemConfig(instance_type='toy')
    data_toy = generate_supply_chain_data(config_toy, seed=42)
    data_toy.summary()
    data_toy.save(config_toy.data_file)

    # Generate full instance
    print("\n" + "="*60)
    print("GENERATING FULL INSTANCE")
    print("="*60)
    config_full = ProblemConfig(instance_type='full')
    data_full = generate_supply_chain_data(config_full, seed=42)
    data_full.summary()
    data_full.save(config_full.data_file)

    print("\nData generation test complete!")
