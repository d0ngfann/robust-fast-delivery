"""
generate_50_seeds.py  -  Dataset Generator

Generates 50 datasets (seeds 1-50) for a given instance type.
Each .pkl file contains a SupplyChainData object with all parameters, coordinates,
distances, costs, demands, and 4 DI scenarios (HD/MD/LD/Mixed).

Depends on: config.py (ProblemConfig), data_gen.py (generate_supply_chain_data)
Output: ../data/DH_data_{instance_type}_seed{1-50}.pkl

Usage (run from codes/ directory):
    python generate_50_seeds.py          # Default: 'full' (R=50)
    python generate_50_seeds.py full200  # Large-scale (R=200, 4x capacity)
"""
import sys
import os
from config import ProblemConfig
from data_gen import generate_supply_chain_data

# Data directory: ../data/ relative to this script's location
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def main():
    # Parse optional instance_type argument
    instance_type = sys.argv[1] if len(sys.argv) > 1 else 'full'

    config = ProblemConfig(instance_type=instance_type)

    print("=" * 60)
    print(f"GENERATING 50 DATA INSTANCES ({instance_type}, seed 1-50)")
    print(f"  R={config.R}, J={config.J}, I={config.I}, K={config.K}, M={config.M}")
    print(f"  Capacity scale: {config.capacity_scale}x")
    print(f"  Output directory: {os.path.abspath(DATA_DIR)}")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    for seed in range(1, 51):
        print(f"\nGenerating seed {seed}/50...")

        # Generate data with this seed
        data = generate_supply_chain_data(config, seed=seed)

        # Save to seed-specific file in ../data/
        filepath = os.path.join(DATA_DIR, f"DH_data_{instance_type}_seed{seed}.pkl")
        data.save(filepath)

        # Print progress
        if seed % 10 == 0:
            print(f"Progress: {seed}/50 completed")

    print("\n" + "=" * 60)
    print("ALL 50 INSTANCES GENERATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
