# Replication Code: Robust Supply Chain Design with Delivery-Speed-Dependent Demand

This repository contains the Python implementation for the numerical experiments in:

> **Two-Stage Robust Supply Chain Design with Delivery-Speed-Dependent Demand: A Column-and-Constraint Generation Approach**  
> *European Journal of Operational Research*, submitted 2026.

## Requirements

- Python 3.11+
- Gurobi 10.0+ with a valid license  
  (Academic license available at https://www.gurobi.com/academia/academic-program-and-licenses/)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Activate a Gurobi academic license:
```bash
grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

## Repository Structure

```
├── codes/
│   ├── config.py              # Problem dimensions, solver settings, and cost parameters
│   ├── data_gen.py            # Data generation: coordinates, costs, demands, DI functions
│   ├── master.py              # Master Problem (first-stage MIP formulation)
│   ├── sub.py                 # Subproblem (worst-case scenario via dual + McCormick linearization)
│   ├── algo.py                # C&CG algorithm main loop
│   ├── main.py                # Entry point for Optimal policy runs
│   ├── master_fixed_mode.py   # Master Problem variant for fixed-mode policies
│   ├── algo_fixed_mode.py     # C&CG variant for FM0/FM1/FM2 policies
│   ├── main_fixed_mode.py     # Entry point for fixed-mode policy runs
│   ├── generate_50_seeds.py   # Regenerates all instance files in data/
│   ├── run_exp1.py            # Experiment 1: R=50, Γ=10, 800 runs
│   ├── run_exp2.py            # Experiment 2: R=200, Γ sensitivity, 4,000 runs
│   ├── run_breakeven.py       # Breakeven analysis: TC₂ sensitivity, 500 runs
│   ├── run_linear_di.py       # Robustness check: linear vs. exponential DI, 320 runs
│   └── run_coverage.py        # Service coverage sensitivity, 480 runs
├── data/                      # Pre-generated instance files (100 .pkl files)
├── requirements.txt
└── README.md
```

## Reproducing the Results

All scripts must be run from the `codes/` directory:

```bash
cd codes
```

### Quick test
```bash
# Optimal policy
python main.py full 10 --seed 1 --di HD

# Fixed-mode policy (FM2)
python main_fixed_mode.py full 10 --seed 1 --di HD --mode 2
```

### Full experiments

| Script | Description | Approx. time |
|---|---|---|
| `run_exp1.py` | Exp 1: base results (R=50, Γ=10) | ~1 hour |
| `run_exp2.py` | Exp 2: Γ sensitivity (R=200) | ~15 hours (4 workers) |
| `run_breakeven.py` | Breakeven: TC₂ sensitivity | ~40 min |
| `run_linear_di.py` | Robustness check: DI function form | ~30 min |
| `run_coverage.py` | Service coverage sensitivity | ~40 min |

```bash
python run_exp1.py
python run_exp2.py
python run_breakeven.py
python run_linear_di.py
python run_coverage.py
```

Results are saved as CSV files to `../result/{exp1,exp2,breakeven,linear_di,coverage}/`.

### Instance data

The `data/` folder contains all 100 pre-generated instances (seeds 1–50 for both R=50 and R=200).
To regenerate from scratch:
```bash
python generate_50_seeds.py full      # R=50 instances (seeds 1–50)
python generate_50_seeds.py full200   # R=200 instances (seeds 1–50)
```

## Command-line Arguments

| Argument | Default | Description |
|---|---|---|
| `instance_type` | — | `full` (R=50) or `full200` (R=200) |
| `gamma` | — | Uncertainty budget Γ (integer) |
| `--seed N` | — | Dataset seed (1–50) |
| `--di {HD,MD,LD,Mixed}` | `HD` | Demand sensitivity scenario |
| `--mode {0,1,2}` | — | Fixed transport mode (`main_fixed_mode.py` only) |
| `--tc2 VALUE` | — | Override TC₂ for breakeven analysis |
| `--di-func {exponential,linear}` | `exponential` | DI function form for robustness check |
| `--coverage {tight,moderate,relaxed,no_limit}` | `no_limit` | Service coverage constraint |
