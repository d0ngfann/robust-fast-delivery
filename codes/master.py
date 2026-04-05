"""
master.py  -  Master Problem (MP) Formulation

First-stage optimization: plant/DC opening, routing, mode selection, optimality cuts.
Gurobi MIP model. Scenarios are added dynamically by the C&CG algorithm.

Depends on: config.py, data_gen.py (SupplyChainData)
Used by: algo.py (CCGAlgorithm creates and solves this)
Extended by: master_fixed_mode.py (inherits, fixes alpha to single mode)

Key methods:
  - add_scenario(): adds a new worst-case scenario with linearized operational profit
  - solve(): solves current MP, returns first-stage solution + theta
  - apply_coverage_constraint(D_bar): fixes alpha[j,r,m]=0 where distance > threshold
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from data_gen import SupplyChainData
from config import ProblemConfig


class MasterProblem:
    """Master Problem for C&CG algorithm."""

    def __init__(self, data: SupplyChainData, config: ProblemConfig):
        """
        Initialize Master Problem.

        Args:
            data: SupplyChainData instance with all parameters
            config: ProblemConfig instance
        """
        self.data = data
        self.config = config
        self.K = data.K
        self.I = data.I
        self.J = data.J
        self.R = data.R
        self.M = data.M

        # Gurobi model
        self.model = gp.Model("MasterProblem")
        self._configure_solver()

        # Decision variables
        self.x = {}  # Product-specific plant opening: x[k,i]
        self.y = {}  # DC opening: y[j]
        self.z = {}  # Product-specific plant-DC route: z[k,i,j]
        self.w = {}  # DC-Customer assignment: w[j,r]
        self.beta = {}  # Customer mode selection: beta[r,m]
        self.alpha = {}  # Mode-route combination: alpha[j,r,m]
        self.theta = None  # Worst-case operational profit

        # Second-stage variables per scenario (will be added dynamically)
        self.scenarios = []  # List of scenario data: [(scenario_id, eta_plus, eta_minus), ...]
        self.A_ij = {}  # Plant-DC flow: A_ij[k,i,j,l]
        self.A_jr = {}  # DC-Customer flow: A_jr[k,j,r,l]
        self.u = {}  # Shortage: u[r,k,l]
        self.X = {}  # Linearization variable: X[j,r,m,k,l] = alpha[j,r,m] * A_jr[k,j,r,l]

        # Build model
        self._build_variables()
        self._build_objective()
        self._build_network_constraints()

        self.model.update()

    def _configure_solver(self):
        """Configure Gurobi solver parameters."""
        self.model.setParam('TimeLimit', self.config.gurobi_time_limit)
        self.model.setParam('MIPGap', self.config.gurobi_mip_gap)
        self.model.setParam('Threads', self.config.gurobi_threads)
        self.model.setParam('OutputFlag', self.config.gurobi_output_flag)
        self.model.setParam('FeasibilityTol', self.config.gurobi_feasibility_tol)
        self.model.setParam('IntFeasTol', self.config.gurobi_int_feas_tol)
        self.model.setParam('OptimalityTol', self.config.gurobi_opt_tol)

    def _build_variables(self):
        """Create first-stage decision variables."""
        # Product-specific plant opening variables
        for k in range(self.K):
            for i in range(self.I):
                self.x[(k, i)] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}")

        # DC opening variables
        for j in range(self.J):
            self.y[j] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{j}")

        # Product-specific plant-DC route variables
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    self.z[(k, i, j)] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{k}_{i}_{j}")

        # DC-Customer assignment variables
        for j in range(self.J):
            for r in range(self.R):
                self.w[(j, r)] = self.model.addVar(vtype=GRB.BINARY, name=f"w_{j}_{r}")

        # Customer mode selection variables
        for r in range(self.R):
            for m in range(self.M):
                self.beta[(r, m)] = self.model.addVar(vtype=GRB.BINARY, name=f"beta_{r}_{m}")

        # Mode-route combination variables
        for j in range(self.J):
            for r in range(self.R):
                for m in range(self.M):
                    self.alpha[(j, r, m)] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"alpha_{j}_{r}_{m}"
                    )

        # Auxiliary variable for worst-case profit
        self.theta = self.model.addVar(
            vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="theta"
        )

    def _build_objective(self):
        """Build objective function: max -OC - FC + theta."""
        # Ordering cost: OC = Sum_j O_j * y_j
        OC = gp.quicksum(self.data.O[j] * self.y[j] for j in range(self.J))

        # Fixed costs: FC = plant + DC + routes
        # Plant fixed costs: Sum_k Sum_i C_plant_{ki} * x_{ki} (PRODUCT-SPECIFIC)
        plant_cost = gp.quicksum(
            self.data.C_plant[(k, i)] * self.x[(k, i)]
            for k in range(self.K)
            for i in range(self.I)
        )

        # DC fixed costs: Sum_j C_dc_j * y_j
        dc_cost = gp.quicksum(self.data.C_dc[j] * self.y[j] for j in range(self.J))

        # Route fixed costs plant-to-DC: Sum_k Sum_i Sum_j L1_{kij} * z_{kij} (PRODUCT-SPECIFIC)
        route1_cost = gp.quicksum(
            self.data.L1[(k, i, j)] * self.z[(k, i, j)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Route fixed costs DC-to-customer: Sum_j Sum_r L2_{jr} * w_jr
        route2_cost = gp.quicksum(
            self.data.L2[(j, r)] * self.w[(j, r)]
            for j in range(self.J)
            for r in range(self.R)
        )

        FC = plant_cost + dc_cost + route1_cost + route2_cost

        # Total objective: maximize -OC - FC + theta
        self.model.setObjective(-OC - FC + self.theta, GRB.MAXIMIZE)

    def _build_network_constraints(self):
        """Build network topology constraints."""
        # Single sourcing: each customer served by exactly one DC
        # Sum_j w_jr = 1, for all r
        for r in range(self.R):
            self.model.addConstr(
                gp.quicksum(self.w[(j, r)] for j in range(self.J)) == 1,
                name=f"single_source_r{r}"
            )

        # Mode selection consistency: Sum_m alpha_jrm = w_jr, for all j,r
        for j in range(self.J):
            for r in range(self.R):
                self.model.addConstr(
                    gp.quicksum(self.alpha[(j, r, m)] for m in range(self.M)) == self.w[(j, r)],
                    name=f"mode_consistency_j{j}_r{r}"
                )

        # Beta consistency: Sum_j alpha_jrm = beta_rm, for all r,m
        for r in range(self.R):
            for m in range(self.M):
                self.model.addConstr(
                    gp.quicksum(self.alpha[(j, r, m)] for j in range(self.J)) == self.beta[(r, m)],
                    name=f"beta_consistency_r{r}_m{m}"
                )

        # PRODUCT-SPECIFIC PLANT CONSTRAINTS
        # Each product must have at least one plant opened: Sum_i x_{ki} >= 1, for all k
        for k in range(self.K):
            self.model.addConstr(
                gp.quicksum(self.x[(k, i)] for i in range(self.I)) >= 1,
                name=f"product_plant_k{k}"
            )

        # Plant opening constraints: z_{kij} <= x_{ki}, for all k,i,j
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    self.model.addConstr(
                        self.z[(k, i, j)] <= self.x[(k, i)],
                        name=f"plant_open_k{k}_i{i}_j{j}"
                    )

        # DC opening constraints for routes: z_{kij} <= y_j, for all k,i,j
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    self.model.addConstr(
                        self.z[(k, i, j)] <= self.y[j],
                        name=f"dc_open_route_k{k}_i{i}_j{j}"
                    )

        # DC opening constraints for customers: w_jr <= y_j, for all j,r
        for j in range(self.J):
            for r in range(self.R):
                self.model.addConstr(
                    self.w[(j, r)] <= self.y[j],
                    name=f"dc_serve_j{j}_r{r}"
                )

    def add_scenario(self, scenario_id, eta_plus, eta_minus):
        """
        Add a new scenario with optimality cut and operational constraints.

        Args:
            scenario_id: Unique identifier for this scenario
            eta_plus: dict {(r,k): value} of eta^+ variables
            eta_minus: dict {(r,k): value} of eta^- variables

        Note: Per algorithm_framework.tex Line 222, ALL scenarios use the SAME beta VARIABLES.
              d_tilde_rk^(l) = Sum_m mu_rk DI_mk beta_rm + (eta+_rk^(l) - eta-_rk^(l)) mu_hat_rk
              This is endogenous demand: beta (mode choice) affects demand realization.
              Master optimizes beta jointly across all scenarios.
        """
        l = scenario_id
        self.scenarios.append((l, eta_plus, eta_minus))

        # Calculate realized demand for this scenario
        # d_rk^(l) = Sum_m mu_rk * DI_mk * beta_rm + (eta+_rk - eta-_rk) * mu_hat_rk
        # beta_rm is a VARIABLE (same beta for all scenarios)

        d_realized = {}
        for r in range(self.R):
            for k in range(self.K):
                # Endogenous nominal demand: Sum_m mu_rk * DI_mk * beta_rm (VARIABLES!)
                nominal_expr = gp.quicksum(
                    self.data.mu[(r, k)] * self.data.DI[(m, k)] * self.beta[(r, m)]
                    for m in range(self.M)
                )
                # Uncertainty deviation: (eta+_rk^(l) - eta-_rk^(l)) * mu_hat_rk
                uncertainty = (eta_plus[(r, k)] - eta_minus[(r, k)]) * self.data.mu_hat[(r, k)]
                d_realized[(r, k)] = nominal_expr + uncertainty

        print(f"    [DEBUG] Scenario {l} added (demand uses shared beta VARIABLES)")

        # Add second-stage variables for this scenario
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    self.A_ij[(k, i, j, l)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name=f"A_ij_{k}_{i}_{j}_{l}"
                    )

        for k in range(self.K):
            for j in range(self.J):
                for r in range(self.R):
                    self.A_jr[(k, j, r, l)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name=f"A_jr_{k}_{j}_{r}_{l}"
                    )

        for r in range(self.R):
            for k in range(self.K):
                self.u[(r, k, l)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"u_{r}_{k}_{l}"
                )

        # Add linearization variables X_jrm^{k(l)} = alpha_jrm * A_jr^{k(l)}
        for j in range(self.J):
            for r in range(self.R):
                for m in range(self.M):
                    for k in range(self.K):
                        self.X[(j, r, m, k, l)] = self.model.addVar(
                            vtype=GRB.CONTINUOUS, lb=0, name=f"X_{j}_{r}_{m}_{k}_{l}"
                        )

        self.model.update()

        # Add linearization constraints for X (Big-M method)
        # X_jrm^{k(l)} = alpha_jrm * A_jr^{k(l)}
        for j in range(self.J):
            M_j = self.data.MC[j]  # Big-M = DC capacity for tightness
            for r in range(self.R):
                for m in range(self.M):
                    for k in range(self.K):
                        X_var = self.X[(j, r, m, k, l)]
                        alpha_var = self.alpha[(j, r, m)]
                        A_var = self.A_jr[(k, j, r, l)]

                        # X <= M * alpha
                        self.model.addConstr(
                            X_var <= M_j * alpha_var,
                            name=f"lin1_j{j}_r{r}_m{m}_k{k}_l{l}"
                        )
                        # X <= A
                        self.model.addConstr(
                            X_var <= A_var,
                            name=f"lin2_j{j}_r{r}_m{m}_k{k}_l{l}"
                        )
                        # X >= A - M(1 - alpha)
                        self.model.addConstr(
                            X_var >= A_var - M_j * (1 - alpha_var),
                            name=f"lin3_j{j}_r{r}_m{m}_k{k}_l{l}"
                        )

        # Add operational constraints for this scenario
        # Plant capacity: Sum_j A_ij^{k(l)} <= MP_ki, for all k,i
        for k in range(self.K):
            for i in range(self.I):
                self.model.addConstr(
                    gp.quicksum(self.A_ij[(k, i, j, l)] for j in range(self.J)) <= self.data.MP[(k, i)],
                    name=f"plant_cap_k{k}_i{i}_l{l}"
                )

        # DC capacity: Sum_k Sum_i A_ij^{k(l)} <= MC_j, for all j
        for j in range(self.J):
            self.model.addConstr(
                gp.quicksum(
                    self.A_ij[(k, i, j, l)]
                    for k in range(self.K)
                    for i in range(self.I)
                ) <= self.data.MC[j],
                name=f"dc_cap_j{j}_l{l}"
            )

        # Route activation plant-to-DC: A_ij^{k(l)} <= MC_j * z_{kij}, for all k,i,j (PRODUCT-SPECIFIC)
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    self.model.addConstr(
                        self.A_ij[(k, i, j, l)] <= self.data.MC[j] * self.z[(k, i, j)],
                        name=f"route_ij_k{k}_i{i}_j{j}_l{l}"
                    )

        # Route activation DC-to-customer: A_jr^{k(l)} <= MC_j * w_jr, for all k,j,r
        for k in range(self.K):
            for j in range(self.J):
                for r in range(self.R):
                    self.model.addConstr(
                        self.A_jr[(k, j, r, l)] <= self.data.MC[j] * self.w[(j, r)],
                        name=f"route_jr_k{k}_j{j}_r{r}_l{l}"
                    )

        # Demand satisfaction: Sum_j A_jr^{k(l)} + u_rk^{(l)} = d_rk^{(l)}, for all k,r
        for k in range(self.K):
            for r in range(self.R):
                self.model.addConstr(
                    gp.quicksum(self.A_jr[(k, j, r, l)] for j in range(self.J)) + self.u[(r, k, l)]
                    == d_realized[(r, k)],
                    name=f"demand_k{k}_r{r}_l{l}"
                )

        # Flow balance at DC: Sum_i A_ij^{k(l)} = Sum_r A_jr^{k(l)}, for all k,j
        for k in range(self.K):
            for j in range(self.J):
                self.model.addConstr(
                    gp.quicksum(self.A_ij[(k, i, j, l)] for i in range(self.I))
                    == gp.quicksum(self.A_jr[(k, j, r, l)] for r in range(self.R)),
                    name=f"balance_k{k}_j{j}_l{l}"
                )

        # Add optimality cut: theta <= Revenue - HC - TC - PC - SC
        # Revenue = Sum_r Sum_k S * (d_rk - u_rk)
        revenue = gp.quicksum(
            self.data.S * (d_realized[(r, k)] - self.u[(r, k, l)])
            for r in range(self.R)
            for k in range(self.K)
        )

        # Holding cost: HC = Sum_k Sum_i Sum_j (h_j/2) * A_ij^{k(l)}
        HC = gp.quicksum(
            (self.data.h[j] / 2) * self.A_ij[(k, i, j, l)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Transportation cost (plant-to-DC): Sum_k Sum_i Sum_j D1_kij * t * A_ij^{k(l)}
        TC1 = gp.quicksum(
            self.data.D1[(k, i, j)] * self.data.t * self.A_ij[(k, i, j, l)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Transportation cost (DC-to-customer): Sum_k Sum_j Sum_r Sum_m D2_jr * TC_m * X_jrm^{k(l)}
        TC2 = gp.quicksum(
            self.data.D2[(j, r)] * self.data.TC[m] * self.X[(j, r, m, k, l)]
            for k in range(self.K)
            for j in range(self.J)
            for r in range(self.R)
            for m in range(self.M)
        )

        TC = TC1 + TC2

        # Production cost: PC = Sum_k Sum_i Sum_j F_ki * A_ij^{k(l)}
        PC = gp.quicksum(
            self.data.F[(k, i)] * self.A_ij[(k, i, j, l)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Shortage cost: SC = Sum_r Sum_k SC * u_rk^{(l)}
        SC = gp.quicksum(
            self.data.SC * self.u[(r, k, l)]
            for r in range(self.R)
            for k in range(self.K)
        )

        # Add optimality cut
        opt_cut_rhs = revenue - HC - TC - PC - SC
        self.model.addConstr(
            self.theta <= opt_cut_rhs,
            name=f"opt_cut_l{l}"
        )

        self.model.update()

        # DEBUG: Print scenario addition details
        print(f"    [DEBUG] Added scenario {l} to Master Problem:")
        print(f"    [DEBUG]   - Added {self.K * self.I * self.J} A_ij variables")
        print(f"    [DEBUG]   - Added {self.K * self.J * self.R} A_jr variables")
        print(f"    [DEBUG]   - Added {self.R * self.K} u variables")
        print(f"    [DEBUG]   - Added {self.J * self.R * self.M * self.K} X variables")
        print(f"    [DEBUG]   - Added optimality cut: theta <= [operational profit for scenario {l}]")
        print(f"    [DEBUG]   - Total constraints in model: {self.model.NumConstrs}")
        print(f"    [DEBUG]   - Total variables in model: {self.model.NumVars}")

    def solve(self):
        """Solve the master problem."""
        self.model.optimize()
        status = self.model.Status

        # Accept OPTIMAL or TIME_LIMIT with solution
        if status == GRB.OPTIMAL:
            return True
        elif status == GRB.TIME_LIMIT and self.model.SolCount > 0:
            gap_pct = self.model.MIPGap * 100
            print(f"  WARNING: Time limit reached. Using best solution found (gap: {gap_pct:.2f}%)")
            return True
        else:
            print(f"  Master Problem failed to solve! Status: {status}")
            return False

    def get_solution(self):
        """
        Extract solution from the master problem.

        Returns:
            dict with solution values
        """
        status = self.model.Status

        # Accept solution if OPTIMAL or TIME_LIMIT with solution
        if status == GRB.OPTIMAL or (status == GRB.TIME_LIMIT and self.model.SolCount > 0):
            solution = {
                'objective': self.model.ObjVal,
                'theta': self.theta.X,
                'x': {(k, i): self.x[(k, i)].X for k in range(self.K) for i in range(self.I)},
                'y': {j: self.y[j].X for j in range(self.J)},
                'z': {(k, i, j): self.z[(k, i, j)].X for k in range(self.K) for i in range(self.I) for j in range(self.J)},
                'w': {(j, r): self.w[(j, r)].X for j in range(self.J) for r in range(self.R)},
                'beta': {(r, m): self.beta[(r, m)].X for r in range(self.R) for m in range(self.M)},
                'alpha': {(j, r, m): self.alpha[(j, r, m)].X
                         for j in range(self.J) for r in range(self.R) for m in range(self.M)}
            }

            # DEBUG: Show beta values (first few)
            print(f"  [DEBUG-MASTER] Beta values from solution (showing first 3):")
            count = 0
            for key, val in solution['beta'].items():
                if val > 0.5 and count < 3:  # Only show active beta values
                    print(f"  [DEBUG-MASTER]   beta{key} = {val:.4f}")
                    count += 1

            return solution
        else:
            return None

    def apply_coverage_constraint(self, D_bar):
        """
        Apply service coverage constraint: alpha[j,r,m] = 0 if D2[j,r] > D_bar[m].

        This fixes alpha variables to 0 for DC-customer pairs where the distance
        exceeds the maximum coverage distance for that transportation mode.
        Mode 0 (slow) typically has no limit (D_bar[0] = inf).

        Args:
            D_bar: dict {m: max_distance}  -  maximum coverage distance per mode
        """
        count = 0
        for j in range(self.J):
            for r in range(self.R):
                for m in range(self.M):
                    if self.data.D2[(j, r)] > D_bar[m]:
                        self.alpha[(j, r, m)].ub = 0
                        count += 1
        print(f"  [COVERAGE] Fixed {count} alpha variables to 0 "
              f"(out of {self.J * self.R * self.M} total)")
