"""
sub.py  -  Subproblem (SP-Dual) Formulation

Finds worst-case demand scenario (eta+, eta-) given fixed first-stage decisions.
Uses dual reformulation + McCormick linearization for bilinear terms.

Depends on: config.py, data_gen.py (SupplyChainData)
Used by: algo.py (CCGAlgorithm feeds MP solution here, gets worst-case scenario back)

Key flow:
  1. Receives fixed (x, y, z, w, beta, alpha) from Master Problem solution
  2. Maximizes adversarial profit via dual formulation
  3. Returns worst-case (eta_plus, eta_minus) and Z_SP value
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from data_gen import SupplyChainData
from config import ProblemConfig


class Subproblem:
    """Subproblem for identifying worst-case demand scenarios."""

    def __init__(self, data: SupplyChainData, config: ProblemConfig):
        """
        Initialize Subproblem.

        Args:
            data: SupplyChainData instance
            config: ProblemConfig instance with Gamma set
        """
        self.data = data
        self.config = config
        self.K = data.K
        self.I = data.I
        self.J = data.J
        self.R = data.R
        self.M = data.M

        if config.Gamma is None:
            raise ValueError("Gamma must be set in config before creating Subproblem")
        self.Gamma = config.Gamma

        # Gurobi model
        self.model = gp.Model("Subproblem")
        self._configure_solver()

        # Decision variables
        # Dual variables
        self.pi = {}  # Plant capacity dual: pi[k,i]
        self.sigma = {}  # DC capacity dual: sigma[j]
        self.psi = {}  # Route plant-DC dual: psi[k,i,j]
        self.phi = {}  # Route DC-customer dual: phi[k,j,r]
        self.gamma = {}  # Demand satisfaction dual: gamma[r,k]
        self.kappa = {}  # Flow balance dual: kappa[k,j]

        # Uncertainty variables
        self.eta_plus = {}  # Demand increase: eta_plus[r,k]
        self.eta_minus = {}  # Demand decrease: eta_minus[r,k]

        # McCormick linearization variables
        self.p_plus = {}  # p_plus[r,k] = eta_plus[r,k] * gamma[r,k]
        self.p_minus = {}  # p_minus[r,k] = eta_minus[r,k] * gamma[r,k]
        # xi[r,k] = p_plus[r,k] - p_minus[r,k]

        # Fixed first-stage solution (will be set before solving)
        self.fixed_x = None
        self.fixed_y = None
        self.fixed_z = None
        self.fixed_w = None
        self.fixed_beta = None
        self.fixed_alpha = None

        # Track constraints that depend on fixed first-stage variables
        # These will be removed and re-added when first-stage solution changes
        self.dual_feasibility_constrs = []

        # Build model
        self._build_variables()
        self._build_constraints()
        # Objective and dual feasibility will be built after fixing first-stage variables

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
        """Create decision variables."""
        # Dual variables
        # pi[k,i]: Plant capacity dual (>= 0)
        for k in range(self.K):
            for i in range(self.I):
                self.pi[(k, i)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"pi_{k}_{i}"
                )

        # sigma[j]: DC capacity dual (>= 0)
        for j in range(self.J):
            self.sigma[j] = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=f"sigma_{j}"
            )

        # psi[k,i,j]: Route plant-DC dual (>= 0)
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    self.psi[(k, i, j)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name=f"psi_{k}_{i}_{j}"
                    )

        # phi[k,j,r]: Route DC-customer dual (>= 0)
        for k in range(self.K):
            for j in range(self.J):
                for r in range(self.R):
                    self.phi[(k, j, r)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, name=f"phi_{k}_{j}_{r}"
                    )

        # gamma[r,k]: Demand satisfaction dual (unrestricted, but bounded)
        # Bounds: gamma^L = -(S + SC), gamma^U = S
        gamma_L = -(self.data.S + self.data.SC)
        gamma_U = self.data.S
        for r in range(self.R):
            for k in range(self.K):
                self.gamma[(r, k)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=gamma_L, ub=gamma_U, name=f"gamma_{r}_{k}"
                )

        # kappa[k,j]: Flow balance dual (unrestricted)
        for k in range(self.K):
            for j in range(self.J):
                self.kappa[(k, j)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"kappa_{k}_{j}"
                )

        # Uncertainty variables (binary)
        for r in range(self.R):
            for k in range(self.K):
                self.eta_plus[(r, k)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"eta_plus_{r}_{k}"
                )
                self.eta_minus[(r, k)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"eta_minus_{r}_{k}"
                )

        # McCormick linearization variables
        # Bounds for p: based on gamma bounds and binary eta
        # p_plus, p_minus  in  [gamma_L, gamma_U] when eta = 1, else 0
        for r in range(self.R):
            for k in range(self.K):
                self.p_plus[(r, k)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=gamma_L, ub=gamma_U, name=f"p_plus_{r}_{k}"
                )
                self.p_minus[(r, k)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=gamma_L, ub=gamma_U, name=f"p_minus_{r}_{k}"
                )

    def _build_constraints(self):
        """Build constraints (before fixing first-stage variables)."""
        # Uncertainty set constraints
        # Budget constraint: Sum_r (eta_plus + eta_minus) <= Gamma, for all k
        for k in range(self.K):
            self.model.addConstr(
                gp.quicksum(
                    self.eta_plus[(r, k)] + self.eta_minus[(r, k)]
                    for r in range(self.R)
                ) <= self.Gamma,
                name=f"budget_k{k}"
            )

        # Mutual exclusivity: eta_plus + eta_minus <= 1, for all r,k
        for r in range(self.R):
            for k in range(self.K):
                self.model.addConstr(
                    self.eta_plus[(r, k)] + self.eta_minus[(r, k)] <= 1,
                    name=f"mutex_r{r}_k{k}"
                )

        # McCormick linearization for p_plus = eta_plus * gamma
        gamma_L = -(self.data.S + self.data.SC)
        gamma_U = self.data.S

        for r in range(self.R):
            for k in range(self.K):
                eta_p = self.eta_plus[(r, k)]
                gamma_var = self.gamma[(r, k)]
                p_p = self.p_plus[(r, k)]

                # p_plus >= gamma_L * eta_plus
                self.model.addConstr(
                    p_p >= gamma_L * eta_p,
                    name=f"mc_pp1_r{r}_k{k}"
                )
                # p_plus <= gamma_U * eta_plus
                self.model.addConstr(
                    p_p <= gamma_U * eta_p,
                    name=f"mc_pp2_r{r}_k{k}"
                )
                # p_plus >= gamma - gamma_U * (1 - eta_plus)
                self.model.addConstr(
                    p_p >= gamma_var - gamma_U * (1 - eta_p),
                    name=f"mc_pp3_r{r}_k{k}"
                )
                # p_plus <= gamma - gamma_L * (1 - eta_plus)
                self.model.addConstr(
                    p_p <= gamma_var - gamma_L * (1 - eta_p),
                    name=f"mc_pp4_r{r}_k{k}"
                )

        # McCormick linearization for p_minus = eta_minus * gamma
        for r in range(self.R):
            for k in range(self.K):
                eta_m = self.eta_minus[(r, k)]
                gamma_var = self.gamma[(r, k)]
                p_m = self.p_minus[(r, k)]

                # p_minus >= gamma_L * eta_minus
                self.model.addConstr(
                    p_m >= gamma_L * eta_m,
                    name=f"mc_pm1_r{r}_k{k}"
                )
                # p_minus <= gamma_U * eta_minus
                self.model.addConstr(
                    p_m <= gamma_U * eta_m,
                    name=f"mc_pm2_r{r}_k{k}"
                )
                # p_minus >= gamma - gamma_U * (1 - eta_minus)
                self.model.addConstr(
                    p_m >= gamma_var - gamma_U * (1 - eta_m),
                    name=f"mc_pm3_r{r}_k{k}"
                )
                # p_minus <= gamma - gamma_L * (1 - eta_minus)
                self.model.addConstr(
                    p_m <= gamma_var - gamma_L * (1 - eta_m),
                    name=f"mc_pm4_r{r}_k{k}"
                )

    def fix_first_stage(self, solution):
        """
        Fix first-stage variables from Master Problem solution.

        Args:
            solution: dict with keys 'x', 'y', 'z', 'w', 'beta', 'alpha'

        Note: This method is called every iteration with potentially different first-stage values.
        We must remove old dual feasibility constraints (which depend on fixed_alpha)
        and rebuild them with the new fixed values.
        """
        # DEBUG: Check if beta values have changed
        beta_changed = False
        if self.fixed_beta is not None:
            for key in solution['beta']:
                if abs(solution['beta'][key] - self.fixed_beta[key]) > 1e-6:
                    beta_changed = True
                    break

        print(f"\n  [DEBUG-SUBPROBLEM] fix_first_stage() called")
        if self.fixed_beta is not None:
            print(f"  [DEBUG-SUBPROBLEM] Beta values changed: {beta_changed}")
            if beta_changed:
                # Show a few changed beta values
                count = 0
                for key in solution['beta']:
                    if abs(solution['beta'][key] - self.fixed_beta[key]) > 1e-6 and count < 3:
                        print(f"  [DEBUG-SUBPROBLEM]   beta{key}: {self.fixed_beta[key]:.4f} -> {solution['beta'][key]:.4f}")
                        count += 1
        else:
            print(f"  [DEBUG-SUBPROBLEM] First call (no previous beta)")

        self.fixed_x = solution['x']
        self.fixed_y = solution['y']
        self.fixed_z = solution['z']
        self.fixed_w = solution['w']
        self.fixed_beta = solution['beta']
        self.fixed_alpha = solution['alpha']

        # Remove old dual feasibility constraints (they depend on fixed_alpha)
        if self.dual_feasibility_constrs:
            num_old_constrs = len(self.dual_feasibility_constrs)
            print(f"  [DEBUG-SUBPROBLEM] Removing {num_old_constrs} old dual feasibility constraints")
            for constr in self.dual_feasibility_constrs:
                self.model.remove(constr)
            self.dual_feasibility_constrs = []
            self.model.update()

        # Add new dual feasibility constraints with updated fixed values
        print(f"  [DEBUG-SUBPROBLEM] Adding new dual feasibility constraints")
        self._build_dual_feasibility()
        num_new_constrs = len(self.dual_feasibility_constrs)
        print(f"  [DEBUG-SUBPROBLEM] Added {num_new_constrs} new dual feasibility constraints")

        # Update objective (depends on fixed_beta)
        print(f"  [DEBUG-SUBPROBLEM] Updating objective function")
        self._build_objective()

        self.model.update()

        # DEBUG: Show model fingerprint
        print(f"  [DEBUG-SUBPROBLEM] Model fingerprint after update: {hex(self.model.fingerprint)}")

    def _build_dual_feasibility(self):
        """
        Build dual feasibility constraints with fixed first-stage variables.

        Note: These constraints depend on fixed_alpha, so they must be rebuilt
        whenever the first-stage solution changes.
        """
        # Dual feasibility for A_ij^k:
        # pi[k,i] + sigma[j] + psi[k,i,j] + kappa[k,j] >= -h_j/2 - D1[k,i,j]*t - F[k,i]
        for k in range(self.K):
            for i in range(self.I):
                for j in range(self.J):
                    rhs = -(self.data.h[j] / 2) - self.data.D1[(k, i, j)] * self.data.t - self.data.F[(k, i)]
                    constr = self.model.addConstr(
                        self.pi[(k, i)] + self.sigma[j] + self.psi[(k, i, j)] + self.kappa[(k, j)] >= rhs,
                        name=f"dual_Aij_k{k}_i{i}_j{j}"
                    )
                    self.dual_feasibility_constrs.append(constr)

        # Dual feasibility for A_jr^k:
        # phi[k,j,r] + gamma[r,k] - kappa[k,j] >= -Sum_m D2[j,r] * TC[m] * alpha[j,r,m]
        # This RHS depends on fixed_alpha, so it changes when alpha changes
        for k in range(self.K):
            for j in range(self.J):
                for r in range(self.R):
                    rhs = -sum(
                        self.data.D2[(j, r)] * self.data.TC[m] * self.fixed_alpha[(j, r, m)]
                        for m in range(self.M)
                    )
                    constr = self.model.addConstr(
                        self.phi[(k, j, r)] + self.gamma[(r, k)] - self.kappa[(k, j)] >= rhs,
                        name=f"dual_Ajr_k{k}_j{j}_r{r}"
                    )
                    self.dual_feasibility_constrs.append(constr)

        # Dual feasibility for u_rk:
        # gamma[r,k] >= -(S + SC)
        # This is already enforced by variable bounds, but we can add explicit constraint for clarity
        for r in range(self.R):
            for k in range(self.K):
                constr = self.model.addConstr(
                    self.gamma[(r, k)] >= -(self.data.S + self.data.SC),
                    name=f"dual_u_r{r}_k{k}"
                )
                self.dual_feasibility_constrs.append(constr)

    def _build_objective(self):
        """Build objective function (dual objective to minimize)."""
        # Dual objective: minimize
        # Sum_k Sum_i MP[k,i] * pi[k,i]
        # + Sum_j MC[j] * sigma[j]
        # + Sum_k Sum_i Sum_j MC[j] * z_ij * psi[k,i,j]
        # + Sum_k Sum_j Sum_r MC[j] * w_jr * phi[k,j,r]
        # + Sum_r Sum_k (Sum_m mu[r,k] * DI[m,k] * beta[r,m]) * gamma[r,k]
        # + Sum_r Sum_k mu_hat[r,k] * (p_plus[r,k] - p_minus[r,k])

        obj = 0

        # Plant capacity term
        obj += gp.quicksum(
            self.data.MP[(k, i)] * self.pi[(k, i)]
            for k in range(self.K)
            for i in range(self.I)
        )

        # DC capacity term
        obj += gp.quicksum(
            self.data.MC[j] * self.sigma[j]
            for j in range(self.J)
        )

        # Route plant-DC term (PRODUCT-SPECIFIC z_{kij})
        obj += gp.quicksum(
            self.data.MC[j] * self.fixed_z[(k, i, j)] * self.psi[(k, i, j)]
            for k in range(self.K)
            for i in range(self.I)
            for j in range(self.J)
        )

        # Route DC-customer term
        obj += gp.quicksum(
            self.data.MC[j] * self.fixed_w[(j, r)] * self.phi[(k, j, r)]
            for k in range(self.K)
            for j in range(self.J)
            for r in range(self.R)
        )

        # Nominal demand term
        obj += gp.quicksum(
            sum(self.data.mu[(r, k)] * self.data.DI[(m, k)] * self.fixed_beta[(r, m)]
                for m in range(self.M)) * self.gamma[(r, k)]
            for r in range(self.R)
            for k in range(self.K)
        )

        # Uncertainty term (linearized): mu_hat_rk x xi_rk where xi = (eta+-eta-)xgamma
        obj += gp.quicksum(
            self.data.mu_hat[(r, k)] * (self.p_plus[(r, k)] - self.p_minus[(r, k)])
            for r in range(self.R)
            for k in range(self.K)
        )

        # CRITICAL: Add Sxmu_hatx(eta+-eta-) term for correct minimization over eta
        # Subproblem minimizes: Sxd(eta) + dual_obj(eta)
        # where d(eta) = d_nominal + (eta+-eta-)xmu_hat
        # So we need to minimize: Sx(eta+-eta-)xmu_hat + [capacity_terms + dxgamma]
        # The Sx(eta+-eta-)xmu_hat term must be added to the objective!
        obj += gp.quicksum(
            self.data.S * self.data.mu_hat[(r, k)] * (self.eta_plus[(r, k)] - self.eta_minus[(r, k)])
            for r in range(self.R)
            for k in range(self.K)
        )

        self.model.setObjective(obj, GRB.MINIMIZE)

    def solve(self):
        """
        Solve the subproblem.

        Returns:
            bool: True if solved successfully (OPTIMAL or TIME_LIMIT with solution)
        """
        self.model.optimize()
        status = self.model.Status

        # Accept OPTIMAL solution
        if status == GRB.OPTIMAL:
            return True

        # Accept TIME_LIMIT with at least one solution found
        # WARNING: This solution may be suboptimal
        # Since we are MINIMIZING, suboptimal Z_SP may be LARGER than true minimum
        # This causes LB to be over-estimated, potentially leading to negative gap
        elif status == GRB.TIME_LIMIT and self.model.SolCount > 0:
            mip_gap = self.model.MIPGap * 100
            print(f"  [!]  WARNING: Subproblem time limit reached!")
            print(f"  MIP Gap: {mip_gap:.4f}%")
            print(f"  Using best solution found (may be suboptimal)")
            print(f"  Note: Z_SP may be over-estimated (minimization problem)")
            return True

        # Failed to solve
        else:
            return False

    def get_worst_case_scenario(self):
        """
        Extract worst-case scenario from subproblem solution.

        Returns:
            tuple: (Z_SP, eta_plus, eta_minus)
                Z_SP: Worst-case operational profit
                eta_plus: dict {(r,k): value}
                eta_minus: dict {(r,k): value}
        """
        # Check if we have a valid solution (OPTIMAL or TIME_LIMIT with solution)
        status = self.model.Status
        if status != GRB.OPTIMAL and not (status == GRB.TIME_LIMIT and self.model.SolCount > 0):
            return None, None, None

        # Extract eta values
        eta_plus = {}
        eta_minus = {}
        for r in range(self.R):
            for k in range(self.K):
                eta_plus[(r, k)] = round(self.eta_plus[(r, k)].X)
                eta_minus[(r, k)] = round(self.eta_minus[(r, k)].X)

        # Z_SP calculation after adding Sxmu_hatx(eta+-eta-) to objective:
        # model.ObjVal now includes:
        #   capacity_terms + d_nominalxgamma + mu_hatx(eta+-eta-)xgamma + Sxmu_hatx(eta+-eta-)
        # Therefore: Z_SP = Sxd_nominal + model.ObjVal

        dual_obj = self.model.ObjVal

        # Calculate Sxd_nominal (constant part only)
        revenue_S_times_d_nominal = 0.0
        for r in range(self.R):
            for k in range(self.K):
                d_nominal = sum(
                    self.data.mu[(r, k)] * self.data.DI[(m, k)] * self.fixed_beta[(r, m)]
                    for m in range(self.M)
                )
                revenue_S_times_d_nominal += self.data.S * d_nominal

        # Z_SP = Sxd_nominal + model.ObjVal
        # (model.ObjVal already includes Sxmu_hatx(eta+-eta-) term now)
        Z_SP = revenue_S_times_d_nominal + dual_obj

        return Z_SP, eta_plus, eta_minus

    def get_dual_solution(self):
        """Get dual variable values (for debugging/analysis)."""
        status = self.model.Status
        if status != GRB.OPTIMAL and not (status == GRB.TIME_LIMIT and self.model.SolCount > 0):
            return None

        dual_sol = {
            'pi': {(k, i): self.pi[(k, i)].X for k in range(self.K) for i in range(self.I)},
            'sigma': {j: self.sigma[j].X for j in range(self.J)},
            'psi': {(k, i, j): self.psi[(k, i, j)].X
                    for k in range(self.K) for i in range(self.I) for j in range(self.J)},
            'phi': {(k, j, r): self.phi[(k, j, r)].X
                    for k in range(self.K) for j in range(self.J) for r in range(self.R)},
            'gamma': {(r, k): self.gamma[(r, k)].X for r in range(self.R) for k in range(self.K)},
            'kappa': {(k, j): self.kappa[(k, j)].X for k in range(self.K) for j in range(self.J)}
        }

        return dual_sol
