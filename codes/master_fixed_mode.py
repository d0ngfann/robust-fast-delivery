"""
master_fixed_mode.py  -  Master Problem with Fixed Transportation Mode

Inherits from master.py (MasterProblem). Forces all customers to use a single mode
by fixing alpha[j,r,m]=0 for all m != fixed_mode.

Depends on: master.py (MasterProblem), config.py, data_gen.py
Used by: algo_fixed_mode.py (CCGAlgorithmFixedMode creates this instead of MasterProblem)
"""

from master import MasterProblem
from data_gen import SupplyChainData
from config import ProblemConfig


class MasterProblemFixedMode(MasterProblem):
    """Master Problem with a fixed transportation mode."""

    def __init__(self, data: SupplyChainData, config: ProblemConfig, fixed_mode: int):
        """
        Initialize Master Problem with fixed mode.

        Args:
            data: SupplyChainData instance with all parameters
            config: ProblemConfig instance
            fixed_mode: The transportation mode to force (0, 1, or 2)
        """
        if fixed_mode < 0 or fixed_mode >= data.M:
            raise ValueError(f"fixed_mode must be between 0 and {data.M - 1}, got {fixed_mode}")

        self.fixed_mode = fixed_mode

        # Call parent constructor (builds variables, objective, constraints)
        super().__init__(data, config)

        # Fix mode variables after parent builds them
        self._fix_mode_variables()

    def _fix_mode_variables(self):
        """
        Fix beta and alpha variables to enforce a single transportation mode.

        beta[r, m] = 1 if m == fixed_mode, else 0
        alpha[j, r, m] can only be active if m == fixed_mode
        """
        print(f"  [FIXED MODE] Forcing transportation mode {self.fixed_mode} for all customers")

        # Fix beta[r,m]: customer r uses mode m
        # Only fixed_mode can be selected
        for r in range(self.R):
            for m in range(self.M):
                if m == self.fixed_mode:
                    # This mode must be selected
                    self.beta[(r, m)].lb = 1
                    self.beta[(r, m)].ub = 1
                else:
                    # This mode cannot be selected
                    self.beta[(r, m)].lb = 0
                    self.beta[(r, m)].ub = 0

        # Fix alpha[j,r,m]: DC j serves customer r via mode m
        # Only fixed_mode can be active (alpha[j,r,fixed_mode] = w[j,r] due to constraints)
        for j in range(self.J):
            for r in range(self.R):
                for m in range(self.M):
                    if m != self.fixed_mode:
                        # Non-fixed modes cannot be used
                        self.alpha[(j, r, m)].lb = 0
                        self.alpha[(j, r, m)].ub = 0

        self.model.update()

        # Count fixed variables
        num_beta_fixed = self.R * self.M
        num_alpha_fixed = self.J * self.R * (self.M - 1)
        print(f"  [FIXED MODE] Fixed {num_beta_fixed} beta variables")
        print(f"  [FIXED MODE] Fixed {num_alpha_fixed} alpha variables to 0")
        print(f"  [FIXED MODE] Mode {self.fixed_mode} (TC={self.data.TC[self.fixed_mode]:.2f}, "
              f"DI multipliers: {[self.data.DI[(self.fixed_mode, k)] for k in range(self.K)]})")
