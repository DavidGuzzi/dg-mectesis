"""
VECM (Vector Error Correction Model) Data Generating Process. Exp 2.7.
"""

import numpy as np
from .base import BaseDGP


class VECMBivariateDGP(BaseDGP):
    """
    Bivariate VECM with cointegration rank 1:
        Delta Y_t = alpha * beta' * Y_{t-1} + Gamma1 * Delta Y_{t-1} + eps_t

    Default parameters (matching experiments.md Exp 2.7):
        beta  = (1, -1)'       cointegration vector
        alpha = (-0.4, 0.2)'   adjustment speeds
        Gamma1 = diag(0.3, 0.3) short-run dynamics
        Sigma = I_2            innovation covariance

    Each individual series is I(1); the combination Y1 - Y2 is I(0).

    simulate() returns ndarray of shape (T, 2).
    """

    def __init__(self, seed: int = None,
                 alpha: list = None, beta: list = None,
                 Gamma1: list = None, Sigma: list = None,
                 burn_in: int = 200):
        super().__init__(seed)
        self.alpha = np.asarray(alpha if alpha is not None else [-0.4, 0.2], dtype=float)
        self.beta = np.asarray(beta if beta is not None else [1.0, -1.0], dtype=float)
        self.Gamma1 = np.asarray(
            Gamma1 if Gamma1 is not None else [[0.3, 0.0], [0.0, 0.3]], dtype=float
        )
        self.Sigma = np.asarray(Sigma if Sigma is not None else [[1.0, 0.0], [0.0, 1.0]], dtype=float)
        self.burn_in = burn_in

    def simulate(self, T: int, **kwargs) -> np.ndarray:
        total = T + self.burn_in
        eps = self.rng.multivariate_normal(np.zeros(2), self.Sigma, size=total)

        # We need Y_{t-1} and Delta Y_{t-1}
        Y = np.zeros((total + 2, 2))  # Y[0] and Y[1] are initial zeros

        for t in range(2, total + 2):
            ecm_term = self.alpha * (self.beta @ Y[t - 1])       # (2,)
            dyn_term = self.Gamma1 @ (Y[t - 1] - Y[t - 2])      # (2,)
            Y[t] = Y[t - 1] + ecm_term + dyn_term + eps[t - 2]

        return Y[2 + self.burn_in:]

    def get_theoretical_properties(self) -> dict:
        return {
            "k": 2,
            "coint_rank": 1,
            "beta": self.beta.tolist(),
            "alpha": self.alpha.tolist(),
            "Gamma1": self.Gamma1.tolist(),
            "Sigma": self.Sigma.tolist(),
        }
