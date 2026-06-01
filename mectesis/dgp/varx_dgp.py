"""
VARX Data Generating Process — bivariate VAR with scalar exogenous covariate.

Returns {"y": (T, 2), "X": (T, 1)} so CovariateMultivariateEngine can split
the target and covariate arrays into train/test sets.
"""

import numpy as np
from .base import BaseDGP


class VARX_DGP(BaseDGP):
    """
    Bivariate VAR(1) with scalar exogenous covariate.

    Y_t = A * Y_{t-1} + gamma * X_t + eps_t,  eps_t ~ N(0, Sigma)
    X_t = rho_x * X_{t-1} + eta_t,             eta_t ~ N(0, sigma_x^2)

    Parameters
    ----------
    A : list of list, shape (2, 2)
        VAR coefficient matrix.
    gamma : list, length 2
        Effect of scalar X_t on each target variable.
    Sigma : list of list, shape (2, 2)
        Innovation covariance matrix.
    sigma_x : float
        Standard deviation of the covariate innovation.
    rho_x : float
        AR(1) persistence of the covariate (|rho_x| < 1).
    seed : int, optional
    """

    def __init__(
        self,
        A=None,
        gamma=None,
        Sigma=None,
        sigma_x: float = 1.0,
        rho_x: float = 0.7,
        seed: int = None,
        burn_in: int = 200,
    ):
        super().__init__(seed)
        self.A       = np.array(A if A is not None else [[0.5, 0.1], [0.1, 0.5]])
        self.gamma   = np.array(gamma if gamma is not None else [0.5, 0.3])
        self.Sigma   = np.array(Sigma if Sigma is not None else [[1.0, 0.3], [0.3, 1.0]])
        self.sigma_x = sigma_x
        self.rho_x   = rho_x
        self.burn_in = burn_in

    def simulate(self, T: int, **kwargs) -> dict:
        total = T + self.burn_in
        L = np.linalg.cholesky(self.Sigma)

        eta = self.rng.normal(0.0, self.sigma_x, total)
        z   = self.rng.standard_normal((total, 2))
        eps = (L @ z.T).T  # (total, 2)

        X = np.empty(total)
        Y = np.zeros((total, 2))
        X[0] = eta[0]
        Y[0] = eps[0]

        for t in range(1, total):
            X[t] = self.rho_x * X[t - 1] + eta[t]
            Y[t] = self.A @ Y[t - 1] + self.gamma * X[t] + eps[t]

        return {
            "y": Y[self.burn_in:],           # (T, 2)
            "X": X[self.burn_in:, np.newaxis],  # (T, 1)
        }

    def get_theoretical_properties(self) -> dict:
        return {
            "A": self.A.tolist(),
            "gamma": self.gamma.tolist(),
            "Sigma": self.Sigma.tolist(),
        }
