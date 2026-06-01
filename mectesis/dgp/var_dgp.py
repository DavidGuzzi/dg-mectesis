"""
VAR and VAR+GARCH-diagonal Data Generating Processes.
"""

import numpy as np
from .base import BaseDGP


class VARDGP(BaseDGP):
    """
    VAR(p) process: Y_t = A_1 Y_{t-1} + ... + A_p Y_{t-p} + eps_t,
    eps_t ~ N(0, Sigma).

    simulate() returns ndarray of shape (T, k).
    """

    def __init__(self, seed: int = None, A_list: list = None,
                 Sigma: np.ndarray = None, burn_in: int = 200):
        super().__init__(seed)
        self.A_list = [np.asarray(A, dtype=float) for A in A_list]
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.burn_in = burn_in
        self.k = self.A_list[0].shape[0]
        self.p = len(self.A_list)

    def simulate(self, T: int, **kwargs) -> np.ndarray:
        k, p = self.k, self.p
        total = T + self.burn_in
        eps = self.rng.multivariate_normal(np.zeros(k), self.Sigma, size=total)
        Y = np.zeros((total + p, k))
        for t in range(p, total + p):
            Y[t] = sum(self.A_list[lag] @ Y[t - lag - 1] for lag in range(p)) + eps[t - p]
        return Y[p + self.burn_in:]

    def get_theoretical_properties(self) -> dict:
        return {
            "k": self.k,
            "p": self.p,
            "A_list": [A.tolist() for A in self.A_list],
            "Sigma": self.Sigma.tolist(),
        }


class VARGARCHDiagonalDGP(BaseDGP):
    """
    VAR(1) for the conditional mean + diagonal GARCH(1,1) on residuals. Exp 2.6.

    Each equation i has its own GARCH(1,1):
        sigma^2_{it} = omega_i + alpha_i * u^2_{i,t-1} + beta_i * sigma^2_{i,t-1}
        u_{it} = sigma_{it} * z_{it},  z_{it} ~ N(0,1)

    simulate() returns ndarray of shape (T, k).
    """

    def __init__(self, seed: int = None, A1: np.ndarray = None,
                 omegas: list = None, alphas: list = None, betas: list = None,
                 burn_in: int = 500):
        super().__init__(seed)
        self.A1 = np.asarray(A1, dtype=float)
        self.k = self.A1.shape[0]
        self.omegas = np.asarray(omegas, dtype=float)
        self.alphas = np.asarray(alphas, dtype=float)
        self.betas = np.asarray(betas, dtype=float)
        self.burn_in = burn_in

    def simulate(self, T: int, **kwargs) -> np.ndarray:
        k = self.k
        total = T + self.burn_in
        z = self.rng.standard_normal((total, k))
        Y = np.zeros((total + 1, k))
        sig2 = self.omegas / (1.0 - self.alphas - self.betas)  # unconditional variance
        sig2 = np.maximum(sig2, 1e-8)
        u_prev = np.zeros(k)

        for t in range(total):
            sig2 = self.omegas + self.alphas * u_prev**2 + self.betas * sig2
            sig2 = np.maximum(sig2, 1e-8)
            u_t = np.sqrt(sig2) * z[t]
            Y[t + 1] = self.A1 @ Y[t] + u_t
            u_prev = u_t

        return Y[1 + self.burn_in:]

    def get_theoretical_properties(self) -> dict:
        return {
            "k": self.k,
            "A1": self.A1.tolist(),
            "omegas": self.omegas.tolist(),
            "alphas": self.alphas.tolist(),
            "betas": self.betas.tolist(),
        }
