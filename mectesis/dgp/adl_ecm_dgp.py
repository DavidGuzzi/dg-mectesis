"""
ADL-ECM Data Generating Process — cointegrated Y and X.

X_t ~ I(1) (random walk).
Y_t - X_t ~ I(0) → cointegration vector (1, -1).

Error-correction representation:
    ΔY_t = alpha_ecm * (Y_{t-1} - X_{t-1}) + ΔX_t + eta_t

Returns {"y": (T,), "X": (T, 1)}.
"""

import numpy as np
from .base import BaseDGP


class ADL_ECM_DGP(BaseDGP):
    """
    Cointegrated bivariate system where X is the I(1) covariate.

    DGP:
        X_t = X_{t-1} + u_t,           u_t ~ N(0, sigma_x^2)
        Y_t = Y_{t-1} + alpha_ecm * (Y_{t-1} - X_{t-1}) + (X_t - X_{t-1}) + eta_t

    Equivalent VECM form (cointegration vector beta=(1,-1)):
        ΔY_t = alpha_ecm * (Y_{t-1} - X_{t-1}) + ΔX_t + eta_t

    Parameters
    ----------
    alpha_ecm : float
        Error-correction speed of adjustment. Should be negative (e.g. -0.3)
        so that deviations from Y = X are corrected.
    sigma : float
        Standard deviation of the idiosyncratic innovation eta_t.
    sigma_x : float
        Standard deviation of the random walk innovation u_t.
    seed : int, optional
    """

    def __init__(self, seed: int = None):
        super().__init__(seed)

    def simulate(
        self,
        T: int,
        alpha_ecm: float = -0.3,
        sigma: float = 1.0,
        sigma_x: float = 1.0,
        burn_in: int = 50,
    ) -> dict:
        total = T + burn_in
        u   = self.rng.normal(0.0, sigma_x, total)
        eta = self.rng.normal(0.0, sigma, total)

        X = np.empty(total)
        Y = np.empty(total)
        X[0] = u[0]
        Y[0] = eta[0]

        for t in range(1, total):
            X[t] = X[t - 1] + u[t]
            ecm  = Y[t - 1] - X[t - 1]
            dX   = X[t] - X[t - 1]
            Y[t] = Y[t - 1] + alpha_ecm * ecm + dX + eta[t]

        return {
            "y": Y[burn_in:],
            "X": X[burn_in:, np.newaxis],  # (T, 1)
        }

    def get_theoretical_properties(
        self,
        alpha_ecm: float = -0.3,
        sigma: float = 1.0,
        sigma_x: float = 1.0,
    ) -> dict:
        return {
            "cointegration_vector": [1.0, -1.0],
            "ecm_speed": alpha_ecm,
            "integrated_order": 1,
        }
