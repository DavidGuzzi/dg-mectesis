"""
ARIMAX Data Generating Processes (DGP) with exogenous covariates.

All classes return a dict {"y": np.ndarray (T,), "X": np.ndarray (T, p)}
so that downstream engines can split y and X into train/test sets and pass
X_future (the known future covariate values) to forecasting models.
"""

import numpy as np
from .base import BaseDGP


class ARIMAX_DGP(BaseDGP):
    """
    AR(1) process driven by a scalar exogenous covariate.

    Mean:     Y_t = phi * Y_{t-1} + beta * X_t + eps_t,  eps_t ~ N(0, sigma_y^2)
    Covariate: X_t = rho_x * X_{t-1} + eta_t,             eta_t ~ N(0, sigma_x^2)

    X_t is stationary (|rho_x| < 1) and assumed fully observed (past & future).
    """

    def __init__(self, seed: int = None):
        super().__init__(seed)

    def simulate(
        self,
        T: int,
        phi: float = 0.6,
        beta: float = 0.8,
        sigma_y: float = 1.0,
        sigma_x: float = 1.0,
        rho_x: float = 0.7,
        burn_in: int = 200,
    ) -> dict:
        total = T + burn_in
        eta = self.rng.normal(0.0, sigma_x, total)
        eps = self.rng.normal(0.0, sigma_y, total)

        X = np.empty(total)
        Y = np.empty(total)
        X[0] = eta[0]
        Y[0] = eps[0]
        for t in range(1, total):
            X[t] = rho_x * X[t - 1] + eta[t]
            Y[t] = phi * Y[t - 1] + beta * X[t] + eps[t]

        return {
            "y": Y[burn_in:],
            "X": X[burn_in:, np.newaxis],  # (T, 1)
        }

    def get_theoretical_properties(
        self,
        phi: float = 0.6,
        beta: float = 0.8,
        sigma_y: float = 1.0,
        sigma_x: float = 1.0,
        rho_x: float = 0.7,
    ) -> dict:
        var_x = sigma_x ** 2 / (1.0 - rho_x ** 2) if abs(rho_x) < 1.0 else np.nan
        return {
            "mean_y": 0.0,
            "ar_coef": phi,
            "exog_coef": beta,
            "var_x": var_x,
        }


class ARIMAX2Cov_DGP(BaseDGP):
    """
    AR(1) process with two independent stationary exogenous covariates.

    Y_t = phi * Y_{t-1} + beta1 * X1_t + beta2 * X2_t + eps_t
    X_i,t = rho_x * X_i,{t-1} + eta_i,t   (i = 1, 2)
    """

    def __init__(self, seed: int = None):
        super().__init__(seed)

    def simulate(
        self,
        T: int,
        phi: float = 0.6,
        beta1: float = 0.8,
        beta2: float = 0.4,
        sigma_y: float = 1.0,
        sigma_x: float = 1.0,
        rho_x: float = 0.7,
        burn_in: int = 200,
    ) -> dict:
        total = T + burn_in
        eta1 = self.rng.normal(0.0, sigma_x, total)
        eta2 = self.rng.normal(0.0, sigma_x, total)
        eps  = self.rng.normal(0.0, sigma_y, total)

        X1 = np.empty(total)
        X2 = np.empty(total)
        Y  = np.empty(total)
        X1[0] = eta1[0]
        X2[0] = eta2[0]
        Y[0]  = eps[0]
        for t in range(1, total):
            X1[t] = rho_x * X1[t - 1] + eta1[t]
            X2[t] = rho_x * X2[t - 1] + eta2[t]
            Y[t]  = phi * Y[t - 1] + beta1 * X1[t] + beta2 * X2[t] + eps[t]

        X = np.column_stack([X1[burn_in:], X2[burn_in:]])  # (T, 2)
        return {
            "y": Y[burn_in:],
            "X": X,
        }

    def get_theoretical_properties(self, **kwargs) -> dict:
        return {"mean_y": 0.0}


class ARIMAX_GARCH_DGP(BaseDGP):
    """
    AR(1)–GARCH(1,1) process where X_t enters both the mean and the variance equation.

    Mean:     Y_t = phi * Y_{t-1} + beta_mean * X_t + eps_t
    Variance: sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta_garch * sigma_{t-1}^2
                          + delta_var * X_t^2
    Covariate: X_t = rho_x * X_{t-1} + eta_t   (stationary, eta ~ N(0, sigma_x^2))
    """

    def __init__(self, seed: int = None):
        super().__init__(seed)

    def simulate(
        self,
        T: int,
        phi: float = 0.4,
        beta_mean: float = 0.5,
        omega: float = 0.1,
        alpha: float = 0.1,
        beta_garch: float = 0.75,
        delta_var: float = 0.1,
        sigma_x: float = 1.0,
        rho_x: float = 0.7,
        burn_in: int = 500,
    ) -> dict:
        total = T + burn_in
        eta = self.rng.normal(0.0, sigma_x, total)
        z   = self.rng.standard_normal(total)

        X      = np.empty(total)
        Y      = np.empty(total)
        eps    = np.empty(total)
        sigma2 = np.empty(total)

        X[0] = eta[0]
        persistence = alpha + beta_garch
        sigma2[0] = (omega + delta_var * X[0] ** 2) / (1.0 - persistence) if persistence < 1.0 else omega
        eps[0]    = np.sqrt(max(sigma2[0], 1e-10)) * z[0]
        Y[0]      = eps[0]

        for t in range(1, total):
            X[t]      = rho_x * X[t - 1] + eta[t]
            sigma2[t] = (omega
                         + alpha * eps[t - 1] ** 2
                         + beta_garch * sigma2[t - 1]
                         + delta_var * X[t] ** 2)
            sigma2[t] = max(sigma2[t], 1e-10)
            eps[t]    = np.sqrt(sigma2[t]) * z[t]
            Y[t]      = phi * Y[t - 1] + beta_mean * X[t] + eps[t]

        return {
            "y": Y[burn_in:],
            "X": X[burn_in:, np.newaxis],  # (T, 1)
        }

    def get_theoretical_properties(self, **kwargs) -> dict:
        return {"mean_y": 0.0}
