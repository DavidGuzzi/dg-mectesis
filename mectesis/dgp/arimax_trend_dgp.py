"""
ARIMAX with deterministic linear trend.

Y_t = alpha + delta * t + Y_t^stationary,   donde Y_t^stationary sigue
un ARIMAX(1) con covariable exogena:

    Y_t^stat = phi * Y_{t-1}^stat + beta * X_t + eps_t
    X_t      = rho_x * X_{t-1} + eta_t

La tendencia deterministica delta*t se agrega al output final (no afecta la
dinamica estacionaria interna). Esto evita que el burn-in contamine la
tendencia y permite que el modelo SARIMAX(p,0,q) con trend="ct" recupere
exactamente la especificacion.
"""

import numpy as np
from .base import BaseDGP


class ARIMAX_TREND_DGP(BaseDGP):
    """
    AR(1) + tendencia deterministica + covariable exogena.

    Y_t = alpha + delta * t + phi * Y_{t-1}^stat + beta * X_t + eps_t
    X_t = rho_x * X_{t-1} + eta_t,   |rho_x| < 1
    """

    def __init__(self, seed: int = None):
        super().__init__(seed)

    def simulate(
        self,
        T: int,
        phi: float = 0.6,
        beta: float = 0.5,
        alpha: float = 0.0,
        delta: float = 0.05,
        sigma_y: float = 1.0,
        sigma_x: float = 1.0,
        rho_x: float = 0.7,
        burn_in: int = 200,
    ) -> dict:
        total = T + burn_in
        eta = self.rng.normal(0.0, sigma_x, total)
        eps = self.rng.normal(0.0, sigma_y, total)

        X = np.empty(total)
        Y_stat = np.empty(total)
        X[0] = eta[0]
        Y_stat[0] = eps[0]
        for t in range(1, total):
            X[t]      = rho_x * X[t - 1] + eta[t]
            Y_stat[t] = phi * Y_stat[t - 1] + beta * X[t] + eps[t]

        Y = Y_stat[burn_in:] + alpha + delta * np.arange(T)
        return {
            "y": Y,
            "X": X[burn_in:, np.newaxis],  # (T, 1)
        }

    def get_theoretical_properties(
        self,
        phi: float = 0.6,
        beta: float = 0.5,
        alpha: float = 0.0,
        delta: float = 0.05,
        sigma_y: float = 1.0,
        sigma_x: float = 1.0,
        rho_x: float = 0.7,
    ) -> dict:
        return {
            "type": "ARIMAX(1) + deterministic linear trend",
            "alpha": alpha,
            "delta": delta,
            "ar_coef": phi,
            "exog_coef": beta,
            "rho_x": rho_x,
        }
