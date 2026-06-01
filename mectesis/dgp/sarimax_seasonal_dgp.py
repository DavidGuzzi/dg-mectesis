"""
SARIMAX seasonal data generating process.

Forma multiplicativa SARIMA(1, 0, 0)(1, 0, 0)[s] con covariable:

    (1 - phi * L)(1 - Phi * L^s) Y_t = beta * X_t + eps_t

Expansion:

    Y_t = phi * Y_{t-1} + Phi * Y_{t-s} - phi * Phi * Y_{t-s-1}
          + beta * X_t + eps_t

X_t es una covariable estacionaria AR(1):

    X_t = rho_x * X_{t-1} + eta_t

Esta especificacion es exactamente recuperada por el modelo
`SARIMAXModel(order=(1,0,0), seasonal_order=(1,0,0,s))` con
`trend="c"`. Para estacionariedad necesitamos |phi| < 1 y |Phi| < 1.
"""

import numpy as np
from .base import BaseDGP


class SARIMAX_SEASONAL_DGP(BaseDGP):
    """
    SARIMA(1,0,0)(1,0,0)[s] estacional con covariable exogena.

    Soporta s=4 (trimestral) y s=12 (mensual).
    """

    def __init__(self, seed: int = None):
        super().__init__(seed)

    def simulate(
        self,
        T: int,
        s: int = 4,
        phi: float = 0.5,
        Phi: float = 0.5,
        beta: float = 0.5,
        sigma_y: float = 1.0,
        sigma_x: float = 1.0,
        rho_x: float = 0.7,
        burn_in: int = None,
    ) -> dict:
        if burn_in is None:
            burn_in = max(200, 6 * s)
        total = T + burn_in
        eta = self.rng.normal(0.0, sigma_x, total)
        eps = self.rng.normal(0.0, sigma_y, total)

        X = np.empty(total)
        Y = np.zeros(total)
        X[0] = eta[0]
        Y[0] = beta * X[0] + eps[0]

        for t in range(1, total):
            X[t] = rho_x * X[t - 1] + eta[t]
            if t > s:
                Y[t] = (phi * Y[t - 1]
                        + Phi * Y[t - s]
                        - phi * Phi * Y[t - s - 1]
                        + beta * X[t] + eps[t])
            elif t == s:
                Y[t] = phi * Y[t - 1] + Phi * Y[0] + beta * X[t] + eps[t]
            else:
                Y[t] = phi * Y[t - 1] + beta * X[t] + eps[t]

        return {
            "y": Y[burn_in:],
            "X": X[burn_in:, np.newaxis],  # (T, 1)
        }

    def get_theoretical_properties(
        self,
        s: int = 4,
        phi: float = 0.5,
        Phi: float = 0.5,
        beta: float = 0.5,
        sigma_y: float = 1.0,
        sigma_x: float = 1.0,
        rho_x: float = 0.7,
    ) -> dict:
        return {
            "type": f"SARIMAX(1,0,0)(1,0,0)[{s}]",
            "s": s,
            "phi": phi,
            "Phi": Phi,
            "exog_coef": beta,
            "rho_x": rho_x,
        }
