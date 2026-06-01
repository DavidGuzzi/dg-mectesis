"""
AR(1) with Deterministic Trend Data Generating Process.
"""

import numpy as np
from .base import BaseDGP


class AR1WithTrend(BaseDGP):
    """
    AR(1) with deterministic trend: Y_t = intercept + trend_coeff * t + phi * Y_{t-1} + eps_t

    Covers Exp 1.5 with intercept=5, trend_coeff=0.1, phi=0.6.
    """

    def simulate(self, T: int, intercept: float = 5.0, trend_coeff: float = 0.1,
                 phi: float = 0.6, sigma: float = 1.0, burn_in: int = 50) -> np.ndarray:
        """
        Simulate AR(1) with deterministic linear trend.

        Parameters
        ----------
        T : int
            Length of the returned series.
        intercept : float, optional
            Constant term. Default is 5.0.
        trend_coeff : float, optional
            Slope of the deterministic trend. Default is 0.1.
        phi : float, optional
            AR(1) coefficient. Default is 0.6.
        sigma : float, optional
            Standard deviation of innovations. Default is 1.0.
        burn_in : int, optional
            Burn-in periods to stabilize the AR component. Default is 50.

        Returns
        -------
        np.ndarray
            Simulated series of shape (T,) where time indices are t = 1, ..., T.

        Notes
        -----
        During burn-in, t_actual = t - burn_in is negative, which helps anchor the
        AR component near the trend before the returned portion begins.
        """
        total_T = T + burn_in
        y = np.empty(total_T)
        y[0] = 0.0
        eps = self.rng.normal(0.0, sigma, size=total_T)

        for t in range(1, total_T):
            t_actual = t - burn_in  # negative during burn-in, 1..T after
            y[t] = intercept + trend_coeff * t_actual + phi * y[t - 1] + eps[t]

        return y[burn_in:]

    def get_theoretical_properties(self, intercept: float = 5.0,
                                   trend_coeff: float = 0.1,
                                   phi: float = 0.6,
                                   sigma: float = 1.0) -> dict:
        return {
            "type": "trend-stationary AR(1)",
            "phi": phi,
            "intercept": intercept,
            "trend_coeff": trend_coeff,
            "sigma": sigma,
            "note": "Stationary around deterministic trend when |phi| < 1"
        }
