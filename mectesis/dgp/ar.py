"""
AR(1) Data Generating Process.
"""

import numpy as np
from .base import BaseDGP


class AR1(BaseDGP):
    """
    AR(1) process: y_t = mu + phi * (y_{t-1} - mu) + eps_t

    where eps_t ~ N(0, sigma^2)
    """

    def simulate(self, T: int, phi: float, mu: float = 0.0,
                 sigma: float = 1.0, burn_in: int = 200) -> np.ndarray:
        """
        Simulate an AR(1) process.

        Parameters
        ----------
        T : int
            Length of the time series (after burn-in).
        phi : float
            Autoregressive coefficient. Must satisfy |phi| < 1 for stationarity.
        mu : float, optional
            Mean of the process. Default is 0.0.
        sigma : float, optional
            Standard deviation of the innovation term. Default is 1.0.
        burn_in : int, optional
            Number of initial observations to discard to ensure stationarity.
            Default is 200.

        Returns
        -------
        np.ndarray
            Simulated AR(1) time series of shape (T,).

        Notes
        -----
        The process is initialized at y_0 = mu and evolved for T + burn_in steps.
        The first burn_in observations are discarded to mitigate initialization bias.
        """
        total_T = T + burn_in
        y = np.empty(total_T)
        y[0] = mu

        for t in range(1, total_T):
            eps = self.rng.normal(0.0, sigma)
            y[t] = mu + phi * (y[t-1] - mu) + eps

        return y[burn_in:]

    def get_theoretical_properties(self, phi: float = None, mu: float = None,
                                   sigma: float = None) -> dict:
        """
        Return theoretical properties of the AR(1) process.

        Parameters
        ----------
        phi : float, optional
            Autoregressive coefficient.
        mu : float, optional
            Mean of the process.
        sigma : float, optional
            Standard deviation of innovations.

        Returns
        -------
        dict
            Dictionary containing:
            - mean: E[y_t] = mu
            - variance: Var(y_t) = sigma^2 / (1 - phi^2)
            - autocorrelation: rho(k) = phi^k

        Notes
        -----
        Properties are only defined for stationary processes (|phi| < 1).
        """
        if phi is None or abs(phi) >= 1:
            return {
                "mean": mu if mu is not None else np.nan,
                "variance": np.nan,
                "note": "Process is non-stationary (|phi| >= 1)"
            }

        variance = (sigma ** 2) / (1 - phi ** 2) if sigma is not None else np.nan

        return {
            "mean": mu if mu is not None else 0.0,
            "variance": variance,
            "autocorrelation": lambda k: phi ** k,
            "phi": phi,
            "sigma": sigma
        }
