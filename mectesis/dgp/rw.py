"""
Random Walk Data Generating Process.
"""

import numpy as np
from .base import BaseDGP


class RandomWalk(BaseDGP):
    """
    Random Walk (I(1)) process: Y_t = drift + Y_{t-1} + eps_t

    where eps_t ~ N(0, sigma^2).

    With drift=0.0 covers Exp 1.3 (RW sin drift).
    With drift=0.5 covers Exp 1.4 (RW con drift).
    """

    def simulate(self, T: int, drift: float = 0.0, sigma: float = 1.0,
                 y0: float = 0.0) -> np.ndarray:
        """
        Simulate a Random Walk process.

        Parameters
        ----------
        T : int
            Length of the time series.
        drift : float, optional
            Deterministic drift term. Default is 0.0.
        sigma : float, optional
            Standard deviation of innovations. Default is 1.0.
        y0 : float, optional
            Initial value. Default is 0.0.

        Returns
        -------
        np.ndarray
            Simulated series of shape (T,).
        """
        y = np.empty(T)
        y[0] = y0
        eps = self.rng.normal(0.0, sigma, size=T)
        for t in range(1, T):
            y[t] = drift + y[t - 1] + eps[t]
        return y

    def get_theoretical_properties(self, drift: float = 0.0,
                                   sigma: float = 1.0) -> dict:
        return {
            "mean": "undefined (non-stationary)",
            "variance": "grows as t * sigma^2",
            "drift": drift,
            "sigma": sigma,
            "note": "I(1) process — non-stationary"
        }
