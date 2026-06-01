"""
Seasonal Data Generating Processes.
"""

import numpy as np
from .base import BaseDGP


class SeasonalDGP(BaseDGP):
    """
    Seasonal AR DGP supporting two modes:

    - integrated=False: (1 - phi*L)(1 - Phi*L^s) Y_t = eps_t
        Expanding: Y_t = phi*Y_{t-1} + Phi*Y_{t-s} - phi*Phi*Y_{t-s-1} + eps_t
        Covers Exp 1.6 (quarterly, s=4, phi=0.5, Phi=0.3).

    - integrated=True: (1 - L)(1 - L^s) Y_t = eps_t
        Expanding: Y_t = Y_{t-1} + Y_{t-s} - Y_{t-s-1} + eps_t
        Covers Exp 1.7 (monthly, s=12).
    """

    def simulate(self, T: int, phi: float = 0.5, Phi: float = 0.3,
                 s: int = 4, integrated: bool = False,
                 sigma: float = 1.0, burn_in: int = 200) -> np.ndarray:
        """
        Simulate seasonal process.

        Parameters
        ----------
        T : int
            Length of the returned series.
        phi : float, optional
            Non-seasonal AR coefficient (used only when integrated=False). Default 0.5.
        Phi : float, optional
            Seasonal AR coefficient (used only when integrated=False). Default 0.3.
        s : int, optional
            Seasonal period. Default is 4 (quarterly).
        integrated : bool, optional
            If True, use doubly integrated form (1-L)(1-L^s)Y_t = eps_t.
            If False, use stationary SAR form. Default is False.
        sigma : float, optional
            Standard deviation of innovations. Default is 1.0.
        burn_in : int, optional
            Burn-in periods (only relevant for stationary form). Default is 200.

        Returns
        -------
        np.ndarray
            Simulated series of shape (T,).
        """
        if integrated:
            return self._simulate_integrated(T, s, sigma)
        else:
            return self._simulate_stationary(T, phi, Phi, s, sigma, burn_in)

    def _simulate_stationary(self, T: int, phi: float, Phi: float,
                              s: int, sigma: float, burn_in: int) -> np.ndarray:
        total_T = T + burn_in
        y = np.zeros(total_T)
        eps = self.rng.normal(0.0, sigma, size=total_T)

        # Need s+1 lags: start from max(1, s+1)
        start = s + 1
        for t in range(start, total_T):
            y[t] = (phi * y[t - 1]
                    + Phi * y[t - s]
                    - phi * Phi * y[t - s - 1]
                    + eps[t])

        return y[burn_in:]

    def _simulate_integrated(self, T: int, s: int, sigma: float) -> np.ndarray:
        # (1-L)(1-L^s)Y_t = eps_t → Y_t = Y_{t-1} + Y_{t-s} - Y_{t-s-1} + eps_t
        # Non-stationary: no burn-in needed, just enough lags
        n = T + s + 1
        y = np.zeros(n)
        eps = self.rng.normal(0.0, sigma, size=n)

        start = s + 1
        for t in range(start, n):
            y[t] = y[t - 1] + y[t - s] - y[t - s - 1] + eps[t]

        return y[s + 1:]  # drop the initial zero-padding

    def get_theoretical_properties(self, phi: float = 0.5, Phi: float = 0.3,
                                   s: int = 4, integrated: bool = False,
                                   sigma: float = 1.0) -> dict:
        if integrated:
            return {
                "type": f"Doubly integrated seasonal: (1-L)(1-L^{s})Y_t = eps_t",
                "s": s,
                "sigma": sigma,
                "note": "Non-stationary — I(1) x I_s(1)"
            }
        return {
            "type": f"Stationary SAR(1)(1)_{s}: (1-{phi}L)(1-{Phi}L^{s})Y_t = eps_t",
            "phi": phi,
            "Phi": Phi,
            "s": s,
            "sigma": sigma,
            "note": "Stationary when |phi| < 1 and |Phi| < 1"
        }
