"""
AR(1) with Structural Break Data Generating Process.
"""

import numpy as np
from .base import BaseDGP


class AR1WithBreak(BaseDGP):
    """
    AR(1) with a structural break in the AR coefficient at break_fraction * T.

    Y_t = phi_before * Y_{t-1} + eps_t  for t <= break_idx
    Y_t = phi_after  * Y_{t-1} + eps_t  for t >  break_idx

    Covers Exp 1.8 with phi_before=0.3, phi_after=0.8.
    """

    def simulate(self, T: int, phi_before: float = 0.3, phi_after: float = 0.8,
                 sigma: float = 1.0, break_fraction: float = 0.5,
                 burn_in: int = 50) -> np.ndarray:
        """
        Simulate AR(1) with structural break.

        Parameters
        ----------
        T : int
            Length of the returned series.
        phi_before : float, optional
            AR coefficient in the first regime. Default is 0.3.
        phi_after : float, optional
            AR coefficient in the second regime. Default is 0.8.
        sigma : float, optional
            Standard deviation of innovations (same in both regimes). Default 1.0.
        break_fraction : float, optional
            Position of the break as a fraction of total series length. Default 0.5.
        burn_in : int, optional
            Burn-in periods for the first regime. Default is 50.

        Returns
        -------
        np.ndarray
            Simulated series of shape (T,).

        Notes
        -----
        The break occurs at index int(T * break_fraction) of the returned series,
        so it is always at T//2 for the default break_fraction=0.5.
        """
        total_T = T + burn_in
        break_idx_full = int(total_T * break_fraction)

        y = np.empty(total_T)
        y[0] = 0.0
        eps = self.rng.normal(0.0, sigma, size=total_T)

        for t in range(1, total_T):
            phi = phi_before if t <= break_idx_full else phi_after
            y[t] = phi * y[t - 1] + eps[t]

        return y[burn_in:]

    def get_theoretical_properties(self, phi_before: float = 0.3,
                                   phi_after: float = 0.8,
                                   sigma: float = 1.0,
                                   break_fraction: float = 0.5) -> dict:
        return {
            "type": "AR(1) with structural break",
            "phi_before": phi_before,
            "phi_after": phi_after,
            "break_fraction": break_fraction,
            "sigma": sigma,
            "note": "Both regimes stationary when |phi_before| < 1 and |phi_after| < 1"
        }
