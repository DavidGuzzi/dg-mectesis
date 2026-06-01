"""
General ARMA family DGPs: ARpDGP, MAqDGP, ARMApqDGP, ARMApqWithTrendDGP.

All parameters (order coefficients, sigma) are set at construction time.
simulate() accepts an optional sigma override to match the MonteCarloEngine
calling convention (dgp.simulate(T=T, **dgp_params)).
"""

import numpy as np
from scipy.signal import lfilter
from statsmodels.tsa.arima_process import ArmaProcess

from .base import BaseDGP


class ARpDGP(BaseDGP):
    """
    AR(p) process:  y_t = φ₁·y_{t-1} + ... + φ_p·y_{t-p} + ε_t

    Parameters
    ----------
    phis : list of float
        AR coefficients [φ₁, ..., φ_p]. Process must be stationary.
    sigma : float
        Innovation standard deviation.
    burn_in : int
        Number of initial samples discarded to remove start-up effects.
    """

    def __init__(self, phis, sigma: float = 1.0, burn_in: int = 200, seed=None):
        super().__init__(seed)
        self.phis = list(phis)
        self.sigma = sigma
        self.burn_in = burn_in
        if self.phis:
            ar_poly = np.r_[1, -np.array(self.phis)]
            if not ArmaProcess(ar_poly, [1.0]).isstationary:
                raise ValueError(f"AR process with phis={phis} is not stationary")

    def simulate(self, T: int, sigma: float = None) -> np.ndarray:
        sigma = self.sigma if sigma is None else sigma
        total = T + self.burn_in
        eps = self.rng.normal(0.0, sigma, size=total)
        if self.phis:
            ar_poly = np.r_[1, -np.array(self.phis)]
            y = lfilter([1.0], ar_poly, eps)
        else:
            y = eps
        return y[self.burn_in:]

    def get_theoretical_properties(self, **kwargs) -> dict:
        return {
            "type": f"AR({len(self.phis)})",
            "phis": self.phis,
            "sigma": self.sigma,
        }


class MAqDGP(BaseDGP):
    """
    MA(q) process:  y_t = ε_t + θ₁·ε_{t-1} + ... + θ_q·ε_{t-q}

    Parameters
    ----------
    thetas : list of float
        MA coefficients [θ₁, ..., θ_q]. Process must be invertible.
    sigma : float
        Innovation standard deviation.
    burn_in : int
        Burn-in periods (MA has finite memory; 50 suffices for q ≤ 4).
    """

    def __init__(self, thetas, sigma: float = 1.0, burn_in: int = 50, seed=None):
        super().__init__(seed)
        self.thetas = list(thetas)
        self.sigma = sigma
        self.burn_in = burn_in
        if self.thetas:
            ma_poly = np.r_[1, np.array(self.thetas)]
            if not ArmaProcess([1.0], ma_poly).isinvertible:
                raise ValueError(f"MA process with thetas={thetas} is not invertible")

    def simulate(self, T: int, sigma: float = None) -> np.ndarray:
        sigma = self.sigma if sigma is None else sigma
        total = T + self.burn_in
        eps = self.rng.normal(0.0, sigma, size=total)
        if self.thetas:
            ma_poly = np.r_[1, np.array(self.thetas)]
            y = lfilter(ma_poly, [1.0], eps)
        else:
            y = eps
        return y[self.burn_in:]

    def get_theoretical_properties(self, **kwargs) -> dict:
        return {
            "type": f"MA({len(self.thetas)})",
            "thetas": self.thetas,
            "sigma": self.sigma,
        }


class ARMApqDGP(BaseDGP):
    """
    ARMA(p,q) process:
        y_t = φ₁·y_{t-1}+...+φ_p·y_{t-p} + ε_t + θ₁·ε_{t-1}+...+θ_q·ε_{t-q}

    Empty phis or thetas reduce to pure MA or AR, respectively.

    Parameters
    ----------
    phis : list of float
        AR coefficients. Pass [] for pure MA.
    thetas : list of float
        MA coefficients. Pass [] for pure AR.
    sigma : float
        Innovation standard deviation.
    burn_in : int
        Number of initial samples discarded.
    """

    def __init__(self, phis, thetas, sigma: float = 1.0, burn_in: int = 200, seed=None):
        super().__init__(seed)
        self.phis = list(phis)
        self.thetas = list(thetas)
        self.sigma = sigma
        self.burn_in = burn_in
        ar_poly = np.r_[1, -np.array(self.phis)] if self.phis else np.array([1.0])
        ma_poly = np.r_[1, np.array(self.thetas)] if self.thetas else np.array([1.0])
        proc = ArmaProcess(ar_poly, ma_poly)
        if self.phis and not proc.isstationary:
            raise ValueError(
                f"ARMA({len(self.phis)},{len(self.thetas)}) with phis={phis} is not stationary"
            )
        if self.thetas and not proc.isinvertible:
            raise ValueError(
                f"ARMA({len(self.phis)},{len(self.thetas)}) with thetas={thetas} is not invertible"
            )

    def simulate(self, T: int, sigma: float = None) -> np.ndarray:
        sigma = self.sigma if sigma is None else sigma
        total = T + self.burn_in
        eps = self.rng.normal(0.0, sigma, size=total)
        ar_poly = np.r_[1, -np.array(self.phis)] if self.phis else np.array([1.0])
        ma_poly = np.r_[1, np.array(self.thetas)] if self.thetas else np.array([1.0])
        y = lfilter(ma_poly, ar_poly, eps)
        return y[self.burn_in:]

    def get_theoretical_properties(self, **kwargs) -> dict:
        return {
            "type": f"ARMA({len(self.phis)},{len(self.thetas)})",
            "phis": self.phis,
            "thetas": self.thetas,
            "sigma": self.sigma,
        }


class ARMApqWithTrendDGP(ARMApqDGP):
    """
    ARMA(p,q) + deterministic trend:  Y_t = alpha + delta·t + ARMA_t

    Parameters
    ----------
    phis, thetas : list of float
        AR and MA coefficients (same as ARMApqDGP).
    delta : float
        Trend slope coefficient.
    alpha : float
        Intercept (level at t=0).
    sigma : float
        Innovation standard deviation.
    burn_in : int
        Number of initial samples discarded.
    """

    def __init__(
        self,
        phis,
        thetas,
        delta: float,
        alpha: float = 0.0,
        sigma: float = 1.0,
        burn_in: int = 200,
        seed=None,
    ):
        super().__init__(phis, thetas, sigma=sigma, burn_in=burn_in, seed=seed)
        self.delta = delta
        self.alpha = alpha

    def simulate(self, T: int, sigma: float = None) -> np.ndarray:
        arma = super().simulate(T, sigma=sigma)
        t = np.arange(1, T + 1, dtype=float)
        return self.alpha + self.delta * t + arma

    def get_theoretical_properties(self, **kwargs) -> dict:
        props = super().get_theoretical_properties()
        props["type"] += "+trend"
        props["delta"] = self.delta
        props["alpha"] = self.alpha
        return props
