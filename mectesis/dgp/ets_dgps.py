"""
State-space DGPs for ETS / exponential-smoothing family experiments.
"""

import numpy as np
from .base import BaseDGP


class LocalLevelDGP(BaseDGP):
    """
    Local level model (ETS(A,N,N) state-space form):
        ℓ_t = ℓ_{t-1} + η_t,   η_t ~ N(0, σ_η²)
        Y_t  = ℓ_t  + ε_t,     ε_t ~ N(0, σ_ε²)
    """

    def simulate(self, T: int, sigma_eps: float = 1.0,
                 sigma_eta: float = 0.3, l0: float = 0.0) -> np.ndarray:
        level = np.empty(T + 1)
        level[0] = l0
        eta = self.rng.normal(0.0, sigma_eta, size=T)
        for t in range(T):
            level[t + 1] = level[t] + eta[t]
        eps = self.rng.normal(0.0, sigma_eps, size=T)
        return level[1:] + eps

    def get_theoretical_properties(self) -> dict:
        return {"type": "local_level", "stationary": False}


class LocalTrendDGP(BaseDGP):
    """
    Local linear trend model (ETS(A,A,N) state-space form):
        ℓ_t = ℓ_{t-1} + b_{t-1} + η_t,   η_t ~ N(0, σ_η²)
        b_t  = b_{t-1} + ζ_t,              ζ_t ~ N(0, σ_ζ²)
        Y_t  = ℓ_t + ε_t,                  ε_t ~ N(0, σ_ε²)
    """

    def simulate(self, T: int, sigma_eps: float = 1.0,
                 sigma_eta: float = 0.2, sigma_zeta: float = 0.1,
                 l0: float = 0.0, b0: float = 0.1) -> np.ndarray:
        level = np.empty(T + 1)
        slope = np.empty(T + 1)
        level[0] = l0
        slope[0] = b0
        eta  = self.rng.normal(0.0, sigma_eta,  size=T)
        zeta = self.rng.normal(0.0, sigma_zeta, size=T)
        for t in range(T):
            level[t + 1] = level[t] + slope[t] + eta[t]
            slope[t + 1] = slope[t] + zeta[t]
        eps = self.rng.normal(0.0, sigma_eps, size=T)
        return level[1:] + eps

    def get_theoretical_properties(self) -> dict:
        return {"type": "local_trend", "stationary": False}


class DampedTrendDGP(BaseDGP):
    """
    Damped trend model (ETS(A,Ad,N) state-space form):
        ℓ_t = ℓ_{t-1} + φ·b_{t-1} + η_t,   η_t ~ N(0, σ_η²)
        b_t  = φ·b_{t-1} + ζ_t,              ζ_t ~ N(0, σ_ζ²)
        Y_t  = ℓ_t + ε_t,                    ε_t ~ N(0, σ_ε²)
    """

    def simulate(self, T: int, phi: float = 0.9, sigma_eps: float = 1.0,
                 sigma_eta: float = 0.2, sigma_zeta: float = 0.1,
                 l0: float = 0.0, b0: float = 0.1) -> np.ndarray:
        level = np.empty(T + 1)
        slope = np.empty(T + 1)
        level[0] = l0
        slope[0] = b0
        eta  = self.rng.normal(0.0, sigma_eta,  size=T)
        zeta = self.rng.normal(0.0, sigma_zeta, size=T)
        for t in range(T):
            level[t + 1] = level[t] + phi * slope[t] + eta[t]
            slope[t + 1] = phi * slope[t] + zeta[t]
        eps = self.rng.normal(0.0, sigma_eps, size=T)
        return level[1:] + eps

    def get_theoretical_properties(self) -> dict:
        return {"type": "damped_trend", "stationary": False}


class DeterministicSeasonalDGP(BaseDGP):
    """
    Pure deterministic seasonality:
        Y_t = μ + s_{t mod m} + ε_t,   Σ s_j = 0

    The seasonal pattern is a discrete sine wave scaled to std ≈ 2 and
    zero-mean, giving visible seasonal amplitude without dominating the noise.
    """

    def simulate(self, T: int, mu: float = 5.0,
                 sigma_eps: float = 1.0, s: int = 12) -> np.ndarray:
        j = np.arange(s)
        pattern = np.sin(2.0 * np.pi * j / s)
        pattern -= pattern.mean()
        pattern = pattern / pattern.std() * 2.0
        eps = self.rng.normal(0.0, sigma_eps, size=T)
        seasonal = np.array([pattern[t % s] for t in range(T)])
        return mu + seasonal + eps

    def get_theoretical_properties(self) -> dict:
        return {"type": "deterministic_seasonal", "stationary": True}


class SeasonalRandomWalkDGP(BaseDGP):
    """
    Seasonal random walk  (SARIMA(0,0,0)(0,1,0)_s DGP):
        Y_t = Y_{t-m} + ε_t,   ε_t ~ N(0, σ²)
    """

    def simulate(self, T: int, s: int = 12,
                 sigma: float = 1.0, burn_in: int = 100) -> np.ndarray:
        total = T + burn_in
        # pre-allocate with s extra slots for the initial lag buffer
        y = np.zeros(total + s)
        eps = self.rng.normal(0.0, sigma, size=total)
        for t in range(total):
            y[s + t] = y[t] + eps[t]
        return y[s + burn_in:]

    def get_theoretical_properties(self) -> dict:
        return {"type": "seasonal_random_walk", "stationary": False}


class LocalLevelSeasonalDGP(BaseDGP):
    """
    Full ETS(A,A,A) state-space DGP:
        ℓ_t = ℓ_{t-1} + b_{t-1} + η_t,   η_t ~ N(0, σ_η²)
        b_t  = b_{t-1} + ζ_t,              ζ_t ~ N(0, σ_ζ²)
        γ_t  = γ_{t-m} + ω_t,              ω_t ~ N(0, σ_ω²)
        Y_t  = ℓ_t + γ_t + ε_t,           ε_t ~ N(0, σ_ε²)

    Initial seasonal states are set to a zero-mean discrete sine pattern.
    """

    def simulate(self, T: int, sigma_eps: float = 0.5,
                 sigma_eta: float = 0.1, sigma_zeta: float = 0.05,
                 sigma_omega: float = 0.1, l0: float = 5.0,
                 b0: float = 0.1, s: int = 12) -> np.ndarray:
        # Initial seasonal pattern (sum-zero sine)
        j = np.arange(s)
        gamma = np.sin(2.0 * np.pi * j / s)
        gamma -= gamma.mean()

        level = l0
        slope = b0
        # gamma[i] holds the seasonal state for phase i (mod s)
        y = np.empty(T)
        eta   = self.rng.normal(0.0, sigma_eta,   size=T)
        zeta  = self.rng.normal(0.0, sigma_zeta,  size=T)
        omega = self.rng.normal(0.0, sigma_omega, size=T)
        eps   = self.rng.normal(0.0, sigma_eps,   size=T)

        for t in range(T):
            phase = t % s
            gamma_t = gamma[phase]
            y[t] = level + gamma_t + eps[t]
            new_level = level + slope + eta[t]
            slope = slope + zeta[t]
            gamma[phase] = gamma_t + omega[t]
            level = new_level

        return y

    def get_theoretical_properties(self) -> dict:
        return {"type": "local_level_seasonal", "stationary": False}
