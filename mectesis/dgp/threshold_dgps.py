"""
Threshold / Smooth-Transition DGPs: SETARDGp, LSTARDGp, ESTARDGp.

These non-linear DGPs are compared against a linear ARIMA(1,0,0) benchmark
to test whether Chronos-2 can detect deterministic regime-switching that a
linear model cannot capture.
"""

import numpy as np
from .base import BaseDGP


class SETARDGp(BaseDGP):
    """
    Self-Exciting Threshold AR (SETAR) with two regimes:

        y_t = φ₁·y_{t-1} + ε_t   if  y_{t-d} ≤ threshold
        y_t = φ₂·y_{t-1} + ε_t   if  y_{t-d} >  threshold

    The regime indicator is determined by the observable lag y_{t-d},
    contrasting with the latent-state Markov Switching AR already in the
    codebase.

    Parameters
    ----------
    phi1 : float
        AR coefficient in the lower regime.
    phi2 : float
        AR coefficient in the upper regime.
    threshold : float
        Threshold value k. Default 0.0.
    delay : int
        Delay parameter d (which lag determines the regime). Default 1.
    sigma : float
        Innovation standard deviation.
    burn_in : int
        Burn-in periods discarded.
    """

    def __init__(
        self,
        phi1: float,
        phi2: float,
        threshold: float = 0.0,
        delay: int = 1,
        sigma: float = 1.0,
        burn_in: int = 200,
        seed=None,
    ):
        super().__init__(seed)
        self.phi1 = phi1
        self.phi2 = phi2
        self.threshold = threshold
        self.delay = delay
        self.sigma = sigma
        self.burn_in = burn_in

    def simulate(self, T: int, sigma: float = None) -> np.ndarray:
        sigma = self.sigma if sigma is None else sigma
        d = self.delay
        total = T + self.burn_in
        eps = self.rng.normal(0.0, sigma, size=total)
        y = np.zeros(total)
        for t in range(1, total):
            indicator = y[t - d] if t > d else 0.0
            phi = self.phi1 if indicator <= self.threshold else self.phi2
            y[t] = phi * y[t - 1] + eps[t]
        return y[self.burn_in :]

    def get_theoretical_properties(self, **kwargs) -> dict:
        return {
            "type": f"SETAR(2;1) threshold={self.threshold}",
            "phi1": self.phi1,
            "phi2": self.phi2,
            "threshold": self.threshold,
            "delay": self.delay,
            "sigma": self.sigma,
        }


class LSTARDGp(BaseDGP):
    """
    Logistic Smooth Transition AR (LSTAR):

        G(y_{t-d}; γ, c) = 1 / (1 + exp(−γ·(y_{t-d} − c)))   ∈ (0, 1)
        y_t = [φ₁·(1 − G) + φ₂·G]·y_{t-1} + ε_t

    G ≈ 0 (lower regime) when y_{t-d} << c;
    G ≈ 1 (upper regime) when y_{t-d} >> c.
    Captures directional asymmetry (e.g., business-cycle expansions vs.
    contractions).

    Parameters
    ----------
    phi1 : float
        AR coefficient in the lower regime (G → 0).
    phi2 : float
        AR coefficient in the upper regime (G → 1).
    gamma : float
        Transition speed. Larger γ → sharper transition.
    c : float
        Transition midpoint (threshold location). Default 0.0.
    delay : int
        Delay parameter d. Default 1.
    sigma : float
        Innovation standard deviation.
    burn_in : int
        Burn-in periods discarded.
    """

    def __init__(
        self,
        phi1: float,
        phi2: float,
        gamma: float,
        c: float = 0.0,
        delay: int = 1,
        sigma: float = 1.0,
        burn_in: int = 200,
        seed=None,
    ):
        super().__init__(seed)
        self.phi1 = phi1
        self.phi2 = phi2
        self.gamma = gamma
        self.c = c
        self.delay = delay
        self.sigma = sigma
        self.burn_in = burn_in

    def _transition(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-self.gamma * (x - self.c)))

    def simulate(self, T: int, sigma: float = None) -> np.ndarray:
        sigma = self.sigma if sigma is None else sigma
        d = self.delay
        total = T + self.burn_in
        eps = self.rng.normal(0.0, sigma, size=total)
        y = np.zeros(total)
        for t in range(1, total):
            indicator = y[t - d] if t > d else 0.0
            G = self._transition(indicator)
            phi = self.phi1 * (1.0 - G) + self.phi2 * G
            y[t] = phi * y[t - 1] + eps[t]
        return y[self.burn_in :]

    def get_theoretical_properties(self, **kwargs) -> dict:
        return {
            "type": "LSTAR(1)",
            "phi1": self.phi1,
            "phi2": self.phi2,
            "gamma": self.gamma,
            "c": self.c,
            "delay": self.delay,
            "sigma": self.sigma,
        }


class ESTARDGp(BaseDGP):
    """
    Exponential Smooth Transition AR (ESTAR):

        G(y_{t-d}; γ, c) = 1 − exp(−γ·(y_{t-d} − c)²)   ∈ [0, 1)
        y_t = [φ₁·(1 − G) + φ₂·G]·y_{t-1} + ε_t

    G ≈ 0 (inner regime) when y_{t-d} ≈ c;
    G ≈ 1 (outer regime) when y_{t-d} is far from c.
    Captures symmetric mean-reversion around a band — canonical model
    for exchange rates with target zones.

    Parameters
    ----------
    phi1 : float
        AR coefficient in the inner regime (G → 0, near equilibrium).
    phi2 : float
        AR coefficient in the outer regime (G → 1, far from equilibrium).
    gamma : float
        Transition speed. Controls how quickly G rises away from c.
    c : float
        Equilibrium center. Default 0.0.
    delay : int
        Delay parameter d. Default 1.
    sigma : float
        Innovation standard deviation.
    burn_in : int
        Burn-in periods discarded.
    """

    def __init__(
        self,
        phi1: float,
        phi2: float,
        gamma: float,
        c: float = 0.0,
        delay: int = 1,
        sigma: float = 1.0,
        burn_in: int = 200,
        seed=None,
    ):
        super().__init__(seed)
        self.phi1 = phi1
        self.phi2 = phi2
        self.gamma = gamma
        self.c = c
        self.delay = delay
        self.sigma = sigma
        self.burn_in = burn_in

    def _transition(self, x: float) -> float:
        return 1.0 - np.exp(-self.gamma * (x - self.c) ** 2)

    def simulate(self, T: int, sigma: float = None) -> np.ndarray:
        sigma = self.sigma if sigma is None else sigma
        d = self.delay
        total = T + self.burn_in
        eps = self.rng.normal(0.0, sigma, size=total)
        y = np.zeros(total)
        for t in range(1, total):
            indicator = y[t - d] if t > d else 0.0
            G = self._transition(indicator)
            phi = self.phi1 * (1.0 - G) + self.phi2 * G
            y[t] = phi * y[t - 1] + eps[t]
        return y[self.burn_in :]

    def get_theoretical_properties(self, **kwargs) -> dict:
        return {
            "type": "ESTAR(1)",
            "phi1": self.phi1,
            "phi2": self.phi2,
            "gamma": self.gamma,
            "c": self.c,
            "delay": self.delay,
            "sigma": self.sigma,
        }
