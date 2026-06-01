"""
Markov Switching AR Data Generating Process.
"""

import numpy as np
from .base import BaseDGP


class MarkovSwitchingAR(BaseDGP):
    """
    AR(1) Markov Switching process with 2 regimes (Hamilton, 1989).

    Y_t = mu[S_t] + phi[S_t] * Y_{t-1} + sigma[S_t] * eps_t,  eps_t ~ N(0,1)

    S_t follows a first-order, ergodic Markov chain:
        P(S_t=j | S_{t-1}=i) given by row-stochastic matrix
            P = [[p00,   1-p00],
                 [1-p11, p11  ]]
    """

    def simulate(
        self,
        T: int,
        mu: tuple = (0.5, -0.5),
        phi: tuple = (0.3, 0.8),
        sigma: tuple = (1.0, 1.5),
        p00: float = 0.9,
        p11: float = 0.85,
        burn_in: int = 500,
    ) -> np.ndarray:
        mu = np.asarray(mu, dtype=float)
        phi = np.asarray(phi, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        total_T = T + burn_in

        # Row-stochastic transition matrix: P[i, j] = P(S_t=j | S_{t-1}=i)
        P = np.array([[p00, 1.0 - p00], [1.0 - p11, p11]])

        # Ergodic (stationary) probability for regime 0
        pi0 = (1.0 - p11) / (2.0 - p00 - p11)

        # Simulate Markov states starting from ergodic distribution
        states = np.empty(total_T, dtype=int)
        states[0] = self.rng.choice(2, p=[pi0, 1.0 - pi0])
        for t in range(1, total_T):
            states[t] = self.rng.choice(2, p=P[states[t - 1]])

        # Simulate observations
        z = self.rng.standard_normal(total_T)
        y = np.empty(total_T)
        s0 = states[0]
        y[0] = sigma[s0] * z[0]

        for t in range(1, total_T):
            s = states[t]
            y[t] = mu[s] + phi[s] * y[t - 1] + sigma[s] * z[t]

        return y[burn_in:]

    def get_theoretical_properties(
        self,
        mu: tuple = (0.5, -0.5),
        phi: tuple = (0.3, 0.8),
        sigma: tuple = (1.0, 1.5),
        p00: float = 0.9,
        p11: float = 0.85,
    ) -> dict:
        pi0 = (1.0 - p11) / (2.0 - p00 - p11)
        pi1 = 1.0 - pi0
        # Within-regime unconditional mean: E[Y|S=i] = mu[i] / (1 - phi[i])
        E0 = mu[0] / (1.0 - phi[0])
        E1 = mu[1] / (1.0 - phi[1])
        return {
            "ergodic_prob_regime_0": pi0,
            "ergodic_prob_regime_1": pi1,
            "unconditional_mean": pi0 * E0 + pi1 * E1,
            "avg_duration_regime_0": 1.0 / (1.0 - p00),
            "avg_duration_regime_1": 1.0 / (1.0 - p11),
        }
