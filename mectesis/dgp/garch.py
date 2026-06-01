"""
ARCH/GARCH/EGARCH Data Generating Processes.
"""

import numpy as np
from .base import BaseDGP


class AR1ARCH(BaseDGP):
    """
    AR(1)–ARCH(1) process.

    Mean:     Y_t = phi * Y_{t-1} + eps_t
    Variance: sigma_t^2 = omega + alpha * eps_{t-1}^2
    """

    def simulate(self, T: int, phi: float = 0.3, omega: float = 0.1,
                 alpha: float = 0.3, burn_in: int = 500) -> np.ndarray:
        total_T = T + burn_in
        z = self.rng.standard_normal(total_T)

        y     = np.empty(total_T)
        eps   = np.empty(total_T)
        sigma2 = np.empty(total_T)

        sigma2[0] = omega / (1.0 - alpha) if alpha < 1.0 else omega
        eps[0]    = np.sqrt(sigma2[0]) * z[0]
        y[0]      = eps[0]

        for t in range(1, total_T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2
            eps[t]    = np.sqrt(sigma2[t]) * z[t]
            y[t]      = phi * y[t - 1] + eps[t]

        return y[burn_in:]

    def get_theoretical_properties(self, phi: float = 0.3, omega: float = 0.1,
                                   alpha: float = 0.3) -> dict:
        uncond_var = omega / (1.0 - alpha) if alpha < 1.0 else np.nan
        return {
            "mean": 0.0,
            "unconditional_variance": uncond_var,
            "arch_persistence": alpha,
        }


class AR1GARCH(BaseDGP):
    """
    AR(1)–GARCH(1,1) process.

    Mean:     Y_t = phi * Y_{t-1} + eps_t
    Variance: sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    """

    def simulate(self, T: int, phi: float = 0.3, omega: float = 0.1,
                 alpha: float = 0.1, beta: float = 0.8,
                 burn_in: int = 500) -> np.ndarray:
        total_T = T + burn_in
        z = self.rng.standard_normal(total_T)

        y      = np.empty(total_T)
        eps    = np.empty(total_T)
        sigma2 = np.empty(total_T)

        persistence = alpha + beta
        sigma2[0] = omega / (1.0 - persistence) if persistence < 1.0 else omega
        eps[0]    = np.sqrt(sigma2[0]) * z[0]
        y[0]      = eps[0]

        for t in range(1, total_T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            eps[t]    = np.sqrt(sigma2[t]) * z[t]
            y[t]      = phi * y[t - 1] + eps[t]

        return y[burn_in:]

    def get_theoretical_properties(self, phi: float = 0.3, omega: float = 0.1,
                                   alpha: float = 0.1, beta: float = 0.8) -> dict:
        persistence = alpha + beta
        uncond_var  = omega / (1.0 - persistence) if persistence < 1.0 else np.nan
        return {
            "mean": 0.0,
            "unconditional_variance": uncond_var,
            "garch_persistence": persistence,
        }


class PureGARCH(BaseDGP):
    """
    GARCH(1,1) with zero mean.

    Y_t     = sigma_t * z_t,  z_t ~ N(0,1)
    sigma_t^2 = omega + alpha * Y_{t-1}^2 + beta * sigma_{t-1}^2
    """

    def simulate(self, T: int, omega: float = 0.1, alpha: float = 0.1,
                 beta: float = 0.8, burn_in: int = 500) -> np.ndarray:
        total_T = T + burn_in
        z = self.rng.standard_normal(total_T)

        y      = np.empty(total_T)
        sigma2 = np.empty(total_T)

        persistence = alpha + beta
        sigma2[0] = omega / (1.0 - persistence) if persistence < 1.0 else omega
        y[0]      = np.sqrt(sigma2[0]) * z[0]

        for t in range(1, total_T):
            sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
            y[t]      = np.sqrt(sigma2[t]) * z[t]

        return y[burn_in:]

    def get_theoretical_properties(self, omega: float = 0.1, alpha: float = 0.1,
                                   beta: float = 0.8) -> dict:
        persistence = alpha + beta
        uncond_var  = omega / (1.0 - persistence) if persistence < 1.0 else np.nan
        return {
            "mean": 0.0,
            "unconditional_variance": uncond_var,
            "garch_persistence": persistence,
        }


class AR1GJRGARCH(BaseDGP):
    """
    AR(1)–GJR–GARCH(1,1,1) with leverage effect.

    Mean:     Y_t = phi * Y_{t-1} + eps_t
    Variance: sigma_t^2 = omega
                          + alpha * eps_{t-1}^2
                          + gamma * eps_{t-1}^2 * 1{eps_{t-1} < 0}
                          + beta  * sigma_{t-1}^2

    Effective persistence: alpha + gamma/2 + beta (E[1{eps<0}] = 0.5 for symmetric z).
    """

    def simulate(self, T: int, phi: float = 0.3, omega: float = 0.1,
                 alpha: float = 0.05, gamma: float = 0.1,
                 beta: float = 0.8, burn_in: int = 500) -> np.ndarray:
        total_T = T + burn_in
        z = self.rng.standard_normal(total_T)

        y      = np.empty(total_T)
        eps    = np.empty(total_T)
        sigma2 = np.empty(total_T)

        eff_persistence = alpha + gamma / 2.0 + beta
        sigma2[0] = omega / (1.0 - eff_persistence) if eff_persistence < 1.0 else omega
        eps[0]    = np.sqrt(sigma2[0]) * z[0]
        y[0]      = eps[0]

        for t in range(1, total_T):
            ind        = 1.0 if eps[t - 1] < 0.0 else 0.0
            sigma2[t]  = (omega
                          + alpha * eps[t - 1] ** 2
                          + gamma * eps[t - 1] ** 2 * ind
                          + beta  * sigma2[t - 1])
            eps[t]     = np.sqrt(sigma2[t]) * z[t]
            y[t]       = phi * y[t - 1] + eps[t]

        return y[burn_in:]

    def get_theoretical_properties(self, phi: float = 0.3, omega: float = 0.1,
                                   alpha: float = 0.05, gamma: float = 0.1,
                                   beta: float = 0.8) -> dict:
        eff_persistence = alpha + gamma / 2.0 + beta
        uncond_var = omega / (1.0 - eff_persistence) if eff_persistence < 1.0 else np.nan
        return {
            "mean": 0.0,
            "unconditional_variance": uncond_var,
            "effective_persistence": eff_persistence,
            "leverage_gamma": gamma,
        }


class AR1EGARCH(BaseDGP):
    """
    AR(1)–EGARCH(1,1) process (Nelson, 1991).

    Mean:     Y_t = phi * Y_{t-1} + eps_t,  eps_t = sigma_t * z_t,  z_t ~ N(0,1)
    Log-var:  ln(sigma_t^2) = omega
                              + beta  * ln(sigma_{t-1}^2)
                              + alpha * (|z_{t-1}| - E[|z|])
                              + gamma * z_{t-1}

    E[|z|] = sqrt(2/pi) for z ~ N(0,1).
    gamma < 0 captures the leverage effect (negative shocks raise vol more).
    """

    _E_ABS_Z = np.sqrt(2.0 / np.pi)

    def simulate(
        self,
        T: int,
        phi: float = 0.3,
        omega: float = 0.02,
        beta: float = 0.9,
        alpha: float = 0.1,
        gamma: float = -0.05,
        burn_in: int = 500,
    ) -> np.ndarray:
        total_T = T + burn_in
        z = self.rng.standard_normal(total_T)

        y      = np.empty(total_T)
        log_s2 = np.empty(total_T)
        eps    = np.empty(total_T)

        # Unconditional log-variance: omega / (1 - beta)
        log_s2[0] = omega / (1.0 - beta) if abs(beta) < 1.0 else 0.0
        eps[0]    = np.exp(0.5 * log_s2[0]) * z[0]
        y[0]      = eps[0]

        for t in range(1, total_T):
            z_lag     = z[t - 1]
            log_s2[t] = (
                omega
                + beta  * log_s2[t - 1]
                + alpha * (abs(z_lag) - self._E_ABS_Z)
                + gamma * z_lag
            )
            eps[t] = np.exp(0.5 * log_s2[t]) * z[t]
            y[t]   = phi * y[t - 1] + eps[t]

        return y[burn_in:]

    def get_theoretical_properties(
        self,
        phi: float = 0.3,
        omega: float = 0.02,
        beta: float = 0.9,
        alpha: float = 0.1,
        gamma: float = -0.05,
    ) -> dict:
        uncond_log_var = omega / (1.0 - beta) if abs(beta) < 1.0 else np.nan
        return {
            "mean": 0.0,
            "egarch_persistence": beta,
            "leverage_gamma": gamma,
            "unconditional_log_variance": uncond_log_var,
            "unconditional_variance": (
                np.exp(uncond_log_var) if not np.isnan(uncond_log_var) else np.nan
            ),
        }
