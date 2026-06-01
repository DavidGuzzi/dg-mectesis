"""
Gaussian Process DGP — KernelSynth style (as in Chronos training data).

Kernels:
  "rbf"         — smooth nonlinear trend (RBF / squared-exponential)
  "periodic"    — periodic pattern
  "rbf+periodic"— composite: trend + seasonality (closest to KernelSynth)
"""

import numpy as np
from .base import BaseDGP


class GPKernelSynthDGP(BaseDGP):
    """
    Sample from a Gaussian Process with configurable kernel composition.

    y_t = f(t) + noise,  f ~ GP(0, K)

    Supported kernels (via the `kernel` parameter):
      - "rbf"          : squared-exponential (smooth trend)
      - "periodic"     : periodic kernel
      - "rbf+periodic" : sum of both (KernelSynth-style)
    """

    def simulate(
        self,
        T: int,
        kernel: str = "rbf+periodic",
        lengthscale_rbf: float = 30.0,
        sigma_rbf: float = 1.0,
        period: float = 12.0,
        lengthscale_per: float = 1.0,
        sigma_per: float = 0.8,
        noise_std: float = 0.3,
    ) -> np.ndarray:
        t = np.arange(T, dtype=float)
        diff = t[:, None] - t[None, :]   # (T, T) signed differences

        K = np.zeros((T, T))

        if "rbf" in kernel:
            K += sigma_rbf ** 2 * np.exp(-(diff ** 2) / (2.0 * lengthscale_rbf ** 2))

        if "periodic" in kernel:
            K += sigma_per ** 2 * np.exp(
                -2.0 * np.sin(np.pi * np.abs(diff) / period) ** 2
                / lengthscale_per ** 2
            )

        # Add diagonal jitter for numerical stability + observation noise
        K += (noise_std ** 2 + 1e-6) * np.eye(T)

        L = np.linalg.cholesky(K)
        return L @ self.rng.standard_normal(T)

    def get_theoretical_properties(
        self,
        kernel: str = "rbf+periodic",
        sigma_rbf: float = 1.0,
        sigma_per: float = 0.8,
        noise_std: float = 0.3,
        **kwargs,
    ) -> dict:
        var = noise_std ** 2
        if "rbf" in kernel:
            var += sigma_rbf ** 2
        if "periodic" in kernel:
            var += sigma_per ** 2
        return {
            "mean": 0.0,
            "variance": var,
            "note": "GP — no finite-order ARMA representation",
        }
