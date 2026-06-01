"""
Base class for Data Generating Processes (DGP).
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseDGP(ABC):
    """
    Abstract base class for data generating processes.

    All DGP implementations must inherit from this class and implement
    the simulate() and get_theoretical_properties() methods.
    """

    def __init__(self, seed: int = None):
        """
        Initialize the DGP with an optional random seed.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. If None, uses system entropy.
        """
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def simulate(self, T: int, **params) -> np.ndarray:
        """
        Simulate a time series of length T.

        Parameters
        ----------
        T : int
            Length of the time series to generate.
        **params : dict
            DGP-specific parameters (e.g., phi, mu, sigma for AR processes).

        Returns
        -------
        np.ndarray
            Simulated time series of shape (T,).
        """
        pass

    @abstractmethod
    def get_theoretical_properties(self) -> dict:
        """
        Return theoretical properties of the DGP.

        Returns
        -------
        dict
            Dictionary containing theoretical properties such as:
            - mean: theoretical mean
            - variance: theoretical variance
            - acf: autocorrelation function (if applicable)
        """
        pass
