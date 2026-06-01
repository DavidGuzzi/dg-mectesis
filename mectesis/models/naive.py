"""
Naive baseline forecasting models.
"""

import numpy as np
from .base import BaseModel


class NaiveModel(BaseModel):
    """
    Naive forecast: repeats the last observed value for all horizons.
    """

    def __init__(self):
        self._last_value = None

    def fit(self, y_train: np.ndarray, **kwargs):
        self._last_value = y_train[-1]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._last_value is None:
            raise ValueError("Model must be fitted before forecasting.")
        return np.full(horizon, self._last_value)

    @property
    def name(self) -> str:
        return "Naive"


class DriftModel(BaseModel):
    """
    Drift forecast: extrapolates the average historical change linearly.

    Forecast at h = y_T + h * (y_T - y_1) / (T - 1)
    """

    def __init__(self):
        self._last_value = None
        self._drift = None

    def fit(self, y_train: np.ndarray, **kwargs):
        T = len(y_train)
        self._last_value = y_train[-1]
        self._drift = (y_train[-1] - y_train[0]) / (T - 1) if T > 1 else 0.0

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._last_value is None:
            raise ValueError("Model must be fitted before forecasting.")
        steps = np.arange(1, horizon + 1)
        return self._last_value + self._drift * steps

    @property
    def name(self) -> str:
        return "Drift"


class SeasonalNaiveModel(BaseModel):
    """
    Seasonal Naive forecast: repeats the last observed value at the same seasonal phase.

    Forecast at h = y_{T - s + ((h-1) % s) + 1}
    """

    def __init__(self, period: int):
        """
        Parameters
        ----------
        period : int
            Seasonal period (e.g., 4 for quarterly, 12 for monthly).
        """
        self.period = period
        self._y_train = None

    def fit(self, y_train: np.ndarray, **kwargs):
        if len(y_train) < self.period:
            raise ValueError(
                f"Training series length ({len(y_train)}) must be >= period ({self.period})."
            )
        self._y_train = y_train

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._y_train is None:
            raise ValueError("Model must be fitted before forecasting.")
        s = self.period
        y_hat = np.empty(horizon)
        for h in range(1, horizon + 1):
            idx = len(self._y_train) - s + ((h - 1) % s)
            y_hat[h - 1] = self._y_train[idx]
        return y_hat

    @property
    def name(self) -> str:
        return f"SeasonalNaive(s={self.period})"
