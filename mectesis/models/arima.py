"""
ARIMA model implementation.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from .base import BaseModel


class ARIMAModel(BaseModel):
    """ARIMA(p, d, q) model using statsmodels."""

    def __init__(self, order: tuple = (1, 0, 0)):
        self.order = order
        self.fitted_model = None
        self._y_train = None

    def fit(self, y_train: np.ndarray, **kwargs):
        self._y_train = y_train
        self._fcst_cache = {}
        model = ARIMA(y_train, order=self.order)
        self.fitted_model = model.fit(**kwargs)

    def _get_forecast(self, horizon: int):
        if horizon not in self._fcst_cache:
            self._fcst_cache[horizon] = self.fitted_model.get_forecast(steps=horizon)
        return self._fcst_cache[horizon]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Call fit() before forecast().")
        return np.array(self._get_forecast(horizon).predicted_mean)

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        if self.fitted_model is None:
            raise ValueError("Call fit() before forecast_intervals().")
        alpha = 1.0 - level
        ci = np.asarray(self._get_forecast(horizon).conf_int(alpha=alpha))
        return ci[:, 0], ci[:, 1]

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_gaussian
        fcst = self._get_forecast(horizon)
        mu = np.array(fcst.predicted_mean)
        sigma = np.maximum(np.array(fcst.se_mean), 1e-8)
        return crps_gaussian(y_true, mu, sigma)

    @property
    def name(self) -> str:
        return f"ARIMA{self.order}"
