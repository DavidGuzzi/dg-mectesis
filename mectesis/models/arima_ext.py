"""
Extended ARIMA models: with deterministic trend and structural break.
"""

import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseModel


class ARIMAWithTrendModel(BaseModel):
    """
    ARIMA with deterministic trend via statsmodels trend parameter.

    Fits: Y_t = alpha + beta*t + phi*Y_{t-1} + eps_t  (trend='ct')
    Covers Exp 1.5.
    """

    def __init__(self, order: tuple = (1, 0, 0), trend: str = "ct"):
        self.order = order
        self.trend = trend
        self.fitted_model = None

    def fit(self, y_train: np.ndarray, **kwargs):
        self._fcst_cache = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y_train, order=self.order, trend=self.trend)
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
        return f"ARIMA{self.order}+trend"


class ARIMAWithBreakModel(BaseModel):
    """
    ARIMA with structural break dummy as exogenous variable (SARIMAX).

    Break indicator: 0 before break_idx, 1 on/after.
    Forecast period is always post-break (exog = all-ones).
    Covers Exp 1.8. Must be instantiated with T_total = full series length T.
    """

    def __init__(self, order: tuple = (1, 0, 0), T_total: int = 200,
                 break_fraction: float = 0.5):
        self.order = order
        self.T_total = T_total
        self.break_fraction = break_fraction
        self.break_idx = int(T_total * break_fraction)
        self.fitted_model = None

    def fit(self, y_train: np.ndarray, **kwargs):
        self._fcst_cache = {}
        n = len(y_train)
        break_exog = (np.arange(n) >= self.break_idx).astype(float).reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train,
                order=self.order,
                exog=break_exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted_model = model.fit(disp=False, **kwargs)

    def _get_forecast(self, horizon: int):
        if horizon not in self._fcst_cache:
            exog_future = np.ones((horizon, 1))
            self._fcst_cache[horizon] = self.fitted_model.get_forecast(
                steps=horizon, exog=exog_future)
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
        return f"ARIMA{self.order}+break"
