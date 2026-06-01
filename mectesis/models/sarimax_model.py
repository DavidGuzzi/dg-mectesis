"""
SARIMAX model wrapper — supports exogenous covariates.

Accepts optional X_train in fit() and X_future in forecast()/forecast_intervals().
When X_train is None the model falls back to a plain ARIMA.
"""

import warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseModel

_FIT_ERRORS = (np.linalg.LinAlgError, RuntimeError, ValueError, Exception)


class SARIMAXModel(BaseModel):
    """
    SARIMAX(p, d, q) x (P, D, Q, s) with optional exogenous regressors.

    Parameters
    ----------
    order : tuple
        ARIMA order (p, d, q).
    seasonal_order : tuple, optional
        Seasonal order (P, D, Q, s). Default (0, 0, 0, 0) = no seasonality.
    trend : str, optional
        Trend specification passed to statsmodels SARIMAX:
        - "c"  : constant (default, backwards compatible)
        - "ct" : constant + linear deterministic trend
        - "t"  : linear deterministic trend (no constant)
        - "n"  : no trend / no constant
    name_suffix : str
        Optional suffix appended to the model name (e.g. 'con X').
    """

    def __init__(self, order: tuple = (1, 0, 0),
                 seasonal_order: tuple = (0, 0, 0, 0),
                 trend: str = "c",
                 name_suffix: str = ""):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self._name_suffix = name_suffix
        self._fitted = None
        self._fit_failed = False
        self._fcst_cache: dict = {}

    def fit(self, y_train: np.ndarray, X_train: np.ndarray = None, **kwargs):
        self._fitted = None
        self._fit_failed = False
        self._fcst_cache = {}
        exog = X_train if X_train is not None and X_train.shape[0] > 0 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = SARIMAX(y_train, exog=exog, order=self.order,
                                seasonal_order=self.seasonal_order,
                                trend=self.trend,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                self._fitted = model.fit(disp=False)
            except _FIT_ERRORS:
                self._fit_failed = True

    def _get_forecast(self, horizon: int, X_future: np.ndarray = None):
        key = (horizon, None if X_future is None else X_future.tobytes())
        if key not in self._fcst_cache:
            if self._fit_failed or self._fitted is None:
                return None
            exog_f = X_future if X_future is not None and X_future.shape[0] > 0 else None
            try:
                self._fcst_cache[key] = self._fitted.get_forecast(
                    steps=horizon, exog=exog_f
                )
            except Exception:
                return None
        return self._fcst_cache[key]

    def forecast(self, horizon: int, X_future: np.ndarray = None, **kwargs) -> np.ndarray:
        fcst = self._get_forecast(horizon, X_future)
        if fcst is None:
            return np.full(horizon, np.nan)
        return np.array(fcst.predicted_mean)

    @property
    def supports_covariates(self) -> bool:
        return True

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95,
                           X_future: np.ndarray = None):
        fcst = self._get_forecast(horizon, X_future)
        if fcst is None:
            nan = np.full(horizon, np.nan)
            return nan, nan
        alpha = 1.0 - level
        try:
            ci = np.asarray(fcst.conf_int(alpha=alpha))
            return ci[:, 0], ci[:, 1]
        except Exception:
            nan = np.full(horizon, np.nan)
            return nan, nan

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int,
                     X_future: np.ndarray = None) -> np.ndarray:
        from properscoring import crps_gaussian
        fcst = self._get_forecast(horizon, X_future)
        if fcst is None:
            return np.full(horizon, np.nan)
        mu    = np.array(fcst.predicted_mean)
        sigma = np.maximum(np.array(fcst.se_mean), 1e-8)
        return crps_gaussian(y_true, mu, sigma)

    @property
    def name(self) -> str:
        base = f"SARIMAX{self.order}"
        if self.seasonal_order != (0, 0, 0, 0):
            base += f"x{self.seasonal_order}"
        return f"{base} {self._name_suffix}".strip()
