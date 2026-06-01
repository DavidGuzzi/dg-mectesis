"""
ARDL / ECM model for cointegrated systems.

For the ADL-ECM experiment (3.6) the DGP is:
    ΔY_t = alpha_ecm * (Y_{t-1} - X_{t-1}) + ΔX_t + eta_t

We estimate this via OLS on the ECM representation:
    ΔY_t = c + alpha * (Y_{t-1} - X_{t-1}) + beta * ΔX_t + eta_t

Forecasting multi-step ahead is done recursively, using the known future X values.
"""

import warnings
import numpy as np
from .base import BaseModel

_FIT_ERRORS = (np.linalg.LinAlgError, RuntimeError, ValueError, Exception)


class ARDLModel(BaseModel):
    """
    Error-Correction Model estimated by OLS.

    fit() estimates the ECM parameters from (y_train, X_train).
    forecast() uses recursion together with the known future X values.
    """

    def __init__(self):
        self._alpha = None
        self._beta  = None
        self._const = None
        self._fit_failed = False
        self._y_last = None
        self._X_last = None
        self._fcst_cache: dict = {}

    def fit(self, y_train: np.ndarray, X_train: np.ndarray = None, **kwargs):
        self._fit_failed = False
        self._fcst_cache = {}
        self._y_last = float(y_train[-1])
        self._X_last = float(X_train[-1, 0]) if X_train is not None else 0.0

        if X_train is None or X_train.shape[0] < 3:
            self._fit_failed = True
            return

        x = X_train[:, 0]
        y = y_train
        n = len(y)

        dy   = np.diff(y)           # ΔY_t,  length n-1
        dx   = np.diff(x)           # ΔX_t,  length n-1
        ecm  = y[:-1] - x[:-1]     # Y_{t-1} - X_{t-1}

        # Design matrix [const, ecm, dx]
        Z = np.column_stack([np.ones(n - 1), ecm, dx])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(Z, dy, rcond=None)
            self._const = float(coeffs[0])
            self._alpha = float(coeffs[1])  # should be negative
            self._beta  = float(coeffs[2])
        except _FIT_ERRORS:
            self._fit_failed = True

    def forecast(self, horizon: int, X_future: np.ndarray = None, **kwargs) -> np.ndarray:
        if self._fit_failed or self._alpha is None:
            return np.full(horizon, np.nan)
        if X_future is None:
            return np.full(horizon, np.nan)

        x_fut = X_future[:, 0]  # (H,)
        y_hat = np.empty(horizon)
        y_prev = self._y_last
        x_prev = self._X_last

        for h in range(horizon):
            x_curr = x_fut[h]
            dx     = x_curr - x_prev
            ecm    = y_prev - x_prev
            dy_hat = self._const + self._alpha * ecm + self._beta * dx
            y_curr = y_prev + dy_hat
            y_hat[h] = y_curr
            y_prev = y_curr
            x_prev = x_curr

        return y_hat

    @property
    def supports_covariates(self) -> bool:
        return True

    @property
    def supports_intervals(self) -> bool:
        return False

    @property
    def supports_crps(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ARDL-ECM"
