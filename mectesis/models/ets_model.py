"""
ETS (Error-Trend-Seasonality) model using statsmodels ExponentialSmoothing.
"""

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .base import BaseModel


class ETSModel(BaseModel):
    """
    ETS additive-error model via statsmodels ExponentialSmoothing.

    Configurations:
        ETS(A,N,N) — trend=None,  damped_trend=False, seasonal=None
        ETS(A,A,N) — trend='add', damped_trend=False, seasonal=None
        ETS(A,Ad,N)— trend='add', damped_trend=True,  seasonal=None
        ETS(A,A,A) — trend='add', damped_trend=False, seasonal='add'
    """

    def __init__(self, trend=None, damped_trend: bool = False,
                 seasonal=None, seasonal_periods: int = None,
                 n_sims: int = 500):
        self._trend = trend
        self._damped_trend = damped_trend
        self._seasonal = seasonal
        self._seasonal_periods = seasonal_periods
        self._n_sims = n_sims
        self._result = None
        self._sims_cache: dict = {}

    @property
    def name(self) -> str:
        if self._trend is None:
            t = "N"
        elif self._damped_trend:
            t = "Ad"
        else:
            t = "A"
        s = "N" if self._seasonal is None else "A"
        return f"ETS(A,{t},{s})"

    def fit(self, y_train: np.ndarray, **kwargs):
        self._sims_cache = {}
        model = ExponentialSmoothing(
            y_train,
            trend=self._trend,
            damped_trend=self._damped_trend if self._trend is not None else False,
            seasonal=self._seasonal,
            seasonal_periods=self._seasonal_periods,
            initialization_method="estimated",
        )
        self._result = model.fit(optimized=True)

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        return np.asarray(self._result.forecast(horizon))

    def _get_sims(self, horizon: int) -> np.ndarray:
        """Return simulation paths of shape (horizon, n_sims)."""
        if horizon not in self._sims_cache:
            sims = self._result.simulate(
                nsimulations=horizon,
                anchor="end",
                repetitions=self._n_sims,
            )
            self._sims_cache[horizon] = np.asarray(sims)
        return self._sims_cache[horizon]

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        sims = self._get_sims(horizon)          # (horizon, n_sims)
        alpha = (1.0 - level) / 2.0
        lo = np.quantile(sims, alpha,       axis=1)
        hi = np.quantile(sims, 1.0 - alpha, axis=1)
        return lo, hi

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_ensemble
        sims = self._get_sims(horizon)          # (horizon, n_sims)
        return np.array([
            crps_ensemble(y_true[h], sims[h]) for h in range(horizon)
        ])
