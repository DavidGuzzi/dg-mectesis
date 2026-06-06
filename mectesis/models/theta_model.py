"""
Theta model using statsmodels ThetaModel.
"""

import numpy as np
from .base import BaseModel


class ThetaModel(BaseModel):
    """
    Theta forecasting method via statsmodels.tsa.forecasting.theta.ThetaModel.

    Prediction intervals use prediction_intervals(steps, alpha=1-level).
    CRPS assumes a Gaussian predictive distribution derived from the 95 % interval.
    """

    def __init__(self, deseasonalize: bool = False):
        self._deseasonalize = deseasonalize
        self._result = None
        self._pi95_cache: dict = {}

    @property
    def name(self) -> str:
        return "Theta"

    def fit(self, y_train: np.ndarray, **kwargs):
        from statsmodels.tsa.forecasting.theta import ThetaModel as _Theta
        self._pi95_cache = {}
        self._result = _Theta(
            y_train,
            deseasonalize=self._deseasonalize,
        ).fit(disp=False)

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        return np.asarray(self._result.forecast(horizon))

    def _pi95(self, horizon: int):
        """Cache 95 % prediction interval DataFrame for CRPS reuse."""
        if horizon not in self._pi95_cache:
            self._pi95_cache[horizon] = self._result.prediction_intervals(
                horizon, alpha=0.05
            )
        return self._pi95_cache[horizon]

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        pi = self._result.prediction_intervals(horizon, alpha=1.0 - level)
        lo = np.asarray(pi["lower"])
        hi = np.asarray(pi["upper"])
        return lo, hi

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        # Infer sigma from the 95% PI assuming a symmetric Gaussian band
        # centered at the point forecast. This is consistent with
        # statsmodels' default ThetaModel.prediction_intervals (Gaussian-based).
        # If that mechanism ever changes (t-Student, empirical, asymmetric),
        # this conversion becomes inconsistent — verify before upgrading
        # statsmodels.
        from properscoring import crps_gaussian
        from scipy.stats import norm

        mu = np.asarray(self._result.forecast(horizon))
        pi = self._pi95(horizon)
        lo = np.asarray(pi["lower"])
        hi = np.asarray(pi["upper"])
        sigma = np.maximum((hi - lo) / (2.0 * norm.ppf(0.975)), 1e-8)
        return crps_gaussian(y_true[:horizon], mu, sigma)
