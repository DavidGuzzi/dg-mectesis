"""
Chronos-2 Time Series Foundation Model implementation.
"""

import numpy as np
import torch
from chronos import Chronos2Pipeline
from .base import BaseModel

# K=39 uniform levels for the proper CRPS estimator (trapezoidal pinball).
# Step 0.025 naturally includes 0.025, 0.10, 0.50, 0.90, 0.975 — all the
# levels needed for 80% and 95% intervals.
_CRPS_LEVELS = [round(0.025 * k, 3) for k in range(1, 40)]  # 0.025, 0.050, ..., 0.975
_CACHED_LEVELS = list(_CRPS_LEVELS)


class ChronosModel(BaseModel):
    """
    Amazon Chronos-2 TSFM wrapper.

    Zero-shot: fit() stores the context, forecast() runs inference.
    All quantile levels needed for point forecast + 80%/95% intervals
    are computed in a single pipeline call and cached per horizon,
    avoiding redundant inference when forecast_intervals() is called
    after forecast().
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.pipeline = None
        self.y_train = None
        self._cache: dict = {}
        self._load_pipeline()

    def _load_pipeline(self):
        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=self.device,
            dtype=torch.bfloat16,
        )

    def fit(self, y_train: np.ndarray, **kwargs):
        """Store training context and clear quantile cache."""
        self.y_train = y_train
        self._cache = {}

    def _all_quantiles(self, horizon: int) -> dict:
        """
        Single inference call for all required quantile levels.
        Result is cached by horizon so subsequent forecast_intervals()
        calls are free.

        Returns
        -------
        dict {level_float: np.ndarray of shape (horizon,)}
        """
        if horizon in self._cache:
            return self._cache[horizon]

        if self.y_train is None:
            raise ValueError("Call fit() before forecast().")

        inputs = [{"target": self.y_train}]
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=horizon,
            quantile_levels=_CACHED_LEVELS,
            batch_size=1,
        )
        # quantiles is a list; quantiles[0] shape: [n_variates=1, horizon, n_levels]
        q = quantiles[0]  # tensor [1, horizon, n_levels]
        result = {
            level: q[0, :, i].numpy()
            for i, level in enumerate(_CACHED_LEVELS)
        }
        self._cache[horizon] = result
        return result

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        """Return median (0.5 quantile) forecast."""
        return self._all_quantiles(horizon)[0.5]

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        """
        Return (lower, upper) from cached quantiles.

        Supported levels: 0.80 (uses 0.10/0.90) and 0.95 (uses 0.025/0.975).
        """
        q = self._all_quantiles(horizon)
        if level == 0.80:
            return q[0.1], q[0.9]
        elif level == 0.95:
            return q[0.025], q[0.975]
        else:
            raise ValueError(
                f"Chronos interval level {level} not supported. Use 0.80 or 0.95."
            )

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        """Proper CRPS via trapezoidal pinball loss over K=19 uniform quantiles."""
        from mectesis.metrics import crps_from_quantiles
        q = self._all_quantiles(horizon)
        levels = np.array(_CRPS_LEVELS)
        quantiles_arr = np.stack([q[l] for l in _CRPS_LEVELS], axis=1)  # (horizon, K)
        return crps_from_quantiles(y_true, quantiles_arr, levels)

    @property
    def name(self) -> str:
        return "Chronos-2"
