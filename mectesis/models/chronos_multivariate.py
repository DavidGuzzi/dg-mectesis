"""
Chronos-2 multivariate wrappers.

ChronosMultivariateModel: native joint multivariate mode.
    Passes all k variables together as (1, k, T_train) — one inference call.
    Output shape: (k, H, n_quantiles) from pipeline.predict_quantiles.

ChronosPerVarModel: k independent univariate forecasts (secondary baseline).
    Allows measuring how much Chronos-2 benefits from seeing all variables jointly.
"""

import numpy as np
from .base import BaseModel

# K=39 uniform levels for the proper CRPS estimator (trapezoidal pinball).
# Step 0.025 naturally includes 0.025, 0.10, 0.50, 0.90, 0.975 — all the
# levels needed for 80% and 95% intervals.
_CRPS_LEVELS = [round(0.025 * k, 3) for k in range(1, 40)]  # 0.025, 0.050, ..., 0.975
_CACHED_LEVELS = list(_CRPS_LEVELS)

# Named indices into _CACHED_LEVELS for forecast/forecast_intervals.
_IDX_LO95 = _CACHED_LEVELS.index(0.025)
_IDX_LO80 = _CACHED_LEVELS.index(0.10)
_IDX_MED  = _CACHED_LEVELS.index(0.50)
_IDX_HI80 = _CACHED_LEVELS.index(0.90)
_IDX_HI95 = _CACHED_LEVELS.index(0.975)


class ChronosMultivariateModel(BaseModel):
    """
    Chronos-2 in native joint multivariate mode.

    All k variables are passed as a single (1, k, T_train) input tensor.
    The model processes them jointly in one forward pass, returning
    (k, H, n_quantiles). This is how Chronos-2 should be used for
    multivariate series — not k separate univariate calls.

    Cache: {horizon: ndarray(k, H, n_levels)} — one entry per unique horizon.
    """

    def __init__(self, chronos_model):
        self.chronos = chronos_model   # ChronosModel instance (holds the pipeline)
        self._y_train = None           # (T_train, k)
        self._k = None
        self._cache: dict = {}         # {horizon: ndarray(k, H, n_levels)}

    def fit(self, y_train: np.ndarray, **kwargs):
        self._y_train = y_train
        self._k = y_train.shape[1]
        self._cache = {}

    def _all_quantiles(self, horizon: int) -> np.ndarray:
        """
        Single joint inference for all k variables at once.
        Returns ndarray of shape (k, horizon, n_levels).
        """
        if horizon in self._cache:
            return self._cache[horizon]

        if self._y_train is None:
            raise ValueError("Call fit() before forecast().")

        # Input shape required by predict_quantiles: (batch_size, k, T_train)
        inputs = self._y_train.T[np.newaxis, :, :]  # (1, k, T_train)

        quantiles, _ = self.chronos.pipeline.predict_quantiles(
            inputs,
            prediction_length=horizon,
            quantile_levels=_CACHED_LEVELS,
            batch_size=1,
        )
        # quantiles is a list of length batch_size=1
        # quantiles[0] shape: (k, horizon, n_levels) as a torch tensor
        result = quantiles[0].numpy()   # (k, horizon, 5)
        self._cache[horizon] = result
        return result

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        """Return median forecasts of shape (horizon, k)."""
        q = self._all_quantiles(horizon)         # (k, horizon, n_levels)
        return q[:, :, _IDX_MED].T               # (horizon, k)

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(
        self, horizon: int, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (lower, upper) of shape (horizon, k).
        Supported levels: 0.80 and 0.95.
        """
        q = self._all_quantiles(horizon)         # (k, horizon, n_levels)
        if level == 0.95:
            lo = q[:, :, _IDX_LO95].T
            hi = q[:, :, _IDX_HI95].T
        elif level == 0.80:
            lo = q[:, :, _IDX_LO80].T
            hi = q[:, :, _IDX_HI80].T
        else:
            raise ValueError(
                f"Chronos interval level {level} not supported. Use 0.80 or 0.95."
            )
        return lo, hi

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        """
        Return CRPS of shape (horizon, k) via the proper trapezoidal estimator
        over K=19 uniform quantile levels.
        """
        from mectesis.metrics import crps_from_quantiles
        q = self._all_quantiles(horizon)         # (k, horizon, n_levels)
        idx = [_CACHED_LEVELS.index(l) for l in _CRPS_LEVELS]
        quantiles_arr = q[:, :, idx].transpose(1, 0, 2)   # (horizon, k, K)
        levels = np.array(_CRPS_LEVELS)
        return crps_from_quantiles(y_true, quantiles_arr, levels)   # (horizon, k)

    @property
    def name(self) -> str:
        return "Chronos-2 (joint)"


class ChronosPerVarModel(BaseModel):
    """
    Chronos-2 running k independent univariate forecasts — secondary baseline.

    Each variable is forecast separately with no cross-variable information.
    Useful to quantify how much the native joint mode (ChronosMultivariateModel)
    actually benefits from seeing all variables together.

    This is equivalent to what Chronos-1 would do, as Chronos-1 did not support
    multivariate inputs.
    """

    def __init__(self, chronos_model):
        self.chronos = chronos_model
        self._y_train = None   # (T_train, k)
        self._k = None
        self._var_caches: dict = {}  # {var_idx: {horizon: {level: ndarray(H,)}}}

    def fit(self, y_train: np.ndarray, **kwargs):
        self._y_train = y_train
        self._k = y_train.shape[1]
        self._var_caches = {}

    def _get_var_quantiles(self, var_idx: int, horizon: int) -> dict:
        """Run Chronos univariately on variable var_idx and cache the quantiles."""
        if var_idx not in self._var_caches:
            self._var_caches[var_idx] = {}
        if horizon not in self._var_caches[var_idx]:
            self.chronos.fit(self._y_train[:, var_idx])
            self._var_caches[var_idx][horizon] = self.chronos._all_quantiles(horizon)
        return self._var_caches[var_idx][horizon]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        """Return median forecasts of shape (horizon, k)."""
        medians = [self._get_var_quantiles(j, horizon)[0.5] for j in range(self._k)]
        return np.column_stack(medians)  # (horizon, k)

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(
        self, horizon: int, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) of shape (horizon, k)."""
        los, his = [], []
        for j in range(self._k):
            q = self._get_var_quantiles(j, horizon)
            if level == 0.95:
                los.append(q[0.025])
                his.append(q[0.975])
            elif level == 0.80:
                los.append(q[0.1])
                his.append(q[0.9])
            else:
                raise ValueError(
                    f"Chronos interval level {level} not supported. Use 0.80 or 0.95."
                )
        return np.column_stack(los), np.column_stack(his)  # each (horizon, k)

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        """
        Return CRPS of shape (horizon, k) via the proper trapezoidal estimator
        over K=19 uniform quantile levels.
        """
        from mectesis.metrics import crps_from_quantiles
        levels = np.array(_CRPS_LEVELS)
        result = np.empty((horizon, self._k))
        for j in range(self._k):
            q = self._get_var_quantiles(j, horizon)
            quantiles_arr = np.stack([q[l] for l in _CRPS_LEVELS], axis=1)  # (horizon, K)
            result[:, j] = crps_from_quantiles(y_true[:, j], quantiles_arr, levels)
        return result

    @property
    def name(self) -> str:
        return "Chronos-2 (ind.)"
