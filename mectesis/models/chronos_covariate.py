"""
Chronos-2 wrappers that accept exogenous covariates.

ChronosCovariateModel   — univariate target with past/future covariates.
ChronosMultivariateCovariateModel — multivariate target with past/future covariates.

Both wrap a shared ChronosModel instance (which holds the pipeline) to avoid
loading the model twice.
"""

import numpy as np
from .base import BaseModel

# K=39 uniform levels for the proper CRPS estimator (trapezoidal pinball).
# Step 0.025 naturally includes 0.025, 0.10, 0.50, 0.90, 0.975 — all the
# levels needed for 80% and 95% intervals.
_CRPS_LEVELS = [round(0.025 * k, 3) for k in range(1, 40)]  # 0.025, 0.050, ..., 0.975
_CACHED_LEVELS = list(_CRPS_LEVELS)


class ChronosCovariateModel(BaseModel):
    """
    Chronos-2 with past and future exogenous covariates (univariate target).

    Uses the list-of-dicts API:
        inputs = [{"target": y, "past_covariates": {...}, "future_covariates": {...}}]

    Parameters
    ----------
    chronos_model : ChronosModel
        A pre-loaded ChronosModel instance (shares the pipeline).
    n_covariates : int
        Expected number of covariate columns in X_train / X_future.
    cov_names : list of str, optional
        Names for the covariate columns (defaults to ["x0", "x1", ...]).
    """

    def __init__(self, chronos_model, n_covariates: int = 1, cov_names=None):
        self._chronos = chronos_model
        self.n_covariates = n_covariates
        self._cov_names = cov_names or [f"x{i}" for i in range(n_covariates)]
        self._y_train = None
        self._X_train = None
        self._X_future = None
        self._cache: dict = {}

    @property
    def supports_covariates(self) -> bool:
        return True

    def fit(self, y_train: np.ndarray, X_train: np.ndarray = None, **kwargs):
        self._y_train  = y_train
        self._X_train  = X_train   # (T_train, p) or None
        self._X_future = None
        self._cache    = {}

    def _build_input(self, horizon: int, X_future: np.ndarray):
        """Build the list-of-dicts input for predict_quantiles."""
        inp = {"target": self._y_train}
        if self._X_train is not None and X_future is not None:
            inp["past_covariates"] = {
                name: self._X_train[:, i].astype(float)
                for i, name in enumerate(self._cov_names)
            }
            inp["future_covariates"] = {
                name: X_future[:horizon, i].astype(float)
                for i, name in enumerate(self._cov_names)
            }
        return inp

    def _all_quantiles(self, horizon: int, X_future: np.ndarray = None) -> dict:
        key = (horizon, None if X_future is None else X_future.tobytes())
        if key in self._cache:
            return self._cache[key]

        inp = self._build_input(horizon, X_future)
        quantiles, _ = self._chronos.pipeline.predict_quantiles(
            inputs=[inp],
            prediction_length=horizon,
            quantile_levels=_CACHED_LEVELS,
            batch_size=1,
        )
        # quantiles[0]: tensor [1, horizon, n_levels]
        q = quantiles[0]
        result = {
            level: q[0, :, i].numpy()
            for i, level in enumerate(_CACHED_LEVELS)
        }
        self._cache[key] = result
        return result

    def forecast(self, horizon: int, X_future: np.ndarray = None, **kwargs) -> np.ndarray:
        return self._all_quantiles(horizon, X_future)[0.5]

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95,
                           X_future: np.ndarray = None):
        q = self._all_quantiles(horizon, X_future)
        if level == 0.80:
            return q[0.1], q[0.9]
        elif level == 0.95:
            return q[0.025], q[0.975]
        raise ValueError(f"Unsupported level {level}. Use 0.80 or 0.95.")

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int,
                     X_future: np.ndarray = None) -> np.ndarray:
        """Proper CRPS via trapezoidal pinball loss over K=19 uniform quantiles."""
        from mectesis.metrics import crps_from_quantiles
        q = self._all_quantiles(horizon, X_future)
        levels = np.array(_CRPS_LEVELS)
        quantiles_arr = np.stack([q[l] for l in _CRPS_LEVELS], axis=1)  # (horizon, K)
        return crps_from_quantiles(y_true, quantiles_arr, levels)

    @property
    def name(self) -> str:
        return "Chronos-2 (con X)"


class ChronosMultivariateCovariateModel(BaseModel):
    """
    Chronos-2 joint multivariate with past and future exogenous covariates.

    target shape:  (T_train, k)   →  (1, k, T_train) for predict_quantiles
    Output shape:  (k, horizon, n_levels)

    Parameters
    ----------
    chronos_model : ChronosModel
        Pre-loaded ChronosModel instance.
    n_covariates : int
        Number of covariate columns.
    cov_names : list of str, optional
    """

    def __init__(self, chronos_model, n_covariates: int = 1, cov_names=None):
        self._chronos = chronos_model
        self.n_covariates = n_covariates
        self._cov_names = cov_names or [f"x{i}" for i in range(n_covariates)]
        self._y_train = None
        self._X_train = None
        self._cache: dict = {}

    @property
    def supports_covariates(self) -> bool:
        return True

    def fit(self, y_train: np.ndarray, X_train: np.ndarray = None, **kwargs):
        # y_train: (T_train, k)
        self._y_train = y_train
        self._X_train = X_train
        self._cache   = {}

    def _all_quantiles(self, horizon: int, X_future: np.ndarray = None) -> dict:
        key = (horizon, None if X_future is None else X_future.tobytes())
        if key in self._cache:
            return self._cache[key]

        # Pass k separate 1-D dicts so covariate shapes are unambiguous.
        k = self._y_train.shape[1]
        inps = []
        for j in range(k):
            d = {"target": self._y_train[:, j].astype(float)}
            if self._X_train is not None and X_future is not None:
                d["past_covariates"] = {
                    name: self._X_train[:, i].astype(float)
                    for i, name in enumerate(self._cov_names)
                }
                d["future_covariates"] = {
                    name: X_future[:horizon, i].astype(float)
                    for i, name in enumerate(self._cov_names)
                }
            inps.append(d)

        quantiles, _ = self._chronos.pipeline.predict_quantiles(
            inputs=inps,
            prediction_length=horizon,
            quantile_levels=_CACHED_LEVELS,
            batch_size=k,
        )
        # quantiles: list of k tensors, each [1, horizon, n_levels]
        result = {
            level: np.stack([quantiles[j][0, :, i].numpy() for j in range(k)], axis=0)
            for i, level in enumerate(_CACHED_LEVELS)
        }
        self._cache[key] = result
        return result

    def forecast(self, horizon: int, X_future: np.ndarray = None, **kwargs) -> np.ndarray:
        # Returns (horizon, k)
        return self._all_quantiles(horizon, X_future)[0.5].T

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95,
                           X_future: np.ndarray = None):
        q = self._all_quantiles(horizon, X_future)
        if level == 0.80:
            lo, hi = q[0.1].T, q[0.9].T     # (horizon, k)
        elif level == 0.95:
            lo, hi = q[0.025].T, q[0.975].T
        else:
            raise ValueError(f"Unsupported level {level}. Use 0.80 or 0.95.")
        return lo, hi

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int,
                     X_future: np.ndarray = None) -> np.ndarray:
        """
        Return CRPS of shape (horizon, k) via the proper trapezoidal estimator
        over K=19 uniform quantile levels.
        """
        from mectesis.metrics import crps_from_quantiles
        q = self._all_quantiles(horizon, X_future)
        # q[l]: (k, horizon) → stack to (k, horizon, K) → transpose to (horizon, k, K)
        quantiles_arr = np.stack([q[l] for l in _CRPS_LEVELS], axis=2)  # (k, horizon, K)
        quantiles_arr = quantiles_arr.transpose(1, 0, 2)                # (horizon, k, K)
        levels = np.array(_CRPS_LEVELS)
        return crps_from_quantiles(y_true, quantiles_arr, levels)        # (horizon, k)

    @property
    def name(self) -> str:
        return "Chronos-2 joint (con X)"
