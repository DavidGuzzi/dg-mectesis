"""
Rolling-origin (a.k.a. walk-forward) backtest for forecasting models on
a single empirical time series.

Metrics reported per (model × horizon):
    n_origins, rmse, mae, crps, mase, bias

Coverage / Winkler / interval-width metrics are intentionally NOT in the
default output (intervals are still computed and stored for fan-charts
via `predict_at_last_origin`).

Supports three modes:
  - univariate:   y is pd.Series
  - covariates:   y is pd.Series, X is pd.DataFrame (exog at fit/forecast)
  - multivariate: y is pd.DataFrame; models return (horizon, k)
"""

from typing import Callable

import numpy as np
import pandas as pd

from ..models.base import BaseModel


def _mase_scale(y_train: np.ndarray, season: int = 12) -> np.ndarray:
    """
    Per-variable seasonal-naïve scaling for MASE (M4 convention).

    For multivariate input shape (T, k) returns shape (k,);
    for univariate shape (T,)  returns shape (1,).
    """
    arr = np.atleast_2d(y_train.T).T if y_train.ndim == 1 else y_train
    T = arr.shape[0]
    s = season if T > 2 * season else 1
    diffs = np.abs(arr[s:] - arr[:-s])
    scale = np.nanmean(diffs, axis=0)
    scale = np.where(scale > 0, scale, np.nan)
    return scale


class RollingOriginBacktest:
    """
    Refit-at-every-origin backtest.

    A fresh model is instantiated via `model_factory()` at each origin to
    avoid carry-over state.

    Parameters
    ----------
    model_factory : callable() -> BaseModel
    y : pd.Series | pd.DataFrame
        Target series (Series → univariate / DataFrame → multivariate).
    horizons : list[int]
    initial_window : int
        Number of observations used to fit the first origin.
    scheme : {'expanding', 'sliding'}
    step : int
        Stride between consecutive origins.
    X : pd.DataFrame | None
        Exogenous regressors aligned with `y` (covariate mode).
    season : int
        Seasonal period for MASE scaling (default 12 = monthly).
    """

    def __init__(
        self,
        model_factory: Callable[[], BaseModel],
        y: pd.Series | pd.DataFrame,
        horizons: list[int],
        initial_window: int = 72,
        scheme: str = "expanding",
        step: int = 1,
        X: pd.DataFrame | None = None,
        season: int = 12,
    ):
        if scheme not in ("expanding", "sliding"):
            raise ValueError("scheme must be 'expanding' or 'sliding'")
        self.model_factory = model_factory
        self.y = y
        self.horizons = sorted(horizons)
        self.initial_window = initial_window
        self.scheme = scheme
        self.step = step
        self.X = X
        self.season = season
        self._multivariate = isinstance(y, pd.DataFrame)

    def _origins(self) -> list[int]:
        max_h = max(self.horizons)
        T = self.y.shape[0]
        last = T - max_h
        return list(range(self.initial_window, last + 1, self.step))

    def _slice_train(self, t: int):
        start = 0 if self.scheme == "expanding" else max(0, t - self.initial_window)
        return start, t

    def run(self, verbose: bool = False) -> dict[int, pd.DataFrame]:
        origins = self._origins()
        n_orig = len(origins)
        if n_orig == 0:
            raise ValueError(
                f"No valid origins for T={self.y.shape[0]}, "
                f"initial_window={self.initial_window}, max_h={max(self.horizons)}"
            )

        max_h = max(self.horizons)
        y_arr = self.y.to_numpy()
        X_arr = self.X.to_numpy() if self.X is not None else None
        k = y_arr.shape[1] if self._multivariate else 1

        shape = (n_orig, max_h, k) if self._multivariate else (n_orig, max_h)
        errors = np.full(shape, np.nan)
        crps = np.full(shape, np.nan)
        scale = np.full((n_orig, k), np.nan)
        has_crps = False

        for i, t in enumerate(origins):
            start, end = self._slice_train(t)
            y_train = y_arr[start:end]
            y_test = y_arr[end:end + max_h]
            scale[i] = _mase_scale(y_train, season=self.season)

            model = self.model_factory()
            fit_kwargs, forecast_kwargs = {}, {}
            if self.X is not None:
                fit_kwargs["X_train"] = X_arr[start:end]
                forecast_kwargs["X_future"] = X_arr[end:end + max_h]

            model.fit(y_train, **fit_kwargs)
            y_hat = model.forecast(max_h, **forecast_kwargs)
            errors[i] = y_test - y_hat

            if model.supports_crps:
                has_crps = True
                crps[i] = (
                    model.compute_crps(y_test, max_h, **forecast_kwargs)
                    if model.supports_covariates
                    else model.compute_crps(y_test, max_h)
                )

            if verbose:
                print(f"  origin {i+1}/{n_orig} (t={t})")

        return self._summarise(errors, crps, scale, has_crps, k)

    def _summarise(self, errors, crps, scale, has_crps, k):
        out: dict[int, pd.DataFrame] = {}
        for h in self.horizons:
            idx = h - 1
            if self._multivariate:
                rows = []
                for j, name in enumerate(self.y.columns):
                    err = errors[:, idx, j]
                    row = self._point_row(err)
                    row["variable"] = name
                    row["mase"] = float(np.nanmean(np.abs(err) / scale[:, j]))
                    if has_crps:
                        row["crps"] = float(np.nanmean(crps[:, idx, j]))
                    rows.append(row)
                df = pd.DataFrame(rows)
            else:
                err = errors[:, idx]
                row = self._point_row(err)
                row["mase"] = float(np.nanmean(np.abs(err) / scale[:, 0]))
                if has_crps:
                    row["crps"] = float(np.nanmean(crps[:, idx]))
                df = pd.DataFrame([row])
            out[h] = df
        return out

    @staticmethod
    def _point_row(err: np.ndarray) -> dict:
        return {
            "n_origins": int(np.sum(~np.isnan(err))),
            "rmse": float(np.sqrt(np.nanmean(err ** 2))),
            "mae": float(np.nanmean(np.abs(err))),
            "bias": float(np.nanmean(err)),
        }

    def predict_at_last_origin(self) -> dict:
        """
        Refit once at the last origin and return point forecast + 80/95 % bands,
        useful for fan-charts. Returns:
            {
              "origin_idx": int (index in y / panel),
              "horizon":    int (max horizon),
              "mean": ndarray(h,) or (h, k),
              "lo80": ..., "hi80": ..., "lo95": ..., "hi95": ...,
              "model_name": str,
            }
        """
        origins = self._origins()
        t = origins[-1]
        start, end = self._slice_train(t)
        max_h = max(self.horizons)
        y_arr = self.y.to_numpy()
        X_arr = self.X.to_numpy() if self.X is not None else None

        model = self.model_factory()
        fit_kwargs, forecast_kwargs = {}, {}
        if self.X is not None:
            fit_kwargs["X_train"] = X_arr[start:end]
            forecast_kwargs["X_future"] = X_arr[end:end + max_h]
        model.fit(y_arr[start:end], **fit_kwargs)

        out = {
            "origin_idx": t,
            "horizon": max_h,
            "mean": np.asarray(model.forecast(max_h, **forecast_kwargs)),
            "model_name": model.name,
        }
        if model.supports_intervals:
            for lv, tag in ((0.80, "80"), (0.95, "95")):
                if model.supports_covariates:
                    lo, hi = model.forecast_intervals(max_h, level=lv, **forecast_kwargs)
                else:
                    lo, hi = model.forecast_intervals(max_h, level=lv)
                out[f"lo{tag}"] = np.asarray(lo)
                out[f"hi{tag}"] = np.asarray(hi)
        return out


def compare_models(
    factories: dict[str, Callable[[], BaseModel]],
    y: pd.Series | pd.DataFrame,
    horizons: list[int],
    initial_window: int = 72,
    scheme: str = "expanding",
    X: pd.DataFrame | None = None,
    season: int = 12,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run RollingOriginBacktest for each factory and return a long-format
    DataFrame with (model, horizon[, variable]) as identifiers.
    """
    rows = []
    for name, factory in factories.items():
        if verbose:
            print(f"[backtest] {name}")
        bt = RollingOriginBacktest(
            factory, y, horizons,
            initial_window=initial_window, scheme=scheme,
            X=X, season=season,
        )
        res = bt.run(verbose=False)
        for h, df in res.items():
            tmp = df.copy()
            tmp.insert(0, "horizon", h)
            tmp.insert(0, "model", name)
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def predict_all_at_last_origin(
    factories: dict[str, Callable[[], BaseModel]],
    y: pd.Series | pd.DataFrame,
    horizons: list[int],
    initial_window: int = 72,
    scheme: str = "expanding",
    X: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """
    Run `predict_at_last_origin` for each factory. Returns
    `{model_name: {mean, lo80, hi80, lo95, hi95, origin_idx, horizon, model_name}}`.
    """
    out = {}
    for name, factory in factories.items():
        bt = RollingOriginBacktest(
            factory, y, horizons,
            initial_window=initial_window, scheme=scheme, X=X,
        )
        out[name] = bt.predict_at_last_origin()
    return out


def to_wide_table(
    long_df: pd.DataFrame,
    metrics: tuple[str, ...] = ("rmse", "mae", "crps", "mase"),
    index: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Pivot a long-format result (output of `compare_models`) into a wide
    table with MultiIndex `(metric, horizon)` columns and `model`
    (plus `variable` if multivariate) as index.
    """
    if index is None:
        index = ("variable", "model") if "variable" in long_df.columns else ("model",)
    present = [m for m in metrics if m in long_df.columns]
    pivot = long_df.pivot_table(
        index=list(index), columns="horizon", values=list(present)
    )
    pivot = pivot.reorder_levels([0, 1], axis=1).sort_index(axis=1, level=[0, 1])
    return pivot
