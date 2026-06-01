"""
Transformations for empirical series: log-diff, alignment, etc.
"""

import numpy as np
import pandas as pd


def log_diff(x: pd.Series, scale: float = 1.0) -> pd.Series:
    return scale * np.log(x).diff().dropna()


def to_monthly_inflation(ipc: pd.Series) -> pd.Series:
    return log_diff(ipc, scale=100.0).rename("pi_mensual")


def to_yoy_inflation(ipc: pd.Series) -> pd.Series:
    return (100.0 * (np.log(ipc) - np.log(ipc.shift(12)))).dropna().rename("pi_yoy")


def align_panel(df: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    return df.dropna(how="any" if how == "inner" else "all")


def select_optimal_lags(
    y: pd.Series,
    X: pd.DataFrame,
    max_lag: int = 12,
    min_lag: int = 1,
    granger_panel: pd.DataFrame | None = None,
    alpha: float = 0.10,
    granger_maxlag: int = 6,
) -> dict[str, int]:
    """Pick the lag (in [min_lag, max_lag]) that maximizes |corr(y_t, x_{t-lag})|
    for each column of X. Lags are strictly predictive (>= 1) to avoid
    reverse-endogeneity leakage. If `granger_panel` is provided, columns whose
    `min p-value over lags 1..granger_maxlag` of `x -> y` exceeds `alpha`
    are dropped from the returned map.

    Convention: lag>0 → x.shift(lag), so y_t is regressed on x's past.

    `granger_maxlag` is independent of the selected lag — a covariable can be
    optimally predictive at lag=1 by cross-correlation while its Granger
    evidence peaks at lag=2 or 3; we want to keep it.
    """
    if min_lag < 1:
        raise ValueError(f"min_lag must be >= 1 (got {min_lag}); use cross_corr directly for non-predictive lags")
    if max_lag < min_lag:
        raise ValueError(f"max_lag ({max_lag}) must be >= min_lag ({min_lag})")
    if granger_maxlag < 1:
        raise ValueError(f"granger_maxlag must be >= 1 (got {granger_maxlag})")

    lag_map: dict[str, int] = {}
    for col in X.columns:
        x = X[col]
        best_lag, best_abs = None, -np.inf
        for lag in range(min_lag, max_lag + 1):
            df = pd.concat([y, x.shift(lag)], axis=1).dropna()
            if df.shape[0] < 3:
                continue
            corr = float(df.corr().iloc[0, 1])
            if np.isfinite(corr) and abs(corr) > best_abs:
                best_abs, best_lag = abs(corr), lag
        if best_lag is None:
            continue
        lag_map[col] = best_lag

    if granger_panel is not None:
        from .diagnostics import granger_matrix
        gm = granger_matrix(granger_panel, maxlag=granger_maxlag)
        y_name = y.name
        for col in list(lag_map.keys()):
            if y_name in gm.index and col in gm.columns:
                pval = gm.loc[y_name, col]
                if not np.isfinite(pval) or pval > alpha:
                    lag_map.pop(col)
    return lag_map


def apply_lags(X: pd.DataFrame, lag_map: dict[str, int]) -> pd.DataFrame:
    """Shift each column of X by its lag in `lag_map` (strictly positive) and
    rename to `{col}__lag{L}`. Index is preserved; the first L observations
    of each lagged column are NaN. Use `.dropna()` downstream to align with `y`.
    """
    for col, lag in lag_map.items():
        if lag < 1:
            raise ValueError(f"apply_lags requires lag >= 1 (got {lag} for {col}) to avoid look-ahead leakage")
        if col not in X.columns:
            raise KeyError(f"column {col!r} not in X")
    out = pd.DataFrame(index=X.index)
    for col, lag in lag_map.items():
        out[f"{col}__lag{lag}"] = X[col].shift(lag)
    return out
