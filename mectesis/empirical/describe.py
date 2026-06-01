"""
Series description: summary statistics and standardised plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL


def summary_stats(y: pd.Series) -> pd.DataFrame:
    s = y.dropna()
    return pd.DataFrame([{
        "n": int(s.shape[0]),
        "start": s.index.min(),
        "end": s.index.max(),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
    }])


def plot_series(y: pd.Series, title: str = "", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    y.plot(ax=ax, color="#1f4068")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_acf_pacf(y: pd.Series, lags: int = 36, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    plot_acf(y.dropna(), lags=lags, ax=axes[0])
    plot_pacf(y.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF — {title}")
    axes[1].set_title(f"PACF — {title}")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_decomposition(y: pd.Series, period: int = 12, title: str = ""):
    res = STL(y.dropna(), period=period, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, comp, name in zip(
        axes,
        [y.dropna(), res.trend, res.seasonal, res.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
    ):
        comp.plot(ax=ax, color="#1f4068")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"STL — {title}")
    fig.tight_layout()
    return fig


def plot_panel(df: pd.DataFrame, title: str = "Panel de series"):
    n = df.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, df.columns):
        df[col].plot(ax=ax, color="#1f4068")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


_MODEL_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


def plot_forecast_fan(
    y: pd.Series,
    forecasts: dict,
    origin_idx: int,
    horizon: int,
    history_tail: int = 36,
    title: str = "",
    ax=None,
    variable_idx: int | None = None,
    colors: dict | None = None,
    show_legend: bool = True,
    linewidth: float = 1.4,
):
    """
    Overlay multiple model forecasts on the recent history, with 80 % and
    95 % prediction bands per model.

    Parameters
    ----------
    y : pd.Series
        Realised target. Plotted in dark blue.
    forecasts : dict[str, dict]
        Output of `predict_all_at_last_origin`. Each value must contain
        `mean`, `lo80`, `hi80`, `lo95`, `hi95`. For multivariate models
        pass `variable_idx` to pick the column.
    origin_idx : int
        Index in `y` where the forecast starts.
    horizon : int
    history_tail : int
        How many historical points to draw before the origin.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))

    hist_start = max(0, origin_idx - history_tail)
    hist = y.iloc[hist_start:origin_idx + 1]
    ax.plot(hist.index, hist.values, color="#1f4068", linewidth=linewidth, label="observado")
    if origin_idx + horizon <= len(y):
        realised = y.iloc[origin_idx:origin_idx + horizon + 1]
        ax.plot(realised.index, realised.values, color="#1f4068", linewidth=linewidth,
                linestyle="--", alpha=0.55, label="realizado")

    future_idx = y.index[origin_idx:origin_idx + horizon]
    for i, (name, f) in enumerate(forecasts.items()):
        if colors is not None and name in colors:
            color = colors[name]
        else:
            color = _MODEL_COLORS[i % len(_MODEL_COLORS)]
        mean = f["mean"]
        if variable_idx is not None and mean.ndim > 1:
            mean = mean[:, variable_idx]
        ax.plot(future_idx, mean, color=color, linewidth=linewidth, label=name)
        for tag, alpha in (("80", 0.18), ("95", 0.09)):
            lo, hi = f.get(f"lo{tag}"), f.get(f"hi{tag}")
            if lo is None or hi is None:
                continue
            if variable_idx is not None and lo.ndim > 1:
                lo, hi = lo[:, variable_idx], hi[:, variable_idx]
            ax.fill_between(future_idx, lo, hi, color=color, alpha=alpha)

    ax.axvline(y.index[origin_idx], color="grey", linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(loc="best", fontsize=8, ncol=2)
    return ax


def plot_forecast_grid(
    y: pd.Series,
    forecasts: dict,
    origin_idx: int,
    horizon: int,
    history_tail: int = 36,
    ncols: int = 2,
    figsize_per: tuple = (6.0, 3.0),
    title: str = "",
    variable_idx: int | None = None,
    colors: dict | None = None,
):
    """
    Open `plot_forecast_fan` per-model into a grid: one subplot per
    model with the same historical tail and forecast bands. Avoids the
    band-overlap legibility issue of the overlay version.
    """
    n = len(forecasts)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        sharey=True, sharex=True,
    )
    axes = np.atleast_1d(axes).flatten()
    for i, (name, f) in enumerate(forecasts.items()):
        plot_forecast_fan(
            y, {name: f}, origin_idx=origin_idx, horizon=horizon,
            history_tail=history_tail, title=name, ax=axes[i],
            variable_idx=variable_idx, colors=colors,
        )
        axes[i].legend(loc="best", fontsize=7, ncol=1)
    for j in range(n, nrows * ncols):
        axes[j].set_visible(False)
    if title:
        fig.suptitle(title, y=1.01, fontsize=11)
    fig.tight_layout()
    return fig


def plot_forecast_matrix(
    Y: pd.DataFrame,
    forecasts: dict,
    origin_idx: int,
    horizon: int,
    history_tail: int = 36,
    figsize_per: tuple = (4.0, 2.4),
    title: str = "",
):
    """
    Matrix layout for multivariate forecasts: rows = variables, columns
    = models. Each cell is `plot_forecast_fan` restricted to that
    (variable, model) pair.
    """
    nvar = Y.shape[1]
    nmod = len(forecasts)
    fig, axes = plt.subplots(
        nvar, nmod,
        figsize=(figsize_per[0] * nmod, figsize_per[1] * nvar),
        sharex=True,
    )
    axes = np.atleast_2d(axes)
    for j, var in enumerate(Y.columns):
        for i, (name, f) in enumerate(forecasts.items()):
            ax = axes[j, i]
            plot_forecast_fan(
                Y[var], {name: f}, origin_idx=origin_idx, horizon=horizon,
                history_tail=history_tail, title="",
                ax=ax, variable_idx=j,
            )
            ax.legend().remove() if ax.get_legend() else None
            if j == 0:
                ax.set_title(name, fontsize=10)
            if i == 0:
                ax.set_ylabel(var, fontsize=10)
    if title:
        fig.suptitle(title, y=1.01, fontsize=11)
    fig.tight_layout()
    return fig
