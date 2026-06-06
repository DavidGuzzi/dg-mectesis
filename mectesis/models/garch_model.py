"""
ARCH/GARCH/EGARCH forecasting models using the arch library (Kevin Sheppard).
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from .base import BaseModel


def _z(level: float) -> float:
    return stats.norm.ppf(1.0 - (1.0 - level) / 2.0)


class ARARCHModel(BaseModel):
    """
    AR(p)+ARCH(q) model — core for Exp 1.9.

    Fitted via arch_model(mean='AR', vol='GARCH', q=0).
    ARCH(q) is GARCH(q, 0): q lagged squared residuals, no lagged variance.
    """

    def __init__(self, ar_lags: int = 1, p: int = 1):
        self.ar_lags = ar_lags
        self.p = p
        self._fitted    = None
        self._scale     = 100.0
        self._fc_cache  = {}

    def fit(self, y_train: np.ndarray, **kwargs):
        self._fc_cache = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                y_train * self._scale,
                mean='AR', lags=self.ar_lags,
                vol='GARCH', p=self.p, q=0,
                dist='normal', rescale=False,
            )
            self._fitted = am.fit(disp='off')

    def _get_forecast(self, horizon: int):
        if horizon not in self._fc_cache:
            self._fc_cache[horizon] = self._fitted.forecast(
                horizon=horizon, reindex=False)
        return self._fc_cache[horizon]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._fitted is None:
            raise ValueError("Call fit() before forecast().")
        return self._get_forecast(horizon).mean.values[-1] / self._scale

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        if self._fitted is None:
            raise ValueError("Call fit() before forecast_intervals().")
        fc      = self._get_forecast(horizon)
        mean_fc = fc.mean.values[-1] / self._scale
        var_fc  = fc.variance.values[-1] / (self._scale ** 2)
        std_fc  = np.sqrt(np.maximum(var_fc, 0.0))
        z       = _z(level)
        return mean_fc - z * std_fc, mean_fc + z * std_fc

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_gaussian
        fc = self._get_forecast(horizon)
        mean_fc = fc.mean.values[-1] / self._scale
        std_fc = np.maximum(np.sqrt(fc.variance.values[-1]) / self._scale, 1e-8)
        return crps_gaussian(y_true, mean_fc, std_fc)

    @property
    def name(self) -> str:
        return f"AR({self.ar_lags})+ARCH({self.p})"


class ARGARCHModel(BaseModel):
    """
    AR(p)+GARCH(p,q) model — core for Exp 1.10, additional for Exp 1.12.
    """

    def __init__(self, ar_lags: int = 1, p: int = 1, q: int = 1):
        self.ar_lags = ar_lags
        self.p = p
        self.q = q
        self._fitted   = None
        self._scale    = 100.0
        self._fc_cache = {}

    def fit(self, y_train: np.ndarray, **kwargs):
        self._fc_cache = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                y_train * self._scale,
                mean='AR', lags=self.ar_lags,
                vol='GARCH', p=self.p, q=self.q,
                dist='normal', rescale=False,
            )
            self._fitted = am.fit(disp='off')

    def _get_forecast(self, horizon: int):
        if horizon not in self._fc_cache:
            self._fc_cache[horizon] = self._fitted.forecast(
                horizon=horizon, reindex=False)
        return self._fc_cache[horizon]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._fitted is None:
            raise ValueError("Call fit() before forecast().")
        return self._get_forecast(horizon).mean.values[-1] / self._scale

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        if self._fitted is None:
            raise ValueError("Call fit() before forecast_intervals().")
        fc      = self._get_forecast(horizon)
        mean_fc = fc.mean.values[-1] / self._scale
        var_fc  = fc.variance.values[-1] / (self._scale ** 2)
        std_fc  = np.sqrt(np.maximum(var_fc, 0.0))
        z       = _z(level)
        return mean_fc - z * std_fc, mean_fc + z * std_fc

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_gaussian
        fc = self._get_forecast(horizon)
        mean_fc = fc.mean.values[-1] / self._scale
        std_fc = np.maximum(np.sqrt(fc.variance.values[-1]) / self._scale, 1e-8)
        return crps_gaussian(y_true, mean_fc, std_fc)

    @property
    def name(self) -> str:
        return f"AR({self.ar_lags})+GARCH({self.p},{self.q})"


class ARIMAXGARCHModel(BaseModel):
    """
    AR(p)+GARCH(p,q) with exogenous covariate in the mean — challenger benchmark
    for the C-D block (ARIMAX-GARCH DGP). Mirrors ARGARCHModel but plumbs X_train
    into fit() and X_future into forecast() / forecast_intervals() / compute_crps()
    via arch_model(mean='ARX', x=...).

    The variance equation is plain GARCH(p,q): the arch library does not support
    exogenous-in-variance, so the delta_var * X_t^2 term of the DGP is not
    modelled. This is a deliberate misspecification (see
    docs/benchmark_garch_covariables_pendiente.md).

    When X_train is None at fit time, the model falls back to mean='AR' and
    behaves like ARGARCHModel.
    """

    def __init__(self, ar_lags: int = 1, p: int = 1, q: int = 1,
                 name_suffix: str = "con X"):
        self.ar_lags     = ar_lags
        self.p           = p
        self.q           = q
        self._name_suffix = name_suffix
        self._fitted     = None
        self._scale      = 100.0
        self._fc_cache   = {}
        self._has_exog   = False
        self._x_cols     = None

    @staticmethod
    def _as_df(X: np.ndarray, cols) -> pd.DataFrame:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        return pd.DataFrame(X, columns=cols)

    def fit(self, y_train: np.ndarray, X_train: np.ndarray = None, **kwargs):
        self._fc_cache = {}
        self._has_exog = X_train is not None and np.asarray(X_train).shape[0] > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._has_exog:
                X_arr = np.asarray(X_train)
                if X_arr.ndim == 1:
                    X_arr = X_arr[:, np.newaxis]
                self._x_cols = [f"x{i}" for i in range(X_arr.shape[1])]
                X_df = pd.DataFrame(X_arr, columns=self._x_cols)
                am = arch_model(
                    y_train * self._scale,
                    x=X_df,
                    mean='ARX', lags=self.ar_lags,
                    vol='GARCH', p=self.p, q=self.q,
                    dist='normal', rescale=False,
                )
            else:
                self._x_cols = None
                am = arch_model(
                    y_train * self._scale,
                    mean='AR', lags=self.ar_lags,
                    vol='GARCH', p=self.p, q=self.q,
                    dist='normal', rescale=False,
                )
            self._fitted = am.fit(disp='off')

    def _get_forecast(self, horizon: int, X_future: np.ndarray = None):
        key = (horizon, id(X_future) if X_future is not None else None)
        if key not in self._fc_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self._has_exog:
                    if X_future is None:
                        raise ValueError(
                            "ARIMAXGARCHModel fitted with X_train requires "
                            "X_future at forecast time."
                        )
                    X_arr = np.asarray(X_future)
                    if X_arr.ndim == 1:
                        X_arr = X_arr[:, np.newaxis]
                    # arch's ARX.forecast expects x as dict[col -> (n_origins, horizon)].
                    # For single-origin forecasts, n_origins=1, so reshape each
                    # column's first `horizon` values into (1, horizon).
                    x_dict = {
                        col: X_arr[:horizon, i].reshape(1, horizon)
                        for i, col in enumerate(self._x_cols)
                    }
                    self._fc_cache[key] = self._fitted.forecast(
                        horizon=horizon, x=x_dict, reindex=False,
                    )
                else:
                    self._fc_cache[key] = self._fitted.forecast(
                        horizon=horizon, reindex=False,
                    )
        return self._fc_cache[key]

    def forecast(self, horizon: int, X_future: np.ndarray = None,
                 **kwargs) -> np.ndarray:
        if self._fitted is None:
            raise ValueError("Call fit() before forecast().")
        return self._get_forecast(horizon, X_future).mean.values[-1] / self._scale

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95,
                           X_future: np.ndarray = None, **kwargs):
        if self._fitted is None:
            raise ValueError("Call fit() before forecast_intervals().")
        fc      = self._get_forecast(horizon, X_future)
        mean_fc = fc.mean.values[-1] / self._scale
        var_fc  = fc.variance.values[-1] / (self._scale ** 2)
        std_fc  = np.sqrt(np.maximum(var_fc, 0.0))
        z       = _z(level)
        return mean_fc - z * std_fc, mean_fc + z * std_fc

    @property
    def supports_crps(self) -> bool:
        return True

    @property
    def supports_covariates(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int,
                     X_future: np.ndarray = None, **kwargs) -> np.ndarray:
        from properscoring import crps_gaussian
        fc      = self._get_forecast(horizon, X_future)
        mean_fc = fc.mean.values[-1] / self._scale
        std_fc  = np.maximum(np.sqrt(fc.variance.values[-1]) / self._scale, 1e-8)
        return crps_gaussian(y_true, mean_fc, std_fc)

    @property
    def name(self) -> str:
        base = f"AR({self.ar_lags})+GARCH({self.p},{self.q})"
        return f"{base} {self._name_suffix}".strip()


class GARCHModel(BaseModel):
    """
    Zero-mean GARCH(p,q) — core for Exp 1.11.
    Point forecasts are 0; intervals widen with conditional variance.
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self._fitted   = None
        self._scale    = 100.0
        self._fc_cache = {}

    def fit(self, y_train: np.ndarray, **kwargs):
        self._fc_cache = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                y_train * self._scale,
                mean='Zero',
                vol='GARCH', p=self.p, q=self.q,
                dist='normal', rescale=False,
            )
            self._fitted = am.fit(disp='off')

    def _get_forecast(self, horizon: int):
        if horizon not in self._fc_cache:
            self._fc_cache[horizon] = self._fitted.forecast(
                horizon=horizon, reindex=False)
        return self._fc_cache[horizon]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._fitted is None:
            raise ValueError("Call fit() before forecast().")
        fc = self._get_forecast(horizon)
        # Zero-mean model: mean forecasts are 0
        try:
            mean_fc = fc.mean.values[-1] / self._scale
        except (AttributeError, IndexError):
            mean_fc = np.zeros(horizon)
        return mean_fc

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        if self._fitted is None:
            raise ValueError("Call fit() before forecast_intervals().")
        fc = self._get_forecast(horizon)
        try:
            mean_fc = fc.mean.values[-1] / self._scale
        except (AttributeError, IndexError):
            mean_fc = np.zeros(horizon)
        var_fc  = fc.variance.values[-1] / (self._scale ** 2)
        std_fc  = np.sqrt(np.maximum(var_fc, 0.0))
        z       = _z(level)
        return mean_fc - z * std_fc, mean_fc + z * std_fc

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_gaussian
        fc = self._get_forecast(horizon)
        try:
            mean_fc = fc.mean.values[-1] / self._scale
        except (AttributeError, IndexError):
            mean_fc = np.zeros(horizon)
        std_fc = np.maximum(np.sqrt(fc.variance.values[-1]) / self._scale, 1e-8)
        return crps_gaussian(y_true, mean_fc, std_fc)

    @property
    def name(self) -> str:
        return f"GARCH({self.p},{self.q})-ZeroMean"


class ARGJRGARCHModel(BaseModel):
    """
    AR(p)+GJR-GARCH(p,o,q) with leverage effect — core for Exp 1.12.

    The 'o' parameter adds the asymmetric term: gamma*eps_{t-1}^2*1{eps_{t-1}<0}.
    """

    def __init__(self, ar_lags: int = 1, p: int = 1, o: int = 1, q: int = 1):
        self.ar_lags = ar_lags
        self.p = p
        self.o = o
        self.q = q
        self._fitted   = None
        self._scale    = 100.0
        self._fc_cache = {}

    def fit(self, y_train: np.ndarray, **kwargs):
        self._fc_cache = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                y_train * self._scale,
                mean='AR', lags=self.ar_lags,
                vol='GARCH', p=self.p, o=self.o, q=self.q,
                dist='normal', rescale=False,
            )
            self._fitted = am.fit(disp='off')

    def _get_forecast(self, horizon: int):
        if horizon not in self._fc_cache:
            self._fc_cache[horizon] = self._fitted.forecast(
                horizon=horizon, reindex=False)
        return self._fc_cache[horizon]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._fitted is None:
            raise ValueError("Call fit() before forecast().")
        return self._get_forecast(horizon).mean.values[-1] / self._scale

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        if self._fitted is None:
            raise ValueError("Call fit() before forecast_intervals().")
        fc      = self._get_forecast(horizon)
        mean_fc = fc.mean.values[-1] / self._scale
        var_fc  = fc.variance.values[-1] / (self._scale ** 2)
        std_fc  = np.sqrt(np.maximum(var_fc, 0.0))
        z       = _z(level)
        return mean_fc - z * std_fc, mean_fc + z * std_fc

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_gaussian
        fc = self._get_forecast(horizon)
        mean_fc = fc.mean.values[-1] / self._scale
        std_fc = np.maximum(np.sqrt(fc.variance.values[-1]) / self._scale, 1e-8)
        return crps_gaussian(y_true, mean_fc, std_fc)

    @property
    def name(self) -> str:
        return f"AR({self.ar_lags})+GJR-GARCH({self.p},{self.o},{self.q})"


class AREGARCHModel(BaseModel):
    """
    AR(p)+EGARCH(p,o,q) model — core for Exp 1.21.

    EGARCH models log-variance, ensuring positivity without
    sign restrictions. Multi-step variance forecasts use arch's simulation
    method since no analytic closed form exists for h > 1.

    Intervals and CRPS use the simulated predictive distribution:
    mean ± z * std where std = sqrt(Var[Y_{T+h} | Y_T]) across simulations.
    """

    def __init__(
        self,
        ar_lags: int = 1,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        n_sim: int = 200,
    ):
        self.ar_lags = ar_lags
        self.p = p
        self.o = o
        self.q = q
        self.n_sim = n_sim
        self._fitted     = None
        self._fit_failed = False
        self._scale      = 100.0
        self._fc_cache   = {}

    def fit(self, y_train: np.ndarray, **kwargs):
        self._fc_cache   = {}
        self._fit_failed = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                y_train * self._scale,
                mean='AR', lags=self.ar_lags,
                vol='EGARCH', p=self.p, o=self.o, q=self.q,
                dist='normal', rescale=False,
            )
            self._fitted = am.fit(disp='off')
        if self._fitted.convergence_flag != 0:
            self._fit_failed = True

    def _get_forecast(self, horizon: int):
        if horizon not in self._fc_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fc_cache[horizon] = self._fitted.forecast(
                    horizon=horizon,
                    method='simulation',
                    simulations=self.n_sim,
                    reindex=False,
                )
        return self._fc_cache[horizon]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._fit_failed or self._fitted is None:
            return np.full(horizon, np.nan)
        return self._get_forecast(horizon).mean.values[-1] / self._scale

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        if self._fit_failed or self._fitted is None:
            nan = np.full(horizon, np.nan)
            return nan, nan
        fc      = self._get_forecast(horizon)
        mean_fc = fc.mean.values[-1] / self._scale
        var_fc  = fc.variance.values[-1] / (self._scale ** 2)
        std_fc  = np.sqrt(np.maximum(var_fc, 0.0))
        z       = _z(level)
        return mean_fc - z * std_fc, mean_fc + z * std_fc

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        if self._fit_failed or self._fitted is None:
            return np.full(horizon, np.nan)
        from properscoring import crps_gaussian
        fc      = self._get_forecast(horizon)
        mean_fc = fc.mean.values[-1] / self._scale
        std_fc  = np.maximum(np.sqrt(fc.variance.values[-1]) / self._scale, 1e-8)
        return crps_gaussian(y_true, mean_fc, std_fc)

    @property
    def name(self) -> str:
        return f"AR({self.ar_lags})+EGARCH({self.p},{self.o},{self.q})"
