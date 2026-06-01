"""
VAR, VECM, and VAR+GARCH-diagonal forecasting models.
"""

import warnings
import numpy as np
from .base import BaseModel

_FIT_ERRORS = (np.linalg.LinAlgError, RuntimeError, ValueError, Exception)


class VARModel(BaseModel):
    """
    VAR(p) model via statsmodels. Accepts y_train of shape (T_train, k).
    Supports joint prediction intervals via the analytic MSE formula.
    """

    def __init__(self, lags: int = 1):
        self.lags = lags
        self._fitted = None
        self._fit_failed = False
        self._y_last = None

    def fit(self, y_train: np.ndarray, **kwargs):
        from statsmodels.tsa.vector_ar.var_model import VAR
        self._fit_failed = False
        self._fitted = None
        self._y_last = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = VAR(y_train)
                self._fitted = model.fit(maxlags=self.lags, ic=None, trend='c')
                self._y_last = y_train[-self.lags:]  # (lags, k)
            except _FIT_ERRORS:
                self._fit_failed = True

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._fit_failed or self._fitted is None:
            k = self._fitted.k_ar if self._fitted is not None else 1
            return np.full((horizon, k), np.nan)
        try:
            return self._fitted.forecast(y=self._y_last, steps=horizon)  # (horizon, k)
        except Exception:
            k = self._y_last.shape[1] if self._y_last is not None else 1
            return np.full((horizon, k), np.nan)

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(
        self, horizon: int, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._fit_failed or self._fitted is None:
            k = 1
            return np.full((horizon, k), np.nan), np.full((horizon, k), np.nan)
        alpha = 1.0 - level
        try:
            fc, lower, upper = self._fitted.forecast_interval(
                y=self._y_last, steps=horizon, alpha=alpha
            )
            return lower, upper  # each (horizon, k)
        except Exception:
            k = self._y_last.shape[1] if self._y_last is not None else 1
            nan = np.full((horizon, k), np.nan)
            return nan, nan

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_gaussian
        if self._fit_failed or self._fitted is None:
            k = self._y_last.shape[1] if self._y_last is not None else 1
            return np.full((horizon, k), np.nan)
        try:
            mu = self._fitted.forecast(y=self._y_last, steps=horizon)  # (horizon, k)
            cov_list = self._fitted.forecast_cov(steps=horizon)        # list of (k, k)
            k = mu.shape[1]
            result = np.empty((horizon, k))
            for h in range(horizon):
                sigma = np.sqrt(np.maximum(np.diag(cov_list[h]), 1e-12))  # (k,)
                result[h] = crps_gaussian(y_true[h], mu[h], sigma)
            return result
        except Exception:
            k = self._y_last.shape[1] if self._y_last is not None else 1
            return np.full((horizon, k), np.nan)

    @property
    def name(self) -> str:
        return f"VAR({self.lags})"


class VECMModel(BaseModel):
    """
    VECM via statsmodels. Accepts y_train of shape (T_train, k).

    Point forecasts via statsmodels predict().
    Intervals and CRPS via residual bootstrap: simulates n_sim forward paths
    using the estimated VECM-in-levels equation
        Y_t = (I + α β') Y_{t-1} + α δ + ε_t
    with iid resampled residuals.  Works for k_ar_diff=1 (default).
    """

    def __init__(self, coint_rank: int = 1, k_ar_diff: int = 1,
                 n_sim: int = 500, seed: int = 42):
        self.coint_rank = coint_rank
        self.k_ar_diff = k_ar_diff
        self._n_sim = n_sim
        self._rng = np.random.default_rng(seed)
        self._fitted = None
        self._fit_failed = False
        self._k = None
        self._Pi = None           # (I + alpha @ beta.T), shape (k, k)
        self._const_ecm = None    # alpha @ det_coef_coint.flatten(), shape (k,)
        self._resid = None        # (T_eff, k) — bootstrap pool
        self._y_last_level = None # (k,) — last observed level Y_T
        self._paths_cache: dict = {}

    def fit(self, y_train: np.ndarray, **kwargs):
        from statsmodels.tsa.vector_ar.vecm import VECM
        self._fit_failed = False
        self._fitted = None
        self._k = y_train.shape[1]
        self._Pi = None
        self._const_ecm = None
        self._resid = None
        self._y_last_level = None
        self._paths_cache = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = VECM(
                    y_train,
                    k_ar_diff=self.k_ar_diff,
                    coint_rank=self.coint_rank,
                    deterministic='ci',
                )
                self._fitted = model.fit()
                k = self._k
                alpha = self._fitted.alpha               # (k, r)
                beta = self._fitted.beta                 # (k, r)
                self._Pi = np.eye(k) + alpha @ beta.T   # (k, k)
                det = getattr(self._fitted, 'det_coef_coint', None)
                if det is not None and det.size > 0:
                    self._const_ecm = alpha @ det.flatten()[:self.coint_rank]
                else:
                    self._const_ecm = np.zeros(k)
                self._resid = self._fitted.resid         # (T_eff, k)
                self._y_last_level = y_train[-1].copy()  # (k,) last level
            except _FIT_ERRORS:
                self._fit_failed = True

    def _get_paths(self, horizon: int) -> np.ndarray:
        """Residual-bootstrap paths: (n_sim, horizon, k)."""
        if horizon in self._paths_cache:
            return self._paths_cache[horizon]
        n_resid = self._resid.shape[0]
        k = self._k
        paths = np.empty((self._n_sim, horizon, k))
        for sim in range(self._n_sim):
            y = self._y_last_level.copy()
            for h in range(horizon):
                eps = self._resid[self._rng.integers(0, n_resid)]
                y = self._Pi @ y + self._const_ecm + eps
                paths[sim, h] = y
        self._paths_cache[horizon] = paths
        return paths

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._fit_failed or self._fitted is None:
            k = self._k if self._k is not None else 2
            return np.full((horizon, k), np.nan)
        try:
            return self._fitted.predict(steps=horizon)  # (horizon, k)
        except Exception:
            k = self._k if self._k is not None else 2
            return np.full((horizon, k), np.nan)

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(
        self, horizon: int, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._fit_failed or self._fitted is None:
            k = self._k if self._k is not None else 2
            return np.full((horizon, k), np.nan), np.full((horizon, k), np.nan)
        try:
            paths = self._get_paths(horizon)   # (n_sim, horizon, k)
            alpha = 1.0 - level
            lo = np.quantile(paths, alpha / 2.0, axis=0)       # (horizon, k)
            hi = np.quantile(paths, 1.0 - alpha / 2.0, axis=0)
            return lo, hi
        except Exception:
            k = self._k if self._k is not None else 2
            return np.full((horizon, k), np.nan), np.full((horizon, k), np.nan)

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_ensemble
        if self._fit_failed or self._fitted is None:
            k = self._k if self._k is not None else 2
            return np.full((horizon, k), np.nan)
        try:
            paths = self._get_paths(horizon)   # (n_sim, horizon, k)
            result = np.empty((horizon, self._k))
            for j in range(self._k):
                # paths[:, :, j].T → (horizon, n_sim) ensemble per step
                result[:, j] = crps_ensemble(y_true[:, j], paths[:, :, j].T)
            return result
        except Exception:
            return np.full((horizon, self._k), np.nan)

    @property
    def name(self) -> str:
        return f"VECM(r={self.coint_rank})"


class VARGARCHDiagonalModel(BaseModel):
    """
    VAR(1) mean + GARCH(1,1) diagonal on residuals. Exp 2.6.

    Point forecasts: VAR(1) conditional mean.
    Intervals: simulation-based using per-equation GARCH-conditional variance.
    Paths are cached so forecast(), forecast_intervals(), and compute_crps()
    share the same simulation.
    """

    def __init__(self, n_sim: int = 500, seed: int = 42):
        self.n_sim = n_sim
        self._rng = np.random.default_rng(seed)
        self._var_fitted = None
        self._garch_params = None  # list of dicts per equation: {omega, alpha, beta}
        self._fit_failed = False
        self._y_last = None
        self._resid_last = None
        self._sigma2_last = None
        self._k = None
        self._paths_cache: dict = {}

    def fit(self, y_train: np.ndarray, **kwargs):
        from statsmodels.tsa.vector_ar.var_model import VAR
        from arch import arch_model

        self._fit_failed = False
        self._var_fitted = None
        self._garch_params = None
        self._paths_cache = {}
        self._k = y_train.shape[1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Step 1: fit VAR(1)
                var_model = VAR(y_train)
                self._var_fitted = var_model.fit(maxlags=1, ic=None, trend='c')
                self._y_last = y_train[-1:]  # (1, k)

                # Step 2: get in-sample residuals, fit GARCH(1,1) per equation
                resid = self._var_fitted.resid  # (T_train - 1, k)
                self._garch_params = []
                self._sigma2_last = np.zeros(self._k)
                self._resid_last = resid[-1]  # last residual (k,)

                for j in range(self._k):
                    am = arch_model(resid[:, j], mean='Zero', vol='GARCH', p=1, q=1)
                    gres = am.fit(disp='off', show_warning=False)
                    omega = float(gres.params['omega'])
                    alpha = float(gres.params['alpha[1]'])
                    beta = float(gres.params['beta[1]'])
                    # clip to ensure stationarity
                    alpha = np.clip(alpha, 1e-6, 0.5)
                    beta = np.clip(beta, 1e-6, 0.98 - alpha)
                    self._garch_params.append({
                        'omega': omega, 'alpha': alpha, 'beta': beta
                    })
                    # last conditional variance
                    uncond = omega / (1.0 - alpha - beta)
                    self._sigma2_last[j] = max(uncond, 1e-8)

            except _FIT_ERRORS:
                self._fit_failed = True

    def _get_paths(self, horizon: int) -> np.ndarray:
        """Simulate n_sim forward paths → (n_sim, horizon, k)."""
        if horizon in self._paths_cache:
            return self._paths_cache[horizon]

        A = self._var_fitted.coefs[0]        # (k, k)
        intercept = self._var_fitted.intercept  # (k,)
        k = self._k
        gp = self._garch_params

        paths = np.empty((self.n_sim, horizon, k))
        z = self._rng.standard_normal((self.n_sim, horizon, k))

        for sim in range(self.n_sim):
            y_prev = self._y_last[0].copy()   # (k,)
            u_prev = self._resid_last.copy()  # (k,)
            sig2 = self._sigma2_last.copy()   # (k,)

            for h in range(horizon):
                sig2 = np.array([
                    max(gp[j]['omega'] + gp[j]['alpha'] * u_prev[j]**2
                        + gp[j]['beta'] * sig2[j], 1e-8)
                    for j in range(k)
                ])
                innovations = np.sqrt(sig2) * z[sim, h]
                y_next = intercept + A @ y_prev + innovations
                paths[sim, h] = y_next
                y_prev = y_next
                u_prev = innovations

        self._paths_cache[horizon] = paths
        return paths

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._fit_failed or self._var_fitted is None:
            k = self._k if self._k else 2
            return np.full((horizon, k), np.nan)
        try:
            return self._get_paths(horizon).mean(axis=0)  # (horizon, k)
        except Exception:
            return np.full((horizon, self._k), np.nan)

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(
        self, horizon: int, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._fit_failed or self._var_fitted is None:
            k = self._k if self._k else 2
            return np.full((horizon, k), np.nan), np.full((horizon, k), np.nan)
        paths = self._get_paths(horizon)  # (n_sim, horizon, k)
        alpha = 1.0 - level
        lo = np.quantile(paths, alpha / 2.0, axis=0)  # (horizon, k)
        hi = np.quantile(paths, 1.0 - alpha / 2.0, axis=0)
        return lo, hi

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_ensemble
        if self._fit_failed or self._var_fitted is None:
            return np.full((horizon, self._k or 2), np.nan)
        try:
            paths = self._get_paths(horizon)   # (n_sim, horizon, k)
            result = np.empty((horizon, self._k))
            for j in range(self._k):
                result[:, j] = crps_ensemble(y_true[:, j], paths[:, :, j].T)
            return result
        except Exception:
            return np.full((horizon, self._k or 2), np.nan)

    @property
    def name(self) -> str:
        return "VAR(1)+GARCH-diag"
