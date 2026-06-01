"""
Markov Switching AR forecasting model.

Fitted via statsmodels MarkovAutoregression (EM + MLE).
Forecasts are simulation-based: given the filtered regime probabilities at T
and the estimated transition matrix and AR parameters, we simulate N forward
paths and aggregate (mean for point forecasts, quantiles for intervals).
"""

import warnings
import numpy as np
from .base import BaseModel

_FIT_ERRORS = (np.linalg.LinAlgError, RuntimeError, ValueError)


class MarkovSwitchingARModel(BaseModel):
    """
    AR(1) Markov Switching model — core for Exp 1.20.

    Because S_t is latent, multi-step analytic forecasts are complex.
    Instead, n_sim forward paths are simulated from the estimated model at
    each forecast call; paths are cached per horizon so that forecast(),
    forecast_intervals(), and compute_crps() share the same simulation.

    When statsmodels fails to converge (SVD non-convergence, degenerate
    steady-state probabilities, etc.) on a particular Monte Carlo draw,
    fit() sets _fit_failed=True and all forecast methods return NaN arrays
    so that draw is excluded from aggregate statistics via np.nanmean.
    """

    def __init__(
        self,
        k_regimes: int = 2,
        order: int = 1,
        n_sim: int = 1000,
        seed: int = 42,
    ):
        self.k_regimes = k_regimes
        self.order = order
        self.n_sim = n_sim
        self._rng = np.random.default_rng(seed)
        self._fitted = None
        self._fit_failed = False
        self._y_last: float = 0.0
        self._paths_cache: dict = {}

    def fit(self, y_train: np.ndarray, **kwargs):
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )
        self._paths_cache = {}
        self._fit_failed = False
        self._fitted = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovAutoregression(
                y_train,
                k_regimes=self.k_regimes,
                order=self.order,
                switching_ar=True,
                switching_variance=True,
            )
            try:
                self._fitted = model.fit(
                    disp=False,
                    em_iter=5,
                    search_reps=1,
                    maxiter=100,
                )
            except _FIT_ERRORS:
                # EM or optimizer failed; retry without EM pre-conditioning.
                try:
                    self._fitted = model.fit(
                        disp=False,
                        em_iter=0,
                        search_reps=0,
                        maxiter=200,
                    )
                except _FIT_ERRORS:
                    self._fit_failed = True
                    return

        self._y_last = float(y_train[-1])

    def _extract_params(self):
        res = self._fitted
        raw = res.params
        p = raw.to_dict() if hasattr(raw, 'to_dict') else dict(zip(res.model.param_names, raw))
        k = self.k_regimes

        mu = np.array([p[f"const[{i}]"] for i in range(k)])
        phi = np.array([p[f"ar.L1[{i}]"] for i in range(k)])
        sigma = np.array([np.sqrt(abs(p[f"sigma2[{i}]"])) for i in range(k)])

        # Row-stochastic transition matrix (statsmodels names: p[i->j])
        p00 = np.clip(float(p["p[0->0]"]), 0.0, 1.0)
        p10 = np.clip(float(p["p[1->0]"]), 0.0, 1.0)
        if not (np.isfinite(p00) and np.isfinite(p10)):
            p00, p10 = 0.9, 0.1
        P = np.array([[p00, 1.0 - p00], [p10, 1.0 - p10]])

        # Filtered regime probabilities at last observation (ndarray or DataFrame)
        fmp = res.filtered_marginal_probabilities
        filt = fmp.iloc[-1].values.astype(float) if hasattr(fmp, 'iloc') else np.asarray(fmp[-1], dtype=float)
        if not np.all(np.isfinite(filt)) or filt.sum() == 0:
            filt = np.ones(k, dtype=float)
        filt = np.clip(filt, 1e-10, 1.0)
        filt /= filt.sum()

        return mu, phi, sigma, P, filt

    def _get_paths(self, horizon: int) -> np.ndarray:
        if horizon not in self._paths_cache:
            mu, phi, sigma, P, filt_probs = self._extract_params()
            rng = self._rng
            paths = np.empty((self.n_sim, horizon))

            for sim in range(self.n_sim):
                s = rng.choice(self.k_regimes, p=filt_probs)
                y_prev = self._y_last
                for h in range(horizon):
                    s = rng.choice(self.k_regimes, p=P[s])
                    y_next = (
                        mu[s]
                        + phi[s] * y_prev
                        + sigma[s] * rng.standard_normal()
                    )
                    paths[sim, h] = y_next
                    y_prev = y_next

            self._paths_cache[horizon] = paths
        return self._paths_cache[horizon]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self._fit_failed or self._fitted is None:
            return np.full(horizon, np.nan)
        return self._get_paths(horizon).mean(axis=0)

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(
        self, horizon: int, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._fit_failed or self._fitted is None:
            nan = np.full(horizon, np.nan)
            return nan, nan
        paths = self._get_paths(horizon)
        alpha = 1.0 - level
        lo = np.quantile(paths, alpha / 2.0, axis=0)
        hi = np.quantile(paths, 1.0 - alpha / 2.0, axis=0)
        return lo, hi

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        if self._fit_failed or self._fitted is None:
            return np.full(horizon, np.nan)
        from properscoring import crps_ensemble
        paths = self._get_paths(horizon)
        return np.array([
            crps_ensemble(float(y_true[h]), paths[:, h])
            for h in range(horizon)
        ])

    @property
    def name(self) -> str:
        return f"MS-AR({self.order})"
