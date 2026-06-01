"""
Monte Carlo engines for forecast experiments with exogenous covariates.

DGPs in this family return {"y": np.ndarray, "X": np.ndarray} instead of
a plain ndarray so that the engine can split Y and X into train/test sets
and pass X_future (the known future covariate values) to models.

CovariateMonteCarloEngine
    Univariate target Y (T,) + covariates X (T, p).
    Returns {model_name: DataFrame} — same structure as MonteCarloEngine.

CovariateMultivariateEngine
    Multivariate target Y (T, k) + covariates X (T, p).
    Returns {model_name: {var_idx: DataFrame}} — same as MultivariateMonteCarloEngine.

Models that do not use covariates simply ignore the X_train / X_future kwargs
passed by the engine via **kwargs.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from mectesis.dgp.base import BaseDGP
from mectesis.models.base import BaseModel
from mectesis.metrics.decomposition import BiasVarianceMSE


class CovariateMonteCarloEngine:
    """
    Monte Carlo engine for univariate target + covariates.

    The DGP must implement:
        simulate(T, **dgp_params) -> {"y": (T,), "X": (T, p)}

    Models receive:
        fit(y_train, X_train=X_train)
        forecast(horizon, X_future=X_future)
        forecast_intervals(horizon, level=level, X_future=X_future)
        compute_crps(y_test, horizon, X_future=X_future)

    Models that do not accept X_train / X_future simply use **kwargs and ignore them.
    """

    def __init__(self, dgp: BaseDGP, models: List[BaseModel], seed: int = None):
        self.dgp    = dgp
        self.models = models
        self.seed   = seed

    def run_monte_carlo(
        self,
        n_sim: int,
        T: int,
        horizon: int,
        dgp_params: dict,
        levels: Tuple[float, ...] = (0.80, 0.95),
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run n_sim Monte Carlo replications with covariates.

        Returns {model_name: DataFrame} — same structure as MonteCarloEngine.
        """
        models_iv   = [m for m in self.models if m.supports_intervals]
        models_crps = [m for m in self.models if m.supports_crps]
        level_keys  = [int(l * 100) for l in levels]

        if verbose:
            print(f"  {n_sim} reps | T={T} | h=1–{horizon} | "
                  f"modelos: {[m.name for m in self.models]}")

        error_mats = {m.name: np.full((n_sim, horizon), np.nan) for m in self.models}
        cov_mats = {
            m.name: {lk: np.full((n_sim, horizon), np.nan) for lk in level_keys}
            for m in models_iv
        }
        wid_mats = {
            m.name: {lk: np.full((n_sim, horizon), np.nan) for lk in level_keys}
            for m in models_iv
        }
        winkler_mats = {
            m.name: {lk: np.full((n_sim, horizon), np.nan) for lk in level_keys}
            for m in models_iv
        }
        crps_mats = {
            m.name: np.full((n_sim, horizon), np.nan) for m in models_crps
        }

        t_start  = time.time()
        log_every = max(1, n_sim // 10)

        for s in range(n_sim):
            data = self.dgp.simulate(T=T, **dgp_params)
            y = data["y"]   # (T,)
            X = data["X"]   # (T, p)

            y_train, y_test = y[:T - horizon], y[T - horizon:]
            X_train, X_future = X[:T - horizon], X[T - horizon:]

            for model in self.models:
                cov  = model.supports_covariates
                fkw  = {"X_train": X_train} if cov else {}
                pkw  = {"X_future": X_future} if cov else {}
                try:
                    model.fit(y_train, **fkw)
                    y_hat = model.forecast(horizon, **pkw)
                    if y_hat is None or np.all(np.isnan(y_hat)):
                        continue
                    error_mats[model.name][s] = y_test - y_hat

                    if model in models_iv:
                        for level, lk in zip(levels, level_keys):
                            lo, hi = model.forecast_intervals(
                                horizon, level=level, **pkw
                            )
                            if lo is None or np.all(np.isnan(lo)):
                                continue
                            cov_mats[model.name][lk][s]     = (y_test >= lo) & (y_test <= hi)
                            wid_mats[model.name][lk][s]     = hi - lo
                            alpha_iv = 1.0 - level
                            penalty  = 2.0 / alpha_iv
                            winkler_mats[model.name][lk][s] = (
                                (hi - lo)
                                + penalty * np.maximum(lo - y_test, 0.0)
                                + penalty * np.maximum(y_test - hi, 0.0)
                            )

                    if model in models_crps:
                        crps = model.compute_crps(y_test, horizon, **pkw)
                        if crps is not None and not np.all(np.isnan(crps)):
                            crps_mats[model.name][s] = crps

                except Exception as exc:
                    if s == 0:
                        import warnings
                        warnings.warn(f"[{model.name}] falló en rep 0: {exc}", stacklevel=2)

            if verbose and (s + 1) % log_every == 0:
                elapsed = time.time() - t_start
                rate    = (s + 1) / elapsed
                eta     = (n_sim - s - 1) / rate
                print(f"    [{s+1:>{len(str(n_sim))}}/{n_sim}] "
                      f"{elapsed:5.0f}s transcurridos  ETA ~{eta:.0f}s")

        if verbose:
            print(f"  ✓ Completado en {time.time() - t_start:.1f}s")

        results = {}
        for model in self.models:
            mname   = model.name
            em      = error_mats[mname]
            valid   = ~np.any(np.isnan(em), axis=1)

            if valid.sum() == 0:
                rows = [{"horizon": h, "bias": np.nan, "variance": np.nan,
                         "mse": np.nan, "rmse": np.nan, "mae": np.nan}
                        for h in list(range(1, horizon + 1)) + ["avg_all"]]
                results[mname] = pd.DataFrame(rows)
                continue

            em_clean = em[valid]
            cov_d = wid_d = winkler_d = crps_d = None

            if model in models_iv:
                cov_d     = {lk: cov_mats[mname][lk][valid]     for lk in level_keys}
                wid_d     = {lk: wid_mats[mname][lk][valid]     for lk in level_keys}
                winkler_d = {lk: winkler_mats[mname][lk][valid] for lk in level_keys}

            if model in models_crps:
                crps_d = crps_mats[mname][valid]

            results[mname] = BiasVarianceMSE.compute_summary_table(
                em_clean,
                coverage_data=cov_d,
                width_data=wid_d,
                winkler_data=winkler_d,
                crps_data=crps_d,
            )

        return results


class CovariateMultivariateEngine:
    """
    Monte Carlo engine for multivariate target (T, k) + covariates (T, p).

    The DGP must implement:
        simulate(T, **dgp_params) -> {"y": (T, k), "X": (T, p)}

    Returns {model_name: {var_idx: DataFrame}} — same as MultivariateMonteCarloEngine.
    """

    def __init__(self, dgp: BaseDGP, models: List[BaseModel], seed: int = None):
        self.dgp    = dgp
        self.models = models
        self.seed   = seed

    def run_monte_carlo(
        self,
        n_sim: int,
        T: int,
        horizon: int,
        dgp_params: dict,
        levels: Tuple[float, ...] = (0.80, 0.95),
        verbose: bool = True,
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Run n_sim replications. Returns {model_name: {var_idx: DataFrame}}.
        """
        models_iv   = [m for m in self.models if m.supports_intervals]
        models_crps = [m for m in self.models if m.supports_crps]
        level_keys  = [int(l * 100) for l in levels]

        # Probe to get k
        probe = self.dgp.simulate(T=T, **dgp_params)
        k     = probe["y"].shape[1]

        if verbose:
            print(f"  {n_sim} reps | T={T} | h=1–{horizon} | k={k} variables | "
                  f"modelos: {[m.name for m in self.models]}")

        error_mats = {m.name: np.full((n_sim, horizon, k), np.nan) for m in self.models}
        cov_mats = {
            m.name: {lk: np.full((n_sim, horizon, k), np.nan) for lk in level_keys}
            for m in models_iv
        }
        wid_mats = {
            m.name: {lk: np.full((n_sim, horizon, k), np.nan) for lk in level_keys}
            for m in models_iv
        }
        winkler_mats = {
            m.name: {lk: np.full((n_sim, horizon, k), np.nan) for lk in level_keys}
            for m in models_iv
        }
        crps_mats = {
            m.name: np.full((n_sim, horizon, k), np.nan) for m in models_crps
        }

        t_start   = time.time()
        log_every = max(1, n_sim // 10)

        for s in range(n_sim):
            data = self.dgp.simulate(T=T, **dgp_params)
            y = data["y"]  # (T, k)
            X = data["X"]  # (T, p)

            y_train, y_test = y[:T - horizon], y[T - horizon:]
            X_train, X_future = X[:T - horizon], X[T - horizon:]

            for model in self.models:
                cov  = model.supports_covariates
                fkw  = {"X_train": X_train} if cov else {}
                pkw  = {"X_future": X_future} if cov else {}
                try:
                    model.fit(y_train, **fkw)
                    y_hat = model.forecast(horizon, **pkw)

                    if y_hat is None or np.all(np.isnan(y_hat)):
                        continue
                    if y_hat.ndim == 1:
                        y_hat = y_hat[:, np.newaxis]
                    if y_hat.shape != (horizon, k):
                        continue

                    error_mats[model.name][s] = y_test - y_hat

                    if model in models_iv:
                        for level, lk in zip(levels, level_keys):
                            lo, hi = model.forecast_intervals(
                                horizon, level=level, **pkw
                            )
                            if lo is None or np.all(np.isnan(lo)):
                                continue
                            if lo.ndim == 1:
                                lo = lo[:, np.newaxis]
                                hi = hi[:, np.newaxis]
                            if lo.shape != (horizon, k):
                                continue
                            cov_mats[model.name][lk][s]     = (y_test >= lo) & (y_test <= hi)
                            wid_mats[model.name][lk][s]     = hi - lo
                            alpha_iv = 1.0 - level
                            penalty  = 2.0 / alpha_iv
                            winkler_mats[model.name][lk][s] = (
                                (hi - lo)
                                + penalty * np.maximum(lo - y_test, 0.0)
                                + penalty * np.maximum(y_test - hi, 0.0)
                            )

                    if model in models_crps:
                        crps = model.compute_crps(y_test, horizon, **pkw)
                        if crps is not None and not np.all(np.isnan(crps)):
                            if crps.ndim == 1:
                                crps = crps[:, np.newaxis]
                            if crps.shape == (horizon, k):
                                crps_mats[model.name][s] = crps

                except Exception:
                    pass

            if verbose and (s + 1) % log_every == 0:
                elapsed = time.time() - t_start
                rate    = (s + 1) / elapsed
                eta     = (n_sim - s - 1) / rate
                print(f"    [{s+1:>{len(str(n_sim))}}/{n_sim}] "
                      f"{elapsed:5.0f}s transcurridos  ETA ~{eta:.0f}s")

        if verbose:
            print(f"  ✓ Completado en {time.time() - t_start:.1f}s")

        results = {}
        for model in self.models:
            mname     = model.name
            var_results = {}
            for j in range(k):
                em_j  = error_mats[mname][:, :, j]
                valid = ~np.any(np.isnan(em_j), axis=1)

                if valid.sum() == 0:
                    var_results[j] = self._nan_df(horizon)
                    continue

                em_clean = em_j[valid]
                cov_d = wid_d = winkler_d = crps_d = None

                if model in models_iv:
                    cov_d     = {lk: cov_mats[mname][lk][:, :, j][valid]     for lk in level_keys}
                    wid_d     = {lk: wid_mats[mname][lk][:, :, j][valid]     for lk in level_keys}
                    winkler_d = {lk: winkler_mats[mname][lk][:, :, j][valid] for lk in level_keys}

                if model in models_crps:
                    crps_d = crps_mats[mname][:, :, j][valid]

                var_results[j] = BiasVarianceMSE.compute_summary_table(
                    em_clean,
                    coverage_data=cov_d,
                    width_data=wid_d,
                    winkler_data=winkler_d,
                    crps_data=crps_d,
                )

            results[mname] = var_results

        return results

    @staticmethod
    def _nan_df(horizon: int) -> pd.DataFrame:
        rows = [
            {"horizon": h, "bias": np.nan, "variance": np.nan,
             "mse": np.nan, "rmse": np.nan, "mae": np.nan}
            for h in list(range(1, horizon + 1)) + ["avg_all"]
        ]
        return pd.DataFrame(rows)
