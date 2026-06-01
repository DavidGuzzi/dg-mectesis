"""
Monte Carlo simulation engine for forecast comparison.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from mectesis.dgp.base import BaseDGP
from mectesis.models.base import BaseModel
from mectesis.metrics.decomposition import BiasVarianceMSE


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for comparing forecast models.

    Supports point metrics (bias, variance, MSE, RMSE, MAE) and
    prediction interval metrics (coverage and width at specified levels).
    """

    def __init__(self, dgp: BaseDGP, models: List[BaseModel], seed: int = None):
        self.dgp = dgp
        self.models = models
        self.seed = seed

    def run_single_simulation(self, T: int, horizon: int,
                              dgp_params: dict) -> Dict[str, np.ndarray]:
        """
        Run one replication. Returns {model_name: error_array(horizon,)}.
        Kept for backward compatibility; run_monte_carlo uses inlined logic.
        """
        y = self.dgp.simulate(T=T, **dgp_params)
        T_train = T - horizon
        y_train, y_test = y[:T_train], y[T_train:]
        assert len(y_test) == horizon

        errors = {}
        for model in self.models:
            model.fit(y_train)
            y_hat = model.forecast(horizon)
            errors[model.name] = y_test - y_hat
        return errors

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
        Run n_sim Monte Carlo replications and return aggregated metrics.

        Parameters
        ----------
        n_sim : int
        T : int
        horizon : int
        dgp_params : dict
        levels : tuple of float
            Prediction interval levels, e.g. (0.80, 0.95).
        verbose : bool
            Print progress with timing.

        Returns
        -------
        dict {model_name: DataFrame}
            Columns: horizon, bias, variance, mse, rmse, mae
                     [, cov_80, width_80, cov_95, width_95] if supported.
            One row per horizon step (1..H) plus an "avg_all" summary row.
        """
        models_iv = [m for m in self.models if m.supports_intervals]
        models_crps = [m for m in self.models if m.supports_crps]
        level_keys = [int(l * 100) for l in levels]

        if verbose:
            iv_names = [m.name for m in models_iv] or ["ninguno"]
            print(f"  {n_sim} reps | T={T} | h=1–{horizon} | "
                  f"modelos: {[m.name for m in self.models]}")
            print(f"  Intervalos ({level_keys}%): {iv_names}")

        # ── Allocate matrices ──────────────────────────────────────────────
        error_mats = {m.name: np.empty((n_sim, horizon)) for m in self.models}
        cov_mats = {
            m.name: {lk: np.empty((n_sim, horizon)) for lk in level_keys}
            for m in models_iv
        }
        wid_mats = {
            m.name: {lk: np.empty((n_sim, horizon)) for lk in level_keys}
            for m in models_iv
        }
        winkler_mats = {
            m.name: {lk: np.empty((n_sim, horizon)) for lk in level_keys}
            for m in models_iv
        }
        crps_mats = {
            m.name: np.empty((n_sim, horizon)) for m in models_crps
        }

        # ── Simulation loop ────────────────────────────────────────────────
        t_start = time.time()
        log_every = max(1, n_sim // 10)
        _fail_counts = {m.name: 0 for m in self.models}

        for s in range(n_sim):
            y = self.dgp.simulate(T=T, **dgp_params)
            y_train, y_test = y[:T - horizon], y[T - horizon:]

            for model in self.models:
                try:
                    model.fit(y_train)
                    y_hat = model.forecast(horizon)
                    error_mats[model.name][s] = y_test - y_hat

                    if model in models_iv:
                        for level, lk in zip(levels, level_keys):
                            lo, hi = model.forecast_intervals(horizon, level=level)
                            cov_mats[model.name][lk][s] = (y_test >= lo) & (y_test <= hi)
                            wid_mats[model.name][lk][s] = hi - lo
                            alpha = 1.0 - level
                            penalty = 2.0 / alpha
                            winkler_mats[model.name][lk][s] = (
                                (hi - lo)
                                + penalty * np.maximum(lo - y_test, 0.0)
                                + penalty * np.maximum(y_test - hi, 0.0)
                            )

                    if model in models_crps:
                        crps_mats[model.name][s] = model.compute_crps(y_test, horizon)

                except Exception as e:
                    _fail_counts[model.name] += 1
                    if _fail_counts[model.name] == 1:
                        print(f"  [!] {model.name} rep={s}: {type(e).__name__}: {e}")
                    error_mats[model.name][s] = np.nan
                    if model in models_iv:
                        for lk in level_keys:
                            cov_mats[model.name][lk][s] = np.nan
                            wid_mats[model.name][lk][s] = np.nan
                            winkler_mats[model.name][lk][s] = np.nan
                    if model in models_crps:
                        crps_mats[model.name][s] = np.nan

            if verbose and (s + 1) % log_every == 0:
                elapsed = time.time() - t_start
                rate = (s + 1) / elapsed
                eta = (n_sim - s - 1) / rate
                print(f"    [{s+1:>{len(str(n_sim))}}/{n_sim}] "
                      f"{elapsed:5.0f}s transcurridos  "
                      f"ETA ~{eta:.0f}s")

        if verbose:
            print(f"  ✓ Completado en {time.time() - t_start:.1f}s")
        for mname, cnt in _fail_counts.items():
            if cnt > 0:
                print(f"  ⚠ {mname}: {cnt}/{n_sim} réplicas fallidas (excluidas del cómputo)")

        # ── Aggregate metrics ──────────────────────────────────────────────
        results = {}
        for model in self.models:
            mname = model.name
            if model in models_iv:
                cov_d = {lk: cov_mats[mname][lk] for lk in level_keys}
                wid_d = {lk: wid_mats[mname][lk] for lk in level_keys}
                winkler_d = {lk: winkler_mats[mname][lk] for lk in level_keys}
            else:
                cov_d, wid_d, winkler_d = None, None, None

            crps_d = crps_mats[mname] if model in models_crps else None

            results[mname] = BiasVarianceMSE.compute_summary_table(
                error_mats[mname],
                coverage_data=cov_d,
                width_data=wid_d,
                winkler_data=winkler_d,
                crps_data=crps_d,
            )

        return results
