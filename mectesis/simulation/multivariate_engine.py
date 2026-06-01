"""
Monte Carlo engine for multivariate forecast comparison.

DGPs return (T, k) arrays; models accept and produce (T, k) and (H, k) arrays.
Results are returned as {model_name: {var_idx: DataFrame}}, one DataFrame per
variable with the same per-horizon structure as the univariate engine.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from mectesis.dgp.base import BaseDGP
from mectesis.models.base import BaseModel
from mectesis.metrics.decomposition import BiasVarianceMSE
from mectesis.metrics.multivariate import trace_msfe, avg_marginal_crps


class MultivariateMonteCarloEngine:
    """
    Monte Carlo engine for comparing multivariate forecast models.

    Differences from MonteCarloEngine:
    - DGP simulate() returns (T, k) instead of (T,)
    - Model fit() receives (T_train, k), forecast() returns (H, k)
    - forecast_intervals() returns (lo (H,k), hi (H,k))
    - compute_crps() returns (H, k)
    - Metrics are computed per variable; result is {model: {var_idx: DataFrame}}
    - NaN rows (failed fits) are excluded variable-wise before aggregation.
    """

    def __init__(self, dgp: BaseDGP, models: List[BaseModel], seed: int = None):
        self.dgp = dgp
        self.models = models
        self.seed = seed

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

        Each DataFrame has the same structure as the univariate engine:
        columns horizon, bias, variance, mse, rmse, mae [, crps, cov_80, ...],
        one row per forecast step 1..H plus an "avg_all" summary row.
        """
        models_iv = [m for m in self.models if m.supports_intervals]
        models_crps = [m for m in self.models if m.supports_crps]
        level_keys = [int(l * 100) for l in levels]

        # Infer k from a quick trial simulation
        y_probe = self.dgp.simulate(T=T, **dgp_params)
        k = y_probe.shape[1]

        if verbose:
            iv_names = [m.name for m in models_iv] or ["ninguno"]
            print(f"  {n_sim} reps | T={T} | h=1–{horizon} | k={k} variables | "
                  f"modelos: {[m.name for m in self.models]}")
            print(f"  Intervalos ({level_keys}%): {iv_names}")

        # ── Allocate matrices (NaN-filled so failed reps are excluded) ──────
        # error_mats[mname]: (n_sim, horizon, k)
        error_mats = {m.name: np.full((n_sim, horizon, k), np.nan) for m in self.models}

        # cov/wid/winkler: (n_sim, horizon, k) per level
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

        # ── Simulation loop ────────────────────────────────────────────────
        t_start = time.time()
        log_every = max(1, n_sim // 10)

        for s in range(n_sim):
            y = self.dgp.simulate(T=T, **dgp_params)   # (T, k)
            y_train = y[:T - horizon]                   # (T_train, k)
            y_test = y[T - horizon:]                    # (horizon, k)

            for model in self.models:
                model.fit(y_train)
                y_hat = model.forecast(horizon)         # (horizon, k) or (horizon,) NaN

                # Guard: skip nan or wrong-shape forecasts
                if y_hat is None or np.all(np.isnan(y_hat)):
                    continue
                if y_hat.ndim == 1:
                    y_hat = y_hat[:, np.newaxis]
                if y_hat.shape != (horizon, k):
                    continue

                error_mats[model.name][s] = y_test - y_hat  # (horizon, k)

                if model in models_iv:
                    for level, lk in zip(levels, level_keys):
                        lo, hi = model.forecast_intervals(horizon, level=level)
                        if lo is None or np.all(np.isnan(lo)):
                            continue
                        if lo.ndim == 1:
                            lo = lo[:, np.newaxis]
                            hi = hi[:, np.newaxis]
                        if lo.shape != (horizon, k):
                            continue
                        cov_mats[model.name][lk][s] = (y_test >= lo) & (y_test <= hi)
                        wid_mats[model.name][lk][s] = hi - lo
                        alpha_iv = 1.0 - level
                        penalty = 2.0 / alpha_iv
                        winkler_mats[model.name][lk][s] = (
                            (hi - lo)
                            + penalty * np.maximum(lo - y_test, 0.0)
                            + penalty * np.maximum(y_test - hi, 0.0)
                        )

                if model in models_crps:
                    crps = model.compute_crps(y_test, horizon)  # (horizon, k)
                    if crps is not None and not np.all(np.isnan(crps)):
                        if crps.ndim == 1:
                            crps = crps[:, np.newaxis]
                        if crps.shape == (horizon, k):
                            crps_mats[model.name][s] = crps

            if verbose and (s + 1) % log_every == 0:
                elapsed = time.time() - t_start
                rate = (s + 1) / elapsed
                eta = (n_sim - s - 1) / rate
                print(f"    [{s+1:>{len(str(n_sim))}}/{n_sim}] "
                      f"{elapsed:5.0f}s transcurridos  "
                      f"ETA ~{eta:.0f}s")

        if verbose:
            print(f"  ✓ Completado en {time.time() - t_start:.1f}s")

        # ── Aggregate metrics per model per variable ───────────────────────
        results = {}
        for model in self.models:
            mname = model.name
            var_results = {}
            for j in range(k):
                em_j = error_mats[mname][:, :, j]      # (n_sim, horizon)
                valid = ~np.any(np.isnan(em_j), axis=1)  # (n_sim,) bool

                if valid.sum() == 0:
                    # All reps failed — return NaN DataFrame
                    var_results[j] = self._nan_df(horizon)
                    continue

                em_clean = em_j[valid]

                # Coverage/width/winkler for this variable
                if model in models_iv:
                    cov_d = {}
                    wid_d = {}
                    winkler_d = {}
                    for lk in level_keys:
                        cm_j = cov_mats[mname][lk][:, :, j][valid]
                        wm_j = wid_mats[mname][lk][:, :, j][valid]
                        wk_j = winkler_mats[mname][lk][:, :, j][valid]
                        cov_d[lk] = cm_j
                        wid_d[lk] = wm_j
                        winkler_d[lk] = wk_j
                else:
                    cov_d, wid_d, winkler_d = None, None, None

                crps_d = None
                if model in models_crps:
                    cr_j = crps_mats[mname][:, :, j][valid]
                    crps_d = cr_j

                var_results[j] = BiasVarianceMSE.compute_summary_table(
                    em_clean,
                    coverage_data=cov_d,
                    width_data=wid_d,
                    winkler_data=winkler_d,
                    crps_data=crps_d,
                )

            # Joint multivariate row (var_idx = -1): trace_msfe + avg_marginal_crps
            # See notebook Cell 1.5 for the justification of these two metrics
            # and the descarted alternatives (ES, VS, multivariate CRPS).
            var_results[-1] = self._compute_joint_row(
                error_mats[mname],
                crps_mats[mname] if model in models_crps else None,
                horizon,
            )

            results[mname] = var_results

        return results

    @staticmethod
    def _compute_joint_row(error_mat: np.ndarray, crps_mat,
                           horizon: int) -> pd.DataFrame:
        """
        Build the joint metrics DataFrame (one row per h, plus 'avg_all').

        Columns: horizon, trace_msfe, avg_crps.
        Replications where any variable has NaN are dropped.
        """
        valid = ~np.any(np.isnan(error_mat), axis=(1, 2))
        if valid.sum() == 0:
            rows = [
                {"horizon": h, "trace_msfe": np.nan, "avg_crps": np.nan}
                for h in list(range(1, horizon + 1)) + ["avg_all"]
            ]
            return pd.DataFrame(rows)

        em = error_mat[valid]                          # (n_valid, horizon, k)
        tm = trace_msfe(em)                            # (horizon,)

        if crps_mat is not None:
            cm = crps_mat[valid]                       # (n_valid, horizon, k)
            if np.all(np.isnan(cm)):
                acrps = np.full(horizon, np.nan)
            else:
                acrps = avg_marginal_crps(cm)
        else:
            acrps = np.full(horizon, np.nan)

        rows = []
        for h in range(horizon):
            rows.append({
                "horizon": h + 1,
                "trace_msfe": float(tm[h]),
                "avg_crps": (float(acrps[h])
                             if np.isfinite(acrps[h]) else np.nan),
            })
        rows.append({
            "horizon": "avg_all",
            "trace_msfe": float(np.mean(tm)),
            "avg_crps": (float(np.nanmean(acrps))
                         if np.any(np.isfinite(acrps)) else np.nan),
        })
        return pd.DataFrame(rows)

    @staticmethod
    def _nan_df(horizon: int) -> pd.DataFrame:
        """Return a DataFrame full of NaN as placeholder for failed models."""
        rows = [
            {"horizon": h, "bias": np.nan, "variance": np.nan,
             "mse": np.nan, "rmse": np.nan, "mae": np.nan}
            for h in list(range(1, horizon + 1)) + ["avg_all"]
        ]
        return pd.DataFrame(rows)
