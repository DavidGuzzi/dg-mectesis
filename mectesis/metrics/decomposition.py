"""
Bias-Variance-MSE-MAE decomposition and interval metrics for forecast evaluation.
"""

import numpy as np
import pandas as pd


class BiasVarianceMSE:
    """
    Compute forecast error metrics from Monte Carlo error matrices.

    Supports point metrics (bias, variance, MSE, RMSE, MAE) and
    prediction interval metrics (coverage and mean width at multiple levels).
    """

    @staticmethod
    def compute_from_errors(errors_matrix: np.ndarray) -> dict:
        """
        Compute point metrics from forecast errors.

        Parameters
        ----------
        errors_matrix : np.ndarray, shape (n_sim, horizon)
            errors = y_true - y_pred

        Returns
        -------
        dict with keys: bias, variance, mse, rmse, mae (each shape (horizon,))
        """
        bias     = np.nanmean(errors_matrix, axis=0)
        variance = np.nanvar(errors_matrix, axis=0, ddof=1)
        mse      = np.nanmean(errors_matrix ** 2, axis=0)
        rmse     = np.sqrt(mse)
        mae      = np.nanmean(np.abs(errors_matrix), axis=0)

        return {"bias": bias, "variance": variance, "mse": mse, "rmse": rmse, "mae": mae}

    @staticmethod
    def compute_summary_table(
        errors_matrix: np.ndarray,
        coverage_data: dict = None,
        width_data: dict = None,
        winkler_data: dict = None,
        crps_data: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Build per-horizon summary table with point and interval metrics.

        Parameters
        ----------
        errors_matrix : np.ndarray, shape (n_sim, horizon)
        coverage_data : dict, optional
            {level_int: np.ndarray(n_sim, horizon)} e.g. {80: ..., 95: ...}
        width_data : dict, optional
            {level_int: np.ndarray(n_sim, horizon)}
        winkler_data : dict, optional
            {level_int: np.ndarray(n_sim, horizon)}
            Winkler (Interval) Score per replication and horizon.
        crps_data : np.ndarray, optional, shape (n_sim, horizon)
            CRPS scores per replication and horizon.

        Returns
        -------
        pd.DataFrame with columns:
            horizon, bias, variance, mse, rmse, mae
            [, crps]  if crps_data provided
            [, cov_80, cov_95, width_80, width_95, winkler_80, winkler_95]  if interval data provided
        """
        horizon = errors_matrix.shape[1]
        m = BiasVarianceMSE.compute_from_errors(errors_matrix)

        row_dict = {
            "horizon":  np.arange(1, horizon + 1),
            "bias":     m["bias"],
            "variance": m["variance"],
            "mse":      m["mse"],
            "rmse":     m["rmse"],
            "mae":      m["mae"],
        }

        agg_dict = {
            "horizon":  "avg_all",
            "bias":     m["bias"].mean(),
            "variance": m["variance"].mean(),
            "mse":      m["mse"].mean(),
            "rmse":     m["rmse"].mean(),
            "mae":      m["mae"].mean(),
        }

        # CRPS column (right after point metrics)
        if crps_data is not None:
            row_dict["crps"] = np.nanmean(crps_data, axis=0)
            agg_dict["crps"] = float(np.nanmean(crps_data))

        # Interval columns
        for prefix, data in [
            ("cov", coverage_data),
            ("width", width_data),
            ("winkler", winkler_data),
        ]:
            if data:
                for level_key, mat in sorted(data.items()):
                    col = f"{prefix}_{level_key}"
                    row_dict[col] = np.nanmean(mat, axis=0)
                    agg_dict[col] = float(np.nanmean(mat))

        df = pd.DataFrame(row_dict)
        df_agg = pd.DataFrame([agg_dict])

        return pd.concat([df, df_agg], ignore_index=True)
