"""
Multivariate aggregated scoring metrics.

See notebook `experimentos_multivariados_v4_cloud.ipynb` Cell 1.5 for the
methodological justification: why these two metrics are reported instead of
Energy Score / Variogram Score, and which alternatives were ruled out.
"""

import numpy as np


def trace_msfe(errors_matrix: np.ndarray) -> np.ndarray:
    """
    Trace of the MSFE matrix per horizon.

    Parameters
    ----------
    errors_matrix : np.ndarray
        Shape (n_sim, horizon, k). Forecast errors e_h = y_test - y_hat
        across `n_sim` Monte Carlo replications, `horizon` lead times, and
        `k` variables.

    Returns
    -------
    np.ndarray
        Shape (horizon,). Per-horizon value equals
            tr(E[e_h e_h^T]) = sum_i E[e_{ih}^2] = sum_i MSE_i(h).

    Notes
    -----
    Standard multivariate point metric in econometrics.
    Reference: Lutkepohl (2005), New Introduction to Multiple Time Series
    Analysis, Springer, ch. 2; Hamilton (1994), Time Series Analysis,
    Princeton University Press, ch. 11.
    """
    if errors_matrix.ndim != 3:
        raise ValueError(
            f"errors_matrix must have shape (n_sim, horizon, k); got {errors_matrix.shape}"
        )
    return np.mean(np.sum(errors_matrix ** 2, axis=2), axis=0)


def avg_marginal_crps(crps_per_var: np.ndarray) -> np.ndarray:
    """
    Average marginal CRPS across variables, per horizon.

    Parameters
    ----------
    crps_per_var : np.ndarray
        Shape (n_sim, horizon, k). Per-simulation, per-step, per-variable CRPS.

    Returns
    -------
    np.ndarray
        Shape (horizon,). Mean across Monte Carlo simulations of
            (1/k) * sum_j CRPS_j(h)
        for each horizon h.

    Notes
    -----
    A linear positive combination of proper scoring rules is also proper,
    so the average preserves the propriety of CRPS for the joint evaluation
    of marginals.
    Reference: Gneiting & Raftery (2007), Strictly Proper Scoring Rules,
    Prediction, and Estimation, JASA 102(477), sec. 2.
    """
    if crps_per_var.ndim != 3:
        raise ValueError(
            f"crps_per_var must have shape (n_sim, horizon, k); got {crps_per_var.shape}"
        )
    per_sim_per_h = np.mean(crps_per_var, axis=2)   # (n_sim, horizon)
    return np.mean(per_sim_per_h, axis=0)
