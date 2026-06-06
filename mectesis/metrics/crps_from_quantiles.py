"""Proper CRPS estimator from quantiles via trapezoidal pinball loss."""
import numpy as np


def crps_from_quantiles(y_true, quantiles, levels):
    """
    Proper CRPS estimator from a set of predictive quantiles.

    Uses the integral representation
        CRPS(F, y) = 2 * integral_0^1 QL_tau(F^{-1}(tau), y) d tau,
    approximated with the trapezoidal rule on (sorted) levels, using
    QL_0 = QL_1 = 0 as virtual endpoints. Convergence is O(1/K^2) and
    the estimator is a proper scoring rule (positive linear combination
    of proper pinball losses).

    Parameters
    ----------
    y_true : np.ndarray
        Shape (horizon,) for univariate or (horizon, k) for multivariate.
    quantiles : np.ndarray
        Shape (horizon, K) for univariate or (horizon, k, K) for multivariate.
        Quantile values of the predictive distribution at the given `levels`.
    levels : np.ndarray
        Shape (K,). Quantile levels tau_k in (0, 1). Must be sorted ascending;
        do not need to be uniformly spaced.

    Returns
    -------
    np.ndarray
        Shape (horizon,) or (horizon, k) — CRPS per step (and per variable).
    """
    y_true = np.asarray(y_true, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    K = levels.shape[0]

    # Trapezoidal weights with virtual endpoints tau=0 and tau=1 (QL=0 there).
    extended = np.concatenate(([0.0], levels, [1.0]))
    weights = (extended[2:] - extended[:-2]) / 2.0   # shape (K,)

    if y_true.ndim == 1:
        diff = y_true[:, None] - quantiles            # (horizon, K)
        w = weights.reshape((1, K))
        lev = levels.reshape((1, K))
    else:
        diff = y_true[:, :, None] - quantiles         # (horizon, k, K)
        w = weights.reshape((1, 1, K))
        lev = levels.reshape((1, 1, K))

    indicator = (diff < 0).astype(float)
    pinball = (lev - indicator) * diff
    return 2.0 * np.sum(w * pinball, axis=-1)
