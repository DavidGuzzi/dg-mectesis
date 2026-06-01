"""
VARX model — bivariate VAR with scalar exogenous covariate.

Estimated equation-by-equation via OLS (same approach as statsmodels VARModel
for the plain VAR). This avoids the numerical instability of the VARMAX
state-space Kalman filter when the optimizer diverges.

Accepts optional X_train (T_train, p_x) in fit() and X_future (H, p_x) in
forecast(). When X_train is None behaves as a standard VAR(p) via OLS.
"""

import warnings
import numpy as np
from scipy.stats import norm as _norm
from .base import BaseModel

_FIT_ERRORS = (np.linalg.LinAlgError, RuntimeError, ValueError, Exception)


class VARMAXModel(BaseModel):
    """
    VARX(p) estimated by equation-by-equation OLS.

    Design matrix per time step t:
        z_t = [1, Y_{t-1}, ..., Y_{t-p}, X_t]

    Coefficient matrix beta: shape (1 + k*p + p_x, k).
    Residual covariance Sigma: (k, k).
    Prediction intervals from analytic h-step MSE via companion-matrix recursion.

    Parameters
    ----------
    order : int
        VAR lag order p.
    """

    def __init__(self, order: int = 1):
        self.order    = order
        self._beta    = None   # (1 + k*p + p_x, k)
        self._Sigma   = None   # (k, k) innovation covariance
        self._y_last  = None   # (p, k) last training obs for recursion
        self._X_last  = None   # (p_x,) last training X
        self._k       = None
        self._p_x     = 0
        self._fit_failed = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, y_train: np.ndarray, X_train: np.ndarray = None, **kwargs):
        self._fit_failed = False
        self._beta = self._Sigma = self._y_last = self._X_last = None

        T, k  = y_train.shape
        p     = self.order
        n     = T - p
        p_x   = X_train.shape[1] if X_train is not None else 0
        ncols = 1 + k * p + p_x

        if n < ncols + 1:
            self._fit_failed = True
            return

        try:
            # Build design matrix Z (n, ncols)
            rows = []
            for t in range(p, T):
                row = [1.0]
                for lag in range(1, p + 1):
                    row.extend(y_train[t - lag].tolist())
                if X_train is not None:
                    row.extend(X_train[t].tolist())
                rows.append(row)

            Z = np.array(rows, dtype=float)      # (n, ncols)
            Y = y_train[p:].astype(float)         # (n, k)

            beta, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)  # (ncols, k)
            self._beta = beta

            resid = Y - Z @ beta                  # (n, k)
            denom = max(n - ncols, 1)
            self._Sigma = (resid.T @ resid) / denom  # (k, k)

            # Reject fits whose OLS dynamics are explosive (spectral radius >= 1).
            A_mats = [beta[1 + lag * k: 1 + (lag + 1) * k].T for lag in range(p)]
            if p == 1:
                F_check = A_mats[0]
            else:
                top = np.hstack(A_mats)
                bot = np.hstack([np.eye(k * (p - 1)), np.zeros((k * (p - 1), k))])
                F_check = np.vstack([top, bot])
            if np.max(np.abs(np.linalg.eigvals(F_check))) >= 1.0:
                self._fit_failed = True
                return

            self._y_last = y_train[-p:].copy()    # (p, k); empty if p=0
            self._X_last = X_train[-1].copy() if X_train is not None else np.zeros(0)
            self._k  = k
            self._p_x = p_x

        except _FIT_ERRORS:
            self._fit_failed = True

    # ── Forecast ──────────────────────────────────────────────────────────────

    def _check_fit(self) -> bool:
        return not self._fit_failed and self._beta is not None

    def forecast(self, horizon: int, X_future: np.ndarray = None, **kwargs) -> np.ndarray:
        if not self._check_fit():
            return np.full((horizon, self._k or 2), np.nan)

        k, p = self._k, self.order
        # rolling buffer: index -1 = most recent
        hist = list(self._y_last)           # list of (k,) arrays, oldest→newest

        y_hat = np.empty((horizon, k))
        for h in range(horizon):
            row = [1.0]
            for lag in range(1, p + 1):
                row.extend(hist[-lag].tolist())
            if X_future is not None:
                row.extend(X_future[h].tolist())
            elif self._p_x > 0:
                row.extend(self._X_last.tolist())
            row_arr = np.array(row, dtype=float)

            y_h = self._beta.T @ row_arr    # (k,)
            y_hat[h] = y_h
            hist.append(y_h)

        return y_hat

    # ── Intervals ─────────────────────────────────────────────────────────────

    @property
    def supports_covariates(self) -> bool:
        return True

    @property
    def supports_intervals(self) -> bool:
        return True

    def _companion_mse(self, horizon: int):
        """
        Return list of h-step forecast MSE matrices (k, k), h=1..horizon.

        Uses companion-matrix recursion:
            Sigma_h = Sigma_{h-1} + F^{h-1} * Sigma_comp * F'^{h-1}

        where F is the (k*p, k*p) companion matrix and Sigma_comp has
        Sigma_eps in the top-left (k, k) block.
        """
        k, p = self._k, self.order

        # Extract AR coefficient blocks A_1,...,A_p each (k, k)
        # beta rows: [intercept (1), A_1 (k), A_2 (k), ..., A_p (k), X_cols]
        A = []
        for lag in range(p):
            start = 1 + lag * k
            A.append(self._beta[start: start + k].T)    # (k, k)

        # Build companion matrix F (k*p, k*p)
        if p == 1:
            F = A[0]
        else:
            top = np.hstack(A)                            # (k, k*p)
            bot = np.hstack([np.eye(k * (p - 1)),
                              np.zeros((k * (p - 1), k))])
            F = np.vstack([top, bot])                     # (k*p, k*p)

        # Companion-form innovation covariance
        kp = k * p
        Sc = np.zeros((kp, kp))
        Sc[:k, :k] = self._Sigma

        # Accumulate MSE
        mse_list = []
        M   = np.zeros((kp, kp))
        Fi  = np.eye(kp)
        for _ in range(horizon):
            M = M + Fi @ Sc @ Fi.T
            mse_list.append(M[:k, :k].copy())
            Fi = Fi @ F

        return mse_list

    def forecast_intervals(self, horizon: int, level: float = 0.95,
                           X_future: np.ndarray = None):
        if not self._check_fit():
            nan = np.full((horizon, self._k or 2), np.nan)
            return nan, nan

        z    = _norm.ppf((1.0 + level) / 2.0)
        y_hat = self.forecast(horizon, X_future=X_future)
        mse_list = self._companion_mse(horizon)

        k  = self._k
        lo = np.empty((horizon, k))
        hi = np.empty((horizon, k))
        for h in range(horizon):
            std_h = np.sqrt(np.maximum(np.diag(mse_list[h]), 0.0))
            lo[h] = y_hat[h] - z * std_h
            hi[h] = y_hat[h] + z * std_h

        return lo, hi

    # ── CRPS ──────────────────────────────────────────────────────────────────

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int,
                     X_future: np.ndarray = None) -> np.ndarray:
        from properscoring import crps_gaussian
        if not self._check_fit():
            return np.full((horizon, self._k or 2), np.nan)

        y_hat    = self.forecast(horizon, X_future=X_future)
        mse_list = self._companion_mse(horizon)
        k        = self._k
        out      = np.empty((horizon, k))

        for h in range(horizon):
            std_h = np.sqrt(np.maximum(np.diag(mse_list[h]), 0.0))
            for j in range(k):
                out[h, j] = crps_gaussian(float(y_true[h, j]),
                                          float(y_hat[h, j]),
                                          max(float(std_h[j]), 1e-8))
        return out

    @property
    def name(self) -> str:
        return f"VARMAX({self.order}) con X"
