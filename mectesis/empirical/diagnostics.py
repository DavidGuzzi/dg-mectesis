"""
Statistical diagnostics for empirical series. Every test returns a
pandas DataFrame with at least columns {stat, pvalue, conclusion} so
results can be concatenated by the caller with consistent formatting.
"""

from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import (
    adfuller, kpss, grangercausalitytests, acf
)
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


def _conclusion_pvalue(pvalue: float, alpha: float, h0: str) -> str:
    return f"reject H0 ({h0}) @ {alpha}" if pvalue < alpha else f"do not reject H0 ({h0}) @ {alpha}"


def adf_test(y: pd.Series, regression: str = "c", alpha: float = 0.05) -> pd.DataFrame:
    stat, pvalue, used_lag, nobs, *_ = adfuller(y.dropna(), regression=regression, autolag="AIC")
    return pd.DataFrame([{
        "test": "ADF",
        "regression": regression,
        "stat": stat,
        "pvalue": pvalue,
        "used_lag": used_lag,
        "nobs": nobs,
        "conclusion": _conclusion_pvalue(pvalue, alpha, "unit root"),
    }])


def kpss_test(y: pd.Series, regression: str = "c", alpha: float = 0.05) -> pd.DataFrame:
    stat, pvalue, lags, _ = kpss(y.dropna(), regression=regression, nlags="auto")
    return pd.DataFrame([{
        "test": "KPSS",
        "regression": regression,
        "stat": stat,
        "pvalue": pvalue,
        "used_lag": lags,
        "nobs": int(y.dropna().shape[0]),
        "conclusion": _conclusion_pvalue(pvalue, alpha, "stationary"),
    }])


def zivot_andrews_test(y: pd.Series, regression: str = "c", alpha: float = 0.05) -> pd.DataFrame:
    from statsmodels.tsa.stattools import zivot_andrews
    stat, pvalue, _, _, _ = zivot_andrews(y.dropna(), regression=regression)
    return pd.DataFrame([{
        "test": "Zivot-Andrews",
        "regression": regression,
        "stat": stat,
        "pvalue": pvalue,
        "conclusion": _conclusion_pvalue(pvalue, alpha, "unit root w/break"),
    }])


def unit_root_battery(y: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    rows = [
        adf_test(y, "c", alpha),
        adf_test(y, "ct", alpha),
        kpss_test(y, "c", alpha),
        kpss_test(y, "ct", alpha),
    ]
    try:
        rows.append(zivot_andrews_test(y, "c", alpha))
    except Exception as e:
        rows.append(pd.DataFrame([{"test": "Zivot-Andrews", "stat": np.nan,
                                    "pvalue": np.nan,
                                    "conclusion": f"skipped: {type(e).__name__}"}]))
    return pd.concat(rows, ignore_index=True)


def serial_correlation(y: pd.Series, lags=(12, 24), alpha: float = 0.05) -> pd.DataFrame:
    res = acorr_ljungbox(y.dropna(), lags=list(lags), return_df=True)
    res = res.reset_index().rename(columns={"index": "lag", "lb_stat": "stat", "lb_pvalue": "pvalue"})
    res["test"] = "Ljung-Box"
    res["conclusion"] = res["pvalue"].apply(lambda p: _conclusion_pvalue(p, alpha, "no autocorr"))
    return res[["test", "lag", "stat", "pvalue", "conclusion"]]


def heteroscedasticity(y: pd.Series, nlags: int = 12, alpha: float = 0.05) -> pd.DataFrame:
    stat, pvalue, f_stat, f_pvalue = het_arch(y.dropna(), nlags=nlags)
    return pd.DataFrame([{
        "test": "ARCH-LM",
        "nlags": nlags,
        "stat": stat,
        "pvalue": pvalue,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "conclusion": _conclusion_pvalue(pvalue, alpha, "no ARCH"),
    }])


def normality(y: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    jb, jb_p, skew, kurt = jarque_bera(y.dropna())
    return pd.DataFrame([{
        "test": "Jarque-Bera",
        "stat": jb,
        "pvalue": jb_p,
        "skew": skew,
        "kurtosis": kurt,
        "conclusion": _conclusion_pvalue(jb_p, alpha, "normality"),
    }])


def structural_breaks(y: pd.Series, n_bkps: int | None = None, model: str = "l2",
                       pen: float | None = None) -> pd.DataFrame:
    """Bai-Perron-style breakpoints via `ruptures` (PELT)."""
    import ruptures as rpt
    arr = y.dropna().to_numpy()
    algo = rpt.Pelt(model=model).fit(arr)
    if n_bkps is not None:
        bkps = algo.predict(n_bkps=n_bkps)
    else:
        bkps = algo.predict(pen=pen if pen is not None else np.log(arr.shape[0]) * arr.var())
    index = y.dropna().index
    dates = [index[i - 1] for i in bkps[:-1]] if len(bkps) > 1 else []
    return pd.DataFrame([{
        "test": "PELT (Bai-Perron-style)",
        "n_breaks": len(dates),
        "break_dates": [d.strftime("%Y-%m") for d in dates],
    }])


def stl_decomposition(y: pd.Series, period: int = 12) -> pd.DataFrame:
    res = STL(y.dropna(), period=period, robust=True).fit()
    return pd.DataFrame({
        "observed": y.dropna(),
        "trend": res.trend,
        "seasonal": res.seasonal,
        "resid": res.resid,
    })


def granger_matrix(df: pd.DataFrame, maxlag: int = 6, alpha: float = 0.05) -> pd.DataFrame:
    """p-values of Granger causality test: column → row (does col cause row?)."""
    cols = list(df.columns)
    out = pd.DataFrame(np.nan, index=cols, columns=cols)
    for r, c in product(cols, cols):
        if r == c:
            continue
        try:
            res = grangercausalitytests(df[[r, c]].dropna(), maxlag=maxlag)
            pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)]
            out.loc[r, c] = float(np.min(pvals))
        except Exception:
            out.loc[r, c] = np.nan
    return out


def johansen_test(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> pd.DataFrame:
    res = coint_johansen(df.dropna(), det_order=det_order, k_ar_diff=k_ar_diff)
    n = df.shape[1]
    rows = []
    for r in range(n):
        rows.append({
            "test": "Johansen trace",
            "H0": f"rank <= {r}",
            "stat": res.lr1[r],
            "cv_90": res.cvt[r, 0],
            "cv_95": res.cvt[r, 1],
            "cv_99": res.cvt[r, 2],
            "conclusion": "reject" if res.lr1[r] > res.cvt[r, 1] else "do not reject",
        })
        rows.append({
            "test": "Johansen max-eig",
            "H0": f"rank = {r}",
            "stat": res.lr2[r],
            "cv_90": res.cvm[r, 0],
            "cv_95": res.cvm[r, 1],
            "cv_99": res.cvm[r, 2],
            "conclusion": "reject" if res.lr2[r] > res.cvm[r, 1] else "do not reject",
        })
    return pd.DataFrame(rows)


def contemp_corr(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return df.corr(method=method)


def cross_corr(y: pd.Series, x: pd.Series, max_lag: int = 24) -> pd.DataFrame:
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        s1, s2 = (y, x.shift(lag)) if lag >= 0 else (y.shift(-lag), x)
        df = pd.concat([s1, s2], axis=1).dropna()
        if df.shape[0] < 3:
            corr = np.nan
        else:
            corr = float(df.corr().iloc[0, 1])
        rows.append({"lag": lag, "corr": corr})
    return pd.DataFrame(rows)
