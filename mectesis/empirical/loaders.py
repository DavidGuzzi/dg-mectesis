"""
Loader for the single empirical dataset `data/raw/data.csv`.

Format conventions (Argentine BCRA/INDEC export):
  - separator  ';' , encoding 'latin-1' , decimal ','  , thousands '.'
  - dates as 'd/m/Y' (end-of-month markers, e.g. 31/12/2023)
  - percentages stored as text suffixed with '%'      (e.g. '1,1%')
  - level series (TCM)  stored without '%'            (e.g. '1.396,34')
  - missing REM values flagged as the literal 'sin_datos'

The CSV is parsed once by `load_dataset()`; the per-series helpers
(load_ipc, load_tcn, …) and `build_panel()` are thin wrappers that
slice the cached panel. Series semantics:

  ipc     inflación mensual %                  (target principal, ya en %)
  tpm     tasa de política monetaria %         (nivel)
  badlar  tasa BADLAR bancos privados %        (nivel)
  tcm     tipo de cambio mayorista A3500       (nivel, ARS/USD)
  m2      variación interanual M2 privado %    (promedio móvil 30 días)
  rem     expectativa de inflación 12m %       (mediana REM)
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "data.csv"

_COLS = {
    "Mes": "fecha",
    "IPC": "ipc",
    "Tasa de política monetaria": "tpm",
    "Tasa Badlar": "badlar",
    "Tipo de Cambio Mayorista": "tcm",
    "M2": "m2",
    "REM (mediana próximos 12 meses)": "rem",
}
_PCT_COLS = ["ipc", "tpm", "badlar", "m2", "rem"]
_LEVEL_COLS = ["tcm"]


def _parse_pct(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.replace("%", "", regex=False)
         .str.replace(".", "", regex=False)   # thousands sep (shouldn't appear but safe)
         .str.replace(",", ".", regex=False)
         .astype(float)
    )


def _parse_level(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.replace(".", "", regex=False)   # thousands sep (e.g. '1.396,34')
         .str.replace(",", ".", regex=False)
         .astype(float)
    )


@lru_cache(maxsize=4)
def load_dataset(
    path: str | None = None,
    interpolate_rem: bool = True,
) -> pd.DataFrame:
    """
    Load and parse `data/raw/data.csv` into a clean monthly DataFrame.

    Parameters
    ----------
    path : str | None
        Override the default path.
    interpolate_rem : bool
        Linearly interpolate REM gaps (Q1-2016 hole left by the INDEC
        intervention). Other columns are not interpolated.
    """
    p = Path(path) if path else _DEFAULT_PATH
    df = pd.read_csv(
        p, sep=";", encoding="latin-1", na_values=["sin_datos"]
    )
    # Some Latin-1 BOMs leave the first header mis-encoded; normalise by position.
    df.columns = list(_COLS.values())
    df = df.dropna(how="all").dropna(subset=["fecha"]).copy()

    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["fecha"]).set_index("fecha").sort_index()

    for col in _PCT_COLS:
        df[col] = _parse_pct(df[col])
    for col in _LEVEL_COLS:
        df[col] = _parse_level(df[col])

    # Raw dates are end-of-month markers; convert to month-start.
    df.index = df.index.to_period("M").to_timestamp()
    df = df.sort_index().asfreq("MS")

    if interpolate_rem:
        df["rem"] = df["rem"].interpolate(method="linear", limit_area="inside")

    return df


def _series(col: str, **kwargs) -> pd.Series:
    return load_dataset(**kwargs)[col].rename(col)


def load_ipc(**kwargs) -> pd.Series:
    """Inflación mensual % (target principal)."""
    return _series("ipc", **kwargs)


def load_tpm(**kwargs) -> pd.Series:
    """Tasa de política monetaria %."""
    return _series("tpm", **kwargs)


def load_badlar(**kwargs) -> pd.Series:
    """Tasa BADLAR bancos privados %."""
    return _series("badlar", **kwargs)


def load_tcm(**kwargs) -> pd.Series:
    """Tipo de cambio mayorista A3500 (ARS/USD)."""
    return _series("tcm", **kwargs)


def load_m2(**kwargs) -> pd.Series:
    """M2 privado, variación interanual % (promedio móvil 30 días)."""
    return _series("m2", **kwargs)


def load_rem(**kwargs) -> pd.Series:
    """REM mediana expectativa de inflación 12m %."""
    return _series("rem", **kwargs)


def build_panel(
    start: str | None = "2016-12-01",
    end: str | None = "2026-05-01",
    columns: list[str] | None = None,
    interpolate_rem: bool = True,
) -> pd.DataFrame:
    """
    Build the monthly panel filtered to [start, end].

    Default columns: ipc, tpm, badlar, tcm, m2, rem.
    """
    df = load_dataset(interpolate_rem=interpolate_rem)
    if columns is not None:
        df = df[columns]
    return df.loc[start:end]
