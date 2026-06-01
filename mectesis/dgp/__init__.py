"""Data Generating Processes (DGP) module."""

from .base import BaseDGP
from .ar import AR1
from .rw import RandomWalk
from .ar_trend import AR1WithTrend
from .seasonal import SeasonalDGP
from .ar_break import AR1WithBreak
from .garch import AR1ARCH, AR1GARCH, PureGARCH, AR1GJRGARCH, AR1EGARCH
from .markov_switching import MarkovSwitchingAR
from .ets_dgps import (
    LocalLevelDGP, LocalTrendDGP, DampedTrendDGP,
    DeterministicSeasonalDGP, SeasonalRandomWalkDGP, LocalLevelSeasonalDGP,
)
from .var_dgp import VARDGP, VARGARCHDiagonalDGP
from .vecm_dgp import VECMBivariateDGP
from .arimax_dgp import ARIMAX_DGP, ARIMAX2Cov_DGP, ARIMAX_GARCH_DGP
from .arimax_trend_dgp import ARIMAX_TREND_DGP
from .sarimax_seasonal_dgp import SARIMAX_SEASONAL_DGP
from .varx_dgp import VARX_DGP
from .adl_ecm_dgp import ADL_ECM_DGP
from .gp_dgp import GPKernelSynthDGP
from .arma_general import ARpDGP, MAqDGP, ARMApqDGP, ARMApqWithTrendDGP
from .threshold_dgps import SETARDGp, LSTARDGp, ESTARDGp

__all__ = [
    "BaseDGP",
    "AR1", "RandomWalk", "AR1WithTrend", "SeasonalDGP", "AR1WithBreak",
    "AR1ARCH", "AR1GARCH", "PureGARCH", "AR1GJRGARCH", "AR1EGARCH",
    "MarkovSwitchingAR",
    "LocalLevelDGP", "LocalTrendDGP", "DampedTrendDGP",
    "DeterministicSeasonalDGP", "SeasonalRandomWalkDGP", "LocalLevelSeasonalDGP",
    "VARDGP", "VARGARCHDiagonalDGP", "VECMBivariateDGP",
    "ARIMAX_DGP", "ARIMAX2Cov_DGP", "ARIMAX_GARCH_DGP",
    "ARIMAX_TREND_DGP",
    "SARIMAX_SEASONAL_DGP",
    "VARX_DGP",
    "ADL_ECM_DGP",
    "GPKernelSynthDGP",
    "ARpDGP", "MAqDGP", "ARMApqDGP", "ARMApqWithTrendDGP",
    "SETARDGp", "LSTARDGp", "ESTARDGp",
]
