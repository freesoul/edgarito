"""Compatibility facade for annual driver-based FCFF forecasting.

The FCFF implementation is private to the forecasting package.  The public
service and its historical generic-service alias remain available here.  The
module used to contain the complete implementation, though, so callers also
historically imported its model types and a handful of implementation helpers
from this path.  Keep those names bound locally rather than relying on a
single ``__getattr__`` target: the moved implementation is split across
models, metrics, and the internal FCFF package.
"""

import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.financial_observation_availability import (
    FinancialObservationAvailabilityService,
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting._fcff import service as _implementation
from edgarito.services.forecasting._fcff.contracts import (
    PERCENT,
    _ForecastContext,
    _HistoricalDrivers,
)
from edgarito.services.forecasting._fcff.service import (
    OPERATING_WORKING_CAPITAL_CONCEPTS,
    FcffForecastService,
    FreeCashFlowForecastService,
)
from edgarito.services.forecasting.models import (
    FcffForecast,
    FcffForecastDcfStub,
    FcffForecastDriver,
    FcffForecastObservation,
    FcffForecastParameters,
    FcffForecastYtdAnchor,
    ForecastAssumptionSource,
    ForecastSeedType,
    ForecastValue,
    MonetaryForecastConstraint,
)
from edgarito.services.metrics.calculator import operating_working_capital_value

# These are deliberate compatibility exports, not unused implementation
# imports.  They were all importable from this module before the split.
# ruff: noqa: F401

__all__ = ["FcffForecastService", "FreeCashFlowForecastService"]


def __getattr__(name: str):
    """Resolve legacy private helper imports from the moved implementation."""

    return getattr(_implementation, name)


def __dir__():
    return sorted({*globals(), *vars(_implementation)})
