"""Internal contracts shared by the FCFF forecast components."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from decimal import Decimal

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.services.forecasting.models import ForecastSeedType

PERCENT = Decimal(100)


@dataclass(frozen=True)
class _HistoricalDrivers:
    fiscal_year: int
    period_end: datetime.date
    unit: str
    revenue: Decimal
    operating_income: Decimal
    pretax_income: Decimal
    income_tax_expense: Decimal
    depreciation_and_amortization: Decimal
    capital_expenditures: Decimal
    operating_working_capital: Decimal
    fiscal_period: FiscalPeriod = FiscalPeriod.FY


@dataclass(frozen=True)
class _ForecastContext:
    base: _HistoricalDrivers
    latest_annual: _HistoricalDrivers
    path_periods: tuple[_HistoricalDrivers, ...]
    seed_type: ForecastSeedType
    seed_methodology: str
    seed_period_end: datetime.date
    current_fiscal_year: int | None = None
    actual_ytd: _HistoricalDrivers | None = None
    actual_quarters: int = 0
    fiscal_year_end: datetime.date | None = None


__all__ = ["PERCENT", "_HistoricalDrivers", "_ForecastContext"]
