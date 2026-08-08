import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import FinancialConcept


class RedFlagCategory(str, Enum):
    FCF_VS_EARNINGS = "fcf_vs_earnings"
    DEBT = "debt"
    DILUTION_SBC = "dilution_sbc"
    ACQUISITIONS = "acquisitions"
    MARGINS_GROWTH = "margins_growth"
    ROIC = "roic"
    CASH_CONVERSION = "cash_conversion"
    CONCENTRATION = "concentration"
    ACCOUNTING_QUALITY = "accounting_quality"


class RedFlagSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RedFlagSourceObservation(BaseModel):
    """A normalized observation retained as auditable red-flag evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concept: FinancialConcept
    value: Decimal
    unit: str
    granularity: Granularity
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_end: datetime.date
    provider: str
    source_concept: str


class RedFlagEvidence(BaseModel):
    """One deterministic calculation supporting a flag."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    value: Decimal
    unit: str
    threshold: Decimal | None = None
    threshold_unit: str | None = None
    comparison: str
    formula: str
    fiscal_year: int
    fiscal_period: FiscalPeriod
    period_end: datetime.date
    granularity: Granularity
    input_concepts: tuple[FinancialConcept, ...] = ()
    source_observations: tuple[RedFlagSourceObservation, ...] = ()


class RedFlag(BaseModel):
    """A threshold breach; missing data is represented by a warning instead."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    category: RedFlagCategory
    severity: RedFlagSeverity
    message: str
    evidence: tuple[RedFlagEvidence, ...] = Field(default_factory=tuple)


class RedFlagWarning(BaseModel):
    """A limitation that prevents a rule from claiming a clean result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    category: RedFlagCategory | None = None
    message: str
    period: tuple[int, FiscalPeriod] | None = None
    required_concepts: tuple[FinancialConcept, ...] = ()


class RedFlagsReport(BaseModel):
    """Complete deterministic output from the investment red-flags service."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    provider: str
    company_id: str
    company_name: str
    ticker: str | None = None
    granularity: Granularity
    configuration_name: str
    evaluated_periods: tuple[tuple[int, FiscalPeriod], ...] = ()
    flags: tuple[RedFlag, ...] = ()
    warnings: tuple[RedFlagWarning, ...] = ()

    @property
    def has_flags(self) -> bool:
        return bool(self.flags)

    @property
    def data_complete(self) -> bool:
        return not self.warnings

    @property
    def is_clean(self) -> bool:
        """Only complete data with no breaches can be called clean."""
        return not self.flags and self.data_complete


__all__ = [
    "RedFlag",
    "RedFlagCategory",
    "RedFlagEvidence",
    "RedFlagSeverity",
    "RedFlagSourceObservation",
    "RedFlagWarning",
    "RedFlagsReport",
]
