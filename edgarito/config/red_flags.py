import json
import sysconfig
from decimal import Decimal
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.schemas.normalization.financials import FinancialConcept
from edgarito.schemas.red_flags import RedFlagCategory, RedFlagSeverity

DEFAULT_RED_FLAGS_PROFILE_PATH = Path("configs/red_flags/default.json")


class _RedFlagsConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def require_finite_decimals(self):
        for value in self.__dict__.values():
            if isinstance(value, Decimal) and not value.is_finite():
                raise ValueError("Red-flags configuration values must be finite")
        return self


class _RuleConfiguration(_RedFlagsConfigModel):
    enabled: bool = True
    severity: RedFlagSeverity = RedFlagSeverity.MEDIUM


class FcfVsEarningsConfiguration(_RuleConfiguration):
    minimum_fcf_to_net_income_pct: Decimal = Field(default=Decimal("80"), ge=0)
    flag_negative_fcf_with_positive_earnings: bool = True


class DebtConfiguration(_RuleConfiguration):
    maximum_net_debt_to_ebitda: Decimal = Field(default=Decimal("3.5"), ge=0)
    minimum_interest_coverage: Decimal = Field(default=Decimal("3"), ge=0)


class DilutionSbcConfiguration(_RuleConfiguration):
    maximum_share_count_growth_pct: Decimal = Field(default=Decimal("3"), ge=0)
    maximum_diluted_share_premium_pct: Decimal = Field(default=Decimal("5"), ge=0)
    maximum_sbc_to_revenue_pct: Decimal = Field(default=Decimal("5"), ge=0)


class AcquisitionsConfiguration(_RuleConfiguration):
    maximum_acquisition_to_revenue_pct: Decimal = Field(
        default=Decimal("10"), ge=0
    )
    maximum_acquisition_to_fcf_pct: Decimal = Field(default=Decimal("50"), ge=0)
    maximum_goodwill_growth_pct: Decimal = Field(default=Decimal("15"), ge=0)


class MarginsGrowthConfiguration(_RuleConfiguration):
    minimum_revenue_growth_pct: Optional[Decimal] = Field(default=None)
    minimum_operating_margin_pct: Optional[Decimal] = Field(default=None)
    maximum_operating_margin_decline_pp: Decimal = Field(default=Decimal("3"), ge=0)

    @model_validator(mode="after")
    def validate_rates(self):
        for name in ("minimum_revenue_growth_pct", "minimum_operating_margin_pct"):
            value = getattr(self, name)
            if value is not None and value <= Decimal("-100"):
                raise ValueError(f"{name} must be greater than -100%")
        return self


class RoicConfiguration(_RuleConfiguration):
    minimum_roic_pct: Decimal = Field(default=Decimal("8"), ge=Decimal("-100"))
    maximum_roic_decline_pp: Decimal = Field(default=Decimal("3"), ge=0)


class CashConversionConfiguration(_RuleConfiguration):
    minimum_operating_cash_flow_to_net_income_pct: Decimal = Field(
        default=Decimal("80"), ge=0
    )


class ConcentrationConfiguration(_RuleConfiguration):
    """Thresholds reserved for dimensional data not in normalized financials."""

    maximum_customer_concentration_pct: Decimal = Field(default=Decimal("25"), ge=0)
    maximum_segment_concentration_pct: Decimal = Field(default=Decimal("40"), ge=0)


class AccountingQualityConfiguration(_RuleConfiguration):
    maximum_receivables_growth_premium_pp: Decimal = Field(
        default=Decimal("10"), ge=0
    )
    maximum_inventory_growth_premium_pp: Decimal = Field(default=Decimal("10"), ge=0)
    maximum_goodwill_to_assets_pct: Decimal = Field(default=Decimal("50"), ge=0)


class RedFlagsConfiguration(_RedFlagsConfigModel):
    """Versioned, provider-neutral thresholds for investment red-flag checks."""

    schema_version: Literal[1] = 1
    name: str = Field(default="default", min_length=1)
    description: str | None = None
    history_periods: int = Field(default=5, ge=2, le=20)
    fcf_vs_earnings: FcfVsEarningsConfiguration = Field(
        default_factory=FcfVsEarningsConfiguration
    )
    debt: DebtConfiguration = Field(default_factory=DebtConfiguration)
    dilution_sbc: DilutionSbcConfiguration = Field(
        default_factory=DilutionSbcConfiguration
    )
    acquisitions: AcquisitionsConfiguration = Field(
        default_factory=AcquisitionsConfiguration
    )
    margins_growth: MarginsGrowthConfiguration = Field(
        default_factory=MarginsGrowthConfiguration
    )
    roic: RoicConfiguration = Field(default_factory=RoicConfiguration)
    cash_conversion: CashConversionConfiguration = Field(
        default_factory=CashConversionConfiguration
    )
    concentration: ConcentrationConfiguration = Field(
        default_factory=ConcentrationConfiguration
    )
    accounting_quality: AccountingQualityConfiguration = Field(
        default_factory=AccountingQualityConfiguration
    )

    @field_validator("name", "description")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Red-flags profile text fields cannot be blank")
        return normalized

    @staticmethod
    def required_concepts(category: RedFlagCategory) -> tuple[FinancialConcept, ...]:
        """Return the atomic concepts that can support a category."""
        concepts = {
            RedFlagCategory.FCF_VS_EARNINGS: {
                FinancialConcept.OPERATING_CASH_FLOW,
                FinancialConcept.CAPITAL_EXPENDITURES,
                FinancialConcept.NET_INCOME,
            },
            RedFlagCategory.DEBT: {
                FinancialConcept.SHORT_TERM_DEBT,
                FinancialConcept.LONG_TERM_DEBT_CURRENT,
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
                FinancialConcept.CASH_AND_EQUIVALENTS,
                FinancialConcept.OPERATING_INCOME,
                FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
                FinancialConcept.INTEREST_EXPENSE,
            },
            RedFlagCategory.DILUTION_SBC: {
                FinancialConcept.SHARES_OUTSTANDING,
                FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES,
                FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
                FinancialConcept.STOCK_BASED_COMPENSATION,
                FinancialConcept.REVENUE,
            },
            RedFlagCategory.ACQUISITIONS: {
                FinancialConcept.ACQUISITION_CASH_PAID,
                FinancialConcept.REVENUE,
                FinancialConcept.GOODWILL,
                FinancialConcept.OPERATING_CASH_FLOW,
                FinancialConcept.CAPITAL_EXPENDITURES,
            },
            RedFlagCategory.MARGINS_GROWTH: {
                FinancialConcept.REVENUE,
                FinancialConcept.OPERATING_INCOME,
            },
            RedFlagCategory.ROIC: {
                FinancialConcept.OPERATING_INCOME,
                FinancialConcept.PRETAX_INCOME,
                FinancialConcept.INCOME_TAX_EXPENSE,
                FinancialConcept.STOCKHOLDERS_EQUITY,
                FinancialConcept.CASH_AND_EQUIVALENTS,
                FinancialConcept.SHORT_TERM_DEBT,
                FinancialConcept.LONG_TERM_DEBT_CURRENT,
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
            },
            RedFlagCategory.CASH_CONVERSION: {
                FinancialConcept.OPERATING_CASH_FLOW,
                FinancialConcept.NET_INCOME,
            },
            RedFlagCategory.CONCENTRATION: set(),
            RedFlagCategory.ACCOUNTING_QUALITY: {
                FinancialConcept.ACCOUNTS_RECEIVABLE,
                FinancialConcept.INVENTORY,
                FinancialConcept.REVENUE,
                FinancialConcept.GOODWILL,
                FinancialConcept.TOTAL_ASSETS,
            },
        }[category]
        return tuple(sorted(concepts, key=lambda concept: concept.value))

    @property
    def enabled_categories(self) -> frozenset[RedFlagCategory]:
        return frozenset(
            category
            for category in RedFlagCategory
            if getattr(self, category.value).enabled
        )

class RedFlagsProfileLoader:
    """Load the packaged default or an explicitly selected red-flags profile."""

    @staticmethod
    def load(path: str | Path | None = None) -> RedFlagsConfiguration:
        if path is None:
            profile_path = RedFlagsProfileLoader.default_path()
            source = str(profile_path)
            try:
                content = profile_path.read_text(encoding="utf-8")
            except (FileNotFoundError, OSError) as exc:
                raise FileNotFoundError(
                    f"Default red-flags profile is unavailable at {source}"
                ) from exc
        else:
            profile_path = Path(path).expanduser()
            source = str(profile_path)
            if not profile_path.is_file():
                raise FileNotFoundError(f"Red-flags profile not found: {source}")
            try:
                content = profile_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Cannot read red-flags profile: {source}") from exc
        try:
            payload = json.loads(content, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in red-flags profile {source}: "
                f"line {exc.lineno}, column {exc.colno}"
            ) from exc
        try:
            return RedFlagsConfiguration.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid red-flags profile {source}: {exc}") from exc

    @staticmethod
    def default_path() -> Path:
        source_checkout = Path(__file__).resolve().parents[2] / (
            DEFAULT_RED_FLAGS_PROFILE_PATH
        )
        installed_data = Path(sysconfig.get_path("data")) / (
            DEFAULT_RED_FLAGS_PROFILE_PATH
        )
        for candidate in (source_checkout, installed_data):
            if candidate.is_file():
                return candidate
        return source_checkout


# The shorter name is useful to callers while retaining the explicit profile
# terminology used by the valuation configuration.
RedFlagsConfigurationLoader = RedFlagsProfileLoader


__all__ = [
    "AccountingQualityConfiguration",
    "AcquisitionsConfiguration",
    "CashConversionConfiguration",
    "ConcentrationConfiguration",
    "DebtConfiguration",
    "DEFAULT_RED_FLAGS_PROFILE_PATH",
    "DilutionSbcConfiguration",
    "FcfVsEarningsConfiguration",
    "MarginsGrowthConfiguration",
    "RedFlagsConfiguration",
    "RedFlagsConfigurationLoader",
    "RedFlagsProfileLoader",
    "RoicConfiguration",
]
