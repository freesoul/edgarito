import json
import sysconfig
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.services.forecasting.models import (
    FcffForecastParameters,
    SimplifiedFcfForecastParameters,
)
from edgarito.services.valuation.models import (
    BusinessArchetype,
    CashFlowTiming,
    CompanyLifecycle,
    Cyclicality,
    EconomicTrait,
    TerminalMetric,
    TerminalValueMethod,
    ValuationInput,
)

DEFAULT_VALUATION_PROFILE_PATH = Path("configs/valuation/default.json")


class ForecastMethod(str, Enum):
    FCFF = "fcff"
    SIMPLIFIED = "simplified"


class _ProfileModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ForecastConfiguration(_ProfileModel):
    default_method: ForecastMethod = ForecastMethod.FCFF
    fcff: FcffForecastParameters = Field(default_factory=FcffForecastParameters)
    simplified: SimplifiedFcfForecastParameters = Field(
        default_factory=SimplifiedFcfForecastParameters
    )


class DiscountRateConfiguration(_ProfileModel):
    """Optional inputs used to derive or explicitly provide discount rates."""

    risk_free_rate: Optional[Decimal] = None
    unlevered_beta: Optional[Decimal] = None
    levered_beta: Optional[Decimal] = None
    equity_risk_premium: Optional[Decimal] = None
    country_risk_premium: Optional[Decimal] = None
    cost_of_equity: Optional[Decimal] = None
    pretax_cost_of_debt: Optional[Decimal] = None
    normalized_tax_rate: Optional[Decimal] = None
    market_value_equity: Optional[Decimal] = None
    market_value_debt: Optional[Decimal] = None
    wacc: Optional[Decimal] = None

    @field_validator("*")
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Discount-rate profile values must be finite")
        return value

    @model_validator(mode="after")
    def validate_ranges(self) -> "DiscountRateConfiguration":
        if self.normalized_tax_rate is not None and not (
            Decimal(0) <= self.normalized_tax_rate <= Decimal(100)
        ):
            raise ValueError("normalized_tax_rate must be between 0% and 100%")
        for name in ("market_value_equity", "market_value_debt"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("cost_of_equity", "pretax_cost_of_debt", "wacc"):
            value = getattr(self, name)
            if value is not None and value <= Decimal("-100"):
                raise ValueError(f"{name} must be greater than -100%")
        return self

    @property
    def can_calculate_cost_of_equity(self) -> bool:
        beta_ready = self.levered_beta is not None or (
            self.unlevered_beta is not None
            and self.normalized_tax_rate is not None
            and self.market_value_equity is not None
            and self.market_value_equity > 0
            and self.market_value_debt is not None
        )
        return self.cost_of_equity is not None or (
            self.risk_free_rate is not None
            and self.equity_risk_premium is not None
            and beta_ready
        )

    @property
    def can_calculate_wacc(self) -> bool:
        return self.wacc is not None or (
            self.can_calculate_cost_of_equity
            and self.pretax_cost_of_debt is not None
            and self.normalized_tax_rate is not None
            and self.market_value_equity is not None
            and self.market_value_debt is not None
            and self.market_value_equity + self.market_value_debt > 0
        )


class TerminalValueConfiguration(_ProfileModel):
    method: TerminalValueMethod = TerminalValueMethod.PERPETUITY_GROWTH
    perpetual_growth_rate: Optional[Decimal] = None
    exit_multiple: Optional[Decimal] = None
    exit_metric: TerminalMetric = TerminalMetric.EBITDA

    @field_validator("perpetual_growth_rate", "exit_multiple")
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Terminal-value profile values must be finite")
        return value

    @model_validator(mode="after")
    def validate_ranges(self) -> "TerminalValueConfiguration":
        if (
            self.perpetual_growth_rate is not None
            and self.perpetual_growth_rate <= Decimal("-100")
        ):
            raise ValueError("perpetual_growth_rate must be greater than -100%")
        if self.exit_multiple is not None and self.exit_multiple < 0:
            raise ValueError("exit_multiple cannot be negative")
        return self


class CapitalBridgeConfiguration(_ProfileModel):
    net_debt: Optional[Decimal] = None
    gross_debt: Optional[Decimal] = None
    cash_and_equivalents: Optional[Decimal] = None
    diluted_shares: Optional[Decimal] = None

    @field_validator("net_debt", "gross_debt", "cash_and_equivalents", "diluted_shares")
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Capital-bridge profile values must be finite")
        return value

    @model_validator(mode="after")
    def validate_bridge(self) -> "CapitalBridgeConfiguration":
        if self.diluted_shares is not None and self.diluted_shares <= 0:
            raise ValueError("diluted_shares must be positive")
        components = (self.gross_debt, self.cash_and_equivalents)
        if (components[0] is None) != (components[1] is None):
            raise ValueError("gross_debt and cash_and_equivalents must be set together")
        if self.gross_debt is not None and self.cash_and_equivalents is not None:
            if self.gross_debt < 0 or self.cash_and_equivalents < 0:
                raise ValueError(
                    "gross_debt and cash_and_equivalents cannot be negative"
                )
            derived_net_debt = self.gross_debt - self.cash_and_equivalents
            if self.net_debt is not None and self.net_debt != derived_net_debt:
                raise ValueError(
                    "net_debt must equal gross_debt - cash_and_equivalents"
                )
        return self


class ValuationCalculationConfiguration(_ProfileModel):
    cash_flow_timing: CashFlowTiming = CashFlowTiming.END_OF_PERIOD
    discount_rates: DiscountRateConfiguration = Field(
        default_factory=DiscountRateConfiguration
    )
    terminal_value: TerminalValueConfiguration = Field(
        default_factory=TerminalValueConfiguration
    )
    capital_bridge: CapitalBridgeConfiguration = Field(
        default_factory=CapitalBridgeConfiguration
    )


class ModelSelectionConfiguration(_ProfileModel):
    business_archetype: Optional[BusinessArchetype] = None
    lifecycle: Optional[CompanyLifecycle] = None
    cyclicality: Optional[Cyclicality] = None
    economic_traits: frozenset[EconomicTrait] = frozenset()
    available_inputs: frozenset[ValuationInput] = frozenset()
    peer_count: Optional[int] = Field(default=None, ge=0)


class ComparableSelectionConfiguration(_ProfileModel):
    max_peers: int = Field(default=8, ge=1, le=50)
    preferred_minimum: int = Field(default=5, ge=1, le=50)
    minimum_score: int = Field(default=50, ge=0, le=100)
    require_same_sector: bool = True

    @model_validator(mode="after")
    def validate_peer_counts(self) -> "ComparableSelectionConfiguration":
        if self.preferred_minimum > self.max_peers:
            raise ValueError("preferred_minimum cannot exceed max_peers")
        return self


class SpecializedInputConfiguration(_ProfileModel):
    history: int = Field(default=5, ge=1, le=100)


class ForecastValuationProfile(_ProfileModel):
    """Versioned parameters shared by forecast and valuation CLI workflows."""

    schema_version: Literal[1] = 1
    name: str = Field(default="default", min_length=1)
    description: Optional[str] = None
    forecast: ForecastConfiguration = Field(default_factory=ForecastConfiguration)
    valuation: ValuationCalculationConfiguration = Field(
        default_factory=ValuationCalculationConfiguration
    )
    model_selection: ModelSelectionConfiguration = Field(
        default_factory=ModelSelectionConfiguration
    )
    comparables: ComparableSelectionConfiguration = Field(
        default_factory=ComparableSelectionConfiguration
    )
    specialized_inputs: SpecializedInputConfiguration = Field(
        default_factory=SpecializedInputConfiguration
    )

    @field_validator("name", "description")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Profile text fields cannot be blank")
        return normalized

    @property
    def configured_valuation_inputs(self) -> frozenset[ValuationInput]:
        configured = set(self.model_selection.available_inputs)
        rates = self.valuation.discount_rates
        if rates.can_calculate_cost_of_equity:
            configured.add(ValuationInput.COST_OF_EQUITY)
        if rates.can_calculate_wacc:
            configured.add(ValuationInput.WACC)
        if self.valuation.terminal_value.perpetual_growth_rate is not None:
            configured.add(ValuationInput.TERMINAL_GROWTH)
        return frozenset(configured)


class ValuationProfileLoader:
    """Load the packaged default or a user-selected JSON profile."""

    @staticmethod
    def load(path: str | Path | None = None) -> ForecastValuationProfile:
        if path is None:
            profile_path = ValuationProfileLoader.default_path()
            source = str(profile_path)
            try:
                content = profile_path.read_text(encoding="utf-8")
            except (FileNotFoundError, OSError) as exc:
                raise FileNotFoundError(
                    f"Default valuation profile is unavailable at {source}"
                ) from exc
        else:
            profile_path = Path(path).expanduser()
            source = str(profile_path)
            if not profile_path.is_file():
                raise FileNotFoundError(f"Valuation profile not found: {source}")
            try:
                content = profile_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Cannot read valuation profile: {source}") from exc
        try:
            payload = json.loads(content, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in valuation profile {source}: "
                f"line {exc.lineno}, column {exc.colno}"
            ) from exc
        try:
            return ForecastValuationProfile.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid valuation profile {source}: {exc}") from exc

    @staticmethod
    def default_path() -> Path:
        source_checkout = Path(__file__).resolve().parents[2] / (
            DEFAULT_VALUATION_PROFILE_PATH
        )
        installed_data = Path(sysconfig.get_path("data")) / (
            DEFAULT_VALUATION_PROFILE_PATH
        )
        for candidate in (source_checkout, installed_data):
            if candidate.is_file():
                return candidate
        return source_checkout


__all__ = [
    "CashFlowTiming",
    "CapitalBridgeConfiguration",
    "ComparableSelectionConfiguration",
    "DEFAULT_VALUATION_PROFILE_PATH",
    "DiscountRateConfiguration",
    "ForecastConfiguration",
    "ForecastMethod",
    "ForecastValuationProfile",
    "ModelSelectionConfiguration",
    "SpecializedInputConfiguration",
    "TerminalValueConfiguration",
    "ValuationCalculationConfiguration",
    "ValuationProfileLoader",
]
