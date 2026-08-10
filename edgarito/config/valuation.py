import datetime
import json
import re
import sysconfig
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.schemas.normalization.classification import Sector
from edgarito.schemas.valuation.intrinsic import (
    PipelineProject,
    PropertyAsset,
    ResourceProject,
    SotpAdjustment,
    SotpComponent,
)
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
    FinancialInstitutionKind,
    RelativeValuationBasis,
    TerminalMetric,
    TerminalValueMethod,
    ValuationInput,
    ValuationProfile,
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
    non_operating_assets: Optional[Decimal] = None

    @field_validator(
        "net_debt",
        "gross_debt",
        "cash_and_equivalents",
        "diluted_shares",
        "non_operating_assets",
    )
    @classmethod
    def require_finite(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Capital-bridge profile values must be finite")
        return value

    @model_validator(mode="after")
    def validate_bridge(self) -> "CapitalBridgeConfiguration":
        if self.diluted_shares is not None and self.diluted_shares <= 0:
            raise ValueError("diluted_shares must be positive")
        if self.non_operating_assets is not None and self.non_operating_assets < 0:
            raise ValueError("non_operating_assets cannot be negative")
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


class ShareRepurchaseConfiguration(_ProfileModel):
    """Optional future buyback schedule used for a remaining-holder analysis."""

    annual_cash_amounts: tuple[Decimal, ...] = ()
    initial_purchase_price: Optional[Decimal] = None
    price_growth_rate: Optional[Decimal] = None
    discount_rate: Optional[Decimal] = None
    source: Optional[str] = None

    @field_validator("annual_cash_amounts")
    @classmethod
    def validate_cash_amounts(cls, values: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        if any(not value.is_finite() or value <= 0 for value in values):
            raise ValueError(
                "Share-repurchase cash amounts must be finite and positive"
            )
        return values

    @field_validator("initial_purchase_price", "price_growth_rate", "discount_rate")
    @classmethod
    def validate_optional_number(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("Share-repurchase assumptions must be finite")
        return value

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Share-repurchase source cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_assumptions(self) -> "ShareRepurchaseConfiguration":
        if self.initial_purchase_price is not None and self.initial_purchase_price <= 0:
            raise ValueError("initial_purchase_price must be positive")
        for name in ("price_growth_rate", "discount_rate"):
            value = getattr(self, name)
            if value is not None and value <= Decimal("-100"):
                raise ValueError(f"{name} must be greater than -100%")
        if not self.annual_cash_amounts and any(
            value is not None
            for value in (
                self.initial_purchase_price,
                self.price_growth_rate,
                self.discount_rate,
                self.source,
            )
        ):
            raise ValueError("Share-repurchase assumptions require annual_cash_amounts")
        return self


class MultistageValuationConfiguration(_ProfileModel):
    """Adaptive convergence from current operating drivers to terminal growth."""

    enabled: bool = True
    stable_growth_rate: Optional[Decimal] = None
    convergence_tolerance: Decimal = Decimal("1")
    max_annual_growth_fade: Decimal = Decimal("3")
    growth_gap_per_high_growth_year: Decimal = Decimal("10")
    minimum_transition_years: int = Field(default=3, ge=1, le=15)
    maximum_transition_years: int = Field(default=10, ge=1, le=20)
    maximum_high_growth_years: int = Field(default=3, ge=0, le=10)
    extend_to_stable: bool = True
    fade_reinvestment_to_terminal: bool = True
    terminal_return_on_invested_capital: Optional[Decimal] = None
    depreciable_asset_life_years: Optional[int] = Field(default=None, ge=2, le=30)

    @field_validator(
        "convergence_tolerance",
        "max_annual_growth_fade",
        "growth_gap_per_high_growth_year",
    )
    @classmethod
    def validate_positive_rate(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value <= 0:
            raise ValueError("Multistage rate parameters must be finite and positive")
        return value

    @field_validator("terminal_return_on_invested_capital")
    @classmethod
    def validate_terminal_roic(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and (not value.is_finite() or value <= 0):
            raise ValueError("Terminal ROIC must be finite and positive")
        return value

    @field_validator("stable_growth_rate")
    @classmethod
    def validate_stable_growth(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and (not value.is_finite() or value <= Decimal("-100")):
            raise ValueError("Stable growth must be finite and greater than -100%")
        return value

    @model_validator(mode="after")
    def validate_stage_bounds(self) -> "MultistageValuationConfiguration":
        if self.minimum_transition_years > self.maximum_transition_years:
            raise ValueError(
                "minimum_transition_years cannot exceed maximum_transition_years"
            )
        return self


class DecisionAnalysisConfiguration(_ProfileModel):
    """Deterministic uncertainty policy for decision-oriented valuation."""

    enabled: bool = True
    revenue_growth_delta: Decimal = Field(default=Decimal("2"), ge=0, le=25)
    operating_margin_delta: Decimal = Field(default=Decimal("2"), ge=0, le=25)
    bear_wacc_delta: Decimal = Field(default=Decimal("0.75"), ge=0, le=10)
    bull_wacc_delta: Decimal = Field(default=Decimal("0.50"), ge=0, le=10)
    terminal_growth_delta: Decimal = Field(default=Decimal("0.25"), ge=0, le=5)
    terminal_roic_spread_change: Decimal = Field(default=Decimal("0.25"), ge=0, le=1)
    fair_value_band: Decimal = Field(default=Decimal("5"), ge=0, le=50)
    sensitivity_size: int = Field(default=5, ge=3, le=9)

    @field_validator("sensitivity_size")
    @classmethod
    def require_odd_sensitivity_size(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("sensitivity_size must be odd")
        return value


class FcfeConfiguration(_ProfileModel):
    explicit_fcfe: tuple[Decimal, ...] = ()
    net_income: tuple[Decimal, ...] = ()
    depreciation_and_amortization: tuple[Decimal, ...] = ()
    capital_expenditures: tuple[Decimal, ...] = ()
    working_capital_changes: tuple[Decimal, ...] = ()
    net_borrowing: tuple[Decimal, ...] = ()
    debt_financing_ratio: Optional[Decimal] = Field(default=None, ge=0, le=1)
    required_common_equity_changes: tuple[Decimal, ...] = ()
    terminal_return_on_equity: Optional[Decimal] = None


class DividendDiscountConfiguration(_ProfileModel):
    mode: Literal["gordon", "multistage"] = "multistage"
    dividends: tuple[Decimal, ...] = ()
    earnings: tuple[Decimal, ...] = ()
    payout_ratios: tuple[Decimal, ...] = ()
    terminal_return_on_equity: Optional[Decimal] = None
    terminal_payout_ratio: Optional[Decimal] = Field(default=None, ge=0, le=1)


class ResidualIncomeConfiguration(_ProfileModel):
    starting_book_value: Optional[Decimal] = Field(default=None, gt=0)
    book_value_basis: Literal["common_equity", "tangible_common_equity"] = (
        "common_equity"
    )
    return_on_equity_path: tuple[Decimal, ...] = ()
    payout_ratio_path: tuple[Decimal, ...] = ()
    excess_return_persistence: Decimal = Field(default=Decimal("0.75"), ge=0, le=1)


class SotpConfiguration(_ProfileModel):
    components: tuple[SotpComponent, ...] = ()
    adjustments: tuple[SotpAdjustment, ...] = ()
    holding_company_discount: Decimal = Field(default=Decimal(0), ge=0, lt=1)


class ReitConfiguration(_ProfileModel):
    ffo: Optional[Decimal] = None
    recurring_affo_adjustments: tuple[Decimal, ...] = ()
    affo_forecast: tuple[Decimal, ...] = ()
    properties: tuple[PropertyAsset, ...] = ()


class FinancialInstitutionConfiguration(_ProfileModel):
    kind: FinancialInstitutionKind = FinancialInstitutionKind.OTHER
    required_common_equity_changes: tuple[Decimal, ...] = ()
    regulatory_capital_constraints: tuple[str, ...] = ()
    actuarial_detail_supplied: bool = False


class ResourceConfiguration(_ProfileModel):
    projects: tuple[ResourceProject, ...] = ()


class PipelineConfiguration(_ProfileModel):
    projects: tuple[PipelineProject, ...] = ()


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
    share_repurchases: ShareRepurchaseConfiguration = Field(
        default_factory=ShareRepurchaseConfiguration
    )
    multistage: MultistageValuationConfiguration = Field(
        default_factory=MultistageValuationConfiguration
    )
    decision_analysis: DecisionAnalysisConfiguration = Field(
        default_factory=DecisionAnalysisConfiguration
    )
    fcfe: FcfeConfiguration = Field(default_factory=FcfeConfiguration)
    dividend_discount: DividendDiscountConfiguration = Field(
        default_factory=DividendDiscountConfiguration
    )
    residual_income: ResidualIncomeConfiguration = Field(
        default_factory=ResidualIncomeConfiguration
    )
    sotp: SotpConfiguration = Field(default_factory=SotpConfiguration)
    reit: ReitConfiguration = Field(default_factory=ReitConfiguration)
    financial_institution: FinancialInstitutionConfiguration = Field(
        default_factory=FinancialInstitutionConfiguration
    )
    resources: ResourceConfiguration = Field(default_factory=ResourceConfiguration)
    pipelines: PipelineConfiguration = Field(default_factory=PipelineConfiguration)


class ModelSelectionConfiguration(_ProfileModel):
    sector: Optional[Sector] = None
    industry: Optional[str] = None
    business_archetype: Optional[BusinessArchetype] = None
    financial_institution_kind: Optional[FinancialInstitutionKind] = None
    lifecycle: Optional[CompanyLifecycle] = None
    cyclicality: Optional[Cyclicality] = None
    economic_traits: frozenset[EconomicTrait] = frozenset()
    available_inputs: frozenset[ValuationInput] = frozenset()
    peer_count: Optional[int] = Field(default=None, ge=0)

    @field_validator("industry")
    @classmethod
    def normalize_industry(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Model-selection industry cannot be blank")
        return normalized


class ComparableSelectionConfiguration(_ProfileModel):
    peers: tuple[str, ...] = ()
    max_peers: int = Field(default=8, ge=1, le=50)
    preferred_minimum: int = Field(default=5, ge=1, le=50)
    minimum_score: int = Field(default=50, ge=0, le=100)
    require_same_sector: bool = True

    @field_validator("peers", mode="before")
    @classmethod
    def normalize_peers(cls, values) -> tuple[str, ...]:
        if values is None:
            return ()
        if isinstance(values, str):
            values = (values,)
        peers = []
        seen = set()
        for value in values:
            symbol = str(value).strip().upper()
            if not re.fullmatch(r"[A-Z0-9][A-Z0-9._^-]*", symbol):
                raise ValueError(f"Invalid comparable peer symbol: {value!r}")
            if symbol not in seen:
                seen.add(symbol)
                peers.append(symbol)
        return tuple(peers)

    @model_validator(mode="after")
    def validate_peer_counts(self) -> "ComparableSelectionConfiguration":
        if self.preferred_minimum > self.max_peers:
            raise ValueError("preferred_minimum cannot exceed max_peers")
        return self


class MultipleResolutionConfiguration(_ProfileModel):
    method: Literal["premium_persistence", "blended"] = "premium_persistence"
    use_target_history: bool = True
    use_peer_median: bool = True
    use_fundamental_anchor: bool = True
    forecast_premium_mean_reversion: bool = True
    minimum_peer_sample: int = Field(default=4, ge=1, le=50)
    minimum_premium_history_observations: int = Field(
        default=8,
        ge=4,
        le=100,
        description=(
            "Minimum synchronized target/peer premium observations required before "
            "AR(1) and premium-persistence blending is enabled"
        ),
    )
    annual_premium_decay: Decimal = Field(default=Decimal("0.10"), ge=0, le=1)
    premium_persistence_prior: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)
    full_premium_history_observations: int = Field(default=12, ge=4, le=100)
    insufficient_history_persistence: Optional[Decimal] = Field(
        default=None, ge=0, le=1
    )
    persistence_range_width: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Deprecated compatibility field; ranges now use peer and premium "
            "evidence rather than a fixed persistence step"
        ),
    )
    winsorize_percentiles: tuple[Decimal, Decimal] = (
        Decimal("10"),
        Decimal("90"),
    )

    @model_validator(mode="after")
    def validate_policy(self) -> "MultipleResolutionConfiguration":
        if not self.use_fundamental_anchor:
            raise ValueError(
                "Relative multiple resolution requires a fundamental anchor"
            )
        lower, upper = self.winsorize_percentiles
        if not Decimal(0) <= lower < upper <= Decimal(100):
            raise ValueError("winsorize_percentiles must be increasing within 0-100")
        return self


class RelativeValuationConfiguration(_ProfileModel):
    enabled: bool = False
    basis: RelativeValuationBasis = RelativeValuationBasis.EV_TO_EBITDA
    horizon_years: Decimal = Field(default=Decimal(1), gt=0, le=10)
    multiple_resolution: MultipleResolutionConfiguration = Field(
        default_factory=MultipleResolutionConfiguration
    )


class SpecializedInputConfiguration(_ProfileModel):
    history: int = Field(default=5, ge=1, le=100)


class ForecastValuationProfile(_ProfileModel):
    """Versioned parameters shared by forecast and valuation CLI workflows."""

    schema_version: Literal[1, 2] = 2
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
    relative_valuation: RelativeValuationConfiguration = Field(
        default_factory=RelativeValuationConfiguration
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
        if self.valuation.fcfe.explicit_fcfe:
            configured.add(ValuationInput.EQUITY_CASH_FLOW_FORECAST)
        if self.valuation.dividend_discount.dividends:
            configured.add(ValuationInput.DIVIDEND_FORECAST)
        if (
            sum(
                value is not None
                for value in (
                    self.valuation.dividend_discount.terminal_return_on_equity,
                    self.valuation.dividend_discount.terminal_payout_ratio,
                    self.valuation.terminal_value.perpetual_growth_rate,
                )
            )
            >= 2
        ):
            configured.add(ValuationInput.PAYOUT_POLICY)
        if self.valuation.residual_income.return_on_equity_path:
            configured.add(ValuationInput.FORECAST_ROE)
        if self.valuation.sotp.components:
            configured.add(ValuationInput.SEGMENT_VALUES)
        if self.valuation.reit.properties:
            configured.add(ValuationInput.ASSET_LEVEL_VALUES)
        if self.valuation.reit.affo_forecast:
            configured.add(ValuationInput.AFFO)
        if self.valuation.resources.projects:
            configured.update(
                {ValuationInput.ASSET_LEVEL_VALUES, ValuationInput.RESERVE_DATA}
            )
        if self.valuation.pipelines.projects:
            configured.update(
                {ValuationInput.ASSET_LEVEL_VALUES, ValuationInput.PIPELINE_DATA}
            )
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

    @classmethod
    def ticker_path(cls, ticker: str) -> Path:
        normalized = re.sub(r"[^a-z0-9._-]+", "-", ticker.strip().casefold()).strip(
            ".-"
        )
        if not normalized or normalized in {"default", ".", ".."}:
            raise ValueError(f"Ticker cannot be used as a profile name: {ticker!r}")
        return cls.default_path().parent / f"{normalized}.json"

    @classmethod
    def load_for_ticker(
        cls,
        ticker: str,
        explicit_path: str | Path | None = None,
    ) -> tuple[ForecastValuationProfile, Path, bool]:
        """Load an explicit/ticker/default profile and report generation need."""
        if explicit_path is not None:
            path = Path(explicit_path).expanduser()
            return cls.load(path), path, False
        ticker_path = cls.ticker_path(ticker)
        if ticker_path.is_file():
            return cls.load(ticker_path), ticker_path, False
        return cls.load(), ticker_path, True

    @classmethod
    def create_generated(
        cls,
        *,
        ticker: str,
        base_profile: ForecastValuationProfile,
        inferred_profile: ValuationProfile,
        terminal_roic: Decimal,
        terminal_roic_confidence: str,
        generated_on: datetime.date,
        peers: tuple[str, ...] = (),
        path: Path | None = None,
    ) -> tuple[ForecastValuationProfile, Path, bool]:
        """Create a ticker profile once, preserving dynamic point-in-time inputs."""
        profile_path = path or cls.ticker_path(ticker)
        if profile_path.is_file():
            return cls.load(profile_path), profile_path, False

        generated = cls.build_generated(
            ticker=ticker,
            base_profile=base_profile,
            inferred_profile=inferred_profile,
            terminal_roic=terminal_roic,
            terminal_roic_confidence=terminal_roic_confidence,
            generated_on=generated_on,
            peers=peers,
            path=profile_path,
        )
        payload = generated.model_dump(mode="json")
        payload["model_selection"]["economic_traits"] = sorted(
            payload["model_selection"]["economic_traits"]
        )
        payload["model_selection"]["available_inputs"] = sorted(
            payload["model_selection"]["available_inputs"]
        )
        content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        try:
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            with profile_path.open("x", encoding="utf-8") as handle:
                handle.write(content)
        except FileExistsError:
            return cls.load(profile_path), profile_path, False
        except OSError as exc:
            raise ValueError(
                f"Cannot create generated valuation profile {profile_path}: {exc}"
            ) from exc
        return generated, profile_path, True

    @classmethod
    def build_generated(
        cls,
        *,
        ticker: str,
        base_profile: ForecastValuationProfile,
        inferred_profile: ValuationProfile,
        terminal_roic: Decimal,
        terminal_roic_confidence: str,
        generated_on: datetime.date,
        peers: tuple[str, ...] = (),
        path: Path | None = None,
    ) -> ForecastValuationProfile:
        """Build a generated profile in memory so discovery can finish before write."""
        profile_path = path or cls.ticker_path(ticker)
        normalized_peers = ComparableSelectionConfiguration.normalize_peers(peers)
        configured = base_profile.model_selection
        selection = configured.model_copy(
            update={
                "sector": configured.sector or inferred_profile.sector,
                "industry": configured.industry or inferred_profile.industry,
                "business_archetype": (
                    configured.business_archetype or inferred_profile.business_archetype
                ),
                "financial_institution_kind": (
                    configured.financial_institution_kind
                    or inferred_profile.financial_institution_kind
                ),
                "lifecycle": configured.lifecycle or inferred_profile.lifecycle,
                "cyclicality": configured.cyclicality or inferred_profile.cyclicality,
                "economic_traits": frozenset(
                    configured.economic_traits | inferred_profile.economic_traits
                ),
                "peer_count": (
                    configured.peer_count
                    if configured.peer_count is not None
                    else len(normalized_peers) or None
                ),
            }
        )
        multistage = base_profile.valuation.multistage.model_copy(
            update={"terminal_return_on_invested_capital": terminal_roic}
        )
        valuation = base_profile.valuation.model_copy(update={"multistage": multistage})
        comparables = base_profile.comparables.model_copy(
            update={
                "peers": base_profile.comparables.peers or normalized_peers,
            }
        )
        generated = base_profile.model_copy(
            update={
                "name": profile_path.stem,
                "description": (
                    f"Auto-generated for {ticker.strip().upper()} on "
                    f"{generated_on.isoformat()}. Structural company inference and "
                    f"terminal ROIC ({terminal_roic_confidence} confidence) are "
                    "materialized for tuning, including economically selected peers "
                    "when discovery succeeds; market rates, terminal growth, forecast "
                    "run-rate, and capital-bridge values remain dynamic."
                ),
                "model_selection": selection,
                "valuation": valuation,
                "comparables": comparables,
            }
        )
        return generated


__all__ = [
    "CashFlowTiming",
    "CapitalBridgeConfiguration",
    "ComparableSelectionConfiguration",
    "DEFAULT_VALUATION_PROFILE_PATH",
    "DiscountRateConfiguration",
    "DividendDiscountConfiguration",
    "FcfeConfiguration",
    "FinancialInstitutionConfiguration",
    "ForecastConfiguration",
    "ForecastMethod",
    "ForecastValuationProfile",
    "ModelSelectionConfiguration",
    "MultipleResolutionConfiguration",
    "MultistageValuationConfiguration",
    "RelativeValuationConfiguration",
    "ReitConfiguration",
    "ResidualIncomeConfiguration",
    "ResourceConfiguration",
    "PipelineConfiguration",
    "SotpConfiguration",
    "SpecializedInputConfiguration",
    "TerminalValueConfiguration",
    "ValuationCalculationConfiguration",
    "ValuationProfileLoader",
]
