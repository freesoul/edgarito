"""Provider-neutral, validated forecasting contracts."""

import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from edgarito.schemas.forward import (
    ForwardEstimateProviderDiagnostic as _ForwardEstimateProviderDiagnostic,
)
from edgarito.schemas.forward import (
    ForwardRevenueEstimate as _ForwardRevenueEstimate,
)
from edgarito.schemas.guidance.management import (
    MonetaryForecastConstraint as _MonetaryForecastConstraint,
)
from edgarito.schemas.identifiers import SecurityIdentifiers as _SecurityIdentifiers


class ForecastAssumptionSource(str, Enum):
    EXPLICIT = "explicit"
    MANAGEMENT_GUIDANCE = "management_guidance"
    TRAILING_AVERAGE = "trailing_average"
    FORWARD_EVIDENCE = "forward_evidence"
    NORMALIZED_HISTORICAL = "normalized_historical"
    CURRENT_RUN_RATE = "current_run_rate"
    ADAPTIVE_MULTISTAGE = "adaptive_multistage"


class ForecastValue(BaseModel):
    """A numeric forecast cell together with its audit provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: Decimal
    source: str
    method: str
    confidence: str

    @field_validator("value")
    @classmethod
    def require_finite_value(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Forecast values must be finite")
        return value

    @field_validator("source", "method")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Forecast value provenance cannot be blank")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError("Forecast value confidence must be high, medium, or low")
        return normalized


class _CaseInsensitiveStrEnum(str, Enum):
    """String enum that accepts the serialized value without case surprises."""

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized = value.strip().casefold()
            for member in cls:
                if member.value.casefold() == normalized:
                    return member
        return None


class FcffForecastMethod(_CaseInsensitiveStrEnum):
    """Available FCFF planning methods.

    ``driver_based`` is intentionally a planning value for this foundation
    step.  Its execution remains explicitly unsupported by the orchestration
    service rather than silently falling back to another method.
    """

    NORMALIZED = "normalized"
    HYBRID = "hybrid"
    DRIVER_BASED = "driver_based"
    AUTO = "auto"


class ForecastScope(_CaseInsensitiveStrEnum):
    """Scope of a forecast decision or manual override."""

    COMPANY = "company"
    SEGMENT = "segment"


class ForecastMetric(_CaseInsensitiveStrEnum):
    """Provider-neutral FCFF metrics used by planning decisions."""

    REVENUE = "revenue"
    SEGMENT_REVENUE = "segment_revenue"
    GROSS_MARGIN = "gross_margin"
    GROSS_PROFIT = "gross_profit"
    R_AND_D = "r_and_d"
    SG_AND_A = "sg_and_a"
    OTHER_OPERATING_ITEMS = "other_operating_items"
    EBIT = "ebit"
    REVENUE_GROWTH = "revenue_growth"
    OPERATING_MARGIN = "operating_margin"
    TAX = "tax"
    TAX_RATE = "tax_rate"
    DEPRECIATION_AND_AMORTIZATION = "depreciation_and_amortization"
    DEPRECIATION_TO_REVENUE = "depreciation_to_revenue"
    CAPEX = "capex"
    CAPEX_TO_REVENUE = "capex_to_revenue"
    OPERATING_WORKING_CAPITAL = "operating_working_capital"
    OPERATING_WORKING_CAPITAL_TO_REVENUE = "operating_working_capital_to_revenue"
    DELTA_NWC = "delta_nwc"
    FCFF = "fcff"


class ForecastStrategy(_CaseInsensitiveStrEnum):
    """How a decision obtains or derives its forecast value."""

    DRIVER = "driver"
    CONSOLIDATED = "consolidated"
    EXPLICIT = "explicit"
    RATIO = "ratio"
    RESIDUAL = "residual"
    IGNORE = "ignore"


class ForecastValueBasis(_CaseInsensitiveStrEnum):
    """Interpretation of an explicit operating-economics path.

    Gross-margin paths predate this field and remain percentage-point paths by
    contract.  New expense paths must declare their basis so an amount cannot
    accidentally be treated as a percentage (or vice versa).
    """

    ABSOLUTE = "absolute"
    PERCENT_OF_REVENUE = "percent_of_revenue"

    @classmethod
    def _missing_(cls, value):
        member = super()._missing_(value)
        if member is not None:
            return member
        normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
        if normalized in {
            "ratio",
            "percent",
            "percentage",
            "percentage_points",
            "percentage_points_of_revenue",
            "percent_of_sales",
        }:
            return cls.PERCENT_OF_REVENUE
        if normalized in {"amount", "currency", "absolute_amount"}:
            return cls.ABSOLUTE
        return None


class ForecastProvenance(BaseModel):
    """Small provider-neutral provenance payload for a manual override."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    source: Optional[str] = None
    origin: Optional[str] = None
    methodology: Optional[str] = None
    reference: Optional[str] = None

    @field_validator("source", "origin", "methodology", "reference")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Forecast provenance text cannot be blank")
        return normalized

    @model_validator(mode="after")
    def require_detail(self) -> "ForecastProvenance":
        if not any(
            value is not None
            for value in (self.source, self.origin, self.methodology, self.reference)
        ):
            raise ValueError("Forecast provenance requires at least one detail")
        return self


def _metric_value(value: ForecastMetric | str) -> str:
    return value.value if isinstance(value, ForecastMetric) else str(value)


def _validate_economics_explicit_record(metric, strategy, path, basis=None) -> None:
    """Validate shared gross-economics manual paths without a second override type."""

    normalized_metric = _metric_value(metric).strip().casefold().replace("-", "_")
    normalized_metric = {
        "research_and_development": "r_and_d",
        "research_and_development_expense": "r_and_d",
        "selling_general_and_administrative": "sg_and_a",
        "selling_general_and_administrative_expense": "sg_and_a",
        "operating_income": "ebit",
        "operating_income_loss": "ebit",
    }.get(normalized_metric, normalized_metric)
    if normalized_metric not in {
        "gross_margin",
        "gross_profit",
        "r_and_d",
        "sg_and_a",
        "other_operating_items",
        "ebit",
    }:
        return
    if strategy == ForecastStrategy.EXPLICIT and path is None:
        raise ValueError(f"Explicit {normalized_metric} records require explicit_path")
    if path is None:
        return
    if normalized_metric == "gross_margin" and any(
        item < Decimal("-100") or item > Decimal("100") for item in path
    ):
        raise ValueError(
            "Gross-margin explicit paths must be between -100 and 100 percentage points"
        )
    if normalized_metric == "gross_profit" and any(item < 0 for item in path):
        raise ValueError("Gross-profit explicit paths cannot be negative")
    if normalized_metric in {"r_and_d", "sg_and_a"}:
        if strategy in {ForecastStrategy.EXPLICIT, ForecastStrategy.RATIO}:
            if path is None:
                raise ValueError(
                    f"{strategy.value} {normalized_metric} records require explicit_path"
                )
            if basis is None:
                raise ValueError(
                    f"{strategy.value} {normalized_metric} records require an explicit value basis"
                )
            expected = (
                ForecastValueBasis.ABSOLUTE
                if strategy == ForecastStrategy.EXPLICIT
                else ForecastValueBasis.PERCENT_OF_REVENUE
            )
            if basis != expected:
                raise ValueError(
                    f"{strategy.value} {normalized_metric} paths require basis={expected.value}"
                )
        if path is not None and any(item < 0 for item in path):
            raise ValueError(f"{normalized_metric} paths cannot be negative")
    elif normalized_metric in {"other_operating_items", "ebit"}:
        if strategy == ForecastStrategy.RATIO:
            raise ValueError(
                f"Ratio {normalized_metric} paths are not supported; use absolute signed values"
            )
        if strategy == ForecastStrategy.EXPLICIT:
            if path is None:
                raise ValueError(
                    f"Explicit {normalized_metric} records require explicit_path"
                )
            if basis is None:
                raise ValueError(
                    f"Explicit {normalized_metric} records require an explicit value basis"
                )
            if basis != ForecastValueBasis.ABSOLUTE:
                raise ValueError(
                    f"Explicit {normalized_metric} paths require basis=absolute"
                )


def _validate_economics_scope(scope, metric) -> None:
    """Reject phase-two company-only metrics at the decision boundary."""

    normalized_metric = _metric_value(metric).strip().casefold().replace("-", "_")
    if normalized_metric in {
        "other_operating_item",
        "other_operating_income",
        "other_operating_expense",
        "recurring_other_operating_items",
    }:
        normalized_metric = "other_operating_items"
    if normalized_metric in {"ebit", "operating_income", "operating_income_loss"}:
        normalized_metric = "ebit"
    if scope == ForecastScope.SEGMENT and normalized_metric in {
        "other_operating_items",
        "ebit",
    }:
        raise ValueError(
            f"Segment {normalized_metric} decisions/overrides are unsupported; "
            "only segment R&D and SG&A are supported in this phase"
        )


def _decision_key(value: "ForecastDecision") -> tuple[str, str, str]:
    return (value.scope.value, value.scope_id, _metric_value(value.metric))


def _override_key(value: "ForecastOverride") -> tuple[str, str, str]:
    return (value.scope.value, value.scope_id, _metric_value(value.metric))


def _normalize_keyed_records(value):
    if not isinstance(value, Mapping):
        return tuple(value)
    if {"scope", "metric", "strategy"}.issubset(value):
        return (value,)
    records = []
    for key, item in value.items():
        if not isinstance(item, Mapping):
            records.append(item)
            continue
        payload = dict(item)
        if isinstance(key, tuple):
            if len(key) == 2:
                payload.setdefault("scope", key[0])
                payload.setdefault("metric", key[1])
            elif len(key) == 3:
                payload.setdefault("scope", key[0])
                payload.setdefault("scope_id", key[1])
                payload.setdefault("metric", key[2])
        records.append(payload)
    return tuple(records)


class ForecastDecision(BaseModel):
    """One immutable, auditable choice for a scoped forecast metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: ForecastScope
    scope_id: str = Field(
        default="company",
        validation_alias=AliasChoices("scope_id", "segment_id", "scope_key"),
    )
    metric: ForecastMetric | str
    strategy: ForecastStrategy
    rationale: str = "Deterministic FCFF planning decision"
    confidence: str = "medium"
    explicit_path: Optional[tuple[Decimal, ...]] = Field(
        default=None,
        validation_alias=AliasChoices("explicit_path", "path"),
    )
    basis: ForecastValueBasis | None = Field(
        default=None,
        validation_alias=AliasChoices("basis", "value_basis", "path_basis"),
    )
    provenance: Optional[str | ForecastProvenance] = None

    @field_validator("scope_id", "rationale")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Forecast decision text cannot be blank")
        return normalized

    @field_validator("metric", mode="before")
    @classmethod
    def normalize_metric(cls, value):
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if not normalized:
                raise ValueError("Forecast decision metric cannot be blank")
            try:
                return ForecastMetric(normalized)
            except ValueError:
                return normalized
        return value

    @field_validator("explicit_path", mode="before")
    @classmethod
    def normalize_explicit_path(cls, value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, Decimal)):
            return (value,)
        return tuple(value)

    @field_validator("explicit_path")
    @classmethod
    def validate_explicit_path(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and not value:
            raise ValueError("Forecast explicit paths cannot be empty")
        if value is not None and any(not item.is_finite() for item in value):
            raise ValueError("Forecast explicit paths must contain finite values")
        return value

    @model_validator(mode="after")
    def validate_scope_identity(self) -> "ForecastDecision":
        if self.scope == ForecastScope.COMPANY and self.scope_id != "company":
            raise ValueError("Company forecast decisions require scope_id='company'")
        if self.scope == ForecastScope.SEGMENT and self.scope_id == "company":
            raise ValueError("Segment forecast decisions require a segment scope_id")
        _validate_economics_scope(self.scope, self.metric)
        _validate_economics_explicit_record(
            self.metric, self.strategy, self.explicit_path, self.basis
        )
        return self

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError(
                "Forecast decision confidence must be high, medium, or low"
            )
        return normalized

    @property
    def key(self) -> tuple[str, str, str]:
        return _decision_key(self)


class ForecastOverride(BaseModel):
    """A manual strategy/path override keyed by scope and metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: ForecastScope
    scope_id: str = Field(
        default="company",
        validation_alias=AliasChoices("scope_id", "segment_id", "scope_key"),
    )
    metric: ForecastMetric | str
    strategy: ForecastStrategy
    explicit_path: Optional[tuple[Decimal, ...]] = Field(
        default=None,
        validation_alias=AliasChoices("explicit_path", "path"),
    )
    basis: ForecastValueBasis | None = Field(
        default=None,
        validation_alias=AliasChoices("basis", "value_basis", "path_basis"),
    )
    provenance: Optional[str | ForecastProvenance] = None

    @field_validator("scope_id")
    @classmethod
    def normalize_scope_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Forecast override scope_id cannot be blank")
        return normalized

    @field_validator("metric", mode="before")
    @classmethod
    def normalize_metric(cls, value):
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if not normalized:
                raise ValueError("Forecast override metric cannot be blank")
            try:
                return ForecastMetric(normalized)
            except ValueError:
                return normalized
        return value

    @field_validator("explicit_path", mode="before")
    @classmethod
    def normalize_explicit_path(cls, value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, Decimal)):
            return (value,)
        return tuple(value)

    @field_validator("explicit_path")
    @classmethod
    def validate_explicit_path(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and not value:
            raise ValueError("Forecast override paths cannot be empty")
        if value is not None and any(not item.is_finite() for item in value):
            raise ValueError("Forecast override paths must contain finite values")
        return value

    @model_validator(mode="after")
    def validate_scope_identity(self) -> "ForecastOverride":
        if self.scope == ForecastScope.COMPANY and self.scope_id != "company":
            raise ValueError("Company forecast overrides require scope_id='company'")
        if self.scope == ForecastScope.SEGMENT and self.scope_id == "company":
            raise ValueError("Segment forecast overrides require a segment scope_id")
        _validate_economics_scope(self.scope, self.metric)
        _validate_economics_explicit_record(
            self.metric, self.strategy, self.explicit_path, self.basis
        )
        return self

    @property
    def key(self) -> tuple[str, str, str]:
        return _override_key(self)

    @property
    def path(self) -> Optional[tuple[Decimal, ...]]:
        """Compatibility alias for the explicit override path."""

        return self.explicit_path


class ForecastPlan(BaseModel):
    """Immutable method resolution and decision audit for one FCFF forecast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested: FcffForecastMethod = Field(
        validation_alias=AliasChoices("requested", "requested_method")
    )
    resolved: FcffForecastMethod = Field(
        validation_alias=AliasChoices("resolved", "resolved_method")
    )
    decisions: tuple[ForecastDecision, ...] = ()
    overrides: tuple[ForecastOverride, ...] = ()
    rationale: str = "Deterministic FCFF forecast method resolution"
    warnings: tuple[str, ...] = ()
    audit: tuple[str, ...] = Field(
        default=(), validation_alias=AliasChoices("audit", "audit_records")
    )
    confidence: str = "medium"

    @field_validator("decisions", "overrides", mode="before")
    @classmethod
    def normalize_records(cls, value):
        if value is None:
            return ()
        if isinstance(value, Mapping):
            return _normalize_keyed_records(value)
        return tuple(value)

    @field_validator("warnings", "audit", mode="before")
    @classmethod
    def normalize_text_records(cls, value):
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    @field_validator("warnings", "audit")
    @classmethod
    def validate_text_records(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("Forecast plan audit text cannot be blank")
        return normalized

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Forecast plan rationale cannot be blank")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError("Forecast plan confidence must be high, medium, or low")
        return normalized

    @model_validator(mode="after")
    def validate_records(self) -> "ForecastPlan":
        decisions = tuple(sorted(self.decisions, key=_decision_key))
        overrides = tuple(sorted(self.overrides, key=_override_key))
        decision_keys = tuple(_decision_key(item) for item in decisions)
        override_keys = tuple(_override_key(item) for item in overrides)
        if len(decision_keys) != len(set(decision_keys)):
            raise ValueError(
                "Forecast plan decisions must be unique by scope and metric"
            )
        if len(override_keys) != len(set(override_keys)):
            raise ValueError("Forecast overrides must be unique by scope and metric")
        if self.requested == FcffForecastMethod.AUTO:
            if self.resolved not in {
                FcffForecastMethod.NORMALIZED,
                FcffForecastMethod.HYBRID,
            }:
                raise ValueError(
                    "AUTO forecast plans must resolve to normalized or hybrid"
                )
        elif self.resolved != self.requested:
            raise ValueError(
                "Explicit forecast plans must resolve to their requested method"
            )
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "overrides", overrides)
        return self

    @property
    def requested_method(self) -> FcffForecastMethod:
        return self.requested

    @property
    def resolved_method(self) -> FcffForecastMethod:
        return self.resolved

    @property
    def method(self) -> FcffForecastMethod:
        """Alias for the concrete method selected for execution."""

        return self.resolved

    @property
    def audit_records(self) -> tuple[str, ...]:
        return self.audit

    def decision(
        self,
        scope: ForecastScope | str,
        metric: ForecastMetric | str,
        scope_id: str = "company",
    ) -> ForecastDecision | None:
        key = (ForecastScope(scope).value, scope_id, _metric_value(metric))
        return next(
            (item for item in self.decisions if _decision_key(item) == key), None
        )


# Explicit aliases keep the FCFF-specific names available without introducing
# a second schema or changing the existing ``config.valuation.ForecastMethod``.
FcffForecastDecision = ForecastDecision
FcffForecastMetric = ForecastMetric
FcffForecastOverride = ForecastOverride
FcffForecastPlan = ForecastPlan
FcffForecastScope = ForecastScope
FcffForecastStrategy = ForecastStrategy


class ForecastSeedType(str, Enum):
    FISCAL_YEAR = "FY"
    TTM = "TTM"
    YTD_PLUS_FORECAST = "YTD+forecast"
    YTD_RUN_RATE = "YTD run-rate"


class SimplifiedFcfForecastParameters(BaseModel):
    """Inputs for a revenue-times-FCF-margin forecast.

    Rates and margins use percentage points: ``5`` means 5%, not 0.05. A
    one-value path is repeated for every projected year; otherwise it must
    contain exactly one value per year.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_years: int = Field(default=5, ge=1, le=30)
    revenue_growth: Optional[tuple[Decimal, ...]] = None
    free_cash_flow_margin: Optional[tuple[Decimal, ...]] = None
    historical_window: int = Field(default=3, ge=1, le=10)

    @field_validator("revenue_growth", "free_cash_flow_margin", mode="before")
    @classmethod
    def normalize_path(cls, value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, Decimal)):
            return (value,)
        return tuple(value)

    @field_validator("revenue_growth")
    @classmethod
    def validate_growth_path(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(not rate.is_finite() for rate in value):
            raise ValueError("Revenue growth must contain finite values")
        if value is not None and any(
            rate <= Decimal("-100") or rate > Decimal("1000") for rate in value
        ):
            raise ValueError(
                "Revenue growth must be greater than -100% and at most 1000%"
            )
        return value

    @field_validator("free_cash_flow_margin")
    @classmethod
    def validate_margin_path(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(not margin.is_finite() for margin in value):
            raise ValueError("Free cash flow margin must contain finite values")
        if value is not None and any(abs(margin) > Decimal("500") for margin in value):
            raise ValueError("Free cash flow margin must be between -500% and 500%")
        return value

    @model_validator(mode="after")
    def validate_path_lengths(self) -> "SimplifiedFcfForecastParameters":
        for name, path in (
            ("revenue_growth", self.revenue_growth),
            ("free_cash_flow_margin", self.free_cash_flow_margin),
        ):
            if path is not None and len(path) not in (1, self.forecast_years):
                raise ValueError(
                    f"{name} must contain one value or {self.forecast_years} values"
                )
        return self


class SimplifiedFcfForecastObservation(BaseModel):
    forecast_year: int
    fiscal_year: int
    period_end: datetime.date
    revenue_growth: Decimal
    revenue: Decimal
    free_cash_flow_margin: Decimal
    free_cash_flow: Decimal
    unit: str
    formula: str = "revenue × free cash flow margin"


class SimplifiedFcfForecast(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    identifiers: Optional[_SecurityIdentifiers] = None
    method: str = "revenue_margin"

    base_fiscal_year: int
    base_period_end: datetime.date
    base_revenue: Decimal
    base_free_cash_flow: Decimal
    unit: str

    parameters: SimplifiedFcfForecastParameters
    historical_fiscal_years: tuple[int, ...]
    revenue_growth_source: ForecastAssumptionSource
    free_cash_flow_margin_source: ForecastAssumptionSource
    observations: list[SimplifiedFcfForecastObservation] = Field(default_factory=list)


class FcffForecastDriver(str, Enum):
    REVENUE_GROWTH = "revenue_growth"
    OPERATING_MARGIN = "operating_margin"
    TAX_RATE = "tax_rate"
    DEPRECIATION_TO_REVENUE = "depreciation_to_revenue"
    CAPEX_TO_REVENUE = "capex_to_revenue"
    OPERATING_WORKING_CAPITAL_TO_REVENUE = "operating_working_capital_to_revenue"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class FcffForecastParameters(BaseModel):
    """Year-specific operating drivers for an unlevered FCFF forecast.

    Every value uses percentage points. A one-value path is repeated for each
    forecast year. A shorter explicit path is extended with its final value;
    omitted paths are inferred from complete annual historical periods. Absolute
    CAPEX constraints are keyed by fiscal year and applied after revenue is known.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_years: int = Field(default=5, ge=1, le=30)
    revenue_growth: Optional[tuple[Decimal, ...]] = None
    operating_margin: Optional[tuple[Decimal, ...]] = None
    tax_rate: Optional[tuple[Decimal, ...]] = None
    depreciation_to_revenue: Optional[tuple[Decimal, ...]] = None
    capex_to_revenue: Optional[tuple[Decimal, ...]] = None
    capex_constraints: dict[int, _MonetaryForecastConstraint] = Field(
        default_factory=dict
    )
    operating_working_capital_to_revenue: Optional[tuple[Decimal, ...]] = None
    revenue_anchors: dict[int, Decimal] = Field(default_factory=dict)
    revenue_anchor_sources: dict[int, ForecastAssumptionSource] = Field(
        default_factory=dict
    )
    assumption_source_overrides: dict[FcffForecastDriver, ForecastAssumptionSource] = (
        Field(default_factory=dict)
    )
    historical_window: int = Field(default=3, ge=1, le=10)

    @field_validator(
        "revenue_growth",
        "operating_margin",
        "tax_rate",
        "depreciation_to_revenue",
        "capex_to_revenue",
        "operating_working_capital_to_revenue",
        mode="before",
    )
    @classmethod
    def normalize_path(cls, value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, Decimal)):
            return (value,)
        return tuple(value)

    @field_validator(
        "revenue_growth",
        "operating_margin",
        "tax_rate",
        "depreciation_to_revenue",
        "capex_to_revenue",
        "operating_working_capital_to_revenue",
    )
    @classmethod
    def validate_finite_path(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(not item.is_finite() for item in value):
            raise ValueError("FCFF driver paths must contain finite values")
        return value

    @field_validator("revenue_growth")
    @classmethod
    def validate_growth(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(
            item <= Decimal("-100") or item > Decimal("1000") for item in value
        ):
            raise ValueError(
                "Revenue growth must be greater than -100% and at most 1000%"
            )
        return value

    @field_validator("tax_rate")
    @classmethod
    def validate_tax_rate(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(
            item < Decimal(0) or item > Decimal(100) for item in value
        ):
            raise ValueError("Tax rate must be between 0% and 100%")
        return value

    @field_validator("depreciation_to_revenue", "capex_to_revenue")
    @classmethod
    def validate_nonnegative_ratio(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(
            item < Decimal(0) or item > Decimal(500) for item in value
        ):
            raise ValueError("D&A and capex ratios must be between 0% and 500%")
        return value

    @field_validator("operating_margin", "operating_working_capital_to_revenue")
    @classmethod
    def validate_signed_ratio(
        cls, value: Optional[tuple[Decimal, ...]]
    ) -> Optional[tuple[Decimal, ...]]:
        if value is not None and any(abs(item) > Decimal(500) for item in value):
            raise ValueError(
                "Operating margin and working-capital ratio must be "
                "between -500% and 500%"
            )
        return value

    @model_validator(mode="after")
    def validate_path_lengths(self) -> "FcffForecastParameters":
        for driver in FcffForecastDriver:
            path = getattr(self, driver.value)
            if path is not None and not 1 <= len(path) <= self.forecast_years:
                raise ValueError(
                    f"{driver.value} must contain between one and "
                    f"{self.forecast_years} values"
                )
        return self

    @field_validator("revenue_anchors")
    @classmethod
    def validate_revenue_anchors(cls, value: dict[int, Decimal]) -> dict[int, Decimal]:
        if any(year < 1900 or year > 2200 for year in value):
            raise ValueError("Revenue anchor fiscal years are invalid")
        if any(not amount.is_finite() or amount <= 0 for amount in value.values()):
            raise ValueError("Revenue anchors must be finite and positive")
        return value

    @field_validator("capex_constraints")
    @classmethod
    def validate_capex_constraints(
        cls, value: dict[int, _MonetaryForecastConstraint]
    ) -> dict[int, _MonetaryForecastConstraint]:
        if any(year < 1900 or year > 2200 for year in value):
            raise ValueError("CAPEX constraint fiscal years are invalid")
        return value


class FcffForecastObservation(BaseModel):
    forecast_year: int
    fiscal_year: int
    period_end: datetime.date
    revenue_growth: Decimal
    revenue: Decimal
    operating_margin: Decimal
    operating_income: Decimal
    tax_rate: Decimal
    nopat: Decimal
    depreciation_to_revenue: Decimal
    depreciation_and_amortization: Decimal
    capex_to_revenue: Decimal
    capital_expenditures: Decimal
    operating_working_capital_to_revenue: Decimal
    operating_working_capital: Decimal
    change_in_operating_working_capital: Decimal
    fcff: Decimal
    unit: str
    cell_audits: dict[str, ForecastValue] = Field(default_factory=dict)
    formula: str = (
        "NOPAT + depreciation and amortization - capital expenditures - "
        "change in operating working capital"
    )


class FcffForecastYtdAnchor(BaseModel):
    """Actual-to-date inputs used to build a YTD-plus-forecast first year.

    The ordinary forecast seed is intentionally kept on the existing FCFF
    fields.  This optional object preserves the additional flow and balance
    sheet values needed when the first projected fiscal year combines actual
    quarters with a forecast of the remaining period.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fiscal_year: int
    ytd_period_end: datetime.date
    fiscal_year_end: datetime.date
    actual_quarters: int = Field(ge=1, le=3)
    actual_revenue: Decimal
    actual_operating_income: Decimal
    actual_pretax_income: Decimal
    actual_income_tax_expense: Decimal
    actual_tax_rate: Optional[Decimal] = None
    actual_depreciation_and_amortization: Decimal
    actual_capital_expenditures: Decimal
    actual_operating_working_capital: Decimal
    latest_annual_revenue: Decimal
    revenue_anchor: Optional[Decimal] = None

    # These are the resolved first-year assumptions before the service turns
    # the first-year outputs into effective driver percentages.
    revenue_growth: Decimal
    operating_margin: Decimal
    tax_rate: Decimal
    depreciation_to_revenue: Decimal
    capex_to_revenue: Decimal
    operating_working_capital_to_revenue: Decimal


class FcffForecastDcfStub(BaseModel):
    """The post-YTD FCFF flow used by a DCF for a YTD-plus-forecast seed.

    ``FcffForecastObservation`` deliberately remains a full-fiscal-year
    observation for reporting.  This object is the separate cash flow that is
    still available after the actual YTD balance-sheet date and is therefore
    the only amount a DCF should use for that first explicit period.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_year: int = Field(default=1, ge=1)
    fiscal_year: int
    period_start: datetime.date
    period_end: datetime.date
    unit: str

    annual_nopat: Decimal
    actual_ytd_nopat: Decimal
    annual_depreciation_and_amortization: Decimal
    actual_ytd_depreciation_and_amortization: Decimal
    annual_capital_expenditures: Decimal
    actual_ytd_capital_expenditures: Decimal
    fiscal_year_end_operating_working_capital: Decimal
    actual_ytd_operating_working_capital: Decimal
    fcff: Decimal
    formula: str = (
        "(annual NOPAT - actual YTD NOPAT) + "
        "(annual D&A - actual YTD D&A) - "
        "(annual CAPEX - actual YTD CAPEX) - "
        "(FY-end OWC - actual YTD OWC)"
    )

    @field_validator(
        "annual_nopat",
        "actual_ytd_nopat",
        "annual_depreciation_and_amortization",
        "actual_ytd_depreciation_and_amortization",
        "annual_capital_expenditures",
        "actual_ytd_capital_expenditures",
        "fiscal_year_end_operating_working_capital",
        "actual_ytd_operating_working_capital",
        "fcff",
    )
    @classmethod
    def require_finite_value(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("FCFF DCF stub values must be finite")
        return value

    @field_validator("unit", "formula")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("FCFF DCF stub text fields cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_period_and_formula(self) -> "FcffForecastDcfStub":
        if self.period_start >= self.period_end:
            raise ValueError("FCFF DCF stub period must end after its start date")
        expected_fcff = (
            self.annual_nopat
            - self.actual_ytd_nopat
            + self.annual_depreciation_and_amortization
            - self.actual_ytd_depreciation_and_amortization
            - self.annual_capital_expenditures
            + self.actual_ytd_capital_expenditures
            - self.fiscal_year_end_operating_working_capital
            + self.actual_ytd_operating_working_capital
        )
        if self.fcff != expected_fcff:
            raise ValueError(
                "FCFF DCF stub FCFF does not match the remaining-period formula"
            )
        return self

    @property
    def remaining_fcff(self) -> Decimal:
        """Return the remaining-period FCFF under an explicit name."""

        return self.fcff

    @property
    def ytd_period_end(self) -> datetime.date:
        """Compatibility alias for the actual balance-sheet date."""

        return self.period_start

    @property
    def fiscal_year_end(self) -> datetime.date:
        """Compatibility alias for the full-year observation end date."""

        return self.period_end

    @property
    def annual_operating_working_capital(self) -> Decimal:
        """Compatibility alias for the full-year OWC balance."""

        return self.fiscal_year_end_operating_working_capital


class FcffForecast(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    identifiers: Optional[_SecurityIdentifiers] = None
    method: str = "driver_based_fcff"
    seed_type: ForecastSeedType = ForecastSeedType.FISCAL_YEAR
    seed_methodology: str = "Latest complete fiscal year"
    seed_period_end: Optional[datetime.date] = None
    current_fiscal_year: Optional[int] = None
    actual_quarters: int = Field(default=0, ge=0, le=4)
    financial_snapshot_retrieved_at: Optional[datetime.datetime] = None
    availability_mode: Optional[str] = None

    base_fiscal_year: int
    base_period_end: datetime.date
    base_revenue: Decimal
    base_operating_income: Decimal
    base_tax_rate: Decimal
    base_nopat: Decimal
    base_depreciation_and_amortization: Decimal
    base_capital_expenditures: Decimal
    base_operating_working_capital: Decimal
    base_fcff: Optional[Decimal] = None
    unit: str

    parameters: FcffForecastParameters
    historical_fiscal_years: tuple[int, ...]
    assumption_sources: dict[FcffForecastDriver, ForecastAssumptionSource]
    assumption_source_paths: dict[
        FcffForecastDriver, tuple[ForecastAssumptionSource, ...]
    ] = Field(default_factory=dict)
    adaptive_stages: tuple[str, ...] = ()
    observations: list[FcffForecastObservation] = Field(default_factory=list)
    warnings: tuple[str, ...] = ()
    ytd_anchor: Optional[FcffForecastYtdAnchor] = None
    dcf_stub: Optional[FcffForecastDcfStub] = None
    capex_constraints_applied: tuple[int, ...] = ()
    # The first projected observation can be a current/YTD estimate.  Keep it
    # separate from the annual history used to normalize the forward growth
    # regime so adaptive forecasting does not mistake a partial-year result for
    # a sustainable long-run anchor.
    current_growth_rate: Optional[Decimal] = None
    normalized_historical_growth: Optional[Decimal] = None
    normalized_historical_growth_path: tuple[Decimal, ...] = ()
    # Optional provider-neutral operating-reconciliation audit.  These fields
    # are populated only when structured operating evidence is injected; the
    # ordinary FCFF path remains unchanged when it is absent.
    operating_driver_coverage: Optional[Decimal] = None
    operating_reconstruction_error: Optional[Decimal] = None
    operating_confidence: Optional[str] = None
    operating_own_supported_years: tuple[int, ...] = ()
    operating_consensus_years: tuple[int, ...] = ()
    operating_divergence_by_year: dict[int, Decimal] = Field(default_factory=dict)
    operating_divergence: Optional[Decimal] = None
    operating_transition_start_year: Optional[int] = None
    operating_warnings: tuple[str, ...] = ()
    operating_selected_revenue_by_year: dict[int, Decimal] = Field(default_factory=dict)
    operating_source_by_year: dict[int, str] = Field(default_factory=dict)
    operating_confidence_by_year: dict[int, str] = Field(default_factory=dict)

    @property
    def current_growth(self) -> Optional[Decimal]:
        return self.current_growth_rate

    @property
    def normalized_forward_growth(self) -> Optional[Decimal]:
        return self.normalized_historical_growth

    @model_validator(mode="before")
    @classmethod
    def normalize_dcf_stub_alias(cls, values):
        if (
            isinstance(values, dict)
            and "dcf_stub" not in values
            and "dcf_remaining_stub" in values
        ):
            normalized = dict(values)
            normalized["dcf_stub"] = normalized.pop("dcf_remaining_stub")
            return normalized
        return values

    @property
    def dcf_remaining_stub(self) -> Optional[FcffForecastDcfStub]:
        """Return the optional DCF-only remaining-period representation."""

        return self.dcf_stub

    @property
    def operating_coverage(self) -> Optional[Decimal]:
        """Short alias for the operating driver reconstruction coverage."""

        return self.operating_driver_coverage

    @property
    def operating_supported_years(self) -> tuple[int, ...]:
        """Short alias for independently supported operating years."""

        return self.operating_own_supported_years

    @property
    def own_supported_years(self) -> tuple[int, ...]:
        return self.operating_own_supported_years

    @property
    def consensus_years(self) -> tuple[int, ...]:
        return self.operating_consensus_years

    @property
    def divergence(self) -> Optional[Decimal]:
        return self.operating_divergence

    @property
    def transition_start_year(self) -> Optional[int]:
        return self.operating_transition_start_year


class AdaptiveMultistagePlan(BaseModel):
    """Stages selected to converge an FCFF projection to perpetual growth."""

    model_config = ConfigDict(frozen=True)

    requested_years: int = Field(ge=1, le=30)
    effective_years: int = Field(ge=1, le=30)
    high_growth_years: int = Field(ge=0, le=30)
    transition_years: int = Field(ge=0, le=30)
    stable_years: int = Field(ge=0, le=30)
    initial_growth_rate: Decimal
    terminal_growth_rate: Decimal
    max_annual_growth_fade: Decimal = Field(gt=0)
    extended_to_stable: bool = False
    explicit_growth_prefix_years: int = Field(default=0, ge=0, le=30)
    terminal_return_on_invested_capital: Optional[Decimal] = None
    terminal_roic_source: Optional[str] = None
    terminal_roic_methodology: Optional[str] = None
    terminal_roic_confidence: Optional[str] = None
    terminal_roic_warnings: tuple[str, ...] = ()
    terminal_reinvestment_rate: Optional[Decimal] = None
    terminal_capex_to_revenue: Optional[Decimal] = None
    capex_transition_years: int = Field(default=0, ge=0, le=30)
    depreciable_asset_life_years: Optional[int] = Field(default=None, ge=2, le=30)
    capex_benefits_modeled: bool = False
    capex_benefits_disclosure: str = (
        "Associated revenue and margin benefits from the CAPEX transition are not "
        "modeled by AdaptiveMultistagePlan."
    )
    forward_evidence_score: Decimal = Decimal("0")
    forward_evidence_summary: tuple[str, ...] = ()
    # ``high_growth_years`` is retained as a serialized compatibility field.
    # Its adaptive-stage meaning is now near-term years, not a claim that the
    # selected rate is economically high.
    current_growth_years: int = Field(default=0, ge=0, le=30)
    current_growth_rate: Optional[Decimal] = None
    forward_growth_rate: Optional[Decimal] = None
    forward_growth_path: tuple[Decimal, ...] = ()
    historical_growth_path: tuple[Decimal, ...] = ()
    management_guidance_path: tuple[Decimal, ...] = ()
    forward_estimates_path: tuple[Decimal, ...] = ()
    forward_growth_path_by_year: tuple[tuple[int, Decimal], ...] = ()
    guidance_growth_path_by_year: tuple[tuple[int, Decimal], ...] = ()
    forward_revenue_estimates: tuple[_ForwardRevenueEstimate, ...] = ()
    forward_estimate_provider: Optional[str] = None
    forward_estimate_years: tuple[int, ...] = ()
    forward_estimate_growth_path: tuple[Decimal, ...] = ()
    forward_estimate_diagnostics: tuple[_ForwardEstimateProviderDiagnostic, ...] = ()
    forward_growth_source: Optional[str] = None
    forward_growth_confidence: Optional[str] = None
    stable_state_supported: bool = False
    current_growth_near_terminal: bool = False
    warnings: tuple[str, ...] = ()
    operating_driver_coverage: Optional[Decimal] = None
    operating_reconstruction_error: Optional[Decimal] = None
    operating_confidence: Optional[str] = None
    operating_own_supported_years: tuple[int, ...] = ()
    operating_consensus_years: tuple[int, ...] = ()
    operating_divergence_by_year: dict[int, Decimal] = Field(default_factory=dict)
    operating_divergence: Optional[Decimal] = None
    operating_transition_start_year: Optional[int] = None
    operating_warnings: tuple[str, ...] = ()
    operating_selected_revenue_by_year: dict[int, Decimal] = Field(default_factory=dict)
    operating_source_by_year: dict[int, str] = Field(default_factory=dict)
    operating_confidence_by_year: dict[int, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stages(self) -> "AdaptiveMultistagePlan":
        if self.effective_years < self.requested_years:
            raise ValueError("effective_years cannot be below requested_years")
        if (
            self.current_growth_years
            + self.high_growth_years
            + self.explicit_growth_prefix_years
            + self.transition_years
            + self.stable_years
            != self.effective_years
        ):
            raise ValueError("Adaptive stages must span the effective forecast")
        return self

    @property
    def near_term_years(self) -> int:
        """Compatibility-friendly name for the initial forward regime."""

        return self.high_growth_years

    @property
    def initial_stage_years(self) -> int:
        return self.high_growth_years

    @property
    def forward_growth_anchor(self) -> Optional[Decimal]:
        return self.forward_growth_rate

    @property
    def normalized_forward_growth(self) -> Optional[Decimal]:
        return self.forward_growth_rate

    @property
    def terminal_ready(self) -> bool:
        return self.stable_state_supported

    @property
    def stable_state_eligible(self) -> bool:
        return self.stable_state_supported

    @property
    def own_supported_years(self) -> tuple[int, ...]:
        return self.operating_own_supported_years

    @property
    def consensus_years(self) -> tuple[int, ...]:
        return self.operating_consensus_years

    @property
    def divergence(self) -> Optional[Decimal]:
        return self.operating_divergence

    @property
    def transition_start_year(self) -> Optional[int]:
        return self.operating_transition_start_year


class ForwardGrowthEvidence(BaseModel):
    """Forward indicators that justify delaying the historical growth fade."""

    model_config = ConfigDict(frozen=True)

    backlog: bool = False
    guidance: bool = False
    capacity: bool = False
    growth_visibility: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    lifecycle: str = "unknown"
    growth_path: tuple[Decimal, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "growth_path",
            "forward_growth_path",
            "consensus_growth_path",
            "explicit_path",
            "path",
        ),
    )
    guidance_growth_path: tuple[Decimal, ...] = ()
    growth_path_by_year: tuple[tuple[int, Decimal], ...] = ()
    source: Optional[str] = None
    forward_revenue_estimates: tuple[_ForwardRevenueEstimate, ...] = ()
    forward_estimate_provider: Optional[str] = None
    forward_estimate_years: tuple[int, ...] = ()
    forward_estimate_growth_path: tuple[Decimal, ...] = ()
    forward_estimate_diagnostics: tuple[_ForwardEstimateProviderDiagnostic, ...] = ()
    growth_anchor: Optional[Decimal] = Field(
        default=None,
        validation_alias=AliasChoices(
            "growth_anchor",
            "forward_growth_anchor",
            "normalized_growth",
            "consensus_growth",
            "forward_anchor",
        ),
    )
    confidence: Optional[str] = None
    # Exact fiscal-year positions for management growth applications.  The
    # compact ``guidance_growth_path`` remains for compatibility, while this
    # mapping prevents consensus-filled years from being mislabeled as
    # management evidence during later reconciliation.
    guidance_growth_path_by_year: tuple[tuple[int, Decimal], ...] = ()

    @field_validator("growth_path", "guidance_growth_path", mode="before")
    @classmethod
    def normalize_growth_path(cls, value):
        if value is None:
            return ()
        if isinstance(value, (str, int, float, Decimal)):
            return (value,)
        return tuple(value)

    @field_validator("guidance_growth_path_by_year", mode="before")
    @classmethod
    def normalize_guidance_year_path(cls, value):
        if value is None:
            return ()
        return tuple((int(year), rate) for year, rate in value)

    @field_validator("growth_path", "guidance_growth_path")
    @classmethod
    def validate_growth_path(cls, value: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        if any(
            not item.is_finite() or item <= Decimal("-100") or item > Decimal("1000")
            for item in value
        ):
            raise ValueError(
                "Forward growth evidence must be finite and greater than -100%"
            )
        return value

    @field_validator("growth_path_by_year", mode="before")
    @classmethod
    def normalize_year_path(cls, value):
        if value is None:
            return ()
        return tuple((int(year), rate) for year, rate in value)

    @field_validator("growth_anchor")
    @classmethod
    def validate_growth_anchor(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and (
            not value.is_finite() or value <= Decimal("-100") or value > Decimal("1000")
        ):
            raise ValueError(
                "Forward growth evidence anchor must be finite and greater than -100%"
            )
        return value

    @field_validator("confidence")
    @classmethod
    def validate_optional_confidence(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError("Forward growth confidence must be high, medium, or low")
        return normalized

    @model_validator(mode="after")
    def normalize_anchor(self) -> "ForwardGrowthEvidence":
        if self.growth_anchor is None and self.growth_path:
            object.__setattr__(self, "growth_anchor", self.growth_path[-1])
        return self

    @property
    def forward_growth_path(self) -> tuple[Decimal, ...]:
        return self.growth_path

    @property
    def forward_growth_anchor(self) -> Optional[Decimal]:
        return self.growth_anchor

    @property
    def score(self) -> Decimal:
        lifecycle_score = {
            "growth": Decimal("1"),
            "unprofitable_growth": Decimal("0.9"),
            "mature": Decimal("0.25"),
            "declining": Decimal("0"),
            "distressed": Decimal("0"),
            "pre_revenue": Decimal("0.8"),
        }.get(self.lifecycle, Decimal("0.1"))
        return (
            Decimal("0.20") * Decimal(self.backlog)
            + Decimal("0.20") * Decimal(self.guidance)
            + Decimal("0.15") * Decimal(self.capacity)
            + Decimal("0.20") * self.growth_visibility
            + Decimal("0.25") * lifecycle_score
        )

    @property
    def summary(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, present in (
                ("forward growth path", bool(self.growth_path)),
                ("backlog/bookings", self.backlog),
                ("forward guidance", self.guidance),
                ("capacity", self.capacity),
                ("growth visibility", self.growth_visibility > 0),
                (f"{self.lifecycle} lifecycle", self.lifecycle != "unknown"),
            )
            if present
        )


class ForwardGrowthOutlook(BaseModel):
    """Resolved forward revenue-growth outlook used by adaptive FCFF.

    ``current_growth`` is deliberately not the same value as
    ``normalized_growth``.  The former may describe a partial current fiscal
    year, while the latter is the forward anchor that drives the near-term and
    transition stages.  ``growth_path`` is reserved for explicit quantitative
    forward evidence and is preserved before the adaptive fade.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    growth_path: tuple[Decimal, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "growth_path",
            "forward_path",
            "forward_growth_path",
            "consensus_growth_path",
            "explicit_path",
            "path",
        ),
    )
    historical_growth_path: tuple[Decimal, ...] = ()
    management_guidance_path: tuple[Decimal, ...] = ()
    forward_estimates_path: tuple[Decimal, ...] = ()
    growth_path_by_year: tuple[tuple[int, Decimal], ...] = ()
    guidance_growth_path_by_year: tuple[tuple[int, Decimal], ...] = ()
    forward_revenue_estimates: tuple[_ForwardRevenueEstimate, ...] = ()
    forward_estimate_provider: Optional[str] = None
    forward_estimate_years: tuple[int, ...] = ()
    forward_estimate_growth_path: tuple[Decimal, ...] = ()
    forward_estimate_diagnostics: tuple[_ForwardEstimateProviderDiagnostic, ...] = ()
    normalized_growth: Optional[Decimal] = Field(
        default=None,
        validation_alias=AliasChoices(
            "normalized_growth",
            "normalized_forward_growth",
            "normalized_forward_anchor",
            "forward_growth_anchor",
            "anchor",
            "consensus_growth",
        ),
    )
    source: str = ForecastAssumptionSource.NORMALIZED_HISTORICAL.value
    confidence: str = "medium"
    current_growth: Optional[Decimal] = None
    stable_state_supported: bool = False
    current_growth_near_terminal: bool = False
    warnings: tuple[str, ...] = ()

    @field_validator(
        "growth_path",
        "historical_growth_path",
        "management_guidance_path",
        "forward_estimates_path",
        mode="before",
    )
    @classmethod
    def normalize_path(cls, value):
        if value is None:
            return ()
        if isinstance(value, (str, int, float, Decimal)):
            return (value,)
        return tuple(value)

    @field_validator("growth_path_by_year", mode="before")
    @classmethod
    def normalize_year_path(cls, value):
        if value is None:
            return ()
        return tuple((int(year), rate) for year, rate in value)

    @field_validator("guidance_growth_path_by_year", mode="before")
    @classmethod
    def normalize_guidance_year_path(cls, value):
        if value is None:
            return ()
        return tuple((int(year), rate) for year, rate in value)

    @field_validator(
        "growth_path",
        "historical_growth_path",
        "management_guidance_path",
        "forward_estimates_path",
    )
    @classmethod
    def validate_path(cls, value: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        if any(
            not item.is_finite() or item <= Decimal("-100") or item > Decimal("1000")
            for item in value
        ):
            raise ValueError(
                "Forward growth path must be finite and greater than -100%"
            )
        return value

    @field_validator("normalized_growth", "current_growth")
    @classmethod
    def validate_rate(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and (
            not value.is_finite() or value <= Decimal("-100") or value > Decimal("1000")
        ):
            raise ValueError(
                "Forward growth rates must be finite and greater than -100%"
            )
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError("Forward growth confidence must be high, medium, or low")
        return normalized

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Forward growth source cannot be blank")
        return normalized

    @model_validator(mode="after")
    def require_anchor(self) -> "ForwardGrowthOutlook":
        if self.normalized_growth is None and self.growth_path:
            object.__setattr__(self, "normalized_growth", self.growth_path[-1])
        if self.normalized_growth is None:
            raise ValueError("Forward growth outlook requires an anchor or path")
        return self

    @property
    def path(self) -> tuple[Decimal, ...]:
        return self.growth_path

    @property
    def forward_path(self) -> tuple[Decimal, ...]:
        return self.growth_path

    @property
    def anchor(self) -> Decimal:
        assert self.normalized_growth is not None
        return self.normalized_growth

    @property
    def forward_growth_anchor(self) -> Decimal:
        return self.anchor

    @property
    def normalized_forward_growth(self) -> Decimal:
        return self.anchor

    @property
    def terminal_ready(self) -> bool:
        return self.stable_state_supported

    @property
    def stable_state_eligible(self) -> bool:
        return self.stable_state_supported


__all__ = [
    "AdaptiveMultistagePlan",
    "FcffForecastMethod",
    "FcffForecastDecision",
    "FcffForecastMetric",
    "FcffForecastOverride",
    "FcffForecastPlan",
    "FcffForecastScope",
    "FcffForecastStrategy",
    "FcffForecast",
    "FcffForecastDcfStub",
    "FcffForecastDriver",
    "FcffForecastObservation",
    "FcffForecastParameters",
    "FcffForecastYtdAnchor",
    "ForecastAssumptionSource",
    "ForecastDecision",
    "ForecastMetric",
    "ForecastOverride",
    "ForecastPlan",
    "ForecastProvenance",
    "ForecastScope",
    "ForecastSeedType",
    "ForecastStrategy",
    "ForecastValue",
    "ForecastValueBasis",
    "ForwardGrowthEvidence",
    "ForwardGrowthOutlook",
    "SimplifiedFcfForecast",
    "SimplifiedFcfForecastObservation",
    "SimplifiedFcfForecastParameters",
]
