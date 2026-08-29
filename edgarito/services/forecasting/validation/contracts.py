"""Immutable, provider-neutral contracts for forecast sanity validation.

The validation package deliberately owns these small contracts instead of
depending on one of the forecast or valuation implementations.  Adapters can
therefore pass a normalised forecast, a hybrid forecast, or a driver-based
forecast without making the validator aware of how the artifact was built.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, ClassVar

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")


def decimal_value(value: Any) -> Decimal:
    """Coerce a scalar to a finite Decimal without inheriting float noise."""

    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid financial numbers")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot coerce {value!r} to Decimal") from exc
    if not result.is_finite():
        raise ValueError("Financial numbers must be finite")
    return result


class ValidationSeverity(str, Enum):
    """Ordered impact levels used by validation findings."""

    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def _missing_(cls, value: Any):
        if isinstance(value, str):
            normalized = value.strip().casefold()
            return next((member for member in cls if member.value == normalized), None)
        return None

    @property
    def rank(self) -> int:
        return (
            ValidationSeverity.INFO,
            ValidationSeverity.WARNING,
            ValidationSeverity.HIGH,
            ValidationSeverity.CRITICAL,
        ).index(self)


class ValidationCategory(str, Enum):
    """Stable categories that allow consumers to group findings."""

    HORIZON = "horizon"
    GROWTH = "growth"
    MARGIN = "margin"
    OPERATING_MARGIN = "operating_margin"
    REINVESTMENT = "reinvestment"
    WORKING_CAPITAL = "working_capital"
    TERMINAL = "terminal"
    TERMINAL_ECONOMICS = "terminal_economics"
    DISCONTINUITY = "discontinuity"
    COMPOUNDING = "compounding"
    CONSISTENCY = "consistency"

    @classmethod
    def _missing_(cls, value: Any):
        if isinstance(value, str):
            normalized = value.strip().casefold()
            return next((member for member in cls if member.value == normalized), None)
        return None


# Friendly aliases for callers that prefer the shorter names.
Severity = ValidationSeverity
Category = ValidationCategory
ForecastValidationSeverity = ValidationSeverity
ForecastValidationCategory = ValidationCategory


class ForecastValidationFinding(BaseModel):
    """One immutable, auditable validation observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    fiscal_year: int | None = Field(
        default=None,
        validation_alias=AliasChoices("fiscal_year", "year"),
    )
    metric: str | None = None
    observed_value: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("observed_value", "observed"),
    )
    threshold: Decimal | None = None
    reference_value: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("reference_value", "reference"),
    )
    explanation: str | None = None

    @field_validator("observed_value", "threshold", "reference_value", mode="before")
    @classmethod
    def coerce_numbers(cls, value: Any) -> Decimal | None:
        return None if value is None else decimal_value(value)

    @field_validator("code", "message")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Validation finding text cannot be blank")
        return normalized

    @field_validator("metric", "explanation")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @property
    def year(self) -> int | None:
        """Compatibility alias for consumers using ``year`` terminology."""

        return self.fiscal_year

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.fiscal_year is None,
            self.fiscal_year if self.fiscal_year is not None else 0,
            self.category.value,
            self.code,
            self.metric or "",
            self.severity.rank,
            self.message,
            str(self.observed_value) if self.observed_value is not None else "",
            str(self.threshold) if self.threshold is not None else "",
            str(self.reference_value) if self.reference_value is not None else "",
            self.explanation or "",
        )


# A shorter name is convenient in rule modules and remains a public contract.
ValidationFinding = ForecastValidationFinding
Finding = ForecastValidationFinding


class TerminalMetrics(BaseModel):
    """Optional DCF and terminal-economics values in percentage-point units."""

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        populate_by_name=True,
        from_attributes=True,
    )

    terminal_growth_rate: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "terminal_growth_rate", "perpetual_growth_rate", "terminal_g", "g"
        ),
    )
    wacc: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "wacc", "terminal_wacc", "discount_rate", "discount_rate_pct"
        ),
    )
    terminal_value: Decimal | None = None
    terminal_value_pv: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "terminal_value_pv",
            "pv_terminal_value",
            "pv_tv",
            "terminal_present_value",
        ),
    )
    terminal_value_share_pct: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "terminal_value_share_pct",
            "terminal_value_percentage",
            "terminal_share_pct",
            "terminal_value_share",
            "terminal_share",
        ),
    )
    enterprise_value: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("enterprise_value", "ev"),
    )
    explicit_forecast_pv: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "explicit_forecast_pv",
            "pv_explicit_forecast",
            "explicit_pv",
            "pv_explicit",
        ),
    )
    terminal_nopat: Decimal | None = None
    terminal_fcff: Decimal | None = None
    terminal_operating_margin: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("terminal_operating_margin", "terminal_margin"),
    )
    terminal_fcff_margin: Decimal | None = None
    terminal_revenue: Decimal | None = None
    terminal_roic: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "terminal_roic",
            "terminal_return_on_invested_capital",
            "return_on_invested_capital",
        ),
    )
    terminal_reinvestment_rate: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "terminal_reinvestment_rate",
            "terminal_reinvestment_rate_pct",
            "reinvestment_rate",
        ),
    )
    terminal_reinvestment: Decimal | None = None
    terminal_capex_to_revenue: Decimal | None = None
    terminal_delta_nwc: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("terminal_delta_nwc", "terminal_change_nwc"),
    )
    terminal_da: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "terminal_da", "terminal_depreciation_and_amortization"
        ),
    )
    terminal_capex: Decimal | None = None

    @field_validator(
        "terminal_growth_rate",
        "wacc",
        "terminal_value",
        "terminal_value_pv",
        "terminal_value_share_pct",
        "enterprise_value",
        "explicit_forecast_pv",
        "terminal_nopat",
        "terminal_fcff",
        "terminal_operating_margin",
        "terminal_fcff_margin",
        "terminal_revenue",
        "terminal_roic",
        "terminal_reinvestment_rate",
        "terminal_reinvestment",
        "terminal_capex_to_revenue",
        "terminal_delta_nwc",
        "terminal_da",
        "terminal_capex",
        mode="before",
    )
    @classmethod
    def coerce_numbers(cls, value: Any) -> Decimal | None:
        return None if value is None else decimal_value(value)


class ForecastYearRow(BaseModel):
    """One forecast year with optional values for every independent rule."""

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        populate_by_name=True,
        from_attributes=True,
    )

    fiscal_year: int = Field(
        validation_alias=AliasChoices("fiscal_year", "year", "calendar_year")
    )
    forecast_year: int | None = None
    revenue: Decimal | None = None
    fcff: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "fcff", "free_cash_flow_to_firm", "free_cash_flow"
        ),
    )
    ebit: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("ebit", "operating_income", "operating_profit"),
    )
    other_operating_income: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "other_operating_income",
            "positive_operating_items",
            "operating_income_adjustments",
        ),
    )
    operating_margin: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("operating_margin", "ebit_margin"),
    )
    gross_profit: Decimal | None = None
    nopat: Decimal | None = None
    tax: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("tax", "tax_expense", "income_tax_expense"),
    )
    tax_rate: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("tax_rate", "effective_tax_rate"),
    )
    depreciation_and_amortization: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "depreciation_and_amortization", "da", "depreciation"
        ),
    )
    capex: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "capex", "capital_expenditures", "capital_expenditure"
        ),
    )
    delta_nwc: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "delta_nwc",
            "change_in_operating_working_capital",
            "change_in_nwc",
            "delta_working_capital",
        ),
    )
    reinvestment_rate: Decimal | None = None

    _DECIMAL_FIELDS: ClassVar[tuple[str, ...]] = (
        "revenue",
        "fcff",
        "ebit",
        "other_operating_income",
        "operating_margin",
        "gross_profit",
        "nopat",
        "tax",
        "tax_rate",
        "depreciation_and_amortization",
        "capex",
        "delta_nwc",
        "reinvestment_rate",
    )

    @field_validator(*_DECIMAL_FIELDS, mode="before")
    @classmethod
    def coerce_numbers(cls, value: Any) -> Decimal | None:
        return None if value is None else decimal_value(value)

    @property
    def year(self) -> int:
        return self.fiscal_year

    @property
    def operating_income(self) -> Decimal | None:
        return self.ebit

    @property
    def da(self) -> Decimal | None:
        return self.depreciation_and_amortization


ForecastRow = ForecastYearRow


class ForecastValidationContext(BaseModel):
    """A copied, partial-data view consumed by validation rules."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    rows: tuple[ForecastYearRow, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "rows", "forecast_rows", "observations", "forecast", "periods", "years"
        ),
    )
    rows_supplied: bool = False
    historical_rows: tuple[ForecastYearRow, ...] = ()
    terminal: TerminalMetrics | None = None
    methodology: str | None = None
    unit: str | None = None

    @model_validator(mode="before")
    @classmethod
    def collect_flat_terminal_metrics(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        if "rows_supplied" not in payload:
            payload["rows_supplied"] = any(
                name in payload
                for name in (
                    "rows",
                    "forecast_rows",
                    "observations",
                    "forecast",
                    "periods",
                    "years",
                    "explicit_forecast",
                )
            )
        if payload.get("terminal") is not None:
            return payload
        for nested_name in ("terminal_metrics", "valuation", "dcf", "valuation_result"):
            nested = payload.get(nested_name)
            if nested is not None:
                payload["terminal"] = _merge_terminal_payload(payload, nested) or nested
                return payload
        terminal_names = {
            "terminal_growth_rate",
            "perpetual_growth_rate",
            "terminal_g",
            "g",
            "wacc",
            "terminal_wacc",
            "discount_rate",
            "terminal_value",
            "terminal_value_pv",
            "terminal_present_value",
            "pv_terminal_value",
            "pv_tv",
            "terminal_value_percentage",
            "terminal_value_share_pct",
            "terminal_share_pct",
            "terminal_value_share",
            "terminal_share",
            "enterprise_value",
            "ev",
            "explicit_forecast_pv",
            "pv_explicit_forecast",
            "explicit_pv",
            "pv_explicit",
            "terminal_nopat",
            "terminal_fcff",
            "terminal_operating_margin",
            "terminal_fcff_margin",
            "terminal_revenue",
            "terminal_roic",
            "terminal_return_on_invested_capital",
            "return_on_invested_capital",
            "terminal_reinvestment_rate",
            "terminal_reinvestment",
            "terminal_capex_to_revenue",
            "terminal_delta_nwc",
            "terminal_change_nwc",
            "terminal_da",
            "terminal_depreciation_and_amortization",
            "terminal_capex",
        }
        flat_terminal = {
            name: payload.pop(name)
            for name in terminal_names
            if name in payload and payload[name] is not None
        }
        if flat_terminal:
            payload["terminal"] = flat_terminal
        return payload

    @field_validator("rows", "historical_rows", mode="before")
    @classmethod
    def coerce_rows(cls, value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, Mapping):
            value = tuple(value.values())
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            return (value,)
        return tuple(value)

    @field_validator("methodology", "unit")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @property
    def forecast_rows(self) -> tuple[ForecastYearRow, ...]:
        return self.rows

    @classmethod
    def from_artifact(cls, artifact: Any) -> "ForecastValidationContext":
        """Adapt a model, dataclass, or mapping without importing its type."""

        if isinstance(artifact, cls):
            return artifact
        if isinstance(artifact, Mapping):
            payload = dict(artifact)
        elif hasattr(artifact, "model_dump"):
            payload = dict(artifact.model_dump(mode="python"))
        elif isinstance(artifact, Iterable) and not isinstance(artifact, (str, bytes)):
            return cls(rows=tuple(artifact))
        else:
            payload = _object_payload(artifact)

        row_source = None
        rows_supplied = False
        for name in (
            "rows",
            "forecast_rows",
            "observations",
            "explicit_forecast",
            "periods",
            "years",
        ):
            if name in payload:
                row_source = payload[name]
                rows_supplied = True
                break
        # Composite artifacts commonly place the forecast under ``forecast``.
        if row_source is None and "forecast" in payload:
            forecast_payload = payload["forecast"]
            if _looks_like_rows(forecast_payload):
                row_source = forecast_payload
                rows_supplied = True
            else:
                nested_forecast = _mapping_payload(forecast_payload)
                for name in (
                    "rows",
                    "forecast_rows",
                    "observations",
                    "explicit_forecast",
                    "periods",
                    "years",
                ):
                    if name in nested_forecast:
                        row_source = nested_forecast[name]
                        rows_supplied = True
                        break
        if row_source is None and _looks_like_rows(payload):
            row_source = payload
            rows_supplied = True

        historical = _first_present(payload, "historical_rows", "history", "historical")
        terminal_payload = _first_present(
            payload,
            "terminal",
            "terminal_metrics",
            "valuation",
            "dcf",
            "valuation_result",
        )
        if terminal_payload is None:
            terminal_payload = payload
        merged_terminal = _merge_terminal_payload(payload, terminal_payload)

        context_payload: dict[str, Any] = {
            "rows": row_source if row_source is not None else (),
            "rows_supplied": rows_supplied,
            "historical_rows": historical or (),
            "terminal": merged_terminal or None,
            "methodology": _first_present(
                payload, "methodology", "method", "forecast_method"
            ),
            "unit": _first_present(payload, "unit", "currency"),
        }
        return cls.model_validate(context_payload)


ForecastValidationArtifact = ForecastValidationContext


class ForecastValidationResult(BaseModel):
    """Immutable findings with stable ordering, counts, and serialization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    findings: tuple[ForecastValidationFinding, ...] = ()

    @field_validator("findings", mode="before")
    @classmethod
    def normalize_findings(cls, value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        return tuple(value)

    @field_validator("findings")
    @classmethod
    def sort_findings(
        cls, value: tuple[ForecastValidationFinding, ...]
    ) -> tuple[ForecastValidationFinding, ...]:
        return tuple(sorted(value, key=ForecastValidationFinding.sort_key))

    @property
    def highest_severity(self) -> ValidationSeverity | None:
        if not self.findings:
            return None
        return max(
            (item.severity for item in self.findings), key=lambda item: item.rank
        )

    @property
    def warning_count(self) -> int:
        return sum(
            item.severity == ValidationSeverity.WARNING for item in self.findings
        )

    @property
    def high_count(self) -> int:
        return sum(item.severity == ValidationSeverity.HIGH for item in self.findings)

    @property
    def critical_count(self) -> int:
        return sum(
            item.severity == ValidationSeverity.CRITICAL for item in self.findings
        )

    @property
    def error_count(self) -> int:
        return self.high_count + self.critical_count

    @property
    def info_count(self) -> int:
        return sum(item.severity == ValidationSeverity.INFO for item in self.findings)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "info": self.info_count,
            "warning": self.warning_count,
            "high": self.high_count,
            "critical": self.critical_count,
            "error": self.error_count,
            "total": len(self.findings),
        }

    def deterministic_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary with stable key/value forms."""

        return {
            "findings": [
                {
                    "code": finding.code,
                    "severity": finding.severity.value,
                    "category": finding.category.value,
                    "message": finding.message,
                    "fiscal_year": finding.fiscal_year,
                    "metric": finding.metric,
                    "observed_value": (
                        str(finding.observed_value)
                        if finding.observed_value is not None
                        else None
                    ),
                    "threshold": (
                        str(finding.threshold)
                        if finding.threshold is not None
                        else None
                    ),
                    "reference_value": (
                        str(finding.reference_value)
                        if finding.reference_value is not None
                        else None
                    ),
                    "explanation": finding.explanation,
                }
                for finding in self.findings
            ],
            "counts": self.counts,
            "highest_severity": (
                self.highest_severity.value
                if self.highest_severity is not None
                else None
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.deterministic_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    as_dict = deterministic_dict
    serialize = to_json


ValidationResult = ForecastValidationResult


def _object_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    try:
        return dict(vars(value))
    except TypeError:
        return {
            name: getattr(value, name)
            for name in (
                "rows",
                "forecast_rows",
                "observations",
                "forecast",
                "periods",
                "years",
                "terminal",
                "valuation",
                "dcf",
                "multistage_plan",
                "wacc",
                "perpetual_growth_rate",
                "enterprise_value",
                "unit",
            )
            if hasattr(value, name)
        }


def _mapping_payload(value: Any) -> dict[str, Any]:
    return _object_payload(value)


def _first_present(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _looks_like_rows(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Mapping):
        if any(key in value for key in ("fiscal_year", "year", "revenue", "fcff")):
            return True
        return bool(value) and all(_looks_like_rows(item) for item in value.values())
    if isinstance(value, Sequence):
        return True
    return any(
        hasattr(value, key) for key in ("fiscal_year", "year", "revenue", "fcff")
    )


def _merge_terminal_payload(
    root: Mapping[str, Any], nested: Any
) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if nested is not None:
        nested_payload = _mapping_payload(nested)
        container_names = {
            "parameters",
            "multistage_plan",
            "terminal_value",
            "terminal_present_value",
            "explicit_forecast_present_value",
        }
        for key, value in nested_payload.items():
            if key not in container_names and _is_scalar(value):
                merged[key] = value
        parameters = nested_payload.get("parameters")
        if parameters is not None:
            _merge_scalar_fields(merged, _mapping_payload(parameters))

        multistage_plan = nested_payload.get("multistage_plan")
        if multistage_plan is not None:
            _merge_scalar_fields(merged, _mapping_payload(multistage_plan))

        terminal_value = nested_payload.get("terminal_value")
        if terminal_value is not None:
            terminal_value_payload = _mapping_payload(terminal_value)
            scalar_value = terminal_value_payload.get("terminal_value")
            if _is_scalar(scalar_value):
                merged["terminal_value"] = scalar_value
            _merge_scalar_fields(merged, terminal_value_payload)

        terminal_present_value = nested_payload.get("terminal_present_value")
        if terminal_present_value is not None:
            terminal_pv_payload = _mapping_payload(terminal_present_value)
            present_value = terminal_pv_payload.get("present_value")
            if _is_scalar(present_value):
                merged["terminal_value_pv"] = present_value
            _merge_scalar_fields(merged, terminal_pv_payload)

        explicit_present_value = nested_payload.get("explicit_forecast_present_value")
        if explicit_present_value is not None:
            explicit_pv_payload = _mapping_payload(explicit_present_value)
            total_present_value = explicit_pv_payload.get("total_present_value")
            if _is_scalar(total_present_value):
                merged["explicit_forecast_pv"] = total_present_value
            _merge_scalar_fields(merged, explicit_pv_payload)

    root_multistage_plan = root.get("multistage_plan")
    if root_multistage_plan is not None:
        _merge_scalar_fields(merged, _mapping_payload(root_multistage_plan))
    for key in (
        "terminal_growth_rate",
        "perpetual_growth_rate",
        "terminal_g",
        "g",
        "wacc",
        "terminal_wacc",
        "discount_rate",
        "terminal_value",
        "terminal_value_pv",
        "terminal_value_share_pct",
        "terminal_value_percentage",
        "terminal_share_pct",
        "terminal_value_share",
        "terminal_share",
        "pv_terminal_value",
        "pv_tv",
        "terminal_present_value",
        "enterprise_value",
        "ev",
        "explicit_forecast_pv",
        "pv_explicit_forecast",
        "explicit_pv",
        "pv_explicit",
        "terminal_nopat",
        "terminal_fcff",
        "terminal_operating_margin",
        "terminal_fcff_margin",
        "terminal_revenue",
        "terminal_roic",
        "terminal_return_on_invested_capital",
        "return_on_invested_capital",
        "terminal_reinvestment_rate",
        "terminal_reinvestment",
        "terminal_capex_to_revenue",
        "terminal_delta_nwc",
        "terminal_change_nwc",
        "terminal_da",
        "terminal_depreciation_and_amortization",
        "terminal_capex",
    ):
        if (
            key in root
            and root[key] is not None
            and not (
                key in {"terminal_value", "terminal_present_value"}
                and not isinstance(root[key], (str, bytes, Decimal, int, float))
            )
        ):
            merged[key] = root[key]
    return merged or None


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, bytes, Decimal, int, float)) and not isinstance(
        value, bool
    )


def _merge_scalar_fields(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if _is_scalar(value):
            target[key] = value


__all__ = [
    "Category",
    "Finding",
    "ForecastRow",
    "ForecastValidationArtifact",
    "ForecastValidationCategory",
    "ForecastValidationContext",
    "ForecastValidationFinding",
    "ForecastValidationResult",
    "ForecastValidationSeverity",
    "ForecastYearRow",
    "HUNDRED",
    "ONE",
    "Severity",
    "TerminalMetrics",
    "ValidationCategory",
    "ValidationFinding",
    "ValidationResult",
    "ValidationSeverity",
    "ZERO",
    "decimal_value",
]
