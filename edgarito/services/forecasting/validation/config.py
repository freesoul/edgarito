"""Explicit thresholds for the deterministic forecast validation rules."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .contracts import decimal_value


class ForecastValidationConfig(BaseModel):
    """Conservative defaults intended to find arithmetic absurdities, not quality.

    Percent and percentage-point settings use the same convention as the
    forecasting contracts in this repository: ``50`` means 50 percent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    minimum_explicit_years: int = Field(default=2, ge=1)
    maximum_explicit_years: int = Field(default=20, ge=1)
    missing_year_severity: str = "warning"

    max_fcff_growth_pct: Decimal = Field(
        default=Decimal("50"),
        validation_alias=AliasChoices(
            "max_fcff_growth_pct",
            "max_fcff_yoy_growth_pct",
            "fcff_growth_threshold_pct",
            "fcff_growth_warning_threshold_pct",
        ),
    )
    max_fcff_jump_pct: Decimal = Decimal("100")
    repeated_hypergrowth_years: int = Field(default=3, ge=2)
    near_zero_denominator: Decimal = Field(
        default=Decimal("0.01"),
        validation_alias=AliasChoices("near_zero_denominator", "near_zero_threshold"),
    )

    max_fcff_margin_pct: Decimal = Decimal("100")
    min_fcff_margin_pct: Decimal = Decimal("-100")
    max_fcff_margin_change_points: Decimal = Decimal("50")
    max_terminal_fcff_margin_pct: Decimal = Decimal("100")

    max_operating_margin_pct: Decimal = Decimal("100")
    min_operating_margin_pct: Decimal = Decimal("-100")
    max_operating_margin_change_points: Decimal = Decimal("30")
    mechanical_expansion_years: int = Field(default=3, ge=2)
    mechanical_expansion_min_points: Decimal = Decimal("1")
    mechanical_expansion_tolerance_points: Decimal = Decimal("0.1")
    max_terminal_operating_margin_pct: Decimal = Decimal("100")

    max_capex_to_revenue_pct: Decimal = Decimal("200")
    max_da_to_revenue_pct: Decimal = Decimal("200")
    max_capex_to_da_ratio: Decimal = Decimal("10")
    min_capex_to_da_ratio: Decimal = Decimal("0")
    negligible_capex_to_revenue_pct: Decimal = Decimal("1")
    high_growth_for_reinvestment_pct: Decimal = Decimal("50")
    persistent_reinvestment_years: int = Field(default=3, ge=2)
    max_da_to_capex_ratio: Decimal = Decimal("10")
    max_capex_ratio_collapse_points: Decimal = Decimal("25")

    max_delta_nwc_to_revenue_pct: Decimal = Decimal("100")
    max_delta_nwc_to_delta_revenue_pct: Decimal = Decimal("1000")
    max_working_capital_source_pct: Decimal = Decimal("50")
    persistent_working_capital_years: int = Field(default=3, ge=2)

    terminal_value_share_warning_pct: Decimal = Field(
        default=Decimal("75"),
        validation_alias=AliasChoices(
            "terminal_value_share_warning_pct",
            "terminal_share_warning_pct",
            "terminal_value_share_threshold_pct",
        ),
    )
    terminal_growth_stepdown_points: Decimal = Decimal("10")
    terminal_margin_discontinuity_points: Decimal = Decimal("20")
    terminal_reinvestment_discontinuity_points: Decimal = Decimal("20")
    terminal_identity_tolerance_pct: Decimal = Decimal("5")
    minimum_terminal_reinvestment_pct: Decimal = Decimal("1")

    compounding_revenue_multiple: Decimal = Decimal("25")
    compounding_fcff_multiple: Decimal = Decimal("100")

    fcff_identity_tolerance_pct: Decimal = Field(
        default=Decimal("1"),
        validation_alias=AliasChoices(
            "fcff_identity_tolerance_pct",
            "fcff_identity_relative_tolerance_pct",
            "identity_tolerance_pct",
            "identity_tolerance",
        ),
    )
    absolute_identity_tolerance: Decimal = Decimal("0.01")

    @field_validator(
        "max_fcff_growth_pct",
        "max_fcff_jump_pct",
        "near_zero_denominator",
        "max_fcff_margin_pct",
        "min_fcff_margin_pct",
        "max_fcff_margin_change_points",
        "max_terminal_fcff_margin_pct",
        "max_operating_margin_pct",
        "min_operating_margin_pct",
        "max_operating_margin_change_points",
        "mechanical_expansion_min_points",
        "mechanical_expansion_tolerance_points",
        "max_terminal_operating_margin_pct",
        "max_capex_to_revenue_pct",
        "max_da_to_revenue_pct",
        "max_capex_to_da_ratio",
        "min_capex_to_da_ratio",
        "negligible_capex_to_revenue_pct",
        "high_growth_for_reinvestment_pct",
        "max_da_to_capex_ratio",
        "max_capex_ratio_collapse_points",
        "max_delta_nwc_to_revenue_pct",
        "max_delta_nwc_to_delta_revenue_pct",
        "max_working_capital_source_pct",
        "terminal_value_share_warning_pct",
        "terminal_growth_stepdown_points",
        "terminal_margin_discontinuity_points",
        "terminal_reinvestment_discontinuity_points",
        "terminal_identity_tolerance_pct",
        "minimum_terminal_reinvestment_pct",
        "compounding_revenue_multiple",
        "compounding_fcff_multiple",
        "fcff_identity_tolerance_pct",
        "absolute_identity_tolerance",
        mode="before",
    )
    @classmethod
    def coerce_decimals(cls, value: Any) -> Decimal:
        return decimal_value(value)

    @field_validator("missing_year_severity")
    @classmethod
    def validate_severity(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in {"info", "warning", "high", "critical"}:
            raise ValueError(
                "missing_year_severity must be info, warning, high, or critical"
            )
        return normalized

    @field_validator("maximum_explicit_years")
    @classmethod
    def validate_horizon_bounds(cls, value: int, info) -> int:
        minimum = info.data.get("minimum_explicit_years")
        if minimum is not None and value < minimum:
            raise ValueError(
                "maximum_explicit_years cannot be below minimum_explicit_years"
            )
        return value


ValidationConfig = ForecastValidationConfig


__all__ = ["ForecastValidationConfig", "ValidationConfig"]
