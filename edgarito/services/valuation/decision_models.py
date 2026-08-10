from __future__ import annotations

import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DecisionScenario(str, Enum):
    BEAR = "bear"
    BASE = "base"
    BULL = "bull"


class RelativeScenarioTimeBasis(str, Enum):
    PRESENT_DAY = "present_day"
    TARGET_DATE = "target_date"


class ValuationAssessmentBand(str, Enum):
    STRONGLY_CHEAP = "strongly cheap"
    CHEAP = "cheap"
    FAIR = "fairly valued"
    EXPENSIVE = "expensive"
    STRONGLY_EXPENSIVE = "strongly expensive"


class ReverseDcfVariable(str, Enum):
    REVENUE_GROWTH = "revenue_growth"
    OPERATING_MARGIN = "operating_margin"
    TERMINAL_ROIC = "terminal_roic"
    TERMINAL_GROWTH = "terminal_growth"
    WACC = "wacc"


class ReverseDcfStatus(str, Enum):
    SOLVED = "solved"
    NO_SOLUTION = "no_solution"


class _DecisionModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ScenarioAssumption(_DecisionModel):
    name: str
    value: Decimal
    base_value: Decimal
    unit: str = "percent"
    changed: bool
    source: str
    methodology: str


class IntrinsicScenarioCase(_DecisionModel):
    scenario: DecisionScenario
    value_per_share: Decimal | None = None
    assumptions: tuple[ScenarioAssumption, ...]
    methodology: str
    available: bool = True
    invalid_reason: str | None = None
    warnings: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def infer_unavailable_state(cls, values):
        """Allow invalid cases to be represented without a fake valuation.

        A missing value is the important part of the public contract: an
        unavailable scenario must not carry Base (or any other scenario's)
        number as a placeholder.  Inferring ``available=False`` when callers
        provide an invalid reason keeps construction backwards-compatible for
        integrations that only know about the new reason field.
        """

        if isinstance(values, dict) and (
            values.get("value_per_share") is None
            or values.get("invalid_reason") is not None
        ):
            values = dict(values)
            values.setdefault("available", False)
        return values

    @model_validator(mode="after")
    def validate_availability(self):
        if self.available:
            if self.value_per_share is None:
                raise ValueError(
                    "Available intrinsic scenarios require a value per share"
                )
            if self.invalid_reason is not None:
                raise ValueError(
                    "Available intrinsic scenarios cannot have an invalid reason"
                )
        elif self.value_per_share is not None:
            raise ValueError(
                "Unavailable intrinsic scenarios cannot publish a value per share"
            )
        elif not self.invalid_reason:
            raise ValueError(
                "Unavailable intrinsic scenarios require a clear invalid reason"
            )
        return self

    @property
    def is_available(self) -> bool:
        """Compatibility alias for consumers using predicate-style naming."""

        return self.available


class RelativeScenarioCase(_DecisionModel):
    scenario: DecisionScenario
    value_per_share: Decimal
    multiple: Decimal = Field(gt=0)
    methodology: str
    time_basis: RelativeScenarioTimeBasis = RelativeScenarioTimeBasis.PRESENT_DAY
    target_date: datetime.date | None = None
    horizon_years: Decimal | None = Field(default=None, gt=0)
    horizon_upside_downside: Decimal | None = None


class SensitivityCell(_DecisionModel):
    row_value: Decimal
    column_value: Decimal
    value_per_share: Decimal | None = None
    invalid_reason: str | None = None

    @model_validator(mode="after")
    def validate_value_or_reason(self):
        if (self.value_per_share is None) == (self.invalid_reason is None):
            raise ValueError(
                "Sensitivity cells require either a value or an invalid reason"
            )
        return self


class SensitivityTable(_DecisionModel):
    name: str
    row_label: str
    column_label: str
    row_values: tuple[Decimal, ...]
    column_values: tuple[Decimal, ...]
    cells: tuple[tuple[SensitivityCell, ...], ...]
    methodology: str

    @model_validator(mode="after")
    def validate_shape(self):
        if len(self.cells) != len(self.row_values):
            raise ValueError("Sensitivity row count does not match row values")
        if any(len(row) != len(self.column_values) for row in self.cells):
            raise ValueError("Sensitivity column count does not match column values")
        return self


class PriceComparison(_DecisionModel):
    label: str
    model: str
    value_per_share: Decimal
    upside_downside: Decimal
    margin_of_safety: Decimal | None


class ValuationAssessment(_DecisionModel):
    intrinsic: ValuationAssessmentBand
    relative: ValuationAssessmentBand | None = None
    overall: str
    model_dispersion: str | None = None
    rationale: tuple[str, ...] = ()


class ReverseDcfSolution(_DecisionModel):
    variable: ReverseDcfVariable
    status: ReverseDcfStatus
    base_value: Decimal
    implied_value: Decimal | None = None
    lower_bound: Decimal
    upper_bound: Decimal
    achieved_price: Decimal | None = None
    target_price: Decimal
    unit: str = "percent"
    methodology: str
    explanation: str


class DecisionValuationResult(_DecisionModel):
    ticker: str | None = None
    company_name: str
    currency: str
    current_price: Decimal = Field(gt=0)
    intrinsic_scenarios: tuple[IntrinsicScenarioCase, ...]
    relative_scenarios: tuple[RelativeScenarioCase, ...] = ()
    sensitivity_tables: tuple[SensitivityTable, ...] = ()
    price_comparisons: tuple[PriceComparison, ...]
    assessment: ValuationAssessment
    reverse_dcf: tuple[ReverseDcfSolution, ...] = ()
    methodology: str
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_scenarios(self):
        intrinsic_names = tuple(item.scenario for item in self.intrinsic_scenarios)
        expected = (
            DecisionScenario.BEAR,
            DecisionScenario.BASE,
            DecisionScenario.BULL,
        )
        if intrinsic_names != expected:
            raise ValueError("Intrinsic scenarios must be ordered bear, base, bull")
        base = self.intrinsic_scenarios[1]
        if not base.available or base.value_per_share is None:
            raise ValueError("The independently calculated Base scenario is required")
        if all(item.available for item in self.intrinsic_scenarios):
            bear, _, bull = self.intrinsic_scenarios
            assert bear.value_per_share is not None
            assert bull.value_per_share is not None
            if not bear.value_per_share < base.value_per_share < bull.value_per_share:
                raise ValueError(
                    "Available intrinsic scenarios must satisfy strict "
                    "Bear < Base < Bull ordering"
                )
        return self


__all__ = [
    "DecisionScenario",
    "DecisionValuationResult",
    "IntrinsicScenarioCase",
    "PriceComparison",
    "RelativeScenarioCase",
    "RelativeScenarioTimeBasis",
    "ReverseDcfSolution",
    "ReverseDcfStatus",
    "ReverseDcfVariable",
    "ScenarioAssumption",
    "SensitivityCell",
    "SensitivityTable",
    "ValuationAssessment",
    "ValuationAssessmentBand",
]
