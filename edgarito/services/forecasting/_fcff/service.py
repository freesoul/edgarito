"""Orchestration for annual driver-based FCFF forecasting."""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Optional

from edgarito.schemas.forecasting import (
    FcffForecast,
    FcffForecastParameters,
    ForecastSeedType,
    ForecastValue,
)
from edgarito.schemas.guidance.management import MonetaryForecastConstraint
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.services.financials.availability import (
    FinancialObservationAvailabilityService,
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting._fcff import audit, calculation
from edgarito.services.forecasting._fcff.audit import (
    _AUDIT_TOLERANCE,
    _ECONOMIC_AUDIT_FIELDS,
    audit_close,
    build_cell_audits,
    build_legacy_inconsistent_audits,
    build_observation_audits,
    confidence,
    driver_components,
    driver_method,
    driver_source,
    economic_identity_issues,
    incomplete_quarter_warnings,
    revenue_anchor_source,
    revenue_components,
    source_confidence,
    stage_method,
)
from edgarito.services.forecasting._fcff.context import (
    aggregate_quarters,
    annualize_ytd,
    complete_annual_periods,
    complete_quarterly_periods,
    fiscal_year_end,
    forecast_context,
    future_date,
    quarter_index,
    raise_missing_inputs,
)
from edgarito.services.forecasting._fcff.paths import (
    current_run_rate_growth,
    driver_paths,
    effective_tax_rate,
    expand_path,
    historical_fcff,
    historical_values,
)
from edgarito.services.metrics.calculator import (
    OPERATING_WORKING_CAPITAL_CONCEPTS,
)


class FcffForecastService:
    """Forecast annual unlevered FCFF from explicit operating drivers."""

    _ECONOMIC_AUDIT_FIELDS = _ECONOMIC_AUDIT_FIELDS
    _AUDIT_TOLERANCE = _AUDIT_TOLERANCE
    _CORE_REQUIRED_CONCEPTS = frozenset(
        {
            FinancialConcept.REVENUE,
            FinancialConcept.OPERATING_INCOME,
            FinancialConcept.PRETAX_INCOME,
            FinancialConcept.INCOME_TAX_EXPENSE,
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
            FinancialConcept.CAPITAL_EXPENDITURES,
        }
    )
    _REQUIRED_CONCEPTS = _CORE_REQUIRED_CONCEPTS | OPERATING_WORKING_CAPITAL_CONCEPTS

    def __init__(
        self,
        availability_service: FinancialObservationAvailabilityService | None = None,
    ) -> None:
        self._availability_service = (
            availability_service or FinancialObservationAvailabilityService()
        )

    @classmethod
    def required_concepts(cls) -> set[FinancialConcept]:
        return set(cls._REQUIRED_CONCEPTS)

    def build_cell_audits(
        self, forecast: FcffForecast
    ) -> tuple[dict[str, ForecastValue], ...]:
        return build_cell_audits(self, forecast)

    def regenerate_cell_audits(self, forecast: FcffForecast) -> FcffForecast:
        audits = self.build_cell_audits(forecast)
        forecast.observations = [
            observation.model_copy(update={"cell_audits": cell_audits})
            for observation, cell_audits in zip(
                forecast.observations, audits, strict=True
            )
        ]
        return forecast

    def economic_identity_issues(self, forecast: FcffForecast) -> tuple[str, ...]:
        return economic_identity_issues(forecast)

    def build_legacy_inconsistent_audits(
        self,
        forecast: FcffForecast,
        issues: tuple[str, ...],
    ) -> tuple[dict[str, ForecastValue], ...]:
        return build_legacy_inconsistent_audits(forecast, issues)

    @classmethod
    def _audit_close(cls, actual: Decimal, expected: Decimal) -> bool:
        return audit_close(actual, expected)

    def forecast(
        self,
        financials: NormalizedCompanyFinancials,
        parameters: Optional[FcffForecastParameters] = None,
        *,
        as_of: datetime.date | None = None,
        availability_mode: ObservationAvailabilityMode = (
            ObservationAvailabilityMode.POINT_IN_TIME
        ),
    ) -> FcffForecast:
        parameters = parameters or FcffForecastParameters()
        availability_mode = ObservationAvailabilityMode(availability_mode)
        if as_of is not None:
            financials = financials.model_copy(
                update={
                    "observations": [
                        item
                        for item in financials.observations
                        if self._availability_service.is_available(
                            item,
                            as_of=as_of,
                            mode=availability_mode,
                            snapshot_retrieved_at=financials.retrieved_at,
                        )
                    ]
                }
            )
        periods = self._complete_annual_periods(financials)
        if not periods:
            self._raise_missing_inputs(financials)
        context_value = self._forecast_context(
            financials, periods, parameters, as_of
        )
        forecast = calculation.build_forecast(
            self,
            financials,
            parameters,
            context_value,
            periods,
            availability_mode,
        )
        return self.regenerate_cell_audits(forecast)

    @staticmethod
    def _capex_constraint_for(
        parameters: FcffForecastParameters, fiscal_year: int
    ) -> MonetaryForecastConstraint | None:
        return parameters.capex_constraints.get(fiscal_year)

    @staticmethod
    def _apply_capex_constraint(
        provisional_capex: Decimal, constraint: MonetaryForecastConstraint
    ) -> Decimal:
        if (
            constraint.point is not None
            and constraint.minimum is None
            and constraint.maximum is None
        ):
            return constraint.point
        constrained = (
            constraint.point if constraint.point is not None else provisional_capex
        )
        if constraint.minimum is not None:
            constrained = max(constrained, constraint.minimum)
        if constraint.maximum is not None:
            constrained = min(constrained, constraint.maximum)
        return constrained

    def _build_observation_audits(
        self,
        forecast: FcffForecast,
        index: int,
        observation,
    ) -> dict[str, ForecastValue]:
        return build_observation_audits(self, forecast, index, observation)

    @classmethod
    def _driver_source(cls, forecast, driver, index) -> str:
        return driver_source(forecast, driver, index)

    @classmethod
    def _driver_components(cls, forecast, driver, index):
        return driver_components(forecast, driver, index)

    @classmethod
    def _revenue_components(cls, forecast, index):
        return revenue_components(forecast, index)

    @staticmethod
    def _revenue_anchor_source(forecast, fiscal_year):
        return revenue_anchor_source(forecast, fiscal_year)

    @staticmethod
    def _driver_method(source, driver_label, stage=None):
        return driver_method(source, driver_label, stage)

    @staticmethod
    def _stage_method(method, stage):
        return stage_method(method, stage)

    @classmethod
    def _audit_value(cls, value, sources, method, *, derived=False):
        return audit.audit_value(value, sources, method, derived=derived)

    @staticmethod
    def _confidence(sources):
        return confidence(sources)

    @staticmethod
    def _source_confidence(source):
        return source_confidence(source)

    @classmethod
    def _incomplete_quarter_warnings(cls, financials, selected_seed_end):
        return incomplete_quarter_warnings(
            cls._CORE_REQUIRED_CONCEPTS,
            financials,
            selected_seed_end,
        )

    def _forecast_context(self, financials, annual_periods, parameters, as_of):
        return forecast_context(self, financials, annual_periods, parameters, as_of)

    @staticmethod
    def _forecast_seed_type(name: str) -> ForecastSeedType:
        return {
            "fiscal_year": ForecastSeedType.FISCAL_YEAR,
            "ttm": ForecastSeedType.TTM,
            "ytd_plus_forecast": ForecastSeedType.YTD_PLUS_FORECAST,
            "ytd_run_rate": ForecastSeedType.YTD_RUN_RATE,
        }[name]

    @staticmethod
    def _annualize_ytd(actual, quarter_count, fiscal_end):
        return annualize_ytd(actual, quarter_count, fiscal_end)

    @classmethod
    def _complete_quarterly_periods(cls, financials):
        return complete_quarterly_periods(financials, cls._CORE_REQUIRED_CONCEPTS)

    @staticmethod
    def _aggregate_quarters(periods):
        return aggregate_quarters(periods)

    @staticmethod
    def _quarter_index(period):
        return quarter_index(period)

    @staticmethod
    def _fiscal_year_end(fiscal_year, annual_end):
        return fiscal_year_end(fiscal_year, annual_end)

    def _driver_paths(
        self,
        parameters,
        historical_periods,
        *,
        fallback_revenue_growth=None,
    ):
        return driver_paths(
            self,
            parameters,
            historical_periods,
            fallback_revenue_growth=fallback_revenue_growth,
        )

    @staticmethod
    def _current_run_rate_growth(context):
        return current_run_rate_growth(context)

    def _historical_values(self, driver, periods):
        return historical_values(self, driver, periods)

    @staticmethod
    def _effective_tax_rate(period):
        return effective_tax_rate(period)

    @staticmethod
    def _historical_fcff(current, previous, nopat):
        return historical_fcff(current, previous, nopat)

    @classmethod
    def _complete_annual_periods(cls, financials):
        return complete_annual_periods(financials, cls._CORE_REQUIRED_CONCEPTS)

    @classmethod
    def _raise_missing_inputs(cls, financials):
        return raise_missing_inputs(financials, cls._CORE_REQUIRED_CONCEPTS)

    @staticmethod
    def _expand_path(path, years):
        return expand_path(path, years)

    @staticmethod
    def _future_date(base_date, years):
        return future_date(base_date, years)


__all__ = ["FcffForecastService"]
