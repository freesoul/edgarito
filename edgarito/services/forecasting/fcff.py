import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.financial_observation_availability import (
    FinancialObservationAvailabilityService,
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting.models import (
    FcffForecast,
    FcffForecastDriver,
    FcffForecastObservation,
    FcffForecastParameters,
    FcffForecastYtdAnchor,
    ForecastAssumptionSource,
    ForecastSeedType,
    ForecastValue,
    MonetaryForecastConstraint,
)
from edgarito.services.metrics.calculator import (
    OPERATING_WORKING_CAPITAL_CONCEPTS,
    operating_working_capital_value,
)

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


class FcffForecastService:
    """Forecast annual unlevered FCFF from explicit operating drivers."""

    _ECONOMIC_AUDIT_FIELDS = (
        "revenue_growth",
        "revenue",
        "operating_margin",
        "operating_income",
        "tax_rate",
        "nopat",
        "depreciation_and_amortization",
        "capital_expenditures",
        "change_in_operating_working_capital",
        "fcff",
    )
    _AUDIT_TOLERANCE = Decimal("1e-18")

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
        """Build audit metadata from the forecast's current drivers and cells.

        Keeping this derivation separate from forecast arithmetic lets adaptive
        forecasting refresh provenance after it changes driver sources or extends
        the projection horizon.
        """

        return tuple(
            self._build_observation_audits(forecast, index, observation)
            for index, observation in enumerate(forecast.observations)
        )

    def regenerate_cell_audits(self, forecast: FcffForecast) -> FcffForecast:
        """Refresh every observation's economic-cell provenance in place."""

        audits = self.build_cell_audits(forecast)
        forecast.observations = [
            observation.model_copy(update={"cell_audits": cell_audits})
            for observation, cell_audits in zip(
                forecast.observations, audits, strict=True
            )
        ]
        return forecast

    def economic_identity_issues(self, forecast: FcffForecast) -> tuple[str, ...]:
        """Return material FCFF identity mismatches in supplied observations."""

        issues: list[str] = []
        previous_working_capital = forecast.base_operating_working_capital
        for observation in forecast.observations:
            expected = {
                "operating_income": (
                    observation.revenue * observation.operating_margin / PERCENT
                ),
                "nopat": observation.operating_income
                * (Decimal(1) - observation.tax_rate / PERCENT),
                "depreciation_and_amortization": (
                    observation.revenue * observation.depreciation_to_revenue / PERCENT
                ),
                "capital_expenditures": (
                    observation.revenue * observation.capex_to_revenue / PERCENT
                ),
                "operating_working_capital": (
                    observation.revenue
                    * observation.operating_working_capital_to_revenue
                    / PERCENT
                ),
                "change_in_operating_working_capital": (
                    observation.operating_working_capital - previous_working_capital
                ),
                "fcff": (
                    observation.nopat
                    + observation.depreciation_and_amortization
                    - observation.capital_expenditures
                    - observation.change_in_operating_working_capital
                ),
            }
            for field, expected_value in expected.items():
                actual_value = getattr(observation, field)
                if not self._audit_close(actual_value, expected_value):
                    issues.append(
                        f"FY{observation.fiscal_year}E {field}: expected "
                        f"{expected_value}, got {actual_value}"
                    )
            previous_working_capital = observation.operating_working_capital
        return tuple(issues)

    def build_legacy_inconsistent_audits(
        self,
        forecast: FcffForecast,
        issues: tuple[str, ...],
    ) -> tuple[dict[str, ForecastValue], ...]:
        """Represent legacy inconsistent observations without changing arithmetic."""

        method = "legacy/inconsistent economic identities: " + "; ".join(issues)
        return tuple(
            {
                field: ForecastValue(
                    value=getattr(observation, field),
                    source="unknown/legacy/inconsistent",
                    method=method,
                    confidence="low",
                )
                for field in self._ECONOMIC_AUDIT_FIELDS
            }
            for observation in forecast.observations
        )

    @classmethod
    def _audit_close(cls, actual: Decimal, expected: Decimal) -> bool:
        scale = max(abs(actual), abs(expected), Decimal(1))
        return abs(actual - expected) <= scale * cls._AUDIT_TOLERANCE

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

        context = self._forecast_context(financials, periods, parameters, as_of)
        historical_periods = periods[-parameters.historical_window :]
        paths, sources = self._driver_paths(parameters, list(context.path_periods))
        base = context.base
        previous = (
            periods[-2]
            if len(periods) > 1 and context.seed_type == ForecastSeedType.FISCAL_YEAR
            else None
        )
        base_tax_rate = self._effective_tax_rate(base)
        if base_tax_rate is None:
            base_tax_rate = paths[FcffForecastDriver.TAX_RATE][0]
        base_nopat = base.operating_income * (Decimal(1) - base_tax_rate / PERCENT)
        base_fcff = self._historical_fcff(base, previous, base_nopat)

        projected_revenue = base.revenue
        previous_working_capital = base.operating_working_capital
        observations = []
        capex_constraints_applied: list[int] = []
        ytd_capex_ratio = None
        first_fiscal_year = (
            context.current_fiscal_year
            if context.current_fiscal_year is not None
            else base.fiscal_year + 1
        )
        for index in range(parameters.forecast_years):
            fiscal_year = first_fiscal_year + index
            growth = paths[FcffForecastDriver.REVENUE_GROWTH][index]
            operating_margin = paths[FcffForecastDriver.OPERATING_MARGIN][index]
            tax_rate = paths[FcffForecastDriver.TAX_RATE][index]
            depreciation_ratio = paths[FcffForecastDriver.DEPRECIATION_TO_REVENUE][
                index
            ]
            capex_ratio = paths[FcffForecastDriver.CAPEX_TO_REVENUE][index]
            working_capital_ratio = paths[
                FcffForecastDriver.OPERATING_WORKING_CAPITAL_TO_REVENUE
            ][index]

            if index == 0 and context.actual_ytd is not None:
                actual = context.actual_ytd
                revenue_anchor = parameters.revenue_anchors.get(fiscal_year)
                if revenue_anchor is not None and revenue_anchor < actual.revenue:
                    raise ValueError(
                        f"FY{fiscal_year} revenue anchor is below reported YTD revenue"
                    )
                projected_revenue = revenue_anchor or max(
                    actual.revenue,
                    context.latest_annual.revenue * (Decimal(1) + growth / PERCENT),
                )
                remaining_revenue = projected_revenue - actual.revenue
                operating_income = (
                    actual.operating_income
                    + remaining_revenue * operating_margin / PERCENT
                )
                actual_tax_rate = self._effective_tax_rate(actual)
                actual_nopat = actual.operating_income * (
                    Decimal(1)
                    - (actual_tax_rate if actual_tax_rate is not None else tax_rate)
                    / PERCENT
                )
                projected_nopat = (
                    remaining_revenue
                    * operating_margin
                    / PERCENT
                    * (Decimal(1) - tax_rate / PERCENT)
                )
                nopat = actual_nopat + projected_nopat
                depreciation = (
                    actual.depreciation_and_amortization
                    + remaining_revenue * depreciation_ratio / PERCENT
                )
                provisional_capex = (
                    actual.capital_expenditures
                    + remaining_revenue * capex_ratio / PERCENT
                )
                capital_expenditures = provisional_capex
                ytd_capex_ratio = capex_ratio
                operating_margin = operating_income / projected_revenue * PERCENT
                tax_rate = (
                    (Decimal(1) - nopat / operating_income) * PERCENT
                    if operating_income > 0
                    else tax_rate
                )
                depreciation_ratio = depreciation / projected_revenue * PERCENT
                capex_ratio = capital_expenditures / projected_revenue * PERCENT
                growth = (
                    projected_revenue / context.latest_annual.revenue - Decimal(1)
                ) * PERCENT
            else:
                previous_revenue = projected_revenue
                projected_revenue = parameters.revenue_anchors.get(
                    fiscal_year
                ) or projected_revenue * (Decimal(1) + growth / PERCENT)
                if fiscal_year in parameters.revenue_anchors:
                    growth = (
                        projected_revenue / previous_revenue - Decimal(1)
                    ) * PERCENT
                operating_income = projected_revenue * operating_margin / PERCENT
                nopat = operating_income * (Decimal(1) - tax_rate / PERCENT)
                depreciation = projected_revenue * depreciation_ratio / PERCENT
                provisional_capex = projected_revenue * capex_ratio / PERCENT
                capital_expenditures = provisional_capex

            capex_constraint = self._capex_constraint_for(parameters, fiscal_year)
            if capex_constraint is not None:
                constrained_capex = self._apply_capex_constraint(
                    provisional_capex, capex_constraint
                )
                if constrained_capex != provisional_capex:
                    capex_constraints_applied.append(fiscal_year)
                capital_expenditures = constrained_capex
                capex_ratio = capital_expenditures / projected_revenue * PERCENT
                if index == 0 and context.actual_ytd is not None:
                    if remaining_revenue != 0:
                        ytd_capex_ratio = (
                            (
                                capital_expenditures
                                - context.actual_ytd.capital_expenditures
                            )
                            / remaining_revenue
                            * PERCENT
                        )
                    else:
                        ytd_capex_ratio = capex_ratio
            operating_working_capital = (
                projected_revenue * working_capital_ratio / PERCENT
            )
            change_in_working_capital = (
                operating_working_capital - previous_working_capital
            )
            fcff = (
                nopat + depreciation - capital_expenditures - change_in_working_capital
            )
            forecast_year = index + 1
            first_period_end = context.fiscal_year_end or self._future_date(
                context.latest_annual.period_end, 1
            )
            observations.append(
                FcffForecastObservation(
                    forecast_year=forecast_year,
                    fiscal_year=fiscal_year,
                    period_end=self._future_date(first_period_end, index),
                    revenue_growth=growth,
                    revenue=projected_revenue,
                    operating_margin=operating_margin,
                    operating_income=operating_income,
                    tax_rate=tax_rate,
                    nopat=nopat,
                    depreciation_to_revenue=depreciation_ratio,
                    depreciation_and_amortization=depreciation,
                    capex_to_revenue=capex_ratio,
                    capital_expenditures=capital_expenditures,
                    operating_working_capital_to_revenue=working_capital_ratio,
                    operating_working_capital=operating_working_capital,
                    change_in_operating_working_capital=change_in_working_capital,
                    fcff=fcff,
                    unit=base.unit,
                )
            )
            previous_working_capital = operating_working_capital

        ytd_anchor_capex_ratio = (
            ytd_capex_ratio
            if context.actual_ytd is not None and ytd_capex_ratio is not None
            else paths[FcffForecastDriver.CAPEX_TO_REVENUE][0]
        )
        ytd_anchor = None
        if (
            context.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
            and context.actual_ytd is not None
        ):
            actual_ytd = context.actual_ytd
            ytd_anchor = FcffForecastYtdAnchor(
                fiscal_year=first_fiscal_year,
                ytd_period_end=actual_ytd.period_end,
                fiscal_year_end=context.fiscal_year_end or actual_ytd.period_end,
                actual_quarters=context.actual_quarters,
                actual_revenue=actual_ytd.revenue,
                actual_operating_income=actual_ytd.operating_income,
                actual_pretax_income=actual_ytd.pretax_income,
                actual_income_tax_expense=actual_ytd.income_tax_expense,
                actual_tax_rate=self._effective_tax_rate(actual_ytd),
                actual_depreciation_and_amortization=(
                    actual_ytd.depreciation_and_amortization
                ),
                actual_capital_expenditures=actual_ytd.capital_expenditures,
                actual_operating_working_capital=actual_ytd.operating_working_capital,
                latest_annual_revenue=context.latest_annual.revenue,
                revenue_anchor=parameters.revenue_anchors.get(first_fiscal_year),
                revenue_growth=paths[FcffForecastDriver.REVENUE_GROWTH][0],
                operating_margin=paths[FcffForecastDriver.OPERATING_MARGIN][0],
                tax_rate=paths[FcffForecastDriver.TAX_RATE][0],
                depreciation_to_revenue=paths[
                    FcffForecastDriver.DEPRECIATION_TO_REVENUE
                ][0],
                capex_to_revenue=ytd_anchor_capex_ratio,
                operating_working_capital_to_revenue=paths[
                    FcffForecastDriver.OPERATING_WORKING_CAPITAL_TO_REVENUE
                ][0],
            )

        forecast_fiscal_years = {
            first_fiscal_year + index for index in range(parameters.forecast_years)
        }
        for year in forecast_fiscal_years:
            constraint = self._capex_constraint_for(parameters, year)
            if constraint is not None and constraint.source == (
                ForecastAssumptionSource.MANAGEMENT_GUIDANCE.value
            ):
                sources[FcffForecastDriver.CAPEX_TO_REVENUE] = (
                    ForecastAssumptionSource.MANAGEMENT_GUIDANCE
                )
                break

        forecast = FcffForecast(
            provider=financials.provider,
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker,
            identifiers=financials.identifiers,
            seed_type=context.seed_type,
            seed_methodology=context.seed_methodology,
            seed_period_end=context.seed_period_end,
            current_fiscal_year=context.current_fiscal_year,
            actual_quarters=context.actual_quarters,
            financial_snapshot_retrieved_at=financials.retrieved_at,
            availability_mode=availability_mode.value,
            base_fiscal_year=base.fiscal_year,
            base_period_end=context.seed_period_end,
            base_revenue=base.revenue,
            base_operating_income=base.operating_income,
            base_tax_rate=base_tax_rate,
            base_nopat=base_nopat,
            base_depreciation_and_amortization=base.depreciation_and_amortization,
            base_capital_expenditures=base.capital_expenditures,
            base_operating_working_capital=base.operating_working_capital,
            base_fcff=base_fcff,
            unit=base.unit,
            parameters=parameters,
            historical_fiscal_years=tuple(
                period.fiscal_year for period in historical_periods
            ),
            assumption_sources=sources,
            observations=observations,
            warnings=self._incomplete_quarter_warnings(
                financials, context.seed_period_end
            ),
            ytd_anchor=ytd_anchor,
            capex_constraints_applied=tuple(capex_constraints_applied),
        )
        return self.regenerate_cell_audits(forecast)

    @staticmethod
    def _capex_constraint_for(
        parameters: FcffForecastParameters, fiscal_year: int
    ) -> MonetaryForecastConstraint | None:
        if (
            parameters.capex_to_revenue is not None
            and parameters.assumption_source_overrides.get(
                FcffForecastDriver.CAPEX_TO_REVENUE
            )
            != ForecastAssumptionSource.MANAGEMENT_GUIDANCE
        ):
            return None
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
        constrained = provisional_capex
        if constraint.minimum is not None:
            constrained = max(constrained, constraint.minimum)
        if constraint.maximum is not None:
            constrained = min(constrained, constraint.maximum)
        return constrained

    def _build_observation_audits(
        self,
        forecast: FcffForecast,
        index: int,
        observation: FcffForecastObservation,
    ) -> dict[str, ForecastValue]:
        growth_source = self._driver_source(
            forecast, FcffForecastDriver.REVENUE_GROWTH, index
        )
        margin_source = self._driver_source(
            forecast, FcffForecastDriver.OPERATING_MARGIN, index
        )
        tax_source = self._driver_source(forecast, FcffForecastDriver.TAX_RATE, index)
        is_ytd_seed = (
            forecast.ytd_anchor is not None
            and observation.forecast_year == 1
            and forecast.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
        )
        is_revenue_anchor = (
            observation.fiscal_year in forecast.parameters.revenue_anchors
        )
        anchor_source = (
            self._revenue_anchor_source(forecast, observation.fiscal_year)
            if is_revenue_anchor
            else None
        )
        anchor_method = (
            f"{anchor_source.replace('_', ' ')} revenue anchor"
            if anchor_source is not None
            else "explicit revenue anchor"
        )
        revenue_components = self._revenue_components(forecast, index)
        if forecast.ytd_anchor is not None:
            revenue_components = (
                *revenue_components,
                ("projected_remainder", "prior_forecast"),
            )
        growth_components = (
            (
                *revenue_components,
                ("projected_remainder", "prior_forecast"),
            )
            if is_ytd_seed
            else (
                (
                    *self._revenue_components(forecast, index - 1),
                    (
                        "revenue_anchor",
                        self._revenue_anchor_source(forecast, observation.fiscal_year),
                    ),
                )
                if is_revenue_anchor
                else (("revenue_growth", growth_source),)
            )
        )
        stage = (
            forecast.adaptive_stages[index]
            if index < len(forecast.adaptive_stages)
            else None
        )
        effective_margin_components = (
            (
                ("ytd_actual_operating_income", "reported"),
                *revenue_components,
                ("projected_remainder", "prior_forecast"),
                ("operating_margin", margin_source),
            )
            if is_ytd_seed
            else (("operating_margin", margin_source),)
        )
        effective_tax_components = (
            (
                ("ytd_actual_nopat", "reported"),
                ("ytd_actual_tax", "reported"),
                *revenue_components,
                ("projected_remainder", "prior_forecast"),
                ("operating_margin", margin_source),
                ("tax_rate", tax_source),
            )
            if is_ytd_seed
            else (("tax_rate", tax_source),)
        )
        depreciation_components = self._driver_components(
            forecast, FcffForecastDriver.DEPRECIATION_TO_REVENUE, index
        )
        capex_components = self._driver_components(
            forecast, FcffForecastDriver.CAPEX_TO_REVENUE, index
        )
        working_capital_components = self._driver_components(
            forecast,
            FcffForecastDriver.OPERATING_WORKING_CAPITAL_TO_REVENUE,
            index,
        )
        capex_constraint = self._capex_constraint_for(
            forecast.parameters, observation.fiscal_year
        )
        capex_constraint_applied = (
            capex_constraint is not None
            and observation.fiscal_year in forecast.capex_constraints_applied
        )
        capex_constraint_components = (
            (
                (
                    f"capex_constraint_fy{observation.fiscal_year}",
                    capex_constraint.source,
                ),
            )
            if capex_constraint_applied
            else ()
        )
        capex_constraint_prefix = (
            (
                f"{capex_constraint.source} {capex_constraint.methodology} capex "
                "constraint; "
            )
            if capex_constraint_applied
            else ""
        )
        capex_method = capex_constraint_prefix + (
            "actual YTD capex + remaining revenue × capex-to-revenue / 100"
            if is_ytd_seed
            else "revenue × capex-to-revenue / 100"
        )
        fcff_method = capex_constraint_prefix + (
            "NOPAT + depreciation and amortization - capital expenditures - "
            "change in operating working capital"
        )
        prior_working_capital = "historical_seed" if index == 0 else "prior_forecast"
        ytd_projection_components = (
            (("projected_remainder", "prior_forecast"),) if is_ytd_seed else ()
        )
        ytd_depreciation_components = (
            (("ytd_actual_depreciation", "reported"),) if is_ytd_seed else ()
        )
        ytd_capex_components = (
            (("ytd_actual_capex", "reported"),) if is_ytd_seed else ()
        )

        return {
            "revenue_growth": self._audit_value(
                observation.revenue_growth,
                growth_components,
                (
                    self._stage_method(
                        "effective growth from reported YTD revenue and "
                        "projected remainder",
                        stage,
                    )
                    if is_ytd_seed
                    else self._stage_method(
                        f"effective growth from {anchor_method} and prior revenue",
                        stage,
                    )
                    if is_revenue_anchor
                    else self._driver_method(growth_source, "revenue growth", stage)
                ),
                derived=is_revenue_anchor or is_ytd_seed,
            ),
            "revenue": self._audit_value(
                observation.revenue,
                revenue_components,
                self._stage_method(
                    (
                        "actual YTD revenue plus explicit revenue anchor"
                        if is_ytd_seed and is_revenue_anchor
                        else anchor_method
                        if is_revenue_anchor
                        else (
                            "actual YTD revenue + forecast remaining revenue"
                            if is_ytd_seed
                            else (
                                "seed revenue × (1 + revenue growth / 100)"
                                if observation.forecast_year == 1
                                else "prior revenue × (1 + revenue growth / 100)"
                            )
                        )
                    ),
                    stage,
                ),
                derived=True,
            ),
            "operating_margin": self._audit_value(
                observation.operating_margin,
                effective_margin_components,
                (
                    "blended YTD actual and remaining forecast operating margin"
                    if is_ytd_seed
                    else self._driver_method(margin_source, "operating margin", stage)
                ),
                derived=is_ytd_seed,
            ),
            "operating_income": self._audit_value(
                observation.operating_income,
                (*revenue_components, *effective_margin_components),
                self._stage_method(
                    (
                        "actual YTD operating income + remaining revenue × operating "
                        "margin / 100"
                        if is_ytd_seed
                        else "revenue × operating margin / 100"
                    ),
                    stage,
                ),
                derived=True,
            ),
            "tax_rate": self._audit_value(
                observation.tax_rate,
                effective_tax_components,
                (
                    "blended YTD actual and remaining forecast tax rate"
                    if is_ytd_seed
                    else self._driver_method(tax_source, "tax rate", stage)
                ),
                derived=is_ytd_seed,
            ),
            "nopat": self._audit_value(
                observation.nopat,
                (
                    *revenue_components,
                    *effective_margin_components,
                    *effective_tax_components,
                ),
                self._stage_method(
                    (
                        "actual YTD NOPAT + remaining-period NOPAT"
                        if is_ytd_seed
                        else "operating income × (1 - tax rate / 100)"
                    ),
                    stage,
                ),
                derived=True,
            ),
            "depreciation_and_amortization": self._audit_value(
                observation.depreciation_and_amortization,
                (
                    *revenue_components,
                    *ytd_depreciation_components,
                    *ytd_projection_components,
                    *depreciation_components,
                ),
                self._stage_method(
                    (
                        "actual YTD D&A + remaining revenue × depreciation-to-revenue / 100"
                        if is_ytd_seed
                        else "revenue × depreciation-to-revenue / 100"
                    ),
                    stage,
                ),
                derived=True,
            ),
            "capital_expenditures": self._audit_value(
                observation.capital_expenditures,
                (
                    *revenue_components,
                    *ytd_capex_components,
                    *ytd_projection_components,
                    *capex_components,
                    *capex_constraint_components,
                ),
                self._stage_method(capex_method, stage),
                derived=True,
            ),
            "change_in_operating_working_capital": self._audit_value(
                observation.change_in_operating_working_capital,
                (
                    *revenue_components,
                    *ytd_projection_components,
                    *working_capital_components,
                    ("prior_working_capital", prior_working_capital),
                ),
                self._stage_method(
                    "operating working capital - prior operating working capital",
                    stage,
                ),
                derived=True,
            ),
            "fcff": self._audit_value(
                observation.fcff,
                (
                    *revenue_components,
                    *effective_margin_components,
                    *effective_tax_components,
                    *ytd_projection_components,
                    *depreciation_components,
                    *capex_components,
                    *capex_constraint_components,
                    *working_capital_components,
                    ("prior_working_capital", prior_working_capital),
                ),
                self._stage_method(fcff_method, stage),
                derived=True,
            ),
        }

    @classmethod
    def _driver_source(
        cls,
        forecast: FcffForecast,
        driver: FcffForecastDriver,
        index: int,
    ) -> str:
        path = forecast.assumption_source_paths.get(driver)
        source = path[index] if path is not None and index < len(path) else None
        if source is None:
            source = forecast.assumption_sources.get(driver)
        return source.value if source is not None else "unknown/legacy"

    @classmethod
    def _driver_components(
        cls,
        forecast: FcffForecast,
        driver: FcffForecastDriver,
        index: int,
    ) -> tuple[tuple[str, str], ...]:
        components = []
        for source_index in range(index + 1):
            source = cls._driver_source(forecast, driver, source_index)
            label = driver.value
            if source_index < index:
                label += f"_fy{forecast.observations[source_index].fiscal_year}"
            components.append((label, source))
        return tuple(components)

    @classmethod
    def _revenue_components(
        cls, forecast: FcffForecast, index: int
    ) -> tuple[tuple[str, str], ...]:
        components: list[tuple[str, str]] = [("seed", "historical_seed")]
        if forecast.ytd_anchor is not None:
            components.append(("ytd_actual", "reported"))
        for source_index in range(index + 1):
            fiscal_year = forecast.observations[source_index].fiscal_year
            if fiscal_year in forecast.parameters.revenue_anchors:
                label = (
                    "revenue_anchor"
                    if source_index == index
                    else f"prior_revenue_anchor_fy{fiscal_year}"
                )
                components.append(
                    (
                        label,
                        cls._revenue_anchor_source(forecast, fiscal_year),
                    )
                )
            else:
                label = (
                    "revenue_growth"
                    if source_index == index
                    else f"prior_revenue_growth_fy{fiscal_year}"
                )
                components.append(
                    (
                        label,
                        cls._driver_source(
                            forecast, FcffForecastDriver.REVENUE_GROWTH, source_index
                        ),
                    )
                )
        return tuple(components)

    @staticmethod
    def _revenue_anchor_source(forecast: FcffForecast, fiscal_year: int) -> str:
        source = forecast.parameters.revenue_anchor_sources.get(
            fiscal_year, ForecastAssumptionSource.EXPLICIT
        )
        return source.value

    @staticmethod
    def _driver_method(source: str, driver_label: str, stage: str | None = None) -> str:
        method_by_source = {
            ForecastAssumptionSource.EXPLICIT.value: "explicit forecast driver",
            ForecastAssumptionSource.MANAGEMENT_GUIDANCE.value: (
                "management guidance forecast driver"
            ),
            ForecastAssumptionSource.TRAILING_AVERAGE.value: (
                "trailing historical average forecast driver"
            ),
            ForecastAssumptionSource.ADAPTIVE_MULTISTAGE.value: (
                "adaptive multistage forecast driver path"
            ),
        }
        stage_label = f" ({stage})" if stage else ""
        return (
            f"{driver_label}{stage_label}: "
            f"{method_by_source.get(source, 'legacy/unknown driver')}"
        )

    @staticmethod
    def _stage_method(method: str, stage: str | None) -> str:
        if stage in {"high_growth", "transition", "stable"}:
            return f"{stage} stage: {method}"
        return method

    @classmethod
    def _audit_value(
        cls,
        value: Decimal,
        sources: tuple[tuple[str, str], ...],
        method: str,
        *,
        derived: bool = False,
    ) -> ForecastValue:
        unique_sources = tuple(dict.fromkeys(sources))
        source = (
            "derived["
            + ",".join(f"{name}={source}" for name, source in unique_sources)
            + "]"
            if derived
            else " + ".join(source for _, source in unique_sources)
        )
        return ForecastValue(
            value=value,
            source=source,
            method=method,
            confidence=cls._confidence(tuple(source for _, source in unique_sources)),
        )

    @staticmethod
    def _confidence(sources: tuple[str, ...]) -> str:
        if not sources or any(source == "unknown/legacy" for source in sources):
            return "low"
        if all(
            FcffForecastService._source_confidence(source) == "high"
            for source in sources
        ):
            return "high"
        if all(
            FcffForecastService._source_confidence(source) in {"high", "medium"}
            for source in sources
        ):
            return "medium"
        return "low"

    @staticmethod
    def _source_confidence(source: str) -> str:
        if source in {
            ForecastAssumptionSource.EXPLICIT.value,
            ForecastAssumptionSource.MANAGEMENT_GUIDANCE.value,
            "historical_seed",
            "reported",
            "prior_forecast",
        }:
            return "high"
        if source in {
            ForecastAssumptionSource.TRAILING_AVERAGE.value,
            ForecastAssumptionSource.ADAPTIVE_MULTISTAGE.value,
        }:
            return "medium"
        return "low"

    @classmethod
    def _incomplete_quarter_warnings(
        cls,
        financials: NormalizedCompanyFinancials,
        selected_seed_end: datetime.date,
    ) -> tuple[str, ...]:
        by_period: dict[
            tuple[int, FiscalPeriod], dict[FinancialConcept, FinancialObservation]
        ] = {}
        for item in financials.observations:
            if (
                item.granularity == Granularity.QUARTERLY
                and item.fiscal_period
                in {
                    FiscalPeriod.Q1,
                    FiscalPeriod.Q2,
                    FiscalPeriod.Q3,
                    FiscalPeriod.Q4,
                }
                and item.period_end > selected_seed_end
            ):
                by_period.setdefault(item.period_key, {}).setdefault(item.concept, item)
        candidates = [
            values
            for values in by_period.values()
            if FinancialConcept.REVENUE in values
        ]
        if not candidates:
            return ()
        values = max(
            candidates,
            key=lambda items: items[FinancialConcept.REVENUE].period_end,
        )
        revenue = values[FinancialConcept.REVENUE]
        missing = sorted(
            cls._CORE_REQUIRED_CONCEPTS - values.keys(),
            key=lambda item: item.value,
        )
        details = [concept.label for concept in missing]
        if operating_working_capital_value(values) is None:
            details.append("Operating Working Capital Components")
        if not details:
            details.append("a coherent single-currency operating dataset")
        return (
            f"FY{revenue.fiscal_year} {revenue.fiscal_period.value} ending "
            f"{revenue.period_end.isoformat()} is incomplete in the "
            f"{financials.provider.upper()} snapshot; forecast seed falls back to "
            f"{selected_seed_end.isoformat()} because "
            f"{', '.join(details)} are unavailable",
        )

    def _forecast_context(self, financials, annual_periods, parameters, as_of):
        latest_annual = annual_periods[-1]
        quarterly = self._complete_quarterly_periods(financials)
        newer = [
            item for item in quarterly if item.period_end > latest_annual.period_end
        ]
        if not newer:
            selected = tuple(annual_periods[-parameters.historical_window :])
            return _ForecastContext(
                base=latest_annual,
                latest_annual=latest_annual,
                path_periods=selected,
                seed_type=ForecastSeedType.FISCAL_YEAR,
                seed_methodology=(
                    f"Latest complete FY{latest_annual.fiscal_year}; no newer complete "
                    "quarterly operating context was available"
                ),
                seed_period_end=latest_annual.period_end,
            )

        latest = newer[-1]
        current = [item for item in newer if item.fiscal_year == latest.fiscal_year]
        current.sort(key=lambda item: item.period_end)
        latest_four = quarterly[-4:]
        has_ltm = len(latest_four) == 4 and all(
            self._quarter_index(right) == self._quarter_index(left) + 1
            for left, right in zip(latest_four, latest_four[1:], strict=False)
        )
        ltm = self._aggregate_quarters(latest_four) if has_ltm else None
        path_periods = list(annual_periods[-parameters.historical_window :])
        if ltm is not None:
            path_periods.append(
                _HistoricalDrivers(
                    **{
                        **ltm.__dict__,
                        "fiscal_year": latest_annual.fiscal_year + 1,
                    }
                )
            )

        if latest.fiscal_period != FiscalPeriod.Q4:
            actual_ytd = self._aggregate_quarters(current)
            fiscal_end = self._fiscal_year_end(
                latest.fiscal_year, latest_annual.period_end
            )
            if as_of is not None and fiscal_end <= as_of:
                next_fiscal_end = self._future_date(fiscal_end, 1)
                if ltm is not None:
                    return _ForecastContext(
                        base=_HistoricalDrivers(
                            **{**ltm.__dict__, "fiscal_year": latest.fiscal_year}
                        ),
                        latest_annual=latest_annual,
                        path_periods=tuple(path_periods),
                        seed_type=ForecastSeedType.TTM,
                        seed_methodology=(
                            f"The FY{latest.fiscal_year} year ended on "
                            f"{fiscal_end.isoformat()} but final-period reporting was "
                            "not yet available; four consecutive reported quarters "
                            "form the current run-rate and the first forecast is the "
                            "next unelapsed fiscal year"
                        ),
                        seed_period_end=latest.period_end,
                        current_fiscal_year=latest.fiscal_year + 1,
                        actual_quarters=4,
                        fiscal_year_end=next_fiscal_end,
                    )

                run_rate = self._annualize_ytd(actual_ytd, len(current), fiscal_end)
                path_periods.append(run_rate)
                return _ForecastContext(
                    base=run_rate,
                    latest_annual=latest_annual,
                    path_periods=tuple(path_periods),
                    seed_type=ForecastSeedType.YTD_RUN_RATE,
                    seed_methodology=(
                        f"The FY{latest.fiscal_year} year ended on "
                        f"{fiscal_end.isoformat()} before final-period reporting was "
                        f"available; {len(current)} reported quarter(s) were annualized "
                        "as a low-confidence run-rate and the first forecast is the "
                        "next unelapsed fiscal year"
                    ),
                    seed_period_end=latest.period_end,
                    current_fiscal_year=latest.fiscal_year + 1,
                    actual_quarters=len(current),
                    fiscal_year_end=next_fiscal_end,
                )
            base = (
                _HistoricalDrivers(
                    **{**ltm.__dict__, "fiscal_year": latest_annual.fiscal_year}
                )
                if ltm is not None
                else latest_annual
            )
            return _ForecastContext(
                base=base,
                latest_annual=latest_annual,
                path_periods=tuple(path_periods),
                seed_type=ForecastSeedType.YTD_PLUS_FORECAST,
                seed_methodology=(
                    f"FY{latest.fiscal_year} estimate uses {len(current)} actual fiscal "
                    f"quarter(s) through {latest.period_end.isoformat()} plus a driver-"
                    "based forecast of the remaining period"
                    + (
                        "; latest-four-quarter metrics seed normalization"
                        if has_ltm
                        else ""
                    )
                ),
                seed_period_end=latest.period_end,
                current_fiscal_year=latest.fiscal_year,
                actual_ytd=actual_ytd,
                actual_quarters=len(current),
                fiscal_year_end=fiscal_end,
            )

        assert ltm is not None
        return _ForecastContext(
            base=ltm,
            latest_annual=latest_annual,
            path_periods=tuple(path_periods),
            seed_type=ForecastSeedType.TTM,
            seed_methodology=(
                "Four consecutive fiscal quarters form a current run-rate; the TTM "
                "is not inserted into completed annual history"
            ),
            seed_period_end=latest.period_end,
            current_fiscal_year=latest.fiscal_year + 1,
            actual_quarters=4,
            fiscal_year_end=self._fiscal_year_end(
                latest.fiscal_year + 1, latest_annual.period_end
            ),
        )

    @staticmethod
    def _annualize_ytd(actual, quarter_count, fiscal_end):
        scale = Decimal(4) / Decimal(quarter_count)
        return _HistoricalDrivers(
            fiscal_year=actual.fiscal_year,
            period_end=fiscal_end,
            unit=actual.unit,
            revenue=actual.revenue * scale,
            operating_income=actual.operating_income * scale,
            pretax_income=actual.pretax_income * scale,
            income_tax_expense=actual.income_tax_expense * scale,
            depreciation_and_amortization=(
                actual.depreciation_and_amortization * scale
            ),
            capital_expenditures=actual.capital_expenditures * scale,
            # Working capital is a point-in-time balance, not an additive flow.
            operating_working_capital=actual.operating_working_capital,
        )

    @classmethod
    def _complete_quarterly_periods(cls, financials):
        by_period: dict[tuple[int, FiscalPeriod], dict] = {}
        for item in financials.observations:
            if item.granularity == Granularity.QUARTERLY and item.fiscal_period in {
                FiscalPeriod.Q1,
                FiscalPeriod.Q2,
                FiscalPeriod.Q3,
                FiscalPeriod.Q4,
            }:
                by_period.setdefault(item.period_key, {}).setdefault(item.concept, item)
        periods = []
        for (fiscal_year, _fiscal_period), values in by_period.items():
            if not cls._CORE_REQUIRED_CONCEPTS <= values.keys():
                continue
            owc = operating_working_capital_value(values)
            if owc is None:
                continue
            units = {
                values[concept].unit for concept in cls._CORE_REQUIRED_CONCEPTS
            } | {owc.unit}
            revenue = values[FinancialConcept.REVENUE]
            if len(units) != 1 or revenue.value <= 0:
                continue
            periods.append(
                _HistoricalDrivers(
                    fiscal_year=fiscal_year,
                    period_end=revenue.period_end,
                    unit=revenue.unit,
                    revenue=revenue.value,
                    operating_income=values[FinancialConcept.OPERATING_INCOME].value,
                    pretax_income=values[FinancialConcept.PRETAX_INCOME].value,
                    income_tax_expense=values[
                        FinancialConcept.INCOME_TAX_EXPENSE
                    ].value,
                    depreciation_and_amortization=values[
                        FinancialConcept.DEPRECIATION_AND_AMORTIZATION
                    ].value,
                    capital_expenditures=abs(
                        values[FinancialConcept.CAPITAL_EXPENDITURES].value
                    ),
                    operating_working_capital=owc.value,
                    fiscal_period=_fiscal_period,
                )
            )
        return sorted(periods, key=lambda item: item.period_end)

    @staticmethod
    def _aggregate_quarters(periods):
        latest = periods[-1]
        return _HistoricalDrivers(
            fiscal_year=latest.fiscal_year,
            period_end=latest.period_end,
            unit=latest.unit,
            revenue=sum((item.revenue for item in periods), Decimal(0)),
            operating_income=sum(
                (item.operating_income for item in periods), Decimal(0)
            ),
            pretax_income=sum((item.pretax_income for item in periods), Decimal(0)),
            income_tax_expense=sum(
                (item.income_tax_expense for item in periods), Decimal(0)
            ),
            depreciation_and_amortization=sum(
                (item.depreciation_and_amortization for item in periods), Decimal(0)
            ),
            capital_expenditures=sum(
                (item.capital_expenditures for item in periods), Decimal(0)
            ),
            operating_working_capital=latest.operating_working_capital,
            fiscal_period=latest.fiscal_period,
        )

    @staticmethod
    def _quarter_index(period):
        quarter = {
            FiscalPeriod.Q1: 0,
            FiscalPeriod.Q2: 1,
            FiscalPeriod.Q3: 2,
            FiscalPeriod.Q4: 3,
        }[period.fiscal_period]
        return period.fiscal_year * 4 + quarter

    @staticmethod
    def _fiscal_year_end(fiscal_year, annual_end):
        try:
            return annual_end.replace(year=fiscal_year)
        except ValueError:
            return annual_end.replace(year=fiscal_year, day=28)

    def _driver_paths(
        self,
        parameters: FcffForecastParameters,
        historical_periods: list[_HistoricalDrivers],
    ) -> tuple[
        dict[FcffForecastDriver, tuple[Decimal, ...]],
        dict[FcffForecastDriver, ForecastAssumptionSource],
    ]:
        paths = {}
        sources = {}
        for driver in FcffForecastDriver:
            explicit_path = getattr(parameters, driver.value)
            if explicit_path is not None:
                paths[driver] = self._expand_path(
                    explicit_path, parameters.forecast_years
                )
                sources[driver] = ForecastAssumptionSource.EXPLICIT
                continue

            historical_values = self._historical_values(driver, historical_periods)
            if not historical_values:
                option = driver.value.replace("_", "-")
                required_history = (
                    "complete, consecutive annual periods"
                    if driver == FcffForecastDriver.REVENUE_GROWTH
                    else "complete annual periods"
                )
                raise ValueError(
                    f"{driver.label} could not be inferred from {required_history}; "
                    f"provide --{option}"
                )
            average = sum(historical_values, Decimal(0)) / len(historical_values)
            paths[driver] = (average,) * parameters.forecast_years
            sources[driver] = ForecastAssumptionSource.TRAILING_AVERAGE
        sources.update(parameters.assumption_source_overrides)
        return paths, sources

    def _historical_values(
        self,
        driver: FcffForecastDriver,
        periods: list[_HistoricalDrivers],
    ) -> list[Decimal]:
        if driver == FcffForecastDriver.REVENUE_GROWTH:
            return [
                (current.revenue - previous.revenue) / previous.revenue * PERCENT
                for previous, current in zip(periods, periods[1:], strict=False)
                if current.fiscal_year == previous.fiscal_year + 1
                and previous.revenue != 0
            ]

        values = []
        for period in periods:
            if period.revenue == 0:
                continue
            if driver == FcffForecastDriver.OPERATING_MARGIN:
                numerator = period.operating_income
            elif driver == FcffForecastDriver.TAX_RATE:
                tax_rate = self._effective_tax_rate(period)
                if tax_rate is not None:
                    values.append(tax_rate)
                continue
            elif driver == FcffForecastDriver.DEPRECIATION_TO_REVENUE:
                numerator = period.depreciation_and_amortization
            elif driver == FcffForecastDriver.CAPEX_TO_REVENUE:
                numerator = period.capital_expenditures
            else:
                numerator = period.operating_working_capital
            values.append(numerator / period.revenue * PERCENT)
        return values

    @classmethod
    def _complete_annual_periods(
        cls, financials: NormalizedCompanyFinancials
    ) -> list[_HistoricalDrivers]:
        by_year: dict[int, dict[FinancialConcept, FinancialObservation]] = {}
        for observation in financials.observations:
            if (
                observation.granularity == Granularity.ANNUAL
                and observation.fiscal_period == FiscalPeriod.FY
            ):
                by_year.setdefault(observation.fiscal_year, {}).setdefault(
                    observation.concept, observation
                )

        periods = []
        for fiscal_year, values in sorted(by_year.items()):
            if not cls._CORE_REQUIRED_CONCEPTS <= values.keys():
                continue
            operating_working_capital = operating_working_capital_value(values)
            if operating_working_capital is None:
                continue
            units = {
                values[concept].unit for concept in cls._CORE_REQUIRED_CONCEPTS
            } | {operating_working_capital.unit}
            if len(units) != 1:
                continue
            revenue = values[FinancialConcept.REVENUE]
            if revenue.value <= 0:
                continue
            periods.append(
                _HistoricalDrivers(
                    fiscal_year=fiscal_year,
                    period_end=revenue.period_end,
                    unit=revenue.unit,
                    revenue=revenue.value,
                    operating_income=values[FinancialConcept.OPERATING_INCOME].value,
                    pretax_income=values[FinancialConcept.PRETAX_INCOME].value,
                    income_tax_expense=values[
                        FinancialConcept.INCOME_TAX_EXPENSE
                    ].value,
                    depreciation_and_amortization=values[
                        FinancialConcept.DEPRECIATION_AND_AMORTIZATION
                    ].value,
                    capital_expenditures=abs(
                        values[FinancialConcept.CAPITAL_EXPENDITURES].value
                    ),
                    operating_working_capital=operating_working_capital.value,
                )
            )
        return periods

    @staticmethod
    def _effective_tax_rate(period: _HistoricalDrivers) -> Optional[Decimal]:
        if period.pretax_income <= 0 or period.income_tax_expense < 0:
            return None
        rate = period.income_tax_expense / period.pretax_income * PERCENT
        return rate if Decimal(0) <= rate <= PERCENT else None

    @staticmethod
    def _historical_fcff(
        current: _HistoricalDrivers,
        previous: Optional[_HistoricalDrivers],
        nopat: Decimal,
    ) -> Optional[Decimal]:
        if previous is None or current.fiscal_year != previous.fiscal_year + 1:
            return None
        change_in_working_capital = (
            current.operating_working_capital - previous.operating_working_capital
        )
        return (
            nopat
            + current.depreciation_and_amortization
            - current.capital_expenditures
            - change_in_working_capital
        )

    @classmethod
    def _raise_missing_inputs(cls, financials: NormalizedCompanyFinancials) -> None:
        annual = [
            observation
            for observation in financials.observations
            if observation.granularity == Granularity.ANNUAL
            and observation.fiscal_period == FiscalPeriod.FY
        ]
        latest_year = max(
            (observation.fiscal_year for observation in annual), default=None
        )
        present = {
            observation.concept
            for observation in annual
            if observation.fiscal_year == latest_year
        }
        missing = sorted(
            cls._CORE_REQUIRED_CONCEPTS - present, key=lambda item: item.value
        )
        details = [concept.value for concept in missing]
        latest_values = {
            observation.concept: observation
            for observation in annual
            if observation.fiscal_year == latest_year
        }
        if operating_working_capital_value(latest_values) is None:
            has_detailed_assets = {
                FinancialConcept.ACCOUNTS_RECEIVABLE,
                FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS,
            } <= present
            has_aggregate_assets = {
                FinancialConcept.CURRENT_ASSETS,
                FinancialConcept.CASH_AND_EQUIVALENTS,
            } <= present
            has_detailed_liabilities = {
                FinancialConcept.ACCOUNTS_PAYABLE,
                FinancialConcept.ACCRUED_LIABILITIES,
                FinancialConcept.DEFERRED_REVENUE_CURRENT,
            } <= present
            has_current_liabilities = FinancialConcept.CURRENT_LIABILITIES in present
            if not has_detailed_assets and not has_aggregate_assets:
                details.append(
                    "working-capital assets (receivables and prepaid/other current "
                    "assets, or total current assets and cash)"
                )
            if not has_current_liabilities and not (
                has_detailed_assets and has_detailed_liabilities
            ):
                details.append(
                    "working-capital liabilities (accrued/payables/deferred revenue "
                    "with detailed assets, or total current liabilities)"
                )
        detail = ", ".join(details)
        suffix = f" Missing for FY{latest_year}: {detail}." if details else ""
        raise ValueError(
            "Driver-based FCFF forecasting requires complete annual revenue, "
            "operating income, pretax income, tax, D&A, capex, and operating "
            f"working-capital components in one currency.{suffix}"
        )

    @staticmethod
    def _expand_path(path: tuple[Decimal, ...], years: int) -> tuple[Decimal, ...]:
        if len(path) == 1:
            return path * years
        return (*path, *((path[-1],) * (years - len(path))))

    @staticmethod
    def _future_date(base_date: datetime.date, years: int) -> datetime.date:
        try:
            return base_date.replace(year=base_date.year + years)
        except ValueError:
            return base_date.replace(year=base_date.year + years, day=28)


# Preserve the old generic service import while changing its semantics to FCFF.
FreeCashFlowForecastService = FcffForecastService


__all__ = ["FcffForecastService", "FreeCashFlowForecastService"]
