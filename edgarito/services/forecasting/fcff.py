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
    ForecastAssumptionSource,
    ForecastSeedType,
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
                capital_expenditures = (
                    actual.capital_expenditures
                    + remaining_revenue * capex_ratio / PERCENT
                )
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
                capital_expenditures = projected_revenue * capex_ratio / PERCENT
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

        return FcffForecast(
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
        )

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
