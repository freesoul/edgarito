import datetime
from decimal import Decimal

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.assumptions import ValuationAssumptionSet
from edgarito.services.forecasting.models import FcffForecast, FcffForecastObservation
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService
from edgarito.services.valuation.discounting import (
    PresentValueService,
    TerminalValueService,
)
from edgarito.services.valuation.models import (
    CashFlow,
    CashFlowTiming,
    FcffDcfCapitalBridge,
    FcffDcfParameters,
    FcffDcfResult,
    TerminalMetric,
    TerminalValueMethod,
)


class FcffDcfCapitalBridgeResolver:
    """Resolve net debt and diluted shares from one normalized annual period."""

    _CONCEPTS = frozenset(
        {
            FinancialConcept.SHORT_TERM_DEBT,
            FinancialConcept.LONG_TERM_DEBT_CURRENT,
            FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
            FinancialConcept.CASH_AND_EQUIVALENTS,
            FinancialConcept.SHARES_OUTSTANDING,
            FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
        }
    )

    @classmethod
    def required_concepts(cls) -> set[FinancialConcept]:
        return set(cls._CONCEPTS)

    def resolve(
        self,
        financials: NormalizedCompanyFinancials,
        *,
        fiscal_year: int,
        period_end: datetime.date,
        unit: str,
        net_debt: Decimal | None = None,
        gross_debt: Decimal | None = None,
        cash_and_equivalents: Decimal | None = None,
        diluted_shares: Decimal | None = None,
    ) -> FcffDcfCapitalBridge:
        annual = [
            observation
            for observation in financials.observations
            if observation.granularity == Granularity.ANNUAL
            and observation.fiscal_period == FiscalPeriod.FY
            and observation.fiscal_year == fiscal_year
        ]
        by_concept = {}
        for observation in annual:
            by_concept.setdefault(observation.concept, observation)

        component_overrides = (gross_debt, cash_and_equivalents)
        if (component_overrides[0] is None) != (component_overrides[1] is None):
            raise ValueError(
                "FCFF DCF manual gross debt and cash must be provided together"
            )
        if gross_debt is not None and cash_and_equivalents is not None:
            if gross_debt < 0 or cash_and_equivalents < 0:
                raise ValueError("Gross debt and cash overrides cannot be negative")
            derived_net_debt = gross_debt - cash_and_equivalents
            if net_debt is not None and net_debt != derived_net_debt:
                raise ValueError(
                    "Net debt override does not match gross debt minus cash"
                )
            net_debt = derived_net_debt
            net_debt_source = "explicit gross debt minus explicit cash override"
        elif net_debt is None and financials.provider.casefold() == "yahoo":
            gross_debt, cash_and_equivalents, net_debt_source = (
                self._resolve_yahoo_net_debt(by_concept, unit, fiscal_year)
            )
            net_debt = gross_debt - cash_and_equivalents
        elif net_debt is None:
            metrics = FinancialMetricsService().calculate(
                financials,
                granularity=Granularity.ANNUAL,
                metrics={FinancialMetric.GROSS_DEBT, FinancialMetric.NET_DEBT},
            )
            net_debt_metric = next(
                (
                    item
                    for item in metrics.observations
                    if item.metric == FinancialMetric.NET_DEBT
                    and item.fiscal_year == fiscal_year
                    and item.fiscal_period == FiscalPeriod.FY
                ),
                None,
            )
            gross_debt_metric = next(
                (
                    item
                    for item in metrics.observations
                    if item.metric == FinancialMetric.GROSS_DEBT
                    and item.fiscal_year == fiscal_year
                    and item.fiscal_period == FiscalPeriod.FY
                ),
                None,
            )
            cash_observation = by_concept.get(FinancialConcept.CASH_AND_EQUIVALENTS)
            if net_debt_metric is None or gross_debt_metric is None:
                raise ValueError(
                    f"FCFF DCF requires complete debt and cash inputs for FY{fiscal_year}; "
                    "provide valuation.capital_bridge.net_debt in the profile or "
                    "--net-debt"
                )
            if net_debt_metric.unit != unit or gross_debt_metric.unit != unit:
                raise ValueError(
                    "Net debt and forecast cash flows must use one currency"
                )
            if cash_observation is None or cash_observation.unit != unit:
                raise ValueError("Cash and forecast cash flows must use one currency")
            net_debt = net_debt_metric.value
            gross_debt = gross_debt_metric.value
            cash_and_equivalents = cash_observation.value
            net_debt_source = net_debt_metric.formula
        else:
            net_debt_source = "explicit profile or CLI override"

        if diluted_shares is None:
            share_observation = by_concept.get(
                FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES
            ) or by_concept.get(FinancialConcept.SHARES_OUTSTANDING)
            if share_observation is None or share_observation.value <= 0:
                raise ValueError(
                    f"FCFF DCF requires positive diluted shares for FY{fiscal_year}; "
                    "provide valuation.capital_bridge.diluted_shares in the profile "
                    "or --shares"
                )
            diluted_shares = share_observation.value
            shares_source = share_observation.concept.value
        else:
            shares_source = "explicit profile or CLI override"

        return FcffDcfCapitalBridge(
            fiscal_year=fiscal_year,
            period_end=period_end,
            unit=unit,
            net_debt=net_debt,
            diluted_shares=diluted_shares,
            net_debt_source=net_debt_source,
            shares_source=shares_source,
            gross_debt=gross_debt,
            cash_and_equivalents=cash_and_equivalents,
        )

    @staticmethod
    def _resolve_yahoo_net_debt(
        by_concept: dict[FinancialConcept, FinancialObservation],
        unit: str,
        fiscal_year: int,
    ) -> tuple[Decimal, Decimal, str]:
        aggregate_current = by_concept.get(FinancialConcept.SHORT_TERM_DEBT)
        current_portion = by_concept.get(FinancialConcept.LONG_TERM_DEBT_CURRENT)
        noncurrent = by_concept.get(FinancialConcept.LONG_TERM_DEBT_NONCURRENT)
        cash = by_concept.get(FinancialConcept.CASH_AND_EQUIVALENTS)
        current = aggregate_current or current_portion
        debt_observations = [
            observation
            for observation in (current, noncurrent)
            if observation is not None
        ]
        if not debt_observations or cash is None:
            raise ValueError(
                f"FCFF DCF requires Yahoo debt and cash inputs for FY{fiscal_year}; "
                "provide valuation.capital_bridge.net_debt, set gross_debt and "
                "cash_and_equivalents together, or use --net-debt"
            )
        if any(observation.unit != unit for observation in [*debt_observations, cash]):
            raise ValueError("Debt, cash, and forecast must use one currency")
        if any(observation.value < 0 for observation in [*debt_observations, cash]):
            raise ValueError("Reported debt and cash cannot be negative")
        gross_debt = sum(
            (observation.value for observation in debt_observations), Decimal(0)
        )
        debt_sources = " + ".join(
            observation.source_concept for observation in debt_observations
        )
        source = (
            f"Yahoo {debt_sources} - {cash.source_concept}; aggregate CurrentDebt "
            "takes precedence over a separately reported current portion"
        )
        return gross_debt, cash.value, source


class FcffDcfService:
    """Value forecast FCFF and bridge enterprise value to diluted equity value."""

    def value(
        self,
        forecast: FcffForecast,
        parameters: FcffDcfParameters,
        capital_bridge: FcffDcfCapitalBridge,
        assumptions: ValuationAssumptionSet | None = None,
    ) -> FcffDcfResult:
        if not forecast.observations:
            raise ValueError("FCFF DCF requires at least one forecast cash flow")
        if capital_bridge.fiscal_year != forecast.base_fiscal_year:
            raise ValueError("Capital bridge must match the forecast base fiscal year")
        if capital_bridge.unit != forecast.unit:
            raise ValueError("Capital bridge and forecast must use one currency")
        expected_years = list(range(1, len(forecast.observations) + 1))
        if [item.forecast_year for item in forecast.observations] != expected_years:
            raise ValueError("FCFF forecast years must be consecutive and start at one")
        if any(item.unit != forecast.unit for item in forecast.observations):
            raise ValueError("All FCFF forecast observations must use one currency")

        timing_offset = (
            Decimal("0.5")
            if parameters.cash_flow_timing == CashFlowTiming.MID_YEAR
            else Decimal(0)
        )
        explicit_cash_flows = tuple(
            CashFlow(
                amount=item.fcff,
                period=Decimal(item.forecast_year) - timing_offset,
                label=f"FY{item.fiscal_year}E FCFF",
            )
            for item in forecast.observations
        )
        explicit_present_value = PresentValueService.discount(
            explicit_cash_flows,
            parameters.wacc,
            forecast.unit,
        )

        final = forecast.observations[-1]
        if parameters.terminal_method == TerminalValueMethod.PERPETUITY_GROWTH:
            assert parameters.perpetual_growth_rate is not None
            terminal_value = TerminalValueService.perpetuity_growth(
                final.fcff,
                parameters.wacc,
                parameters.perpetual_growth_rate,
            )
        else:
            assert parameters.exit_multiple is not None
            terminal_value = TerminalValueService.exit_multiple(
                self._terminal_metric(final, parameters.exit_metric),
                parameters.exit_multiple,
            )

        terminal_period = Decimal(final.forecast_year)
        terminal_present_values = PresentValueService.discount(
            (
                CashFlow(
                    amount=terminal_value.terminal_value,
                    period=terminal_period,
                    label="Terminal value",
                ),
            ),
            parameters.wacc,
            forecast.unit,
        )
        terminal_present_value = terminal_present_values.cash_flows[0]
        enterprise_value = (
            explicit_present_value.total_present_value
            + terminal_present_value.present_value
        )
        equity_value = enterprise_value - capital_bridge.net_debt
        value_per_share = equity_value / capital_bridge.diluted_shares
        terminal_percentage = (
            terminal_present_value.present_value / enterprise_value * Decimal(100)
            if enterprise_value != 0
            else None
        )
        warnings = []
        if terminal_percentage is not None and terminal_percentage > Decimal(75):
            warnings.append(
                "Discounted terminal value exceeds 75% of enterprise value; "
                "the result is highly sensitive to terminal assumptions"
            )
        if equity_value <= 0:
            warnings.append("Enterprise value does not cover reported net debt")
        if parameters.cash_flow_timing == CashFlowTiming.MID_YEAR:
            warnings.append(
                "Explicit FCFF uses mid-year timing; terminal value remains at "
                "the end of the final forecast year"
            )

        return FcffDcfResult(
            provider=forecast.provider,
            company_id=forecast.company_id,
            company_name=forecast.company_name,
            ticker=forecast.ticker,
            valuation_date=forecast.base_period_end,
            unit=forecast.unit,
            parameters=parameters,
            assumptions=assumptions,
            capital_bridge=capital_bridge,
            explicit_forecast_present_value=explicit_present_value,
            terminal_value=terminal_value,
            terminal_present_value=terminal_present_value,
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            value_per_share=value_per_share,
            terminal_value_percentage=terminal_percentage,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _terminal_metric(
        final: FcffForecastObservation, metric: TerminalMetric
    ) -> Decimal:
        if metric == TerminalMetric.EBITDA:
            return final.operating_income + final.depreciation_and_amortization
        if metric == TerminalMetric.EBIT:
            return final.operating_income
        if metric == TerminalMetric.FCFF:
            return final.fcff
        return final.revenue


__all__ = ["FcffDcfCapitalBridgeResolver", "FcffDcfService"]
