import datetime
from decimal import Decimal

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.assumptions import (
    ValuationAssumptionKind,
    ValuationAssumptionSet,
)
from edgarito.services.forecasting.models import (
    AdaptiveMultistagePlan,
    FcffForecast,
    FcffForecastObservation,
)
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
    ShareRepurchaseParameters,
    ShareRepurchasePeriod,
    ShareRepurchaseResult,
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
            FinancialConcept.SHORT_TERM_INVESTMENTS,
            FinancialConcept.NONCURRENT_INVESTMENTS,
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
        non_operating_assets: Decimal | None = None,
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

        if non_operating_assets is None:
            investment_observations = [
                by_concept[concept]
                for concept in (
                    FinancialConcept.SHORT_TERM_INVESTMENTS,
                    FinancialConcept.NONCURRENT_INVESTMENTS,
                )
                if concept in by_concept
            ]
            if any(item.unit != unit for item in investment_observations):
                raise ValueError(
                    "Non-operating investments and forecast must use one currency"
                )
            non_operating_assets = sum(
                (item.value for item in investment_observations), Decimal(0)
            )
            non_operating_assets_source = (
                " + ".join(item.source_concept for item in investment_observations)
                if investment_observations
                else "none reported"
            )
        else:
            if non_operating_assets < 0:
                raise ValueError("Non-operating assets override cannot be negative")
            non_operating_assets_source = "explicit profile or CLI override"

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
            non_operating_assets=non_operating_assets,
            non_operating_assets_source=non_operating_assets_source,
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

    _TERMINAL_GROWTH_GAP_WARNING = Decimal("1")
    _DAYS_PER_YEAR = Decimal("365")

    def value(
        self,
        forecast: FcffForecast,
        parameters: FcffDcfParameters,
        capital_bridge: FcffDcfCapitalBridge,
        assumptions: ValuationAssumptionSet | None = None,
        multistage_plan: AdaptiveMultistagePlan | None = None,
        valuation_date: datetime.date | None = None,
        share_repurchase_parameters: ShareRepurchaseParameters | None = None,
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

        selected_valuation_date = valuation_date or forecast.base_period_end
        if selected_valuation_date < forecast.base_period_end:
            raise ValueError("Valuation date cannot precede the forecast base date")
        first_period_end = forecast.observations[0].period_end
        if selected_valuation_date >= first_period_end:
            raise ValueError(
                "Valuation date must precede the first forecast period end; update "
                "the financial base period or provide a forecast with a future first "
                "cash-flow date"
            )

        timing_offset = (
            Decimal("0.5")
            if parameters.cash_flow_timing == CashFlowTiming.MID_YEAR
            else Decimal(0)
        )
        use_calendar_periods = valuation_date is not None
        explicit_cash_flows = tuple(
            CashFlow(
                amount=item.fcff,
                period=self._discount_period(
                    item,
                    selected_valuation_date,
                    timing_offset,
                    use_calendar_periods=use_calendar_periods,
                ),
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

        terminal_period = (
            self._year_fraction(selected_valuation_date, final.period_end)
            if use_calendar_periods
            else Decimal(final.forecast_year)
        )
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
        equity_value = (
            enterprise_value
            - capital_bridge.net_debt
            + capital_bridge.non_operating_assets
        )
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
        if parameters.terminal_method == TerminalValueMethod.EXIT_MULTIPLE:
            warnings.append(
                "Exit-multiple terminal value assumes the selected market multiple "
                "persists through the final forecast year; treat it as a "
                "market-relative scenario, not a standalone intrinsic estimate"
            )
        transition_warning = self._terminal_transition_warning(forecast, parameters)
        if transition_warning is not None:
            warnings.append(transition_warning)
        if equity_value <= 0:
            warnings.append("Enterprise value does not cover reported net debt")
        if parameters.cash_flow_timing == CashFlowTiming.MID_YEAR:
            warnings.append(
                "Explicit FCFF uses mid-year timing; terminal value remains at "
                "the end of the final forecast year"
            )
        if capital_bridge.period_end < selected_valuation_date:
            warnings.append(
                f"Capital bridge is dated {capital_bridge.period_end.isoformat()}, "
                f"before the {selected_valuation_date.isoformat()} valuation date; "
                "use current debt, cash, and shares when available"
            )
        share_repurchases = None
        if share_repurchase_parameters is not None:
            share_repurchases = self._model_share_repurchases(
                forecast=forecast,
                parameters=share_repurchase_parameters,
                dcf_parameters=parameters,
                assumptions=assumptions,
                capital_bridge=capital_bridge,
                equity_value=equity_value,
                value_per_share=value_per_share,
                valuation_date=selected_valuation_date,
                use_calendar_periods=use_calendar_periods,
            )
            for repurchase, forecast_observation in zip(
                share_repurchases.periods,
                forecast.observations,
                strict=False,
            ):
                if repurchase.cash_spent > forecast_observation.fcff:
                    warnings.append(
                        f"FY{repurchase.fiscal_year} planned buybacks exceed forecast "
                        "FCFF; execution requires existing cash, borrowing, or other "
                        "funding"
                    )
            if abs(share_repurchases.accretion_percentage) >= Decimal("0.5"):
                direction = (
                    "accretive"
                    if share_repurchases.accretion_percentage > 0
                    else "dilutive"
                )
                warnings.append(
                    f"Modeled buybacks are {direction} to remaining holders because "
                    "the assumed execution-price path differs from the model-implied "
                    "fair-value path"
                )

        return FcffDcfResult(
            provider=forecast.provider,
            company_id=forecast.company_id,
            company_name=forecast.company_name,
            ticker=forecast.ticker,
            valuation_date=selected_valuation_date,
            unit=forecast.unit,
            parameters=parameters,
            assumptions=assumptions,
            multistage_plan=multistage_plan,
            capital_bridge=capital_bridge,
            explicit_forecast_present_value=explicit_present_value,
            terminal_value=terminal_value,
            terminal_present_value=terminal_present_value,
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            value_per_share=value_per_share,
            share_repurchases=share_repurchases,
            terminal_value_percentage=terminal_percentage,
            warnings=tuple(warnings),
        )

    @classmethod
    def _discount_period(
        cls,
        observation: FcffForecastObservation,
        valuation_date: datetime.date,
        timing_offset: Decimal,
        *,
        use_calendar_periods: bool,
    ) -> Decimal:
        if not use_calendar_periods:
            return Decimal(observation.forecast_year) - timing_offset
        period = cls._year_fraction(valuation_date, observation.period_end)
        period -= timing_offset
        if period < 0:
            raise ValueError(
                "Mid-year cash-flow timing falls before the valuation date; update "
                "the financial base period or use end-of-period timing"
            )
        return period

    @classmethod
    def _year_fraction(cls, start: datetime.date, end: datetime.date) -> Decimal:
        return Decimal((end - start).days) / cls._DAYS_PER_YEAR

    @classmethod
    def _terminal_transition_warning(
        cls,
        forecast: FcffForecast,
        parameters: FcffDcfParameters,
    ) -> str | None:
        if (
            parameters.terminal_method != TerminalValueMethod.PERPETUITY_GROWTH
            or parameters.perpetual_growth_rate is None
        ):
            return None
        final = forecast.observations[-1]
        metric = "revenue"
        explicit_growth = final.revenue_growth
        if len(forecast.observations) >= 2:
            previous_fcff = forecast.observations[-2].fcff
            if previous_fcff > 0 and final.fcff > 0:
                metric = "FCFF"
                explicit_growth = (
                    (final.fcff - previous_fcff) / previous_fcff * Decimal(100)
                )
        gap = abs(explicit_growth - parameters.perpetual_growth_rate)
        if gap < cls._TERMINAL_GROWTH_GAP_WARNING:
            return None
        return (
            f"Final explicit {metric} growth ({explicit_growth:,.1f}%) differs from "
            f"perpetual growth ({parameters.perpetual_growth_rate:,.1f}%) by "
            f"{gap:,.1f} percentage points; the terminal transition is abrupt, so "
            "value may be highly sensitive to --years"
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

    @classmethod
    def _model_share_repurchases(
        cls,
        *,
        forecast: FcffForecast,
        parameters: ShareRepurchaseParameters,
        dcf_parameters: FcffDcfParameters,
        assumptions: ValuationAssumptionSet | None,
        capital_bridge: FcffDcfCapitalBridge,
        equity_value: Decimal,
        value_per_share: Decimal,
        valuation_date: datetime.date,
        use_calendar_periods: bool,
    ) -> ShareRepurchaseResult:
        cash_amounts = parameters.annual_cash_amounts
        if len(cash_amounts) > len(forecast.observations):
            raise ValueError(
                "Share-repurchase schedule exceeds the explicit forecast horizon; "
                "increase --years or shorten valuation.share_repurchases."
                "annual_cash_amounts"
            )
        if equity_value <= 0 or value_per_share <= 0:
            raise ValueError(
                "Share-repurchase analysis requires positive pre-repurchase equity "
                "and per-share values"
            )

        discount_rate = parameters.discount_rate
        if discount_rate is not None:
            discount_rate_source = "explicit profile or CLI assumption"
        else:
            equity_cost = (
                assumptions.find(ValuationAssumptionKind.COST_OF_EQUITY)
                if assumptions is not None
                else None
            )
            if equity_cost is not None:
                discount_rate = equity_cost.value
                discount_rate_source = "resolved cost of equity"
            else:
                discount_rate = dcf_parameters.wacc
                discount_rate_source = "WACC fallback"

        price_growth_rate = parameters.price_growth_rate
        if price_growth_rate is None:
            price_growth_rate = discount_rate
        initial_purchase_price = parameters.initial_purchase_price
        if initial_purchase_price is None:
            initial_purchase_price = value_per_share
            purchase_price_source = "model-implied fair value at valuation date"
        else:
            purchase_price_source = "explicit profile or CLI assumption"

        periods = []
        for cash_spent, observation in zip(
            cash_amounts, forecast.observations, strict=False
        ):
            discount_period = (
                cls._year_fraction(valuation_date, observation.period_end)
                if use_calendar_periods
                else Decimal(observation.forecast_year)
            )
            discount_factor = PresentValueService.discount_factor(
                discount_rate, discount_period
            )
            price_growth_factor = Decimal(1) / PresentValueService.discount_factor(
                price_growth_rate, discount_period
            )
            purchase_price = initial_purchase_price * price_growth_factor
            shares_repurchased = cash_spent / purchase_price
            periods.append(
                ShareRepurchasePeriod(
                    forecast_year=observation.forecast_year,
                    fiscal_year=observation.fiscal_year,
                    period_end=observation.period_end,
                    discount_period=discount_period,
                    cash_spent=cash_spent,
                    present_value_cash_spent=cash_spent * discount_factor,
                    purchase_price=purchase_price,
                    shares_repurchased=shares_repurchased,
                )
            )

        total_cash_spent = sum((period.cash_spent for period in periods), Decimal(0))
        present_value_cash_spent = sum(
            (period.present_value_cash_spent for period in periods), Decimal(0)
        )
        shares_repurchased = sum(
            (period.shares_repurchased for period in periods), Decimal(0)
        )
        ending_shares = capital_bridge.diluted_shares - shares_repurchased
        residual_equity_value = equity_value - present_value_cash_spent
        if ending_shares <= 0:
            raise ValueError("Modeled repurchases exceed diluted shares")
        if residual_equity_value <= 0:
            raise ValueError("PV of modeled repurchases exceeds current equity value")
        value_per_remaining_share = residual_equity_value / ending_shares
        accretion_percentage = (
            value_per_remaining_share / value_per_share - Decimal(1)
        ) * Decimal(100)
        return ShareRepurchaseResult(
            source=parameters.source,
            discount_rate=discount_rate,
            discount_rate_source=discount_rate_source,
            price_growth_rate=price_growth_rate,
            initial_purchase_price=initial_purchase_price,
            purchase_price_source=purchase_price_source,
            starting_shares=capital_bridge.diluted_shares,
            ending_shares=ending_shares,
            shares_repurchased=shares_repurchased,
            total_cash_spent=total_cash_spent,
            present_value_cash_spent=present_value_cash_spent,
            pre_repurchase_equity_value=equity_value,
            residual_equity_value=residual_equity_value,
            pre_repurchase_value_per_share=value_per_share,
            value_per_remaining_share=value_per_remaining_share,
            accretion_percentage=accretion_percentage,
            periods=tuple(periods),
        )


__all__ = ["FcffDcfCapitalBridgeResolver", "FcffDcfService"]
