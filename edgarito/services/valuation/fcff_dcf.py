import datetime
from decimal import Decimal

from edgarito.enums.granularity import Granularity
from edgarito.schemas.forecasting import (
    AdaptiveMultistagePlan,
    FcffForecast,
    FcffForecastDcfStub,
    FcffForecastObservation,
    ForecastSeedType,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.assumptions import (
    ValuationAssumptionKind,
    ValuationAssumptionSet,
)
from edgarito.services.financials.availability import (
    FinancialObservationAvailabilityService,
    ObservationAvailabilityMode,
)
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
    FcffDcfShareDilutionSensitivity,
    ShareCountBasis,
    ShareRepurchaseParameters,
    ShareRepurchasePeriod,
    ShareRepurchaseResult,
    TerminalMetric,
    TerminalValueMethod,
)

_CAPITAL_BRIDGE_STALE_DAYS = 180
_DILUTION_SENSITIVITY_PERCENTAGES = (Decimal("5"), Decimal("10"), Decimal("20"))
_CURRENT_SHARES_LIMITATION_WARNING = (
    "Current shares outstanding are used as the valuation denominator; this is not "
    "a fully diluted share count and excludes potential dilution from options, "
    "RSUs, and other claims"
)
_USER_SUPPLIED_SHARES_WARNING = (
    "A user-supplied share count is used as provided; its fully diluted status is "
    "not verified and options, RSUs, and other claims may be excluded"
)


class FcffDcfCapitalBridgeResolver:
    """Resolve enterprise-to-equity inputs from the latest coherent period."""

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

    def __init__(
        self,
        availability_service: FinancialObservationAvailabilityService | None = None,
    ) -> None:
        self._availability_service = (
            availability_service or FinancialObservationAvailabilityService()
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
        valuation_date: datetime.date | None = None,
        availability_mode: ObservationAvailabilityMode = (
            ObservationAvailabilityMode.POINT_IN_TIME
        ),
    ) -> FcffDcfCapitalBridge:
        as_of = valuation_date or period_end
        availability_mode = ObservationAvailabilityMode(availability_mode)
        eligible = [
            item
            for item in financials.observations
            if item.concept in self._CONCEPTS
            and self._availability_service.is_available(
                item,
                as_of=as_of,
                mode=availability_mode,
                snapshot_retrieved_at=financials.retrieved_at,
            )
        ]
        by_date: dict[datetime.date, dict[FinancialConcept, FinancialObservation]] = {}
        for item in eligible:
            values = by_date.setdefault(item.period_end, {})
            existing = values.get(item.concept)
            if existing is None or self._observation_preference_key(item) > (
                self._observation_preference_key(existing)
            ):
                values[item.concept] = item
        bridge_date = self._latest_coherent_balance_date(by_date) or period_end
        by_concept = by_date.get(bridge_date, {})
        bridge_fiscal_year = next(
            (item.fiscal_year for item in by_concept.values()), fiscal_year
        )
        warnings: list[str] = []

        component_overrides = (gross_debt, cash_and_equivalents)
        debt_is_explicit = net_debt is not None or gross_debt is not None
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
                self._resolve_yahoo_net_debt(by_concept, unit, bridge_fiscal_year)
            )
            net_debt = gross_debt - cash_and_equivalents
        elif net_debt is None:
            gross_debt, cash_and_equivalents, net_debt_source = (
                self._resolve_reported_net_debt(by_concept, unit, bridge_fiscal_year)
            )
            net_debt = gross_debt - cash_and_equivalents
        else:
            net_debt_source = "explicit profile or CLI override"

        share_observation = None
        share_count_basis = ShareCountBasis.UNKNOWN
        current_shares_outstanding = None
        weighted_average_diluted_shares = None
        if diluted_shares is None:
            share_observation = self._latest_share_observation(eligible)
            if share_observation is None or share_observation.value <= 0:
                raise ValueError(
                    f"FCFF DCF requires a positive share count by {as_of}; "
                    "provide valuation.capital_bridge.diluted_shares in the profile "
                    "or --shares"
                )
            diluted_shares = share_observation.value
            if share_observation.concept == FinancialConcept.SHARES_OUTSTANDING:
                share_count_basis = ShareCountBasis.CURRENT_SHARES_OUTSTANDING
                current_shares_outstanding = share_observation.value
                shares_source = (
                    f"{share_observation.source_concept} current shares outstanding; "
                    "not a fully diluted share count"
                )
                warnings.append(_CURRENT_SHARES_LIMITATION_WARNING)
            else:
                share_count_basis = ShareCountBasis.WEIGHTED_AVERAGE_DILUTED
                weighted_average_diluted_shares = share_observation.value
                shares_source = (
                    f"{share_observation.source_concept}; weighted-average diluted "
                    "shares fallback because a current shares-outstanding balance "
                    "was unavailable"
                )
        else:
            share_count_basis = ShareCountBasis.USER_SUPPLIED
            shares_source = (
                "user-supplied share count from explicit profile or CLI override; "
                "fully diluted status not verified"
            )
            warnings.append(_USER_SUPPLIED_SHARES_WARNING)

        if (
            share_observation is not None
            and abs((share_observation.period_end - bridge_date).days)
            > _CAPITAL_BRIDGE_STALE_DAYS
        ):
            warnings.append(
                "Share count and debt/cash are from materially different dates: "
                f"{share_observation.period_end.isoformat()} vs {bridge_date.isoformat()}"
            )
        if (as_of - bridge_date).days > _CAPITAL_BRIDGE_STALE_DAYS:
            warnings.append(
                f"Capital-bridge data are {(as_of - bridge_date).days} days old; "
                "current quarterly values were unavailable"
            )

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
            fiscal_year=bridge_fiscal_year,
            period_end=bridge_date,
            unit=unit,
            net_debt=net_debt,
            diluted_shares=diluted_shares,
            share_count_basis=share_count_basis,
            current_shares_outstanding=current_shares_outstanding,
            weighted_average_diluted_shares=weighted_average_diluted_shares,
            net_debt_source=net_debt_source,
            shares_source=shares_source,
            gross_debt=gross_debt,
            cash_and_equivalents=cash_and_equivalents,
            non_operating_assets=non_operating_assets,
            non_operating_assets_source=non_operating_assets_source,
            debt_date=(
                bridge_date if gross_debt is not None and not debt_is_explicit else None
            ),
            cash_date=(
                bridge_date
                if cash_and_equivalents is not None and not debt_is_explicit
                else None
            ),
            shares_date=(share_observation.period_end if share_observation else None),
            non_operating_assets_date=(
                bridge_date
                if non_operating_assets_source
                not in {
                    "none reported",
                    "explicit profile or CLI override",
                }
                else None
            ),
            warnings=tuple(warnings),
        )

    @classmethod
    def _latest_coherent_balance_date(cls, by_date):
        for granularity in (Granularity.QUARTERLY, Granularity.ANNUAL):
            coherent_dates = [
                selected_on
                for selected_on, values in by_date.items()
                if cls._has_coherent_balance(values, granularity)
            ]
            if coherent_dates:
                return max(coherent_dates)
        return max(by_date, default=None)

    @staticmethod
    def _has_coherent_balance(
        values: dict[FinancialConcept, FinancialObservation],
        granularity: Granularity,
    ) -> bool:
        cash = values.get(FinancialConcept.CASH_AND_EQUIVALENTS)
        debt = any(
            values.get(concept) is not None
            for concept in (
                FinancialConcept.SHORT_TERM_DEBT,
                FinancialConcept.LONG_TERM_DEBT_CURRENT,
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
            )
        )
        return (
            cash is not None
            and debt
            and cash.granularity == granularity
            and any(
                values[concept].granularity == granularity
                for concept in (
                    FinancialConcept.SHORT_TERM_DEBT,
                    FinancialConcept.LONG_TERM_DEBT_CURRENT,
                    FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
                )
                if concept in values
            )
        )

    @staticmethod
    def _observation_preference_key(
        observation: FinancialObservation,
    ) -> tuple[bool, datetime.date, str, bool, str, Decimal]:
        """Rank duplicate observations without depending on input order."""
        return (
            observation.granularity == Granularity.QUARTERLY,
            observation.filed or datetime.date.min,
            observation.accession_number or "",
            bool(observation.form and observation.form.endswith("/A")),
            observation.source_concept,
            observation.value,
        )

    @classmethod
    def _latest_observation(cls, observations):
        return max(
            observations,
            key=lambda item: (item.period_end, *cls._observation_preference_key(item)),
            default=None,
        )

    @classmethod
    def _latest_share_observation(cls, eligible):
        shares = [
            item
            for item in eligible
            if item.concept
            in {
                FinancialConcept.SHARES_OUTSTANDING,
                FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
            }
            and item.value > 0
        ]
        if not shares:
            return None
        current = [
            item
            for item in shares
            if item.concept == FinancialConcept.SHARES_OUTSTANDING
        ]
        quarterly_current = [
            item for item in current if item.granularity == Granularity.QUARTERLY
        ]
        if quarterly_current:
            return cls._latest_observation(quarterly_current)
        if current:
            return cls._latest_observation(current)

        quarterly_weighted = [
            item for item in shares if item.granularity == Granularity.QUARTERLY
        ]
        return cls._latest_observation(quarterly_weighted or shares)

    @classmethod
    def _resolve_reported_net_debt(cls, by_concept, unit, fiscal_year):
        debt_observations = [
            by_concept[concept]
            for concept in (
                FinancialConcept.SHORT_TERM_DEBT,
                FinancialConcept.LONG_TERM_DEBT_CURRENT,
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
            )
            if concept in by_concept
        ]
        cash = by_concept.get(FinancialConcept.CASH_AND_EQUIVALENTS)
        if not debt_observations or cash is None:
            raise ValueError(
                f"FCFF DCF requires complete debt and cash inputs for FY{fiscal_year}; "
                "provide valuation.capital_bridge.net_debt in the profile or --net-debt"
            )
        if any(item.unit != unit for item in [*debt_observations, cash]):
            raise ValueError("Debt, cash, and forecast must use one currency")
        gross_debt = sum((item.value for item in debt_observations), Decimal(0))
        return gross_debt, cash.value, "gross debt - cash and equivalents"

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
        dcf_stub = self._validate_dcf_stub(
            forecast,
            capital_bridge,
            selected_valuation_date,
        )
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
        # A YTD forecast is based at the latest reported balance-sheet date,
        # so its stub and all following periods must use calendar fractions
        # even when the caller leaves valuation_date at its default base date.
        use_calendar_periods = valuation_date is not None or dcf_stub is not None
        explicit_cash_flows = []
        for index, item in enumerate(forecast.observations):
            is_stub = index == 0 and dcf_stub is not None
            period = (
                self._stub_discount_period(
                    dcf_stub,
                    selected_valuation_date,
                    timing_offset,
                )
                if is_stub
                else self._discount_period(
                    item,
                    selected_valuation_date,
                    timing_offset,
                    use_calendar_periods=use_calendar_periods,
                )
            )
            explicit_cash_flows.append(
                CashFlow(
                    amount=dcf_stub.fcff if is_stub else item.fcff,
                    period=period,
                    label=(
                        f"FY{item.fiscal_year}E FCFF remaining stub"
                        if is_stub
                        else f"FY{item.fiscal_year}E FCFF"
                    ),
                )
            )
        explicit_cash_flows = tuple(explicit_cash_flows)
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
        share_dilution_sensitivities = (
            tuple(
                self._share_dilution_sensitivity(
                    equity_value=equity_value,
                    base_share_count=capital_bridge.diluted_shares,
                    dilution_percentage=dilution_percentage,
                )
                for dilution_percentage in _DILUTION_SENSITIVITY_PERCENTAGES
            )
            if equity_value > 0
            else ()
        )
        terminal_percentage = (
            terminal_present_value.present_value / enterprise_value * Decimal(100)
            if enterprise_value != 0
            else None
        )
        warnings = [*forecast.warnings, *capital_bridge.warnings]
        if dcf_stub is not None and capital_bridge.period_end < dcf_stub.period_start:
            warnings.append(
                f"Capital bridge is dated {capital_bridge.period_end.isoformat()} "
                f"before remaining stub start {dcf_stub.period_start.isoformat()}; "
                "current-period debt/cash data were unavailable"
            )
        if (
            capital_bridge.share_count_basis
            == ShareCountBasis.CURRENT_SHARES_OUTSTANDING
        ):
            if not any(
                "current shares outstanding" in warning.casefold()
                and "options" in warning.casefold()
                and "rsu" in warning.casefold()
                and "other claims" in warning.casefold()
                for warning in warnings
            ):
                warnings.append(_CURRENT_SHARES_LIMITATION_WARNING)
        elif capital_bridge.share_count_basis == ShareCountBasis.USER_SUPPLIED:
            if not any(
                "user-supplied share count" in warning.casefold()
                for warning in warnings
            ):
                warnings.append(_USER_SUPPLIED_SHARES_WARNING)
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
            warnings.append(
                "Share-dilution sensitivity is not meaningful for non-positive "
                "equity value and was omitted"
            )
        if parameters.cash_flow_timing == CashFlowTiming.MID_YEAR:
            warnings.append(
                "Explicit FCFF uses mid-year timing; terminal value remains at "
                "the end of the final forecast year"
            )
        if (selected_valuation_date - capital_bridge.period_end).days > (
            _CAPITAL_BRIDGE_STALE_DAYS
        ) and not any(
            warning.startswith(("Capital-bridge data are", "Capital bridge is dated"))
            for warning in warnings
        ):
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

        from edgarito.services.forecasting._fcff.service import FcffForecastService

        forecast_service = FcffForecastService()
        identity_issues = forecast_service.economic_identity_issues(forecast)
        if identity_issues:
            if any(observation.cell_audits for observation in forecast.observations):
                raise ValueError(
                    "FCFF forecast economic identities are inconsistent: "
                    + "; ".join(identity_issues)
                )
            generated_cell_audits = forecast_service.build_legacy_inconsistent_audits(
                forecast, identity_issues
            )
        else:
            generated_cell_audits = forecast_service.build_cell_audits(forecast)
        forecast_cell_audits = {
            observation.fiscal_year: generated_cell_audits[index]
            for index, observation in enumerate(forecast.observations)
        }

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
            forecast_seed_type=forecast.seed_type.value,
            forecast_seed_methodology=forecast.seed_methodology,
            forecast_seed_period_end=forecast.seed_period_end,
            forecast_actual_quarters=forecast.actual_quarters,
            financial_snapshot_retrieved_at=(forecast.financial_snapshot_retrieved_at),
            observation_availability_mode=forecast.availability_mode,
            forecast_assumption_sources={
                driver.value: source.value
                for driver, source in forecast.assumption_sources.items()
            },
            forecast_cell_audits=forecast_cell_audits,
            operating_driver_coverage=forecast.operating_driver_coverage,
            operating_reconstruction_error=forecast.operating_reconstruction_error,
            operating_confidence=forecast.operating_confidence,
            operating_own_supported_years=forecast.operating_own_supported_years,
            operating_consensus_years=forecast.operating_consensus_years,
            operating_divergence_by_year=forecast.operating_divergence_by_year,
            operating_divergence=forecast.operating_divergence,
            operating_transition_start_year=forecast.operating_transition_start_year,
            operating_warnings=forecast.operating_warnings,
            operating_selected_revenue_by_year=forecast.operating_selected_revenue_by_year,
            operating_source_by_year=forecast.operating_source_by_year,
            operating_confidence_by_year=forecast.operating_confidence_by_year,
            capital_bridge=capital_bridge,
            explicit_forecast_present_value=explicit_present_value,
            terminal_value=terminal_value,
            terminal_present_value=terminal_present_value,
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            value_per_share=value_per_share,
            share_dilution_sensitivities=share_dilution_sensitivities,
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
    def _stub_discount_period(
        cls,
        stub: FcffForecastDcfStub,
        valuation_date: datetime.date,
        timing_offset: Decimal,
    ) -> Decimal:
        """Discount a remaining stub at its end or midpoint.

        The ordinary mid-year convention subtracts half a full fiscal year.
        Applying that convention to a short post-YTD stub could place the cash
        flow before the valuation date, so the stub midpoint is calculated from
        its own remaining period instead.
        """

        period = cls._year_fraction(valuation_date, stub.period_end)
        if timing_offset:
            period /= Decimal(2)
        if period < 0:
            raise ValueError(
                "Mid-year cash-flow timing falls before the valuation date; update "
                "the financial base period or use end-of-period timing"
            )
        return period

    @classmethod
    def _validate_dcf_stub(
        cls,
        forecast: FcffForecast,
        capital_bridge: FcffDcfCapitalBridge,
        valuation_date: datetime.date,
    ) -> FcffForecastDcfStub | None:
        """Validate the optional post-YTD flow before it enters the DCF.

        The full-year observation remains the reporting representation.  A
        missing or stale stub on a YTD seed must not silently fall back to that
        full-year amount, because doing so double-counts the reported YTD
        activity against the capital bridge.
        """

        is_ytd = forecast.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
        stub = forecast.dcf_stub
        if not is_ytd:
            if stub is not None:
                raise ValueError(
                    "FCFF DCF remaining stub metadata is only valid for "
                    "ForecastSeedType.YTD_PLUS_FORECAST"
                )
            return None

        if forecast.ytd_anchor is None:
            raise ValueError(
                "YTD FCFF DCF requires YTD anchor metadata for its remaining stub"
            )
        if not isinstance(stub, FcffForecastDcfStub):
            raise ValueError(
                "YTD FCFF DCF requires complete remaining stub metadata; "
                "the DCF stub is missing"
            )

        anchor = forecast.ytd_anchor
        first = forecast.observations[0]
        if stub.forecast_year != first.forecast_year:
            raise ValueError("FCFF DCF stub forecast year must match first observation")
        if stub.forecast_year != 1:
            raise ValueError("FCFF DCF stub must be the first forecast flow")
        if (
            stub.fiscal_year != first.fiscal_year
            or stub.fiscal_year != anchor.fiscal_year
        ):
            raise ValueError("FCFF DCF stub fiscal year must match the YTD anchor")
        if forecast.current_fiscal_year != stub.fiscal_year:
            raise ValueError(
                "FCFF DCF stub fiscal year must match the current forecast year"
            )
        if forecast.actual_quarters != anchor.actual_quarters:
            raise ValueError("FCFF DCF stub quarter count must match the YTD anchor")
        if stub.period_start != forecast.base_period_end:
            raise ValueError(
                "FCFF DCF stub start must match the forecast base period end"
            )
        if stub.period_start != anchor.ytd_period_end:
            raise ValueError("FCFF DCF stub start must match the YTD period end")
        if forecast.seed_period_end is None:
            raise ValueError("YTD FCFF DCF requires a forecast seed period end")
        if stub.period_start != forecast.seed_period_end:
            raise ValueError("FCFF DCF stub start must match the forecast seed period")
        if (
            stub.period_end != first.period_end
            or stub.period_end != anchor.fiscal_year_end
        ):
            raise ValueError(
                "FCFF DCF stub end must match the first forecast fiscal-year end"
            )
        if stub.period_start >= stub.period_end:
            raise ValueError("FCFF DCF stub period must end after its start")
        if stub.unit != forecast.unit or first.unit != forecast.unit:
            raise ValueError("FCFF DCF stub and forecast must use one currency")
        if capital_bridge.period_end > stub.period_start:
            raise ValueError(
                "FCFF DCF capital bridge period must match or precede the remaining "
                "stub start; it cannot follow the remaining stub start "
                f"(bridge {capital_bridge.period_end.isoformat()}; "
                f"stub {stub.period_start.isoformat()})"
            )
        if capital_bridge.period_end > valuation_date:
            raise ValueError(
                "FCFF DCF capital bridge period cannot follow the valuation date"
            )
        if valuation_date < stub.period_start:
            raise ValueError("Valuation date cannot precede the remaining stub start")
        if valuation_date >= stub.period_end:
            raise ValueError(
                "Valuation date must precede the remaining stub period end"
            )

        actual_tax_rate = (
            anchor.actual_tax_rate
            if anchor.actual_tax_rate is not None
            else anchor.tax_rate
        )
        actual_ytd_nopat = anchor.actual_operating_income * (
            Decimal(1) - actual_tax_rate / Decimal(100)
        )
        expected_components = {
            "annual_nopat": first.nopat,
            "actual_ytd_nopat": actual_ytd_nopat,
            "annual_depreciation_and_amortization": (
                first.depreciation_and_amortization
            ),
            "actual_ytd_depreciation_and_amortization": (
                anchor.actual_depreciation_and_amortization
            ),
            "annual_capital_expenditures": first.capital_expenditures,
            "actual_ytd_capital_expenditures": anchor.actual_capital_expenditures,
            "fiscal_year_end_operating_working_capital": (
                first.operating_working_capital
            ),
            "actual_ytd_operating_working_capital": (
                anchor.actual_operating_working_capital
            ),
        }
        for field, expected in expected_components.items():
            if getattr(stub, field) != expected:
                raise ValueError(
                    f"FCFF DCF stub {field} does not match the forecast and YTD anchor"
                )
        expected_fcff = (
            first.nopat
            - actual_ytd_nopat
            + first.depreciation_and_amortization
            - anchor.actual_depreciation_and_amortization
            - first.capital_expenditures
            + anchor.actual_capital_expenditures
            - first.operating_working_capital
            + anchor.actual_operating_working_capital
        )
        if stub.fcff != expected_fcff:
            raise ValueError(
                "FCFF DCF stub FCFF does not match the remaining-period formula"
            )
        return stub

    @classmethod
    def _year_fraction(cls, start: datetime.date, end: datetime.date) -> Decimal:
        return Decimal((end - start).days) / cls._DAYS_PER_YEAR

    @staticmethod
    def _share_dilution_sensitivity(
        *,
        equity_value: Decimal,
        base_share_count: Decimal,
        dilution_percentage: Decimal,
    ) -> FcffDcfShareDilutionSensitivity:
        share_count = base_share_count * (
            Decimal(1) + dilution_percentage / Decimal(100)
        )
        return FcffDcfShareDilutionSensitivity(
            dilution_percentage=dilution_percentage,
            base_share_count=base_share_count,
            share_count=share_count,
            equity_value=equity_value,
            value_per_share=equity_value / share_count,
        )

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
