"""Provider-neutral intrinsic valuation engines and specialized adapters."""

from __future__ import annotations

import datetime
from decimal import Decimal
from statistics import median

from edgarito.schemas.valuation.intrinsic import (
    ComponentValuationMethod,
    ComponentValueBasis,
    DividendDiscountDetails,
    DividendDiscountInput,
    FcfeDcfDetails,
    FcfeDcfInput,
    FcfeForecastPeriod,
    ForecastSummaryPoint,
    InputProvenance,
    IntrinsicValuationContext,
    IntrinsicValuationResult,
    ModelWarning,
    PipelineProject,
    PropertyAsset,
    ResidualIncomeDetails,
    ResidualIncomeInput,
    ResidualIncomePeriod,
    ResolvedModelAssumption,
    ResourceProject,
    SotpAdjustmentKind,
    SotpComponent,
    SotpDetails,
    SotpValuationInput,
    ValuationConfidence,
    ValuedSotpComponent,
    WarningSeverity,
)
from edgarito.services.valuation.discounting import PresentValueService
from edgarito.services.valuation.models import ValuationModel

_HUNDRED = Decimal(100)


def _growth_factor(rate: Decimal) -> Decimal:
    return Decimal(1) + rate / _HUNDRED


def _validate_terminal_spread(
    discount_rate: Decimal,
    growth_rate: Decimal,
    *,
    return_on_equity: Decimal | None = None,
) -> None:
    if growth_rate >= discount_rate:
        raise ValueError("Terminal growth must be below cost of equity")
    if return_on_equity is not None:
        if growth_rate >= return_on_equity:
            raise ValueError("Terminal growth must be below terminal ROE")
        if return_on_equity <= 0:
            raise ValueError("Terminal ROE must be positive")


def _terminal_value(
    cash_flow: Decimal, discount_rate: Decimal, growth: Decimal
) -> Decimal:
    _validate_terminal_spread(discount_rate, growth)
    return cash_flow / ((discount_rate - growth) / _HUNDRED)


def _common_result(
    *,
    model: ValuationModel,
    adapter: str,
    context: IntrinsicValuationContext,
    equity_value: Decimal,
    details: object,
    assumptions: tuple[ResolvedModelAssumption, ...],
    forecast_summary: tuple[ForecastSummaryPoint, ...],
    warnings: tuple[ModelWarning, ...] = (),
) -> IntrinsicValuationResult[object]:
    return IntrinsicValuationResult[object](
        model=model,
        adapter=adapter,
        company_id=context.company_id,
        company_name=context.company_name,
        ticker=context.ticker,
        valuation_date=context.valuation_date,
        currency=context.currency,
        equity_value=equity_value,
        diluted_shares=context.diluted_shares,
        value_per_share=equity_value / context.diluted_shares,
        assumptions=assumptions,
        forecast_summary=forecast_summary,
        confidence=context.confidence,
        warnings=warnings,
        provenance=context.provenance,
        details=details,
    )


class FcfeDcfService:
    """Discount distributable equity cash flows without an enterprise bridge."""

    @staticmethod
    def corporate_period(
        *,
        label: str,
        period: Decimal,
        net_income: Decimal,
        depreciation_and_amortization: Decimal,
        capital_expenditures: Decimal,
        working_capital_change: Decimal,
        debt_financing_ratio: Decimal,
    ) -> FcfeForecastPeriod:
        if not Decimal(0) <= debt_financing_ratio <= Decimal(1):
            raise ValueError("Debt-financing ratio must be between zero and one")
        reinvestment = (
            capital_expenditures
            - depreciation_and_amortization
            + working_capital_change
        )
        net_borrowing = reinvestment * debt_financing_ratio
        fcfe = (
            net_income
            + depreciation_and_amortization
            - capital_expenditures
            - working_capital_change
            + net_borrowing
        )
        return FcfeForecastPeriod(
            label=label,
            period=period,
            net_income=net_income,
            depreciation_and_amortization=depreciation_and_amortization,
            capital_expenditures=capital_expenditures,
            working_capital_change=working_capital_change,
            net_borrowing=net_borrowing,
            fcfe=fcfe,
        )

    @staticmethod
    def regulated_period(
        *,
        label: str,
        period: Decimal,
        net_income: Decimal,
        required_common_equity_change: Decimal,
    ) -> FcfeForecastPeriod:
        return FcfeForecastPeriod(
            label=label,
            period=period,
            net_income=net_income,
            required_common_equity_change=required_common_equity_change,
            fcfe=net_income - required_common_equity_change,
        )

    def value(self, inputs: FcfeDcfInput) -> IntrinsicValuationResult[FcfeDcfDetails]:
        _validate_terminal_spread(
            inputs.cost_of_equity,
            inputs.terminal_growth_rate,
            return_on_equity=inputs.terminal_return_on_equity,
        )
        explicit_pv = sum(
            (
                PresentValueService.present_value(
                    period.fcfe, inputs.cost_of_equity, period.period
                )
                for period in inputs.periods
            ),
            Decimal(0),
        )
        retention = inputs.terminal_growth_rate / inputs.terminal_return_on_equity
        terminal_income = inputs.terminal_net_income or inputs.periods[-1].net_income
        assert terminal_income is not None
        terminal_income *= _growth_factor(inputs.terminal_growth_rate)
        terminal_fcfe = terminal_income * (Decimal(1) - retention)
        terminal_value = _terminal_value(
            terminal_fcfe, inputs.cost_of_equity, inputs.terminal_growth_rate
        )
        terminal_pv = PresentValueService.present_value(
            terminal_value,
            inputs.cost_of_equity,
            inputs.periods[-1].period,
        )
        equity_value = explicit_pv + terminal_pv
        details = FcfeDcfDetails(
            periods=inputs.periods,
            explicit_fcfe_present_value=explicit_pv,
            terminal_fcfe=terminal_fcfe,
            terminal_value=terminal_value,
            terminal_present_value=terminal_pv,
            cost_of_equity=inputs.cost_of_equity,
            terminal_growth_rate=inputs.terminal_growth_rate,
            terminal_return_on_equity=inputs.terminal_return_on_equity,
            terminal_retention_ratio=retention,
            regulated_financial=inputs.regulated_financial,
        )
        result = _common_result(
            model=ValuationModel.EQUITY_DCF,
            adapter="regulated FCFE"
            if inputs.regulated_financial
            else "corporate FCFE",
            context=inputs.context,
            equity_value=equity_value,
            details=details,
            assumptions=(
                ResolvedModelAssumption(
                    name="Cost of equity",
                    value=inputs.cost_of_equity,
                    unit="percent",
                    source="resolved",
                ),
                ResolvedModelAssumption(
                    name="Terminal growth",
                    value=inputs.terminal_growth_rate,
                    unit="percent",
                    source="resolved",
                ),
                ResolvedModelAssumption(
                    name="Terminal ROE",
                    value=inputs.terminal_return_on_equity,
                    unit="percent",
                    source="resolved",
                ),
            ),
            forecast_summary=tuple(
                ForecastSummaryPoint(
                    label=item.label,
                    period=item.period,
                    amount=item.fcfe,
                    present_value=PresentValueService.present_value(
                        item.fcfe, inputs.cost_of_equity, item.period
                    ),
                    unit=inputs.context.currency,
                )
                for item in inputs.periods
            ),
        )
        return IntrinsicValuationResult[FcfeDcfDetails].model_validate(
            result.model_dump()
        )


class DividendDiscountService:
    """Gordon and multistage dividend discount models."""

    @staticmethod
    def resolve_terminal_policy(
        *,
        growth: Decimal | None,
        return_on_equity: Decimal | None,
        payout_ratio: Decimal | None,
    ) -> tuple[Decimal, Decimal, Decimal]:
        if growth is None:
            assert return_on_equity is not None and payout_ratio is not None
            growth = return_on_equity * (Decimal(1) - payout_ratio)
        elif return_on_equity is None:
            assert payout_ratio is not None
            retention = Decimal(1) - payout_ratio
            if retention <= 0:
                raise ValueError("Growth and full payout cannot imply terminal ROE")
            return_on_equity = growth / retention
        elif payout_ratio is None:
            payout_ratio = Decimal(1) - growth / return_on_equity
        implied_growth = return_on_equity * (Decimal(1) - payout_ratio)
        if abs(implied_growth - growth) > Decimal("0.000001"):
            raise ValueError(
                "Terminal ROE, payout, and growth assumptions are inconsistent"
            )
        if not Decimal(0) <= payout_ratio <= Decimal(1):
            raise ValueError("Terminal payout ratio must be between zero and one")
        _validate_terminal_spread(
            Decimal("1e99"), growth, return_on_equity=return_on_equity
        )
        return growth, return_on_equity, payout_ratio

    def value(
        self, inputs: DividendDiscountInput
    ) -> IntrinsicValuationResult[DividendDiscountDetails]:
        growth, terminal_roe, payout = self.resolve_terminal_policy(
            growth=inputs.terminal_growth_rate,
            return_on_equity=inputs.terminal_return_on_equity,
            payout_ratio=inputs.terminal_payout_ratio,
        )
        _validate_terminal_spread(
            inputs.cost_of_equity, growth, return_on_equity=terminal_roe
        )
        warnings: list[ModelWarning] = []
        if inputs.mode == "gordon":
            assert inputs.next_dividend is not None
            explicit_pv = Decimal(0)
            terminal_dividend = inputs.next_dividend
            terminal_value = _terminal_value(
                terminal_dividend, inputs.cost_of_equity, growth
            )
            terminal_pv = terminal_value
        else:
            explicit_pv = sum(
                (
                    PresentValueService.present_value(
                        period.dividends, inputs.cost_of_equity, period.period
                    )
                    for period in inputs.periods
                ),
                Decimal(0),
            )
            last = inputs.periods[-1]
            terminal_earnings = inputs.terminal_earnings or last.earnings
            if terminal_earnings is None:
                if payout == 0:
                    raise ValueError(
                        "Terminal earnings are required when payout is zero"
                    )
                terminal_earnings = last.dividends / payout
            terminal_dividend = terminal_earnings * _growth_factor(growth) * payout
            terminal_value = _terminal_value(
                terminal_dividend, inputs.cost_of_equity, growth
            )
            terminal_pv = PresentValueService.present_value(
                terminal_value, inputs.cost_of_equity, last.period
            )
            payout_values = [
                item.payout_ratio
                for item in inputs.periods
                if item.payout_ratio is not None
            ]
            if payout_values and max(payout_values) - min(payout_values) > Decimal(
                "0.15"
            ):
                warnings.append(
                    ModelWarning(
                        code="unstable_payout",
                        severity=WarningSeverity.MEDIUM,
                        summary="Forecast payout policy is unstable.",
                    )
                )
            if any(value > 1 for value in payout_values):
                warnings.append(
                    ModelWarning(
                        code="payout_above_earnings",
                        severity=WarningSeverity.MEDIUM,
                        summary="A forecast dividend temporarily exceeds earnings.",
                    )
                )
        reference_dividend = (
            inputs.next_dividend
            if inputs.mode == "gordon"
            else inputs.periods[0].dividends
        )
        if inputs.distributable_fcfe not in {None, Decimal(0)}:
            disconnect = abs(reference_dividend - inputs.distributable_fcfe) / abs(
                inputs.distributable_fcfe
            )
            if disconnect > Decimal("0.25"):
                warnings.append(
                    ModelWarning(
                        code="dividend_fcfe_disconnect",
                        severity=WarningSeverity.MEDIUM,
                        summary="Dividend differs from distributable FCFE by more than 25%.",
                    )
                )
        details = DividendDiscountDetails(
            mode=inputs.mode,
            periods=inputs.periods,
            explicit_dividend_present_value=explicit_pv,
            terminal_dividend=terminal_dividend,
            terminal_value=terminal_value,
            terminal_present_value=terminal_pv,
            cost_of_equity=inputs.cost_of_equity,
            terminal_growth_rate=growth,
            terminal_return_on_equity=terminal_roe,
            terminal_payout_ratio=payout,
            terminal_retention_ratio=Decimal(1) - payout,
        )
        result = _common_result(
            model=ValuationModel.DIVIDEND_DISCOUNT,
            adapter=f"{inputs.mode} DDM",
            context=inputs.context,
            equity_value=explicit_pv + terminal_pv,
            details=details,
            assumptions=(
                ResolvedModelAssumption(
                    name="Cost of equity",
                    value=inputs.cost_of_equity,
                    unit="percent",
                    source="resolved",
                ),
                ResolvedModelAssumption(
                    name="Terminal growth",
                    value=growth,
                    unit="percent",
                    source="resolved",
                ),
                ResolvedModelAssumption(
                    name="Terminal ROE",
                    value=terminal_roe,
                    unit="percent",
                    source="resolved",
                ),
                ResolvedModelAssumption(
                    name="Terminal payout",
                    value=payout,
                    unit="ratio",
                    source="resolved",
                ),
            ),
            forecast_summary=tuple(
                ForecastSummaryPoint(
                    label=item.label,
                    period=item.period,
                    amount=item.dividends,
                    present_value=PresentValueService.present_value(
                        item.dividends, inputs.cost_of_equity, item.period
                    ),
                    unit=inputs.context.currency,
                )
                for item in inputs.periods
            ),
            warnings=tuple(warnings),
        )
        return IntrinsicValuationResult[DividendDiscountDetails].model_validate(
            result.model_dump()
        )


class ResidualIncomeService:
    """Value book equity plus finite, fading residual income."""

    @staticmethod
    def infer_policy(
        observations: tuple[tuple[int, Decimal, Decimal], ...],
    ) -> tuple[Decimal, Decimal, ValuationConfidence]:
        """Infer median ROE and payout from year, ROE, payout observations."""
        if len(observations) < 3:
            raise ValueError(
                "Residual-income inference requires three clean annual observations"
            )
        years = tuple(item[0] for item in observations)
        if years != tuple(range(years[0], years[0] + len(years))):
            raise ValueError(
                "Residual-income history must contain consecutive annual periods"
            )
        if any(roe <= 0 or payout < 0 or payout > 1 for _, roe, payout in observations):
            raise ValueError("Residual-income history contains an unclean observation")
        return (
            Decimal(str(median([item[1] for item in observations]))),
            Decimal(str(median([item[2] for item in observations]))),
            ValuationConfidence.MEDIUM,
        )

    def value(
        self, inputs: ResidualIncomeInput
    ) -> IntrinsicValuationResult[ResidualIncomeDetails]:
        periods: list[ResidualIncomePeriod] = []
        book = inputs.starting_book_value
        for index, (roe, payout) in enumerate(
            zip(inputs.return_on_equity_path, inputs.payout_ratio_path, strict=True), 1
        ):
            period, book = self._period(
                index=index,
                book=book,
                roe=roe,
                payout=payout,
                cost_of_equity=inputs.cost_of_equity,
                transition=False,
            )
            periods.append(period)

        excess = inputs.return_on_equity_path[-1] - inputs.cost_of_equity
        transition_years = 0
        while abs(excess) > Decimal("0.25") and transition_years < 20:
            excess *= inputs.excess_return_persistence
            transition_years += 1
            roe = inputs.cost_of_equity + excess
            period, book = self._period(
                index=len(periods) + 1,
                book=book,
                roe=roe,
                payout=inputs.payout_ratio_path[-1],
                cost_of_equity=inputs.cost_of_equity,
                transition=True,
            )
            periods.append(period)

        residual_pv = sum(
            (
                PresentValueService.present_value(
                    item.residual_income, inputs.cost_of_equity, item.period
                )
                for item in periods
            ),
            Decimal(0),
        )
        equity_value = inputs.starting_book_value + residual_pv
        details = ResidualIncomeDetails(
            starting_book_value=inputs.starting_book_value,
            book_value_basis=inputs.book_value_basis,
            periods=tuple(periods),
            residual_income_present_value=residual_pv,
            transition_years=transition_years,
            excess_return_persistence=inputs.excess_return_persistence,
            cost_of_equity=inputs.cost_of_equity,
        )
        warnings = ()
        if transition_years == 20 and abs(excess) > Decimal("0.25"):
            warnings = (
                ModelWarning(
                    code="residual_fade_limit",
                    severity=WarningSeverity.MEDIUM,
                    summary=(
                        "Excess ROE remained above 0.25 percentage points after "
                        "the 20-year transition cap; terminal residual income is zero."
                    ),
                ),
            )
        result = _common_result(
            model=ValuationModel.RESIDUAL_INCOME,
            adapter="book value / fading excess return",
            context=inputs.context,
            equity_value=equity_value,
            details=details,
            assumptions=(
                ResolvedModelAssumption(
                    name="Cost of equity",
                    value=inputs.cost_of_equity,
                    unit="percent",
                    source="resolved",
                ),
                ResolvedModelAssumption(
                    name="Excess-return persistence",
                    value=inputs.excess_return_persistence,
                    unit="ratio",
                    source="profile"
                    if inputs.excess_return_persistence != Decimal("0.75")
                    else "policy default",
                ),
                ResolvedModelAssumption(
                    name="Book value basis",
                    value=inputs.book_value_basis,
                    source="profile or normalized history",
                ),
            ),
            forecast_summary=tuple(
                ForecastSummaryPoint(
                    label=item.label,
                    period=Decimal(item.period),
                    amount=item.residual_income,
                    present_value=PresentValueService.present_value(
                        item.residual_income, inputs.cost_of_equity, item.period
                    ),
                    unit=inputs.context.currency,
                )
                for item in periods
            ),
            warnings=warnings,
        )
        return IntrinsicValuationResult[ResidualIncomeDetails].model_validate(
            result.model_dump()
        )

    @staticmethod
    def _period(
        *,
        index: int,
        book: Decimal,
        roe: Decimal,
        payout: Decimal,
        cost_of_equity: Decimal,
        transition: bool,
    ) -> tuple[ResidualIncomePeriod, Decimal]:
        income = book * roe / _HUNDRED
        dividends = income * payout
        equity_charge = book * cost_of_equity / _HUNDRED
        residual = income - equity_charge
        ending_book = book + income - dividends
        period = ResidualIncomePeriod(
            label=f"Year {index}",
            period=index,
            opening_book_value=book,
            return_on_equity=roe,
            net_income=income,
            payout_ratio=payout,
            dividends=dividends,
            equity_charge=equity_charge,
            residual_income=residual,
            ending_book_value=ending_book,
            transition=transition,
        )
        return period, ending_book


class SotpValuationService:
    """Aggregate provider-neutral components without double counting adjustments."""

    def value(
        self, inputs: SotpValuationInput
    ) -> IntrinsicValuationResult[SotpDetails]:
        included = set().union(
            *(
                set(component.included_balance_sheet_items)
                for component in inputs.components
            )
        )
        duplicated = {
            adjustment.balance_sheet_item
            for adjustment in inputs.adjustments
            if adjustment.balance_sheet_item
            and adjustment.balance_sheet_item in included
        }
        if duplicated:
            names = ", ".join(sorted(duplicated))
            raise ValueError(
                f"SOTP adjustment duplicates component balance-sheet items: {names}"
            )

        valued: list[ValuedSotpComponent] = []
        for component in inputs.components:
            reporting_value = component.value * component.fx_rate_to_reporting_currency
            component_debt = (
                component.component_net_debt * component.fx_rate_to_reporting_currency
            )
            if component.value_basis == ComponentValueBasis.ENTERPRISE:
                equity_before_ownership = reporting_value - component_debt
            else:
                if component.component_net_debt != 0:
                    raise ValueError(
                        "Equity-valued SOTP components cannot subtract net debt"
                    )
                equity_before_ownership = reporting_value
            owned = equity_before_ownership * component.ownership
            valued.append(
                ValuedSotpComponent(
                    name=component.name,
                    method=component.method,
                    reported_value=reporting_value,
                    component_net_debt=component_debt,
                    equity_value_before_ownership=equity_before_ownership,
                    ownership=component.ownership,
                    owned_equity_value=owned,
                    reporting_currency=inputs.context.currency,
                )
            )
        gross = sum((item.owned_equity_value for item in valued), Decimal(0))
        adjustment_totals = {kind: Decimal(0) for kind in SotpAdjustmentKind}
        for adjustment in inputs.adjustments:
            adjustment_totals[adjustment.kind] += (
                adjustment.amount * adjustment.fx_rate_to_reporting_currency
            )
        pre_discount = (
            gross
            + adjustment_totals[SotpAdjustmentKind.NON_OPERATING_ASSET]
            - adjustment_totals[SotpAdjustmentKind.CORPORATE_DEBT]
            - adjustment_totals[SotpAdjustmentKind.OTHER_LIABILITY]
            - adjustment_totals[SotpAdjustmentKind.MINORITY_INTEREST]
        )
        equity_value = pre_discount * (Decimal(1) - inputs.holding_company_discount)
        details = SotpDetails(
            components=tuple(valued),
            owned_gross_asset_value=gross,
            non_operating_assets=adjustment_totals[
                SotpAdjustmentKind.NON_OPERATING_ASSET
            ],
            corporate_debt=adjustment_totals[SotpAdjustmentKind.CORPORATE_DEBT],
            other_liabilities=adjustment_totals[SotpAdjustmentKind.OTHER_LIABILITY],
            minority_interests=adjustment_totals[SotpAdjustmentKind.MINORITY_INTEREST],
            pre_discount_equity_value=pre_discount,
            holding_company_discount=inputs.holding_company_discount,
        )
        provenance = list(inputs.context.provenance)
        for component in inputs.components:
            provenance.extend(component.provenance)
            provenance.append(
                InputProvenance(
                    field=f"{component.name}.fx_rate",
                    source=component.fx_rate_source,
                    observed_on=component.fx_rate_date,
                )
            )
        result = _common_result(
            model=ValuationModel.NAV_SOTP,
            adapter=inputs.adapter,
            context=inputs.context.model_copy(update={"provenance": tuple(provenance)}),
            equity_value=equity_value,
            details=details,
            assumptions=(
                ResolvedModelAssumption(
                    name="Holding-company discount",
                    value=inputs.holding_company_discount,
                    unit="ratio",
                    source="profile",
                ),
            ),
            forecast_summary=tuple(
                ForecastSummaryPoint(
                    label=item.name,
                    period=Decimal(0),
                    amount=item.owned_equity_value,
                    unit=inputs.context.currency,
                )
                for item in valued
            ),
        )
        return IntrinsicValuationResult[SotpDetails].model_validate(result.model_dump())


class PropertyNavAdapter:
    def to_components(
        self,
        properties: tuple[PropertyAsset, ...],
        *,
        valuation_date: datetime.date,
    ) -> tuple[SotpComponent, ...]:
        if not properties:
            raise ValueError(
                "Property NAV requires NOI and cap rate for at least one asset"
            )
        return tuple(
            SotpComponent(
                name=asset.name,
                method=ComponentValuationMethod.ASSET_NAV,
                value_basis=ComponentValueBasis.ENTERPRISE,
                value=asset.noi / (asset.cap_rate / _HUNDRED),
                currency=asset.currency,
                ownership=asset.ownership,
                fx_rate_date=valuation_date,
                fx_rate_source="reporting currency",
                provenance=asset.provenance,
            )
            for asset in properties
        )


class ReitAffoAdapter:
    @staticmethod
    def derive_affo(
        *, ffo: Decimal | None, recurring_adjustments: tuple[Decimal, ...]
    ) -> Decimal:
        if ffo is None or not recurring_adjustments:
            raise ValueError("AFFO requires FFO and explicit recurring adjustments")
        return ffo + sum(recurring_adjustments, Decimal(0))


class ResourceNavAdapter:
    def to_components(
        self, projects: tuple[ResourceProject, ...], *, valuation_date: datetime.date
    ) -> tuple[SotpComponent, ...]:
        scenarios = {project.scenario for project in projects}
        if len(scenarios) > 1:
            raise ValueError(
                "Resource commodity scenarios must be valued separately; use to_scenarios"
            )
        return tuple(
            SotpComponent(
                name=f"{project.name} [{project.scenario}]",
                method=ComponentValuationMethod.SPECIALIZED_ADAPTER,
                value_basis=ComponentValueBasis.EQUITY,
                value=sum(
                    (
                        PresentValueService.present_value(
                            year.cash_flow, project.discount_rate, year.year
                        )
                        for year in project.years
                    ),
                    Decimal(0),
                ),
                currency=project.currency,
                ownership=project.ownership,
                fx_rate_date=valuation_date,
                fx_rate_source="reporting currency",
                provenance=project.provenance,
            )
            for project in projects
        )

    def to_scenarios(
        self, projects: tuple[ResourceProject, ...], *, valuation_date: datetime.date
    ) -> dict[str, tuple[SotpComponent, ...]]:
        scenarios = sorted({project.scenario for project in projects})
        return {
            scenario: self.to_components(
                tuple(project for project in projects if project.scenario == scenario),
                valuation_date=valuation_date,
            )
            for scenario in scenarios
        }


class PipelineRnpvAdapter:
    def to_components(
        self, projects: tuple[PipelineProject, ...], *, valuation_date: datetime.date
    ) -> tuple[SotpComponent, ...]:
        return tuple(
            SotpComponent(
                name=project.name,
                method=ComponentValuationMethod.RNPV,
                value_basis=ComponentValueBasis.EQUITY,
                value=sum(
                    (
                        PresentValueService.present_value(
                            year.success_cash_flow * project.success_probability
                            - year.development_cost,
                            project.discount_rate,
                            year.year,
                        )
                        for year in project.years
                    ),
                    Decimal(0),
                ),
                currency=project.currency,
                ownership=project.ownership,
                fx_rate_date=valuation_date,
                fx_rate_source="reporting currency",
                provenance=(project.success_probability_provenance,)
                + project.provenance,
            )
            for project in projects
        )
