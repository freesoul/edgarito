import argparse
import asyncio
import datetime
import logging
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from edgarito.cli.comparables import (
    _build_comparable_report,
    _resolve_comparable_peer_symbols,
    _run_comparables,
)
from edgarito.cli.parser import build_parser
from edgarito.cli.presentation.console import (
    ClassificationConsolePresenter,
    FinancialsConsolePresenter,
    ForecastConsolePresenter,
    IndependentValuationModelsConsolePresenter,
    MetricsConsolePresenter,
    RedFlagsConsolePresenter,
    SpecializedExtractionConsolePresenter,
    ValuationReportConsolePresenter,
    ValuationSelectionConsolePresenter,
)
from edgarito.config.red_flags import RedFlagsProfileLoader
from edgarito.config.valuation import (
    ForecastMethod,
    ForecastValuationProfile,
    ValuationProfileLoader,
)
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.logger import configure_logger
from edgarito.schemas.guidance.management import GuidanceOverlayResult
from edgarito.schemas.market import ReferenceSeriesKind, ReferenceValueUnit
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.assumptions import ValuationAssumptionKind
from edgarito.schemas.valuation.intrinsic import (
    DividendDiscountInput,
    DividendForecastPeriod,
    FcfeDcfInput,
    FcfeForecastPeriod,
    IntrinsicValuationContext,
    ResidualIncomeInput,
    SotpValuationInput,
    ValuationConfidence,
)
from edgarito.schemas.valuation.relative import (
    ForwardValuationMetric,
    RelativeNumeratorBasis,
)
from edgarito.schemas.valuation.specialized import SpecializedInputType
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.export import (
    CompanyAnalysisReportService,
    ExcelRenderer,
    ValuationExcelRenderer,
)
from edgarito.services.financial_observation_availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting import (
    AdaptiveMultistageFcffForecastService,
    FcffForecastParameters,
    FcffForecastService,
    ForwardGrowthEvidence,
    SimplifiedFcfForecastParameters,
    SimplifiedFcfForecastService,
)
from edgarito.services.guidance.extraction import ManagementGuidanceExtractor
from edgarito.services.guidance.overlay import GuidanceForecastOverlay
from edgarito.services.guidance.service import ManagementGuidanceService
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService
from edgarito.services.normalization.classification import (
    CompanyClassificationNormalizer,
)
from edgarito.services.normalization.yahoo_market import YahooMarketNormalizer
from edgarito.services.openai import OpenAIClient
from edgarito.services.providers.damodaran import DamodaranClient
from edgarito.services.providers.ecb import EcbClient
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.providers.fred import FredClient
from edgarito.services.providers.treasury import TreasuryClient
from edgarito.services.providers.yahoo import YahooFinanceClient
from edgarito.services.reconciliation.classification import (
    CompanyClassificationService,
)
from edgarito.services.reconciliation.financials import FinancialDataService
from edgarito.services.red_flags import InvestmentRedFlagsService
from edgarito.services.valuation import (
    BusinessArchetype,
    CashFlowTiming,
    CompanyLifecycle,
    ComparableImpliedValuationService,
    Cyclicality,
    DecisionScenarioPolicy,
    DecisionValuationService,
    DiscountRateService,
    EcbMarketDataCurrencyConverter,
    EconomicTrait,
    FcffDcfCapitalBridgeResolver,
    FcffDcfParameters,
    FcffDcfService,
    ForwardPeerMultiplesService,
    HistoricalMultiplesService,
    IntrinsicDecisionContext,
    MultipleResolver,
    RelativeValuationBasis,
    ShareRepurchaseParameters,
    SpecializedValuationExtractor,
    TerminalMetric,
    TerminalRoicResolver,
    TerminalValueMethod,
    ValuationAssumptionResolver,
    ValuationInput,
    ValuationModelSelector,
    ValuationProfileBuilder,
    ValuationProfileOverrides,
)
from edgarito.services.valuation.assumptions import CostOfEquityResolver
from edgarito.services.valuation.execution import ValuationExecutor
from edgarito.services.valuation.intrinsic import (
    DividendDiscountService,
    FcfeDcfService,
    PipelineRnpvAdapter,
    PropertyNavAdapter,
    ReitAffoAdapter,
    ResidualIncomeService,
    ResourceNavAdapter,
    SotpValuationService,
)
from edgarito.services.valuation.models import (
    MultipleConfidence,
    ResolvedMultiple,
    ValuationModel,
)
from edgarito.services.valuation.relative import (
    EQUITY_RELATIVE_BASES,
    ProviderNeutralRelativeValuationService,
)
from edgarito.settings import (
    ALPHAVANTAGE_API_KEY,
    CLASSIFICATION_PROVIDER_CONFIGURATION,
    FMP_API_KEY,
    FRED_API_KEY,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
    OPENFIGI_API_KEY,
    PROVIDER_CONFIGURATION,
)


async def _run_financials(args: argparse.Namespace) -> int:
    _validate_limit(args.limit)
    granularity = _granularity(args.period)

    concepts = (
        {FinancialConcept(value) for value in args.concept} if args.concept else None
    )
    financials = await _retrieve_financials(args, granularity, concepts)

    print(FinancialsConsolePresenter().render(financials, limit=args.limit))
    return 0


async def _run_metrics(args: argparse.Namespace) -> int:
    _validate_limit(args.limit)
    granularity = _granularity(args.period)
    selected_metrics = (
        {FinancialMetric(value) for value in args.metric} if args.metric else None
    )
    concepts = FinancialMetricsService.required_concepts(selected_metrics)
    financials = await _retrieve_financials(args, granularity, concepts)
    metrics = FinancialMetricsService().calculate(
        financials,
        granularity=granularity,
        metrics=selected_metrics,
    )

    print(MetricsConsolePresenter().render(metrics, limit=args.limit))
    return 0


async def _run_export(args: argparse.Namespace) -> int:
    granularity = _granularity(args.period)
    financials = await _retrieve_financials(args, granularity, None)
    metrics = FinancialMetricsService().calculate(
        financials,
        granularity=granularity,
    )
    report = CompanyAnalysisReportService().compose(
        financials=financials,
        metrics=metrics,
    )
    output = ExcelRenderer().render(report, args.output)
    print(f"Exported Excel workbook to {output}")
    return 0


async def _run_red_flags(args: argparse.Namespace) -> int:
    configuration = RedFlagsProfileLoader.load(args.profile)
    granularity = Granularity(args.period)
    concepts = {
        concept
        for category in configuration.enabled_categories
        for concept in configuration.required_concepts(category)
    }
    financials = await _retrieve_financials(args, granularity, concepts)
    report = InvestmentRedFlagsService(configuration).analyze(
        financials,
        granularity=granularity,
    )

    print(RedFlagsConsolePresenter().render(report, verbose=args.verbose))
    return 0


def _load_selected_valuation_profile(args):
    ticker = getattr(args, "ticker", None)
    if ticker:
        profile, _, _ = ValuationProfileLoader.load_for_ticker(ticker, args.profile)
        return profile
    return ValuationProfileLoader.load(args.profile)


async def _run_forecast(args: argparse.Namespace) -> int:
    profile = _load_selected_valuation_profile(args)
    forecast_method = (
        ForecastMethod(args.forecast_method)
        if args.forecast_method is not None
        else profile.forecast.default_method
    )
    fcff_driver_arguments = (
        args.operating_margin,
        args.tax_rate,
        args.depreciation_to_revenue,
        args.capex_to_revenue,
        args.operating_working_capital_to_revenue,
    )
    if forecast_method == ForecastMethod.SIMPLIFIED:
        if any(value is not None for value in fcff_driver_arguments):
            raise ValueError(
                "FCFF driver options cannot be used with --forecast-method simplified"
            )
        configured = profile.forecast.simplified
        parameters = SimplifiedFcfForecastParameters(
            forecast_years=(
                args.years if args.years is not None else configured.forecast_years
            ),
            revenue_growth=(
                args.revenue_growth
                if args.revenue_growth is not None
                else configured.revenue_growth
            ),
            free_cash_flow_margin=(
                args.fcf_margin
                if args.fcf_margin is not None
                else configured.free_cash_flow_margin
            ),
            historical_window=(
                args.historical_window
                if args.historical_window is not None
                else configured.historical_window
            ),
        )
        service = SimplifiedFcfForecastService()
    else:
        if args.fcf_margin is not None:
            raise ValueError(
                "--fcf-margin requires --forecast-method simplified; use the "
                "FCFF operating, tax, D&A, capex, and working-capital drivers"
            )
        configured = profile.forecast.fcff
        parameters = _fcff_parameters(args, configured)
        service = FcffForecastService()
    financials = await _retrieve_financials(
        args,
        Granularity.ANNUAL,
        service.required_concepts(),
    )
    forecast = service.forecast(financials, parameters)
    print(ForecastConsolePresenter().render(forecast))
    return 0


async def _run_profile_intrinsic_valuation(
    args: argparse.Namespace,
    profile: ForecastValuationProfile,
    selected_model: str,
) -> int:
    """Execute a profile-backed equity or asset model without an FCFF bridge."""
    concepts = ValuationProfileBuilder.required_concepts() | {
        FinancialConcept.NET_INCOME_COMMON,
        FinancialConcept.COMMON_EQUITY,
        FinancialConcept.DIVIDENDS_PAID,
        FinancialConcept.GOODWILL,
        FinancialConcept.INTANGIBLE_ASSETS_NET,
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
        FinancialConcept.CAPITAL_EXPENDITURES,
    }
    financials = await _retrieve_financials(args, Granularity.ANNUAL, concepts)
    valuation_date = datetime.date.today()
    annual = [
        item
        for item in financials.observations
        if item.granularity == Granularity.ANNUAL
        and item.fiscal_period == FiscalPeriod.FY
    ]
    if not annual:
        raise ValueError("Intrinsic equity models require annual normalized history")
    latest_by_concept = {}
    for item in sorted(annual, key=lambda value: value.period_end):
        latest_by_concept[item.concept] = item
    monetary = next(
        (
            item
            for item in sorted(annual, key=lambda value: value.period_end, reverse=True)
            if item.unit != "shares" and "/shares" not in item.unit
        ),
        None,
    )
    if monetary is None:
        raise ValueError("Could not resolve a reporting currency for valuation")
    currency = monetary.unit
    bridge = profile.valuation.capital_bridge
    share_observation = latest_by_concept.get(
        FinancialConcept.SHARES_OUTSTANDING
    ) or latest_by_concept.get(FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES)
    shares = (
        args.shares
        or bridge.diluted_shares
        or (share_observation.value if share_observation is not None else None)
    )
    if shares is None or shares <= 0:
        raise ValueError(
            "Intrinsic equity valuation requires a positive diluted share count"
        )

    terminal_growth = (
        args.terminal_growth
        if args.terminal_growth is not None
        else profile.valuation.terminal_value.perpetual_growth_rate
    )
    discount_configuration = profile.valuation.discount_rates
    archetype = profile.model_selection.business_archetype
    needs_equity_rate = selected_model in {
        "fcfe-dcf",
        "reit-affo",
        "ddm",
        "residual-income",
    } or (
        selected_model == "auto-specialized"
        and archetype
        in {
            BusinessArchetype.FINANCIAL_INTERMEDIARY,
            BusinessArchetype.REIT_PROPERTY,
        }
    )
    needs_terminal_growth = selected_model in {
        "fcfe-dcf",
        "reit-affo",
        "ddm",
    } or (
        selected_model == "auto-specialized"
        and archetype
        in {
            BusinessArchetype.FINANCIAL_INTERMEDIARY,
            BusinessArchetype.REIT_PROPERTY,
        }
    )
    needs_cost = needs_equity_rate and (
        args.cost_of_equity is None and discount_configuration.cost_of_equity is None
    )
    automatic = await _retrieve_automatic_assumption_inputs(
        args,
        financials,
        currency,
        needs_wacc=needs_cost,
        needs_terminal=needs_terminal_growth and terminal_growth is None,
        sector_override=profile.model_selection.sector,
        industry_override=profile.model_selection.industry,
    )
    resolved_equity = (
        CostOfEquityResolver().resolve(
            configuration=discount_configuration,
            valuation_date=valuation_date,
            currency=currency,
            company_id=financials.company_id,
            classification=automatic.get("classification"),
            risk_free_series=automatic.get("risk_free_series"),
            country_snapshot=automatic.get("country_snapshot"),
            industry_snapshot=automatic.get("industry_snapshot"),
            company_beta=automatic.get("company_beta"),
            cost_of_equity_override=args.cost_of_equity,
        )
        if needs_equity_rate
        else None
    )
    if needs_terminal_growth and terminal_growth is None:
        assert resolved_equity is not None
        terminal_growth, _ = ValuationAssumptionResolver()._derive_terminal_growth(
            wacc=resolved_equity.cost_of_equity,
            selected_on=valuation_date,
            currency=currency,
            company_id=financials.company_id,
            inflation_series=automatic.get("inflation_series"),
            risk_free_series=automatic.get("risk_free_series"),
        )
    if terminal_growth is None:
        terminal_growth = Decimal(0)

    available_inputs = set(profile.configured_valuation_inputs)
    available_inputs.add(ValuationInput.DILUTED_SHARES)
    if needs_equity_rate:
        available_inputs.add(ValuationInput.COST_OF_EQUITY)
    if needs_terminal_growth:
        available_inputs.add(ValuationInput.TERMINAL_GROWTH)
    if selected_model in {"fcfe-dcf", "reit-affo"}:
        available_inputs.add(ValuationInput.EQUITY_CASH_FLOW_FORECAST)
    elif selected_model == "ddm":
        available_inputs.update(
            {ValuationInput.DIVIDEND_FORECAST, ValuationInput.PAYOUT_POLICY}
        )
    elif selected_model == "residual-income":
        available_inputs.add(ValuationInput.FORECAST_ROE)
    elif selected_model in {
        "nav",
        "property-nav",
        "resource-nav",
        "pipeline-rnpv",
    }:
        available_inputs.add(ValuationInput.ASSET_LEVEL_VALUES)
        if selected_model == "nav":
            available_inputs.add(ValuationInput.SEGMENT_VALUES)
    if selected_model == "resource-nav":
        available_inputs.add(ValuationInput.RESERVE_DATA)
    if selected_model == "pipeline-rnpv":
        available_inputs.add(ValuationInput.PIPELINE_DATA)
    overrides = ValuationProfileOverrides(
        sector=profile.model_selection.sector,
        industry=profile.model_selection.industry,
        business_archetype=profile.model_selection.business_archetype,
        financial_institution_kind=(
            profile.model_selection.financial_institution_kind
            or profile.valuation.financial_institution.kind
        ),
        actuarial_detail_supplied=(
            profile.valuation.financial_institution.actuarial_detail_supplied
        ),
        regulatory_capital_constraints_supplied=bool(
            profile.valuation.financial_institution.regulatory_capital_constraints
        ),
        lifecycle=profile.model_selection.lifecycle,
        cyclicality=profile.model_selection.cyclicality,
        economic_traits=set(profile.model_selection.economic_traits),
        available_inputs=available_inputs,
        peer_count=profile.model_selection.peer_count,
    )
    economic_profile = ValuationProfileBuilder().build(
        financials, automatic.get("classification"), overrides
    )
    selection = ValuationModelSelector().select(economic_profile)
    context = IntrinsicValuationContext(
        company_id=financials.company_id,
        company_name=financials.company_name,
        ticker=financials.ticker or args.ticker,
        valuation_date=valuation_date,
        currency=currency,
        diluted_shares=shares,
        confidence=ValuationConfidence.MEDIUM,
    )
    terminal_roe_override = args.terminal_roe
    equity_rate = (
        resolved_equity.cost_of_equity if resolved_equity is not None else Decimal(0)
    )
    requested_models = None
    runners = {}
    if selected_model == "auto-specialized":
        candidates = {
            BusinessArchetype.FINANCIAL_INTERMEDIARY: (
                "residual-income",
                "fcfe-dcf",
                "ddm",
            ),
            BusinessArchetype.REIT_PROPERTY: (
                "property-nav",
                "reit-affo",
                "ddm",
            ),
            BusinessArchetype.RESOURCE_PRODUCER: ("resource-nav",),
            BusinessArchetype.PROJECT_PIPELINE: ("pipeline-rnpv",),
            BusinessArchetype.HOLDING_COMPANY: ("nav",),
            BusinessArchetype.CONGLOMERATE: ("nav",),
        }.get(economic_profile.business_archetype, ())
        for candidate in candidates:
            try:
                model, runner = _profile_model_runner(
                    selected_model=candidate,
                    profile=profile,
                    context=context,
                    annual=annual,
                    cost_of_equity=equity_rate,
                    terminal_growth=terminal_growth,
                    terminal_roe_override=terminal_roe_override,
                    args=args,
                )
            except ValueError:
                continue
            runners.setdefault(model, runner)
    else:
        try:
            model, runner = _profile_model_runner(
                selected_model=selected_model,
                profile=profile,
                context=context,
                annual=annual,
                cost_of_equity=equity_rate,
                terminal_growth=terminal_growth,
                terminal_roe_override=terminal_roe_override,
                args=args,
            )
        except ValueError as exc:
            model = {
                "fcfe-dcf": ValuationModel.EQUITY_DCF,
                "reit-affo": ValuationModel.EQUITY_DCF,
                "ddm": ValuationModel.DIVIDEND_DISCOUNT,
                "residual-income": ValuationModel.RESIDUAL_INCOME,
                "nav": ValuationModel.NAV_SOTP,
                "property-nav": ValuationModel.NAV_SOTP,
                "resource-nav": ValuationModel.NAV_SOTP,
                "pipeline-rnpv": ValuationModel.NAV_SOTP,
            }[selected_model]
            message = str(exc)

            def runner(message=message):
                raise ValueError(message)

        runners[model] = runner
        requested_models = {model}
    run = ValuationExecutor().execute(
        selection=selection,
        runners=runners,
        requested_models=requested_models,
    )
    print(
        IndependentValuationModelsConsolePresenter().render(
            run, verbose=args.verbose or args.audit
        )
    )
    return 0


def _profile_model_runner(
    *,
    selected_model: str,
    profile: ForecastValuationProfile,
    context: IntrinsicValuationContext,
    annual,
    cost_of_equity: Decimal,
    terminal_growth: Decimal,
    terminal_roe_override: Decimal | None,
    args: argparse.Namespace,
):
    valuation = profile.valuation
    latest = {}
    by_year = {}
    for item in sorted(annual, key=lambda value: value.period_end):
        latest[item.concept] = item
        by_year.setdefault(item.fiscal_year, {})[item.concept] = item

    if selected_model in {"fcfe-dcf", "reit-affo"}:
        configured_fcfe = valuation.fcfe
        regulated_financial = False
        if selected_model == "reit-affo":
            configured = valuation.reit
            values = configured.affo_forecast
            if not values:
                values = (
                    ReitAffoAdapter.derive_affo(
                        ffo=configured.ffo,
                        recurring_adjustments=configured.recurring_affo_adjustments,
                    ),
                )
        else:
            values = tuple(args.fcfe or configured_fcfe.explicit_fcfe)
        terminal_roe = (
            terminal_roe_override
            or valuation.fcfe.terminal_return_on_equity
            or valuation.dividend_discount.terminal_return_on_equity
        )
        if terminal_roe is None:
            raise ValueError("FCFE DCF requires terminal ROE")
        retention = terminal_growth / terminal_roe
        if retention >= 1:
            raise ValueError("Terminal growth must be below terminal ROE")
        if values:
            terminal_income = values[-1] / (Decimal(1) - retention)
            periods = tuple(
                FcfeForecastPeriod(
                    label=f"Year {index}",
                    period=Decimal(index),
                    fcfe=value,
                    explicit_fcfe=True,
                )
                for index, value in enumerate(values, 1)
            )
        else:
            net_income = configured_fcfe.net_income
            required_equity = (
                configured_fcfe.required_common_equity_changes
                or valuation.financial_institution.required_common_equity_changes
            )
            if net_income and len(net_income) == len(required_equity):
                regulated_financial = True
                periods = tuple(
                    FcfeDcfService.regulated_period(
                        label=f"Year {index}",
                        period=Decimal(index),
                        net_income=income,
                        required_common_equity_change=required_equity[index - 1],
                    )
                    for index, income in enumerate(net_income, 1)
                )
            else:
                paths = (
                    net_income,
                    configured_fcfe.depreciation_and_amortization,
                    configured_fcfe.capital_expenditures,
                    configured_fcfe.working_capital_changes,
                )
                if not net_income or any(
                    len(path) != len(net_income) for path in paths
                ):
                    raise ValueError(
                        "FCFE requires an explicit path or complete net-income, D&A, "
                        "capex, and working-capital paths"
                    )
                borrowing = tuple(args.net_borrowing or configured_fcfe.net_borrowing)
                debt_ratio = (
                    args.debt_financing_ratio
                    if args.debt_financing_ratio is not None
                    else configured_fcfe.debt_financing_ratio
                )
                if not borrowing and debt_ratio is None:
                    raise ValueError(
                        "Corporate FCFE requires net borrowing or a debt-financing ratio"
                    )
                if borrowing and len(borrowing) != len(net_income):
                    raise ValueError(
                        "Net borrowing must have one value per FCFE period"
                    )
                built = []
                for index, income in enumerate(net_income, 1):
                    if borrowing:
                        da = paths[1][index - 1]
                        capex = paths[2][index - 1]
                        working_capital = paths[3][index - 1]
                        debt = borrowing[index - 1]
                        built.append(
                            FcfeForecastPeriod(
                                label=f"Year {index}",
                                period=Decimal(index),
                                net_income=income,
                                depreciation_and_amortization=da,
                                capital_expenditures=capex,
                                working_capital_change=working_capital,
                                net_borrowing=debt,
                                fcfe=income + da - capex - working_capital + debt,
                            )
                        )
                    else:
                        built.append(
                            FcfeDcfService.corporate_period(
                                label=f"Year {index}",
                                period=Decimal(index),
                                net_income=income,
                                depreciation_and_amortization=paths[1][index - 1],
                                capital_expenditures=paths[2][index - 1],
                                working_capital_change=paths[3][index - 1],
                                debt_financing_ratio=debt_ratio,
                            )
                        )
                periods = tuple(built)
            terminal_income = net_income[-1]

        def run_fcfe():
            result = FcfeDcfService().value(
                FcfeDcfInput(
                    context=context,
                    periods=periods,
                    cost_of_equity=cost_of_equity,
                    terminal_growth_rate=terminal_growth,
                    terminal_return_on_equity=terminal_roe,
                    terminal_net_income=terminal_income,
                    regulated_financial=regulated_financial,
                )
            )
            if selected_model == "reit-affo":
                return result.model_copy(update={"adapter": "REIT discounted AFFO"})
            return result

        return ValuationModel.EQUITY_DCF, run_fcfe

    if selected_model == "ddm":
        configured = valuation.dividend_discount
        mode = configured.mode
        dividends = tuple(args.dividend or configured.dividends)
        earnings = configured.earnings
        payout_path = tuple(args.payout_ratio or configured.payout_ratios)
        if not dividends and earnings and len(earnings) == len(payout_path):
            dividends = tuple(
                earning * payout
                for earning, payout in zip(earnings, payout_path, strict=True)
            )
        if not dividends:
            historical_dividends = [
                values[FinancialConcept.DIVIDENDS_PAID]
                for _, values in sorted(by_year.items())
                if FinancialConcept.DIVIDENDS_PAID in values
                and values[FinancialConcept.DIVIDENDS_PAID].value > 0
            ]
            if historical_dividends:
                dividends = (
                    historical_dividends[-1].value
                    * (Decimal(1) + terminal_growth / Decimal(100)),
                )
                mode = "gordon"
        if not dividends:
            raise ValueError("DDM requires dividends or earnings multiplied by payout")
        periods = tuple(
            DividendForecastPeriod(
                label=f"Year {index}",
                period=Decimal(index),
                earnings=(
                    earnings[index - 1] if len(earnings) == len(dividends) else None
                ),
                dividends=dividend,
                payout_ratio=(
                    payout_path[index - 1]
                    if len(payout_path) == len(dividends)
                    and len(earnings) == len(dividends)
                    and dividend == earnings[index - 1] * payout_path[index - 1]
                    else None
                ),
            )
            for index, dividend in enumerate(dividends, 1)
        )
        terminal_roe = terminal_roe_override or configured.terminal_return_on_equity
        payout = payout_path[-1] if payout_path else configured.terminal_payout_ratio
        if payout is None:
            historical_payouts = []
            for _, values in sorted(by_year.items()):
                income = values.get(FinancialConcept.NET_INCOME_COMMON) or values.get(
                    FinancialConcept.NET_INCOME
                )
                dividend = values.get(FinancialConcept.DIVIDENDS_PAID)
                if (
                    income is not None
                    and dividend is not None
                    and income.value > 0
                    and income.unit == dividend.unit
                ):
                    historical_payouts.append(dividend.value / income.value)
            if len(historical_payouts) >= 3:
                payout = sorted(historical_payouts[-3:])[1]

        def run_ddm():
            return DividendDiscountService().value(
                DividendDiscountInput(
                    context=context,
                    mode=mode,
                    cost_of_equity=cost_of_equity,
                    terminal_growth_rate=terminal_growth,
                    terminal_return_on_equity=terminal_roe,
                    terminal_payout_ratio=payout,
                    periods=periods if mode == "multistage" else (),
                    next_dividend=dividends[0] if mode == "gordon" else None,
                    terminal_earnings=earnings[-1] if earnings else None,
                )
            )

        return ValuationModel.DIVIDEND_DISCOUNT, run_ddm

    if selected_model == "residual-income":
        configured = valuation.residual_income
        book = configured.starting_book_value
        basis = configured.book_value_basis
        if book is None:
            common = latest.get(FinancialConcept.COMMON_EQUITY) or latest.get(
                FinancialConcept.STOCKHOLDERS_EQUITY
            )
            if common is None:
                raise ValueError("Residual income requires positive common equity")
            book = common.value
            if basis == "tangible_common_equity":
                book -= sum(
                    (
                        latest[concept].value
                        for concept in (
                            FinancialConcept.GOODWILL,
                            FinancialConcept.INTANGIBLE_ASSETS_NET,
                        )
                        if concept in latest and latest[concept].unit == common.unit
                    ),
                    Decimal(0),
                )
        roe_path = tuple(args.forecast_roe or configured.return_on_equity_path)
        payout_path = tuple(args.payout_ratio or configured.payout_ratio_path)
        if not roe_path or not payout_path:
            inferred = []
            for year in sorted(by_year):
                values = by_year[year]
                income = values.get(FinancialConcept.NET_INCOME_COMMON) or values.get(
                    FinancialConcept.NET_INCOME
                )
                equity = values.get(FinancialConcept.COMMON_EQUITY) or values.get(
                    FinancialConcept.STOCKHOLDERS_EQUITY
                )
                dividends = values.get(FinancialConcept.DIVIDENDS_PAID)
                if (
                    income is not None
                    and equity is not None
                    and dividends is not None
                    and income.value > 0
                    and equity.value > 0
                    and income.unit == equity.unit == dividends.unit
                ):
                    inferred.append(
                        (
                            year,
                            income.value / equity.value * Decimal(100),
                            dividends.value / income.value,
                        )
                    )
            normalized_roe, normalized_payout, _ = ResidualIncomeService.infer_policy(
                tuple(inferred[-3:])
            )
            roe_path = roe_path or (normalized_roe,) * 5
            payout_path = payout_path or (normalized_payout,) * len(roe_path)
        persistence = (
            args.excess_return_persistence
            if args.excess_return_persistence is not None
            else configured.excess_return_persistence
        )

        def run_residual():
            return ResidualIncomeService().value(
                ResidualIncomeInput(
                    context=context,
                    starting_book_value=book,
                    book_value_basis=basis,
                    return_on_equity_path=roe_path,
                    payout_ratio_path=payout_path,
                    cost_of_equity=cost_of_equity,
                    excess_return_persistence=persistence,
                )
            )

        return ValuationModel.RESIDUAL_INCOME, run_residual

    components = valuation.sotp.components
    adjustments = valuation.sotp.adjustments
    adapter = "generic SOTP"
    if selected_model == "property-nav":
        components = PropertyNavAdapter().to_components(
            valuation.reit.properties, valuation_date=context.valuation_date
        )
        adapter = "property NOI / cap-rate NAV"
    elif selected_model == "resource-nav":
        scenario_components = ResourceNavAdapter().to_scenarios(
            valuation.resources.projects, valuation_date=context.valuation_date
        )

        def run_resource_scenarios():
            return tuple(
                SotpValuationService().value(
                    SotpValuationInput(
                        context=context,
                        components=scenario_values,
                        adjustments=adjustments,
                        holding_company_discount=(
                            valuation.sotp.holding_company_discount
                        ),
                        adapter=f"finite resource project NAV [{scenario}]",
                    )
                )
                for scenario, scenario_values in scenario_components.items()
            )

        return ValuationModel.NAV_SOTP, run_resource_scenarios
    elif selected_model == "pipeline-rnpv":
        components = PipelineRnpvAdapter().to_components(
            valuation.pipelines.projects, valuation_date=context.valuation_date
        )
        adapter = "pipeline probability-adjusted rNPV"
    elif selected_model != "nav":
        raise ValueError(f"Unsupported intrinsic model: {selected_model}")

    def run_sotp():
        return SotpValuationService().value(
            SotpValuationInput(
                context=context,
                components=components,
                adjustments=adjustments,
                holding_company_discount=valuation.sotp.holding_company_discount,
                adapter=adapter,
            )
        )

    return ValuationModel.NAV_SOTP, run_sotp


def _equity_relative_valuation(
    *,
    basis: RelativeValuationBasis,
    report,
    profile: ForecastValuationProfile,
    intrinsic,
    valuation_date: datetime.date,
    horizon_years: Decimal,
    discount_rate: Decimal,
):
    fundamentals = report.target.fundamentals
    metric = None
    label = ""
    if basis == RelativeValuationBasis.PE:
        earnings = profile.valuation.dividend_discount.earnings
        metric = (
            earnings[0]
            if earnings
            else (
                fundamentals.net_income_common
                if fundamentals.net_income_common is not None
                else fundamentals.net_income
            )
        )
        label = (
            "forward common earnings" if earnings else "LTM common earnings fallback"
        )
    elif basis == RelativeValuationBasis.PRICE_TO_BOOK:
        metric = (
            profile.valuation.residual_income.starting_book_value
            or fundamentals.book_equity
        )
        label = "common book equity"
    elif basis == RelativeValuationBasis.PRICE_TO_TANGIBLE_BOOK:
        metric = fundamentals.tangible_book_equity
        label = "tangible common equity"
    elif basis == RelativeValuationBasis.PRICE_TO_AFFO:
        affo = profile.valuation.reit.affo_forecast
        if affo:
            metric = affo[0]
            label = "forward AFFO"
        else:
            metric = ReitAffoAdapter.derive_affo(
                ffo=profile.valuation.reit.ffo,
                recurring_adjustments=(
                    profile.valuation.reit.recurring_affo_adjustments
                ),
            )
            label = "current AFFO"
    elif basis == RelativeValuationBasis.PRICE_TO_NAV:
        components = profile.valuation.sotp.components
        if not components and profile.valuation.reit.properties:
            components = PropertyNavAdapter().to_components(
                profile.valuation.reit.properties,
                valuation_date=valuation_date,
            )
        nav = SotpValuationService().value(
            SotpValuationInput(
                context=IntrinsicValuationContext(
                    company_id=report.target.company_id,
                    company_name=report.target.company_name,
                    ticker=report.target.ticker,
                    valuation_date=valuation_date,
                    currency=report.target.currency,
                    diluted_shares=report.target.fundamentals.shares,
                ),
                components=components,
                adjustments=profile.valuation.sotp.adjustments,
                holding_company_discount=(
                    profile.valuation.sotp.holding_company_discount
                ),
            )
        )
        metric = nav.equity_value
        label = "profile NAV"
    if metric is None or metric <= 0:
        raise ValueError(f"positive target denominator unavailable for {basis.value}")
    summary = next((item for item in report.summaries if item.basis == basis), None)
    if summary is None:
        raise ValueError(f"peer denominator unavailable for {basis.value}")
    policy = profile.relative_valuation.multiple_resolution
    if summary.sample_size < policy.minimum_peer_sample:
        raise ValueError(
            f"only {summary.sample_size} usable peer denominators; "
            f"policy requires {policy.minimum_peer_sample}"
        )
    fundamental = intrinsic.equity_value / metric
    point = summary.median
    lower = summary.percentile_25 or summary.minimum
    upper = summary.percentile_75 or summary.maximum
    confidence = (
        MultipleConfidence.HIGH
        if summary.sample_size >= 8
        else MultipleConfidence.MEDIUM
        if summary.sample_size >= policy.minimum_peer_sample
        else MultipleConfidence.LOW
    )
    market_cap = report.target.market_capitalization
    current_anchor = market_cap / metric if market_cap is not None else None
    resolved_multiple = ResolvedMultiple(
        basis=basis,
        point_estimate=point,
        lower_bound=max(Decimal("0.01"), lower),
        upper_bound=upper,
        fundamental_anchor=fundamental,
        peer_anchor=summary.median,
        peer_anchor_source="current peer denominator evidence",
        peer_anchor_percentile_25=summary.percentile_25,
        peer_anchor_percentile_75=summary.percentile_75,
        current_target_anchor=current_anchor,
        market_anchor=summary.median,
        observed_premium=(
            current_anchor / summary.median - Decimal(1)
            if current_anchor is not None
            else None
        ),
        resolved_premium=Decimal(0),
        historical_persistence=Decimal(0),
        fundamental_support=Decimal(1),
        horizon_retention=Decimal(1),
        persistence_factor=Decimal(0),
        sample_size=summary.sample_size,
        peer_confidence=confidence,
        confidence=confidence,
        methodology=(
            "Independent peer median with peer percentile bounds; the DCF-implied "
            "equity multiple is retained as a diagnostic only"
        ),
        warnings=(
            ("Target denominator uses current/LTM fallback rather than a forecast",)
            if "fallback" in label
            else ()
        ),
    )
    target_date = valuation_date + datetime.timedelta(
        days=int(horizon_years * Decimal(365))
    )
    return ProviderNeutralRelativeValuationService().value(
        valuation_date=valuation_date,
        horizon_years=horizon_years,
        metric=ForwardValuationMetric(
            basis=basis,
            amount=metric,
            label=label,
            target_date=target_date,
            currency=report.target.currency,
            numerator_basis=RelativeNumeratorBasis.EQUITY_VALUE,
        ),
        diluted_shares=report.target.fundamentals.shares,
        discount_rate=discount_rate,
        resolved_multiple=resolved_multiple,
        current_price=report.target.price,
    )


async def _run_valuation(args: argparse.Namespace) -> int:
    generated_profile_path = None
    should_generate_profile = False
    peer_report = None
    decision_result = None
    additional_warnings: list[str] = []
    if args.ticker:
        profile, generated_profile_path, should_generate_profile = (
            ValuationProfileLoader.load_for_ticker(args.ticker, args.profile)
        )
    else:
        profile = ValuationProfileLoader.load(args.profile)
    selected_model = args.model
    if selected_model == "auto":
        archetype = profile.model_selection.business_archetype
        if archetype is None:
            try:
                classification = await _retrieve_classification(
                    args, provider=None, crosscheck=False
                )
                sector = profile.model_selection.sector or classification.sector
                industry = profile.model_selection.industry or classification.industry
                archetype = ValuationProfileBuilder._archetype(
                    sector, ValuationProfileBuilder._key(industry)
                )
            except (RuntimeError, ValueError):
                archetype = None
        if archetype is not None and profile.model_selection.business_archetype is None:
            profile = profile.model_copy(
                update={
                    "model_selection": profile.model_selection.model_copy(
                        update={"business_archetype": archetype}
                    )
                }
            )
        selected_model = {
            BusinessArchetype.FINANCIAL_INTERMEDIARY: "auto-specialized",
            BusinessArchetype.REIT_PROPERTY: "auto-specialized",
            BusinessArchetype.RESOURCE_PRODUCER: "auto-specialized",
            BusinessArchetype.PROJECT_PIPELINE: "auto-specialized",
            BusinessArchetype.HOLDING_COMPANY: "auto-specialized",
            BusinessArchetype.CONGLOMERATE: "auto-specialized",
        }.get(
            archetype,
            "both" if profile.relative_valuation.enabled else "fcff-dcf",
        )
    if getattr(args, "excel_output", None) is not None and selected_model not in {
        "fcff-dcf",
        "comparables",
        "both",
    }:
        raise ValueError(
            "Excel valuation export supports FCFF DCF and relative valuation; the selected "
            f"model ({selected_model}) does not produce an FCFF DCF result"
        )
    if selected_model not in {"fcff-dcf", "comparables", "both"}:
        return await _run_profile_intrinsic_valuation(args, profile, selected_model)
    forecast_parameters = _fcff_parameters(args, profile.forecast.fcff)
    terminal_configuration = profile.valuation.terminal_value
    terminal_method = (
        TerminalValueMethod(args.terminal_method)
        if args.terminal_method is not None
        else terminal_configuration.method
    )
    if (
        getattr(args, "excel_output", None) is not None
        and selected_model == "both"
        and terminal_method != TerminalValueMethod.PERPETUITY_GROWTH
    ):
        raise ValueError(
            "Excel export with --model both requires a perpetuity-growth terminal "
            "method because relative valuation requires a perpetuity-growth DCF "
            "anchor"
        )
    cash_flow_timing = (
        CashFlowTiming(args.cash_flow_timing)
        if args.cash_flow_timing is not None
        else profile.valuation.cash_flow_timing
    )
    forecast_service = FcffForecastService()
    bridge_resolver = FcffDcfCapitalBridgeResolver()
    required_concepts = (
        forecast_service.required_concepts()
        | bridge_resolver.required_concepts()
        | ValuationProfileBuilder.required_concepts()
        | {
            FinancialConcept.INTEREST_EXPENSE,
            FinancialConcept.STOCKHOLDERS_EQUITY,
        }
    )
    financials = await _retrieve_financials(
        args,
        None,
        required_concepts,
    )
    valuation_date = datetime.date.today()
    additional_warnings.extend(_financial_snapshot_warnings(financials, args))
    forecast = forecast_service.forecast(
        financials,
        forecast_parameters,
        as_of=valuation_date,
        availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
    )
    guidance_overlay: GuidanceOverlayResult | None = None
    if OPENAI_API_KEY:
        original_forecast_parameters = forecast_parameters
        try:
            (
                candidate_parameters,
                candidate_overlay,
            ) = await _management_guidance_overlay(
                args,
                financials,
                original_forecast_parameters,
                forecast,
                valuation_date,
            )
            additional_warnings.extend(candidate_overlay.warnings)
            candidate_forecast = forecast
            if candidate_overlay.applications:
                candidate_forecast = forecast_service.forecast(
                    financials,
                    candidate_parameters,
                    as_of=valuation_date,
                    availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
                )
            forecast_parameters = candidate_parameters
            guidance_overlay = candidate_overlay
            forecast = candidate_forecast
        except Exception as exc:
            forecast_parameters = original_forecast_parameters
            additional_warnings.append(
                "Management-guidance extraction unavailable; historical forecast "
                f"retained ({exc})"
            )
            guidance_overlay = None
    elif args.verbose or args.audit:
        additional_warnings.append(
            "AI management-guidance extraction skipped because OpenAI is not configured"
        )
    seed_forecast = forecast
    bridge_configuration = profile.valuation.capital_bridge
    has_cli_debt_bridge = any(
        value is not None for value in (args.net_debt, args.gross_debt, args.cash)
    )
    capital_bridge = bridge_resolver.resolve(
        financials,
        fiscal_year=forecast.base_fiscal_year,
        period_end=forecast.base_period_end,
        unit=forecast.unit,
        net_debt=(
            args.net_debt if has_cli_debt_bridge else bridge_configuration.net_debt
        ),
        gross_debt=(
            args.gross_debt if has_cli_debt_bridge else bridge_configuration.gross_debt
        ),
        cash_and_equivalents=(
            args.cash
            if has_cli_debt_bridge
            else bridge_configuration.cash_and_equivalents
        ),
        diluted_shares=(
            args.shares
            if args.shares is not None
            else bridge_configuration.diluted_shares
        ),
        non_operating_assets=(
            args.non_operating_assets
            if args.non_operating_assets is not None
            else bridge_configuration.non_operating_assets
        ),
        valuation_date=valuation_date,
        availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
    )
    discount_configuration = profile.valuation.discount_rates
    needs_automatic_wacc = (
        args.wacc is None
        and discount_configuration.wacc is None
        and not discount_configuration.can_calculate_wacc
    )
    needs_automatic_terminal = (
        terminal_method == TerminalValueMethod.PERPETUITY_GROWTH
        and args.terminal_growth is None
        and terminal_configuration.perpetual_growth_rate is None
    )
    automatic_inputs = await _retrieve_automatic_assumption_inputs(
        args,
        financials,
        forecast.unit,
        needs_wacc=needs_automatic_wacc,
        needs_terminal=needs_automatic_terminal,
        sector_override=profile.model_selection.sector,
        industry_override=profile.model_selection.industry,
    )
    resolved = ValuationAssumptionResolver().resolve(
        financials=financials,
        capital_bridge=capital_bridge,
        discount_configuration=discount_configuration,
        terminal_configuration=terminal_configuration,
        terminal_is_perpetuity=(
            terminal_method == TerminalValueMethod.PERPETUITY_GROWTH
        ),
        valuation_date=valuation_date,
        wacc_override=args.wacc,
        terminal_growth_override=args.terminal_growth,
        **automatic_inputs,
    )
    profile_context = ValuationProfileBuilder().build(
        financials,
        automatic_inputs.get("classification"),
        ValuationProfileOverrides(
            sector=profile.model_selection.sector,
            industry=profile.model_selection.industry,
            business_archetype=profile.model_selection.business_archetype,
            financial_institution_kind=profile.model_selection.financial_institution_kind,
            actuarial_detail_supplied=(
                profile.valuation.financial_institution.actuarial_detail_supplied
            ),
            regulatory_capital_constraints_supplied=bool(
                profile.valuation.financial_institution.regulatory_capital_constraints
            ),
            lifecycle=profile.model_selection.lifecycle,
            cyclicality=profile.model_selection.cyclicality,
            economic_traits=set(profile.model_selection.economic_traits),
        ),
    )
    comparable_bundle = None
    comparable_error = None
    relative_result = None
    provider_relative_result = None
    if selected_model in {"comparables", "both"}:
        provider_symbols = _parse_provider_symbols(args.provider_symbol)
        fallback_symbol = args.ticker or financials.ticker
        if fallback_symbol is None:
            comparable_error = (
                "Automatic peer discovery requires a ticker or a provider symbol"
            )
        else:
            target_symbol = (
                provider_symbols.get(ProviderName.YAHOO, fallback_symbol)
                .strip()
                .upper()
            )
            peer_symbols, peer_source = _resolve_comparable_peer_symbols(
                args,
                profile,
                target_symbol,
            )
            try:
                comparable_bundle = await _build_comparable_report(
                    args,
                    profile,
                    target_symbol,
                    peer_symbols,
                    peer_source=peer_source,
                    as_of=valuation_date,
                    availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
                )
            except (RuntimeError, ValueError) as exc:
                comparable_error = str(exc)

    peer_roics = ()
    if comparable_bundle is not None:
        peer_roics = comparable_bundle.reliable_peer_roics
    configured_terminal_roic = (
        args.terminal_roic
        if args.terminal_roic is not None
        else profile.valuation.multistage.terminal_return_on_invested_capital
    )
    terminal_roic = TerminalRoicResolver().resolve(
        financials,
        wacc=resolved.wacc,
        terminal_growth=(
            resolved.perpetual_growth_rate
            if resolved.perpetual_growth_rate is not None
            else profile.valuation.multistage.stable_growth_rate or Decimal(0)
        ),
        valuation_date=valuation_date,
        currency=forecast.unit,
        explicit_roic=configured_terminal_roic,
        explicit_source=(
            "explicit CLI override"
            if args.terminal_roic is not None
            else "explicit valuation profile"
        ),
        lifecycle=profile_context.lifecycle,
        cyclicality=profile_context.cyclicality,
        peer_roics=peer_roics,
    )
    if should_generate_profile and args.ticker and generated_profile_path is not None:
        discovered_peers = (
            comparable_bundle.report.universe.selected_tickers
            if comparable_bundle is not None
            and comparable_bundle.report.universe.discovery_confidence != "low"
            else ()
        )
        profile, generated_profile_path, created = (
            ValuationProfileLoader.create_generated(
                ticker=args.ticker,
                base_profile=profile,
                inferred_profile=profile_context,
                terminal_roic=terminal_roic.value,
                terminal_roic_confidence=terminal_roic.confidence,
                generated_on=valuation_date,
                peers=discovered_peers,
                path=generated_profile_path,
            )
        )
        should_generate_profile = False
        if created:
            print(f"Generated valuation profile: {generated_profile_path.resolve()}")
    resolved = replace(
        resolved,
        assumption_set=resolved.assumption_set.model_copy(
            update={
                "assumptions": (
                    *resolved.assumption_set.assumptions,
                    terminal_roic.assumption,
                )
            }
        ),
    )
    multistage_plan = None
    multistage_configuration = profile.valuation.multistage.model_copy(
        update={
            "terminal_return_on_invested_capital": terminal_roic.value,
        }
    )
    use_multistage = args.projection_method == "adaptive" or (
        args.projection_method is None
        and multistage_configuration.enabled
        and (
            terminal_method == TerminalValueMethod.PERPETUITY_GROWTH
            or multistage_configuration.stable_growth_rate is not None
        )
    )
    tax_assumption = resolved.assumption_set.find(
        ValuationAssumptionKind.NORMALIZED_TAX_RATE
    )
    if use_multistage:
        stable_growth_rate = (
            resolved.perpetual_growth_rate
            if terminal_method == TerminalValueMethod.PERPETUITY_GROWTH
            else multistage_configuration.stable_growth_rate
        )
        if stable_growth_rate is None:
            raise ValueError(
                "Adaptive multistage projection with an exit multiple requires "
                "valuation.multistage.stable_growth_rate in the profile"
            )
        forecast, multistage_plan = AdaptiveMultistageFcffForecastService(
            forecast_service
        ).forecast(
            financials,
            forecast,
            forecast_parameters,
            stable_growth_rate,
            multistage_configuration,
            normalized_tax_rate=(
                tax_assumption.value if tax_assumption is not None else None
            ),
            forward_evidence=_forward_growth_evidence(
                profile_context.lifecycle,
                profile_context.economic_traits,
                guidance_overlay,
            ),
            as_of=valuation_date,
            availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
        )
        multistage_plan = multistage_plan.model_copy(
            update={
                "terminal_roic_source": terminal_roic.source,
                "terminal_roic_methodology": terminal_roic.methodology,
                "terminal_roic_confidence": terminal_roic.confidence,
                "terminal_roic_warnings": terminal_roic.warnings,
            }
        )
    parameters = FcffDcfParameters(
        wacc=resolved.wacc,
        wacc_source=resolved.wacc_source,
        cash_flow_timing=cash_flow_timing,
        terminal_method=terminal_method,
        perpetual_growth_rate=resolved.perpetual_growth_rate,
        perpetual_growth_source=resolved.perpetual_growth_source,
        exit_multiple=(
            (
                args.exit_multiple
                if args.exit_multiple is not None
                else terminal_configuration.exit_multiple
            )
            if terminal_method == TerminalValueMethod.EXIT_MULTIPLE
            else None
        ),
        exit_metric=(
            TerminalMetric(args.exit_metric)
            if args.exit_metric is not None
            else terminal_configuration.exit_metric
        ),
    )
    repurchase_configuration = profile.valuation.share_repurchases
    repurchase_cash = (
        tuple(args.buyback_cash)
        if args.buyback_cash is not None
        else repurchase_configuration.annual_cash_amounts
    )
    share_repurchase_parameters = None
    if not args.no_buybacks and repurchase_cash:
        share_repurchase_parameters = ShareRepurchaseParameters(
            annual_cash_amounts=repurchase_cash,
            initial_purchase_price=(
                args.buyback_price
                if args.buyback_price is not None
                else repurchase_configuration.initial_purchase_price
            ),
            price_growth_rate=(
                args.buyback_price_growth
                if args.buyback_price_growth is not None
                else repurchase_configuration.price_growth_rate
            ),
            discount_rate=(
                args.buyback_discount_rate
                if args.buyback_discount_rate is not None
                else repurchase_configuration.discount_rate
            ),
            source=(
                "CLI override"
                if args.buyback_cash is not None
                else repurchase_configuration.source or "valuation profile"
            ),
        )
    elif not args.no_buybacks and any(
        value is not None
        for value in (
            args.buyback_price,
            args.buyback_price_growth,
            args.buyback_discount_rate,
        )
    ):
        raise ValueError(
            "Buyback price or rate assumptions require --buyback-cash or a "
            "profile repurchase schedule"
        )
    result = FcffDcfService().value(
        forecast,
        parameters,
        capital_bridge,
        resolved.assumption_set,
        multistage_plan,
        valuation_date,
        share_repurchase_parameters,
    )
    if terminal_roic.warnings:
        result = result.model_copy(
            update={
                "warnings": tuple(
                    dict.fromkeys([*result.warnings, *terminal_roic.warnings])
                )
            }
        )
    if selected_model == "comparables":
        additional_warnings.extend(result.warnings)
    if selected_model in {"comparables", "both"}:
        if terminal_method != TerminalValueMethod.PERPETUITY_GROWTH:
            raise ValueError(
                "Relative multiple resolution requires a perpetuity-growth DCF "
                "for its independent fundamental anchor"
            )
        relative_configuration = profile.relative_valuation
        basis = RelativeValuationBasis(
            args.relative_basis or relative_configuration.basis
        )
        horizon_years = (
            args.horizon_years
            if args.horizon_years is not None
            else relative_configuration.horizon_years
        )
        if horizon_years <= 0:
            raise ValueError("--horizon-years must be positive")
        if comparable_bundle is None:
            additional_warnings.append(
                "Relative valuation skipped: automatic peer evidence could not be "
                f"prepared ({comparable_error or 'unknown provider failure'})"
            )
        else:
            report = comparable_bundle.report
            comparable_financials = comparable_bundle.target_financials
            comparable_market = comparable_bundle.target_market
            comparable_peer_sources = comparable_bundle.peer_sources
            if basis not in EQUITY_RELATIVE_BASES:
                report = ForwardPeerMultiplesService().build(
                    report,
                    {
                        symbol: financials
                        for symbol, (
                            financials,
                            _market,
                        ) in comparable_peer_sources.items()
                    },
                    basis,
                    valuation_date,
                    horizon_years,
                )
            peer_report = report
            relative_ready = True
            if report.universe.discovery_confidence == "low":
                additional_warnings.append(
                    "Relative valuation skipped: selected peer evidence has low "
                    "economic-comparability confidence"
                )
                relative_ready = False
            elif len(report.universe.selected_tickers) < (
                relative_configuration.multiple_resolution.minimum_peer_sample
            ):
                additional_warnings.append(
                    "Relative valuation skipped: peer evidence is below the "
                    f"configured minimum sample of "
                    f"{relative_configuration.multiple_resolution.minimum_peer_sample}"
                )
                relative_ready = False
            if relative_ready and basis in EQUITY_RELATIVE_BASES:
                try:
                    provider_relative_result = _equity_relative_valuation(
                        basis=basis,
                        report=report,
                        profile=profile,
                        intrinsic=result,
                        valuation_date=valuation_date,
                        horizon_years=horizon_years,
                        discount_rate=(
                            resolved.assumption_set.find(
                                ValuationAssumptionKind.COST_OF_EQUITY
                            ).value
                            if resolved.assumption_set.find(
                                ValuationAssumptionKind.COST_OF_EQUITY
                            )
                            is not None
                            else resolved.wacc
                        ),
                    )
                except ValueError as exc:
                    additional_warnings.append(f"Relative valuation skipped: {exc}")
            if relative_ready and basis not in EQUITY_RELATIVE_BASES:
                target_history = HistoricalMultiplesService().compute(
                    comparable_financials,
                    comparable_market,
                    basis,
                )
                peer_histories = tuple(
                    HistoricalMultiplesService().compute(financials, market, basis)
                    for financials, market in comparable_peer_sources.values()
                )
                resolved_multiple = MultipleResolver().resolve(
                    basis=basis,
                    target=report.target,
                    target_history=target_history,
                    peer_histories=peer_histories,
                    peer_report=report,
                    target_forecast=forecast,
                    intrinsic_valuation=result,
                    horizon_years=horizon_years,
                    policy=relative_configuration.multiple_resolution,
                )
                if report.warnings:
                    resolved_multiple = resolved_multiple.model_copy(
                        update={
                            "warnings": tuple(
                                dict.fromkeys(
                                    [*resolved_multiple.warnings, *report.warnings]
                                )
                            )
                        }
                    )
                relative_result = ComparableImpliedValuationService().value(
                    target_forecast=forecast,
                    capital_bridge=capital_bridge,
                    projected_shares=capital_bridge.diluted_shares,
                    resolved_multiple=resolved_multiple,
                    valuation_date=valuation_date,
                    horizon_years=horizon_years,
                    discount_rate=resolved.wacc,
                    current_price=report.target.price,
                    analyst_target_price=args.analyst_target_price,
                    intrinsic_value_per_share=result.value_per_share,
                )
    if getattr(args, "excel_output", None) is not None:
        output = ValuationExcelRenderer().render(
            forecast,
            result,
            args.excel_output,
            relative=relative_result or provider_relative_result,
            peer_report=peer_report,
            discount_timing_basis="calendar",
        )
        print(f"Exported valuation Excel workbook to {output}")
    current_price = (
        relative_result.current_price if relative_result is not None else None
    )
    if current_price is None and provider_relative_result is not None:
        current_price = provider_relative_result.current_price
    if current_price is None and comparable_bundle is not None:
        current_price = comparable_bundle.report.target.price
    market_data = automatic_inputs.get("market_data")
    if current_price is None and market_data is not None:
        latest_price = market_data.latest_price
        current_price = latest_price.close if latest_price is not None else None
    decision_configuration = profile.valuation.decision_analysis
    if current_price is not None and decision_configuration.enabled:
        decision_context = IntrinsicDecisionContext(
            financials=financials,
            requested_parameters=forecast_parameters,
            seed_forecast=seed_forecast,
            base_forecast=forecast,
            base_result=result,
            capital_bridge=capital_bridge,
            terminal_roic=terminal_roic.value,
            multistage_configuration=multistage_configuration,
            use_multistage=use_multistage,
            valuation_date=valuation_date,
            availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
            normalized_tax_rate=(
                tax_assumption.value if tax_assumption is not None else None
            ),
            share_repurchase_parameters=share_repurchase_parameters,
            forward_evidence=_forward_growth_evidence(
                profile_context.lifecycle,
                profile_context.economic_traits,
                guidance_overlay,
            ),
            flexible_revenue_growth=(
                args.revenue_growth is None
                and profile.forecast.fcff.revenue_growth is None
                and not (
                    guidance_overlay
                    and any(
                        item.driver == "revenue_growth"
                        for item in guidance_overlay.applications
                    )
                )
            ),
            flexible_operating_margin=(
                args.operating_margin is None
                and profile.forecast.fcff.operating_margin is None
                and not (
                    guidance_overlay
                    and any(
                        item.driver == "operating_margin"
                        for item in guidance_overlay.applications
                    )
                )
            ),
            flexible_terminal_roic=configured_terminal_roic is None,
            flexible_wacc=(args.wacc is None and discount_configuration.wacc is None),
            flexible_terminal_growth=(
                args.terminal_growth is None
                and terminal_configuration.perpetual_growth_rate is None
            ),
        )
        try:
            decision_policy = DecisionScenarioPolicy(
                revenue_growth_delta=decision_configuration.revenue_growth_delta,
                operating_margin_delta=decision_configuration.operating_margin_delta,
                bear_wacc_delta=decision_configuration.bear_wacc_delta,
                bull_wacc_delta=decision_configuration.bull_wacc_delta,
                terminal_growth_delta=decision_configuration.terminal_growth_delta,
                terminal_roic_spread_change=(
                    decision_configuration.terminal_roic_spread_change
                ),
                fair_value_band=decision_configuration.fair_value_band,
                sensitivity_size=decision_configuration.sensitivity_size,
            )
            decision_result = DecisionValuationService(decision_policy).build(
                decision_context,
                current_price,
                relative_result or provider_relative_result,
            )
        except ValueError as exc:
            additional_warnings.append(f"Decision analysis unavailable: {exc}")
    elif current_price is None and decision_configuration.enabled:
        additional_warnings.append(
            "Decision analysis skipped: no current market price was available"
        )
    report_output = ValuationReportConsolePresenter().render(
        intrinsic=result if selected_model in {"fcff-dcf", "both"} else None,
        peer_report=peer_report,
        relative=relative_result,
        provider_relative=provider_relative_result,
        decision=decision_result,
        profile_name=profile.name,
        show_scenarios=args.scenarios,
        show_sensitivity=args.sensitivity,
        show_reverse_dcf=args.reverse_dcf,
        verbose=args.verbose or args.audit,
        additional_warnings=tuple(additional_warnings),
        management_guidance=guidance_overlay,
    )
    if report_output:
        print(report_output)
    return 0


async def _run_valuation_models(args: argparse.Namespace) -> int:
    valuation_profile = _load_selected_valuation_profile(args)
    configuration = valuation_profile.model_selection
    financials = await _retrieve_financials(
        args,
        Granularity.ANNUAL,
        ValuationProfileBuilder.required_concepts(),
    )
    classification = await _retrieve_classification(
        args,
        provider=(
            ProviderName(args.classification_provider)
            if args.classification_provider
            else None
        ),
        crosscheck=False,
    )
    overrides = ValuationProfileOverrides(
        sector=configuration.sector,
        industry=configuration.industry,
        business_archetype=(
            BusinessArchetype(args.business_type)
            if args.business_type
            else configuration.business_archetype
        ),
        financial_institution_kind=configuration.financial_institution_kind,
        actuarial_detail_supplied=(
            valuation_profile.valuation.financial_institution.actuarial_detail_supplied
        ),
        regulatory_capital_constraints_supplied=bool(
            valuation_profile.valuation.financial_institution.regulatory_capital_constraints
        ),
        lifecycle=(
            CompanyLifecycle(args.lifecycle)
            if args.lifecycle
            else configuration.lifecycle
        ),
        cyclicality=(
            Cyclicality(args.cyclicality)
            if args.cyclicality
            else configuration.cyclicality
        ),
        economic_traits=(
            {EconomicTrait(value) for value in args.trait}
            if args.trait is not None
            else set(configuration.economic_traits)
        ),
        available_inputs=(
            {
                *valuation_profile.configured_valuation_inputs,
                *(ValuationInput(value) for value in args.available_input),
            }
            if args.available_input is not None
            else set(valuation_profile.configured_valuation_inputs)
        ),
        peer_count=(
            args.peer_count if args.peer_count is not None else configuration.peer_count
        ),
    )
    profile = ValuationProfileBuilder().build(financials, classification, overrides)
    selection = ValuationModelSelector().select(profile)
    print(ValuationSelectionConsolePresenter().render(selection))
    return 0


async def _run_classification(args: argparse.Namespace) -> int:
    classification = await _retrieve_classification(
        args,
        provider=ProviderName(args.provider) if args.provider else None,
        crosscheck=args.crosscheck,
    )
    print(ClassificationConsolePresenter().render(classification))
    return 0


async def _run_specialized_inputs(args: argparse.Namespace) -> int:
    configuration = _load_selected_valuation_profile(args).specialized_inputs
    history = args.history if args.history is not None else configuration.history
    if history < 1:
        raise ValueError("--history must be at least 1")
    if not args.user_agent:
        raise ValueError("SEC retrieval requires EDGARITO_USER_AGENT / user_agent")
    async with EdgarClient(
        FileSystemCache(Path(args.cache_dir)), args.user_agent
    ) as client:
        cik = args.cik
        if cik is None:
            cik = await client.get_cik(
                args.ticker,
                use_cache=not args.refresh,
                make_cache=True,
            )
        facts = await client.get_company_facts(
            cik,
            use_cache=not args.refresh,
            make_cache=True,
        )
    extraction = SpecializedValuationExtractor().extract(
        facts,
        SpecializedInputType(args.type),
        ticker=args.ticker,
        historical_periods=history,
    )
    print(SpecializedExtractionConsolePresenter().render(extraction))
    return 0


async def _retrieve_classification(
    args: argparse.Namespace,
    *,
    provider: Optional[ProviderName],
    crosscheck: bool,
):
    async with CompanyClassificationService(
        cache=FileSystemCache(Path(args.cache_dir)),
        provider_configuration=CLASSIFICATION_PROVIDER_CONFIGURATION,
        alphavantage_api_key=ALPHAVANTAGE_API_KEY,
        fmp_api_key=FMP_API_KEY,
        openfigi_api_key=OPENFIGI_API_KEY,
    ) as service:
        classification = await service.retrieve(
            ticker=args.ticker,
            cik=args.cik,
            isin=args.isin,
            exchange=args.exchange,
            exchange_symbols=_parse_mappings(args.exchange_symbol, "--exchange-symbol"),
            provider_symbols=_parse_provider_symbols(args.provider_symbol),
            provider=provider,
            use_cache=not args.refresh,
            make_cache=True,
            crosscheck=crosscheck,
        )
    return classification


async def _retrieve_financials(
    args: argparse.Namespace,
    granularity: Optional[Granularity],
    concepts: Optional[set[FinancialConcept]],
) -> NormalizedCompanyFinancials:
    cache = FileSystemCache(Path(args.cache_dir))
    async with FinancialDataService(
        cache=cache,
        provider_configuration=PROVIDER_CONFIGURATION,
        user_agent=args.user_agent,
        alphavantage_api_key=ALPHAVANTAGE_API_KEY,
        fmp_api_key=FMP_API_KEY,
        openfigi_api_key=OPENFIGI_API_KEY,
    ) as service:
        return await service.retrieve(
            ticker=args.ticker,
            cik=args.cik,
            isin=args.isin,
            exchange=args.exchange,
            exchange_symbols=_parse_mappings(args.exchange_symbol, "--exchange-symbol"),
            provider_symbols=_parse_provider_symbols(args.provider_symbol),
            market=Market(args.market),
            provider=ProviderName(args.provider) if args.provider else None,
            granularity=granularity,
            concepts=concepts,
            use_cache=not args.refresh,
            make_cache=True,
            crosscheck=args.crosscheck,
        )


def _financial_snapshot_warnings(
    financials: NormalizedCompanyFinancials,
    args: argparse.Namespace,
) -> list[str]:
    """Surface current-snapshot freshness without changing generic cache policy."""
    if financials.provider.casefold() != "yahoo":
        return []
    max_age_hours = getattr(args, "financial_snapshot_max_age_hours", 24)
    if max_age_hours <= 0:
        raise ValueError("--financial-snapshot-max-age-hours must be positive")
    retrieved_at = financials.retrieved_at
    if retrieved_at is None:
        return [
            "Yahoo financial snapshot retrieval time is unavailable; cache freshness "
            "cannot be established. Use --refresh for a current snapshot"
        ]
    age = datetime.datetime.now(datetime.timezone.utc) - retrieved_at.astimezone(
        datetime.timezone.utc
    )
    if age < datetime.timedelta(0):
        return [
            "Yahoo financial snapshot retrieval time is in the future; provenance "
            "should be checked"
        ]
    threshold = datetime.timedelta(hours=max_age_hours)
    if age <= threshold:
        return []
    age_hours = int(age.total_seconds() // 3600)
    return [
        f"Yahoo financial snapshot is stale ({age_hours} hours old; retrieved "
        f"{retrieved_at.astimezone(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC). "
        "Use --refresh for a current snapshot"
    ]


def _granularity(period: str) -> Optional[Granularity]:
    return None if period == "all" else Granularity(period)


def _fcff_parameters(args: argparse.Namespace, configured) -> FcffForecastParameters:
    return FcffForecastParameters(
        forecast_years=(
            args.years if args.years is not None else configured.forecast_years
        ),
        revenue_growth=(
            args.revenue_growth
            if args.revenue_growth is not None
            else configured.revenue_growth
        ),
        operating_margin=(
            args.operating_margin
            if args.operating_margin is not None
            else configured.operating_margin
        ),
        tax_rate=(args.tax_rate if args.tax_rate is not None else configured.tax_rate),
        depreciation_to_revenue=(
            args.depreciation_to_revenue
            if args.depreciation_to_revenue is not None
            else configured.depreciation_to_revenue
        ),
        capex_to_revenue=(
            args.capex_to_revenue
            if args.capex_to_revenue is not None
            else configured.capex_to_revenue
        ),
        operating_working_capital_to_revenue=(
            args.operating_working_capital_to_revenue
            if args.operating_working_capital_to_revenue is not None
            else configured.operating_working_capital_to_revenue
        ),
        revenue_anchors=(
            {} if args.revenue_growth is not None else configured.revenue_anchors
        ),
        assumption_source_overrides={
            driver: source
            for driver, source in configured.assumption_source_overrides.items()
            if not (
                args.revenue_growth is not None and driver.value == "revenue_growth"
            )
        },
        historical_window=(
            args.historical_window
            if args.historical_window is not None
            else configured.historical_window
        ),
    )


async def _management_guidance_overlay(
    args: argparse.Namespace,
    financials: NormalizedCompanyFinancials,
    parameters: FcffForecastParameters,
    baseline,
    valuation_date: datetime.date,
) -> tuple[FcffForecastParameters, GuidanceOverlayResult]:
    if not args.user_agent:
        return parameters, GuidanceOverlayResult(
            warnings=(
                "Management guidance skipped: SEC retrieval requires --user-agent",
            )
        )
    cache = FileSystemCache(args.cache_dir)
    openai_client = OpenAIClient(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        reasoning_effort=OPENAI_REASONING_EFFORT,
    )
    try:
        async with EdgarClient(cache, args.user_agent) as edgar:
            discovery = await ManagementGuidanceService(
                edgar,
                ManagementGuidanceExtractor(openai_client, cache),
            ).retrieve(
                ticker=args.ticker or financials.ticker,
                cik=args.cik,
                as_of=valuation_date,
                refresh_sec=args.refresh,
            )
        overlaid, overlay = GuidanceForecastOverlay().apply(
            discovery.records,
            baseline=baseline,
            parameters=parameters,
        )
        validation_rejections = tuple(
            f"Extraction rejected: {item.reason}" for item in discovery.rejected
        )
        return overlaid, overlay.model_copy(
            update={
                "rejected_reasons": tuple(
                    dict.fromkeys([*overlay.rejected_reasons, *validation_rejections])
                ),
                "warnings": tuple(
                    dict.fromkeys([*overlay.warnings, *discovery.warnings])
                ),
                "cache_hits": discovery.cache_hits,
                "cache_misses": discovery.cache_misses,
                "filings_inspected": discovery.filings_inspected,
                "documents_inspected": discovery.documents_inspected,
                "extracted_guidance_records": discovery.extracted_guidance_records,
                "rejected_records": (
                    discovery.rejected_records + len(overlay.rejected_reasons)
                ),
            }
        )
    finally:
        await openai_client.close()


def _forward_growth_evidence(lifecycle, economic_traits, guidance_overlay):
    """Turn verified forward indicators into a bounded stage-duration signal."""
    records = ()
    if guidance_overlay is not None:
        records = (*guidance_overlay.applications, *guidance_overlay.evidence_only)
    evidence_records = tuple(
        getattr(item, "guidance", item) for item in records
    )
    traits = {getattr(item, "value", str(item)) for item in economic_traits}
    backlog = "backlog_driven" in traits
    for item in guidance_overlay.applications if guidance_overlay is not None else ():
        metric = getattr(item.guidance.metric, "value", "")
        backlog = backlog or metric in {"backlog", "bookings"}
    guidance = any(
        getattr(getattr(item, "metric", None), "value", "")
        in {"revenue", "revenue_growth"}
        for item in evidence_records
    )
    text = " ".join(
        getattr(item, "supporting_text", "").casefold() for item in evidence_records
    )
    capacity = any(
        term in text for term in ("capacity", "cleanroom", "fab", "shipment")
    )
    visible_years = {
        item.fiscal_year
        for item in evidence_records
        if getattr(item, "fiscal_year", None) is not None
        and getattr(getattr(item, "period_type", None), "value", "")
        in {"multi_year_target", "long_term_target"}
    }
    growth_visibility = min(Decimal("1"), Decimal(len(visible_years)) / Decimal("3"))
    forward_growth_records = sorted(
        (
            item.fiscal_year,
            item.midpoint,
        )
        for item in evidence_records
        if getattr(getattr(item, "metric", None), "value", "")
        == "revenue_growth"
        and item.midpoint is not None
    )
    return ForwardGrowthEvidence(
        backlog=backlog,
        guidance=guidance,
        capacity=capacity,
        growth_visibility=growth_visibility,
        lifecycle=getattr(lifecycle, "value", str(lifecycle)),
        growth_path=tuple(value for _year, value in forward_growth_records),
        confidence="high" if forward_growth_records else None,
    )


async def _retrieve_automatic_assumption_inputs(
    args: argparse.Namespace,
    financials: NormalizedCompanyFinancials,
    currency: str,
    *,
    needs_wacc: bool,
    needs_terminal: bool,
    sector_override=None,
    industry_override: Optional[str] = None,
) -> dict:
    inputs = {
        "classification": None,
        "market_data": None,
        "risk_free_series": None,
        "inflation_series": None,
        "country_snapshot": None,
        "industry_snapshot": None,
        "company_beta": None,
    }
    if not (needs_wacc or needs_terminal):
        return inputs

    cache = FileSystemCache(Path(args.cache_dir))
    use_cache = not args.refresh
    symbol = financials.ticker or args.ticker
    if needs_wacc:
        if not symbol:
            raise ValueError(
                "Automatic WACC requires a ticker to retrieve Yahoo classification "
                "and market capitalization; provide --ticker or explicit WACC inputs"
            )
        try:
            async with YahooFinanceClient(cache) as yahoo:
                source, history = await asyncio.gather(
                    yahoo.get_company_financials(
                        symbol, use_cache=use_cache, make_cache=True
                    ),
                    yahoo.get_price_history(
                        symbol,
                        period="1mo",
                        use_cache=use_cache,
                        make_cache=True,
                    ),
                )
                classification = CompanyClassificationNormalizer().normalize_yahoo(
                    source
                )
                if (
                    classification.industry is None
                    or classification.country is None
                    or source.beta is None
                ):
                    source = await yahoo.get_company_financials(
                        symbol, use_cache=False, make_cache=True
                    )
                    classification = CompanyClassificationNormalizer().normalize_yahoo(
                        source
                    )
                classification = _apply_classification_overrides(
                    classification,
                    sector=sector_override,
                    industry=industry_override,
                )
            inputs["classification"] = classification
            inputs["company_beta"] = source.beta
            market_data = YahooMarketNormalizer().normalize(history)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "Automatic WACC could not retrieve Yahoo classification/price data; "
                "set WACC or the missing CAPM/capital-weight inputs in the profile. "
                f"Cause: {exc}"
            ) from exc
        if market_data.currency != currency.strip().upper():
            try:
                async with EcbClient(cache) as ecb:
                    market_data = await EcbMarketDataCurrencyConverter(ecb).convert(
                        market_data,
                        currency,
                        use_cache=use_cache,
                        make_cache=True,
                    )
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"Automatic WACC could not align the Yahoo quote currency "
                    f"({market_data.currency}) with the financial-statement currency "
                    f"({currency}); provide an explicit WACC or market-value equity. "
                    f"Cause: {exc}"
                ) from exc
        inputs["market_data"] = market_data

        try:
            async with DamodaranClient(cache) as damodaran:
                country_snapshot, industry_snapshot = await asyncio.gather(
                    damodaran.get_country_risk_premiums(
                        use_cache=use_cache, make_cache=True
                    ),
                    damodaran.get_industry_betas(use_cache=use_cache, make_cache=True),
                )
            inputs["country_snapshot"] = country_snapshot
            inputs["industry_snapshot"] = industry_snapshot
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "Automatic WACC could not retrieve the versioned Damodaran country "
                "and industry references; set beta, ERP, country premium, and tax "
                f"inputs in the profile. Cause: {exc}"
            ) from exc

    normalized_currency = currency.strip().upper()
    try:
        if normalized_currency == "EUR":
            start = datetime.date.today() - datetime.timedelta(days=365 * 6)
            async with EcbClient(cache) as ecb:
                risk_free_task = ecb.get_series(
                    "YC",
                    "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                    kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
                    unit=ReferenceValueUnit.PERCENTAGE_POINTS,
                    start_period=datetime.date.today() - datetime.timedelta(days=45),
                    end_period=datetime.date.today(),
                    use_cache=use_cache,
                    make_cache=True,
                )
                if needs_terminal:
                    risk_free, inflation = await asyncio.gather(
                        risk_free_task,
                        ecb.get_series(
                            "HICP",
                            "M.U2.N.000000.4D0.ANR",
                            kind=ReferenceSeriesKind.INFLATION_RATE,
                            unit=ReferenceValueUnit.PERCENT_CHANGE,
                            start_period=start,
                            end_period=datetime.date.today(),
                            use_cache=use_cache,
                            make_cache=True,
                        ),
                    )
                    inputs["inflation_series"] = inflation
                else:
                    risk_free = await risk_free_task
            inputs["risk_free_series"] = risk_free
        elif normalized_currency == "DKK":
            today = datetime.date.today()
            inflation_start = today - datetime.timedelta(days=365 * 6)
            async with EcbClient(cache) as ecb:
                risk_free_task = ecb.get_series(
                    "IRS",
                    "M.DK.L.L40.CI.0000.DKK.N.Z",
                    kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
                    unit=ReferenceValueUnit.PERCENTAGE_POINTS,
                    start_period=today - datetime.timedelta(days=120),
                    end_period=today,
                    use_cache=use_cache,
                    make_cache=True,
                )
                if needs_terminal:
                    risk_free, inflation = await asyncio.gather(
                        risk_free_task,
                        ecb.get_series(
                            "HICP",
                            "M.DK.N.000000.4D0.ANR",
                            kind=ReferenceSeriesKind.INFLATION_RATE,
                            unit=ReferenceValueUnit.PERCENT_CHANGE,
                            start_period=inflation_start,
                            end_period=today,
                            use_cache=use_cache,
                            make_cache=True,
                        ),
                    )
                    inputs["inflation_series"] = inflation
                else:
                    risk_free = await risk_free_task
            inputs["risk_free_series"] = risk_free
        elif normalized_currency == "USD":
            async with TreasuryClient(cache) as treasury:
                inputs["risk_free_series"] = await treasury.get_par_yield(
                    120,
                    use_cache=use_cache,
                    make_cache=True,
                )
            if needs_terminal and FRED_API_KEY:
                async with FredClient(cache, FRED_API_KEY) as fred:
                    inputs["inflation_series"] = await fred.get_series(
                        "FPCPITOTLZGUSA",
                        kind=ReferenceSeriesKind.INFLATION_RATE,
                        unit=ReferenceValueUnit.PERCENT_CHANGE,
                        observation_start=datetime.date.today()
                        - datetime.timedelta(days=365 * 15),
                        observation_end=datetime.date.today(),
                        country="US",
                        use_cache=use_cache,
                        make_cache=True,
                    )
        else:
            raise ValueError(
                f"automatic macro assumptions currently support DKK, EUR, and USD, not "
                f"{normalized_currency}; set risk_free_rate/WACC and terminal growth "
                "in the profile"
            )
    except RuntimeError as exc:
        raise ValueError(
            "Automatic valuation assumptions could not retrieve the sovereign-yield "
            "or inflation series; provide risk_free_rate/WACC and terminal growth in "
            f"the profile. Cause: {exc}"
        ) from exc
    return inputs


def _apply_classification_overrides(
    classification,
    *,
    sector=None,
    industry: Optional[str] = None,
):
    """Apply explicit valuation-profile economics without erasing raw labels."""
    updates = {}
    if sector is not None:
        updates["sector"] = sector
        updates["sector_taxonomy"] = "valuation-profile"
    if industry is not None:
        updates["industry"] = industry
        updates["industry_taxonomy"] = "valuation-profile"
    return classification.model_copy(update=updates) if updates else classification


def _resolve_wacc(override: Optional[Decimal], configuration) -> tuple[Decimal, str]:
    if override is not None:
        return override, "explicit CLI override"
    if configuration.wacc is not None:
        return configuration.wacc, "explicit valuation profile"

    beta = configuration.levered_beta
    if beta is None and configuration.unlevered_beta is not None:
        required = {
            "market_value_debt": configuration.market_value_debt,
            "market_value_equity": configuration.market_value_equity,
            "normalized_tax_rate": configuration.normalized_tax_rate,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Levering the profile beta requires: " + ", ".join(missing)
            )
        assert configuration.market_value_debt is not None
        assert configuration.market_value_equity is not None
        assert configuration.normalized_tax_rate is not None
        beta = DiscountRateService.lever_beta(
            configuration.unlevered_beta,
            configuration.market_value_debt,
            configuration.market_value_equity,
            configuration.normalized_tax_rate,
        )

    cost_of_equity = configuration.cost_of_equity
    if cost_of_equity is None:
        capm_inputs = {
            "risk_free_rate": configuration.risk_free_rate,
            "levered_beta": beta,
            "equity_risk_premium": configuration.equity_risk_premium,
        }
        missing = [name for name, value in capm_inputs.items() if value is None]
        if missing:
            raise ValueError(
                "FCFF DCF requires WACC. Provide --wacc, set valuation.discount_rates.wacc, "
                "or complete the profile CAPM/WACC inputs. Missing: "
                + ", ".join(missing)
            )
        assert configuration.risk_free_rate is not None
        assert beta is not None
        assert configuration.equity_risk_premium is not None
        cost_of_equity = DiscountRateService.cost_of_equity(
            configuration.risk_free_rate,
            beta,
            configuration.equity_risk_premium,
            configuration.country_risk_premium or Decimal(0),
        ).cost_of_equity

    wacc_inputs = {
        "pretax_cost_of_debt": configuration.pretax_cost_of_debt,
        "normalized_tax_rate": configuration.normalized_tax_rate,
        "market_value_equity": configuration.market_value_equity,
        "market_value_debt": configuration.market_value_debt,
    }
    missing = [name for name, value in wacc_inputs.items() if value is None]
    if missing:
        raise ValueError(
            "FCFF DCF WACC calculation is missing profile inputs: " + ", ".join(missing)
        )
    assert configuration.pretax_cost_of_debt is not None
    assert configuration.normalized_tax_rate is not None
    assert configuration.market_value_equity is not None
    assert configuration.market_value_debt is not None
    result = DiscountRateService.wacc(
        cost_of_equity,
        configuration.pretax_cost_of_debt,
        configuration.normalized_tax_rate,
        configuration.market_value_equity,
        configuration.market_value_debt,
    )
    return result.wacc, "derived from valuation profile CAPM and capital weights"


def _validate_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("--limit must be at least 1")


def _parse_mappings(values: Optional[list[str]], option: str) -> dict[str, str]:
    mappings = {}
    for value in values or []:
        key, separator, mapped_value = value.partition("=")
        key = key.strip()
        mapped_value = mapped_value.strip()
        if not separator or not key or not mapped_value:
            raise ValueError(f"{option} must use NAME=SYMBOL syntax")
        normalized_key = key.lower()
        if normalized_key in mappings:
            raise ValueError(f"Duplicate {option} mapping for {key}")
        mappings[normalized_key] = mapped_value
    return mappings


def _parse_provider_symbols(values: Optional[list[str]]) -> dict[ProviderName, str]:
    raw_mappings = _parse_mappings(values, "--provider-symbol")
    try:
        return {
            ProviderName(provider): symbol for provider, symbol in raw_mappings.items()
        }
    except ValueError as exc:
        choices = ", ".join(provider.value for provider in ProviderName)
        raise ValueError(
            f"Unknown provider in --provider-symbol; choose one of: {choices}"
        ) from exc


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logger(
        logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    )
    try:
        if args.command == "financials":
            return asyncio.run(_run_financials(args))
        if args.command == "metrics":
            return asyncio.run(_run_metrics(args))
        if args.command == "export":
            return asyncio.run(_run_export(args))
        if args.command == "red-flags":
            return asyncio.run(_run_red_flags(args))
        if args.command == "forecast":
            return asyncio.run(_run_forecast(args))
        if args.command == "valuation":
            return asyncio.run(_run_valuation(args))
        if args.command == "valuation-models":
            return asyncio.run(_run_valuation_models(args))
        if args.command == "classification":
            return asyncio.run(_run_classification(args))
        if args.command == "comparables":
            return asyncio.run(_run_comparables(args))
        if args.command == "specialized-inputs":
            return asyncio.run(_run_specialized_inputs(args))
    except (ValueError, RuntimeError, FileNotFoundError, ValidationError) as exc:
        parser.error(str(exc))
    return 1
