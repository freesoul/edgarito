# ruff: noqa: F401, F811

import argparse
import datetime
import logging
import time
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from edgarito.cli.comparables import (
    _build_comparable_report as build_comparable_report,
)
from edgarito.cli.comparables import (
    _resolve_comparable_peer_symbols as resolve_comparable_peer_symbols,
)
from edgarito.cli.presentation.console import (
    IndependentValuationModelsConsolePresenter,
    SpecializedExtractionConsolePresenter,
    ValuationReportConsolePresenter,
    ValuationSelectionConsolePresenter,
)
from edgarito.cli.use_cases import valuation_dcf as dcf_use_case
from edgarito.cli.use_cases import valuation_forecast as forecast_use_case
from edgarito.cli.use_cases import valuation_intrinsic as intrinsic_use_case
from edgarito.cli.use_cases import valuation_output as output_use_case
from edgarito.cli.use_cases import valuation_relative as relative_valuation_use_case
from edgarito.cli.use_cases.context import (
    ValuationDependencyContext,
    call_with_context,
    dependency,
)
from edgarito.cli.use_cases.financial_retrieval import (
    parse_provider_symbols,
    retrieve_classification,
    retrieve_financials,
)
from edgarito.cli.use_cases.forecast import (
    fcff_parameters,
    load_selected_valuation_profile,
    market_for_args,
    resolve_depreciable_asset_life_configuration,
)
from edgarito.cli.use_cases.forward_assumptions import (
    financial_snapshot_warnings,
    forward_growth_evidence,
    management_guidance_overlay,
    materialize_forward_revenue_anchors,
    merge_forward_growth_evidence,
    retrieve_automatic_assumption_inputs,
    retrieve_forward_estimates,
)
from edgarito.cli.use_cases.operating_evidence import (
    default_operating_vocabulary_audit,
    operating_evidence_provider,
    operating_quality_audit,
    retain_operating_audit_metadata,
    retrieve_operating_evidence,
)
from edgarito.config.valuation import (
    ForecastValuationProfile,
    ValuationProfileLoader,
)
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.schemas.guidance.management import GuidanceOverlayResult
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
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
from edgarito.schemas.valuation.selection import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    EconomicTrait,
    MultipleConfidence,
    RelativeValuationBasis,
    ValuationInput,
    ValuationModel,
    ValuationProfileOverrides,
)
from edgarito.schemas.valuation.specialized import SpecializedInputType
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.export import (
    ValuationExcelRenderer,
)
from edgarito.services.financials.availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting._fcff.service import FcffForecastService
from edgarito.services.forecasting.forward_estimates import (
    ForwardRevenueEstimateService,
)
from edgarito.services.forecasting.multistage import (
    AdaptiveMultistageFcffForecastService,
)
from edgarito.services.operating.contracts import (
    OperatingForecastQualityError,
    OperatingForecastQualityResult,
)
from edgarito.services.operating.integration import OperatingForecastPipelineService
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.valuation import (
    CashFlowTiming,
    ComparableImpliedValuationService,
    DecisionScenarioPolicy,
    DecisionValuationService,
    FcffDcfCapitalBridgeResolver,
    FcffDcfParameters,
    FcffDcfService,
    ForwardPeerMultiplesService,
    HistoricalMultiplesService,
    IntrinsicDecisionContext,
    MultipleResolver,
    ShareRepurchaseParameters,
    SpecializedValuationExtractor,
    TerminalMetric,
    TerminalRoicResolver,
    TerminalValueMethod,
    ValuationAssumptionResolver,
    ValuationModelSelector,
    ValuationProfileBuilder,
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
from edgarito.services.valuation.models import ResolvedMultiple
from edgarito.services.valuation.relative import (
    EQUITY_RELATIVE_BASES,
    ProviderNeutralRelativeValuationService,
)
from edgarito.settings import (
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


def _resolve_valuation_collaborator(dependencies, name: str):
    """Resolve a historically facade-patchable valuation collaborator."""

    return ValuationDependencyContext(dependencies).resolve(name, globals()[name])


@contextmanager
def _valuation_step(name: str):
    """Log progress for valuation stages, including elapsed time."""
    started = time.perf_counter()
    logger.warning("Valuation: %s...", name)
    try:
        yield
    except Exception:
        logger.exception(
            "Valuation: %s failed after %.1fs", name, time.perf_counter() - started
        )
        raise
    else:
        logger.warning(
            "Valuation: %s completed in %.1fs", name, time.perf_counter() - started
        )


# Optional provider-neutral injection seam retained for host applications and
# tests.  Normal valuation creates the SEC/OpenAI discovery provider below when
# its optional credentials are configured.
OPERATING_EVIDENCE_PROVIDER = None


async def _run_profile_intrinsic_valuation(
    args: argparse.Namespace,
    profile: ForecastValuationProfile,
    selected_model: str,
    *,
    dependencies=None,
) -> int:
    """Execute a profile-backed equity or asset model without an FCFF bridge."""
    CostOfEquityResolver = _resolve_valuation_collaborator(
        dependencies, "CostOfEquityResolver"
    )
    ValuationAssumptionResolver = _resolve_valuation_collaborator(
        dependencies, "ValuationAssumptionResolver"
    )
    ValuationProfileOverrides = _resolve_valuation_collaborator(
        dependencies, "ValuationProfileOverrides"
    )
    ValuationProfileBuilder = _resolve_valuation_collaborator(
        dependencies, "ValuationProfileBuilder"
    )
    ValuationModelSelector = _resolve_valuation_collaborator(
        dependencies, "ValuationModelSelector"
    )
    IntrinsicValuationContext = _resolve_valuation_collaborator(
        dependencies, "IntrinsicValuationContext"
    )
    ValuationConfidence = _resolve_valuation_collaborator(
        dependencies, "ValuationConfidence"
    )
    concepts = ValuationProfileBuilder.required_concepts() | {
        FinancialConcept.NET_INCOME_COMMON,
        FinancialConcept.COMMON_EQUITY,
        FinancialConcept.DIVIDENDS_PAID,
        FinancialConcept.GOODWILL,
        FinancialConcept.INTANGIBLE_ASSETS_NET,
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
        FinancialConcept.CAPITAL_EXPENDITURES,
    }
    retrieve = dependency(dependencies, "_retrieve_financials", retrieve_financials)
    financials = await call_with_context(
        retrieve,
        args,
        Granularity.ANNUAL,
        concepts,
        context=dependencies,
    )
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
    automatic = await call_with_context(
        dependency(
            dependencies,
            "_retrieve_automatic_assumption_inputs",
            retrieve_automatic_assumption_inputs,
        ),
        args,
        financials,
        currency,
        needs_wacc=needs_cost,
        needs_terminal=needs_terminal_growth and terminal_growth is None,
        sector_override=profile.model_selection.sector,
        industry_override=profile.model_selection.industry,
        context=dependencies,
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
    runner_factory = dependency(
        dependencies, "_profile_model_runner", _profile_model_runner
    )
    runners, requested_models = intrinsic_use_case.build_intrinsic_runners(
        selected_model=selected_model,
        economic_profile=economic_profile,
        profile=profile,
        context=context,
        annual=annual,
        cost_of_equity=equity_rate,
        terminal_growth=terminal_growth,
        terminal_roe_override=terminal_roe_override,
        args=args,
        runner_factory=runner_factory,
        dependencies=dependencies,
    )
    intrinsic_use_case.execute_intrinsic_models(
        selection=selection,
        runners=runners,
        requested_models=requested_models,
        verbose=args.verbose or args.audit,
        dependencies=dependencies,
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
    dependencies=None,
):
    FcfeForecastPeriod = _resolve_valuation_collaborator(
        dependencies, "FcfeForecastPeriod"
    )
    FcfeDcfService = _resolve_valuation_collaborator(dependencies, "FcfeDcfService")
    FcfeDcfInput = _resolve_valuation_collaborator(dependencies, "FcfeDcfInput")
    DividendForecastPeriod = _resolve_valuation_collaborator(
        dependencies, "DividendForecastPeriod"
    )
    DividendDiscountService = _resolve_valuation_collaborator(
        dependencies, "DividendDiscountService"
    )
    DividendDiscountInput = _resolve_valuation_collaborator(
        dependencies, "DividendDiscountInput"
    )
    ResidualIncomeService = _resolve_valuation_collaborator(
        dependencies, "ResidualIncomeService"
    )
    ResidualIncomeInput = _resolve_valuation_collaborator(
        dependencies, "ResidualIncomeInput"
    )
    PropertyNavAdapter = _resolve_valuation_collaborator(
        dependencies, "PropertyNavAdapter"
    )
    ResourceNavAdapter = _resolve_valuation_collaborator(
        dependencies, "ResourceNavAdapter"
    )
    SotpValuationService = _resolve_valuation_collaborator(
        dependencies, "SotpValuationService"
    )
    SotpValuationInput = _resolve_valuation_collaborator(
        dependencies, "SotpValuationInput"
    )
    PipelineRnpvAdapter = _resolve_valuation_collaborator(
        dependencies, "PipelineRnpvAdapter"
    )
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
async def _run_valuation(args: argparse.Namespace, *, context=None) -> int:
    ValuationProfileLoader = _resolve_valuation_collaborator(
        context, "ValuationProfileLoader"
    )
    ValuationProfileBuilder = _resolve_valuation_collaborator(
        context, "ValuationProfileBuilder"
    )
    OperatingForecastQualityResult = _resolve_valuation_collaborator(
        context, "OperatingForecastQualityResult"
    )
    OperatingForecastQualityError = _resolve_valuation_collaborator(
        context, "OperatingForecastQualityError"
    )
    ValuationAssumptionResolver = _resolve_valuation_collaborator(
        context, "ValuationAssumptionResolver"
    )
    ValuationProfileOverrides = _resolve_valuation_collaborator(
        context, "ValuationProfileOverrides"
    )
    TerminalRoicResolver = _resolve_valuation_collaborator(
        context, "TerminalRoicResolver"
    )
    AdaptiveMultistageFcffForecastService = _resolve_valuation_collaborator(
        context, "AdaptiveMultistageFcffForecastService"
    )
    OperatingForecastPipelineService = _resolve_valuation_collaborator(
        context, "OperatingForecastPipelineService"
    )
    ForwardRevenueEstimateService = _resolve_valuation_collaborator(
        context, "ForwardRevenueEstimateService"
    )
    _market_for_args = dependency(context, "_market_for_args", market_for_args)
    _retrieve_classification = dependency(
        context, "_retrieve_classification", retrieve_classification
    )
    _valuation_step = dependency(context, "_valuation_step", valuation_step)
    _operating_evidence_provider = dependency(
        context, "_operating_evidence_provider", operating_evidence_provider
    )
    _default_operating_vocabulary_audit = dependency(
        context,
        "_default_operating_vocabulary_audit",
        default_operating_vocabulary_audit,
    )
    _retrieve_operating_evidence = dependency(
        context, "_retrieve_operating_evidence", retrieve_operating_evidence
    )
    _operating_quality_audit = dependency(
        context, "_operating_quality_audit", operating_quality_audit
    )
    _retain_operating_audit_metadata = dependency(
        context,
        "_retain_operating_audit_metadata",
        retain_operating_audit_metadata,
    )
    _retrieve_automatic_assumption_inputs = dependency(
        context,
        "_retrieve_automatic_assumption_inputs",
        retrieve_automatic_assumption_inputs,
    )
    _resolve_depreciable_asset_life_configuration = dependency(
        context,
        "_resolve_depreciable_asset_life_configuration",
        resolve_depreciable_asset_life_configuration,
    )
    _retrieve_forward_estimates = dependency(
        context, "_retrieve_forward_estimates", retrieve_forward_estimates
    )
    _materialize_forward_revenue_anchors = dependency(
        context,
        "_materialize_forward_revenue_anchors",
        materialize_forward_revenue_anchors,
    )
    _forward_growth_evidence = dependency(
        context, "_forward_growth_evidence", forward_growth_evidence
    )
    _merge_forward_growth_evidence = dependency(
        context, "_merge_forward_growth_evidence", merge_forward_growth_evidence
    )
    _run_profile_intrinsic_valuation = dependency(
        context,
        "_run_profile_intrinsic_valuation",
        profile_intrinsic_valuation,
    )
    logger.warning("Valuation: starting for %s", args.ticker or "profile")
    market = call_with_context(_market_for_args, args, context=context)
    generated_profile_path = None
    should_generate_profile = False
    peer_report = None
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
                classification = await call_with_context(
                    _retrieve_classification,
                    args,
                    provider=None,
                    crosscheck=False,
                    context=context,
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
        return await call_with_context(
            _run_profile_intrinsic_valuation,
            args,
            profile,
            selected_model,
            context=context,
        )
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
    construction = await forecast_use_case.construct_fcff_forecast(
        args=args,
        profile=profile,
        market=market,
        dependencies=context,
    )
    forecast_parameters = construction.forecast_parameters
    terminal_configuration = construction.terminal_configuration
    terminal_method = construction.terminal_method
    cash_flow_timing = construction.cash_flow_timing
    forecast_service = construction.forecast_service
    bridge_resolver = construction.bridge_resolver
    financials = construction.financials
    valuation_date = construction.valuation_date
    forecast = construction.forecast
    seed_forecast = construction.seed_forecast
    guidance_overlay = construction.guidance_overlay
    additional_warnings.extend(construction.warnings)
    operating_evidence = None
    operating_audit = OperatingForecastQualityResult(
        accepted=False,
        reason="Operating forecast inactive: provider not configured",
        vocabulary_audit=call_with_context(
            _default_operating_vocabulary_audit,
            profile,
            context=context,
        ),
    )
    async with call_with_context(
        _operating_evidence_provider,
        args,
        financials,
        market=market,
        context=context,
    ) as (provider, provider_rejection):
        if provider_rejection is not None:
            operating_audit = OperatingForecastQualityResult(
                accepted=False,
                reason=provider_rejection,
                vocabulary_audit=call_with_context(
                    _default_operating_vocabulary_audit,
                    profile,
                    context=context,
                ),
            )
            additional_warnings.append(provider_rejection)
        if provider_rejection is None:
            classification = None
            if not profile.model_selection.industry:
                try:
                    classification = await call_with_context(
                        _retrieve_classification,
                        args,
                        provider=None,
                        crosscheck=False,
                        context=context,
                    )
                except Exception as exc:
                    additional_warnings.append(
                        f"Operating vocabulary classification unavailable: {exc}"
                    )
            operating_metadata = {
                "industry": profile.model_selection.industry
                or getattr(classification, "industry", None)
                or getattr(financials, "industry", None),
                "sector": profile.model_selection.sector
                or getattr(classification, "sector", None)
                or getattr(financials, "sector", None),
                "business_archetype": profile.model_selection.business_archetype,
            }
            with call_with_context(
                _valuation_step,
                "retrieving operating evidence",
                context=context,
            ):
                (
                    operating_evidence,
                    operating_warnings,
                ) = await call_with_context(
                    _retrieve_operating_evidence,
                    financials,
                    forecast,
                    valuation_date,
                    provider=provider,
                    args=args,
                    metadata=operating_metadata,
                    context=context,
                )
            additional_warnings.extend(operating_warnings)
            if operating_evidence is not None:
                # Discovery evidence is gated only after the deterministic
                # engine reconstructs history; do not reject on missing audit
                # metrics before the pipeline has calculated them.
                operating_audit = call_with_context(
                    _operating_quality_audit,
                    operating_evidence,
                    discovery_warnings=operating_warnings,
                    context=context,
                )
            else:
                operating_audit = OperatingForecastQualityResult(
                    accepted=False,
                    reason=(
                        operating_warnings[-1]
                        if operating_warnings
                        else "Operating forecast rejected: discovery returned no usable evidence"
                    ),
                    warnings=operating_warnings,
                    vocabulary_audit=call_with_context(
                        _default_operating_vocabulary_audit,
                        profile,
                        context=context,
                    ),
                )
        else:
            operating_evidence = None
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
    with call_with_context(
        _valuation_step,
        "retrieving automatic valuation assumptions",
        context=context,
    ):
        automatic_inputs = await call_with_context(
            _retrieve_automatic_assumption_inputs,
            args,
            financials,
            forecast.unit,
            needs_wacc=needs_automatic_wacc,
            needs_terminal=needs_automatic_terminal,
            sector_override=profile.model_selection.sector,
            industry_override=profile.model_selection.industry,
            context=context,
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
    comparable_retrieval = await relative_valuation_use_case.retrieve_comparable_bundle(
        args=args,
        profile=profile,
        financials=financials,
        selected_model=selected_model,
        valuation_date=valuation_date,
        dependencies=context,
    )
    comparable_bundle = comparable_retrieval.bundle
    comparable_error = comparable_retrieval.error

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
                terminal_roic_source=terminal_roic.source,
                terminal_roic_methodology=terminal_roic.methodology,
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
    asset_life_resolution = None
    if use_multistage:
        (
            multistage_configuration,
            asset_life_resolution,
        ) = call_with_context(
            _resolve_depreciable_asset_life_configuration,
            financials,
            profile_context,
            multistage_configuration,
            context=context,
        )
        if asset_life_resolution is not None:
            additional_warnings.extend(asset_life_resolution.warnings)
    tax_assumption = resolved.assumption_set.find(
        ValuationAssumptionKind.NORMALIZED_TAX_RATE
    )
    forward_estimate_result = None
    forward_evidence = None
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
        with call_with_context(
            _valuation_step,
            "retrieving forward estimates",
            context=context,
        ):
            forward_estimate_result = await call_with_context(
                _retrieve_forward_estimates,
                args,
                financials,
                forecast,
                use_cache=not args.refresh,
                make_cache=True,
                context=context,
            )
        additional_warnings.extend(forward_estimate_result.warnings)
        forecast_parameters = call_with_context(
            _materialize_forward_revenue_anchors,
            forecast_parameters,
            forward_estimate_result.estimates,
            forecast,
            additional_warnings,
            context=context,
        )
        management_evidence = call_with_context(
            _forward_growth_evidence,
            profile_context.lifecycle,
            profile_context.economic_traits,
            guidance_overlay,
            tuple(item.fiscal_year for item in forecast.observations),
            context=context,
        )
        consensus_evidence = ForwardRevenueEstimateService.to_growth_evidence(
            forward_estimate_result,
            forecast_years=tuple(item.fiscal_year for item in forecast.observations),
            base_revenue=(
                forecast.ytd_anchor.latest_annual_revenue
                if forecast.ytd_anchor is not None
                else forecast.base_revenue
            ),
            seed_revenues={
                item.fiscal_year: item.revenue for item in forecast.observations
            },
            seed_growth_path=tuple(
                item.revenue_growth for item in forecast.observations
            ),
            lifecycle=getattr(
                profile_context.lifecycle, "value", str(profile_context.lifecycle)
            ),
            backlog=management_evidence.backlog,
            capacity=management_evidence.capacity,
            growth_visibility=management_evidence.growth_visibility,
        )
        forward_evidence = call_with_context(
            _merge_forward_growth_evidence,
            management_evidence,
            consensus_evidence,
            management_has_revenue_guidance=any(
                item.driver == "revenue_growth"
                and getattr(getattr(item.guidance, "metric", None), "value", "")
                == "revenue"
                for item in (
                    guidance_overlay.applications
                    if guidance_overlay is not None
                    else ()
                )
            ),
            forecast_years=tuple(item.fiscal_year for item in forecast.observations),
            context=context,
        )
        if operating_evidence is None:
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
                forward_evidence=forward_evidence,
                as_of=valuation_date,
                availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
            )
        else:
            try:
                pipeline_result = OperatingForecastPipelineService(
                    fcff_service=forecast_service,
                ).forecast(
                    financials,
                    evidence=operating_evidence,
                    parameters=forecast_parameters,
                    consensus_estimates=forward_estimate_result.estimates,
                    terminal_growth_rate=stable_growth_rate,
                    adaptive_configuration=multistage_configuration,
                    normalized_tax_rate=(
                        tax_assumption.value if tax_assumption is not None else None
                    ),
                    forward_evidence=forward_evidence,
                    as_of=valuation_date,
                    availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
                )
            except OperatingForecastQualityError as exc:
                operating_audit = call_with_context(
                    _retain_operating_audit_metadata,
                    exc.result,
                    operating_audit,
                    context=context,
                )
                operating_evidence = None
                additional_warnings.append(
                    f"{exc.result.reason}; standard consensus/historical forecast retained"
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
                    forward_evidence=forward_evidence,
                    as_of=valuation_date,
                    availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
                )
            else:
                operating_audit = call_with_context(
                    _retain_operating_audit_metadata,
                    pipeline_result.quality or operating_audit,
                    operating_audit,
                    context=context,
                )
                forecast = pipeline_result.forecast
                multistage_plan = pipeline_result.adaptive_plan
                forecast_parameters = pipeline_result.integration.parameters
                forward_evidence = pipeline_result.forward_growth or forward_evidence
                additional_warnings.extend(pipeline_result.warnings)
        plan_updates = {
            "terminal_roic_source": terminal_roic.source,
            "terminal_roic_methodology": terminal_roic.methodology,
            "terminal_roic_confidence": terminal_roic.confidence,
            "terminal_roic_warnings": terminal_roic.warnings,
        }
        if asset_life_resolution is not None and asset_life_resolution.warnings:
            plan_updates["warnings"] = tuple(
                dict.fromkeys(
                    [*multistage_plan.warnings, *asset_life_resolution.warnings]
                )
            )
        multistage_plan = multistage_plan.model_copy(update=plan_updates)
    elif operating_evidence is not None:
        try:
            pipeline_result = OperatingForecastPipelineService(
                fcff_service=forecast_service,
            ).forecast(
                financials,
                evidence=operating_evidence,
                parameters=forecast_parameters,
                as_of=valuation_date,
                availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
            )
        except OperatingForecastQualityError as exc:
            operating_audit = call_with_context(
                _retain_operating_audit_metadata,
                exc.result,
                operating_audit,
                context=context,
            )
            operating_evidence = None
            additional_warnings.append(
                f"{exc.result.reason}; standard FCFF forecast retained"
            )
        else:
            operating_audit = call_with_context(
                _retain_operating_audit_metadata,
                pipeline_result.quality or operating_audit,
                operating_audit,
                context=context,
            )
            forecast = pipeline_result.forecast
            forecast_parameters = pipeline_result.integration.parameters
            additional_warnings.extend(pipeline_result.warnings)
    with call_with_context(
        _valuation_step,
        "calculating FCFF DCF valuation",
        context=context,
    ):
        dcf_execution = dcf_use_case.execute_fcff_dcf(
            args=args,
            profile=profile,
            forecast=forecast,
            resolved=resolved,
            capital_bridge=capital_bridge,
            multistage_plan=multistage_plan,
            valuation_date=valuation_date,
            terminal_method=terminal_method,
            cash_flow_timing=cash_flow_timing,
            terminal_configuration=terminal_configuration,
            terminal_roic=terminal_roic,
            asset_life_resolution=asset_life_resolution,
            dependencies=context,
        )
    result = dcf_execution.result
    share_repurchase_parameters = dcf_execution.share_repurchase_parameters
    if selected_model == "comparables":
        additional_warnings.extend(result.warnings)
    relative_stage = relative_valuation_use_case.calculate_relative_valuation(
        args=args,
        profile=profile,
        comparable_bundle=comparable_bundle,
        comparable_error=comparable_error,
        selected_model=selected_model,
        financials=financials,
        forecast=forecast,
        result=result,
        capital_bridge=capital_bridge,
        resolved=resolved,
        valuation_date=valuation_date,
        terminal_method=terminal_method,
        dependencies=context,
    )
    peer_report = relative_stage.peer_report
    relative_result = relative_stage.relative_result
    provider_relative_result = relative_stage.provider_relative_result
    additional_warnings.extend(relative_stage.warnings)
    output_use_case.present_valuation_results(
        args=args,
        selected_model=selected_model,
        profile=profile,
        financials=financials,
        forecast=forecast,
        seed_forecast=seed_forecast,
        forecast_parameters=forecast_parameters,
        result=result,
        capital_bridge=capital_bridge,
        terminal_roic=terminal_roic,
        multistage_configuration=multistage_configuration,
        use_multistage=use_multistage,
        valuation_date=valuation_date,
        tax_assumption=tax_assumption,
        share_repurchase_parameters=share_repurchase_parameters,
        forward_evidence=forward_evidence,
        guidance_overlay=guidance_overlay,
        operating_audit=operating_audit,
        profile_context=profile_context,
        configured_terminal_roic=configured_terminal_roic,
        discount_configuration=discount_configuration,
        terminal_configuration=terminal_configuration,
        automatic_inputs=automatic_inputs,
        relative_result=relative_result,
        provider_relative_result=provider_relative_result,
        peer_report=peer_report,
        additional_warnings=additional_warnings,
        dependencies=context,
    )
    return 0
async def _run_valuation_models(args: argparse.Namespace, *, context=None) -> int:
    ValuationProfileOverrides = _resolve_valuation_collaborator(
        context, "ValuationProfileOverrides"
    )
    ValuationProfileBuilder = _resolve_valuation_collaborator(
        context, "ValuationProfileBuilder"
    )
    ValuationModelSelector = _resolve_valuation_collaborator(
        context, "ValuationModelSelector"
    )
    ValuationSelectionConsolePresenter = _resolve_valuation_collaborator(
        context, "ValuationSelectionConsolePresenter"
    )
    _load_selected_valuation_profile = dependency(
        context,
        "_load_selected_valuation_profile",
        load_selected_valuation_profile,
    )
    _retrieve_financials = dependency(
        context, "_retrieve_financials", retrieve_financials
    )
    _retrieve_classification = dependency(
        context, "_retrieve_classification", retrieve_classification
    )
    valuation_profile = call_with_context(
        _load_selected_valuation_profile,
        args,
        context=context,
    )
    configuration = valuation_profile.model_selection
    financials = await call_with_context(
        _retrieve_financials,
        args,
        Granularity.ANNUAL,
        ValuationProfileBuilder.required_concepts(),
        context=context,
    )
    classification = await call_with_context(
        _retrieve_classification,
        args,
        provider=(
            ProviderName(args.classification_provider)
            if args.classification_provider
            else None
        ),
        crosscheck=False,
        context=context,
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
async def _run_specialized_inputs(args: argparse.Namespace, *, context=None) -> int:
    _load_selected_valuation_profile = dependency(
        context,
        "_load_selected_valuation_profile",
        load_selected_valuation_profile,
    )
    cache_type = dependency(context, "FileSystemCache", FileSystemCache)
    edgar_type = dependency(context, "EdgarClient", EdgarClient)
    extractor_type = dependency(
        context,
        "SpecializedValuationExtractor",
        SpecializedValuationExtractor,
    )
    presenter_type = dependency(
        context,
        "SpecializedExtractionConsolePresenter",
        SpecializedExtractionConsolePresenter,
    )
    configuration = call_with_context(
        _load_selected_valuation_profile,
        args,
        context=context,
    ).specialized_inputs
    history = args.history if args.history is not None else configuration.history
    if history < 1:
        raise ValueError("--history must be at least 1")
    if not args.user_agent:
        raise ValueError("SEC retrieval requires EDGARITO_USER_AGENT / user_agent")
    async with edgar_type(cache_type(Path(args.cache_dir)), args.user_agent) as client:
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
    extraction = extractor_type().extract(
        facts,
        SpecializedInputType(args.type),
        ticker=args.ticker,
        historical_periods=history,
    )
    print(presenter_type().render(extraction))
    return 0


# Focused valuation entry points.
valuation_step = _valuation_step
profile_intrinsic_valuation = _run_profile_intrinsic_valuation
profile_model_runner = _profile_model_runner
_equity_relative_valuation = relative_valuation_use_case.equity_relative_valuation
equity_relative_valuation = _equity_relative_valuation


async def run_valuation(args, *, context=None):
    return await _run_valuation(args, context=context)


async def run_profile_intrinsic_valuation(
    args, profile, selected_model, *, context=None
):
    return await _run_profile_intrinsic_valuation(
        args, profile, selected_model, dependencies=context
    )


async def run_valuation_models(args, *, context=None):
    return await _run_valuation_models(args, context=context)


async def run_specialized_inputs(args, *, context=None):
    return await _run_specialized_inputs(args, context=context)
