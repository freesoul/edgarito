import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.config.valuation import (
    DiscountRateConfiguration,
    ForecastValuationProfile,
)
from edgarito.schemas.valuation.intrinsic import (
    ComponentValuationMethod,
    ComponentValueBasis,
    DividendDiscountInput,
    FcfeDcfInput,
    InputProvenance,
    IntrinsicValuationContext,
    PipelineProject,
    PipelineProjectYear,
    PropertyAsset,
    ResidualIncomeInput,
    ResourceProject,
    ResourceProjectYear,
    SotpAdjustment,
    SotpAdjustmentKind,
    SotpComponent,
    SotpValuationInput,
)
from edgarito.schemas.valuation.relative import (
    ForwardValuationMetric,
    RelativeCapitalBridge,
    RelativeNumeratorBasis,
)
from edgarito.services.valuation.assumptions import CostOfEquityResolver
from edgarito.services.valuation.execution import ValuationExecutor
from edgarito.services.valuation.intrinsic import (
    DividendDiscountService,
    FcfeDcfService,
    PipelineRnpvAdapter,
    PropertyNavAdapter,
    ResidualIncomeService,
    ResourceNavAdapter,
    SotpValuationService,
)
from edgarito.services.valuation.models import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    DataReadiness,
    EconomicTrait,
    FinancialInstitutionKind,
    ModelRole,
    ModelSuitability,
    MultipleConfidence,
    RelativeValuationBasis,
    ResolvedMultiple,
    ValuationModel,
    ValuationProfile,
    ValuationSelection,
)
from edgarito.services.valuation.relative import (
    ProviderNeutralRelativeValuationService,
)
from edgarito.services.valuation.selector import ValuationModelSelector

D = Decimal
TODAY = datetime.date(2026, 8, 7)
PROVENANCE = (InputProvenance(field="test", source="synthetic", observed_on=TODAY),)


def context(shares: str = "10") -> IntrinsicValuationContext:
    return IntrinsicValuationContext(
        company_id="1",
        company_name="Example",
        ticker="EX",
        valuation_date=TODAY,
        currency="USD",
        diluted_shares=D(shares),
    )


def test_cost_of_equity_resolution_never_requires_wacc_or_debt() -> None:
    resolver = CostOfEquityResolver()
    profile_result = resolver.resolve(
        configuration=DiscountRateConfiguration(
            risk_free_rate=D(4),
            levered_beta=D("1.2"),
            equity_risk_premium=D(5),
            country_risk_premium=D(0),
        ),
        valuation_date=TODAY,
        currency="USD",
        company_id="1",
        company_beta=D(2),
    )
    assert profile_result.cost_of_equity == D(10)
    assert profile_result.levered_beta == D("1.2")
    cli_result = resolver.resolve(
        configuration=DiscountRateConfiguration(cost_of_equity=D(9)),
        valuation_date=TODAY,
        currency="USD",
        company_id="1",
        cost_of_equity_override=D(8),
    )
    assert cli_result.cost_of_equity == D(8)
    assert cli_result.levered_beta is None


def test_schema_v2_serializes_typed_specialized_profile_sections() -> None:
    profile = ForecastValuationProfile.model_validate(
        {
            "schema_version": 2,
            "name": "resource",
            "valuation": {
                "resources": {
                    "projects": [_resource_project("base", D(10)).model_dump()]
                }
            },
        }
    )
    restored = ForecastValuationProfile.model_validate_json(profile.model_dump_json())
    assert restored.schema_version == 2
    assert restored.valuation.resources.projects[0].years[0].cash_flow == D(40)


def test_fcfe_reconciles_net_borrowing_without_enterprise_bridge() -> None:
    period = FcfeDcfService.corporate_period(
        label="Year 1",
        period=D(1),
        net_income=D(100),
        depreciation_and_amortization=D(20),
        capital_expenditures=D(50),
        working_capital_change=D(10),
        debt_financing_ratio=D("0.25"),
    )
    assert period.net_borrowing == D(10)
    assert period.fcfe == D(70)
    result = FcfeDcfService().value(
        FcfeDcfInput(
            context=context(),
            periods=(period,),
            cost_of_equity=D(10),
            terminal_growth_rate=D(3),
            terminal_return_on_equity=D(12),
        )
    )
    assert result.equity_value == (
        result.details.explicit_fcfe_present_value
        + result.details.terminal_present_value
    )
    assert "net_debt" not in result.model_dump()


def test_regulated_fcfe_uses_required_equity_growth_not_deposits() -> None:
    period = FcfeDcfService.regulated_period(
        label="Year 1",
        period=D(1),
        net_income=D(100),
        required_common_equity_change=D(25),
    )
    assert period.fcfe == D(75)
    assert period.net_borrowing is None
    assert "deposit" not in period.model_dump()


@pytest.mark.parametrize(
    ("growth", "roe"),
    [("10", "12"), ("3", "3")],
)
def test_fcfe_rejects_invalid_terminal_economics(growth: str, roe: str) -> None:
    period = FcfeDcfService.regulated_period(
        label="Year 1",
        period=D(1),
        net_income=D(100),
        required_common_equity_change=D(20),
    )
    with pytest.raises(ValueError):
        FcfeDcfService().value(
            FcfeDcfInput(
                context=context(),
                periods=(period,),
                cost_of_equity=D(10),
                terminal_growth_rate=D(growth),
                terminal_return_on_equity=D(roe),
                regulated_financial=True,
            )
        )


def test_ddm_derives_each_terminal_policy_leg_and_rejects_bad_triple() -> None:
    service = DividendDiscountService()
    assert service.resolve_terminal_policy(
        growth=D(3), return_on_equity=D(12), payout_ratio=None
    ) == (D(3), D(12), D("0.75"))
    assert service.resolve_terminal_policy(
        growth=None, return_on_equity=D(12), payout_ratio=D("0.75")
    ) == (D(3), D(12), D("0.75"))
    assert service.resolve_terminal_policy(
        growth=D(3), return_on_equity=None, payout_ratio=D("0.75")
    ) == (D(3), D(12), D("0.75"))
    with pytest.raises(ValueError, match="inconsistent"):
        service.resolve_terminal_policy(
            growth=D(4), return_on_equity=D(12), payout_ratio=D("0.75")
        )


def test_gordon_ddm_and_fcfe_disconnect_warning() -> None:
    result = DividendDiscountService().value(
        DividendDiscountInput(
            context=context(),
            mode="gordon",
            cost_of_equity=D(10),
            terminal_growth_rate=D(3),
            terminal_return_on_equity=D(12),
            next_dividend=D(4),
            distributable_fcfe=D(8),
        )
    )
    assert result.equity_value == D(4) / D("0.07")
    assert {warning.code for warning in result.warnings} == {"dividend_fcfe_disconnect"}


def test_residual_income_equals_book_at_roe_equal_to_cost_of_equity() -> None:
    result = ResidualIncomeService().value(
        ResidualIncomeInput(
            context=context(),
            starting_book_value=D(1000),
            book_value_basis="common_equity",
            return_on_equity_path=(D(10), D(10)),
            payout_ratio_path=(D("0.4"), D("0.4")),
            cost_of_equity=D(10),
        )
    )
    assert result.equity_value == D(1000)
    first = result.details.periods[0]
    assert first.ending_book_value == D(1060)
    assert result.details.transition_years == 0


def test_residual_income_policy_inference_requires_clean_consecutive_history() -> None:
    service = ResidualIncomeService()
    with pytest.raises(ValueError, match="three clean"):
        service.infer_policy(((2024, D(12), D("0.4")),))
    with pytest.raises(ValueError, match="consecutive"):
        service.infer_policy(
            (
                (2022, D(12), D("0.4")),
                (2024, D(13), D("0.5")),
                (2025, D(14), D("0.6")),
            )
        )
    roe, payout, confidence = service.infer_policy(
        (
            (2023, D(12), D("0.4")),
            (2024, D(50), D("0.5")),
            (2025, D(14), D("0.6")),
        )
    )
    assert (roe, payout, confidence.value) == (D(14), D("0.5"), "medium")


def test_sotp_applies_component_debt_fx_ownership_and_adjustments() -> None:
    component = SotpComponent(
        name="Division",
        method=ComponentValuationMethod.DCF,
        value_basis=ComponentValueBasis.ENTERPRISE,
        value=D(100),
        currency="EUR",
        ownership=D("0.5"),
        component_net_debt=D(20),
        fx_rate_to_reporting_currency=D("1.2"),
        fx_rate_date=TODAY,
        fx_rate_source="ECB",
        provenance=PROVENANCE,
    )
    adjustment = SotpAdjustment(
        name="Corporate cash",
        kind=SotpAdjustmentKind.NON_OPERATING_ASSET,
        amount=D(10),
        currency="USD",
        fx_rate_date=TODAY,
        fx_rate_source="reporting currency",
    )
    result = SotpValuationService().value(
        SotpValuationInput(
            context=context(), components=(component,), adjustments=(adjustment,)
        )
    )
    assert result.details.owned_gross_asset_value == D("48")
    assert result.equity_value == D(58)


def test_sotp_rejects_duplicate_corporate_balance_sheet_adjustments() -> None:
    component = SotpComponent(
        name="Division",
        method=ComponentValuationMethod.DCF,
        value_basis=ComponentValueBasis.EQUITY,
        value=D(100),
        currency="USD",
        fx_rate_date=TODAY,
        fx_rate_source="reporting currency",
        included_balance_sheet_items=frozenset({"cash"}),
    )
    adjustment = SotpAdjustment(
        name="Cash",
        kind=SotpAdjustmentKind.NON_OPERATING_ASSET,
        amount=D(10),
        currency="USD",
        fx_rate_date=TODAY,
        fx_rate_source="reporting currency",
        balance_sheet_item="cash",
    )
    with pytest.raises(ValueError, match="duplicates"):
        SotpValuationService().value(
            SotpValuationInput(
                context=context(), components=(component,), adjustments=(adjustment,)
            )
        )


def test_lower_reit_cap_rate_increases_property_nav() -> None:
    adapter = PropertyNavAdapter()
    high_cap = adapter.to_components(
        (
            PropertyAsset(
                name="A",
                noi=D(10),
                cap_rate=D(10),
                currency="USD",
                provenance=PROVENANCE,
            ),
        ),
        valuation_date=TODAY,
    )[0]
    low_cap = adapter.to_components(
        (
            PropertyAsset(
                name="A",
                noi=D(10),
                cap_rate=D(5),
                currency="USD",
                provenance=PROVENANCE,
            ),
        ),
        valuation_date=TODAY,
    )[0]
    assert low_cap.value > high_cap.value


def test_resource_schedule_is_finite_reserve_constrained_and_scenario_separated() -> (
    None
):
    with pytest.raises(ValidationError, match="reserves"):
        ResourceProject(
            name="Mine",
            scenario="base",
            reserves=D(5),
            discount_rate=D(10),
            currency="USD",
            years=(
                ResourceProjectYear(
                    year=1,
                    production=D(6),
                    commodity_price=D(10),
                    operating_costs=D(1),
                    sustaining_capex=D(1),
                    development_capex=D(1),
                    taxes_and_royalties=D(1),
                    closure_costs=D(1),
                ),
            ),
            provenance=PROVENANCE,
        )
    base = _resource_project("base", D(10))
    bull = _resource_project("bull", D(15))
    scenarios = ResourceNavAdapter().to_scenarios((base, bull), valuation_date=TODAY)
    assert set(scenarios) == {"base", "bull"}
    assert scenarios["bull"][0].value > scenarios["base"][0].value
    with pytest.raises(ValueError, match="separately"):
        ResourceNavAdapter().to_components((base, bull), valuation_date=TODAY)


def _resource_project(scenario: str, price: Decimal) -> ResourceProject:
    return ResourceProject(
        name="Mine",
        scenario=scenario,
        reserves=D(10),
        discount_rate=D(10),
        currency="USD",
        years=(
            ResourceProjectYear(
                year=1,
                production=D(5),
                commodity_price=price,
                operating_costs=D(5),
                sustaining_capex=D(2),
                development_capex=D(1),
                taxes_and_royalties=D(2),
                closure_costs=D(0),
            ),
        ),
        provenance=PROVENANCE,
    )


def test_pipeline_zero_probability_keeps_expected_development_costs() -> None:
    probability = InputProvenance(field="success_probability", source="profile")
    project = PipelineProject(
        name="Drug A",
        success_probability=D(0),
        success_probability_provenance=probability,
        discount_rate=D(10),
        currency="USD",
        years=(
            PipelineProjectYear(
                year=1, development_cost=D(11), success_cash_flow=D(100)
            ),
        ),
    )
    component = PipelineRnpvAdapter().to_components((project,), valuation_date=TODAY)[0]
    assert component.value == D(-10)


def test_executor_runs_ready_models_independently_and_keeps_blocked_visible() -> None:
    profile = ValuationProfile(
        provider="synthetic",
        company_id="1",
        company_name="Example",
        business_archetype=BusinessArchetype.GENERAL_OPERATING,
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=Cyclicality.LOW,
        economic_traits={EconomicTrait.DIVIDEND_PAYER},
    )
    ready_fcfe = ModelSuitability(
        model=ValuationModel.EQUITY_DCF,
        role=ModelRole.CONDITIONAL,
        suitability_score=70,
        data_readiness=DataReadiness.READY,
    )
    ready_ddm = ready_fcfe.model_copy(
        update={"model": ValuationModel.DIVIDEND_DISCOUNT}
    )
    blocked = ready_fcfe.model_copy(
        update={
            "model": ValuationModel.NAV_SOTP,
            "data_readiness": DataReadiness.BLOCKED,
        }
    )
    selection = ValuationSelection(
        profile=profile, models=[ready_fcfe, ready_ddm, blocked]
    )
    base = DividendDiscountService().value(
        DividendDiscountInput(
            context=context(),
            mode="gordon",
            cost_of_equity=D(10),
            terminal_growth_rate=D(3),
            terminal_return_on_equity=D(12),
            next_dividend=D(4),
        )
    )
    fcfe_result = base.model_copy(update={"model": ValuationModel.EQUITY_DCF})
    run = ValuationExecutor().execute(
        selection=selection,
        runners={
            ValuationModel.EQUITY_DCF: lambda: fcfe_result,
            ValuationModel.DIVIDEND_DISCOUNT: lambda: base,
        },
    )
    assert [item.result.model for item in run.executed_models] == [
        ValuationModel.EQUITY_DCF,
        ValuationModel.DIVIDEND_DISCOUNT,
    ]
    assert run.skipped_models[0].model == ValuationModel.NAV_SOTP
    assert "blended" not in run.model_dump()


def test_executor_preserves_separate_adapter_scenarios() -> None:
    profile = ValuationProfile(
        provider="synthetic",
        company_id="1",
        company_name="Resource",
        business_archetype=BusinessArchetype.RESOURCE_PRODUCER,
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=Cyclicality.HIGH,
    )
    suitability = ModelSuitability(
        model=ValuationModel.NAV_SOTP,
        role=ModelRole.PRIMARY,
        suitability_score=95,
        data_readiness=DataReadiness.READY,
    )
    base = (
        DividendDiscountService()
        .value(
            DividendDiscountInput(
                context=context(),
                mode="gordon",
                cost_of_equity=D(10),
                terminal_growth_rate=D(3),
                terminal_return_on_equity=D(12),
                next_dividend=D(4),
            )
        )
        .model_copy(update={"model": ValuationModel.NAV_SOTP, "adapter": "base"})
    )
    bull = base.model_copy(
        update={
            "adapter": "bull",
            "equity_value": base.equity_value * D("1.2"),
            "value_per_share": base.value_per_share * D("1.2"),
        }
    )
    run = ValuationExecutor().execute(
        selection=ValuationSelection(profile=profile, models=[suitability]),
        runners={ValuationModel.NAV_SOTP: lambda: (base, bull)},
    )
    assert [item.result.adapter for item in run.executed_models] == ["base", "bull"]
    assert run.dispersion is not None


def test_bank_selector_uses_residual_income_primary_and_separate_ddm() -> None:
    profile = ValuationProfile(
        provider="synthetic",
        company_id="bank",
        company_name="Bank",
        business_archetype=BusinessArchetype.FINANCIAL_INTERMEDIARY,
        financial_institution_kind=FinancialInstitutionKind.BANK,
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=Cyclicality.MODERATE,
        economic_traits={
            EconomicTrait.REGULATED_CAPITAL,
            EconomicTrait.DIVIDEND_PAYER,
            EconomicTrait.STABLE_PAYOUT,
        },
    )
    selection = ValuationModelSelector().select(profile)
    assert selection.primary is not None
    assert selection.primary.model == ValuationModel.RESIDUAL_INCOME
    ddm = next(
        item
        for item in selection.models
        if item.model == ValuationModel.DIVIDEND_DISCOUNT
    )
    fcfe = next(
        item for item in selection.models if item.model == ValuationModel.EQUITY_DCF
    )
    fcff = next(
        item for item in selection.models if item.model == ValuationModel.FCFF_DCF
    )
    assert ddm.model != fcfe.model
    assert fcff.data_readiness == DataReadiness.NOT_APPLICABLE


def test_equity_relative_basis_produces_actual_value_without_ev_bridge() -> None:
    metric = ForwardValuationMetric(
        basis=RelativeValuationBasis.PE,
        amount=D(100),
        label="FY2027E common earnings",
        target_date=datetime.date(2027, 8, 7),
        currency="USD",
        numerator_basis=RelativeNumeratorBasis.EQUITY_VALUE,
    )
    result = ProviderNeutralRelativeValuationService().value(
        valuation_date=TODAY,
        horizon_years=D(1),
        metric=metric,
        diluted_shares=D(10),
        discount_rate=D(10),
        resolved_multiple=_resolved_multiple(RelativeValuationBasis.PE, D(15)),
        current_price=D(120),
    )
    assert result.point_case.target_date_equity_value == D(1500)
    assert result.point_case.target_date_value_per_share == D(150)
    assert result.point_case.present_value_per_share < D(150)
    assert result.current_price_implied_multiple == D(12)
    with pytest.raises(ValueError, match="must not apply"):
        ProviderNeutralRelativeValuationService().value(
            valuation_date=TODAY,
            horizon_years=D(1),
            metric=metric,
            diluted_shares=D(10),
            discount_rate=D(10),
            resolved_multiple=_resolved_multiple(RelativeValuationBasis.PE, D(15)),
            capital_bridge=RelativeCapitalBridge(net_debt=D(20)),
        )


def test_enterprise_relative_basis_requires_and_applies_ev_bridge() -> None:
    metric = ForwardValuationMetric(
        basis=RelativeValuationBasis.EV_TO_EBITDA,
        amount=D(100),
        label="FY2027E EBITDA",
        target_date=datetime.date(2027, 8, 7),
        currency="USD",
        numerator_basis=RelativeNumeratorBasis.ENTERPRISE_VALUE,
    )
    result = ProviderNeutralRelativeValuationService().value(
        valuation_date=TODAY,
        horizon_years=D(1),
        metric=metric,
        diluted_shares=D(10),
        discount_rate=D(10),
        resolved_multiple=_resolved_multiple(
            RelativeValuationBasis.EV_TO_EBITDA, D(10)
        ),
        capital_bridge=RelativeCapitalBridge(
            net_debt=D(100), non_operating_assets=D(20)
        ),
    )
    assert result.point_case.target_date_numerator_value == D(1000)
    assert result.point_case.target_date_equity_value == D(920)


def _resolved_multiple(
    basis: RelativeValuationBasis, point: Decimal
) -> ResolvedMultiple:
    return ResolvedMultiple(
        basis=basis,
        point_estimate=point,
        lower_bound=point - D(1),
        upper_bound=point + D(1),
        fundamental_anchor=point,
        historical_persistence=D(0),
        fundamental_support=D(1),
        horizon_retention=D(1),
        persistence_factor=D(0),
        sample_size=5,
        confidence=MultipleConfidence.MEDIUM,
        methodology="synthetic test",
    )
