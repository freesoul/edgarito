import datetime
from decimal import Decimal

import pytest

from edgarito.cli.presentation.decision import DecisionValuationConsolePresenter
from edgarito.cli.presentation.valuation_report import (
    ValuationReportConsolePresenter,
)
from edgarito.config.valuation import MultistageValuationConfiguration
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting import (
    AdaptiveMultistageFcffForecastService,
    FcffForecastParameters,
    FcffForecastService,
)
from edgarito.services.valuation import (
    DecisionScenario,
    DecisionScenarioPolicy,
    DecisionValuationResult,
    DecisionValuationService,
    FcffDcfCapitalBridge,
    FcffDcfParameters,
    FcffDcfService,
    IntrinsicDecisionContext,
    IntrinsicDecisionEngine,
    IntrinsicScenarioCase,
    PriceComparison,
    RelativeScenarioCase,
    RelativeScenarioTimeBasis,
    ReverseDcfService,
    ReverseDcfStatus,
    ReverseDcfVariable,
    ScenarioValuationService,
    SensitivityAnalysisService,
    ValuationAssessment,
    ValuationAssessmentBand,
)

VALUATION_DATE = datetime.date(2024, 12, 31)


def _observation(concept, value, year):
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=year,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(year, 12, 31),
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
    )


def _financials(ticker="DECISION", *, capex_ratio=Decimal("6")):
    observations = []
    for year, revenue in (
        (2022, Decimal("100")),
        (2023, Decimal("110")),
        (2024, Decimal("121")),
    ):
        values = {
            FinancialConcept.REVENUE: revenue,
            FinancialConcept.OPERATING_INCOME: revenue * Decimal("0.22"),
            FinancialConcept.PRETAX_INCOME: revenue * Decimal("0.20"),
            FinancialConcept.INCOME_TAX_EXPENSE: revenue * Decimal("0.04"),
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION: revenue * Decimal("0.04"),
            FinancialConcept.CAPITAL_EXPENDITURES: revenue * capex_ratio / Decimal(100),
            FinancialConcept.ACCOUNTS_RECEIVABLE: revenue * Decimal("0.12"),
            FinancialConcept.INVENTORY: revenue * Decimal("0.05"),
            FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: revenue
            * Decimal("0.03"),
            FinancialConcept.ACCOUNTS_PAYABLE: revenue * Decimal("0.07"),
            FinancialConcept.ACCRUED_LIABILITIES: revenue * Decimal("0.04"),
            FinancialConcept.DEFERRED_REVENUE_CURRENT: revenue * Decimal("0.02"),
        }
        observations.extend(
            _observation(concept, str(value), year) for concept, value in values.items()
        )
    return NormalizedCompanyFinancials(
        provider="test",
        company_id=ticker,
        company_name=f"{ticker} Company",
        ticker=ticker,
        observations=observations,
    )


def _context(
    *,
    ticker="DECISION",
    capex_ratio=Decimal("6"),
    net_debt=Decimal("20"),
    wacc=Decimal("8"),
    terminal_growth=Decimal("2"),
    explicit=False,
):
    financials = _financials(ticker, capex_ratio=capex_ratio)
    parameters = FcffForecastParameters(
        forecast_years=5,
        revenue_growth=Decimal("10") if explicit else None,
        operating_margin=Decimal("22") if explicit else None,
        tax_rate=Decimal("20") if explicit else None,
        depreciation_to_revenue=Decimal("4") if explicit else None,
        capex_to_revenue=capex_ratio if explicit else None,
        operating_working_capital_to_revenue=Decimal("7") if explicit else None,
    )
    service = FcffForecastService()
    seed = service.forecast(financials, parameters, as_of=VALUATION_DATE)
    configuration = MultistageValuationConfiguration(
        terminal_return_on_invested_capital=Decimal("15")
    )
    forecast, plan = AdaptiveMultistageFcffForecastService(service).forecast(
        financials,
        seed,
        parameters,
        terminal_growth,
        configuration,
        normalized_tax_rate=Decimal("20"),
        as_of=VALUATION_DATE,
    )
    bridge = FcffDcfCapitalBridge(
        fiscal_year=2024,
        period_end=VALUATION_DATE,
        unit="USD",
        net_debt=net_debt,
        diluted_shares=Decimal("10"),
        net_debt_source="controlled fixture",
        shares_source="controlled fixture",
    )
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc=wacc, perpetual_growth_rate=terminal_growth),
        bridge,
        multistage_plan=plan,
        valuation_date=VALUATION_DATE,
    )
    return IntrinsicDecisionContext(
        financials=financials,
        requested_parameters=parameters,
        seed_forecast=seed,
        base_forecast=forecast,
        base_result=result,
        capital_bridge=bridge,
        terminal_roic=Decimal("15"),
        multistage_configuration=configuration,
        use_multistage=True,
        valuation_date=VALUATION_DATE,
        normalized_tax_rate=Decimal("20"),
        flexible_revenue_growth=not explicit,
        flexible_operating_margin=not explicit,
        flexible_terminal_roic=not explicit,
        flexible_wacc=not explicit,
        flexible_terminal_growth=not explicit,
    )


def test_scenarios_are_ordered_reproducible_and_change_real_assumptions():
    engine = IntrinsicDecisionEngine(_context())
    service = ScenarioValuationService()

    first = service.build(engine)
    second = service.build(engine)

    assert first == second
    bear, base, bull = first
    assert bear.available and base.available and bull.available
    assert bear.value_per_share < base.value_per_share < bull.value_per_share
    assert bear.assumptions[0].value < base.assumptions[0].value
    assert bull.assumptions[1].value > base.assumptions[1].value
    assert bear.assumptions[3].value > base.assumptions[3].value
    for case in first:
        values = {item.name: item.value for item in case.assumptions}
        assert values["WACC"] > values["Terminal growth"]
        assert values["Terminal ROIC"] > values["Terminal growth"]


def test_scenarios_preserve_explicit_profile_or_cli_assumptions():
    cases = ScenarioValuationService().build(
        IntrinsicDecisionEngine(_context(explicit=True))
    )

    assert cases[1].available
    assert cases[1].value_per_share is not None
    for case in (cases[0], cases[2]):
        assert not case.available
        assert case.value_per_share is None
        assert case.invalid_reason
        assert "explicit assumptions" in case.invalid_reason
        assert not any(item.changed for item in case.assumptions)
        assert all("preserved explicit" in item.source for item in case.assumptions)


def test_scenarios_preserve_an_explicit_negative_terminal_growth():
    cases = ScenarioValuationService().build(
        IntrinsicDecisionEngine(_context(explicit=True, terminal_growth=Decimal("-1")))
    )

    for case in cases:
        terminal_growth = next(
            item for item in case.assumptions if item.name == "Terminal growth"
        )
        assert terminal_growth.value == Decimal("-1")
        assert not terminal_growth.changed


def test_non_monotonic_combined_scenarios_are_explicitly_unavailable():
    context = _context()
    policy = DecisionScenarioPolicy(bull_wacc_delta=Decimal("-20"))
    service = ScenarioValuationService(policy)
    engine = IntrinsicDecisionEngine(context)
    raw_bull = service._variant(engine, DecisionScenario.BULL)
    assert raw_bull.value_per_share < context.base_result.value_per_share

    result = DecisionValuationService(policy).build(
        context,
        Decimal("10"),
        include_sensitivity=False,
        include_reverse_dcf=False,
    )

    bear, base, bull = result.intrinsic_scenarios
    assert bear.available and base.available
    assert not bull.available
    assert bull.value_per_share is None
    assert "non-monotonic" in bull.invalid_reason
    assert "non-monotonic" in " ".join(result.warnings).lower()


def test_scenario_revaluation_reuses_base_multistage_topology():
    context = _context()
    engine = IntrinsicDecisionEngine(context)
    service = ScenarioValuationService()
    values = service._scenario_values(context, DecisionScenario.BULL)

    evaluation = engine.evaluate(
        revenue_growth=values["revenue_growth"],
        operating_margin=values["operating_margin"],
        terminal_roic=values["terminal_roic"],
        terminal_growth=values["terminal_growth"],
        wacc=values["wacc"],
        preserve_projection_structure=True,
    )

    base_plan = context.base_result.multistage_plan
    scenario_plan = evaluation.result.multistage_plan
    assert base_plan is not None and scenario_plan is not None
    assert (
        scenario_plan.explicit_growth_prefix_years,
        scenario_plan.high_growth_years,
        scenario_plan.transition_years,
        scenario_plan.stable_years,
    ) == (
        base_plan.explicit_growth_prefix_years,
        base_plan.high_growth_years,
        base_plan.transition_years,
        base_plan.stable_years,
    )
    assert evaluation.forecast.adaptive_stages == context.base_forecast.adaptive_stages


def test_unavailable_scenarios_are_rendered_without_publishing_base_as_bull():
    context = _context(explicit=True)
    result = DecisionValuationService().build(
        context,
        Decimal("10"),
        include_sensitivity=False,
        include_reverse_dcf=False,
    )

    rendered = DecisionValuationConsolePresenter().render(result, show_scenarios=True)

    assert "Bear scenario unavailable" in rendered
    assert "Bull scenario unavailable" in rendered
    assert "Intrinsic value/share" in rendered
    assert "unavailable" in rendered
    assert not any(
        comparison.label == "Bull" for comparison in result.price_comparisons
    )


def test_sensitivity_is_monotonic_and_marks_invalid_wacc_growth_pairs():
    regular = SensitivityAnalysisService().wacc_terminal_growth(
        IntrinsicDecisionEngine(_context())
    )
    middle_column = len(regular.column_values) // 2
    values_by_wacc = [row[middle_column].value_per_share for row in regular.cells]
    assert all(
        left > right
        for left, right in zip(values_by_wacc, values_by_wacc[1:], strict=False)
    )
    middle_row = len(regular.row_values) // 2
    values_by_growth = [cell.value_per_share for cell in regular.cells[middle_row]]
    assert all(
        left < right
        for left, right in zip(values_by_growth, values_by_growth[1:], strict=False)
    )

    tight = SensitivityAnalysisService().wacc_terminal_growth(
        IntrinsicDecisionEngine(
            _context(wacc=Decimal("3.5"), terminal_growth=Decimal("3"))
        )
    )
    assert any(cell.invalid_reason for row in tight.cells for cell in row)


def test_higher_terminal_roic_raises_value_through_lower_reinvestment():
    engine = IntrinsicDecisionEngine(_context())
    low = engine.evaluate(terminal_roic=Decimal("10"))
    high = engine.evaluate(terminal_roic=Decimal("25"))

    assert high.result.value_per_share > low.result.value_per_share
    assert (
        high.result.multistage_plan.terminal_reinvestment_rate
        < low.result.multistage_plan.terminal_reinvestment_rate
    )


def test_margin_of_safety_sign_and_assessment_bands_are_explainable():
    context = _context()
    base_value = context.base_result.value_per_share
    cheap = DecisionValuationService().build(
        context,
        base_value * Decimal("0.5"),
        include_sensitivity=False,
        include_reverse_dcf=False,
    )
    expensive = DecisionValuationService().build(
        context,
        base_value * Decimal("2"),
        include_sensitivity=False,
        include_reverse_dcf=False,
    )

    cheap_base = next(item for item in cheap.price_comparisons if item.label == "Base")
    expensive_base = next(
        item for item in expensive.price_comparisons if item.label == "Base"
    )
    assert cheap_base.upside_downside > 0
    assert cheap_base.margin_of_safety > 0
    assert expensive_base.upside_downside < 0
    assert expensive_base.margin_of_safety < 0
    assert cheap.assessment.intrinsic == ValuationAssessmentBand.STRONGLY_CHEAP
    assert expensive.assessment.intrinsic == ValuationAssessmentBand.STRONGLY_EXPENSIVE


def test_intrinsic_relative_disagreement_is_not_averaged_away():
    context = _context()
    intrinsic = ScenarioValuationService().build(IntrinsicDecisionEngine(context))
    base = intrinsic[1].value_per_share
    relative = tuple(
        RelativeScenarioCase(
            scenario=scenario,
            value_per_share=value,
            multiple=Decimal("10"),
            methodology="controlled fixture",
        )
        for scenario, value in (
            (DecisionScenario.BEAR, base * Decimal("2.0")),
            (DecisionScenario.BASE, base * Decimal("2.2")),
            (DecisionScenario.BULL, base * Decimal("2.4")),
        )
    )
    assessment = DecisionValuationService()._assessment(
        base * Decimal("1.5"), intrinsic, relative
    )

    assert assessment.intrinsic in {
        ValuationAssessmentBand.EXPENSIVE,
        ValuationAssessmentBand.STRONGLY_EXPENSIVE,
    }
    assert assessment.relative in {
        ValuationAssessmentBand.CHEAP,
        ValuationAssessmentBand.STRONGLY_CHEAP,
    }
    assert assessment.overall.startswith("models disagree")


def test_reverse_dcf_reproduces_price_and_solves_each_variable_independently():
    engine = IntrinsicDecisionEngine(_context())
    target = engine.evaluate(
        revenue_growth=engine.context.base_revenue_growth + Decimal("3")
    ).result.value_per_share
    solutions = ReverseDcfService().solve_all(engine, target)
    growth = next(
        item for item in solutions if item.variable == ReverseDcfVariable.REVENUE_GROWTH
    )

    assert growth.status == ReverseDcfStatus.SOLVED
    assert growth.implied_value > growth.base_value
    assert abs(growth.achieved_price - target) <= Decimal("0.02")
    assert all("alone" in item.methodology for item in solutions)


def test_reverse_dcf_fails_gracefully_outside_economic_bounds():
    solutions = ReverseDcfService().solve_all(
        IntrinsicDecisionEngine(_context()), Decimal("1000000000")
    )

    assert all(item.status == ReverseDcfStatus.NO_SOLUTION for item in solutions)
    assert all(item.implied_value is None for item in solutions)


def test_console_keeps_the_default_summary_concise_and_exposes_optional_details():
    context = _context()
    result = DecisionValuationService().build(
        context, context.base_result.value_per_share
    )
    presenter = DecisionValuationConsolePresenter()

    summary = presenter.render(result)
    details = presenter.render(
        result,
        show_scenarios=True,
        show_sensitivity=True,
        show_reverse_dcf=True,
    )

    assert "DECISION SUMMARY" in summary
    assert "Margin-of-safety convention" not in summary
    assert "SENSITIVITY:" not in summary
    assert "SCENARIOS" in details
    assert "SENSITIVITY: VALUE/SHARE SENSITIVITY" in details
    assert "REVERSE DCF" in details
    assert "rows are not a combined forecast" in details
    assert details.rfind("DECISION SUMMARY") > details.rfind("REVERSE DCF")

    audit = presenter.render(result, verbose=True)
    assert "Margin-of-safety convention" in audit


def test_decision_presenter_separates_target_date_relative_evidence_from_dcf():
    intrinsic = tuple(
        IntrinsicScenarioCase(
            scenario=scenario,
            value_per_share=value,
            assumptions=(),
            methodology="controlled intrinsic DCF",
        )
        for scenario, value in zip(
            (DecisionScenario.BEAR, DecisionScenario.BASE, DecisionScenario.BULL),
            (Decimal("8"), Decimal("10"), Decimal("12")),
            strict=True,
        )
    )
    relative = tuple(
        RelativeScenarioCase(
            scenario=scenario,
            value_per_share=value,
            multiple=Decimal("10"),
            methodology="controlled pure peer target-date evidence",
            time_basis=RelativeScenarioTimeBasis.TARGET_DATE,
            target_date=datetime.date(2025, 12, 31),
            horizon_years=Decimal(1),
            horizon_upside_downside=(value / Decimal("10") - Decimal(1)) * 100,
        )
        for scenario, value in zip(
            (DecisionScenario.BEAR, DecisionScenario.BASE, DecisionScenario.BULL),
            (Decimal("8"), Decimal("10"), Decimal("12")),
            strict=True,
        )
    )
    comparisons = tuple(
        PriceComparison(
            label=scenario.scenario.value.title(),
            model="intrinsic",
            value_per_share=scenario.value_per_share,
            upside_downside=(scenario.value_per_share / Decimal("10") - 1) * 100,
            margin_of_safety=(1 - Decimal("10") / scenario.value_per_share) * 100,
        )
        for scenario in intrinsic
    )
    result = DecisionValuationResult(
        company_name="Controlled Company",
        currency="USD",
        current_price=Decimal("10"),
        intrinsic_scenarios=intrinsic,
        relative_scenarios=relative,
        price_comparisons=comparisons,
        assessment=ValuationAssessment(
            intrinsic=ValuationAssessmentBand.FAIR,
            overall=ValuationAssessmentBand.FAIR.value,
            rationale=("Only present-day intrinsic DCF scenario evidence was used",),
        ),
        methodology="controlled",
    )

    rendered = "\n".join(DecisionValuationConsolePresenter().render_summary(result))

    assert "Present-day intrinsic DCF comparison" in rendered
    assert "TARGET-DATE RELATIVE EVIDENCE" in rendered
    assert "Target-date value/share" in rendered
    assert "Horizon upside/(downside)" in rendered
    assert "excluded from present-day margin-of-safety" in rendered
    assert "Relative assessment: target-date evidence excluded" in rendered
    assert "Peer relative" not in rendered


def test_decision_presenter_displays_mixed_summary_for_target_date_relative_upside():
    intrinsic = tuple(
        IntrinsicScenarioCase(
            scenario=scenario,
            value_per_share=value,
            assumptions=(),
            methodology="controlled intrinsic DCF",
        )
        for scenario, value in zip(
            (DecisionScenario.BEAR, DecisionScenario.BASE, DecisionScenario.BULL),
            (Decimal("8"), Decimal("10"), Decimal("12")),
            strict=True,
        )
    )
    relative = tuple(
        RelativeScenarioCase(
            scenario=scenario,
            value_per_share=value,
            multiple=Decimal("10"),
            methodology="controlled pure peer target-date evidence",
            time_basis=RelativeScenarioTimeBasis.TARGET_DATE,
            target_date=datetime.date(2025, 12, 31),
            horizon_years=Decimal(1),
            horizon_upside_downside=(value / Decimal("10") - 1) * 100,
        )
        for scenario, value in zip(
            (DecisionScenario.BEAR, DecisionScenario.BASE, DecisionScenario.BULL),
            (Decimal("9"), Decimal("11"), Decimal("13")),
            strict=True,
        )
    )
    comparisons = tuple(
        PriceComparison(
            label=scenario.scenario.value.title(),
            model="intrinsic",
            value_per_share=scenario.value_per_share,
            upside_downside=(scenario.value_per_share / Decimal("10") - 1) * 100,
            margin_of_safety=(1 - Decimal("10") / scenario.value_per_share) * 100,
        )
        for scenario in intrinsic
    )
    result = DecisionValuationResult(
        company_name="Controlled Company",
        currency="USD",
        current_price=Decimal("10"),
        intrinsic_scenarios=intrinsic,
        relative_scenarios=relative,
        price_comparisons=comparisons,
        assessment=ValuationAssessment(
            intrinsic=ValuationAssessmentBand.EXPENSIVE,
            overall=ValuationAssessmentBand.EXPENSIVE.value,
        ),
        methodology="controlled",
    )

    rendered = "\n".join(DecisionValuationConsolePresenter().render_summary(result))

    assert "Present-day intrinsic DCF comparison" in rendered
    assert (
        "Overall assessment: mixed — above intrinsic base value, but supported by relative valuation"
        in rendered
    )
    assert "Intrinsic assessment: expensive" in rendered


def test_consolidated_warning_output_deduplicates_shortens_and_adds_severity():
    message = (
        "Projected net debt and diluted shares are held flat because no "
        "capital-structure forecast was supplied"
    )
    rendered = ValuationReportConsolePresenter._render_warnings(
        [message, message, "Enterprise value does not cover reported net debt"],
        verbose=False,
    )
    audit = ValuationReportConsolePresenter._render_warnings([message], verbose=True)

    assert rendered.count("Projected net debt") == 1
    assert "[HIGH] Enterprise value does not cover reported net debt." in rendered
    assert "[INFO] Projected net debt and diluted shares are held flat." in rendered
    assert "because no capital-structure forecast was supplied" not in rendered
    assert "because no capital-structure forecast was supplied" in audit


@pytest.mark.parametrize(
    ("ticker", "capex_ratio", "net_debt"),
    [
        ("MATURE", Decimal("5"), Decimal("10")),
        ("CAPITAL", Decimal("18"), Decimal("80")),
    ],
)
def test_decision_layer_remains_finite_across_stable_and_capital_intensive_cases(
    ticker, capex_ratio, net_debt
):
    result = DecisionValuationService().build(
        _context(ticker=ticker, capex_ratio=capex_ratio, net_debt=net_debt),
        Decimal("10"),
        include_sensitivity=False,
        include_reverse_dcf=False,
    )

    assert all(item.value_per_share.is_finite() for item in result.intrinsic_scenarios)
    assert result.price_comparisons
