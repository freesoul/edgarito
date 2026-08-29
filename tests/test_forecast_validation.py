from copy import deepcopy
from decimal import Decimal

from edgarito.services.forecasting.validation import (
    ForecastValidationConfig,
    ForecastValidationContext,
    ForecastValidationFinding,
    ForecastValidationResult,
    ForecastValidationService,
    Severity,
    TerminalMetrics,
    ValidationCategory,
)
from edgarito.services.forecasting.validation.rules._utils import percent_ratio


def row(year, **values):
    return {"fiscal_year": year, **values}


def reasonable_rows():
    return [
        row(
            2025,
            revenue="100",
            fcff="16",
            ebit="20",
            gross_profit="50",
            nopat="16",
            tax_rate="20",
            da="5",
            capex="4",
            delta_nwc="1",
        ),
        row(
            2026,
            revenue="110",
            fcff="17.6",
            ebit="22",
            gross_profit="55",
            nopat="17.6",
            tax_rate="20",
            da="5.5",
            capex="4.4",
            delta_nwc="1.1",
        ),
        row(
            2027,
            revenue="121",
            fcff="19.36",
            ebit="24.2",
            gross_profit="60.5",
            nopat="19.36",
            tax_rate="20",
            da="6.05",
            capex="4.84",
            delta_nwc="1.21",
        ),
    ]


def validate(rows, terminal=None, config=None):
    return ForecastValidationService(config=config).validate(
        ForecastValidationContext(rows=rows, terminal=terminal)
    )


def codes(result):
    return {finding.code for finding in result.findings}


def test_reasonable_forecast_has_no_high_severity_findings():
    result = validate(reasonable_rows())

    assert result.highest_severity in {None, Severity.INFO, Severity.WARNING}
    assert result.error_count == 0


def test_repeated_fifty_percent_fcff_growth_is_flagged():
    result = validate(
        [
            row(2025, fcff="100"),
            row(2026, fcff="160"),
            row(2027, fcff="256"),
            row(2028, fcff="409.6"),
        ]
    )

    assert "FCFF_GROWTH_EXTREME" in codes(result)
    assert "FCFF_REPEATED_HYPERGROWTH" in codes(result)


def test_near_zero_fcff_denominator_does_not_report_percentage_growth():
    result = validate([row(2025, fcff="0.01"), row(2026, fcff="10")])

    assert "FCFF_GROWTH_NEAR_ZERO_DENOMINATOR" in codes(result)
    assert "FCFF_GROWTH_EXTREME" not in codes(result)


def test_negative_to_positive_fcff_is_a_sign_transition():
    result = validate([row(2025, fcff="-10"), row(2026, fcff="10")])

    assert "SIGN_TRANSITION" in codes(result)
    assert "FCFF_GROWTH_EXTREME" not in codes(result)


def test_extreme_fcff_revenue_margin_is_flagged():
    result = validate([row(2025, revenue="100", fcff="200")])

    assert "EXTREME_FCFF_MARGIN" in codes(result)


def test_impossible_operating_margin_is_flagged():
    result = validate([row(2025, revenue="100", ebit="150")])

    assert "IMPOSSIBLE_OPERATING_MARGIN" in codes(result)


def test_terminal_growth_at_wacc_is_critical():
    result = validate(
        [row(2025, fcff="10")],
        TerminalMetrics(terminal_growth_rate="10", wacc="10"),
    )

    assert result.highest_severity is Severity.CRITICAL
    assert "TERMINAL_GROWTH_NOT_BELOW_WACC" in codes(result)


def test_high_terminal_value_share_is_flagged():
    result = validate(
        [row(2025, fcff="10")],
        TerminalMetrics(terminal_value_pv="90", enterprise_value="100"),
    )

    assert "HIGH_TERMINAL_VALUE_SHARE" in codes(result)


def test_normal_terminal_value_share_is_accepted():
    result = validate(
        reasonable_rows(),
        TerminalMetrics(terminal_value_pv="50", enterprise_value="100"),
    )

    assert "HIGH_TERMINAL_VALUE_SHARE" not in codes(result)


def test_extreme_forecast_compounding_is_flagged():
    result = validate(
        [row(2025, revenue="1", fcff="1"), row(2027, revenue="200", fcff="200")]
    )

    assert "EXTREME_COMPOUNDING_SCALE" in codes(result)


def test_fcff_identity_discrepancy_is_flagged():
    result = validate(
        [row(2025, fcff="20", nopat="16", da="5", capex="4", delta_nwc="1")]
    )

    assert "FCFF_ACCOUNTING_IDENTITY_INCONSISTENT" in codes(result)


def test_correct_fcff_identity_passes():
    result = validate(
        [row(2025, fcff="16", nopat="16", da="5", capex="4", delta_nwc="1")]
    )

    assert "FCFF_ACCOUNTING_IDENTITY_INCONSISTENT" not in codes(result)


def test_missing_metrics_skip_irrelevant_rules():
    result = validate([row(2025, revenue="100"), row(2026, revenue="110")])

    assert not {
        "FCFF_GROWTH_EXTREME",
        "EXTREME_FCFF_MARGIN",
        "IMPOSSIBLE_OPERATING_MARGIN",
        "FCFF_ACCOUNTING_IDENTITY_INCONSISTENT",
    } & codes(result)


def test_terminal_reinvestment_identity_inconsistency_is_detected():
    result = validate(
        [row(2025, fcff="10")],
        TerminalMetrics(
            terminal_growth_rate="2",
            terminal_roic="10",
            terminal_reinvestment_rate="0",
            terminal_nopat="100",
            terminal_fcff="100",
        ),
    )

    assert "TERMINAL_REINVESTMENT_IDENTITY_INCONSISTENT" in codes(result)
    assert "TERMINAL_HIGH_GROWTH_LOW_REINVESTMENT" in codes(result)


def test_configuration_changes_growth_threshold_behavior():
    rows = [row(2025, fcff="100"), row(2026, fcff="140")]

    assert "FCFF_GROWTH_EXTREME" not in codes(validate(rows))
    configured = ForecastValidationConfig(max_fcff_growth_pct="30")
    assert "FCFF_GROWTH_EXTREME" in codes(validate(rows, config=configured))


def test_findings_serialize_deterministically():
    first = ForecastValidationFinding(
        code="B",
        severity=Severity.WARNING,
        category=ValidationCategory.GROWTH,
        message="b",
    )
    second = ForecastValidationFinding(
        code="A",
        severity=Severity.INFO,
        category=ValidationCategory.HORIZON,
        message="a",
    )
    left = ForecastValidationResult(findings=(first, second))
    right = ForecastValidationResult(findings=(second, first))

    assert left.to_json() == right.to_json()
    assert left.counts == {
        "info": 1,
        "warning": 1,
        "high": 0,
        "critical": 0,
        "error": 0,
        "total": 2,
    }


def test_validator_does_not_mutate_input_rows():
    rows = [row(2025, revenue="100", fcff="10"), row(2026, revenue="110", fcff="11")]
    before = [dict(item) for item in rows]

    validate(rows)

    assert rows == before


def test_multiple_rules_report_independently():
    result = validate(
        [row(2025, revenue="100", fcff="250", ebit="150", gross_profit="100")],
        TerminalMetrics(terminal_growth_rate="5", wacc="4"),
    )

    assert {
        "EXTREME_FCFF_MARGIN",
        "IMPOSSIBLE_OPERATING_MARGIN",
        "TERMINAL_GROWTH_NOT_BELOW_WACC",
    } <= codes(result)


def test_composite_dumped_forecast_and_dcf_artifact_is_adapted_without_mutation():
    artifact = {
        "forecast": {
            "method": "driver_based_fcff",
            "unit": "USD millions",
            "observations": [
                {
                    "forecast_year": 1,
                    "fiscal_year": 2025,
                    "revenue": "100",
                    "operating_income": "20",
                    "tax_rate": "20",
                    "nopat": "16",
                    "depreciation_and_amortization": "5",
                    "capital_expenditures": "4",
                    "change_in_operating_working_capital": "1",
                    "fcff": "16",
                },
                {
                    "forecast_year": 2,
                    "fiscal_year": 2026,
                    "revenue": "110",
                    "operating_income": "22",
                    "tax_rate": "20",
                    "nopat": "17.6",
                    "depreciation_and_amortization": "5.5",
                    "capital_expenditures": "4.4",
                    "change_in_operating_working_capital": "1.1",
                    "fcff": "17.6",
                },
            ],
        },
        "valuation": {
            "parameters": {
                "wacc": "10",
                "perpetual_growth_rate": "2",
            },
            "multistage_plan": {
                "terminal_growth_rate": "2",
                "terminal_return_on_invested_capital": "10",
                "terminal_reinvestment_rate": "20",
                "terminal_capex_to_revenue": "5",
            },
            "terminal_value": {
                "method": "perpetuity_growth",
                "terminal_value": "900",
                "final_cash_flow": "17.6",
                "discount_rate": "10",
                "perpetual_growth_rate": "2",
            },
            "terminal_present_value": {
                "amount": "900",
                "present_value": "90",
                "discount_rate": "10",
            },
            "explicit_forecast_present_value": {
                "total_present_value": "10",
            },
            "enterprise_value": "100",
            "terminal_value_percentage": "90",
        },
    }
    before = deepcopy(artifact)

    context = ForecastValidationContext.from_artifact(artifact)
    result = ForecastValidationService().validate(artifact)

    assert context.rows[0].ebit == 20
    assert context.rows[0].capex == 4
    assert context.rows[0].delta_nwc == 1
    assert context.terminal is not None
    assert context.terminal.wacc == 10
    assert context.terminal.terminal_growth_rate == 2
    assert context.terminal.terminal_value == 900
    assert context.terminal.terminal_value_pv == 90
    assert context.terminal.explicit_forecast_pv == 10
    assert context.terminal.enterprise_value == 100
    assert context.terminal.terminal_value_share_pct == 90
    assert context.terminal.terminal_roic == 10
    assert context.terminal.terminal_reinvestment_rate == 20
    assert context.terminal.terminal_capex_to_revenue == 5
    assert "HIGH_TERMINAL_VALUE_SHARE" in codes(result)
    assert artifact == before


def test_percent_ratio_skips_a_denominator_at_or_below_near_zero():
    assert percent_ratio(Decimal("100"), Decimal("0.5"), Decimal("1")) is None
    assert percent_ratio(Decimal("100"), Decimal("2"), Decimal("1")) == Decimal("5000")


def test_standalone_dcf_artifact_without_forecast_rows_is_not_an_empty_forecast():
    artifact = {
        "parameters": {"wacc": "10", "perpetual_growth_rate": "2"},
        "terminal_value": {"terminal_value": "100"},
        "terminal_present_value": {"present_value": "50"},
        "explicit_forecast_present_value": {"total_present_value": "50"},
        "enterprise_value": "100",
        "terminal_value_percentage": "50",
    }

    result = ForecastValidationService().validate(artifact)

    assert "EMPTY_FORECAST" not in codes(result)


def test_explicitly_empty_forecast_still_reports_empty_forecast():
    result = ForecastValidationService().validate({"observations": []})

    assert "EMPTY_FORECAST" in codes(result)


def test_growth_and_repeated_growth_only_use_adjacent_fiscal_years():
    result = validate(
        [
            row(2025, fcff="100"),
            row(2027, fcff="160"),
            row(2029, fcff="256"),
        ]
    )

    assert "FCFF_GROWTH_EXTREME" not in codes(result)
    assert "FCFF_REPEATED_HYPERGROWTH" not in codes(result)


def test_percentage_point_reinvestment_rate_is_not_fractionally_rescaled():
    result = validate(
        [row(2025, fcff="10")],
        TerminalMetrics(
            terminal_growth_rate="2",
            terminal_roic="10",
            terminal_reinvestment_rate="0.2",
            terminal_nopat="100",
            terminal_fcff="100",
        ),
    )

    low_reinvestment = [
        finding
        for finding in result.findings
        if finding.code == "TERMINAL_HIGH_GROWTH_LOW_REINVESTMENT"
    ]
    inconsistent_rate = [
        finding
        for finding in result.findings
        if finding.code == "TERMINAL_REINVESTMENT_IDENTITY_INCONSISTENT"
    ]
    assert len(low_reinvestment) == 1
    assert inconsistent_rate[0].observed_value == Decimal("0.2")
