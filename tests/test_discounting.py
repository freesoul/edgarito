from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.services.valuation import (
    CashFlow,
    CostOfEquityResult,
    DiscountRateService,
    PresentValueResult,
    PresentValueService,
    TerminalValueMethod,
    TerminalValueService,
    WaccResult,
)


def test_capm_cost_of_equity_and_beta_levering_are_percentage_point_based():
    levered_beta = DiscountRateService.lever_beta(
        unlevered_beta="0.8",
        market_value_debt="500",
        market_value_equity="1000",
        normalized_tax_rate="25",
    )
    result = DiscountRateService.cost_of_equity(
        risk_free_rate="4",
        levered_beta=levered_beta,
        equity_risk_premium="5",
        country_risk_premium="0.5",
    )

    assert levered_beta == Decimal("1.1000")
    assert result.cost_of_equity == Decimal("10.0000")
    assert result.formula.startswith("risk-free rate")


def test_wacc_uses_market_value_weights_and_after_tax_debt_cost():
    result = DiscountRateService.wacc(
        cost_of_equity="10",
        pretax_cost_of_debt="5",
        normalized_tax_rate="25",
        market_value_equity="800",
        market_value_debt="200",
    )

    assert result.equity_weight == Decimal("0.8")
    assert result.debt_weight == Decimal("0.2")
    assert result.after_tax_cost_of_debt == Decimal("3.75")
    assert result.wacc == Decimal("8.750")
    assert WaccResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "cost_of_equity": "10",
                "pretax_cost_of_debt": "5",
                "normalized_tax_rate": "101",
                "market_value_equity": "800",
                "market_value_debt": "200",
            },
            "between 0% and 100%",
        ),
        (
            {
                "cost_of_equity": "10",
                "pretax_cost_of_debt": "5",
                "normalized_tax_rate": "25",
                "market_value_equity": "0",
                "market_value_debt": "0",
            },
            "positive total capital",
        ),
    ],
)
def test_wacc_rejects_invalid_capital_or_tax_inputs(arguments, message):
    with pytest.raises(ValueError, match=message):
        DiscountRateService.wacc(**arguments)


def test_present_value_discounts_multiple_cash_flows_and_preserves_the_bridge():
    result = PresentValueService.discount(
        (
            CashFlow(amount="110", period=1, label="Year 1 FCFF"),
            CashFlow(amount="121", period=2, label="Year 2 FCFF"),
        ),
        discount_rate="10",
        unit="USD",
    )

    assert result.total_present_value.quantize(Decimal("0.000001")) == Decimal(
        "200.000000"
    )
    assert result.cash_flows[0].discount_factor.quantize(
        Decimal("0.000001")
    ) == Decimal("0.909091")
    assert result.cash_flows[1].label == "Year 2 FCFF"
    assert PresentValueResult.model_validate_json(result.model_dump_json()) == result


def test_fractional_period_supports_mid_year_discounting():
    factor = PresentValueService.discount_factor("10", "0.5")
    value = PresentValueService.present_value("100", "10", "0.5")

    assert factor.quantize(Decimal("0.000000001")) == Decimal("0.953462589")
    assert value.quantize(Decimal("0.000001")) == Decimal("95.346259")


def test_present_value_rejects_empty_flows_and_invalid_periods():
    with pytest.raises(ValidationError, match="At least one cash flow"):
        PresentValueService.discount((), "8", "USD")
    with pytest.raises(ValueError, match="greater than -100%"):
        PresentValueService.discount_factor("-100", 1)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        CashFlow(amount="100", period="-0.5")


def test_perpetuity_growth_terminal_value_uses_next_period_cash_flow():
    result = TerminalValueService.perpetuity_growth(
        final_cash_flow="100",
        discount_rate="8",
        perpetual_growth_rate="2",
    )

    assert result.method == TerminalValueMethod.PERPETUITY_GROWTH
    assert result.terminal_value == Decimal("1700")
    discounted = PresentValueService.present_value(result.terminal_value, "8", 5)
    assert discounted < result.terminal_value

    repeating = TerminalValueService.perpetuity_growth("123", "9.1", "2.3")
    assert type(repeating).model_validate_json(repeating.model_dump_json()) == repeating


def test_perpetuity_growth_requires_discount_rate_above_growth():
    with pytest.raises(ValueError, match="must exceed"):
        TerminalValueService.perpetuity_growth("100", "3", "3")
    with pytest.raises(ValueError, match="cannot be negative"):
        TerminalValueService.perpetuity_growth("-1", "8", "2")


def test_exit_multiple_terminal_value_is_a_separate_auditable_method():
    result = TerminalValueService.exit_multiple("500", "8")

    assert result.method == TerminalValueMethod.EXIT_MULTIPLE
    assert result.terminal_value == Decimal("4000")
    assert result.discount_rate is None
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_result_schemas_reject_inconsistent_manual_calculations():
    valid = DiscountRateService.cost_of_equity("4", "1.2", "5")
    with pytest.raises(ValidationError, match="does not match"):
        CostOfEquityResult(
            risk_free_rate=valid.risk_free_rate,
            levered_beta=valid.levered_beta,
            equity_risk_premium=valid.equity_risk_premium,
            cost_of_equity="99",
        )
