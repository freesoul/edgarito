import datetime
import json
from decimal import Decimal
from pathlib import Path

import pytest

from edgarito.cli.__main__ import main
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting import (
    FcffForecast,
    FcffForecastObservation,
    FcffForecastParameters,
)
from edgarito.services.valuation import (
    CashFlowTiming,
    FcffDcfCapitalBridge,
    FcffDcfCapitalBridgeResolver,
    FcffDcfParameters,
    FcffDcfResult,
    FcffDcfService,
    ShareRepurchaseParameters,
    TerminalMetric,
    TerminalValueMethod,
)

ROOT = Path(__file__).parents[1]


def test_fcff_dcf_discounts_forecast_terminal_value_and_equity_bridge():
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(
            wacc="10",
            perpetual_growth_rate="2",
        ),
        _capital_bridge(),
    )

    assert result.explicit_forecast_present_value.total_present_value.quantize(
        Decimal("0.001")
    ) == Decimal("181.818")
    assert result.terminal_value.terminal_value == Decimal("1402.5")
    assert result.enterprise_value.quantize(Decimal("0.001")) == Decimal("1340.909")
    assert result.equity_value.quantize(Decimal("0.001")) == Decimal("1240.909")
    assert result.value_per_share.quantize(Decimal("0.001")) == Decimal("124.091")
    assert result.terminal_value_percentage is not None
    assert result.terminal_value_percentage > Decimal(75)
    assert "highly sensitive" in result.warnings[0]
    assert FcffDcfResult.model_validate_json(result.model_dump_json()) == result


def test_mid_year_timing_changes_explicit_cash_flows_but_not_terminal_timing():
    service = FcffDcfService()
    end_period = service.value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    mid_year = service.value(
        _forecast(),
        FcffDcfParameters(
            wacc="10",
            perpetual_growth_rate="2",
            cash_flow_timing=CashFlowTiming.MID_YEAR,
        ),
        _capital_bridge(),
    )

    assert mid_year.explicit_forecast_present_value.cash_flows[0].period == Decimal(
        "0.5"
    )
    assert mid_year.explicit_forecast_present_value.total_present_value > (
        end_period.explicit_forecast_present_value.total_present_value
    )
    assert mid_year.terminal_present_value.period == Decimal(2)
    assert mid_year.terminal_present_value.present_value == (
        end_period.terminal_present_value.present_value
    )
    assert any("mid-year timing" in warning for warning in mid_year.warnings)


def test_current_valuation_date_uses_calendar_stub_periods():
    valuation_date = datetime.date(2026, 7, 1)
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
        valuation_date=valuation_date,
    )

    first = result.explicit_forecast_present_value.cash_flows[0]
    assert result.valuation_date == valuation_date
    assert first.period == Decimal(183) / Decimal(365)
    assert result.terminal_present_value.period == Decimal(548) / Decimal(365)
    assert any("Capital bridge is dated 2025-12-31" in item for item in result.warnings)


def test_current_mid_year_timing_rejects_an_already_elapsed_cash_flow_date():
    with pytest.raises(ValueError, match="Mid-year cash-flow timing"):
        FcffDcfService().value(
            _forecast(),
            FcffDcfParameters(
                wacc="10",
                perpetual_growth_rate="2",
                cash_flow_timing=CashFlowTiming.MID_YEAR,
            ),
            _capital_bridge(),
            valuation_date=datetime.date(2026, 8, 1),
        )


def test_perpetuity_growth_warns_when_explicit_fcff_has_not_converged():
    service = FcffDcfService()
    abrupt = service.value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )
    converged_forecast = _forecast()
    converged_forecast.observations[-1].fcff = Decimal("102")
    converged = service.value(
        converged_forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
    )

    assert any(
        "Final explicit FCFF growth (10.0%)" in warning
        and "sensitive to --years" in warning
        for warning in abrupt.warnings
    )
    assert not any(
        "terminal transition is abrupt" in warning for warning in converged.warnings
    )


def test_exit_multiple_supports_explicit_terminal_metrics():
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(
            wacc="10",
            terminal_method=TerminalValueMethod.EXIT_MULTIPLE,
            exit_multiple="8",
            exit_metric=TerminalMetric.EBITDA,
        ),
        _capital_bridge(),
    )

    assert result.terminal_value.method == TerminalValueMethod.EXIT_MULTIPLE
    assert result.terminal_value.terminal_metric == Decimal("170")
    assert result.terminal_value.terminal_value == Decimal("1360")
    assert any("market-relative scenario" in item for item in result.warnings)


def test_fair_value_buybacks_account_for_cash_and_shares_without_fake_accretion():
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
        share_repurchase_parameters=ShareRepurchaseParameters(
            annual_cash_amounts=(Decimal("100"), Decimal("100")),
            source="test plan",
        ),
    )

    repurchases = result.share_repurchases
    assert repurchases is not None
    assert repurchases.total_cash_spent == Decimal("200")
    assert repurchases.discount_rate == Decimal("10")
    assert repurchases.discount_rate_source == "WACC fallback"
    assert repurchases.purchase_price_source.startswith("model-implied")
    assert repurchases.ending_shares < repurchases.starting_shares
    assert repurchases.residual_equity_value < result.equity_value
    assert repurchases.value_per_remaining_share.quantize(Decimal("0.000001")) == (
        result.value_per_share.quantize(Decimal("0.000001"))
    )
    assert repurchases.accretion_percentage.copy_abs() < Decimal("1e-24")


def test_buybacks_below_intrinsic_value_are_accretive_after_cash_spent():
    result = FcffDcfService().value(
        _forecast(),
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        _capital_bridge(),
        share_repurchase_parameters=ShareRepurchaseParameters(
            annual_cash_amounts=(Decimal("100"), Decimal("100")),
            initial_purchase_price=Decimal("100"),
            price_growth_rate=Decimal("10"),
            discount_rate=Decimal("10"),
        ),
    )

    repurchases = result.share_repurchases
    assert repurchases is not None
    assert repurchases.value_per_remaining_share > result.value_per_share
    assert repurchases.accretion_percentage > 0
    assert any("accretive" in warning for warning in result.warnings)


def test_buyback_schedule_cannot_exceed_explicit_forecast_horizon():
    with pytest.raises(ValueError, match="exceeds the explicit forecast horizon"):
        FcffDcfService().value(
            _forecast(),
            FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
            _capital_bridge(),
            share_repurchase_parameters=ShareRepurchaseParameters(
                annual_cash_amounts=(Decimal("1"), Decimal("1"), Decimal("1")),
            ),
        )


def test_fcff_dcf_rejects_invalid_forecast_or_terminal_economics():
    forecast = _forecast()
    forecast.observations[-1].fcff = Decimal("-1")
    with pytest.raises(ValueError, match="cannot be negative"):
        FcffDcfService().value(
            forecast,
            FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
            _capital_bridge(),
        )
    with pytest.raises(ValueError, match="must exceed"):
        FcffDcfParameters(wacc="2", perpetual_growth_rate="2")


def test_capital_bridge_resolves_normalized_net_debt_and_diluted_shares():
    financials = _financials_with_bridge()
    result = FcffDcfCapitalBridgeResolver().resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
    )

    assert result.gross_debt == Decimal("100")
    assert result.cash_and_equivalents == Decimal("25")
    assert result.net_debt == Decimal("75")
    assert result.diluted_shares == Decimal("10")
    assert result.shares_source == "weighted_average_diluted_shares"


def test_capital_bridge_missing_data_can_be_supplied_explicitly():
    financials = NormalizedCompanyFinancials(
        provider="test",
        company_id="1",
        company_name="Example",
        ticker="EX",
    )
    resolver = FcffDcfCapitalBridgeResolver()
    with pytest.raises(ValueError, match="--net-debt"):
        resolver.resolve(
            financials,
            fiscal_year=2025,
            period_end=datetime.date(2025, 12, 31),
            unit="USD",
        )

    supplied = resolver.resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        net_debt=Decimal("-20"),
        diluted_shares=Decimal("10"),
    )
    assert supplied.net_debt == Decimal("-20")
    assert supplied.gross_debt is None
    assert supplied.shares_source == "explicit profile or CLI override"

    supplied_components = resolver.resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        gross_debt=Decimal("100"),
        cash_and_equivalents=Decimal("25"),
        diluted_shares=Decimal("10"),
    )
    assert supplied_components.net_debt == Decimal("75")
    assert supplied_components.gross_debt == Decimal("100")
    assert supplied_components.net_debt_source.startswith("explicit gross debt")


def test_yahoo_bridge_uses_aggregate_current_debt_without_current_portion():
    financials = _financials_with_bridge()
    financials.provider = "yahoo"
    financials.observations = [
        item
        for item in financials.observations
        if item.concept != FinancialConcept.LONG_TERM_DEBT_CURRENT
    ]

    result = FcffDcfCapitalBridgeResolver().resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
    )

    assert result.gross_debt == Decimal("80")
    assert result.net_debt == Decimal("55")
    assert "aggregate CurrentDebt" in result.net_debt_source


def test_cli_runs_fcff_dcf_from_profile_and_cached_financials(tmp_path, capsys):
    _cache_aapl(tmp_path)
    profile = tmp_path / "dcf.json"
    profile.write_text(
        json.dumps(
            {
                "name": "dcf-test",
                "valuation": {
                    "discount_rates": {"wacc": "8"},
                    "terminal_value": {"perpetual_growth_rate": "2"},
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "valuation",
            "--ticker",
            "AAPL",
            "--years",
            "2",
            "--profile",
            str(profile),
            "--cache-dir",
            str(tmp_path),
            "--user-agent",
            "Edgarito Tests (tests@example.com)",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Model: FCFF DCF" in output
    assert "WACC: 8.00% (explicit valuation profile)" in output
    assert "FY2026E FCFF" in output
    assert "FY2027E FCFF" in output
    assert "Value per share (USD):" in output
    assert "Net debt source: gross debt - cash and equivalents" in output


def _forecast() -> FcffForecast:
    observations = [
        _forecast_observation(1, 2026, "100", "140", "20"),
        _forecast_observation(2, 2027, "110", "150", "20"),
    ]
    return FcffForecast(
        provider="test",
        company_id="1",
        company_name="Example",
        ticker="EX",
        base_fiscal_year=2025,
        base_period_end=datetime.date(2025, 12, 31),
        base_revenue=Decimal("900"),
        base_operating_income=Decimal("130"),
        base_tax_rate=Decimal("20"),
        base_nopat=Decimal("104"),
        base_depreciation_and_amortization=Decimal("20"),
        base_capital_expenditures=Decimal("30"),
        base_operating_working_capital=Decimal("100"),
        base_fcff=Decimal("90"),
        unit="USD",
        parameters=FcffForecastParameters(
            forecast_years=2,
            revenue_growth="5",
            operating_margin="15",
            tax_rate="20",
            depreciation_to_revenue="2",
            capex_to_revenue="3",
            operating_working_capital_to_revenue="10",
        ),
        historical_fiscal_years=(2024, 2025),
        assumption_sources={},
        observations=observations,
    )


def _forecast_observation(
    year: int,
    fiscal_year: int,
    fcff: str,
    operating_income: str,
    depreciation: str,
) -> FcffForecastObservation:
    return FcffForecastObservation(
        forecast_year=year,
        fiscal_year=fiscal_year,
        period_end=datetime.date(fiscal_year, 12, 31),
        revenue_growth=Decimal("5"),
        revenue=Decimal("1000") + Decimal(100 * year),
        operating_margin=Decimal("15"),
        operating_income=Decimal(operating_income),
        tax_rate=Decimal("20"),
        nopat=Decimal(operating_income) * Decimal("0.8"),
        depreciation_to_revenue=Decimal("2"),
        depreciation_and_amortization=Decimal(depreciation),
        capex_to_revenue=Decimal("3"),
        capital_expenditures=Decimal("30"),
        operating_working_capital_to_revenue=Decimal("10"),
        operating_working_capital=Decimal("110"),
        change_in_operating_working_capital=Decimal("10"),
        fcff=Decimal(fcff),
        unit="USD",
    )


def _capital_bridge() -> FcffDcfCapitalBridge:
    return FcffDcfCapitalBridge(
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        net_debt=Decimal("100"),
        diluted_shares=Decimal("10"),
        net_debt_source="test",
        shares_source="test",
    )


def _financials_with_bridge() -> NormalizedCompanyFinancials:
    values = {
        FinancialConcept.SHORT_TERM_DEBT: "10",
        FinancialConcept.LONG_TERM_DEBT_CURRENT: "20",
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "70",
        FinancialConcept.CASH_AND_EQUIVALENTS: "25",
        FinancialConcept.SHARES_OUTSTANDING: "9",
        FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES: "10",
    }
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="1",
        company_name="Example",
        ticker="EX",
        observations=[
            FinancialObservation(
                concept=concept,
                statement=concept.statement,
                value=Decimal(value),
                unit=("shares" if "shares" in concept.value else "USD"),
                granularity=Granularity.ANNUAL,
                fiscal_year=2025,
                fiscal_period=FiscalPeriod.FY,
                period_end=datetime.date(2025, 12, 31),
                provider="test",
                taxonomy="test",
                source_concept=concept.value,
            )
            for concept, value in values.items()
        ],
    )


def _cache_aapl(cache_dir: Path) -> None:
    ticker_path = (
        cache_dir
        / "providers"
        / "edgar"
        / "www.sec.gov"
        / "files"
        / "company_tickers.json"
    )
    facts_path = (
        cache_dir
        / "providers"
        / "edgar"
        / "data.sec.gov"
        / "api"
        / "xbrl"
        / "companyfacts"
        / "CIK0000320193.json"
    )
    ticker_path.parent.mkdir(parents=True)
    facts_path.parent.mkdir(parents=True)
    ticker_path.write_text(
        json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}),
        encoding="utf-8",
    )
    fixture = ROOT / "tests" / "fixtures" / "aapl_facts.json"
    facts_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
