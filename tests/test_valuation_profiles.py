import json
from decimal import Decimal
from pathlib import Path

import pytest

from edgarito.cli.__main__ import build_parser, main
from edgarito.config import ForecastMethod, ValuationProfileLoader
from edgarito.config.valuation import CashFlowTiming, ForecastValuationProfile
from edgarito.schemas.normalization.classification import Sector
from edgarito.services.valuation import ValuationInput

ROOT = Path(__file__).parents[1]
AAPL_FIXTURE = ROOT / "tests" / "fixtures" / "aapl_facts.json"


def test_default_profile_is_loaded_from_the_root_configs_directory():
    path = ValuationProfileLoader.default_path()
    profile = ValuationProfileLoader.load()

    assert path == ROOT / "configs" / "valuation" / "default.json"
    assert profile.name == "default"
    assert profile.forecast.default_method == ForecastMethod.FCFF
    assert profile.forecast.fcff.forecast_years == 5
    assert profile.forecast.fcff.historical_window == 3
    assert profile.valuation.cash_flow_timing == CashFlowTiming.END_OF_PERIOD
    assert profile.valuation.discount_rates.risk_free_rate is None
    assert profile.valuation.discount_rates.wacc is None
    assert profile.valuation.terminal_value.perpetual_growth_rate is None
    assert profile.valuation.multistage.enabled
    assert profile.valuation.multistage.stable_growth_rate is None
    assert profile.valuation.multistage.max_annual_growth_fade == Decimal("3")
    assert profile.valuation.multistage.extend_to_stable
    assert profile.valuation.share_repurchases.annual_cash_amounts == ()
    assert profile.model_selection.sector is None
    assert profile.model_selection.industry is None
    assert profile.comparables.max_peers == 8
    assert profile.specialized_inputs.history == 5
    assert ForecastValuationProfile.model_validate_json(profile.model_dump_json()) == (
        profile
    )


def test_race_profile_overrides_provider_classification_with_luxury_economics():
    profile = ValuationProfileLoader.load(ROOT / "configs" / "valuation" / "race.json")

    assert profile.name == "race"
    assert profile.model_selection.sector == Sector.CONSUMER_DISCRETIONARY
    assert profile.model_selection.industry == "Luxury Goods"
    assert profile.forecast.fcff.operating_margin == (
        Decimal("29.5"),
        Decimal("29.625"),
        Decimal("29.75"),
        Decimal("29.875"),
        Decimal("30"),
    )
    assert profile.forecast.fcff.capex_to_revenue == (Decimal("10.8"),)
    assert profile.valuation.discount_rates.country_risk_premium == Decimal("0.71")
    assert profile.valuation.capital_bridge.net_debt == Decimal("32000000")
    assert profile.valuation.terminal_value.exit_multiple == Decimal("22.5")
    assert profile.valuation.multistage.stable_growth_rate == Decimal("2.9")
    assert profile.valuation.share_repurchases.annual_cash_amounts == (
        Decimal("700000000"),
        Decimal("700000000"),
        Decimal("700000000"),
        Decimal("700000000"),
        Decimal("700000000"),
    )
    assert profile.valuation.share_repurchases.initial_purchase_price is None


def test_ticker_profile_is_auto_selected_but_explicit_profile_wins():
    automatic = ValuationProfileLoader.load_for_ticker("MSFT")
    explicit = ValuationProfileLoader.load_for_ticker(
        "MSFT", ROOT / "configs" / "valuation" / "default.json"
    )

    assert automatic.name == "msft"
    assert (
        automatic.valuation.multistage.terminal_return_on_invested_capital
        == Decimal("40")
    )
    assert automatic.valuation.multistage.depreciable_asset_life_years == 6
    assert explicit.name == "default"


def test_custom_profile_can_partially_override_defaults(tmp_path):
    path = tmp_path / "growth.json"
    path.write_text(
        json.dumps(
            {
                "name": "growth",
                "forecast": {
                    "default_method": "simplified",
                    "simplified": {
                        "forecast_years": 2,
                        "historical_window": 1,
                        "revenue_growth": ["12", "8"],
                        "free_cash_flow_margin": "20",
                    },
                },
                "valuation": {
                    "cash_flow_timing": "mid_year",
                    "discount_rates": {
                        "risk_free_rate": "4.2",
                        "equity_risk_premium": "4.5",
                    },
                    "terminal_value": {"perpetual_growth_rate": "2"},
                },
                "comparables": {"max_peers": 6, "preferred_minimum": 4},
            }
        ),
        encoding="utf-8",
    )

    profile = ValuationProfileLoader.load(path)

    assert profile.forecast.default_method == ForecastMethod.SIMPLIFIED
    assert profile.forecast.simplified.revenue_growth == (
        Decimal("12"),
        Decimal("8"),
    )
    assert profile.forecast.simplified.free_cash_flow_margin == (Decimal("20"),)
    assert profile.forecast.fcff.forecast_years == 5
    assert profile.valuation.cash_flow_timing == CashFlowTiming.MID_YEAR
    assert profile.valuation.discount_rates.risk_free_rate == Decimal("4.2")
    assert profile.valuation.terminal_value.perpetual_growth_rate == Decimal("2")
    assert profile.comparables.minimum_score == 50


def test_profile_validation_rejects_unknown_or_invalid_parameters(tmp_path):
    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps({"forecast": {"fcff": {"revenue_groth": "10"}}}),
        encoding="utf-8",
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"comparables": {"max_peers": 3, "preferred_minimum": 5}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="revenue_groth"):
        ValuationProfileLoader.load(unknown)
    with pytest.raises(ValueError, match="cannot exceed"):
        ValuationProfileLoader.load(invalid)
    with pytest.raises(FileNotFoundError, match="not found"):
        ValuationProfileLoader.load(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="minimum_transition_years"):
        ForecastValuationProfile.model_validate(
            {
                "valuation": {
                    "multistage": {
                        "minimum_transition_years": 8,
                        "maximum_transition_years": 4,
                    }
                }
            }
        )


def test_configured_rates_are_exposed_as_ready_selector_inputs():
    profile = ForecastValuationProfile.model_validate(
        {
            "valuation": {
                "discount_rates": {
                    "risk_free_rate": "4",
                    "levered_beta": "1.1",
                    "equity_risk_premium": "5",
                    "pretax_cost_of_debt": "5",
                    "normalized_tax_rate": "25",
                    "market_value_equity": "800",
                    "market_value_debt": "200",
                },
                "terminal_value": {"perpetual_growth_rate": "2"},
            }
        }
    )

    assert profile.configured_valuation_inputs == frozenset(
        {
            ValuationInput.COST_OF_EQUITY,
            ValuationInput.WACC,
            ValuationInput.TERMINAL_GROWTH,
        }
    )


def test_profile_accepts_a_manual_gross_debt_and_cash_bridge():
    profile = ForecastValuationProfile.model_validate(
        {
            "valuation": {
                "capital_bridge": {
                    "gross_debt": "100",
                    "cash_and_equivalents": "25",
                    "diluted_shares": "10",
                }
            }
        }
    )

    bridge = profile.valuation.capital_bridge
    assert bridge.gross_debt == Decimal("100")
    assert bridge.cash_and_equivalents == Decimal("25")
    with pytest.raises(ValueError, match="must be set together"):
        ForecastValuationProfile.model_validate(
            {"valuation": {"capital_bridge": {"gross_debt": "100"}}}
        )


def test_profile_cli_options_are_unset_until_profile_resolution():
    parser = build_parser()
    forecast = parser.parse_args(["forecast", "--ticker", "AAPL"])
    comparables = parser.parse_args(
        ["comparables", "--ticker", "AAPL", "--peer", "MSFT"]
    )

    assert forecast.profile is None
    assert forecast.forecast_method is None
    assert forecast.years is None
    assert forecast.historical_window is None
    assert comparables.profile is None
    assert comparables.max_peers is None
    assert comparables.require_same_sector is None


def test_cli_uses_profile_forecast_parameters_then_applies_cli_overrides(
    tmp_path, capsys
):
    _cache_aapl(tmp_path)
    profile_path = tmp_path / "fcff.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "two-year-fcff",
                "forecast": {
                    "fcff": {
                        "forecast_years": 2,
                        "historical_window": 2,
                        "revenue_growth": "5",
                        "operating_margin": "25",
                        "tax_rate": "21",
                        "depreciation_to_revenue": "4",
                        "capex_to_revenue": "3",
                        "operating_working_capital_to_revenue": "10",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    common = [
        "forecast",
        "--ticker",
        "AAPL",
        "--profile",
        str(profile_path),
        "--cache-dir",
        str(tmp_path),
        "--user-agent",
        "Edgarito Tests (tests@example.com)",
    ]

    assert main(common) == 0
    profile_output = capsys.readouterr().out
    assert "FY2026E" in profile_output
    assert "FY2027E" in profile_output
    assert "Revenue Growth: explicit" in profile_output

    assert main([*common, "--years", "1", "--revenue-growth", "7"]) == 0
    override_output = capsys.readouterr().out
    assert "FY2026E" in override_output
    assert "FY2027E" not in override_output
    assert "7.0%" in override_output


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
    facts_path.write_text(AAPL_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
