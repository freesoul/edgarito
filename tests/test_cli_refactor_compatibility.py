import asyncio
import datetime
from decimal import Decimal
from types import SimpleNamespace

from edgarito.config.valuation import ForecastMethod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.forecasting import FcffForecastParameters


def _forecast_args():
    return SimpleNamespace(
        ticker="TEST",
        profile=None,
        forecast_method=None,
        operating_margin=None,
        tax_rate=None,
        depreciation_to_revenue=None,
        capex_to_revenue=None,
        operating_working_capital_to_revenue=None,
        years=None,
        revenue_growth=None,
        fcf_margin=None,
        historical_window=None,
    )


def test_historical_forecast_seams_control_nested_workflow(monkeypatch):
    import edgarito.cli.__main__ as cli

    configured = SimpleNamespace(
        forecast_years=2,
        revenue_growth=None,
        operating_margin=None,
        tax_rate=None,
        depreciation_to_revenue=None,
        capex_to_revenue=None,
        operating_working_capital_to_revenue=None,
        revenue_anchors={},
        assumption_source_overrides={},
        historical_window=3,
    )
    profile = SimpleNamespace(
        forecast=SimpleNamespace(
            default_method=ForecastMethod.FCFF,
            fcff=configured,
        )
    )
    seen = []
    forecast_result = object()

    class FakeForecastService:
        def required_concepts(self):
            return {"revenue"}

        def forecast(self, financials, parameters):
            seen.append((financials, parameters))
            return forecast_result

    class FakePresenter:
        def render(self, value):
            assert value is forecast_result
            return "forecast"

    async def retrieve(args, granularity, concepts):
        assert granularity is Granularity.ANNUAL
        assert concepts == {"revenue"}
        return "financials"

    def load(_args):
        seen.append("load")
        return profile

    def parameters(_args, _configured):
        seen.append("parameters")
        return FcffForecastParameters(forecast_years=2)

    monkeypatch.setattr(cli, "_load_selected_valuation_profile", load)
    monkeypatch.setattr(cli, "_fcff_parameters", parameters)
    monkeypatch.setattr(cli, "_retrieve_financials", retrieve)
    monkeypatch.setattr(cli, "FcffForecastService", FakeForecastService)
    monkeypatch.setattr(cli, "ForecastConsolePresenter", FakePresenter)

    assert asyncio.run(cli._run_forecast(_forecast_args())) == 0
    assert seen[:2] == ["load", "parameters"]
    assert seen[2][0] == "financials"


def test_historical_kwargs_seam_controls_operating_evidence(monkeypatch):
    import edgarito.cli.__main__ as cli

    calls = []

    class Provider:
        def discover(self, company_id):
            return {"company_id": company_id, "warnings": ()}

    def call_with_supported_kwargs(resolver, kwargs):
        calls.append(kwargs)
        return resolver(company_id=kwargs["company_id"])

    financials = SimpleNamespace(
        company_id="company",
        ticker="TEST",
        industry=None,
        business_archetype=None,
    )
    forecast = SimpleNamespace(observations=())
    monkeypatch.setattr(cli, "_call_with_supported_kwargs", call_with_supported_kwargs)

    evidence, warnings = asyncio.run(
        cli._retrieve_operating_evidence(
            financials,
            forecast,
            "2026-01-01",
            provider=Provider(),
        )
    )

    assert evidence["company_id"] == "company"
    assert warnings == ()
    assert calls and calls[0]["company_id"] == "company"


def test_historical_classification_override_controls_automatic_inputs(
    monkeypatch, tmp_path
):
    import edgarito.cli.__main__ as cli

    source = SimpleNamespace(beta=SimpleNamespace(value=1))
    classification = SimpleNamespace(industry="raw", country="US")
    override_calls = []

    class FakeYahoo:
        def __init__(self, *args):
            del args

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        async def get_company_financials(self, *args, **kwargs):
            return source

        async def get_price_history(self, *args, **kwargs):
            return "history"

    class FakeClassificationNormalizer:
        def normalize_yahoo(self, value):
            assert value is source
            return classification

    class FakeMarketNormalizer:
        def normalize(self, value):
            assert value == "history"
            return SimpleNamespace(currency="USD")

    class FakeDamodaran:
        def __init__(self, *args):
            del args

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        async def get_country_risk_premiums(self, **kwargs):
            return "country"

        async def get_industry_betas(self, **kwargs):
            return "industry"

    class FakeTreasury:
        def __init__(self, *args):
            del args

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        async def get_par_yield(self, *args, **kwargs):
            return "risk-free"

    def apply_override(value, *, sector=None, industry=None):
        override_calls.append((value, sector, industry))
        return "overridden"

    monkeypatch.setattr(cli, "YahooFinanceClient", FakeYahoo)
    monkeypatch.setattr(cli, "CompanyClassificationNormalizer", FakeClassificationNormalizer)
    monkeypatch.setattr(cli, "YahooMarketNormalizer", FakeMarketNormalizer)
    monkeypatch.setattr(cli, "DamodaranClient", FakeDamodaran)
    monkeypatch.setattr(cli, "TreasuryClient", FakeTreasury)
    monkeypatch.setattr(cli, "_apply_classification_overrides", apply_override)

    args = SimpleNamespace(cache_dir=tmp_path, refresh=False, ticker="TEST")
    result = asyncio.run(
        cli._retrieve_automatic_assumption_inputs(
            args,
            SimpleNamespace(ticker="TEST"),
            "USD",
            needs_wacc=True,
            needs_terminal=False,
            sector_override="Technology",
            industry_override="Semiconductors",
        )
    )

    assert result["classification"] == "overridden"
    assert result["risk_free_series"] == "risk-free"
    assert override_calls == [(classification, "Technology", "Semiconductors")]


def test_comparables_dispatch_preserves_facade_override(monkeypatch):
    import edgarito.cli.__main__ as cli

    seen = []

    async def fake_comparables(args):
        seen.append(args.command)
        return 23

    monkeypatch.setattr(cli, "_run_comparables", fake_comparables)

    assert cli.main(["comparables", "--ticker", "AAPL"]) == 23
    assert seen == ["comparables"]


def test_facade_dispatch_uses_a_snapshot_without_mutating_application(monkeypatch):
    import edgarito.cli.__main__ as cli
    from edgarito.cli.use_cases import application

    original_handler = application._run_comparables
    original_key = application.FMP_API_KEY
    seen = []

    async def fake_comparables(args):
        seen.append(args.command)
        assert application._run_comparables is original_handler
        assert application.FMP_API_KEY is original_key
        return 31

    monkeypatch.setattr(cli, "_run_comparables", fake_comparables)
    monkeypatch.setattr(cli, "FMP_API_KEY", "facade-only-key")

    assert cli.main(["comparables", "--ticker", "AAPL"]) == 31
    assert seen == ["comparables"]
    assert application._run_comparables is original_handler
    assert application.FMP_API_KEY is original_key


def test_facade_service_override_reaches_nested_intrinsic_runner(monkeypatch):
    import edgarito.cli.__main__ as cli

    calls = []

    class FakeDividendDiscountService:
        def __init__(self):
            calls.append("constructed")

        def value(self, value):
            calls.append(value)
            return "fake-ddm-result"

    profile = SimpleNamespace(
        valuation=SimpleNamespace(
            dividend_discount=SimpleNamespace(
                mode="gordon",
                dividends=(Decimal("1"),),
                earnings=(),
                payout_ratios=(),
                terminal_return_on_equity=None,
                terminal_payout_ratio=Decimal("0.5"),
            )
        )
    )
    args = SimpleNamespace(dividend=None, payout_ratio=None)
    from edgarito.schemas.valuation.intrinsic import IntrinsicValuationContext

    context = IntrinsicValuationContext(
        company_id="company",
        company_name="Company",
        valuation_date=datetime.date.today(),
        currency="USD",
        diluted_shares=Decimal("1"),
    )
    monkeypatch.setattr(cli, "DividendDiscountService", FakeDividendDiscountService)

    _model, runner = cli._profile_model_runner(
        selected_model="ddm",
        profile=profile,
        context=context,
        annual=(),
        cost_of_equity=Decimal("0.1"),
        terminal_growth=Decimal("0.02"),
        terminal_roe_override=None,
        args=args,
    )

    assert runner() == "fake-ddm-result"
    assert calls[0] == "constructed"
