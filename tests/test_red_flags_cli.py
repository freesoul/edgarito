import datetime
from decimal import Decimal

import edgarito.cli.__main__ as cli_main
from edgarito.cli import main
from edgarito.cli.parser import build_parser
from edgarito.cli.presentation.console import RedFlagsConsolePresenter
from edgarito.config.red_flags import RedFlagsProfileLoader
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.red_flags import (
    RedFlag,
    RedFlagCategory,
    RedFlagEvidence,
    RedFlagSeverity,
    RedFlagSourceObservation,
    RedFlagsReport,
)


def test_red_flags_parser_uses_retrieval_arguments_and_profile_alias():
    args = build_parser().parse_args(
        [
            "red-flags",
            "--ticker",
            "AAPL",
            "--period",
            "quarterly",
            "--config",
            "custom.json",
            "--refresh",
            "--verbose",
        ]
    )

    assert args.command == "red-flags"
    assert args.ticker == "AAPL"
    assert args.period == "quarterly"
    assert args.profile.name == "custom.json"
    assert args.refresh is True
    assert args.verbose is True


def test_red_flags_cli_loads_profile_retrieves_required_concepts_and_renders(
    monkeypatch, capsys
):
    calls = {}

    async def fake_retrieve(args, granularity, concepts):
        calls["granularity"] = granularity
        calls["concepts"] = concepts
        return NormalizedCompanyFinancials(
            provider="sec",
            company_id="320193",
            company_name="Apple Inc.",
            ticker="AAPL",
            observations=[
                FinancialObservation(
                    concept=FinancialConcept.REVENUE,
                    statement=FinancialConcept.REVENUE.statement,
                    value=Decimal("100"),
                    unit="USD",
                    granularity=granularity,
                    fiscal_year=2025,
                    fiscal_period=FiscalPeriod.FY,
                    period_end=datetime.date(2025, 9, 27),
                    provider="sec",
                    taxonomy="us-gaap",
                    source_concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                )
            ],
        )

    monkeypatch.setattr(cli_main, "_retrieve_financials", fake_retrieve)

    assert main(["red-flags", "--ticker", "AAPL", "--verbose"]) == 0

    configuration = RedFlagsProfileLoader.load()
    expected_concepts = {
        concept
        for category in configuration.enabled_categories
        for concept in configuration.required_concepts(category)
    }
    assert calls == {"granularity": Granularity.ANNUAL, "concepts": expected_concepts}
    output = capsys.readouterr().out
    assert "AAPL - Apple Inc." in output
    assert "Profile: default" in output
    assert "INCOMPLETE" in output
    assert "WARNINGS" in output
    assert "Required concepts:" in output


def test_red_flags_presenter_keeps_concise_output_short_and_expands_verbose_details():
    source = RedFlagSourceObservation(
        concept=FinancialConcept.NET_INCOME,
        value=Decimal("100.123456"),
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=2025,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(2025, 12, 31),
        provider="sec",
        source_concept="NetIncomeLoss",
    )
    evidence = RedFlagEvidence(
        metric="fcf_to_net_income",
        value=Decimal("20.987654"),
        unit="%",
        threshold=Decimal("80.123456"),
        threshold_unit="%",
        comparison="<",
        formula="100 × free cash flow / net income",
        fiscal_year=2025,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(2025, 12, 31),
        granularity=Granularity.ANNUAL,
        input_concepts=(FinancialConcept.NET_INCOME,),
        source_observations=(source,),
    )
    report = RedFlagsReport(
        provider="sec",
        company_id="1",
        company_name="Example Corp",
        ticker="EXM",
        granularity=Granularity.ANNUAL,
        configuration_name="test",
        evaluated_periods=((2025, FiscalPeriod.FY),),
        flags=(
            RedFlag(
                code="fcf_below_earnings",
                category=RedFlagCategory.FCF_VS_EARNINGS,
                severity=RedFlagSeverity.HIGH,
                message="Free cash flow conversion was below the configured floor.",
                evidence=(evidence,),
            ),
        ),
    )

    presenter = RedFlagsConsolePresenter()
    concise = presenter.render(report)
    verbose = presenter.render(report, verbose=True)

    assert "[HIGH] Fcf Below Earnings (FY2025)" in concise
    assert evidence.formula not in concise
    assert "Evidence:" not in concise
    assert "Formula: 100 × free cash flow / net income (<)" in verbose
    assert "Evidence: fcf_to_net_income = 20.99 %; threshold 80.12 %" in verbose
    assert "Net Income: 100.12 USD (FY2025, NetIncomeLoss)" in verbose
