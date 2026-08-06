import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from edgarito.cli import main
from edgarito.schemas.providers.edgar.company_facts import CompanyFacts
from edgarito.schemas.valuation.specialized import (
    ExtractedFieldOrigin,
    ExtractedValuationField,
    ExtractionPeriodKind,
    ExtractionReadiness,
    SpecializedInputType,
)
from edgarito.services.valuation.specialized import (
    BiotechInputExtractor,
    ReitInputExtractor,
    ResourceInputExtractor,
    SotpInputExtractor,
    SpecializedValuationExtractor,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _facts(name: str) -> CompanyFacts:
    return CompanyFacts.model_validate_json(
        (FIXTURES / name).read_text(encoding="utf-8")
    )


def _reit_facts() -> CompanyFacts:
    values = {
        "NetIncomeLoss": 100,
        "DepreciationDepletionAndAmortization": 20,
        "GainLossOnSaleOfRealEstate": 5,
        "ImpairmentOfRealEstate": 2,
    }
    return CompanyFacts.model_validate(
        {
            "cik": 123456,
            "entityName": "Example REIT",
            "facts": {
                "dei": {},
                "us-gaap": {
                    concept: {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-01-01",
                                    "end": "2025-12-31",
                                    "val": value,
                                    "accn": "0000123456-26-000001",
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-15",
                                }
                            ]
                        }
                    }
                    for concept, value in values.items()
                },
            },
        }
    )


def test_reit_extractor_builds_a_traceable_ffo_proxy_but_not_affo():
    result = ReitInputExtractor().extract(_reit_facts(), ticker="REIT")
    proxy = result.latest("ffo_proxy")

    assert result.readiness == ExtractionReadiness.PARTIAL
    assert proxy is not None
    assert proxy.value == Decimal("117")
    assert proxy.origin == ExtractedFieldOrigin.DERIVED_PROXY
    assert "GainLossOnSaleOfRealEstate" in proxy.source_concepts
    assert "recurring building" in " ".join(result.missing_inputs)
    assert "not labeled NAREIT FFO" in " ".join(result.limitations)


def test_resource_extractor_retains_cost_history_and_reports_nav_blockers():
    result = ResourceInputExtractor().extract(
        _facts("xom_facts.json"), ticker="XOM", historical_periods=2
    )

    exploration = result.latest("exploration_expense")
    assert result.readiness == ExtractionReadiness.PARTIAL
    assert exploration is not None
    assert exploration.fiscal_year == 2025
    assert exploration.fiscal_period == "Q3"
    assert exploration.period_kind == ExtractionPeriodKind.YEAR_TO_DATE
    assert exploration.value == Decimal("464000000")
    assert len({field.period_end for field in result.fields}) <= 2
    assert "reserve quantities" in result.missing_inputs[0]


def test_biotech_extractor_gets_r_and_d_but_does_not_invent_pipeline_programs():
    result = BiotechInputExtractor().extract(_facts("jnj_facts.json"), ticker="JNJ")

    research = result.latest("research_and_development_expense")
    assert research is not None
    assert research.fiscal_year == 2025
    assert research.fiscal_period == "Q3"
    assert research.value == Decimal("10413000000")
    assert not any(field.dimensions.get("program") for field in result.fields)
    assert "program and indication names" in result.missing_inputs


def test_sotp_extractor_reports_segment_count_without_fake_segment_members():
    result = SotpInputExtractor().extract(_facts("wmt_facts.json"), ticker="WMT")
    segment_count = result.latest("reportable_segment_count")

    assert segment_count is not None
    assert segment_count.value == Decimal("3")
    assert segment_count.unit.casefold() == "segment"
    assert "named reportable segments" in result.missing_inputs
    assert not any(field.dimensions for field in result.fields)


def test_dispatcher_and_specialized_schema_preserve_type_and_provenance():
    result = SpecializedValuationExtractor().extract(
        _facts("jnj_facts.json"),
        SpecializedInputType.BIOTECH,
        ticker="JNJ",
        historical_periods=1,
    )

    assert result.input_type == SpecializedInputType.BIOTECH
    assert {field.fiscal_year for field in result.fields} == {2025}
    assert {field.fiscal_period for field in result.fields} == {"Q3"}
    assert type(result).model_validate_json(result.model_dump_json()) == result

    with pytest.raises(ValidationError, match="require a derivation"):
        ExtractedValuationField(
            name="proxy",
            value=Decimal(1),
            unit="USD",
            fiscal_year=2025,
            period_end="2025-12-31",
            origin=ExtractedFieldOrigin.DERIVED_PROXY,
            source_concepts=("NetIncomeLoss",),
        )


def test_cli_extracts_cached_sec_specialized_inputs(tmp_path, capsys):
    ticker_path = (
        tmp_path
        / "providers"
        / "edgar"
        / "www.sec.gov"
        / "files"
        / "company_tickers.json"
    )
    facts_path = (
        tmp_path
        / "providers"
        / "edgar"
        / "data.sec.gov"
        / "api"
        / "xbrl"
        / "companyfacts"
        / "CIK0000034088.json"
    )
    ticker_path.parent.mkdir(parents=True)
    facts_path.parent.mkdir(parents=True)
    ticker_path.write_text(
        json.dumps(
            {"0": {"cik_str": 34088, "ticker": "XOM", "title": "Exxon Mobil"}}
        ),
        encoding="utf-8",
    )
    facts_path.write_text(
        (FIXTURES / "xom_facts.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "specialized-inputs",
            "--ticker",
            "XOM",
            "--type",
            "resource",
            "--history",
            "1",
            "--cache-dir",
            str(tmp_path),
            "--user-agent",
            "Edgarito Tests (tests@example.com)",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Extractor: Natural resource" in output
    assert "Readiness: partial" in output
    assert "exploration_expense" in output
    assert "reserve quantities" in output
