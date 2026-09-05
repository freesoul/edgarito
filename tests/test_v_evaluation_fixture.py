"""Focused audit coverage for the frozen Visa evaluation fixture."""

from datetime import date
from decimal import Decimal

from edgarito.evaluation import (
    audit_case,
    canonical_information_content_hash,
    canonical_information_identity,
    load_fixture,
)

SEC_URLS = {
    2022: "https://www.sec.gov/Archives/edgar/data/1403161/000140316122000081/v-20220930.htm",
    2023: "https://www.sec.gov/Archives/edgar/data/1403161/000140316123000099/v-20230930.htm",
}
SEC_2024_URL = "https://www.sec.gov/Archives/edgar/data/1403161/000140316124000058/v-20240930.htm"
SEC_ACCESSIONS = {
    2022: "0001403161-22-000081",
    2023: "0001403161-23-000099",
}
SEC_DATES = {2022: date(2022, 11, 16), 2023: date(2023, 11, 15)}
IR_URLS = {
    2022: "https://s1.q4cdn.com/050606653/files/doc_financials/2022/q4/Q4FY22-Visa-Operational-Performance-Data-FINAL-v2.pdf",
    2023: "https://s1.q4cdn.com/050606653/files/doc_financials/2023/q4/Q4FY23-Visa-Operational-Performance-Data.pdf",
    2024: "https://s1.q4cdn.com/050606653/files/doc_financials/2024/q4/Q4FY24-Visa-Operational-Performance-Data-FINAL.pdf",
}
IR_DATES = {2022: date(2022, 10, 25), 2023: date(2023, 10, 24), 2024: date(2024, 10, 29)}

FINANCIAL_VALUES = {
    2022: {
        "revenue": "29310",
        "operating_income": "18813",
        "pretax_income": "18136",
        "income_tax_expense": "3179",
        "depreciation_and_amortization": "861",
        "capital_expenditures": "970",
        "accounts_receivable": "2020",
        "prepaid_and_other_current_assets": "2668",
        "current_liabilities": "20853",
        "long_term_debt_current": "2250",
        "accounts_payable": "340",
        "accrued_liabilities": "3726",
    },
    2023: {
        "revenue": "32653",
        "operating_income": "21000",
        "pretax_income": "21037",
        "income_tax_expense": "3764",
        "depreciation_and_amortization": "943",
        "capital_expenditures": "1059",
        "accounts_receivable": "2291",
        "prepaid_and_other_current_assets": "2584",
        "current_liabilities": "23098",
        "long_term_debt_current": "0",
        "accounts_payable": "375",
        "accrued_liabilities": "5015",
    },
}
OPERATING_VALUES = {
    2022: {
        "payments_volume": "11607",
        "processed_transactions": "192530",
        "cross_border_growth_ex_intra_europe": "49",
        "cards": "4053",
    },
    2023: {
        "payments_volume": "12338",
        "processed_transactions": "212579",
        "cross_border_growth_ex_intra_europe": "25",
        "cards": "4260",
    },
}


def test_v_fixture_freezes_truthful_financial_and_operating_facts():
    fixture = load_fixture("v")

    assert fixture.manifest.completeness == "partial"
    assert fixture.manifest.operating_evidence_available is True
    assert fixture.case.as_of == date(2023, 11, 15)
    assert fixture.case.fiscal_years == (2024,)
    assert {item.period_end for item in fixture.case.point_in_time_financials.observations} == {
        date(2022, 9, 30),
        date(2023, 9, 30),
    }

    for year, expected in FINANCIAL_VALUES.items():
        rows = {
            row.concept: row
            for row in fixture.manifest.financial_rows
            if row.fiscal_year == year
        }
        assert {concept: row.value for concept, row in rows.items()} == expected
        assert all(
            row.source_id == SEC_ACCESSIONS[year]
            and row.filed == SEC_DATES[year]
            and SEC_URLS[year] in row.provenance
            for row in rows.values()
        )
        assert rows["capital_expenditures"].value in {"970", "1059"}

    observations = {
        (item.fiscal_year, item.driver_id): item
        for item in fixture.case.reasoning_input.observations
    }
    assert {
        (year, metric): str(observations[(year, metric)].value)
        for year, metrics in OPERATING_VALUES.items()
        for metric in metrics
    } == {
        (year, metric): value
        for year, metrics in OPERATING_VALUES.items()
        for metric, value in metrics.items()
    }
    for (year, _metric), observation in observations.items():
        assert observation.segment_id == "company"
        assert observation.scope == "company"
        assert observation.is_total is True
        assert observation.evidence is not None
        assert observation.evidence.accession == IR_URLS[year]
        assert observation.evidence.filing_date == IR_DATES[year]
        assert observation.origin == "first_party_observation"


def test_v_has_no_invented_business_model_definition_or_take_rate():
    fixture = load_fixture("v")
    reasoning = fixture.case.reasoning_input

    assert reasoning.segments == ()
    assert reasoning.definitions == ()
    assert fixture.case.expected_archetypes == ()
    assert {item.driver_id for item in reasoning.observations} == {
        "payments_volume",
        "processed_transactions",
        "cross_border_growth_ex_intra_europe",
        "cards",
    }
    assert all(item.driver_id != "take_rate" for item in reasoning.observations)
    assert "transactions_take_rate" not in {
        str(item) for item in fixture.manifest.expected_archetypes
    }
    assert audit_case(fixture.case).valid


def test_v_availability_manifest_preserves_exact_content_hash_links():
    fixture = load_fixture("v")
    expected = {}
    for item in fixture.case.reasoning_input.observations:
        identity = canonical_information_identity(item, "operating")
        expected[identity] = canonical_information_content_hash(item)
    for item in fixture.case.reasoning_input.historical_facts:
        identity = canonical_information_identity(item, "historical_summary")
        expected[identity] = canonical_information_content_hash(item)

    records = {record.identity: record for record in fixture.case.availability_manifest}
    assert set(records) == set(expected)
    assert {identity: record.content_hash for identity, record in records.items()} == expected
    assert all(
        "2023-11-15" in metadata.as_of.isoformat()
        for metadata in fixture.manifest.evidence_metadata
    )


def test_v_actuals_are_separate_fy2024_outcomes_with_exact_sources():
    fixture = load_fixture("v")
    actuals = fixture.actual_outcomes
    financials = {item.metric: item for item in actuals.observations}

    assert set(financials) == {
        "revenue",
        "ebit",
        "pretax_income",
        "income_tax_expense",
        "depreciation_and_amortization",
        "capex",
        "tax_rate",
    }
    assert {
        metric: str(item.value)
        for metric, item in financials.items()
        if metric != "tax_rate"
    } == {
        "revenue": "35926",
        "ebit": "23595",
        "pretax_income": "23916",
        "income_tax_expense": "4173",
        "depreciation_and_amortization": "1034",
        "capex": "1257",
    }
    assert financials["tax_rate"].value == Decimal("4173") / Decimal("23916") * 100
    assert financials["tax_rate"].value_kind == "derived"
    assert "4,173 / 23,916 * 100" in financials["tax_rate"].reconstruction_provenance
    assert all(
        item.fiscal_year == 2024
        and item.period_end == date(2024, 9, 30)
        and item.source_id == "0001403161-24-000058"
        and item.source_date == date(2024, 11, 13)
        and SEC_2024_URL in (item.reconstruction_provenance or "")
        for item in actuals.observations
    )
    assert all(
        "OWC and FCFF intentionally unavailable" in (item.provenance or "")
        for item in actuals.observations
    )

    operating = {item.driver_id: item for item in actuals.assumption_outcomes}
    assert {
        driver: str(item.actual) for driver, item in operating.items()
    } == {
        "payments_volume": "13190",
        "processed_transactions": "233758",
        "cross_border_growth_ex_intra_europe": "15",
        "cards": "4609",
    }
    assert all(
        item.fiscal_year == 2024
        and item.source_id == IR_URLS[2024]
        and item.source_date == IR_DATES[2024]
        for item in operating.values()
    )
    assert not any("owc" in item.metric.casefold() for item in actuals.observations)
    assert not any("fcff" in item.metric.casefold() for item in actuals.observations)
    assert all(item.fiscal_year in {2022, 2023} for item in fixture.case.reasoning_input.historical_facts)
