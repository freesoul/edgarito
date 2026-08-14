from decimal import Decimal

import pytest

from edgarito.config.classification import CLASSIFICATION
from edgarito.config.guidance import GUIDANCE_DOCUMENTS, GUIDANCE_EXTRACTION
from edgarito.config.operating import (
    OPERATING_EXTRACTION,
    OPERATING_UNITS,
    OPERATING_VOCABULARY,
    OperatingVocabularyLoader,
)
from edgarito.schemas.normalization.classification import Sector
from edgarito.schemas.operating import normalize_operating_unit
from edgarito.services.guidance.documents import (
    _GUIDANCE_AUDIT_PATTERNS,
    _GUIDANCE_CONTEXT_PATTERNS,
    _OPERATING_CONTEXT_PATTERNS,
    GUIDANCE_TERMS,
    OPERATING_TERMS,
)
from edgarito.services.normalization.classification import SECTOR_ALIASES
from edgarito.services.operating.extraction import (
    _OPERATING_AUDIT_PATTERNS,
    operating_keyword_hits,
)
from edgarito.services.operating.vocabulary import (
    GLOBAL_KPI_TERMS,
    INDUSTRY_ALIASES,
    INDUSTRY_KPI_TERMS,
    KpiVocabularyProvider,
)


def test_operating_vocabulary_tables_preserve_legacy_order_and_deduplication():
    assert GLOBAL_KPI_TERMS == (
        ("revenue", "revenue"),
        ("sales", "revenue"),
        ("volume", "volume"),
        ("price", "price"),
        ("customers", "customers"),
        ("capacity", "capacity"),
        ("utilization", "utilization"),
        ("production", "production"),
        ("deliveries", "deliveries"),
        ("shipments", "shipments"),
        ("units", "volume"),
        ("deployments", "deployments"),
        ("subscribers", "subscribers"),
        ("users", "users"),
        ("members", "members"),
        ("arpu", "arpu"),
        ("backlog", "backlog"),
        ("orders", "orders"),
        ("transactions", "transactions"),
        ("stores", "stores"),
        ("sites", "sites"),
    )
    assert INDUSTRY_KPI_TERMS["automotive"] == (
        ("deliveries", "deliveries"),
        ("production", "production"),
        ("asp", "price"),
    )
    assert INDUSTRY_ALIASES["auto_manufacturers"] == "automotive"
    assert KpiVocabularyProvider().normal_terms("Auto Manufacturers")[-1] == (
        "asp",
        "price",
    )


def test_guidance_data_tables_preserve_terms_forms_patterns_and_weights():
    assert GUIDANCE_TERMS[:5] == (
        "earnings",
        "financial results",
        "quarter results",
        "annual results",
        "outlook",
    )
    assert GUIDANCE_TERMS[-5:] == (
        "shipments",
        "deliveries",
        "investment",
        "facility",
        "data center",
    )
    assert OPERATING_TERMS[-3:] == ("millions", "billions", "thousands")
    assert GUIDANCE_DOCUMENTS.current_report_forms == (
        "8-K",
        "8-K/A",
        "6-K",
        "6-K/A",
    )
    assert GUIDANCE_DOCUMENTS.periodic_report_forms == (
        "10-Q",
        "10-Q/A",
        "10-K",
        "10-K/A",
    )
    assert _GUIDANCE_CONTEXT_PATTERNS[0] == (r"\bexpect(?:s|ed|ing)?\b", 10)
    assert _OPERATING_CONTEXT_PATTERNS[-1] == (
        r"\b(?:actual|reported|historical|period ended|three months|six months|nine months)\b",
        4,
    )
    assert _GUIDANCE_AUDIT_PATTERNS == {
        "expect": r"\bexpect(?:s|ed|ing)?\b",
        "capex": r"\bcapex\b",
        "capital expenditures": r"\bcapital expenditures\b",
        "revenue": r"\brevenues?\b",
        "margin": r"\bmargins?\b",
    }


def test_operating_extraction_tables_preserve_audit_counts_and_patterns():
    text = "Deliveries and shipments; prices, ARPU, and facility investments."
    assert _OPERATING_AUDIT_PATTERNS["deliveries"] == r"\bdeliveries\b"
    assert operating_keyword_hits(text)["deliveries"] == 1
    assert operating_keyword_hits(text)["facility"] == 1
    assert OPERATING_EXTRACTION.number_pattern.pattern == (
        r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?"
    )
    assert OPERATING_EXTRACTION.metric_aliases["arpu"] == frozenset(
        {"average_revenue_per_user"}
    )


def test_classification_and_unit_aliases_preserve_legacy_behavior():
    expected_sector = {
        "communication services": Sector.COMMUNICATION_SERVICES,
        "communications": Sector.COMMUNICATION_SERVICES,
        "consumer cyclical": Sector.CONSUMER_DISCRETIONARY,
        "consumer discretionary": Sector.CONSUMER_DISCRETIONARY,
        "consumer defensive": Sector.CONSUMER_STAPLES,
        "consumer staples": Sector.CONSUMER_STAPLES,
        "energy": Sector.ENERGY,
        "finance": Sector.FINANCIALS,
        "financial services": Sector.FINANCIALS,
        "financials": Sector.FINANCIALS,
        "health care": Sector.HEALTHCARE,
        "healthcare": Sector.HEALTHCARE,
        "life sciences": Sector.HEALTHCARE,
        "industrials": Sector.INDUSTRIALS,
        "information technology": Sector.TECHNOLOGY,
        "technology": Sector.TECHNOLOGY,
        "basic materials": Sector.MATERIALS,
        "materials": Sector.MATERIALS,
        "real estate": Sector.REAL_ESTATE,
        "utilities": Sector.UTILITIES,
    }
    assert dict(SECTOR_ALIASES) == expected_sector
    assert dict(CLASSIFICATION.sector_aliases) == expected_sector
    assert OPERATING_UNITS.scale_aliases[0].label == "thousand"
    assert OPERATING_UNITS.scale_aliases[-1].label == "bb"
    assert normalize_operating_unit("USD millions") == ("usd", Decimal("1000000"))


def test_externalized_views_are_immutable_and_missing_files_fail_clearly(tmp_path):
    with pytest.raises(
        FileNotFoundError, match="Operating KPI vocabulary configuration"
    ):
        OperatingVocabularyLoader.load(tmp_path / "missing.json")

    with pytest.raises(TypeError):
        OPERATING_VOCABULARY.industry_aliases["new"] = "value"
    with pytest.raises(AttributeError):
        OPERATING_VOCABULARY.global_terms += (("new", "new"),)
    assert GUIDANCE_EXTRACTION.unit_scales["millions"] == Decimal("1000000")
