import datetime
from decimal import Decimal

from edgarito.schemas.operating import (
    EvidenceReference,
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.services.operating import OperatingHistoryAssembler


def _observation(
    segment,
    metric,
    year,
    value,
    unit,
    period="FY",
    period_key=None,
    accession=None,
    document_name=None,
):
    return OperatingDriverObservation(
        segment_id=segment,
        driver_id=metric,
        fiscal_year=year,
        fiscal_period=period,
        period_key=period_key,
        value=Decimal(value),
        unit=unit,
        origin="reported",
        confidence="high",
        evidence=EvidenceReference(
            provider="sec",
            accession=accession or f"{year}-{metric}",
            document_name=document_name,
            filing_date=datetime.date(year + 1, 1, 1),
            supporting_text=f"Reported {metric} {value} in FY{year}.",
        ),
    )


def _definition():
    return OperatingDriverDefinition(
        driver_id="auto-volume-price",
        archetype=OperatingArchetype.VOLUME_PRICE,
        segment_id="Auto",
        output_metric="revenue",
        input_metrics=("volume", "price"),
        units={"volume": "units", "price": "USD/unit"},
        formula_id="volume_price",
        required_inputs=("volume", "price"),
    )


def test_history_assembles_annual_quarterly_ytd_ltm_units_and_segments():
    result = OperatingHistoryAssembler().assemble(
        [
            _observation(
                "Automotive segment", "segment_revenue", 2025, "100", "USD millions"
            ),
            _observation("Automotive segment", "volume", 2025, "20", "million units"),
            _observation(
                "Energy business", "segment_revenue", 2025, "40", "USD millions"
            ),
            _observation(
                "Automotive segment", "volume", 2025, "5", "million units", "FQ", "Q1"
            ),
            _observation(
                "Automotive segment", "volume", 2025, "11", "million units", "YTD"
            ),
            _observation(
                "Automotive segment",
                "segment_revenue",
                2025,
                "98",
                "USD millions",
                "LTM",
            ),
        ],
        segments=(
            OperatingSegment(segment_id="Automotive segment", name="Automotive"),
            OperatingSegment(segment_id="Energy business", name="Energy"),
        ),
    )

    revenue = next(
        item
        for item in result.observations
        if item.segment_id == "automotive"
        and item.driver_id == "revenue"
        and item.fiscal_period == "FY"
    )
    assert revenue.normalized_value == Decimal("100000000")
    assert revenue.original_unit == "USD millions"
    assert revenue.original_scale == Decimal(1)
    assert result.audit.accepted_periods == ("FY", "FQ", "YTD", "LTM")
    assert result.historical_revenue["automotive"][2025] == Decimal("100000000")
    assert result.company_revenue[2025] == Decimal("140000000")


def test_history_deduplicates_deterministically_derives_price_and_reports_missing_pairs():
    base = _observation("Auto", "segment_revenue", 2025, "100", "USD millions")
    result = OperatingHistoryAssembler().assemble(
        [
            base,
            base,
            _observation("Auto", "volume", 2025, "20", "million units"),
        ],
        segments=(OperatingSegment(segment_id="Auto", name="Auto"),),
        definitions=(_definition(),),
    )

    assert result.audit.deduplicated_observations == 1
    assert result.audit.derived_observations == 1
    assert next(
        item for item in result.observations if item.driver_id == "implied_price"
    ).normalized_value == Decimal("5")
    assert "auto/price" in result.audit.missing_pairs


def test_history_removes_required_optional_alias_collisions():
    definition = OperatingDriverDefinition(
        driver_id="energy-volume-price",
        archetype=OperatingArchetype.VOLUME_PRICE,
        segment_id="energy",
        output_metric="revenue",
        input_metrics=(
            "Megapack deployments",
            "Megapack average selling price",
            "Powerwall deployments",
        ),
        units={
            "Megapack deployments": "deployments",
            "Megapack average selling price": "USD/unit",
            "Powerwall deployments": "deployments",
        },
        formula_id="volume_price",
        required_inputs=("Megapack deployments",),
        optional_inputs=("Megapack average selling price", "Powerwall deployments"),
    )

    result = OperatingHistoryAssembler().assemble((), definitions=(definition,))

    normalized = result.definitions[0]
    assert normalized.required_inputs == ("volume",)
    assert normalized.optional_inputs == ("price",)


def test_history_joins_cross_document_revenue_and_volume_with_auditable_sources():
    result = OperatingHistoryAssembler().assemble(
        [
            _observation("Automotive revenues", "revenue", 2025, "100", "USD millions", accession="a", document_name="10-K.htm"),
            _observation("Vehicle business", "volume", 2025, "20", "million units", accession="b", document_name="8-K.htm"),
        ]
    )

    price = next(item for item in result.observations if item.driver_id == "implied_price")
    assert price.normalized_value == Decimal("5")
    assert price.method.endswith("cross_document")
    assert {item.document_name for item in price.source_provenance} == {"10-K.htm", "8-K.htm"}
    assert result.audit.joins_attempted == 1
    assert result.audit.joins_accepted == 1
    assert result.audit.source_document_count == 2


def test_history_rejects_cross_document_period():
    result = OperatingHistoryAssembler().assemble(
        [
            _observation("Automotive", "revenue", 2025, "100", "USD millions", accession="a", document_name="10-K.htm"),
            _observation("Vehicle business", "volume", 2025, "20", "million units", period="FQ", period_key="Q1", accession="b", document_name="8-K.htm"),
        ]
    )

    assert not any(item.driver_id == "implied_price" for item in result.observations)
    assert result.audit.joins_rejected >= 1
    assert "incompatible_period" in result.audit.join_rejections_by_reason


def test_history_does_not_merge_unrelated_similar_segment_names():
    result = OperatingHistoryAssembler().assemble(
        [
            _observation("Automotive", "revenue", 2025, "100", "USD millions"),
            _observation("Vehicle logistics", "volume", 2025, "20", "million units"),
        ]
    )
    assert {item.segment_id for item in result.observations} == {"automotive", "vehicle_logistics"}
