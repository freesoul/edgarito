import datetime as dt
from decimal import Decimal

import pytest

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.guidance.management import (
    GuidancePeriodType,
    GuidanceScope,
    GuidanceStatus,
    GuidanceValueKind,
    ManagementGuidance,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    FinancialStatement,
)
from edgarito.schemas.operating import EvidenceReference, OperatingInvestmentProgram
from edgarito.schemas.operating_graph import UnresolvedLeafRequirement
from edgarito.services.forecasting.reasoning.contracts import ForecastReasoningInput
from edgarito.services.research.consensus import reconcile_market_size
from edgarito.services.research.contracts import (
    EvidenceProvenance,
    EvidenceSourceType,
    MarketSizeEvidence,
)
from edgarito.services.valuation.factors import (
    EconomicLeafFactorAdapter,
    FactorAugmentedReasoningInput,
    FactorConfidence,
    FactorEstimate,
    FactorKey,
    FactorPeriod,
    FactorRange,
    FactorRequest,
    FinancialsFactorAdapter,
    ResearchFactorAdapter,
    ResolvedFactorReasoningAdapter,
    valuation_root_keys,
)
from edgarito.services.valuation.factors.resolvers import (
    ExistingResearchEvidenceResolver,
)


def _period(year=2025):
    return FactorPeriod(target_year=year, period_type="FY", period_key=f"FY {year}")


def _estimate(key):
    return FactorEstimate(
        key=key,
        range=FactorRange(low=Decimal("1"), base=Decimal("2"), high=Decimal("3")),
        unit=key.unit,
        currency=key.currency,
        info_as_of=dt.date(2025, 1, 1),
        target_period=key.period,
        confidence=FactorConfidence.MEDIUM,
        methodology="test method",
        resolver="test resolver",
        all_availability_dates=(dt.date(2025, 1, 1),),
        created_at=dt.date(2025, 1, 1),
    )


def _research(date, low, base, high, source):
    return MarketSizeEvidence(
        market="payments",
        geography="Worldwide",
        context={"period": "FY 2025"},
        source_date=date,
        source_type=EvidenceSourceType.ANALYST_ESTIMATE,
        low=low,
        base=base,
        high=high,
        provenance=EvidenceProvenance(source=source),
        unit="USD",
        currency="USD",
    )


def _guidance(**changes):
    values = {
        "metric": "revenue",
        "fiscal_year": 2026,
        "period_type": GuidancePeriodType.FISCAL_YEAR,
        "point": Decimal("120"),
        "value_kind": GuidanceValueKind.MONETARY,
        "currency": "USD",
        "unit": "USD",
        "scope": GuidanceScope.CONSOLIDATED,
        "filing_date": dt.date(2025, 2, 1),
        "accession_number": "guidance-1",
        "filing_form": "8-K",
        "source_document": "ex991.htm",
        "source_document_type": "EX-99.1",
        "supporting_text": "Revenue is expected to be 120.",
        "evidence_verified": True,
        "extraction_model": "test",
    }
    values.update(changes)
    return ManagementGuidance(**values)


def test_leaf_adapter_keeps_company_price_identity_and_audit():
    requirement = UnresolvedLeafRequirement(
        node_id="price_per_call",
        fiscal_year=2026,
        reason="missing leaf",
        path=("revenue", "price_per_call"),
        metric="price_per_call",
        unit="USD / tonne",
        currency="USD",
        required_by_relationship_ids=("revenue-identity",),
    )
    request = EconomicLeafFactorAdapter("ACME", information_as_of=dt.date(2025, 12, 31)).adapt(
        requirement
    )
    assert request.key.subject_id == "acme"
    assert request.key.metric == "price_per_call"
    assert request.audit_context["path"] == "revenue|price_per_call"
    assert request.audit_context["required_by_relationship_ids"] == "revenue-identity"


def test_leaf_adapter_requires_an_explicit_information_cutoff():
    leaf = UnresolvedLeafRequirement(
        node_id="risk",
        fiscal_year=2025,
        reason="missing",
        metric="price_per_call",
        unit="USD / tonne",
        currency="USD",
    )
    with pytest.raises(ValueError, match="information_as_of"):
        EconomicLeafFactorAdapter("ACME").adapt(leaf)
    request = EconomicLeafFactorAdapter("ACME").adapt(
        leaf, information_as_of=dt.date(2025, 3, 1)
    )
    assert request.information_as_of == dt.date(2025, 3, 1)


def test_same_global_research_key_does_not_collide_with_company_leaf():
    research_key = FactorKey(
        domain="market",
        subject_type="market",
        subject_id="payments",
        metric="price_per_call",
        period=_period(2025),
        unit="USD / tonne",
        currency="USD",
    )
    leaf = UnresolvedLeafRequirement(
        node_id="risk",
        fiscal_year=2025,
        reason="missing",
        metric="price_per_call",
        unit="USD / tonne",
        currency="USD",
    )
    request = EconomicLeafFactorAdapter("ACME", information_as_of=dt.date(2025, 3, 1)).adapt(
        leaf
    )
    assert request.key != research_key
    assert request.key.subject_id == "acme"


def test_research_and_consensus_preserve_ranges_and_latest_availability():
    first = _research(dt.date(2025, 1, 1), "10", "12", "15", "first")
    second = _research(dt.date(2025, 2, 1), "11", "13", "17", "second")
    adapter = ResearchFactorAdapter()
    direct = adapter.adapt(first)
    consensus = adapter.adapt(reconcile_market_size([first, second]))
    assert direct.range == FactorRange(low=Decimal("10"), base=Decimal("12"), high=Decimal("15"))
    assert consensus.information_available_on == dt.date(2025, 2, 1)
    assert consensus.range == FactorRange(low=Decimal("10.5"), base=Decimal("12.5"), high=Decimal("16"))
    assert len(consensus.evidence_refs) == 2
    assert consensus.dispersion == Decimal("1")


def test_financial_observation_is_exact_historical_period_and_not_future():
    observation = FinancialObservation(
        concept=FinancialConcept.REVENUE,
        statement=FinancialStatement.INCOME_STATEMENT,
        value=Decimal("100"),
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=2024,
        fiscal_period=FiscalPeriod.FY,
        period_end=dt.date(2024, 12, 31),
        provider="sec",
        taxonomy="us-gaap",
        source_concept="RevenueFromContractWithCustomerExcludingAssessedTax",
        accession_number="10-k-1",
        filed=dt.date(2025, 2, 1),
    )
    evidence = FinancialsFactorAdapter("ACME", currency="USD", as_of=dt.date(2025, 3, 1)).adapt(
        observation
    )
    assert evidence.key.period.target_year == 2024
    assert evidence.key.period.period_type.value == "FY"
    assert evidence.observed_on == observation.period_end
    assert evidence.information_available_on == dt.date(2025, 2, 1)
    assert evidence.provenance.reference == observation.source_concept
    with pytest.raises(ValueError, match="cutoff"):
        FinancialsFactorAdapter("ACME", currency="USD", as_of=dt.date(2024, 12, 31)).adapt(
            observation
        )


def test_guidance_requires_current_verified_future_filed_evidence():
    adapter = FinancialsFactorAdapter("ACME", as_of=dt.date(2025, 3, 1))
    evidence = adapter.adapt(_guidance())
    assert evidence.key.period == _period(2026)
    assert evidence.range == FactorRange.from_point(Decimal("120"))
    with pytest.raises(ValueError, match="withdrawn"):
        adapter.adapt(_guidance(status=GuidanceStatus.WITHDRAWN))
    with pytest.raises(ValueError, match="unverified"):
        adapter.adapt(_guidance(evidence_verified=False))


def test_capacity_only_program_is_unresolved_but_monetary_program_is_evidence():
    common = {
        "program_id": "factory",
        "name": "Factory expansion",
        "fiscal_year": 2026,
        "evidence": EvidenceReference(
            provider="sec", accession="10-k", filing_date=dt.date(2025, 1, 1)
        ),
    }
    capacity = OperatingInvestmentProgram(**common, value=100, unit="units", currency=None)
    monetary = OperatingInvestmentProgram(**common, value=100, unit="USD", currency="USD")
    adapter = FinancialsFactorAdapter("ACME")
    assert adapter.adapt(capacity) is None
    assert adapter.adapt(monetary).range == FactorRange.from_point(Decimal("100"))


def test_factor_reasoning_context_retains_ranges_dependencies_without_overrides():
    dependency = FactorKey(
        domain="company",
        subject_type="company",
        subject_id="ACME",
        metric="revenue",
        period=_period(),
        unit="USD",
        currency="USD",
    )
    key = dependency.model_copy(update={"metric": "margin"})
    estimate = _estimate(key).model_copy(
        update={
            "dependencies": (dependency,),
            "dependency_fingerprints": ((dependency.digest, "abc"),),
            "evidence_refs": ("evidence-1",),
        }
    )
    context = ResolvedFactorReasoningAdapter().to_context([estimate])
    assert context.items[0].range.high == Decimal("3")
    assert context.items[0].dependencies == (dependency,)
    assert context.context_hash == context.digest
    base = ForecastReasoningInput(
        company_id="ACME",
        unit="USD",
        as_of=dt.date(2025, 1, 1),
        forecast_years=(2026,),
    )
    augmented = ResolvedFactorReasoningAdapter().augment(base, context)
    assert isinstance(augmented, FactorAugmentedReasoningInput)
    assert augmented.to_forecast_reasoning_input() is base
    assert augmented.manual_overrides == ()
    assert ResolvedFactorReasoningAdapter().to_research_evidence(estimate) == ()


def test_research_resolver_delegates_exact_adapted_evidence_and_root_contract_is_complete():
    source = _research(dt.date(2025, 1, 1), "10", "12", "15", "source")
    evidence = ResearchFactorAdapter().adapt(source)
    result = ExistingResearchEvidenceResolver([source]).resolve(
        FactorRequest(key=evidence.key, information_as_of=dt.date(2025, 2, 1))
    )
    assert result.resolved
    assert result.estimate.evidence_refs == evidence.evidence_refs
    assert result.estimate.source == evidence.source
    assert {key.metric for key in valuation_root_keys(company_id="ACME")} == {
        "risk_free_rate",
        "equity_risk_premium",
        "beta",
        "debt_spread",
        "target_capital_structure",
        "terminal_growth",
        "terminal_roic",
    }
