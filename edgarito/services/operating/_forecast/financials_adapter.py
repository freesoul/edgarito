"""Adapter from normalized company financials to operating evidence.

This is a deterministic boundary only. It does not retrieve another provider,
ask an LLM, allocate company facts, or calculate a forecast.
"""

from __future__ import annotations

import datetime
from typing import Any

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.operating import EvidenceReference, OperatingDriverObservation
from edgarito.services.financials.availability import (
    FinancialObservationAvailabilityService,
    ObservationAvailabilityMode,
)

_CONCEPT_DRIVERS = {
    FinancialConcept.REVENUE: "revenue",
    FinancialConcept.GROSS_PROFIT: "gross_profit",
    FinancialConcept.RESEARCH_AND_DEVELOPMENT_EXPENSE: "r_and_d",
    FinancialConcept.SELLING_GENERAL_AND_ADMINISTRATIVE_EXPENSE: "sg_and_a",
    FinancialConcept.OPERATING_INCOME: "operating_income",
}
_EXPENSES = {"r_and_d", "sg_and_a"}


def normalized_company_financials_to_operating_observations(
    financials: NormalizedCompanyFinancials | Any,
    *,
    as_of: datetime.date | None = None,
    availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
    availability_service: FinancialObservationAvailabilityService | None = None,
) -> tuple[OperatingDriverObservation, ...]:
    """Expose annual/quarterly normalized company facts as company evidence."""

    if not isinstance(financials, NormalizedCompanyFinancials):
        financials = NormalizedCompanyFinancials.model_validate(financials)
    availability = availability_service or FinancialObservationAvailabilityService()
    result: list[OperatingDriverObservation] = []
    for item in financials.observations:
        driver = _CONCEPT_DRIVERS.get(item.concept)
        if driver is None:
            continue
        if as_of is not None and not availability.is_available(
            item,
            as_of=as_of,
            mode=availability_mode,
            snapshot_retrieved_at=financials.retrieved_at,
        ):
            continue
        fiscal_period, period_key = _operating_period(item.fiscal_period)
        reference = EvidenceReference(
            provider=item.provider,
            accession=item.accession_number,
            filing_date=item.filed,
            document_name=item.form or item.source_concept,
            supporting_text=(
                f"Normalized {item.concept.value} from {item.source_concept}"
            ),
        )
        value = abs(item.value) if driver in _EXPENSES else item.value
        result.append(
            OperatingDriverObservation(
                segment_id="company",
                driver_id=driver,
                fiscal_year=item.fiscal_year,
                fiscal_period=fiscal_period,
                period_key=period_key,
                value=value,
                unit=item.unit,
                currency=item.unit.upper() if len(item.unit) == 3 else None,
                scope="company",
                scope_evidence="normalized company financial fact",
                is_total=True,
                # A period-reconstructed normalized fact is still a reported
                # company fact for precedence purposes; its method retains the
                # reconstruction audit instead of demoting it to a forecast.
                origin="reported",
                confidence="high" if item.filed is not None else "medium",
                method=(
                    f"normalized_{item.derivation_kind.value}"
                    if item.derivation_kind is not None
                    else "normalized_company_financial_fact"
                ),
                provenance=reference,
                evidence=reference,
                source_provenance=(reference,),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.fiscal_year,
                item.fiscal_period,
                item.period_key or "",
                item.driver_id,
            ),
        )
    )


class NormalizedFinancialsOperatingAdapter:
    """Small object wrapper for dependency-injection callers."""

    def adapt(self, financials, **kwargs):
        return normalized_company_financials_to_operating_observations(
            financials, **kwargs
        )

    __call__ = adapt


def _operating_period(period: FiscalPeriod | str) -> tuple[str, str | None]:
    normalized = str(getattr(period, "value", period)).strip().upper()
    if normalized == "FY":
        return "FY", None
    if normalized in {"Q1", "Q2", "Q3", "Q4"}:
        return "FQ", normalized
    if normalized in {"YTD", "LTM"}:
        return normalized, None
    return "FY", None


# Descriptive aliases keep the adapter easy to discover without creating a
# second implementation.
adapt_normalized_company_financials = normalized_company_financials_to_operating_observations
normalized_financials_to_operating_observations = normalized_company_financials_to_operating_observations
normalized_financials_to_operating_evidence = normalized_company_financials_to_operating_observations
adapt_normalized_financials = normalized_company_financials_to_operating_observations


__all__ = [
    "adapt_normalized_company_financials",
    "normalized_company_financials_to_operating_observations",
    "normalized_financials_to_operating_observations",
    "normalized_financials_to_operating_evidence",
    "adapt_normalized_financials",
    "NormalizedFinancialsOperatingAdapter",
]
