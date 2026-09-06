"""Adapters for normalized financial and first-party forward evidence.

These adapters only create raw :class:`FactorEvidence`.  In particular, a
reported historical observation is never relabeled as a forward estimate.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.schemas.guidance.management import (
    GuidancePeriodType,
    GuidanceScope,
    GuidanceStatus,
    GuidanceValueKind,
    ManagementGuidance,
)
from edgarito.schemas.normalization.financials import FinancialObservation
from edgarito.schemas.operating import OperatingInvestmentProgram
from edgarito.services.financials.availability import (
    FinancialObservationAvailabilityService,
    ObservationAvailabilityMode,
)
from edgarito.services.valuation.factors.contracts import (
    FactorConfidence,
    FactorDomain,
    FactorEvidence,
    FactorKey,
    FactorPeriod,
    FactorPeriodType,
    FactorProvenance,
    FactorRange,
)
from edgarito.services.valuation.factors.identity import (
    canonicalize_unit,
    stable_digest,
)


def _text(value: Any, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} cannot be blank")
    return result


def _confidence(value: Any, *, default: FactorConfidence = FactorConfidence.MEDIUM):
    value = str(getattr(value, "value", value)).casefold()
    return {
        "low": FactorConfidence.LOW,
        "medium": FactorConfidence.MEDIUM,
        "high": FactorConfidence.HIGH,
        "very_high": FactorConfidence.VERY_HIGH,
    }.get(value, default)


def _financial_period(
    year: int,
    period: FiscalPeriod,
    *,
    period_key: str | None = None,
) -> FactorPeriod:
    if period_key:
        normalized_key = period_key.strip().upper()
        if normalized_key in {"Q1", "Q2", "Q3", "Q4"}:
            normalized_key = f"{normalized_key} {year}"
        elif normalized_key == "FY":
            normalized_key = f"FY {year}"
    else:
        normalized_key = f"FY {year}" if period == FiscalPeriod.FY else f"{period.value} {year}"
    if period == FiscalPeriod.FY:
        return FactorPeriod(
            target_year=year,
            period_type=FactorPeriodType.FY,
            period_key=normalized_key,
        )
    return FactorPeriod(
        target_year=year,
        period_type=FactorPeriodType.FQ,
        period_key=normalized_key,
    )


def _guidance_period(guidance: ManagementGuidance) -> FactorPeriod:
    if guidance.period_type == GuidancePeriodType.LONG_TERM_TARGET:
        return FactorPeriod(
            period_type=FactorPeriodType.LONG_TERM,
            period_key="long_term",
        )
    if guidance.fiscal_year is None:
        raise ValueError("guidance requires an exact future fiscal year")
    if guidance.period_type == GuidancePeriodType.QUARTER:
        if guidance.fiscal_quarter is None:
            raise ValueError("quarter guidance requires an exact fiscal quarter")
        return FactorPeriod(
            target_year=guidance.fiscal_year,
            period_type=FactorPeriodType.FQ,
            period_key=f"Q{guidance.fiscal_quarter} {guidance.fiscal_year}",
        )
    return FactorPeriod(
        target_year=guidance.fiscal_year,
        period_type=FactorPeriodType.FY,
        period_key=f"FY {guidance.fiscal_year}",
    )


def _quarter_end(year: int, quarter: int) -> dt.date:
    month = quarter * 3
    if month == 12:
        return dt.date(year, 12, 31)
    return dt.date(year, month + 1, 1) - dt.timedelta(days=1)


def _period_end(period: FactorPeriod, *, quarter: int | None = None) -> dt.date | None:
    if period.target_year is None:
        return None
    if period.period_type == FactorPeriodType.FY:
        return dt.date(period.target_year, 12, 31)
    if period.period_type == FactorPeriodType.FQ and quarter is not None:
        return _quarter_end(period.target_year, quarter)
    return dt.date(period.target_year, 12, 31)


def _same_target(candidate: FactorKey, target: FactorKey | None) -> FactorKey:
    if target is not None and not isinstance(target, FactorKey):
        target = FactorKey.model_validate(target)
    if target is not None and candidate != target:
        raise ValueError("adapted financial evidence does not match target FactorKey")
    return candidate


@dataclass(frozen=True)
class FinancialEvidenceAvailabilityPolicy:
    """Small explicit wrapper around the repository availability policy."""

    mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME
    snapshot_retrieved_at: dt.datetime | None = None

    def available_on(self, observation: FinancialObservation) -> dt.date:
        return FinancialObservationAvailabilityService().available_on(
            observation,
            mode=ObservationAvailabilityMode(self.mode),
            snapshot_retrieved_at=self.snapshot_retrieved_at,
        )


FinancialAvailabilityPolicy = FinancialEvidenceAvailabilityPolicy


class FinancialsFactorAdapter:
    """Convert financial observations, guidance, and monetary programs."""

    def __init__(
        self,
        company_id: str,
        *,
        currency: str | None = None,
        as_of: dt.date | dt.datetime | None = None,
        availability_policy: FinancialEvidenceAvailabilityPolicy | None = None,
        policy: FinancialEvidenceAvailabilityPolicy | ObservationAvailabilityMode | None = None,
        availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
        snapshot_retrieved_at: dt.datetime | None = None,
    ) -> None:
        self.company_id = _text(company_id, "company_id")
        if self.company_id.casefold() == "global":
            raise ValueError("company_id cannot be the global sentinel")
        self.currency = currency.strip().upper() if currency else None
        self.as_of = as_of
        supplied_policy = availability_policy or policy
        if isinstance(supplied_policy, ObservationAvailabilityMode):
            supplied_policy = FinancialEvidenceAvailabilityPolicy(mode=supplied_policy)
        self.availability_policy = supplied_policy or FinancialEvidenceAvailabilityPolicy(
            mode=availability_mode,
            snapshot_retrieved_at=snapshot_retrieved_at,
        )

    def to_factor_evidence(
        self,
        item: FinancialObservation | ManagementGuidance | OperatingInvestmentProgram,
        *,
        target_key: FactorKey | None = None,
        target_period: FactorPeriod | None = None,
        as_of: dt.date | dt.datetime | None = None,
    ) -> FactorEvidence | None:
        if isinstance(item, FinancialObservation):
            return self.observation(
                item,
                target_key=target_key,
                target_period=target_period,
                as_of=as_of,
            )
        if isinstance(item, ManagementGuidance):
            return self.guidance(
                item,
                target_key=target_key,
                target_period=target_period,
                as_of=as_of,
            )
        if isinstance(item, OperatingInvestmentProgram):
            return self.investment_program(item, target_key=target_key)
        raise TypeError(
            "financial adapter accepts FinancialObservation, ManagementGuidance, "
            "or OperatingInvestmentProgram"
        )

    adapt = to_factor_evidence

    def observation(
        self,
        observation: FinancialObservation,
        *,
        target_key: FactorKey | None = None,
        target_period: FactorPeriod | None = None,
        as_of: dt.date | dt.datetime | None = None,
    ) -> FactorEvidence:
        if not isinstance(observation, FinancialObservation):
            observation = FinancialObservation.model_validate(observation)
        period = _financial_period(observation.fiscal_year, observation.fiscal_period)
        if target_period is not None and period != target_period:
            raise ValueError("financial observation does not match the exact target period")
        currency = getattr(observation, "currency", None) or self.currency
        key = _same_target(
            FactorKey(
                domain=FactorDomain.COMPANY,
                subject_type="company",
                subject_id=self.company_id,
                metric=observation.concept.value,
                period=period,
                unit=observation.unit,
                currency=currency,
            ),
            target_key,
        )
        try:
            available_on = self.availability_policy.available_on(observation)
        except TypeError:
            # Accept the repository service directly as a convenience while
            # keeping this adapter independent of provider retrieval.
            available_on = self.availability_policy.available_on(
                observation,
                mode=ObservationAvailabilityMode.POINT_IN_TIME,
            )
        cutoff = as_of or self.as_of
        if cutoff is not None:
            cutoff_date = cutoff.date() if isinstance(cutoff, dt.datetime) else cutoff
            if observation.period_end > cutoff_date or available_on > cutoff_date:
                raise ValueError("financial observation is not available at the requested cutoff")
        evidence_id = observation.accession_number or stable_digest(
            {
                "company_id": self.company_id,
                "period": period.canonical,
                "concept": observation.source_concept,
                "value": observation.value,
            }
        )
        derivation = observation.derivation_kind.value if observation.derivation_kind else None
        notes = "; ".join(
            part
            for part in (
                f"taxonomy={observation.taxonomy}" if observation.taxonomy else None,
                f"form={observation.form}" if observation.form else None,
                f"derivation_kind={derivation}" if derivation else None,
                f"derivation={observation.derivation}" if observation.derivation else None,
            )
            if part
        )
        provenance = FactorProvenance(
            source=observation.provider,
            source_id=observation.accession_number,
            publisher=observation.provider,
            reference=observation.source_concept,
            notes=notes or None,
        )
        source_type = "reported_first_party_fact" if observation.filed else "derived_observation"
        confidence = FactorConfidence.HIGH if observation.filed else FactorConfidence.MEDIUM
        return FactorEvidence(
            key=key,
            point=observation.value,
            information_available_on=available_on,
            observed_on=observation.period_end,
            source=observation.provider,
            evidence_id=evidence_id,
            all_availability_dates=(available_on,),
            provenance=provenance,
            confidence=confidence,
            source_type=source_type,
            source_types=(source_type,),
            evidence_refs=(evidence_id,),
        )

    adapt_observation = observation
    to_historical_evidence = observation

    def guidance(
        self,
        guidance: ManagementGuidance,
        *,
        target_key: FactorKey | None = None,
        target_period: FactorPeriod | None = None,
        as_of: dt.date | dt.datetime | None = None,
    ) -> FactorEvidence:
        if guidance.status == GuidanceStatus.WITHDRAWN:
            raise ValueError("withdrawn management guidance cannot become factor evidence")
        if not guidance.evidence_verified:
            raise ValueError("unverified management guidance cannot become factor evidence")
        cutoff = as_of or self.as_of
        if cutoff is None:
            raise ValueError("management guidance adaptation requires an as_of date")
        cutoff_date = cutoff.date() if isinstance(cutoff, dt.datetime) else cutoff
        if guidance.filing_date > cutoff_date:
            raise ValueError("management guidance was filed after the requested cutoff")

        period = _guidance_period(guidance)
        if target_period is not None and period != target_period:
            raise ValueError("management guidance does not match the exact target period")
        end = _period_end(
            period,
            quarter=guidance.fiscal_quarter,
        )
        if period.period_type != FactorPeriodType.LONG_TERM and end is not None and end <= cutoff_date:
            raise ValueError("management guidance is not for a future period")

        if guidance.scope == GuidanceScope.CONSOLIDATED:
            domain, subject_type, subject_id = FactorDomain.COMPANY, "company", self.company_id
        elif guidance.scope == GuidanceScope.SEGMENT:
            segment = _text(guidance.segment_name, "guidance segment_name")
            domain, subject_type, subject_id = (
                FactorDomain.BUSINESS,
                "business",
                f"{self.company_id}:{segment}",
            )
        else:
            raise ValueError("management guidance requires a consolidated or segment scope")

        currency = guidance.currency
        if currency is None and guidance.value_kind == GuidanceValueKind.MONETARY:
            currency = self.currency
        if guidance.value_kind == GuidanceValueKind.MONETARY and not currency:
            raise ValueError("monetary guidance requires an exact currency")
        unit = canonicalize_unit(guidance.unit)
        key = _same_target(
            FactorKey(
                domain=domain,
                subject_type=subject_type,
                subject_id=subject_id,
                metric=guidance.metric.value,
                period=period,
                unit=unit,
                currency=currency,
            ),
            target_key,
        )
        values = tuple(value for value in (guidance.point, guidance.low, guidance.high) if value is not None)
        if not values:
            raise ValueError("management guidance requires a numeric point or range")
        if guidance.point is not None and guidance.low is None and guidance.high is None:
            point = guidance.point
            factor_range = FactorRange.from_point(point)
        else:
            low = guidance.low if guidance.low is not None else guidance.midpoint
            high = guidance.high if guidance.high is not None else guidance.midpoint
            base = guidance.point if guidance.point is not None else guidance.midpoint
            factor_range = FactorRange(low=low, base=base, high=high)
        evidence_id = guidance.accession_number
        notes = "; ".join(
            part
            for part in (
                f"status={guidance.status.value}",
                f"basis={guidance.basis.value}",
                f"filing_form={guidance.filing_form}",
                f"source_document={guidance.source_document}",
            )
            if part
        )
        provenance = FactorProvenance(
            source="management_guidance",
            source_id=guidance.accession_number,
            publisher="sec",
            reference=guidance.source_document,
            notes=notes,
        )
        return FactorEvidence(
            key=key,
            low=factor_range.low,
            base=factor_range.base,
            high=factor_range.high,
            information_available_on=guidance.filing_date,
            observed_on=guidance.filing_date,
            source="management_guidance",
            evidence_id=evidence_id,
            all_availability_dates=(guidance.filing_date,),
            provenance=provenance,
            confidence=_confidence(
                guidance.extraction_confidence,
                default=FactorConfidence.HIGH,
            ),
            source_type="reported_first_party_fact",
            source_types=("reported_first_party_fact",),
            evidence_refs=(evidence_id,),
        )

    adapt_guidance = guidance
    to_management_guidance_evidence = guidance

    def investment_program(
        self,
        program: OperatingInvestmentProgram,
        *,
        target_key: FactorKey | None = None,
    ) -> FactorEvidence | None:
        # Capacity/store/unit disclosures remain audit facts, not monetary
        # factors.  Returning None gives callers an explicit unresolved state.
        if not program._is_monetary_unit():
            return None
        if program.fiscal_year is None:
            raise ValueError("monetary investment evidence requires exact timing")
        if not program.currency:
            raise ValueError("monetary investment evidence requires currency")
        if program.segment_id:
            domain, subject_type, subject_id = (
                FactorDomain.BUSINESS,
                "business",
                f"{self.company_id}:{program.segment_id}",
            )
        else:
            domain, subject_type, subject_id = FactorDomain.COMPANY, "company", self.company_id
        period = _financial_period(
            program.fiscal_year,
            FiscalPeriod(program.fiscal_period),
            period_key=program.period_key,
        )
        key = _same_target(
            FactorKey(
                domain=domain,
                subject_type=subject_type,
                subject_id=subject_id,
                metric="capital_expenditures",
                period=period,
                unit=program.unit,
                currency=program.currency,
            ),
            target_key,
        )
        if program.value is not None:
            values = (program.value * program.scale,)
            factor_range = FactorRange.from_point(values[0])
        elif program.low is not None or program.high is not None:
            low = (program.low if program.low is not None else program.high) * program.scale
            high = (program.high if program.high is not None else program.low) * program.scale
            factor_range = FactorRange(low=low, base=(low + high) / 2, high=high)
        else:
            return None
        reference = program.evidence
        source_date = (
            reference.filing_date
            if reference is not None and reference.filing_date is not None
            else dt.date(program.fiscal_year, 12, 31)
        )
        evidence_id = (
            reference.accession
            if reference is not None and reference.accession
            else stable_digest(program.model_dump(mode="python"))
        )
        source = reference.provider if reference is not None else program.source
        provenance = FactorProvenance(
            source=source,
            source_id=reference.accession if reference is not None else None,
            publisher=source,
            reference=reference.document_name if reference is not None else program.program_id,
            locator=reference.source_text_hash if reference is not None else None,
            notes=f"program={program.program_id};status={program.status};purpose={program.purpose or 'unspecified'}",
        )
        source_type = "reported_first_party_fact"
        return FactorEvidence(
            key=key,
            low=factor_range.low,
            base=factor_range.base,
            high=factor_range.high,
            information_available_on=source_date,
            observed_on=source_date,
            source=source,
            evidence_id=evidence_id,
            all_availability_dates=(source_date,),
            provenance=provenance,
            confidence=_confidence(program.confidence),
            source_type=source_type,
            source_types=(source_type,),
            evidence_refs=(evidence_id,),
        )

    adapt_investment_program = investment_program
    to_investment_evidence = investment_program

    def to_factor_evidence_many(
        self,
        items: Iterable[FinancialObservation | ManagementGuidance | OperatingInvestmentProgram],
        **kwargs: Any,
    ) -> tuple[FactorEvidence, ...]:
        values = [
            self.to_factor_evidence(item, **kwargs)
            for item in items
        ]
        unique = {item.fingerprint: item for item in values if item is not None}
        return tuple(
            sorted(unique.values(), key=lambda item: (item.key.digest, item.fingerprint))
        )

    adapt_many = to_factor_evidence_many


class FinancialObservationFactorAdapter(FinancialsFactorAdapter):
    """Focused name for callers adapting only normalized observations."""

    def to_factor_evidence(
        self,
        item: FinancialObservation,
        **kwargs: Any,
    ) -> FactorEvidence:
        return self.observation(item, **kwargs)


class ManagementGuidanceFactorAdapter(FinancialsFactorAdapter):
    """Focused name for callers adapting only current guidance."""

    def to_factor_evidence(
        self,
        item: ManagementGuidance,
        **kwargs: Any,
    ) -> FactorEvidence:
        return self.guidance(item, **kwargs)


class OperatingInvestmentProgramFactorAdapter(FinancialsFactorAdapter):
    """Focused name for monetary operating-program evidence."""

    def to_factor_evidence(
        self,
        item: OperatingInvestmentProgram,
        **kwargs: Any,
    ) -> FactorEvidence | None:
        return self.investment_program(item, **kwargs)


NormalizedFinancialFactorAdapter = FinancialObservationFactorAdapter
ManagementGuidanceAdapter = ManagementGuidanceFactorAdapter
InvestmentProgramFactorAdapter = OperatingInvestmentProgramFactorAdapter


__all__ = [
    "FinancialAvailabilityPolicy",
    "FinancialEvidenceAvailabilityPolicy",
    "FinancialObservationFactorAdapter",
    "FinancialsFactorAdapter",
    "InvestmentProgramFactorAdapter",
    "ManagementGuidanceAdapter",
    "ManagementGuidanceFactorAdapter",
    "NormalizedFinancialFactorAdapter",
    "OperatingInvestmentProgramFactorAdapter",
    "ObservationAvailabilityMode",
]
