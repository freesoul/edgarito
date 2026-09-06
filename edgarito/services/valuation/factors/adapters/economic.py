"""Adapters from economic-graph leaf gaps to factor requests.

The adapter is intentionally a boundary object.  It does not resolve a leaf,
and requester/audit information never participates in :class:`FactorKey`
identity.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from edgarito.schemas.operating_graph import UnresolvedLeafRequirement
from edgarito.services.valuation.factors.contracts import (
    FactorDomain,
    FactorMateriality,
    FactorPeriod,
    FactorPeriodType,
    FactorRequest,
)
from edgarito.services.valuation.factors.identity import canonical_json


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    result = str(value).strip()
    if not result and not allow_empty:
        raise ValueError(f"economic leaf {name} cannot be blank")
    return result


def _period(requirement: UnresolvedLeafRequirement) -> FactorPeriod:
    raw = _text(requirement.fiscal_period, "fiscal_period")
    token = raw.casefold().replace("-", "_").replace(" ", "_")
    if token in {"fy", "year", "annual"}:
        period_type = FactorPeriodType.FY
        period_key = f"FY {requirement.fiscal_year}"
    elif token == "fq":
        period_type = FactorPeriodType.FQ
        period_key = f"FQ {requirement.fiscal_year}"
    elif token in {"q1", "q2", "q3", "q4"}:
        period_type = FactorPeriodType.FQ
        period_key = f"{token.upper()} {requirement.fiscal_year}"
    elif token in {"current_spot", "spot"}:
        period_type = FactorPeriodType.CURRENT_SPOT
        period_key = "current_spot"
    elif token in {"long_term", "longterm"}:
        period_type = FactorPeriodType.LONG_TERM
        period_key = "long_term"
    else:
        raise ValueError(f"unsupported economic leaf fiscal period: {raw!r}")
    return FactorPeriod(
        target_year=requirement.fiscal_year
        if period_type not in {FactorPeriodType.CURRENT_SPOT, FactorPeriodType.LONG_TERM}
        else None,
        period_type=period_type,
        period_key=period_key,
    )


def _scope_key(company_id: str, requirement: UnresolvedLeafRequirement) -> tuple:
    scope = _text(requirement.scope, "scope").casefold()
    scope_id = _text(requirement.scope_id, "scope_id")
    if scope in {"consolidated", "company"}:
        # A company leaf is never a global key.  Keeping the company identifier
        # in subject_id also prevents two requesters from sharing a false leaf.
        return FactorDomain.COMPANY, "company", company_id
    if scope == "operating":
        return FactorDomain.OPERATING, "company", company_id
    if scope in {"business", "segment"}:
        if scope_id.casefold() in {"", "company", "consolidated"}:
            raise ValueError("business economic leaves require a non-company scope_id")
        return FactorDomain.BUSINESS, "business", f"{company_id}:{scope_id}"
    raise ValueError(f"unsupported economic leaf scope: {requirement.scope!r}")


def _materiality(value: Any) -> FactorMateriality:
    raw = getattr(value, "value", value)
    try:
        return FactorMateriality(raw)
    except ValueError:
        return FactorMateriality.UNKNOWN


@dataclass(frozen=True)
class EconomicLeafFactorAdapter:
    """Create one point-in-time factor request for an unresolved leaf."""

    company_id: str
    information_as_of: dt.date | dt.datetime | None = None
    requester: str | None = "economic_graph"

    def __post_init__(self) -> None:
        company_id = _text(self.company_id, "company_id")
        if company_id.casefold() == "global":
            raise ValueError("company_id cannot be the global sentinel")
        object.__setattr__(self, "company_id", company_id)

    def to_request(
        self,
        requirement: UnresolvedLeafRequirement,
        *,
        information_as_of: dt.date | dt.datetime | None = None,
    ) -> FactorRequest:
        if not isinstance(requirement, UnresolvedLeafRequirement):
            requirement = UnresolvedLeafRequirement.model_validate(requirement)
        domain, subject_type, subject_id = _scope_key(self.company_id, requirement)
        metric = _text(requirement.metric, "metric")
        unit = _text(requirement.unit, "unit")
        period = _period(requirement)
        as_of = (
            information_as_of
            if information_as_of is not None
            else self.information_as_of
        )
        if as_of is None:
            raise ValueError(
                "information_as_of must be supplied explicitly at construction "
                "or adaptation time"
            )
        audit = {
            "company_id": self.company_id,
            "requirement_node_id": requirement.node_id,
            "requirement_reason": requirement.reason,
            "scope": requirement.scope,
            "scope_id": requirement.scope_id,
            "metric": requirement.metric,
            "unit": requirement.unit,
            "currency": requirement.currency or "",
            "fiscal_year": str(requirement.fiscal_year),
            "fiscal_period": requirement.fiscal_period,
            "materiality": getattr(requirement.materiality, "value", requirement.materiality),
            "path": "|".join(requirement.path),
            "required_by_relationship_ids": "|".join(
                requirement.required_by_relationship_ids
            ),
        }
        return FactorRequest(
            key={
                "domain": domain,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "metric": metric,
                "period": period,
                "unit": unit,
                "currency": requirement.currency,
            },
            information_as_of=as_of,
            materiality=_materiality(requirement.materiality),
            requester=self.requester,
            audit_context=audit,
        )

    adapt = to_request

    def to_requests(
        self,
        requirements: Iterable[UnresolvedLeafRequirement],
        *,
        information_as_of: dt.date | dt.datetime | None = None,
    ) -> tuple[FactorRequest, ...]:
        """Adapt requirements, deduplicating keys deterministically.

        Duplicate leaves are collapsed only after choosing the lexicographically
        stable audit representative.  This keeps resolver work bounded without
        allowing caller iteration order to affect the result.
        """

        candidates = [
            self.to_request(item, information_as_of=information_as_of)
            for item in requirements
        ]
        by_digest: dict[str, list[FactorRequest]] = {}
        for request in candidates:
            by_digest.setdefault(request.key.digest, []).append(request)
        result: list[FactorRequest] = []
        for digest in sorted(by_digest):
            same_key = sorted(
                by_digest[digest],
                key=lambda item: canonical_json(item.audit_context),
            )
            first = same_key[0]
            if len(same_key) == 1:
                result.append(first)
                continue
            merged = dict(first.audit_context)
            merged["duplicate_requirement_node_ids"] = "|".join(
                sorted({item.audit_context["requirement_node_id"] for item in same_key})
            )
            merged["duplicate_paths"] = "|".join(
                sorted(
                    {
                        item.audit_context["path"]
                        for item in same_key
                        if item.audit_context["path"]
                    }
                )
            )
            result.append(first.model_copy(update={"audit_context": merged}))
        return tuple(result)

    adapt_many = to_requests
    adapt_requirements = to_requests


__all__ = ["EconomicLeafFactorAdapter"]
