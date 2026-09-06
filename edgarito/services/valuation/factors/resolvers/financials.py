"""Resolver wrappers for caller-supplied financial and guidance evidence."""

from __future__ import annotations

from collections.abc import Iterable

from edgarito.schemas.guidance.management import ManagementGuidance
from edgarito.schemas.normalization.financials import (
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.valuation.factors.adapters.financials import (
    FinancialsFactorAdapter,
)
from edgarito.services.valuation.factors.contracts import FactorKey
from edgarito.services.valuation.factors.resolvers.direct_evidence import (
    DirectEvidenceResolver,
)


class _DelegatingFinancialResolver:
    requires_dependencies = False

    def __init__(self, evidence) -> None:
        self.evidence = tuple(evidence)
        self._delegate = DirectEvidenceResolver(self.evidence)

    def eligible_evidence(self, request):
        return self._delegate.eligible_evidence(request)

    def can_resolve(self, request, context=None) -> bool:
        return self._delegate.can_resolve(request, context)

    def resolve(self, request, context=None, **kwargs):
        return self._delegate.resolve(request, context, **kwargs)


class DirectFinancialEvidenceResolver(_DelegatingFinancialResolver):
    """Resolve only normalized historical observations supplied by the caller."""

    PROTOCOL_ID = "direct_financial_evidence"
    resolver_id = PROTOCOL_ID
    protocol_id = PROTOCOL_ID
    priority = 200

    def __init__(
        self,
        observations: Iterable[FinancialObservation] | NormalizedCompanyFinancials = (),
        *,
        company_id: str | None = None,
        currency: str | None = None,
        as_of=None,
        availability_policy=None,
        target_key: FactorKey | None = None,
    ) -> None:
        if isinstance(observations, FinancialObservation):
            observations = (observations,)
        if isinstance(observations, NormalizedCompanyFinancials):
            company_id = company_id or observations.company_id
            observations = observations.observations
        if company_id is None:
            raise ValueError("direct financial evidence requires company_id")
        adapter = FinancialsFactorAdapter(
            company_id,
            currency=currency,
            as_of=as_of,
            availability_policy=availability_policy,
        )
        super().__init__(
            adapter.observation(item, target_key=target_key, as_of=as_of)
            for item in observations
        )


class ManagementGuidanceFactorResolver(_DelegatingFinancialResolver):
    """Resolve only caller-supplied, current, evidence-verified guidance."""

    PROTOCOL_ID = "management_guidance_factor"
    resolver_id = PROTOCOL_ID
    protocol_id = PROTOCOL_ID
    priority = 300

    def __init__(
        self,
        guidance: Iterable[ManagementGuidance] = (),
        *,
        company_id: str,
        as_of,
        currency: str | None = None,
        target_key: FactorKey | None = None,
    ) -> None:
        if isinstance(guidance, ManagementGuidance):
            guidance = (guidance,)
        adapter = FinancialsFactorAdapter(company_id, currency=currency, as_of=as_of)
        super().__init__(
            adapter.guidance(item, target_key=target_key, as_of=as_of)
            for item in guidance
        )


__all__ = [
    "DirectFinancialEvidenceResolver",
    "ManagementGuidanceFactorResolver",
]
