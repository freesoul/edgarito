"""Resolver wrappers for caller-supplied research evidence.

No provider retrieval occurs here; the wrapper only delegates to the existing
point-in-time DirectEvidenceResolver.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from edgarito.services.research.consensus import EvidenceConsensus
from edgarito.services.research.contracts import ResearchEvidence
from edgarito.services.valuation.factors.adapters.research import ResearchFactorAdapter
from edgarito.services.valuation.factors.contracts import FactorKey
from edgarito.services.valuation.factors.resolvers.direct_evidence import (
    DirectEvidenceResolver,
)


class ExistingResearchEvidenceResolver:
    PROTOCOL_ID = "existing_research_evidence"
    resolver_id = PROTOCOL_ID
    protocol_id = PROTOCOL_ID
    priority = 100
    requires_dependencies = False

    def __init__(
        self,
        evidence: Iterable[object] = (),
        *,
        target_key: FactorKey | None = None,
        target_keys: Mapping[object, FactorKey] | None = None,
    ) -> None:
        if isinstance(evidence, (ResearchEvidence, EvidenceConsensus)):
            evidence = (evidence,)
        adapter = ResearchFactorAdapter()
        values = []
        for item in evidence:
            target = target_key
            if target_keys is not None:
                item_id = getattr(item, "evidence_id", None)
                target = target_keys.get(item_id, target)
            values.append(adapter.to_factor_evidence(item, target_key=target))
        self.evidence = tuple(values)
        self._delegate = DirectEvidenceResolver(self.evidence)

    def eligible_evidence(self, request):
        return self._delegate.eligible_evidence(request)

    def can_resolve(self, request, context=None) -> bool:
        return self._delegate.can_resolve(request, context)

    def resolve(self, request, context=None, **kwargs):
        return self._delegate.resolve(request, context, **kwargs)


__all__ = ["ExistingResearchEvidenceResolver"]
