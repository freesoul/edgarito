"""Company investor-relations fallback for unresolved operating gaps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from edgarito.services.operating._discovery.contracts import DiscoveryState
from edgarito.services.operating._discovery.recovery import (
    _assemble,
    gap_terms,
    merge_extraction_entry,
    targeted_gap_context,
)


async def recover_from_ir(
    service: Any,
    state: DiscoveryState,
    *,
    history: Any,
    financials: Any | None,
    profile_metadata: Mapping[str, Any] | None,
    company_id: str,
    as_of,
    extraction_years: tuple[int, ...],
) -> tuple[Any, tuple[Any, ...], str | None]:
    """Use an injected IR provider only when metadata and gaps justify it."""

    unresolved_before_ir = history.audit.gaps_unresolved
    if not unresolved_before_ir:
        return history, (), None
    profile = dict(profile_metadata or {})
    source_profile = getattr(financials, "profile", None)
    if source_profile is not None:
        if hasattr(source_profile, "model_dump"):
            profile = {**source_profile.model_dump(), **profile}
        elif isinstance(source_profile, Mapping):
            profile = {**source_profile, **profile}
    ir_url = next(
        (
            str(profile[key]).strip()
            for key in (
                "investor_website",
                "investorWebsite",
                "ir_url",
                "irUrl",
            )
            if profile.get(key)
        ),
        None,
    )
    if not ir_url:
        return (
            history,
            (),
            "IR fallback unavailable: profile metadata has no investor-relations URL",
        )
    if service._ir_fallback is None:
        return history, (), "IR fallback unavailable: no IR provider configured"

    diagnostic: str | None = None
    ir_resolved: tuple[Any, ...] = ()
    try:
        ir_documents = await service._ir_fallback.retrieve(
            url=ir_url,
            gaps=unresolved_before_ir,
            as_of=as_of,
        )
        for filing, document, clean_text in tuple(ir_documents or ()):
            context_text = targeted_gap_context(
                clean_text,
                unresolved_before_ir,
                gap_terms(unresolved_before_ir),
            )
            if not context_text:
                continue
            entry, cache_hit = await service._extractor.extract(
                filing,
                document,
                clean_text,
                valuation_date=as_of,
                source_text=clean_text,
                context_text=context_text,
                fiscal_years=extraction_years,
                source_provider="company_ir",
            )
            state.hits += int(cache_hit)
            state.misses += int(not cache_hit)
            merge_extraction_entry(entry, state=state)
        history = _assemble(service, state, company_id)
        ir_resolved = tuple(
            gap
            for gap in unresolved_before_ir
            if gap.key not in {item.key for item in history.audit.gaps_unresolved}
        )
    except Exception as exc:
        diagnostic = f"IR fallback failed: {exc}"
        state.warnings.append(diagnostic)
    return history, ir_resolved, diagnostic


__all__ = ["recover_from_ir"]
