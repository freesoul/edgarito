"""Point-in-time provenance rules shared by graph discovery and evaluation."""

from __future__ import annotations

import datetime as _datetime

_EVIDENCE_ORIGINS = frozenset(
    {
        "evidence",
        "extracted_evidence",
        "first_party",
        "first_party_disclosure",
        "first_party_filing",
        "first_party_observation",
        "historical_reported",
        "reported",
    }
)
_DETERMINISTIC_TEMPLATE_ORIGIN = "deterministic_template"


def _origin(value: object) -> str:
    if isinstance(value, str):
        return value.strip().casefold().replace("-", "_").replace(" ", "_")
    return str(getattr(value, "origin", "") or "").strip().casefold().replace(
        "-", "_"
    ).replace(" ", "_")


def is_evidence_provenance(value: object) -> bool:
    """Return whether a provenance value claims a dated evidence source."""

    if value is None:
        return False
    origin = _origin(value)
    if origin == _DETERMINISTIC_TEMPLATE_ORIGIN:
        return False
    if origin in _EVIDENCE_ORIGINS:
        return True
    if not isinstance(value, str):
        source = str(getattr(value, "source", "") or "").strip().casefold()
        if source in _EVIDENCE_ORIGINS:
            return True
        return bool(getattr(value, "evidence_ids", ()))
    return False


def provenance_as_of_issue(
    value: object,
    as_of: _datetime.date | None,
) -> tuple[str, str] | None:
    """Return a diagnostic code/message when evidence provenance is unusable."""

    if as_of is None or not is_evidence_provenance(value):
        return None
    available_on = getattr(value, "available_on", None)
    label = _origin(value) or "evidence"
    if available_on is None:
        return (
            "missing_available_on",
            f"{label} provenance has no available_on date for as_of={as_of.isoformat()}",
        )
    if available_on > as_of:
        return (
            "future_evidence",
            f"{label} provenance was not available as of {as_of.isoformat()}",
        )
    return None


def observation_as_of_issue(
    observation: object,
    as_of: _datetime.date | None,
    reported_origins: frozenset[str],
) -> tuple[str, str] | None:
    """Return the point-in-time issue for a reported observation."""

    if as_of is None:
        return None
    available_on = getattr(observation, "available_on", None)
    origin = str(getattr(observation, "origin", "") or "").casefold()
    if origin not in reported_origins:
        if available_on is not None and available_on > as_of:
            return (
                "future_evidence",
                f"Observation was not available as of {as_of.isoformat()}",
            )
        return None
    if available_on is None:
        return (
            "missing_available_on",
            f"{origin} observation has no available_on date for as_of={as_of.isoformat()}",
        )
    if available_on > as_of:
        return (
            "future_evidence",
            f"{origin} observation was not available as of {as_of.isoformat()}",
        )
    return None


__all__ = [
    "is_evidence_provenance",
    "observation_as_of_issue",
    "provenance_as_of_issue",
]
