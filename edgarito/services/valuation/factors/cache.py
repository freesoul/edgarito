"""Append-only semantic factor cache with optional deterministic persistence."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from edgarito.services.valuation.factors.contracts import (
    CacheState,
    FactorEstimate,
    FactorRequest,
    FactorResolutionStatus,
    FreshnessReason,
)
from edgarito.services.valuation.factors.freshness import (
    FactorFreshnessPolicy,
    FreshnessDecision,
)


class CacheLookup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: CacheState
    estimate: Optional[FactorEstimate] = None
    reason: FreshnessReason
    detail: Optional[str] = None

    @property
    def status(self) -> FactorResolutionStatus:
        return (
            FactorResolutionStatus.CACHE_HIT
            if self.state == CacheState.HIT
            else FactorResolutionStatus.UNRESOLVED
        )

    @property
    def hit(self) -> bool:
        return self.state == CacheState.HIT and self.estimate is not None


class SemanticFactorCache:
    """Cache keyed by ``FactorKey.digest``; values are never overwritten."""

    _INVALIDATIONS_FILE = ".invalidations.json"

    def __init__(self, directory: str | os.PathLike[str] | None = None):
        self.directory = Path(directory) if directory is not None else None
        self._history: dict[str, list[FactorEstimate]] = {}
        self._invalidated: dict[str, set[str]] = {}
        if self.directory is not None:
            self._load()

    def put(self, estimate: FactorEstimate) -> bool:
        """Append an estimate unless this exact computation already exists."""

        digest = estimate.key.digest
        history = self._history.setdefault(digest, [])
        if any(item.fingerprint == estimate.fingerprint for item in history):
            return False
        history.append(estimate)
        history.sort(key=self._sort_key)
        self._persist_key(digest)
        return True

    def history(self, key) -> tuple[FactorEstimate, ...]:
        digest = key.digest
        return tuple(self._history.get(digest, ()))

    def invalidate(self, key, *, fingerprint: Optional[str] = None) -> int:
        """Mark entries stale while retaining every historical version on disk."""

        digest = key.digest
        known = {item.fingerprint for item in self._history.get(digest, ())}
        if fingerprint is None:
            self._invalidated.setdefault(digest, set()).update(known)
            count = len(known)
        else:
            self._invalidated.setdefault(digest, set()).add(fingerprint)
            count = int(fingerprint in known)
        self._persist_invalidations()
        return count

    def lookup(
        self,
        request: FactorRequest,
        policy: Optional[FactorFreshnessPolicy] = None,
        evaluated_at: dt.date | dt.datetime | None = None,
        current_dependency_fingerprints: Optional[Mapping[object, str]] = None,
    ) -> CacheLookup:
        if evaluated_at is None:
            raise ValueError("evaluated_at must be supplied explicitly")
        policy = policy or FactorFreshnessPolicy()
        candidates = self._history.get(request.key.digest, ())
        if not candidates:
            return CacheLookup(state=CacheState.MISS, reason=FreshnessReason.NO_ENTRY)

        current = self._normalize_dependency_fingerprints(
            current_dependency_fingerprints
        )
        decisions: list[tuple[FactorEstimate, FreshnessDecision]] = []
        for estimate in candidates:
            decision = policy.check(
                estimate,
                request,
                evaluated_at,
                current_dependency_fingerprints=current,
                invalidated=estimate.fingerprint
                in self._invalidated.get(request.key.digest, set()),
            )
            decisions.append((estimate, decision))
        eligible = [item for item in decisions if item[1].eligible]
        if eligible:
            estimate, _ = max(eligible, key=lambda item: self._sort_key(item[0]))
            return CacheLookup(
                state=CacheState.HIT,
                estimate=estimate,
                reason=FreshnessReason.ELIGIBLE,
            )

        # Report the most recent candidate's failure, but don't remove future
        # versions: a later request may make one of them point-in-time eligible.
        estimate, decision = max(decisions, key=lambda item: self._sort_key(item[0]))
        return CacheLookup(
            state=CacheState.STALE,
            reason=decision.reason,
            detail=decision.detail,
        )

    def get(self, *args, **kwargs) -> CacheLookup:
        return self.lookup(*args, **kwargs)

    @staticmethod
    def _sort_key(estimate: FactorEstimate):
        def point(value):
            if isinstance(value, dt.datetime):
                if value.tzinfo is not None and value.utcoffset() is not None:
                    return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
                return value
            return dt.datetime.combine(value, dt.time.min)

        return (
            point(estimate.info_as_of),
            point(estimate.created_at),
            estimate.version,
            estimate.fingerprint,
        )

    @staticmethod
    def _normalize_dependency_fingerprints(
        values: Optional[Mapping[object, str]],
    ) -> dict[str, str]:
        if not values:
            return {}
        normalized = {}
        for key, value in values.items():
            normalized_key = key.digest if hasattr(key, "digest") else str(key)
            fingerprint = value.fingerprint if hasattr(value, "fingerprint") else value
            normalized[normalized_key] = str(fingerprint).strip().lower()
        return normalized

    def _load(self) -> None:
        assert self.directory is not None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        for path in self.directory.glob("*.json"):
            if path.name == self._INVALIDATIONS_FILE:
                continue
            if len(path.stem) != 64 or any(
                char not in "0123456789abcdef" for char in path.stem
            ):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                raw_estimates = (
                    payload["estimates"] if isinstance(payload, dict) else payload
                )
                estimates = [
                    FactorEstimate.model_validate(item) for item in raw_estimates
                ]
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                # A corrupt cache must behave like a miss, not prevent startup.
                continue
            if all(estimate.key.digest == path.stem for estimate in estimates):
                self._history[path.stem] = sorted(estimates, key=self._sort_key)
        try:
            payload = json.loads(
                (self.directory / self._INVALIDATIONS_FILE).read_text(encoding="utf-8")
            )
            self._invalidated = {
                str(key): set(str(item) for item in values)
                for key, values in payload.items()
                if isinstance(values, list)
            }
        except (AttributeError, OSError, ValueError, TypeError, json.JSONDecodeError):
            self._invalidated = {}

    def _persist_key(self, digest: str) -> None:
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "estimates": [
                    estimate.model_dump(mode="json", by_alias=False)
                    for estimate in self._history[digest]
                ]
            }
            self._atomic_write(self.directory / f"{digest}.json", payload)
        except OSError:
            # Persistence is optional; the in-memory cache remains usable.
            return

    def _persist_invalidations(self) -> None:
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._atomic_write(
                self.directory / self._INVALIDATIONS_FILE,
                {
                    key: sorted(values)
                    for key, values in sorted(self._invalidated.items())
                },
            )
        except OSError:
            return

    @staticmethod
    def _atomic_write(path: Path, payload: object) -> None:
        content = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)


__all__ = ["CacheLookup", "SemanticFactorCache"]
