"""Centralized eligibility policy for semantic factor estimates."""

from __future__ import annotations

import datetime as dt
from datetime import timedelta
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from edgarito.services.valuation.factors.contracts import (
    FactorDomain,
    FactorEstimate,
    FactorFreshnessMode,
    FactorRequest,
    FreshnessReason,
)
from edgarito.services.valuation.factors.identity import canonicalize_token

_DEFAULT_DOMAIN_TTLS = {
    FactorDomain.COMPANY: timedelta(days=120),
    FactorDomain.BUSINESS: timedelta(days=120),
    FactorDomain.OPERATING: timedelta(days=180),
    FactorDomain.MARKET: timedelta(days=30),
    FactorDomain.COMPETITOR: timedelta(days=90),
    FactorDomain.COMMODITY: timedelta(days=7),
    FactorDomain.MACRO: timedelta(days=90),
    FactorDomain.REGULATORY: timedelta(days=30),
    FactorDomain.GEOPOLITICAL: timedelta(days=14),
    FactorDomain.FINANCING: timedelta(days=30),
}


def _point(value: dt.date | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return value
    return dt.datetime.combine(value, dt.time.min)


class FactorFreshnessRule(BaseModel):
    """One domain/source policy rule; the first matching specificity wins."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: Optional[FactorDomain] = None
    source: Optional[str] = None
    mode: FactorFreshnessMode = FactorFreshnessMode.AUTO
    ttl: Optional[timedelta] = None
    immutable_reusable: bool = True

    @field_validator("domain", mode="before")
    @classmethod
    def normalize_domain(cls, value):
        return FactorDomain(value) if value is not None else None

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: Optional[str]) -> Optional[str]:
        return canonicalize_token(value, field="source") if value is not None else None

    @field_validator("ttl")
    @classmethod
    def valid_ttl(cls, value: Optional[timedelta]) -> Optional[timedelta]:
        if value is not None and value.total_seconds() < 0:
            raise ValueError("freshness TTL cannot be negative")
        return value


class FreshnessDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    eligible: bool
    reason: FreshnessReason
    detail: Optional[str] = None


class FactorFreshnessPolicy(BaseModel):
    """All cache freshness decisions go through this policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: tuple[FactorFreshnessRule, ...] = ()
    default_mode: FactorFreshnessMode = FactorFreshnessMode.AUTO
    default_ttl: Optional[timedelta] = None
    immutable_reusable: bool = True

    @field_validator("default_mode", mode="before")
    @classmethod
    def normalize_default_mode(cls, value):
        return FactorFreshnessMode(value)

    @field_validator("default_ttl")
    @classmethod
    def valid_default_ttl(cls, value: Optional[timedelta]) -> Optional[timedelta]:
        if value is not None and value.total_seconds() < 0:
            raise ValueError("default freshness TTL cannot be negative")
        return value

    def rule_for(self, estimate: FactorEstimate) -> FactorFreshnessRule:
        candidates = []
        source = estimate.source or canonicalize_token(
            estimate.resolver, field="resolver"
        )
        for index, rule in enumerate(self.rules):
            if rule.domain is not None and rule.domain != estimate.key.domain:
                continue
            if rule.source is not None and rule.source != source:
                continue
            specificity = (rule.source is not None) + (rule.domain is not None)
            candidates.append((specificity, -index, rule))
        if candidates:
            selected = max(candidates, key=lambda item: (item[0], item[1]))[2]
            if selected.mode == FactorFreshnessMode.AUTO and selected.ttl is None:
                return selected.model_copy(
                    update={
                        "ttl": _DEFAULT_DOMAIN_TTLS.get(
                            estimate.key.domain, timedelta(days=90)
                        )
                    }
                )
            return selected
        return FactorFreshnessRule(
            mode=self.default_mode,
            ttl=(
                self.default_ttl
                if self.default_ttl is not None
                else _DEFAULT_DOMAIN_TTLS.get(
                    estimate.key.domain, timedelta(days=90)
                )
            ),
            immutable_reusable=self.immutable_reusable,
        )

    def check(
        self,
        estimate: FactorEstimate,
        request: FactorRequest,
        evaluated_at: dt.date | dt.datetime,
        *,
        current_dependency_fingerprints: Optional[dict[str, str]] = None,
        invalidated: bool = False,
    ) -> FreshnessDecision:
        """Check one candidate at an explicit evaluation time."""

        if estimate.key != request.key:
            return FreshnessDecision(
                eligible=False, reason=FreshnessReason.KEY_MISMATCH
            )
        if estimate.target_period != request.key.period:
            return FreshnessDecision(
                eligible=False, reason=FreshnessReason.TARGET_PERIOD_MISMATCH
            )
        if invalidated:
            return FreshnessDecision(eligible=False, reason=FreshnessReason.INVALIDATED)
        if estimate.superseded:
            return FreshnessDecision(eligible=False, reason=FreshnessReason.SUPERSEDED)
        if _point(estimate.info_as_of) > _point(request.information_as_of):
            return FreshnessDecision(
                eligible=False, reason=FreshnessReason.FUTURE_INFORMATION
            )
        if any(
            _point(available_on) > _point(request.information_as_of)
            for available_on in estimate.all_availability_dates
        ):
            return FreshnessDecision(
                eligible=False, reason=FreshnessReason.AVAILABILITY_AFTER_AS_OF
            )
        if estimate.confidence.rank < request.min_confidence.rank:
            return FreshnessDecision(
                eligible=False, reason=FreshnessReason.CONFIDENCE_BELOW_MINIMUM
            )

        expected = estimate.dependency_fingerprint_map
        if expected:
            current = current_dependency_fingerprints or {}
            normalized_current = {
                str(key): str(value).strip().lower() for key, value in current.items()
            }
            for dependency in estimate.dependencies:
                expected_fingerprint = expected.get(dependency.digest)
                if expected_fingerprint is None:
                    return FreshnessDecision(
                        eligible=False,
                        reason=FreshnessReason.DEPENDENCY_FINGERPRINT_MISMATCH,
                    )
                actual = normalized_current.get(dependency.digest)
                if actual is None:
                    actual = normalized_current.get(dependency.semantic_id)
                if actual != expected_fingerprint:
                    return FreshnessDecision(
                        eligible=False,
                        reason=FreshnessReason.DEPENDENCY_FINGERPRINT_MISMATCH,
                    )

        rule = self.rule_for(estimate)
        evaluated = _point(evaluated_at)
        if (
            estimate.expires_at is not None
            and not estimate.immutable
            and evaluated >= _point(estimate.expires_at)
        ):
            return FreshnessDecision(eligible=False, reason=FreshnessReason.EXPIRED)
        if (
            estimate.immutable
            and rule.immutable_reusable
            and (
                rule.mode in {FactorFreshnessMode.AUTO, FactorFreshnessMode.IMMUTABLE}
                or self.immutable_reusable
            )
        ):
            return FreshnessDecision(
                eligible=True, reason=FreshnessReason.ELIGIBLE, detail="immutable"
            )
        if rule.mode == FactorFreshnessMode.IMMUTABLE and not estimate.immutable:
            return FreshnessDecision(
                eligible=False, reason=FreshnessReason.IMMUTABLE_REQUIRED
            )
        if rule.mode == FactorFreshnessMode.EXPIRES and estimate.expires_at is None:
            return FreshnessDecision(eligible=False, reason=FreshnessReason.EXPIRED)
        if rule.mode == FactorFreshnessMode.TTL or (
            rule.mode == FactorFreshnessMode.AUTO and rule.ttl is not None
        ):
            if rule.ttl is None:
                return FreshnessDecision(
                    eligible=False, reason=FreshnessReason.TTL_EXPIRED
                )
            if evaluated >= _point(estimate.created_at) + rule.ttl:
                return FreshnessDecision(
                    eligible=False, reason=FreshnessReason.TTL_EXPIRED
                )
        return FreshnessDecision(eligible=True, reason=FreshnessReason.ELIGIBLE)

    def evaluate(self, *args, **kwargs) -> FreshnessDecision:
        return self.check(*args, **kwargs)

    def is_eligible(self, *args, **kwargs) -> bool:
        return self.check(*args, **kwargs).eligible


__all__ = [
    "FactorFreshnessPolicy",
    "FactorFreshnessRule",
    "FreshnessDecision",
]
