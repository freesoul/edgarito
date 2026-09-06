"""Contracts for valuation-root factor identities.

This is deliberately only an identity layer.  It does not calculate WACC,
terminal value, or any other valuation output.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from edgarito.services.valuation.factors.contracts import (
    FactorDomain,
    FactorKey,
    FactorPeriod,
    FactorPeriodType,
)


class ValuationRootMetric(str, Enum):
    RISK_FREE_RATE = "risk_free_rate"
    EQUITY_RISK_PREMIUM = "equity_risk_premium"
    ERP = "equity_risk_premium"
    BETA = "beta"
    DEBT_SPREAD = "debt_spread"
    TARGET_CAPITAL_STRUCTURE = "target_capital_structure"
    TERMINAL_GROWTH = "terminal_growth"
    TERMINAL_ROIC = "terminal_roic"


VALUATION_ROOT_METRICS = (
    ValuationRootMetric.RISK_FREE_RATE,
    ValuationRootMetric.EQUITY_RISK_PREMIUM,
    ValuationRootMetric.BETA,
    ValuationRootMetric.DEBT_SPREAD,
    ValuationRootMetric.TARGET_CAPITAL_STRUCTURE,
    ValuationRootMetric.TERMINAL_GROWTH,
    ValuationRootMetric.TERMINAL_ROIC,
)

_GLOBAL_ROOTS = {
    ValuationRootMetric.RISK_FREE_RATE,
    ValuationRootMetric.EQUITY_RISK_PREMIUM,
}


def _metric(value: ValuationRootMetric | str) -> ValuationRootMetric:
    if isinstance(value, ValuationRootMetric):
        return value
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"erp", "equity_premium", "equity_risk_premium"}:
        return ValuationRootMetric.EQUITY_RISK_PREMIUM
    return ValuationRootMetric(normalized)


def _default_period() -> FactorPeriod:
    return FactorPeriod(
        period_type=FactorPeriodType.CURRENT_SPOT,
        period_key="current_spot",
    )


def valuation_root_key(
    metric: ValuationRootMetric | str,
    *,
    period: FactorPeriod | None = None,
    company_id: str | None = None,
    subject_id: str | None = None,
    unit: str = "percent",
    currency: str | None = None,
    domain: FactorDomain | str | None = None,
) -> FactorKey:
    """Build a semantic key for one root without invoking valuation logic."""

    root = _metric(metric)
    if subject_id is None:
        subject_id = (
            company_id
            if root not in _GLOBAL_ROOTS and company_id is not None
            else "global"
        )
    if domain is None:
        domain = (
            FactorDomain.MACRO
            if root in {ValuationRootMetric.RISK_FREE_RATE, ValuationRootMetric.EQUITY_RISK_PREMIUM}
            else FactorDomain.FINANCING
            if root in {
                ValuationRootMetric.DEBT_SPREAD,
                ValuationRootMetric.TARGET_CAPITAL_STRUCTURE,
            }
            else FactorDomain.COMPANY
        )
    else:
        domain = FactorDomain(domain)
    subject_type = "macro" if root in {
        ValuationRootMetric.RISK_FREE_RATE,
        ValuationRootMetric.EQUITY_RISK_PREMIUM,
    } else "financing" if domain == FactorDomain.FINANCING else "company"
    return FactorKey(
        domain=domain,
        subject_type=subject_type,
        subject_id=subject_id,
        metric=root.value,
        period=period or _default_period(),
        unit=unit,
        currency=currency,
    )


class ValuationRootFactor(BaseModel):
    """Small auditable wrapper around a root FactorKey."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: FactorKey

    @classmethod
    def from_metric(cls, metric: ValuationRootMetric | str, **kwargs) -> "ValuationRootFactor":
        return cls(key=valuation_root_key(metric, **kwargs))

    @property
    def metric(self) -> str:
        return self.key.metric


def valuation_root_keys(
    *,
    period: FactorPeriod | None = None,
    company_id: str | None = None,
    currency: str | None = None,
) -> tuple[FactorKey, ...]:
    """Return the seven supported root identities in stable metric order."""

    return tuple(
        valuation_root_key(
            metric,
            period=period,
            company_id=company_id,
            currency=currency,
        )
        for metric in VALUATION_ROOT_METRICS
    )


# A descriptive alias for code that treats roots as a contract factory.
build_valuation_root_key = valuation_root_key
ValuationRoot = ValuationRootFactor


__all__ = [
    "VALUATION_ROOT_METRICS",
    "ValuationRootFactor",
    "ValuationRootMetric",
    "ValuationRoot",
    "build_valuation_root_key",
    "valuation_root_key",
    "valuation_root_keys",
]
