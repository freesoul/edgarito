"""Process-state contracts for operating forecasting orchestration.

Validated domain payloads remain in :mod:`edgarito.schemas.operating`.  The
dataclasses here only compose those payloads with service state and audit
details.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from edgarito.schemas.forecasting import (
    AdaptiveMultistagePlan,
    FcffForecast,
    FcffForecastParameters,
    ForwardGrowthEvidence,
)
from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    ResolvedRevenueYear,
)
from edgarito.schemas.operating_history import OperatingHistoryAudit


@dataclass(frozen=True)
class RevenueForecastReconciliation:
    """Detailed result retained alongside the public company forecast."""

    forecast: CompanyOperatingForecast
    resolved_years: tuple[ResolvedRevenueYear, ...]

    @property
    def company_forecast(self) -> CompanyOperatingForecast:
        """Compatibility-friendly name for the selected company forecast."""

        return self.forecast

    @property
    def selected_years(self) -> tuple[ResolvedRevenueYear, ...]:
        """Alias emphasizing that each item is the selected value."""

        return self.resolved_years

    @property
    def selected_revenue_by_year(self) -> dict[int, Decimal]:
        return {item.fiscal_year: item.revenue for item in self.resolved_years}

    @property
    def revenue(self) -> tuple[Decimal, ...]:
        return self.forecast.consolidated_revenue

    @property
    def consolidated_revenue(self) -> tuple[Decimal, ...]:
        return self.forecast.consolidated_revenue

    @property
    def consolidated_growth(self) -> tuple[Decimal | None, ...]:
        return self.forecast.consolidated_growth

    @property
    def fiscal_years(self) -> tuple[int, ...]:
        return self.forecast.fiscal_years

    @property
    def source_by_year(self) -> dict[int, str]:
        return self.forecast.source_by_year

    @property
    def confidence_by_year(self) -> dict[int, str]:
        return self.forecast.confidence_by_year

    @property
    def explicit_years(self) -> tuple[int, ...]:
        return self.forecast.explicit_years

    @property
    def transition_start_year(self) -> int | None:
        return self.forecast.transition_start_year

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.forecast.warnings

    @property
    def operating_economics(self):
        return self.forecast.operating_economics

    @property
    def own_supported_years(self) -> tuple[int, ...]:
        """Years retained from the supported independent operating path."""

        return self.forecast.own_supported_years

    @property
    def consensus_years(self) -> tuple[int, ...]:
        """Years filled by analyst consensus after independent selection."""

        return self.forecast.consensus_years

    @property
    def divergence_by_year(self) -> dict[int, Decimal]:
        """Consensus-versus-independent revenue divergence by fiscal year."""

        return self.forecast.divergence_by_year

    @property
    def divergence(self) -> Decimal | None:
        """Mean absolute consensus-versus-independent divergence."""

        return self.forecast.divergence

    @property
    def selected_source_by_year(self) -> dict[int, str]:
        return self.forecast.selected_source_by_year

    @property
    def selected_confidence_by_year(self) -> dict[int, str]:
        return self.forecast.selected_confidence_by_year


@dataclass(frozen=True)
class OperatingForecastIntegrationResult:
    """Independent, selected, and FCFF-ready outputs from one composition."""

    independent_forecast: CompanyOperatingForecast
    reconciled_forecast: CompanyOperatingForecast
    reconciliation: RevenueForecastReconciliation
    parameters: FcffForecastParameters

    @property
    def company_forecast(self) -> CompanyOperatingForecast:
        return self.reconciled_forecast

    @property
    def reconciled(self) -> CompanyOperatingForecast:
        return self.reconciled_forecast

    @property
    def details(self) -> RevenueForecastReconciliation:
        return self.reconciliation

    @property
    def materialized_parameters(self) -> FcffForecastParameters:
        return self.parameters

    @property
    def fcff_parameters(self) -> FcffForecastParameters:
        return self.parameters

    @property
    def own_supported_years(self) -> tuple[int, ...]:
        return self.reconciled_forecast.own_supported_years

    @property
    def consensus_years(self) -> tuple[int, ...]:
        return self.reconciled_forecast.consensus_years

    @property
    def divergence_by_year(self) -> dict[int, Decimal]:
        return self.reconciled_forecast.divergence_by_year

    @property
    def divergence(self) -> Decimal | None:
        return self.reconciled_forecast.divergence

    @property
    def audit_diagnostics(self) -> dict[str, object]:
        return self.reconciled_forecast.audit_diagnostics

    @property
    def diagnostics(self) -> dict[str, object]:
        return self.audit_diagnostics

    @property
    def operating_economics(self):
        """Return optional gross economics without changing revenue reconciliation."""

        # Gross economics are calculated against the independent segment
        # revenue path.  Reconciliation may clear the reconciled copy when its
        # denominator changes, so the independent artifact is the truthful
        # convenience surface.
        return self.independent_forecast.operating_economics

    @property
    def independent_operating_economics(self):
        return self.independent_forecast.operating_economics

    def materialize_revenue_anchors(
        self, parameters: FcffForecastParameters
    ) -> FcffForecastParameters:
        return self.reconciled_forecast.materialize_revenue_anchors(parameters)


@dataclass(frozen=True)
class OperatingForecastPipelineResult:
    """Operating reconciliation composed into the ordinary FCFF path."""

    integration: OperatingForecastIntegrationResult
    seed_forecast: FcffForecast
    forecast: FcffForecast
    adaptive_plan: AdaptiveMultistagePlan | None = None
    forward_growth: ForwardGrowthEvidence | None = None
    quality: "OperatingForecastQualityResult" | None = None

    @property
    def independent_forecast(self) -> CompanyOperatingForecast:
        return self.integration.independent_forecast

    @property
    def reconciled_forecast(self) -> CompanyOperatingForecast:
        return self.integration.reconciled_forecast

    @property
    def reconciliation(self) -> RevenueForecastReconciliation:
        return self.integration.reconciliation

    @property
    def parameters(self) -> FcffForecastParameters:
        return self.forecast.parameters

    @property
    def fcff_forecast(self) -> FcffForecast:
        return self.forecast

    @property
    def fcff(self) -> FcffForecast:
        return self.forecast

    @property
    def adaptive_multistage_plan(self) -> AdaptiveMultistagePlan | None:
        return self.adaptive_plan

    @property
    def forecast_parameters(self) -> FcffForecastParameters:
        return self.parameters

    @property
    def plan(self) -> AdaptiveMultistagePlan | None:
        return self.adaptive_plan

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.integration.reconciled_forecast.warnings,
                    *self.forecast.warnings,
                )
            )
        )

    @property
    def audit_diagnostics(self) -> dict[str, object]:
        return self.integration.audit_diagnostics

    @property
    def diagnostics(self) -> dict[str, object]:
        return self.audit_diagnostics

    @property
    def operating_economics(self):
        return self.integration.operating_economics


@dataclass(frozen=True)
class OperatingForecastQualityResult:
    """Deterministic activation decision for structured operating evidence."""

    accepted: bool
    reason: str
    definitions_count: int = 0
    observations_count: int = 0
    driver_coverage: Decimal | None = None
    modeled_revenue_share: Decimal | None = None
    reconstruction_error: Decimal | None = None
    confidence: str | None = None
    own_supported_years: tuple[int, ...] = ()
    consensus_years: tuple[int, ...] = ()
    transition_start_year: int | None = None
    warnings: tuple[str, ...] = ()
    audit_records: tuple[Any, ...] = ()
    document_audits: tuple[Any, ...] = ()
    unusable_evidence: tuple[str, ...] = ()
    history_audit: OperatingHistoryAudit | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    filings_inspected: int = 0
    documents_inspected: int = 0
    raw_filings_received: int = 0
    raw_filings_in_range: int = 0
    candidate_filings: int = 0
    filing_inventory_cache_bypass: bool = False
    filing_inventory_fetched_live: bool = False
    filing_inventory_metadata: tuple[str, ...] = ()
    vocabulary_audit: Any | None = None
    vocabulary_terms: tuple[Any, ...] = ()
    exhibits_found: int = 0
    gaps_resolved_sec: tuple[Any, ...] = ()
    gaps_resolved_ir: tuple[Any, ...] = ()
    ir_diagnostic: str | None = None

    @property
    def status(self) -> str:
        return "active" if self.accepted else "rejected"


class OperatingForecastQualityError(ValueError):
    """Raised when structured operating evidence fails the activation gate."""

    def __init__(self, result: OperatingForecastQualityResult) -> None:
        self.result = result
        super().__init__(result.reason)


__all__ = [
    "OperatingForecastIntegrationResult",
    "OperatingForecastPipelineResult",
    "OperatingForecastQualityError",
    "OperatingForecastQualityResult",
    "RevenueForecastReconciliation",
]
