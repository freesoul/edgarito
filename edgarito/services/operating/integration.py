"""Provider-neutral composition of operating forecasting and reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from edgarito.schemas.forward import ForwardRevenueEstimate
from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.services.forecasting.models import FcffForecastParameters
from edgarito.services.operating.forecast import OperatingForecastService
from edgarito.services.operating.reconciliation import (
    RevenueForecastReconciler,
    RevenueForecastReconciliation,
    materialize_revenue_anchors,
)


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


class OperatingForecastIntegrationService:
    """Compose deterministic operating evidence with revenue reconciliation.

    This seam deliberately accepts normalized contracts only.  It performs no
    discovery and has no provider dependencies.
    """

    def __init__(
        self,
        forecast_service: OperatingForecastService | None = None,
        reconciler: RevenueForecastReconciler | None = None,
    ) -> None:
        self.forecast_service = forecast_service or OperatingForecastService()
        self.reconciler = reconciler or RevenueForecastReconciler()

    def integrate(
        self,
        segments: Iterable[OperatingSegment],
        definitions: Iterable[OperatingDriverDefinition],
        observations: Iterable[OperatingDriverObservation] = (),
        management_constraints: Any = (),
        historical_revenue: Mapping[Any, Any] | None = None,
        consensus_estimates: Iterable[ForwardRevenueEstimate]
        | Mapping[Any, Any]
        | Any = (),
        explicit_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        management_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        fiscal_years: Iterable[int] = (),
        parameters: FcffForecastParameters | None = None,
        *,
        fcff_parameters: FcffForecastParameters | None = None,
        company_id: str = "company",
    ) -> OperatingForecastIntegrationResult:
        if parameters is not None and fcff_parameters is not None:
            raise ValueError("Pass either parameters or fcff_parameters, not both")
        parameters = fcff_parameters or parameters
        if parameters is None:
            raise TypeError("fcff_parameters is required")
        independent = self.forecast_service.forecast(
            segments,
            definitions,
            observations,
            management_constraints,
            historical_revenue,
            fiscal_years,
            company_id=company_id,
        )
        reconciliation = self.reconciler.reconcile_with_details(
            independent,
            consensus_estimates=consensus_estimates,
            historical_revenue=historical_revenue,
            explicit_anchors=explicit_anchors,
            management_anchors=management_anchors,
        )
        materialized = materialize_revenue_anchors(parameters, reconciliation)
        return OperatingForecastIntegrationResult(
            independent_forecast=independent,
            reconciled_forecast=reconciliation.forecast,
            reconciliation=reconciliation,
            parameters=materialized,
        )

    forecast = integrate
    compose = integrate
    run = integrate


OperatingForecastIntegration = OperatingForecastIntegrationService

__all__ = [
    "OperatingForecastIntegration",
    "OperatingForecastIntegrationResult",
    "OperatingForecastIntegrationService",
]
