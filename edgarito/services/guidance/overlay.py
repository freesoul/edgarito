from __future__ import annotations

from decimal import Decimal

from edgarito.schemas.guidance.management import (
    GuidanceApplication,
    GuidanceBasis,
    GuidanceMetric,
    GuidanceOverlayResult,
    GuidancePeriodType,
    GuidanceScope,
    GuidanceStatus,
    GuidanceValueKind,
    ManagementGuidance,
)
from edgarito.services.forecasting.models import (
    FcffForecast,
    FcffForecastDriver,
    FcffForecastParameters,
    ForecastAssumptionSource,
)


class GuidanceForecastOverlay:
    """Map verified numerical guidance into provider-neutral FCFF anchors."""

    def apply(
        self,
        records: tuple[ManagementGuidance, ...] | list[ManagementGuidance],
        *,
        baseline: FcffForecast,
        parameters: FcffForecastParameters,
    ) -> tuple[FcffForecastParameters, GuidanceOverlayResult]:
        eligible: list[ManagementGuidance] = []
        evidence_only: list[ManagementGuidance] = []
        rejected: list[str] = []
        years = {item.fiscal_year for item in baseline.observations}
        for record in records:
            reason = self._eligibility_reason(record, years)
            if reason:
                evidence_only.append(record)
                rejected.append(f"{record.period_label} {record.metric.value}: {reason}")
            else:
                eligible.append(record)

        by_metric_year = {
            (record.metric, record.fiscal_year): record for record in eligible
        }
        applications: list[GuidanceApplication] = []
        updates: dict = {}
        source_overrides = dict(parameters.assumption_source_overrides)
        revenue_anchors = dict(parameters.revenue_anchors)

        revenue_guidance = {
            year: record
            for (metric, year), record in by_metric_year.items()
            if metric == GuidanceMetric.REVENUE
            and record.value_kind == GuidanceValueKind.MONETARY
        }
        growth_guidance = {
            year: record
            for (metric, year), record in by_metric_year.items()
            if metric == GuidanceMetric.REVENUE_GROWTH
            and record.value_kind == GuidanceValueKind.PERCENTAGE
        }

        if parameters.revenue_growth is None:
            for year, record in sorted(revenue_guidance.items()):
                value = record.midpoint
                if value is None:
                    continue
                if record.currency != baseline.unit.upper():
                    evidence_only.append(record)
                    rejected.append(
                        f"{record.period_label} revenue: currency {record.currency} "
                        f"does not match forecast unit {baseline.unit}"
                    )
                    continue
                if year in revenue_anchors:
                    evidence_only.append(record)
                    rejected.append(
                        f"{record.period_label} revenue: explicit revenue anchor wins"
                    )
                    continue
                revenue_anchors[year] = value
                applications.append(
                    GuidanceApplication(
                        driver=FcffForecastDriver.REVENUE_GROWTH.value,
                        fiscal_year=year,
                        value=value,
                        guidance=record,
                        methodology="management guidance midpoint revenue anchor",
                    )
                )
            if revenue_anchors != parameters.revenue_anchors:
                updates["revenue_anchors"] = revenue_anchors
                source_overrides[FcffForecastDriver.REVENUE_GROWTH] = (
                    ForecastAssumptionSource.MANAGEMENT_GUIDANCE
                )

            growth_path = [item.revenue_growth for item in baseline.observations]
            growth_changed = False
            for index, observation in enumerate(baseline.observations):
                if observation.fiscal_year in revenue_guidance:
                    continue
                record = growth_guidance.get(observation.fiscal_year)
                if record is None or record.midpoint is None:
                    continue
                growth_path[index] = record.midpoint
                growth_changed = True
                applications.append(
                    GuidanceApplication(
                        driver=FcffForecastDriver.REVENUE_GROWTH.value,
                        fiscal_year=observation.fiscal_year,
                        value=record.midpoint,
                        guidance=record,
                        methodology="management guidance growth-range midpoint",
                    )
                )
            if growth_changed:
                updates["revenue_growth"] = tuple(growth_path)
                source_overrides[FcffForecastDriver.REVENUE_GROWTH] = (
                    ForecastAssumptionSource.MANAGEMENT_GUIDANCE
                )
        else:
            for record in [*revenue_guidance.values(), *growth_guidance.values()]:
                evidence_only.append(record)
                rejected.append(
                    f"{record.period_label} {record.metric.value}: explicit "
                    "CLI/profile revenue growth wins"
                )

        self._apply_percentage_driver(
            eligible,
            baseline,
            parameters,
            driver=FcffForecastDriver.OPERATING_MARGIN,
            metrics={GuidanceMetric.OPERATING_MARGIN, GuidanceMetric.EBIT_MARGIN},
            updates=updates,
            source_overrides=source_overrides,
            applications=applications,
            evidence_only=evidence_only,
            rejected=rejected,
        )
        self._apply_percentage_driver(
            eligible,
            baseline,
            parameters,
            driver=FcffForecastDriver.TAX_RATE,
            metrics={GuidanceMetric.TAX_RATE},
            updates=updates,
            source_overrides=source_overrides,
            applications=applications,
            evidence_only=evidence_only,
            rejected=rejected,
        )

        # Capex maps only when matching full-year absolute revenue guidance exists.
        if parameters.capex_to_revenue is None:
            capex_path = [item.capex_to_revenue for item in baseline.observations]
            changed = False
            for index, observation in enumerate(baseline.observations):
                capex = by_metric_year.get(
                    (GuidanceMetric.CAPEX, observation.fiscal_year)
                )
                revenue = revenue_guidance.get(observation.fiscal_year)
                if capex is None:
                    continue
                if (
                    capex.value_kind != GuidanceValueKind.MONETARY
                    or (
                        revenue is not None
                        and revenue.value_kind != GuidanceValueKind.MONETARY
                    )
                ):
                    evidence_only.append(capex)
                    rejected.append(
                        f"{capex.period_label} capex: monetary guidance is required"
                    )
                    continue
                if (
                    revenue is None
                    or capex.midpoint is None
                    or revenue.midpoint is None
                    or capex.currency != baseline.unit.upper()
                    or revenue.currency != baseline.unit.upper()
                ):
                    evidence_only.append(capex)
                    rejected.append(
                        f"{capex.period_label} capex: matching guided revenue in "
                        "forecast currency is required"
                    )
                    continue
                ratio = capex.midpoint / revenue.midpoint * Decimal(100)
                capex_path[index] = ratio
                changed = True
                applications.append(
                    GuidanceApplication(
                        driver=FcffForecastDriver.CAPEX_TO_REVENUE.value,
                        fiscal_year=observation.fiscal_year,
                        value=ratio,
                        guidance=capex,
                        methodology="guided capex / guided revenue midpoint",
                    )
                )
            if changed:
                updates["capex_to_revenue"] = tuple(capex_path)
                source_overrides[FcffForecastDriver.CAPEX_TO_REVENUE] = (
                    ForecastAssumptionSource.MANAGEMENT_GUIDANCE
                )

        applied_ids = {id(application.guidance) for application in applications}
        for record in eligible:
            if id(record) not in applied_ids and record not in evidence_only:
                evidence_only.append(record)
        if source_overrides != parameters.assumption_source_overrides:
            updates["assumption_source_overrides"] = source_overrides
        overlaid = parameters.model_copy(update=updates)
        return overlaid, GuidanceOverlayResult(
            applications=tuple(applications),
            evidence_only=tuple(dict.fromkeys(evidence_only)),
            rejected_reasons=tuple(dict.fromkeys(rejected)),
        )

    @staticmethod
    def _eligibility_reason(
        record: ManagementGuidance, forecast_years: set[int]
    ) -> str | None:
        if not record.evidence_verified:
            return "supporting evidence is unverified"
        if record.status == GuidanceStatus.WITHDRAWN:
            return "guidance was explicitly withdrawn"
        if record.basis in {
            GuidanceBasis.NON_GAAP,
            GuidanceBasis.CONSTANT_CURRENCY,
        }:
            return f"{record.basis.value} basis is not compatible with reported FCFF drivers"
        if record.scope != GuidanceScope.CONSOLIDATED:
            return "only consolidated guidance maps automatically"
        if record.period_type != GuidancePeriodType.FISCAL_YEAR:
            return "only exact fiscal-year guidance maps automatically"
        if record.fiscal_year not in forecast_years:
            return "period is outside the current FCFF forecast horizon"
        return None

    @staticmethod
    def _apply_percentage_driver(
        records,
        baseline,
        parameters,
        *,
        driver,
        metrics,
        updates,
        source_overrides,
        applications,
        evidence_only,
        rejected,
    ) -> None:
        selected = {
            record.fiscal_year: record
            for record in records
            if record.metric in metrics
            and record.value_kind == GuidanceValueKind.PERCENTAGE
        }
        if not selected:
            return
        if getattr(parameters, driver.value) is not None:
            for record in selected.values():
                evidence_only.append(record)
                rejected.append(
                    f"{record.period_label} {record.metric.value}: explicit "
                    f"CLI/profile {driver.value} wins"
                )
            return
        path = [getattr(item, driver.value) for item in baseline.observations]
        changed = False
        for index, observation in enumerate(baseline.observations):
            record = selected.get(observation.fiscal_year)
            if record is None or record.midpoint is None:
                continue
            path[index] = record.midpoint
            changed = True
            applications.append(
                GuidanceApplication(
                    driver=driver.value,
                    fiscal_year=observation.fiscal_year,
                    value=record.midpoint,
                    guidance=record,
                    methodology=f"management guidance midpoint {driver.value} anchor",
                )
            )
        if changed:
            updates[driver.value] = tuple(path)
            source_overrides[driver] = ForecastAssumptionSource.MANAGEMENT_GUIDANCE
