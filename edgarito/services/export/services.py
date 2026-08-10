"""Adapters from analysis-domain models to canonical export snapshots."""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from edgarito.enums.provider import ProviderName
from edgarito.schemas.normalization.financials import (
    NormalizedCompanyFinancials,
)
from edgarito.schemas.red_flags import RedFlagsReport
from edgarito.schemas.valuation.intrinsic import ValuationRunResult
from edgarito.services.forecasting.models import (
    AdaptiveMultistagePlan,
    FcffForecast,
    SimplifiedFcfForecast,
)
from edgarito.services.metrics.models import CompanyMetrics
from edgarito.services.valuation.models import ValuationProfile

from ._utils import max_period_end, snapshot
from .models import (
    AdaptiveMultistagePlanExport,
    CanonicalFinancialObservation,
    CanonicalMetricObservation,
    CanonicalSecurityIdentifiers,
    CompanyAnalysisReport,
    ExecutedValuationExport,
    FcffForecastExport,
    FcffForecastObservationExport,
    FcffForecastParametersExport,
    FcffForecastYtdAnchorExport,
    FinancialDataExport,
    ForecastExport,
    ForecastKind,
    ForecastSummaryPointExport,
    MetricsExport,
    ModelSuitabilityExport,
    RedFlagEvidenceExport,
    RedFlagExport,
    RedFlagsExport,
    RedFlagSourceObservationExport,
    RedFlagWarningExport,
    SimplifiedFcfForecastExport,
    SimplifiedFcfForecastObservationExport,
    SimplifiedFcfForecastParametersExport,
    SkippedValuationExport,
    ValuationAssumptionExport,
    ValuationDispersionExport,
    ValuationExport,
    ValuationInputProvenanceExport,
    ValuationModelResultExport,
    ValuationProfileExport,
    ValuationSelectionExport,
    ValuationWarningExport,
)


def _copy_identifiers(value):
    if value is None:
        return None
    return CanonicalSecurityIdentifiers(
        ticker=value.ticker,
        isin=value.isin,
        cik=value.cik,
        exchange=value.exchange,
        exchange_symbols=value.exchange_symbols,
        provider_symbols=(
            tuple(
                (ProviderName(provider), symbol)
                for provider, symbol in value.provider_symbols.items()
            )
            if hasattr(value.provider_symbols, "items")
            else value.provider_symbols
        ),
    )


def _copy_datetime(value: datetime.datetime | None) -> datetime.datetime | None:
    return value


def _observation_dates(observations: Iterable[Any]) -> list[datetime.date | None]:
    return [getattr(item, "period_end", None) for item in observations]


def _normalized_identity_values(section) -> set[tuple[str, str | int]]:
    values: set[tuple[str, str | int]] = set()
    ticker = getattr(section, "ticker", None)
    if ticker is not None:
        values.add(("ticker", ticker.strip().upper()))
    identifiers = getattr(section, "identifiers", None)
    if identifiers is None:
        return values
    if identifiers.ticker is not None:
        values.add(("ticker", identifiers.ticker.strip().upper()))
    if identifiers.isin is not None:
        values.add(("isin", identifiers.isin.strip().upper()))
    if identifiers.cik is not None:
        values.add(("cik", identifiers.cik))
    return values


def _validate_composition_identity(sections) -> None:
    names = {
        section.company_name.strip().casefold()
        for section in sections
        if section.company_name is not None
    }
    if len(names) > 1:
        raise ValueError("Analysis sections contain conflicting company_name values")

    tickers = {
        ticker.strip().upper()
        for section in sections
        for ticker in (
            getattr(section, "ticker", None),
            getattr(getattr(section, "identifiers", None), "ticker", None),
        )
        if ticker is not None
    }
    if len(tickers) > 1:
        raise ValueError("Analysis sections contain conflicting ticker values")

    providers = {section.provider for section in sections}
    if len(providers) <= 1:
        return
    identity_sets = [_normalized_identity_values(section) for section in sections]
    if not set.intersection(*identity_sets):
        raise ValueError(
            "Analysis sections from different providers require a shared reliable "
            "identity (ticker, ISIN, or CIK)"
        )


class FinancialDataExportService:
    """Create a detached canonical snapshot of normalized financials."""

    def __init__(self, *, generated_at: datetime.datetime | None = None):
        self.generated_at = generated_at

    def export(self, financials: NormalizedCompanyFinancials) -> FinancialDataExport:
        observations = tuple(
            CanonicalFinancialObservation.model_validate(
                deepcopy(observation.model_dump(mode="python"))
            )
            for observation in financials.observations
        )
        generated_at = (
            self.generated_at
            if self.generated_at is not None
            else financials.retrieved_at
        )
        return FinancialDataExport(
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker,
            identifiers=_copy_identifiers(financials.identifiers),
            provider=financials.provider,
            generated_at=generated_at,
            as_of=max_period_end(_observation_dates(observations)),
            retrieved_at=_copy_datetime(financials.retrieved_at),
            observations=observations,
        )

    def convert(self, financials: NormalizedCompanyFinancials) -> FinancialDataExport:
        return self.export(financials)

    def __call__(self, financials: NormalizedCompanyFinancials) -> FinancialDataExport:
        return self.export(financials)


class MetricsExportService:
    """Create a canonical snapshot without recalculating any metric."""

    def __init__(self, *, generated_at: datetime.datetime | None = None):
        self.generated_at = generated_at

    def export(self, metrics: CompanyMetrics) -> MetricsExport:
        observations = tuple(
            CanonicalMetricObservation.model_validate(
                deepcopy(observation.model_dump(mode="python"))
            )
            for observation in metrics.observations
        )
        return MetricsExport(
            company_id=metrics.company_id,
            company_name=metrics.company_name,
            ticker=metrics.ticker,
            provider=metrics.provider,
            generated_at=self.generated_at,
            as_of=max_period_end(_observation_dates(observations)),
            observations=observations,
        )

    def convert(self, metrics: CompanyMetrics) -> MetricsExport:
        return self.export(metrics)

    def __call__(self, metrics: CompanyMetrics) -> MetricsExport:
        return self.export(metrics)


def _simplified_parameters(source):
    return SimplifiedFcfForecastParametersExport(
        forecast_years=source.forecast_years,
        revenue_growth=source.revenue_growth,
        free_cash_flow_margin=source.free_cash_flow_margin,
        historical_window=source.historical_window,
    )


def _simplified_forecast(source: SimplifiedFcfForecast) -> SimplifiedFcfForecastExport:
    return SimplifiedFcfForecastExport(
        provider=source.provider,
        company_id=source.company_id,
        company_name=source.company_name,
        ticker=source.ticker,
        identifiers=_copy_identifiers(source.identifiers),
        method=source.method,
        base_fiscal_year=source.base_fiscal_year,
        base_period_end=source.base_period_end,
        base_revenue=source.base_revenue,
        base_free_cash_flow=source.base_free_cash_flow,
        unit=source.unit,
        parameters=_simplified_parameters(source.parameters),
        historical_fiscal_years=tuple(source.historical_fiscal_years),
        revenue_growth_source=source.revenue_growth_source,
        free_cash_flow_margin_source=source.free_cash_flow_margin_source,
        observations=tuple(
            SimplifiedFcfForecastObservationExport.model_validate(
                deepcopy(observation.model_dump(mode="python"))
            )
            for observation in source.observations
        ),
    )


def _fcff_parameters(source):
    return FcffForecastParametersExport(
        forecast_years=source.forecast_years,
        revenue_growth=source.revenue_growth,
        operating_margin=source.operating_margin,
        tax_rate=source.tax_rate,
        depreciation_to_revenue=source.depreciation_to_revenue,
        capex_to_revenue=source.capex_to_revenue,
        capex_constraints=tuple(sorted(source.capex_constraints.items())),
        operating_working_capital_to_revenue=source.operating_working_capital_to_revenue,
        revenue_anchors=tuple(sorted(source.revenue_anchors.items())),
        revenue_anchor_sources=tuple(sorted(source.revenue_anchor_sources.items())),
        assumption_source_overrides=tuple(
            sorted(
                source.assumption_source_overrides.items(),
                key=lambda item: item[0].value,
            )
        ),
        historical_window=source.historical_window,
    )


def _fcff_forecast(source: FcffForecast) -> FcffForecastExport:
    return FcffForecastExport(
        provider=source.provider,
        company_id=source.company_id,
        company_name=source.company_name,
        ticker=source.ticker,
        identifiers=_copy_identifiers(source.identifiers),
        method=source.method,
        seed_type=source.seed_type,
        seed_methodology=source.seed_methodology,
        seed_period_end=source.seed_period_end,
        current_fiscal_year=source.current_fiscal_year,
        actual_quarters=source.actual_quarters,
        financial_snapshot_retrieved_at=source.financial_snapshot_retrieved_at,
        availability_mode=source.availability_mode,
        base_fiscal_year=source.base_fiscal_year,
        base_period_end=source.base_period_end,
        base_revenue=source.base_revenue,
        base_operating_income=source.base_operating_income,
        base_tax_rate=source.base_tax_rate,
        base_nopat=source.base_nopat,
        base_depreciation_and_amortization=source.base_depreciation_and_amortization,
        base_capital_expenditures=source.base_capital_expenditures,
        base_operating_working_capital=source.base_operating_working_capital,
        base_fcff=source.base_fcff,
        unit=source.unit,
        parameters=_fcff_parameters(source.parameters),
        historical_fiscal_years=tuple(source.historical_fiscal_years),
        assumption_sources=tuple(
            sorted(source.assumption_sources.items(), key=lambda item: item[0].value)
        ),
        observations=tuple(
            FcffForecastObservationExport.model_validate(
                deepcopy(observation.model_dump(mode="python"))
            )
            for observation in source.observations
        ),
        warnings=tuple(source.warnings),
        capex_constraints_applied=tuple(source.capex_constraints_applied),
        ytd_anchor=(
            FcffForecastYtdAnchorExport.model_validate(
                deepcopy(source.ytd_anchor.model_dump(mode="python"))
            )
            if source.ytd_anchor is not None
            else None
        ),
    )


def _adaptive_plan(source: AdaptiveMultistagePlan | None):
    if source is None:
        return None
    return AdaptiveMultistagePlanExport.model_validate(
        deepcopy(source.model_dump(mode="python"))
    )


class ForecastExportService:
    """Snapshot either supported forecast model and an optional adaptive plan."""

    def __init__(self, *, generated_at: datetime.datetime | None = None):
        self.generated_at = generated_at

    def export(
        self,
        forecast: SimplifiedFcfForecast | FcffForecast,
        *,
        adaptive_plan: AdaptiveMultistagePlan | None = None,
    ) -> ForecastExport:
        if isinstance(forecast, SimplifiedFcfForecast):
            canonical = _simplified_forecast(forecast)
            kind = ForecastKind.SIMPLIFIED_FCF
            generated_at = self.generated_at
            as_of = canonical.base_period_end
        elif isinstance(forecast, FcffForecast):
            canonical = _fcff_forecast(forecast)
            kind = ForecastKind.FCFF
            generated_at = self.generated_at or forecast.financial_snapshot_retrieved_at
            as_of = forecast.seed_period_end or forecast.base_period_end
        else:
            raise TypeError("forecast must be SimplifiedFcfForecast or FcffForecast")
        return ForecastExport(
            company_id=forecast.company_id,
            company_name=forecast.company_name,
            ticker=forecast.ticker,
            identifiers=_copy_identifiers(forecast.identifiers),
            provider=forecast.provider,
            generated_at=generated_at,
            as_of=as_of,
            forecast_type=kind,
            forecast=canonical,
            adaptive_plan=_adaptive_plan(adaptive_plan),
        )

    def convert(
        self,
        forecast: SimplifiedFcfForecast | FcffForecast,
        *,
        adaptive_plan: AdaptiveMultistagePlan | None = None,
    ) -> ForecastExport:
        return self.export(forecast, adaptive_plan=adaptive_plan)

    def __call__(
        self,
        forecast: SimplifiedFcfForecast | FcffForecast,
        *,
        adaptive_plan: AdaptiveMultistagePlan | None = None,
    ) -> ForecastExport:
        return self.export(forecast, adaptive_plan=adaptive_plan)


def _valuation_profile(source: ValuationProfile) -> ValuationProfileExport:
    return ValuationProfileExport(
        provider=source.provider,
        company_id=source.company_id,
        company_name=source.company_name,
        ticker=source.ticker,
        identifiers=_copy_identifiers(source.identifiers),
        sector=source.sector.value if source.sector is not None else None,
        industry=source.industry,
        country=source.country,
        exchange=source.exchange,
        reporting_currency=source.reporting_currency,
        latest_revenue=source.latest_revenue,
        business_archetype=source.business_archetype,
        financial_institution_kind=source.financial_institution_kind,
        actuarial_detail_supplied=source.actuarial_detail_supplied,
        regulatory_capital_constraints_supplied=source.regulatory_capital_constraints_supplied,
        lifecycle=source.lifecycle,
        cyclicality=source.cyclicality,
        economic_traits=tuple(
            sorted(source.economic_traits, key=lambda item: item.value)
        ),
        annual_fiscal_years=tuple(source.annual_fiscal_years),
        revenue_growth_rates=tuple(source.revenue_growth_rates),
        positive_fcf_periods=source.positive_fcf_periods,
        positive_earnings_periods=source.positive_earnings_periods,
        latest_book_equity=source.latest_book_equity,
        available_inputs=tuple(
            sorted(source.available_inputs, key=lambda item: item.value)
        ),
        peer_count=source.peer_count,
        inference_notes=tuple(source.inference_notes),
    )


def _suitability(source) -> ModelSuitabilityExport:
    return ModelSuitabilityExport(
        model=source.model,
        role=source.role,
        suitability_score=source.suitability_score,
        data_readiness=source.data_readiness,
        forecast_profile=source.forecast_profile,
        reasons=tuple(source.reasons),
        limitations=tuple(source.limitations),
        hard_rejections=tuple(source.hard_rejections),
        missing_inputs=tuple(
            sorted(source.missing_inputs, key=lambda item: item.value)
        ),
        relative_bases=tuple(source.relative_bases),
    )


def _valuation_result(source) -> ValuationModelResultExport:
    return ValuationModelResultExport(
        model=source.model,
        adapter=source.adapter,
        company_id=source.company_id,
        company_name=source.company_name,
        ticker=source.ticker,
        valuation_date=source.valuation_date,
        currency=source.currency,
        equity_value=source.equity_value,
        diluted_shares=source.diluted_shares,
        value_per_share=source.value_per_share,
        assumptions=tuple(
            ValuationAssumptionExport.model_validate(
                deepcopy(item.model_dump(mode="python"))
            )
            for item in source.assumptions
        ),
        forecast_summary=tuple(
            ForecastSummaryPointExport.model_validate(
                deepcopy(item.model_dump(mode="python"))
            )
            for item in source.forecast_summary
        ),
        confidence=source.confidence,
        warnings=tuple(
            ValuationWarningExport.model_validate(
                deepcopy(item.model_dump(mode="python"))
            )
            for item in source.warnings
        ),
        provenance=tuple(
            ValuationInputProvenanceExport.model_validate(
                deepcopy(item.model_dump(mode="python"))
            )
            for item in source.provenance
        ),
        details=snapshot(source.details),
    )


class ValuationExportService:
    """Wrap an existing run while retaining every executed and skipped model."""

    def __init__(self, *, generated_at: datetime.datetime | None = None):
        self.generated_at = generated_at

    def export(self, run: ValuationRunResult) -> ValuationExport:
        profile = _valuation_profile(run.economic_profile)
        selection = ValuationSelectionExport(
            profile=profile,
            models=tuple(_suitability(item) for item in run.selection.models),
        )
        executed = tuple(
            ExecutedValuationExport(
                role=item.role,
                suitability=_suitability(item.suitability),
                result=_valuation_result(item.result),
            )
            for item in run.executed_models
        )
        skipped = tuple(
            SkippedValuationExport(
                model=item.model,
                role=item.role,
                readiness=item.readiness,
                missing_inputs=tuple(sorted(item.missing_inputs)),
                reasons=tuple(item.reasons),
            )
            for item in run.skipped_models
        )
        valuation_dates = [item.result.valuation_date for item in run.executed_models]
        return ValuationExport(
            company_id=run.economic_profile.company_id,
            company_name=run.economic_profile.company_name,
            ticker=run.economic_profile.ticker,
            identifiers=_copy_identifiers(run.economic_profile.identifiers),
            provider=run.economic_profile.provider,
            generated_at=self.generated_at,
            as_of=max_period_end(valuation_dates),
            economic_profile=profile,
            selection=selection,
            executed_models=executed,
            skipped_models=skipped,
            relative_cross_checks=tuple(
                snapshot(item) for item in run.relative_cross_checks
            ),
            dispersion=(
                ValuationDispersionExport.model_validate(
                    deepcopy(run.dispersion.model_dump(mode="python"))
                )
                if run.dispersion is not None
                else None
            ),
        )

    def convert(self, run: ValuationRunResult) -> ValuationExport:
        return self.export(run)

    def __call__(self, run: ValuationRunResult) -> ValuationExport:
        return self.export(run)


def _red_flag_source(source) -> RedFlagSourceObservationExport:
    return RedFlagSourceObservationExport.model_validate(
        deepcopy(source.model_dump(mode="python"))
    )


def _red_flag_evidence(source) -> RedFlagEvidenceExport:
    return RedFlagEvidenceExport(
        metric=source.metric,
        value=source.value,
        unit=source.unit,
        threshold=source.threshold,
        threshold_unit=source.threshold_unit,
        comparison=source.comparison,
        formula=source.formula,
        fiscal_year=source.fiscal_year,
        fiscal_period=source.fiscal_period,
        period_end=source.period_end,
        granularity=source.granularity,
        input_concepts=tuple(source.input_concepts),
        source_observations=tuple(
            _red_flag_source(item) for item in source.source_observations
        ),
    )


class RedFlagsExportService:
    """Copy deterministic flags, warnings, and all auditable evidence."""

    def __init__(self, *, generated_at: datetime.datetime | None = None):
        self.generated_at = generated_at

    def export(self, report: RedFlagsReport) -> RedFlagsExport:
        flags = tuple(
            RedFlagExport(
                code=flag.code,
                category=flag.category,
                severity=flag.severity,
                message=flag.message,
                evidence=tuple(_red_flag_evidence(item) for item in flag.evidence),
            )
            for flag in report.flags
        )
        warnings = tuple(
            RedFlagWarningExport(
                code=warning.code,
                category=warning.category,
                message=warning.message,
                period=warning.period,
                required_concepts=tuple(warning.required_concepts),
            )
            for warning in report.warnings
        )
        dates = [evidence.period_end for flag in flags for evidence in flag.evidence]
        return RedFlagsExport(
            schema_version=1,
            source_schema_version=report.schema_version,
            company_id=report.company_id,
            company_name=report.company_name,
            ticker=report.ticker,
            provider=report.provider,
            generated_at=self.generated_at,
            as_of=max_period_end(dates),
            granularity=report.granularity,
            configuration_name=report.configuration_name,
            evaluated_periods=tuple(report.evaluated_periods),
            flags=flags,
            warnings=warnings,
        )

    def convert(self, report: RedFlagsReport) -> RedFlagsExport:
        return self.export(report)

    def __call__(self, report: RedFlagsReport) -> RedFlagsExport:
        return self.export(report)


def _is_canonical(value: Any, expected_type: type) -> bool:
    return isinstance(value, expected_type)


def _detach(value, expected_type):
    return expected_type.model_validate(deepcopy(value.model_dump(mode="python")))


class CompanyAnalysisReportService:
    """Compose independent section snapshots without changing CLI orchestration."""

    def __init__(self, *, generated_at: datetime.datetime | None = None):
        self.generated_at = generated_at

    def compose(
        self,
        *,
        financials: NormalizedCompanyFinancials | FinancialDataExport | None = None,
        metrics: CompanyMetrics | MetricsExport | None = None,
        forecast: (SimplifiedFcfForecast | FcffForecast | ForecastExport | None) = None,
        adaptive_plan: AdaptiveMultistagePlan | None = None,
        valuation: ValuationRunResult | ValuationExport | None = None,
        red_flags: RedFlagsReport | RedFlagsExport | None = None,
    ) -> CompanyAnalysisReport:
        financial_export = (
            _detach(financials, FinancialDataExport)
            if _is_canonical(financials, FinancialDataExport)
            else FinancialDataExportService(generated_at=self.generated_at).export(
                financials
            )
            if financials is not None
            else None
        )
        metrics_export = (
            _detach(metrics, MetricsExport)
            if _is_canonical(metrics, MetricsExport)
            else MetricsExportService(generated_at=self.generated_at).export(metrics)
            if metrics is not None
            else None
        )
        forecast_export = (
            _detach(forecast, ForecastExport)
            if _is_canonical(forecast, ForecastExport)
            else ForecastExportService(generated_at=self.generated_at).export(
                forecast, adaptive_plan=adaptive_plan
            )
            if forecast is not None
            else None
        )
        valuation_export = (
            _detach(valuation, ValuationExport)
            if _is_canonical(valuation, ValuationExport)
            else ValuationExportService(generated_at=self.generated_at).export(
                valuation
            )
            if valuation is not None
            else None
        )
        red_flags_export = (
            _detach(red_flags, RedFlagsExport)
            if _is_canonical(red_flags, RedFlagsExport)
            else RedFlagsExportService(generated_at=self.generated_at).export(red_flags)
            if red_flags is not None
            else None
        )
        sections = tuple(
            section
            for section in (
                financial_export,
                metrics_export,
                forecast_export,
                valuation_export,
                red_flags_export,
            )
            if section is not None
        )
        if not sections:
            raise ValueError("At least one analysis section is required")
        company_ids = {section.company_id for section in sections}
        if len(company_ids) > 1:
            raise ValueError(
                "All analysis sections must share company_id; "
                f"found {sorted(company_ids)!r}"
            )
        _validate_composition_identity(sections)
        first = sections[0]
        generated = self.generated_at or max(
            (
                section.generated_at
                for section in sections
                if section.generated_at is not None
            ),
            default=None,
        )
        as_of = max_period_end([section.as_of for section in sections])
        providers = {section.provider for section in sections}
        return CompanyAnalysisReport(
            company_id=first.company_id,
            company_name=first.company_name,
            ticker=next(
                (section.ticker for section in sections if section.ticker is not None),
                None,
            ),
            identifiers=_copy_identifiers(
                next(
                    (
                        section.identifiers
                        for section in sections
                        if section.identifiers is not None
                    ),
                    None,
                )
            ),
            provider=next(iter(providers)) if len(providers) == 1 else "mixed",
            generated_at=generated,
            as_of=as_of,
            financial_data=financial_export,
            metrics=metrics_export,
            forecast=forecast_export,
            valuation=valuation_export,
            red_flags=red_flags_export,
        )

    def export(self, **sections) -> CompanyAnalysisReport:
        return self.compose(**sections)

    def __call__(self, **sections) -> CompanyAnalysisReport:
        return self.compose(**sections)


# Compatibility-oriented names for callers that use section-specific terminology.
NormalizedFinancialDataExportService = FinancialDataExportService
FinancialsExportService = FinancialDataExportService
ForecastsExportService = ForecastExportService
ValuationReportExportService = ValuationExportService
RedFlagsReportExportService = RedFlagsExportService
