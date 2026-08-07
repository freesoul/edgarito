from __future__ import annotations

from decimal import Decimal

from edgarito.cli.presentation._valuation_format import (
    format_currency,
    section,
    short_warning,
    unique_warnings,
    warning_severity,
)
from edgarito.cli.presentation.dcf import FcffDcfConsolePresenter
from edgarito.cli.presentation.decision import DecisionValuationConsolePresenter
from edgarito.cli.presentation.valuation import (
    ComparableImpliedValuationConsolePresenter,
    ComparableMultiplesConsolePresenter,
)
from edgarito.schemas.guidance.management import (
    GuidanceOverlayResult,
    GuidanceValueKind,
    ManagementGuidance,
)
from edgarito.schemas.valuation.intrinsic import ValuationRunResult
from edgarito.schemas.valuation.relative import ProviderNeutralRelativeValuation
from edgarito.services.valuation import (
    ComparableImpliedValuation,
    ComparableMultiplesReport,
    DecisionValuationResult,
    FcffDcfResult,
)


class ValuationReportConsolePresenter:
    """Compose one detail-first valuation report with the decision summary last."""

    def render(
        self,
        *,
        intrinsic: FcffDcfResult | None,
        peer_report: ComparableMultiplesReport | None,
        relative: ComparableImpliedValuation | None,
        decision: DecisionValuationResult | None,
        profile_name: str | None,
        show_scenarios: bool,
        show_sensitivity: bool,
        show_reverse_dcf: bool,
        verbose: bool,
        provider_relative: ProviderNeutralRelativeValuation | None = None,
        additional_warnings: tuple[str, ...] = (),
        management_guidance: GuidanceOverlayResult | None = None,
    ) -> str:
        blocks: list[str] = []
        if intrinsic is not None:
            blocks.append(
                FcffDcfConsolePresenter().render(
                    intrinsic,
                    profile_name=profile_name,
                    verbose=verbose,
                    include_warnings=False,
                )
            )
        if management_guidance is not None and management_guidance.applications:
            blocks.append(
                self._render_management_guidance(
                    management_guidance, verbose=verbose
                )
            )
        if peer_report is not None:
            blocks.append(
                ComparableMultiplesConsolePresenter().render(
                    peer_report,
                    verbose=verbose,
                    include_warnings=False,
                )
            )
        if relative is not None:
            blocks.append(
                ComparableImpliedValuationConsolePresenter().render(
                    relative,
                    verbose=verbose,
                    include_warnings=False,
                )
            )
        if provider_relative is not None:
            blocks.append(
                ProviderNeutralRelativeValuationConsolePresenter().render(
                    provider_relative
                )
            )
        decision_presenter = DecisionValuationConsolePresenter()
        if decision is not None:
            details = decision_presenter.render_details(
                decision,
                show_scenarios=show_scenarios,
                show_sensitivity=show_sensitivity,
                show_reverse_dcf=show_reverse_dcf,
                verbose=verbose,
            )
            if details:
                blocks.append("\n".join(details))

        warnings = list(additional_warnings)
        if intrinsic is not None:
            warnings.extend(intrinsic.warnings)
        if peer_report is not None:
            warnings.extend(ComparableMultiplesConsolePresenter.warnings(peer_report))
        if relative is not None:
            warnings.extend(relative.warnings)
        if provider_relative is not None:
            warnings.extend(provider_relative.warnings)
        if decision is not None:
            warnings.extend(decision.warnings)
        rendered_warnings = self._render_warnings(warnings, verbose=verbose)
        if rendered_warnings:
            blocks.append(rendered_warnings)

        if decision is not None:
            blocks.append(
                "\n".join(decision_presenter.render_summary(decision, verbose=verbose))
            )
        return "\n\n".join(block for block in blocks if block)

    @classmethod
    def _render_management_guidance(
        cls, result: GuidanceOverlayResult, *, verbose: bool
    ) -> str:
        lines = [*section("MANAGEMENT GUIDANCE")]
        for application in result.applications:
            record = application.guidance
            lines.append(
                f"{record.period_label} {record.metric.value.replace('_', ' '):<20} "
                f"{cls._guidance_range(record):>18}   "
                f"base anchor {cls._guidance_value(application.value, record)}"
            )
        for record in result.evidence_only:
            lines.append(
                f"{record.period_label} {record.metric.value.replace('_', ' '):<20} "
                f"{cls._guidance_range(record):>18}   evidence only"
            )
        source_records = [
            *(item.guidance for item in result.applications),
            *result.evidence_only,
        ]
        first = source_records[0]
        source_labels = tuple(
            dict.fromkeys(
                f"SEC {record.filing_form} filed {record.filing_date.isoformat()}"
                for record in source_records
            )
        )
        lines.extend(
            (
                "",
                f"Source: {', '.join(source_labels)}",
                f"Extraction: OpenAI / {first.extraction_model}",
            )
        )
        if verbose:
            lines.append(
                f"Extraction cache: {result.cache_hits} hit(s), "
                f"{result.cache_misses} miss(es)"
            )
            for record in source_records:
                lines.extend(
                    (
                        f"  {record.accession_number} | {record.source_document} | "
                        f"{record.source_document_type}",
                        f"    Evidence: {record.supporting_text}",
                    )
                )
            lines.extend(f"  Rejected: {reason}" for reason in result.rejected_reasons)
        return "\n".join(lines)

    @classmethod
    def _guidance_range(cls, record: ManagementGuidance) -> str:
        if record.low is not None and record.high is not None:
            return (
                f"{cls._guidance_value(record.low, record)}–"
                f"{cls._guidance_value(record.high, record)}"
            )
        return cls._guidance_value(record.midpoint or Decimal(0), record)

    @staticmethod
    def _guidance_value(value: Decimal, record: ManagementGuidance) -> str:
        if record.value_kind == GuidanceValueKind.PERCENTAGE:
            return f"{value:,.1f}%"
        if record.value_kind == GuidanceValueKind.MONETARY:
            currency = record.currency or record.unit
            absolute = abs(value)
            if absolute >= Decimal(1_000_000_000):
                return f"{currency} {value / Decimal(1_000_000_000):,.1f}B"
            if absolute >= Decimal(1_000_000):
                return f"{currency} {value / Decimal(1_000_000):,.1f}M"
            return f"{currency} {value:,.0f}"
        return f"{value:,.2f}"

    @staticmethod
    def _render_warnings(messages: list[str], *, verbose: bool) -> str:
        warnings = unique_warnings(messages)
        if not warnings:
            return ""
        order = {"HIGH": 0, "MED": 1, "LOW": 2, "INFO": 3}
        warnings.sort(key=lambda item: order[warning_severity(item)])
        lines = [*section("CONSOLIDATED WARNINGS")]
        for warning in warnings:
            severity = warning_severity(warning)
            message = warning if verbose else short_warning(warning)
            lines.append(f"[{severity}] {message}")
        return "\n".join(lines)


class IndependentValuationModelsConsolePresenter:
    """Render independent intrinsic models and finish with a non-blended summary."""

    def render(self, run: ValuationRunResult, *, verbose: bool = False) -> str:
        profile = run.economic_profile
        blocks = [
            "\n".join(
                [
                    *section("VALUATION SETUP"),
                    f"Company             {profile.company_name}",
                    f"Ticker              {profile.ticker or 'unavailable'}",
                    f"Economic profile    {profile.business_archetype.value}",
                    "Policy              Independent model outputs; no blended value",
                ]
            )
        ]
        warnings: list[tuple[str, str]] = []
        for execution in run.executed_models:
            result = execution.result
            lines = [
                *section(result.model.label),
                f"Method               {result.adapter}",
                f"Role                 {execution.role.value}",
                f"Confidence           {result.confidence.value}",
                f"Equity value         {format_currency(result.equity_value, result.currency)}",
                f"Diluted shares       {result.diluted_shares:,.2f}",
                f"Intrinsic value      {format_currency(result.value_per_share, result.currency)}/share",
            ]
            if result.forecast_summary:
                lines.extend(("", "Forecast summary"))
                for point in result.forecast_summary:
                    present = (
                        f" | PV {format_currency(point.present_value, point.unit)}"
                        if point.present_value is not None
                        else ""
                    )
                    lines.append(
                        f"  {point.label:<16} {format_currency(point.amount, point.unit)}{present}"
                    )
            if verbose and result.assumptions:
                lines.extend(("", "Resolved assumptions"))
                for assumption in result.assumptions:
                    unit = f" {assumption.unit}" if assumption.unit else ""
                    lines.append(
                        f"  {assumption.name:<24} {assumption.value}{unit} [{assumption.source}]"
                    )
            blocks.append("\n".join(lines))
            warnings.extend(
                (warning.severity.value.upper(), warning.detail or warning.summary)
                for warning in result.warnings
            )

        if run.skipped_models:
            lines = [*section("MODEL READINESS / SKIPS")]
            for skipped in run.skipped_models:
                missing = (
                    f" | missing: {', '.join(sorted(skipped.missing_inputs))}"
                    if skipped.missing_inputs
                    else ""
                )
                reason = (
                    skipped.reasons[0] if skipped.reasons else "No executable inputs"
                )
                lines.append(
                    f"{skipped.model.label:<28} {skipped.readiness.value:<14}{missing}"
                )
                lines.append(f"  {reason}")
                if verbose:
                    lines.extend(f"  {item}" for item in skipped.reasons[1:])
            blocks.append("\n".join(lines))

        if warnings:
            lines = [*section("CONSOLIDATED WARNINGS")]
            seen: set[str] = set()
            for severity, message in warnings:
                key = message.casefold().rstrip(".")
                if key not in seen:
                    seen.add(key)
                    lines.append(f"[{severity}] {message}")
            blocks.append("\n".join(lines))

        summary = [
            *section("VALUATION MODEL SUMMARY"),
            "Independent model results (not averaged)",
            "",
        ]
        if run.executed_models:
            for execution in run.executed_models:
                result = execution.result
                summary.append(
                    f"{result.model.label:<28} {format_currency(result.value_per_share, result.currency)}/share  "
                    f"[{execution.role.value}, {result.confidence.value}]"
                )
        else:
            summary.append("No intrinsic model was data-ready.")
        blocks.append("\n".join(summary))
        return "\n\n".join(blocks)


class ProviderNeutralRelativeValuationConsolePresenter:
    def render(self, result: ProviderNeutralRelativeValuation) -> str:
        lines = [
            *section("RELATIVE VALUATION"),
            f"Basis: {result.metric.basis.value} | Metric: {result.metric.label}",
            f"Target date: {result.target_date.isoformat()} | Confidence: {result.confidence.value}",
            "",
            "Case            Multiple       Target-date value    Present-value equivalent today",
            "-" * 82,
        ]
        for case in (result.lower_case, result.point_case, result.upper_case):
            lines.append(
                f"{case.label:<14} {case.multiple:>9.2f}x"
                f" {format_currency(case.target_date_value_per_share, result.currency):>23}"
                f" {format_currency(case.present_value_per_share, result.currency):>33}"
            )
        if result.current_price is not None:
            lines.extend(
                (
                    "",
                    f"Current price: {format_currency(result.current_price, result.currency)}",
                    f"Current-price implied multiple: {result.current_price_implied_multiple:.2f}x",
                )
            )
        return "\n".join(lines)


__all__ = [
    "IndependentValuationModelsConsolePresenter",
    "ProviderNeutralRelativeValuationConsolePresenter",
    "ValuationReportConsolePresenter",
]
