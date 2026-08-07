from __future__ import annotations

from edgarito.cli.presentation._valuation_format import (
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
        additional_warnings: tuple[str, ...] = (),
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


__all__ = ["ValuationReportConsolePresenter"]
