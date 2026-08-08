from decimal import Decimal

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.schemas.red_flags import RedFlag, RedFlagEvidence, RedFlagsReport


class RedFlagsConsolePresenter:
    """Render the deterministic red-flags report for terminal users."""

    def render(self, report: RedFlagsReport, *, verbose: bool = False) -> str:
        identifier = report.ticker or f"CIK {report.company_id}"
        status = (
            "RED FLAGS FOUND"
            if report.flags
            else "INCOMPLETE"
            if report.warnings
            else "CLEAN"
        )
        evaluated = ", ".join(
            self._period_label(year, period)
            for year, period in report.evaluated_periods
        )
        lines = [
            f"{identifier} - {report.company_name}",
            f"Provider: {report.provider.upper()} | "
            f"Granularity: {report.granularity.value.upper()} | "
            f"Profile: {report.configuration_name}",
            f"Status: {status} | Flags: {len(report.flags)} | "
            f"Warnings: {len(report.warnings)}",
            f"Evaluated: {evaluated or '-'}",
            "",
            "RED FLAGS",
        ]

        if report.flags:
            for flag in report.flags:
                lines.extend(self._render_flag(flag, verbose=verbose))
        else:
            lines.append("None")

        lines.extend(["", "WARNINGS"])
        if report.warnings:
            for warning in report.warnings:
                category = (
                    f"[{self._label(warning.category.value)}] "
                    if warning.category is not None
                    else ""
                )
                lines.append(f"- {category}{warning.message}")
                if verbose and warning.required_concepts:
                    concepts = ", ".join(
                        concept.label for concept in warning.required_concepts
                    )
                    lines.append(f"  Required concepts: {concepts}")
        else:
            lines.append("None")
        return "\n".join(lines)

    def _render_flag(self, flag: RedFlag, *, verbose: bool) -> list[str]:
        evidence = flag.evidence[0] if flag.evidence else None
        period = (
            f" ({self._period_label(evidence.fiscal_year, evidence.fiscal_period)})"
            if evidence is not None
            else ""
        )
        lines = [
            f"- [{flag.severity.value.upper()}] {self._label(flag.code)}{period}: "
            f"{flag.message}"
        ]
        if verbose:
            for item in flag.evidence:
                lines.extend(self._render_evidence(item))
        return lines

    @classmethod
    def _render_evidence(cls, evidence: RedFlagEvidence) -> list[str]:
        threshold = (
            f"; threshold {cls._format_decimal(evidence.threshold)} "
            f"{evidence.threshold_unit or ''}".rstrip()
            if evidence.threshold is not None
            else ""
        )
        lines = [
            f"  Evidence: {evidence.metric} = "
            f"{cls._format_decimal(evidence.value)} {evidence.unit}{threshold}",
            f"  Formula: {evidence.formula} ({evidence.comparison})",
        ]
        if evidence.input_concepts:
            concepts = ", ".join(concept.label for concept in evidence.input_concepts)
            lines.append(f"  Inputs: {concepts}")
        if evidence.source_observations:
            lines.append("  Sources:")
            for observation in evidence.source_observations:
                period = cls._period_label(
                    observation.fiscal_year, observation.fiscal_period
                )
                lines.append(
                    f"    {observation.concept.label}: "
                    f"{cls._format_decimal(observation.value)} {observation.unit} "
                    f"({period}, {observation.source_concept})"
                )
        return lines

    @staticmethod
    def _period_label(year: int, period: FiscalPeriod) -> str:
        return f"FY{year}" if period == FiscalPeriod.FY else f"FY{year} {period.value}"

    @staticmethod
    def _label(value: str) -> str:
        return value.replace("_", " ").title()

    @staticmethod
    def _format_decimal(value: Decimal | None) -> str:
        if value is None:
            return "-"
        return f"{value:.2f}"


__all__ = ["RedFlagsConsolePresenter"]
