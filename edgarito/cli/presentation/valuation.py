from __future__ import annotations

from decimal import Decimal

from edgarito.cli.presentation._valuation_format import (
    format_currency,
    format_multiple,
    format_percent,
    format_ratio_percent,
    section,
    subsection,
)
from edgarito.services.valuation import (
    ComparableImpliedValuation,
    ComparableMultiplesReport,
    ModelRole,
    ModelSuitability,
    MultipleStatus,
    ValuationSelection,
)


class ValuationSelectionConsolePresenter:
    def render(self, selection: ValuationSelection) -> str:
        profile = selection.profile
        identifier = profile.ticker or f"CIK {profile.company_id}"
        lines = [
            f"{identifier} - {profile.company_name}",
            f"Economic profile: {self._label(profile.business_archetype.value)}",
            f"Sector: {profile.sector.value if profile.sector else '-'} | "
            f"Industry: {profile.industry or '-'}",
            f"Lifecycle: {self._label(profile.lifecycle.value)} | "
            f"Cyclicality: {self._label(profile.cyclicality.value)}",
        ]
        if profile.economic_traits:
            traits = ", ".join(
                self._label(trait.value)
                for trait in sorted(
                    profile.economic_traits, key=lambda item: item.value
                )
            )
            lines.append(f"Economic traits: {traits}")
        if profile.annual_fiscal_years:
            lines.append(
                f"Annual history: FY{profile.annual_fiscal_years[0]}–"
                f"FY{profile.annual_fiscal_years[-1]}"
            )

        for role in (
            ModelRole.PRIMARY,
            ModelRole.CONDITIONAL,
            ModelRole.CROSSCHECK,
            ModelRole.NOT_RECOMMENDED,
        ):
            models = [model for model in selection.models if model.role == role]
            if not models:
                continue
            lines.extend(["", self._label(role.value).upper()])
            for model in models:
                lines.extend(self._render_model(model))
        return "\n".join(lines)

    def _render_model(self, model: ModelSuitability) -> list[str]:
        lines = [
            f"{model.model.label} — suitability {model.suitability_score}/100; "
            f"data {self._label(model.data_readiness.value)}"
        ]
        if model.forecast_profile:
            lines.append(
                f"  Forecast profile: {self._label(model.forecast_profile.value)}"
            )
        if model.relative_bases:
            bases = ", ".join(
                self._label(basis.value) for basis in model.relative_bases
            )
            lines.append(f"  Suggested bases: {bases}")
        for reason in model.reasons:
            lines.append(f"  + {reason}")
        for rejection in model.hard_rejections:
            lines.append(f"  ! {rejection}")
        for limitation in model.limitations:
            lines.append(f"  ~ {limitation}")
        if model.missing_inputs:
            missing = ", ".join(
                self._label(item.value)
                for item in sorted(model.missing_inputs, key=lambda item: item.value)
            )
            lines.append(f"  Missing: {missing}")
        return lines

    @staticmethod
    def _label(value: str) -> str:
        relative_labels = {
            "price_to_earnings": "P/E (PER)",
            "ev_to_ebitda": "EV/EBITDA",
            "ev_to_fcf": "EV/FCF",
        }
        if value in relative_labels:
            return relative_labels[value]
        acronyms = {
            "affo": "AFFO",
            "dcf": "DCF",
            "ddm": "DDM",
            "ebit": "EBIT",
            "ebitda": "EBITDA",
            "ev": "EV",
            "fcf": "FCF",
            "fcfe": "FCFE",
            "fcff": "FCFF",
            "nav": "NAV",
            "roe": "ROE",
            "sotp": "SOTP",
            "wacc": "WACC",
        }
        return " ".join(acronyms.get(part, part.title()) for part in value.split("_"))


class ComparableMultiplesConsolePresenter:
    def render(
        self,
        report: ComparableMultiplesReport,
        *,
        verbose: bool = False,
        include_warnings: bool = True,
    ) -> str:
        target = report.target
        selected = set(report.universe.selected_tickers)
        lines = [
            *section("PEER ANALYSIS"),
            f"{target.ticker} - {target.company_name}",
            f"LTM period: {target.fundamentals.period_start.isoformat()} to "
            f"{target.fundamentals.period_end.isoformat()} | Current price: "
            f"{format_currency(target.price, target.currency)} "
            f"({target.price_date.isoformat()})",
            f"Selected peers: {len(selected)} | Discovery confidence: "
            f"{report.universe.discovery_confidence}",
            "",
            *subsection("PEER SELECTION"),
            f"{'Ticker':<10}{'Score':>8}   Evidence",
            "-" * 58,
        ]
        default_candidates = [
            candidate for candidate in report.universe.candidates if candidate.selected
        ]
        for candidate in default_candidates:
            evidence = ", ".join(self._evidence_tags(candidate.reasons)) or "selected"
            lines.append(f"{candidate.ticker:<10}{candidate.score:>7}/100   {evidence}")
        if not default_candidates:
            lines.append("No peers selected")
        if verbose:
            lines.extend(
                [
                    "",
                    f"Discovery source: {report.universe.discovery_source}",
                    f"Discovery method: {report.universe.discovery_methodology}",
                    "Full selection audit:",
                ]
            )
            for candidate in report.universe.candidates:
                decision = "selected" if candidate.selected else "excluded"
                detail = candidate.exclusions or candidate.reasons
                lines.append(
                    f"  {candidate.ticker} {candidate.score}/100 {decision}: "
                    f"{'; '.join(detail) or '-'}"
                )

        target_multiples = {item.basis: item for item in target.multiples}
        summaries = {item.basis: item for item in report.summaries}
        bases = list(dict.fromkeys([*target_multiples, *summaries]))
        lines.extend(
            [
                "",
                *subsection("PEER MULTIPLES"),
                f"{'Basis':<28} {'Target':>12} {'Peer median':>14} "
                f"{'Peer range':>21} {'N':>4}",
                "-" * 83,
            ]
        )
        for basis in bases:
            target_multiple = target_multiples.get(basis)
            summary = summaries.get(basis)
            target_value = (
                self._format_trading_multiple(
                    target_multiple.value, target_multiple.unit
                )
                if target_multiple
                and target_multiple.status == MultipleStatus.COMPUTED
                and target_multiple.value is not None
                else "-"
            )
            median_value = (
                self._format_trading_multiple(summary.median, target_multiple.unit)
                if summary and target_multiple
                else "-"
            )
            peer_range = (
                f"{self._format_trading_multiple(summary.minimum, target_multiple.unit)}–"
                f"{self._format_trading_multiple(summary.maximum, target_multiple.unit)}"
                if summary and target_multiple
                else "-"
            )
            lines.append(
                f"{ValuationSelectionConsolePresenter._label(basis.value):<28} "
                f"{target_value:>12} {median_value:>14} {peer_range:>21} "
                f"{summary.sample_size if summary else 0:>4}"
            )
        if include_warnings:
            warnings = self.warnings(report)
            if warnings:
                lines.extend(["", *subsection("WARNINGS")])
                lines.extend(f"- {warning}" for warning in warnings)
        return "\n".join(lines)

    @staticmethod
    def warnings(report: ComparableMultiplesReport) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *report.universe.warnings,
                    *report.target.warnings,
                    *(warning for peer in report.peers for warning in peer.warnings),
                    *report.warnings,
                ]
            )
        )

    @staticmethod
    def _evidence_tags(reasons: list[str]) -> tuple[str, ...]:
        joined = " ".join(reasons).casefold()
        candidates = (
            (
                ("same sector", "sector"),
                ("same business archetype", "archetype"),
                ("revenue scale is comparable", "scale"),
            )
            if "same sector" in joined
            else (
                ("same business archetype", "archetype"),
                ("same lifecycle", "lifecycle"),
                ("observable growth", "economics"),
                ("revenue scale is comparable", "scale"),
                ("same cyclicality", "cyclicality"),
            )
        )
        return tuple(label for marker, label in candidates if marker in joined)[:3]

    @staticmethod
    def _format_trading_multiple(value: Decimal, unit: str) -> str:
        return f"{value:,.2f}%" if unit == "percent" else f"{value:,.2f}x"


class ComparableImpliedValuationConsolePresenter:
    def render(
        self,
        result: ComparableImpliedValuation,
        *,
        verbose: bool = False,
        include_warnings: bool = True,
    ) -> str:
        multiple = result.resolved_multiple
        peer_label = {
            "forward": "Peer forward baseline",
            "current_ltm_fallback": "Peer baseline (current LTM)",
            "dcf_fallback": "Peer baseline (DCF fallback)",
        }.get(multiple.peer_anchor_source, "Peer forward baseline")
        target_label = (
            "Target forward comparative multiple:"
            if multiple.premium_evidence_source == "forward_synchronized"
            else "Current target comparative multiple:"
        )
        primary_premium = (
            multiple.forward_synchronized_premium
            if multiple.premium_evidence_source == "forward_synchronized"
            else (
                multiple.statistical_premium
                if multiple.statistical_premium is not None
                else multiple.observed_premium
            )
        )
        lines = [
            *section("RELATIVE VALUATION"),
            f"{result.ticker or result.company_id} - {result.company_name}",
            f"Valuation date: {result.valuation_date.isoformat()} | Target date: "
            f"{result.target_date.isoformat()} ({result.horizon_years:,.1f} years)",
            f"Basis: {ValuationSelectionConsolePresenter._label(result.basis.value)} | "
            f"Forecast metric: {result.forecast_metric_label}",
        ]
        if result.pure_peer_point_case is not None:
            lines.extend(
                [
                    "",
                    *subsection("INDEPENDENT PEER VALUATION"),
                    "Pure peer-implied target-date value (no DCF multiple, premium, "
                    "or WACC discounting): "
                    + format_currency(
                        result.pure_peer_point_case.target_date_value_per_share,
                        result.currency,
                    ),
                    "Pure peer horizon upside/(downside) vs current price: "
                    + format_percent(
                        self._horizon_upside(
                            result.pure_peer_point_case.target_date_value_per_share,
                            result.current_price,
                        ),
                        signed=True,
                    ),
                    "Pure peer multiple range: "
                    + f"{format_multiple(result.pure_peer_lower_case.multiple)}–"
                    + f"{format_multiple(result.pure_peer_upper_case.multiple)}",
                ]
            )
        if result.historical_point_case is not None:
            lines.extend(
                [
                    "",
                    *subsection("HISTORICAL MULTIPLE VALUATION"),
                    "Historical-multiple target-date value: "
                    + format_currency(
                        result.historical_point_case.target_date_value_per_share,
                        result.currency,
                    ),
                    "Historical-multiple horizon upside/(downside) vs current price: "
                    + format_percent(
                        self._horizon_upside(
                            result.historical_point_case.target_date_value_per_share,
                            result.current_price,
                        ),
                        signed=True,
                    ),
                    "Historical multiple range: "
                    + f"{format_multiple(result.historical_lower_case.multiple)}–"
                    + f"{format_multiple(result.historical_upper_case.multiple)}",
                    "Historical multiple valuation confidence: "
                    + self._historical_confidence(multiple),
                ]
            )
        lines.extend(
            [
                "",
                *subsection("COMPOSITE / DCF DIAGNOSTIC"),
                "DCF-blended relative value (target date): "
                + format_currency(
                    result.point_case.implied_value_per_share,
                    result.currency,
                ),
                "DCF-blended relative value (present-day DCF PV diagnostic): "
                + format_currency(
                    result.point_case.present_value_per_share,
                    result.currency,
                ),
                "",
                *subsection("MULTIPLE RESOLUTION"),
                f"{peer_label + ':':<40}{format_multiple(multiple.market_anchor)}",
                f"{'DCF-implied forward multiple (diagnostic):':<40}"
                f"{format_multiple(multiple.fundamental_anchor)}",
                f"{target_label:<40}{format_multiple(multiple.current_target_anchor)}",
                f"{'Premium evidence source:':<40}{multiple.premium_evidence_source}",
                f"{'Primary premium evidence:':<40}"
                f"{format_ratio_percent(primary_premium, signed=True)}",
                f"{'Historical premium:':<40}"
                f"{format_ratio_percent(multiple.historical_peer_premium, signed=True)}",
                f"{'Composite resolved premium:':<40}"
                f"{format_ratio_percent(multiple.resolved_premium, signed=True)}",
                f"{'Composite resolved multiple:':<40}"
                f"{format_multiple(multiple.point_estimate)}",
                f"{'Composite Evidence range:':<40}"
                f"{format_multiple(multiple.lower_bound)}–"
                f"{format_multiple(multiple.upper_bound)}",
                f"{'Confidence:':<40}{multiple.confidence.value}",
            ]
        )
        if multiple.premium_history_sample_size < 8:
            lines.append(
                "Historical persistence: disabled due to insufficient observations "
                f"({multiple.premium_history_sample_size} synchronized premium "
                "observations; minimum 8 required)"
            )
        if verbose:
            lines.extend(["", *self._multiple_audit(result)])
        lines.extend(
            [
                "",
                f"{'Composite case':<18}{'Multiple':>12}{'Target-date value':>24}"
                f"{'DCF PV diagnostic today':>34}",
                "-" * 88,
            ]
        )
        for case in (result.lower_case, result.point_case, result.upper_case):
            lines.append(
                f"{case.label:<18}{case.multiple:>11,.2f}x"
                f"{format_currency(case.implied_value_per_share, result.currency):>24}"
                f"{format_currency(case.present_value_per_share, result.currency):>34}"
            )
        if result.current_price is not None:
            lines.extend(
                [
                    "",
                    f"Current price: {format_currency(result.current_price, result.currency)}",
                    "Current-price implied forward multiple: "
                    + (
                        f"{result.current_price_implied_multiple:,.2f}x "
                        f"{ValuationSelectionConsolePresenter._label(result.basis.value)}"
                        if result.current_price_implied_multiple is not None
                        else "unavailable"
                    ),
                ]
            )
        if (
            result.analyst_target_price is not None
            and result.analyst_target_implied_multiple is not None
        ):
            lines.extend(
                [
                    f"Analyst target-date value: "
                    f"{format_currency(result.analyst_target_price, result.currency)}",
                    "Analyst-target implied multiple: "
                    f"{result.analyst_target_implied_multiple:,.2f}x "
                    f"{ValuationSelectionConsolePresenter._label(result.basis.value)}",
                ]
            )
        if include_warnings and result.warnings:
            lines.extend(["", *subsection("WARNINGS")])
            lines.extend(f"- {warning}" for warning in result.warnings)
        return "\n".join(lines)

    @staticmethod
    def _horizon_upside(value, current_price):
        if current_price is None or current_price <= 0:
            return None
        return (value / current_price - Decimal(1)) * Decimal(100)

    @staticmethod
    def _historical_confidence(multiple) -> str:
        observations = multiple.historical_sample_size
        if observations < 8:
            return (
                f"lower confidence — {observations} observations "
                "(fewer than 8 required for high confidence)"
            )
        return (
            f"{multiple.target_history_confidence.value} — {observations} observations"
        )

    @staticmethod
    def _multiple_audit(result: ComparableImpliedValuation) -> list[str]:
        multiple = result.resolved_multiple
        lines = [
            *subsection("MULTIPLE-RESOLUTION AUDIT"),
            f"DCF-implied premium vs peers (diagnostic): "
            f"{format_ratio_percent(multiple.fundamental_premium, signed=True)}",
            f"Target historical median: {format_multiple(multiple.historical_anchor)}",
            "Target historical IQR: "
            f"{format_multiple(multiple.historical_percentile_25)}–"
            f"{format_multiple(multiple.historical_percentile_75)}",
            f"Historical observations: {multiple.historical_sample_size}",
            f"Historical multiple volatility: "
            f"{format_ratio_percent(multiple.historical_volatility)}",
            f"Historical multiple trend: "
            f"{format_ratio_percent(multiple.historical_trend, signed=True)}",
            f"Current target premium vs baseline: "
            f"{format_ratio_percent(multiple.observed_premium, signed=True)}",
            f"Forward synchronized target/peer premium: "
            f"{format_ratio_percent(multiple.forward_synchronized_premium, signed=True)}",
            f"Forward peer adjustment weight: {multiple.forward_evidence_weight:,.1%}",
            f"Synchronized premium observations: "
            f"{multiple.premium_history_sample_size}",
            "Median premium observation interval: "
            + (
                f"{multiple.premium_observation_interval_years:,.2f} years"
                if multiple.premium_observation_interval_years is not None
                else "unavailable"
            ),
            "Raw AR(1) phi: "
            + (
                f"{multiple.premium_mean_reversion_beta:,.2f}"
                if multiple.premium_mean_reversion_beta is not None
                else "unavailable"
            ),
            f"Shrunk AR(1) phi: {multiple.shrunk_premium_persistence:,.2f}",
            f"Historical statistical premium at horizon: "
            f"{format_ratio_percent(multiple.statistical_premium, signed=True)}",
            f"Historical persistence evidence weight: "
            f"{multiple.premium_history_weight:,.1%}",
            f"Quality-support score: {multiple.fundamental_support:,.1%}",
            f"Historical persistence horizon retention: "
            f"{multiple.horizon_retention:,.1%}",
            f"Composite DCF/relative persistence factor: "
            f"{multiple.persistence_factor:,.1%}",
            "Confidence detail: "
            f"peer={multiple.peer_confidence.value}, "
            f"history={multiple.target_history_confidence.value}, "
            f"persistence={multiple.premium_persistence_confidence.value}",
            f"Peer sample: {multiple.sample_size}",
            f"Methodology: {multiple.methodology}",
        ]
        if result.intrinsic_value_per_share is not None:
            lines.extend(
                [
                    f"Intrinsic value: "
                    f"{format_currency(result.intrinsic_value_per_share, result.currency)}",
                    f"Composite relative value (present value): "
                    f"{format_currency(result.point_case.present_value_per_share, result.currency)}",
                ]
            )
        return lines
