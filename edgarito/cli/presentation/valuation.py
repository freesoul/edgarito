from decimal import Decimal

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
    def render(self, report: ComparableMultiplesReport) -> str:
        target = report.target
        selected = set(report.universe.selected_tickers)
        lines = [
            f"{target.ticker} - {target.company_name}",
            f"LTM period: {target.fundamentals.period_start.isoformat()} to "
            f"{target.fundamentals.period_end.isoformat()} | Price: "
            f"{target.price:,.2f} {target.currency} on {target.price_date.isoformat()}",
            f"Selected peers ({len(selected)}): "
            f"{', '.join(report.universe.selected_tickers) or '-'}",
            f"Candidate source: {report.universe.discovery_source} | "
            f"confidence {report.universe.discovery_confidence}",
            f"Discovery method: {report.universe.discovery_methodology}",
            "",
            "PEER SELECTION",
            f"{'Ticker':<12} {'Score':>7}  Decision / evidence",
            "-" * 78,
        ]
        for candidate in report.universe.candidates:
            decision = "selected" if candidate.selected else "excluded"
            detail = candidate.exclusions or candidate.reasons
            lines.append(
                f"{candidate.ticker:<12} {candidate.score:>6}/100  "
                f"{decision}: {'; '.join(detail) or '-'}"
            )

        target_multiples = {item.basis: item for item in target.multiples}
        summaries = {item.basis: item for item in report.summaries}
        bases = list(dict.fromkeys([*target_multiples, *summaries]))
        lines.extend(
            [
                "",
                "LTM MULTIPLES",
                f"{'Basis':<28} {'Target':>12} {'Peer median':>14} "
                f"{'Peer range':>21} {'N':>4}",
                "-" * 83,
            ]
        )
        for basis in bases:
            target_multiple = target_multiples.get(basis)
            summary = summaries.get(basis)
            target_value = (
                self._format_multiple(target_multiple.value, target_multiple.unit)
                if target_multiple
                and target_multiple.status == MultipleStatus.COMPUTED
                and target_multiple.value is not None
                else "-"
            )
            median_value = (
                self._format_multiple(summary.median, target_multiple.unit)
                if summary and target_multiple
                else "-"
            )
            peer_range = (
                f"{self._format_multiple(summary.minimum, target_multiple.unit)}–"
                f"{self._format_multiple(summary.maximum, target_multiple.unit)}"
                if summary and target_multiple
                else "-"
            )
            lines.append(
                f"{ValuationSelectionConsolePresenter._label(basis.value):<28} "
                f"{target_value:>12} {median_value:>14} {peer_range:>21} "
                f"{summary.sample_size if summary else 0:>4}"
            )

        warnings = [
            *target.warnings,
            *(warning for peer in report.peers for warning in peer.warnings),
            *report.warnings,
        ]
        if warnings:
            lines.extend(["", "WARNINGS"])
            lines.extend(f"- {warning}" for warning in dict.fromkeys(warnings))
        return "\n".join(lines)

    @staticmethod
    def _format_multiple(value: Decimal, unit: str) -> str:
        return f"{value:,.2f}%" if unit == "percent" else f"{value:,.2f}x"


class ComparableImpliedValuationConsolePresenter:
    def render(self, result: ComparableImpliedValuation) -> str:
        multiple = result.resolved_multiple

        def anchor(value):
            return f"{value:,.2f}x" if value is not None else "unavailable"

        peer_label = {
            "forward": "Peer forward baseline",
            "current_ltm_fallback": "Peer baseline (current LTM)",
            "dcf_fallback": "Base multiple (DCF fallback)",
        }.get(multiple.peer_anchor_source, "Peer/base multiple")

        lines = [
            "MARKET-RELATIVE IMPLIED VALUATION",
            f"{result.ticker or result.company_id} - {result.company_name}",
            f"Valuation date: {result.valuation_date.isoformat()} | Target date: "
            f"{result.target_date.isoformat()} ({result.horizon_years:,.2f} years)",
            f"Basis: {ValuationSelectionConsolePresenter._label(result.basis.value)} | "
            f"Metric: {result.forecast_metric_label}",
            "",
            "MULTIPLE RESOLUTION",
            f"{peer_label + ':':<36}{anchor(multiple.market_anchor)}",
            f"DCF-implied forward multiple:   {anchor(multiple.fundamental_anchor)}",
            "DCF-implied premium vs peers:   "
            + (
                f"{multiple.fundamental_premium:+,.1%}"
                if multiple.fundamental_premium is not None
                else "unavailable"
            ),
            f"Target historical median:       {anchor(multiple.historical_anchor)}",
            "Target historical IQR:          "
            + (
                f"{multiple.historical_percentile_25:,.2f}x-"
                f"{multiple.historical_percentile_75:,.2f}x"
                if multiple.historical_percentile_25 is not None
                and multiple.historical_percentile_75 is not None
                else "unavailable"
            ),
            f"Historical observations:         {multiple.historical_sample_size}",
            "Historical multiple volatility: "
            + (
                f"{multiple.historical_volatility:,.1%}"
                if multiple.historical_volatility is not None
                else "unavailable"
            ),
            "Historical multiple trend:      "
            + (
                f"{multiple.historical_trend:+,.1%}"
                if multiple.historical_trend is not None
                else "unavailable"
            ),
            f"Current target comparative multiple: "
            f"{anchor(multiple.current_target_anchor)}",
            "Current target premium vs base: "
            + (
                f"{multiple.observed_premium:+,.1%}"
                if multiple.observed_premium is not None
                else "unavailable"
            ),
            "Historical long-run premium:    "
            + (
                f"{multiple.historical_peer_premium:+,.1%}"
                if multiple.historical_peer_premium is not None
                else " unavailable"
            ),
            f"Synchronized premium observations: "
            f"{multiple.premium_history_sample_size}",
            "Median premium observation interval: "
            + (
                f"{multiple.premium_observation_interval_years:,.2f} years"
                if multiple.premium_observation_interval_years is not None
                else "unavailable"
            ),
            "Raw AR(1) phi (deviation persistence): "
            + (
                f"{multiple.premium_mean_reversion_beta:,.2f}"
                if multiple.premium_mean_reversion_beta is not None
                else "unavailable"
            ),
            f"Shrunk AR(1) phi:               "
            f"{multiple.shrunk_premium_persistence:,.2f}",
            "Statistical premium at horizon: "
            + (
                f"{multiple.statistical_premium:+,.1%}"
                if multiple.statistical_premium is not None
                else "unavailable"
            ),
            f"Premium-history weight:         {multiple.premium_history_weight:,.1%}",
            f"Fundamental quality support:    {multiple.fundamental_support:,.1%}",
            f"Horizon evidence retention:     {multiple.horizon_retention:,.1%}",
            f"Statistical-anchor evidence weight: {multiple.persistence_factor:,.1%}",
            "Resolved target premium:        "
            + (
                f"{multiple.resolved_premium:+,.1%}"
                if multiple.resolved_premium is not None
                else "unavailable"
            ),
            f"Resolved forward multiple:      {multiple.point_estimate:,.2f}x",
            f"Reasonable range:                {multiple.lower_bound:,.2f}x-"
            f"{multiple.upper_bound:,.2f}x",
            "Range evidence: DCF anchor, peer IQR, and synchronized premium IQR",
            "Confidence:",
            f"  peer baseline:                {multiple.peer_confidence.value}",
            f"  target history:               "
            f"{multiple.target_history_confidence.value}",
            f"  premium persistence:          "
            f"{multiple.premium_persistence_confidence.value}",
            f"  overall relative valuation:   {multiple.confidence.value}",
            f"Peer sample: {multiple.sample_size}",
            "",
            f"{'Case':<12}{'Multiple':>12}{'Target-date price':>22}{'Present value':>20}",
            "-" * 66,
        ]
        for case in (result.lower_case, result.point_case, result.upper_case):
            lines.append(
                f"{case.label:<12}{case.multiple:>11,.2f}x"
                f"{case.implied_value_per_share:>18,.2f} {result.currency}"
                f"{case.present_value_per_share:>16,.2f} {result.currency}"
            )
        if result.intrinsic_value_per_share is not None:
            difference = (
                result.point_case.present_value_per_share
                - result.intrinsic_value_per_share
            )
            lines.extend(
                [
                    "",
                    "MODEL COMPARISON",
                    f"Intrinsic FCFF DCF:             "
                    f"{result.intrinsic_value_per_share:,.2f} {result.currency}",
                    f"Relative target-date price:     "
                    f"{result.point_case.implied_value_per_share:,.2f} "
                    f"{result.currency}",
                    f"Relative present-value equivalent today: "
                    f"{result.point_case.present_value_per_share:,.2f} "
                    f"{result.currency}",
                    f"Market-premium difference:      {difference:+,.2f} "
                    f"{result.currency}",
                    "The DCF values forecast cash flows; the relative estimate "
                    "also retains an evidence-constrained market premium.",
                ]
            )
        if result.current_price is not None:
            lines.extend(
                [
                    f"Current price:                   {result.current_price:,.2f} "
                    f"{result.currency}",
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
                    f"Analyst target price:            "
                    f"{result.analyst_target_price:,.2f} {result.currency}",
                    "Analyst target vs resolved target-date price: "
                    f"{result.analyst_target_price - result.point_case.implied_value_per_share:+,.2f} "
                    f"{result.currency}",
                    f"Analyst-target implied multiple: "
                    f"{result.analyst_target_implied_multiple:,.2f}x "
                    f"{ValuationSelectionConsolePresenter._label(result.basis.value)}",
                ]
            )
        if result.warnings:
            lines.extend(["", "WARNINGS"])
            lines.extend(f"- {warning}" for warning in result.warnings)
        return "\n".join(lines)

