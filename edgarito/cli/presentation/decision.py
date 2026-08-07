from __future__ import annotations

from edgarito.cli.presentation._valuation_format import (
    format_currency,
    section,
)
from edgarito.services.valuation import (
    DecisionScenario,
    DecisionValuationResult,
    ReverseDcfStatus,
)


class DecisionValuationConsolePresenter:
    """Render decision evidence without deriving or changing valuation results."""

    def render(
        self,
        result: DecisionValuationResult,
        *,
        show_scenarios: bool = False,
        show_sensitivity: bool = False,
        show_reverse_dcf: bool = False,
        verbose: bool = False,
        include_warnings: bool = True,
    ) -> str:
        lines = self.render_details(
            result,
            show_scenarios=show_scenarios,
            show_sensitivity=show_sensitivity,
            show_reverse_dcf=show_reverse_dcf,
            verbose=verbose,
        )
        if include_warnings and result.warnings:
            if lines:
                lines.append("")
            lines.extend([*section("CONSOLIDATED WARNINGS")])
            lines.extend(f"- {warning}" for warning in result.warnings)
        if lines:
            lines.append("")
        lines.extend(self.render_summary(result, verbose=verbose))
        return "\n".join(lines)

    def render_details(
        self,
        result: DecisionValuationResult,
        *,
        show_scenarios: bool,
        show_sensitivity: bool,
        show_reverse_dcf: bool,
        verbose: bool,
    ) -> list[str]:
        lines: list[str] = []
        if show_scenarios:
            lines.extend(self._scenario_details(result, verbose=verbose))
        if show_sensitivity:
            if lines:
                lines.append("")
            lines.extend(self._sensitivity_details(result, verbose=verbose))
        if show_reverse_dcf:
            if lines:
                lines.append("")
            lines.extend(self._reverse_details(result, verbose=verbose))
        return lines

    def render_summary(
        self, result: DecisionValuationResult, *, verbose: bool = False
    ) -> list[str]:
        lines = [
            *section("DECISION SUMMARY"),
            f"Current price: {format_currency(result.current_price, result.currency)}",
            "",
            f"{'Evidence':<18}{f'Value/share ({result.currency})':>20}"
            f"{'Upside/(downside)':>20}{'Margin of safety':>18}",
            "-" * 76,
        ]
        for comparison in result.price_comparisons:
            margin = (
                f"{comparison.margin_of_safety:+,.1f}%"
                if comparison.margin_of_safety is not None
                else "n/a"
            )
            lines.append(
                f"{comparison.label:<18}{comparison.value_per_share:>20,.2f}"
                f"{comparison.upside_downside:>+19,.1f}%{margin:>18}"
            )
        assessment = result.assessment
        lines.extend(
            [
                "",
                f"Intrinsic assessment: {assessment.intrinsic.value}",
                *(
                    [f"Relative assessment: {assessment.relative.value}"]
                    if assessment.relative is not None
                    else []
                ),
                f"Overall assessment: {assessment.overall}",
            ]
        )
        market_growth = next(
            (
                item
                for item in result.reverse_dcf
                if item.variable.value == "revenue_growth"
            ),
            None,
        )
        if market_growth is not None:
            if market_growth.status == ReverseDcfStatus.SOLVED:
                lines.append(
                    "Main reverse-DCF implied expectation: initial revenue growth "
                    f"{market_growth.implied_value:,.1f}% vs "
                    f"{market_growth.base_value:,.1f}% base"
                )
            else:
                lines.append(
                    "Main reverse-DCF implied expectation: initial revenue growth "
                    "has no solution within "
                    f"{market_growth.lower_bound:,.1f}% to "
                    f"{market_growth.upper_bound:,.1f}%"
                )
        if verbose:
            if assessment.model_dispersion is not None:
                lines.append(
                    f"Intrinsic/relative dispersion: {assessment.model_dispersion}"
                )
            lines.extend(
                [
                    "Margin-of-safety convention: 1 - current price / estimated "
                    "value; negative means price exceeds estimated value.",
                    f"Decision methodology: {result.methodology}",
                ]
            )
        return lines

    @staticmethod
    def _scenario_details(
        result: DecisionValuationResult, *, verbose: bool
    ) -> list[str]:
        lines = [*section("SCENARIOS")]
        headers = tuple(
            case.scenario.value.title() for case in result.intrinsic_scenarios
        )
        lines.extend(
            [
                f"{'Assumption':<30}{headers[0]:>12}{headers[1]:>12}{headers[2]:>12}",
                "-" * 66,
            ]
        )
        base_assumptions = result.intrinsic_scenarios[1].assumptions
        for index, assumption in enumerate(base_assumptions):
            values = tuple(
                case.assumptions[index].value for case in result.intrinsic_scenarios
            )
            lines.append(
                f"{assumption.name:<30}{values[0]:>11,.2f}%"
                f"{values[1]:>11,.2f}%{values[2]:>11,.2f}%"
            )
        values = tuple(case.value_per_share for case in result.intrinsic_scenarios)
        lines.append(
            f"{'Intrinsic value/share':<30}{values[0]:>12,.2f}"
            f"{values[1]:>12,.2f}{values[2]:>12,.2f}"
        )
        if result.relative_scenarios:
            multiples = tuple(case.multiple for case in result.relative_scenarios)
            values = tuple(case.value_per_share for case in result.relative_scenarios)
            lines.extend(
                [
                    f"{'Relative multiple':<30}{multiples[0]:>11,.2f}x"
                    f"{multiples[1]:>11,.2f}x{multiples[2]:>11,.2f}x",
                    f"{'Relative value/share':<30}{values[0]:>12,.2f}"
                    f"{values[1]:>12,.2f}{values[2]:>12,.2f}",
                ]
            )
        if verbose:
            changed = {
                assumption.name
                for case in result.intrinsic_scenarios
                if case.scenario != DecisionScenario.BASE
                for assumption in case.assumptions
                if assumption.changed
            }
            lines.append(
                "Scenario policy changed: "
                + (", ".join(sorted(changed)) if changed else "none")
            )
            lines.append(f"Methodology: {result.methodology}")
        return lines

    @staticmethod
    def _sensitivity_details(
        result: DecisionValuationResult, *, verbose: bool
    ) -> list[str]:
        lines: list[str] = []
        for index, table in enumerate(result.sensitivity_tables):
            if index:
                lines.append("")
            axes_label = f"{table.row_label} / {table.column_label}"
            lines.extend(
                [
                    *section(f"SENSITIVITY: {table.name}"),
                    f"{axes_label:<16}"
                    + "".join(f"{value:>11,.2f}%" for value in table.column_values),
                    "-" * (16 + 11 * len(table.column_values)),
                ]
            )
            for row_value, row in zip(table.row_values, table.cells, strict=True):
                rendered = "".join(
                    (
                        f"{cell.value_per_share:>11,.2f}"
                        if cell.value_per_share is not None
                        else f"{'invalid':>11}"
                    )
                    for cell in row
                )
                lines.append(f"{row_value:>14,.2f}% {rendered}")
            if verbose:
                lines.append(f"Methodology: {table.methodology}")
        return lines

    @staticmethod
    def _reverse_details(
        result: DecisionValuationResult, *, verbose: bool
    ) -> list[str]:
        if not result.reverse_dcf:
            return []
        lines = [
            *section("REVERSE DCF"),
            f"{'Assumption':<24}{'Base':>11}{'Implied':>12}{'Search range':>24}",
            "-" * 71,
        ]
        for solution in result.reverse_dcf:
            label = solution.variable.value.replace("_", " ").title()
            implied = (
                f"{solution.implied_value:,.2f}%"
                if solution.implied_value is not None
                else "no solution"
            )
            search_range = f"{solution.lower_bound:,.2f}%..{solution.upper_bound:,.2f}%"
            lines.append(
                f"{label:<24}{solution.base_value:>10,.2f}%"
                f"{implied:>12}{search_range:>24}"
            )
            if verbose:
                lines.append(f"  {solution.explanation}")
        lines.append(
            "Each assumption is solved independently; rows are not a combined forecast."
        )
        return lines


__all__ = ["DecisionValuationConsolePresenter"]
