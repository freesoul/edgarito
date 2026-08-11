from __future__ import annotations

import datetime
from decimal import Decimal

from edgarito.cli.presentation._valuation_format import section, subsection
from edgarito.schemas.valuation.assumptions import ValuationAssumptionKind
from edgarito.services.valuation import FcffDcfResult


class FcffDcfConsolePresenter:
    def render(
        self,
        result: FcffDcfResult,
        *,
        profile_name: str | None = None,
        verbose: bool = False,
        include_warnings: bool = True,
    ) -> str:
        identifier = result.ticker or result.company_id
        values = [
            result.enterprise_value,
            result.equity_value,
            result.capital_bridge.net_debt,
            result.capital_bridge.non_operating_assets,
            result.terminal_value.terminal_value,
            *(
                item.amount
                for item in result.explicit_forecast_present_value.cash_flows
            ),
        ]
        if result.share_repurchases is not None:
            values.extend(
                [
                    result.share_repurchases.total_cash_spent,
                    result.share_repurchases.present_value_cash_spent,
                    result.share_repurchases.residual_equity_value,
                ]
            )
        scale, suffix = self._scale_values(values)
        share_count_label = result.capital_bridge.share_count_label
        share_dilution_sensitivities = (
            result.share_dilution_sensitivities if result.equity_value > 0 else ()
        )
        share_scale, share_suffix = self._scale_values(
            [
                result.capital_bridge.diluted_shares,
                *(item.share_count for item in share_dilution_sensitivities),
            ]
        )
        amount_unit = f"{result.unit} {suffix}".rstrip()
        timing = result.parameters.cash_flow_timing.value.replace("_", " ")
        terminal_method = result.parameters.terminal_method.value.replace("_", " ")

        lines = [
            *section("VALUATION SETUP"),
            f"{identifier} - {result.company_name}",
            f"Valuation date: {result.valuation_date.isoformat()} | "
            f"Provider: {result.provider.upper()} | Profile: "
            f"{profile_name or 'unspecified'}",
            f"Model: FCFF DCF | Cash-flow timing: {timing} | "
            f"Terminal method: {terminal_method}",
            self._snapshot_label(result),
            "Availability mode: "
            f"{(result.observation_availability_mode or 'unspecified').replace('_', ' ')}",
            "",
            *subsection("DEFAULT ASSUMPTIONS"),
            *self._compact_assumptions(result),
        ]
        if verbose:
            lines.extend(
                [
                    "",
                    *self._economic_model_details(result),
                    "",
                    *self._audit_details(result),
                ]
            )

        lines.extend(
            [
                "",
                *section("INTRINSIC VALUATION"),
                f"FCFF forecast ({amount_unit})",
                f"{'Cash flow':<24}{'Period':>9}{'FCFF':>16}"
                f"{'Discount':>12}{'Present value':>18}",
                "-" * 79,
            ]
        )
        for item in result.explicit_forecast_present_value.cash_flows:
            lines.append(
                f"{(item.label or 'FCFF'):<24}{item.period:>9,.1f}"
                f"{item.amount / scale:>16,.1f}{item.discount_factor:>12,.4f}"
                f"{item.present_value / scale:>18,.1f}"
            )
        terminal = result.terminal_present_value
        lines.append(
            f"{'Terminal value':<24}{terminal.period:>9,.1f}"
            f"{terminal.amount / scale:>16,.1f}{terminal.discount_factor:>12,.4f}"
            f"{terminal.present_value / scale:>18,.1f}"
        )
        lines.extend(
            [
                "",
                *subsection("EV → EQUITY BRIDGE"),
                f"{'Explicit FCFF PV':<30}{result.explicit_forecast_present_value.total_present_value / scale:>16,.1f} {amount_unit}",
                f"{'Terminal PV':<30}{result.terminal_present_value.present_value / scale:>16,.1f} {amount_unit}",
                f"{'Enterprise value':<30}{result.enterprise_value / scale:>16,.1f} {amount_unit}",
                f"{'Less: net debt':<30}{result.capital_bridge.net_debt / scale:>16,.1f} {amount_unit}",
                f"{'Add: non-operating assets':<30}{result.capital_bridge.non_operating_assets / scale:>16,.1f} {amount_unit}",
                f"{'Equity value':<30}{result.equity_value / scale:>16,.1f} {amount_unit}",
                f"{share_count_label:<30}{result.capital_bridge.diluted_shares / share_scale:>16,.1f} {share_suffix or 'units'}",
                f"{'Intrinsic value/share':<30}{result.value_per_share:>16,.2f} {result.unit}",
            ]
        )
        if share_dilution_sensitivities:
            lines.extend(
                self._share_dilution_details(
                    result,
                    share_scale,
                    share_suffix,
                    share_count_label,
                )
            )
        if result.terminal_value_percentage is not None:
            lines.append(
                f"{'Terminal PV / enterprise value':<30}"
                f"{result.terminal_value_percentage:>16,.1f}%"
            )
        if result.share_repurchases is not None:
            lines.extend(
                self._repurchase_details(
                    result, scale, share_scale, amount_unit, share_suffix
                )
            )
        if include_warnings and result.warnings:
            lines.extend(["", *subsection("WARNINGS")])
            lines.extend(f"- {warning}" for warning in result.warnings)
        return "\n".join(lines)

    def _economic_model_details(self, result: FcffDcfResult) -> list[str]:
        lines = [*section("ECONOMIC FCFF MODEL")]
        if not result.forecast_cell_audits:
            lines.append("No economic FCFF cell audit metadata is available.")
            return lines

        years = sorted(result.forecast_cell_audits)
        amount_values = [
            audit.value
            for cells in result.forecast_cell_audits.values()
            for key, audit in cells.items()
            if key
            in {
                "revenue",
                "operating_income",
                "nopat",
                "depreciation_and_amortization",
                "capital_expenditures",
                "change_in_operating_working_capital",
                "fcff",
            }
        ]
        scale, suffix = self._scale_values(amount_values)
        amount_unit = f"{result.unit} {suffix}".rstrip()
        rows = (
            ("revenue_growth", "Revenue growth (%) [driver path]", True),
            (
                "revenue",
                f"Revenue ({amount_unit}) [prior revenue × (1 + growth / 100)]",
                False,
            ),
            ("operating_margin", "Operating margin (%) [driver path]", True),
            (
                "operating_income",
                f"EBIT / operating income ({amount_unit}) [revenue × margin / 100]",
                False,
            ),
            ("tax_rate", "Tax rate (%) [driver path]", True),
            (
                "nopat",
                f"NOPAT ({amount_unit}) [EBIT × (1 - tax rate / 100)]",
                False,
            ),
            (
                "depreciation_and_amortization",
                f"D&A ({amount_unit}) [revenue × D&A/revenue / 100]",
                False,
            ),
            (
                "capital_expenditures",
                f"Capex ({amount_unit}) [revenue × capex/revenue / 100]",
                False,
            ),
            (
                "change_in_operating_working_capital",
                f"Change in operating WC ({amount_unit}) [ending WC - prior WC]",
                False,
            ),
            (
                "fcff",
                f"FCFF ({amount_unit}) [NOPAT + D&A - capex - Δ NWC]",
                False,
            ),
        )
        label_width = max(56, max(len(label) for _, label, _ in rows) + 2)
        lines.extend(
            [
                f"{'Economic cell / formula':<{label_width}}"
                + "".join(f"{'FY' + str(year) + 'E':>14}" for year in years),
                "-" * (label_width + 14 * len(years)),
            ]
        )
        for key, label, is_percent in rows:
            values = []
            for year in years:
                audit = result.forecast_cell_audits[year].get(key)
                values.append(self._format_economic_value(audit, scale, is_percent))
            lines.append(
                f"{label:<{label_width}}" + "".join(f"{value:>14}" for value in values)
            )

        lines.extend(["", *subsection("ECONOMIC CELL PROVENANCE")])
        for year in years:
            lines.append(f"FY{year}E:")
            for key, _label, _is_percent in rows:
                audit = result.forecast_cell_audits[year].get(key)
                if audit is None:
                    lines.append(
                        f"  {key}: source=unknown/legacy | "
                        "method=legacy forecast observation | confidence=low"
                    )
                    continue
                lines.append(
                    f"  {key}: source={audit.source} | "
                    f"method={audit.method} | confidence={audit.confidence}"
                )
        return lines

    def _share_dilution_details(
        self,
        result: FcffDcfResult,
        share_scale: Decimal,
        share_suffix: str,
        share_count_label: str,
    ) -> list[str]:
        lines = [
            "",
            *subsection("SHARE DILUTION SENSITIVITY"),
            "Equity value held constant; only the share-count denominator changes.",
            f"Base denominator: {share_count_label}",
            f"{'Additional dilution':<30}{'Share count':>16}{'Value/share':>18}",
            "-" * 66,
        ]
        for item in result.share_dilution_sensitivities:
            dilution_label = f"+{item.dilution_percentage:,.0f}%"
            lines.append(
                f"{dilution_label:<30}"
                f"{item.share_count / share_scale:>16,.2f}"
                f"{item.value_per_share:>18,.2f} {result.unit}"
            )
        if share_suffix:
            lines.append(f"Share-count unit scale: {share_suffix}")
        return lines

    @staticmethod
    def _format_economic_value(audit, scale: Decimal, is_percent: bool) -> str:
        if audit is None:
            return "n/a"
        if is_percent:
            return f"{audit.value:,.2f}%"
        return f"{audit.value / scale:,.1f}"

    def _compact_assumptions(self, result: FcffDcfResult) -> list[str]:
        cost_of_equity = self._assumption_value(
            result, ValuationAssumptionKind.COST_OF_EQUITY
        )
        beta = self._assumption_value(result, ValuationAssumptionKind.LEVERED_BETA)
        tax_rate = self._assumption_value(
            result, ValuationAssumptionKind.NORMALIZED_TAX_RATE
        )
        terminal_roic = (
            result.multistage_plan.terminal_return_on_invested_capital
            if result.multistage_plan is not None
            else self._assumption_value(result, ValuationAssumptionKind.TERMINAL_ROIC)
        )
        terminal_growth = result.parameters.perpetual_growth_rate
        projection = self._projection_label(result)
        seed_date = (
            result.forecast_seed_period_end.isoformat()
            if result.forecast_seed_period_end
            else "unavailable"
        )
        rows = (
            ("WACC", self._percent(result.parameters.wacc)),
            ("Terminal growth", self._percent(terminal_growth)),
            ("Terminal ROIC", self._percent(terminal_roic)),
            ("Cost of equity", self._percent(cost_of_equity)),
            ("Beta", f"{beta:,.2f}x" if beta is not None else "unavailable"),
            ("Tax rate", self._percent(tax_rate)),
            (
                "Forecast seed / method",
                f"{result.forecast_seed_type} through {seed_date} / "
                f"{self._seed_context(result)}",
            ),
            ("Projection structure", projection),
        )
        return [f"{label + ':':<25}{value}" for label, value in rows]

    def _audit_details(self, result: FcffDcfResult) -> list[str]:
        lines = [*subsection("ASSUMPTION AND PROVENANCE AUDIT")]
        lines.extend(
            [
                f"WACC source: {result.parameters.wacc_source}",
                f"Forecast seed methodology: {result.forecast_seed_methodology}",
            ]
        )
        if result.forecast_assumption_sources:
            lines.append("Forecast driver sources:")
            for driver, source in sorted(result.forecast_assumption_sources.items()):
                lines.append(
                    f"  {driver.replace('_', ' ').title()}: {source.replace('_', '-')}"
                )
        if (
            result.provider.casefold() == "yahoo"
            and result.observation_availability_mode == "current_snapshot"
        ):
            lines.append(
                "Availability policy: Yahoo observations present in the retrieved "
                "current snapshot are current evidence when their period has ended; "
                "historical reconstruction still uses conservative publication lags"
            )
        if result.parameters.perpetual_growth_rate is not None:
            lines.append(
                "Terminal growth source: "
                f"{result.parameters.perpetual_growth_source or 'explicit'}"
            )
        if result.multistage_plan is not None:
            plan = result.multistage_plan
            if plan.forward_growth_rate is not None:
                current_growth_label = (
                    "Current FY growth"
                    if result.forecast_seed_type in {"YTD+forecast", "YTD run-rate", "TTM"}
                    else "First projected FY growth"
                )
                lines.extend(
                    [
                        "FORWARD REVENUE OUTLOOK",
                        f"{current_growth_label}: "
                        f"{self._format_growth(plan.current_growth_rate)} "
                        f"[{result.forecast_seed_type}]",
                        "Forward growth anchor: "
                        f"{self._format_growth(plan.forward_growth_rate)}",
                        f"Source: {plan.forward_growth_source or 'unavailable'}",
                        f"Confidence: {plan.forward_growth_confidence or 'unavailable'}",
                        "Historical inputs: "
                        f"{self._format_growth_path(plan.historical_growth_path)}",
                        "Management guidance: "
                        f"{self._format_growth_path(plan.management_guidance_path)}",
                        "Forward estimates: "
                        f"{self._format_growth_path(plan.forward_estimates_path)}",
                        "Terminal growth: "
                        f"{self._format_growth(result.parameters.perpetual_growth_rate)}",
                        f"Stable-state supported: "
                        f"{'yes' if plan.stable_state_supported else 'no'}",
                    ]
                )
                if plan.forward_estimate_diagnostics:
                    lines.extend(
                        [
                            "FORWARD ESTIMATE RETRIEVAL",
                            *self._forward_estimate_diagnostics(plan),
                        ]
                    )
                if plan.forward_revenue_estimates:
                    estimate_years = plan.forward_estimate_years or tuple(
                        estimate.fiscal_year
                        for estimate in plan.forward_revenue_estimates
                    )
                    years = ", ".join(
                        f"FY{year}" for year in estimate_years
                    )
                    lines.extend(
                        [
                            "Selected provider: "
                            f"{self._forward_provider_label(plan.forward_estimate_provider)}",
                            f"Years: {years or 'unavailable'}",
                        ]
                    )
                    growth_by_year = dict(plan.forward_growth_path_by_year)
                    for estimate in plan.forward_revenue_estimates:
                        value = estimate.midpoint
                        lines.append(
                            f"FY{estimate.fiscal_year} revenue estimate: "
                            f"{self._format_revenue_estimate(value, result.unit)} | "
                            f"source={estimate.source} | "
                            f"implied growth={self._format_growth(growth_by_year.get(estimate.fiscal_year))} | "
                            f"analysts={estimate.analyst_count or 'unavailable'} | "
                            f"confidence={estimate.confidence or 'unavailable'}"
                        )
            if plan.terminal_roic_source:
                lines.append(
                    f"Terminal ROIC: {plan.terminal_roic_source} | confidence "
                    f"{plan.terminal_roic_confidence or 'unspecified'}"
                )
            if plan.terminal_roic_methodology:
                lines.append(f"Terminal ROIC method: {plan.terminal_roic_methodology}")
            if plan.terminal_return_on_invested_capital is not None:
                details = (
                    f"Stable reinvestment: {plan.terminal_reinvestment_rate:,.2f}% "
                    f"of NOPAT at {plan.terminal_return_on_invested_capital:,.2f}% ROIC"
                )
                if plan.terminal_capex_to_revenue is not None:
                    details += (
                        f" | capex/revenue {plan.terminal_capex_to_revenue:,.2f}%"
                    )
                if plan.depreciable_asset_life_years is not None:
                    details += (
                        f" | asset life {plan.depreciable_asset_life_years} years"
                    )
                lines.append(details)
            if plan.capex_transition_years:
                lines.append(
                    f"CAPEX transition: {plan.capex_transition_years} years | "
                    f"{plan.capex_benefits_disclosure}"
                )
        if result.assumptions is not None:
            lines.append("Resolved assumptions:")
            for assumption in result.assumptions.assumptions:
                provenance = assumption.provenance
                source = provenance.provider or provenance.origin.value
                metadata = [source]
                if provenance.dataset:
                    metadata.append(provenance.dataset)
                if provenance.version:
                    metadata.append(provenance.version)
                if provenance.observed_on:
                    metadata.append(f"observed {provenance.observed_on.isoformat()}")
                lines.append(
                    f"  {assumption.kind.value}: {assumption.value:,.3f} "
                    f"[{' | '.join(metadata)}]"
                )
                if provenance.methodology:
                    lines.append(f"    Method: {provenance.methodology}")
                if assumption.rationale:
                    lines.append(f"    Rationale: {assumption.rationale}")
        bridge = result.capital_bridge
        lines.extend(
            [
                f"Net debt source: {bridge.net_debt_source}",
                f"Non-operating assets source: {bridge.non_operating_assets_source}",
                f"Share-count basis: {bridge.share_count_basis.value}",
                f"Shares source: {bridge.shares_source}",
                "Capital bridge dates: "
                f"debt={bridge.debt_date or 'explicit/unknown'}, "
                f"cash={bridge.cash_date or 'explicit/unknown'}, "
                f"shares={bridge.shares_date or 'explicit/unknown'}, "
                "non-operating assets="
                f"{bridge.non_operating_assets_date or 'none/explicit'}",
                f"Debt scope: {bridge.debt_scope}",
            ]
        )
        return lines

    @staticmethod
    def _format_growth(value: Decimal | None) -> str:
        return "unavailable" if value is None else f"{value:+.2f}%"

    @classmethod
    def _format_growth_path(cls, values: tuple[Decimal, ...]) -> str:
        if not values:
            return "unavailable"
        return "[" + ", ".join(cls._format_growth(value) for value in values) + "]"

    @classmethod
    def _forward_estimate_diagnostics(cls, plan) -> list[str]:
        lines = []
        for diagnostic in plan.forward_estimate_diagnostics:
            status = getattr(diagnostic.status, "value", diagnostic.status)
            status = str(status).replace("_", " ")
            provider = {
                "alphavantage": "Alpha Vantage",
                "alpha_vantage": "Alpha Vantage",
                "yahoo": "Yahoo",
            }.get(diagnostic.provider.casefold(), diagnostic.provider)
            detail = f"{provider}: {status}"
            if diagnostic.estimate_count:
                detail += f" — {diagnostic.estimate_count} annual estimate(s)"
            if diagnostic.years:
                detail += " (" + ", ".join(f"FY{year}" for year in diagnostic.years) + ")"
            if diagnostic.reason:
                detail += f" — {diagnostic.reason}"
            lines.append(detail)
        return lines

    @staticmethod
    def _forward_provider_label(provider: str | None) -> str:
        if provider is None:
            return "unavailable"
        return {
            "alphavantage": "Alpha Vantage",
            "alpha_vantage": "Alpha Vantage",
            "yahoo": "Yahoo",
        }.get(provider.casefold(), provider)

    @staticmethod
    def _format_revenue_estimate(value: Decimal | None, unit: str) -> str:
        if value is None:
            return "unavailable"
        absolute = abs(value)
        if absolute >= Decimal("1e9"):
            return f"{unit} {value / Decimal('1e9'):,.1f}B"
        if absolute >= Decimal("1e6"):
            return f"{unit} {value / Decimal('1e6'):,.1f}M"
        return f"{unit} {value:,.0f}"

    def _repurchase_details(
        self,
        result: FcffDcfResult,
        scale: Decimal,
        share_scale: Decimal,
        amount_unit: str,
        share_suffix: str,
    ) -> list[str]:
        repurchases = result.share_repurchases
        if repurchases is None:
            return []
        lines = [
            "",
            *subsection("SHARE REPURCHASE ANALYSIS"),
            f"{'Period':<12}{'Cash spent':>16}{'Purchase price':>17}"
            f"{'Shares retired':>17}{'PV cash':>16}",
            "-" * 78,
        ]
        for period in repurchases.periods:
            lines.append(
                f"{f'FY{period.fiscal_year}E':<12}"
                f"{period.cash_spent / scale:>16,.1f}"
                f"{period.purchase_price:>17,.2f}"
                f"{period.shares_repurchased / share_scale:>17,.2f}"
                f"{period.present_value_cash_spent / scale:>16,.1f}"
            )
        lines.extend(
            [
                f"Total buyback cash: {repurchases.total_cash_spent / scale:,.1f} {amount_unit}",
                f"PV of buyback cash: {repurchases.present_value_cash_spent / scale:,.1f} {amount_unit}",
                f"Remaining diluted shares: {repurchases.ending_shares / share_scale:,.2f} {share_suffix or 'units'}",
                f"Intrinsic value/share after buybacks: {repurchases.value_per_remaining_share:,.2f} {result.unit}",
                f"Buyback accretion/(dilution): {repurchases.accretion_percentage:+,.1f}%",
            ]
        )
        return lines

    @staticmethod
    def _projection_label(result: FcffDcfResult) -> str:
        plan = result.multistage_plan
        if plan is None:
            return "constant explicit forecast"
        stages = []
        if plan.current_growth_years:
            stages.append(f"{plan.current_growth_years} current")
        if plan.explicit_growth_prefix_years:
            stages.append(f"{plan.explicit_growth_prefix_years} explicit")
        if plan.high_growth_years:
            stages.append(f"{plan.high_growth_years} near-term")
        if plan.transition_years:
            stages.append(f"{plan.transition_years} transition")
        if plan.stable_years:
            stages.append(f"{plan.stable_years} stable")
        extension = (
            f"; extended from {plan.requested_years} to {plan.effective_years} years"
            if plan.extended_to_stable
            else f"; {plan.effective_years} years"
        )
        return "adaptive multistage: " + " + ".join(stages) + extension

    @staticmethod
    def _short_seed_method(methodology: str) -> str:
        return methodology.split(";", 1)[0]

    @classmethod
    def _seed_context(cls, result: FcffDcfResult) -> str:
        if result.forecast_actual_quarters:
            noun = "quarter" if result.forecast_actual_quarters == 1 else "quarters"
            source = (
                f" from {result.provider.upper()} snapshot"
                if result.observation_availability_mode == "current_snapshot"
                else ""
            )
            return (
                f"{result.forecast_actual_quarters} currently reported fiscal "
                f"{noun}{source}"
            )
        return cls._short_seed_method(result.forecast_seed_methodology)

    @staticmethod
    def _snapshot_label(result: FcffDcfResult) -> str:
        retrieved_at = result.financial_snapshot_retrieved_at
        if retrieved_at is None:
            return f"Financial snapshot: {result.provider.upper()} retrieval time unavailable"
        timestamp = retrieved_at.astimezone(datetime.timezone.utc)
        return (
            f"Financial snapshot: {result.provider.upper()} retrieved "
            f"{timestamp:%Y-%m-%d %H:%M} UTC"
        )

    @staticmethod
    def _assumption_value(
        result: FcffDcfResult, kind: ValuationAssumptionKind
    ) -> Decimal | None:
        if result.assumptions is None:
            return None
        assumption = result.assumptions.find(kind)
        return assumption.value if assumption is not None else None

    @staticmethod
    def _percent(value: Decimal | None) -> str:
        return f"{value:,.2f}%" if value is not None else "unavailable"

    @staticmethod
    def _scale_values(values: list[Decimal]) -> tuple[Decimal, str]:
        largest = max((abs(value) for value in values), default=Decimal(0))
        if largest >= Decimal("1000000000"):
            return Decimal("1000000000"), "B"
        if largest >= Decimal("1000000"):
            return Decimal("1000000"), "M"
        if largest >= Decimal("1000"):
            return Decimal("1000"), "K"
        return Decimal(1), ""
