from decimal import Decimal

from edgarito.services.valuation import FcffDcfResult


class FcffDcfConsolePresenter:
    def render(self, result: FcffDcfResult, *, profile_name: str | None = None) -> str:
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
        share_scale, share_suffix = self._scale_values(
            [result.capital_bridge.diluted_shares]
        )
        amount_unit = f"{result.unit} {suffix}".rstrip()
        timing = result.parameters.cash_flow_timing.value.replace("_", " ")
        terminal_method = result.parameters.terminal_method.value.replace("_", " ")
        lines = [
            f"{identifier} - {result.company_name}",
            f"Provider: {result.provider.upper()} | Valuation date: "
            f"{result.valuation_date.isoformat()}",
            f"Valuation profile: {profile_name or 'unspecified'}",
            f"Model: FCFF DCF | Timing: {timing}",
            "Forecast seed: "
            f"{result.forecast_seed_type} through "
            f"{result.forecast_seed_period_end.isoformat() if result.forecast_seed_period_end else '-'}",
            f"Forecast seed method: {result.forecast_seed_methodology}",
            f"WACC: {result.parameters.wacc:,.2f}% ({result.parameters.wacc_source})",
            f"Terminal method: {terminal_method}",
        ]
        if result.multistage_plan is not None:
            plan = result.multistage_plan
            stages = []
            if plan.explicit_growth_prefix_years:
                stages.append(f"{plan.explicit_growth_prefix_years} explicit")
            if plan.high_growth_years:
                stages.append(f"{plan.high_growth_years} high-growth")
            if plan.transition_years:
                stages.append(f"{plan.transition_years} transition")
            if plan.stable_years:
                stages.append(f"{plan.stable_years} stable")
            extension = (
                f"; extended from {plan.requested_years} requested years"
                if plan.extended_to_stable
                else ""
            )
            lines.append(
                "Projection: adaptive multistage | "
                f"{' + '.join(stages)} years{extension} | stable growth anchor "
                f"{plan.terminal_growth_rate:,.2f}%"
            )
            if plan.terminal_return_on_invested_capital is not None:
                details = (
                    "Stable reinvestment: "
                    f"{plan.terminal_reinvestment_rate:,.2f}% of NOPAT at "
                    f"{plan.terminal_return_on_invested_capital:,.2f}% terminal ROIC"
                )
                if plan.terminal_capex_to_revenue is not None:
                    details += (
                        f" | terminal capex/revenue "
                        f"{plan.terminal_capex_to_revenue:,.2f}%"
                    )
                if plan.depreciable_asset_life_years is not None:
                    details += (
                        f" | {plan.depreciable_asset_life_years}-year depreciable "
                        "asset life"
                    )
                lines.append(details)
                lines.append(
                    "Terminal ROIC resolution: "
                    f"{plan.terminal_roic_source or 'unspecified'} | confidence "
                    f"{plan.terminal_roic_confidence or 'unspecified'}"
                )
                if plan.terminal_roic_methodology:
                    lines.append(
                        f"Terminal ROIC method: {plan.terminal_roic_methodology}"
                    )
        if result.parameters.perpetual_growth_rate is not None:
            source = result.parameters.perpetual_growth_source or "explicit"
            lines.append(
                "Terminal growth: "
                f"{result.parameters.perpetual_growth_rate:,.2f}% ({source})"
            )
        if result.parameters.exit_multiple is not None:
            metric = result.parameters.exit_metric.value.upper()
            lines.append(
                f"Terminal multiple: {result.parameters.exit_multiple:,.2f}x {metric}"
            )
        if result.assumptions is not None:
            lines.extend(["", "Resolved assumptions:"])
            for assumption in result.assumptions.assumptions:
                provider = (
                    assumption.provenance.provider or assumption.provenance.origin.value
                )
                lines.append(
                    f"  {assumption.kind.value}: {assumption.value:,.3f} [{provider}]"
                )
        lines.extend(
            [
                "",
                f"{'Cash flow':<26}{'Period':>10}{'FCFF':>18}{'Factor':>12}{'PV':>18}",
                "-" * 84,
            ]
        )
        for item in result.explicit_forecast_present_value.cash_flows:
            lines.append(
                f"{(item.label or 'FCFF'):<26}{item.period:>10,.1f}"
                f"{item.amount / scale:>18,.1f}{item.discount_factor:>12,.4f}"
                f"{item.present_value / scale:>18,.1f}"
            )
        terminal = result.terminal_present_value
        lines.append(
            f"{'Terminal value':<26}{terminal.period:>10,.1f}"
            f"{terminal.amount / scale:>18,.1f}{terminal.discount_factor:>12,.4f}"
            f"{terminal.present_value / scale:>18,.1f}"
        )
        lines.extend(
            [
                "",
                f"Explicit FCFF PV ({amount_unit}): "
                f"{result.explicit_forecast_present_value.total_present_value / scale:,.1f}",
                f"Terminal value PV ({amount_unit}): "
                f"{result.terminal_present_value.present_value / scale:,.1f}",
                f"Enterprise value ({amount_unit}): "
                f"{result.enterprise_value / scale:,.1f}",
                f"Less net debt ({amount_unit}): "
                f"{result.capital_bridge.net_debt / scale:,.1f}",
                f"Add non-operating investments ({amount_unit}): "
                f"{result.capital_bridge.non_operating_assets / scale:,.1f}",
                f"Equity value ({amount_unit}): {result.equity_value / scale:,.1f}",
                f"Diluted shares ({share_suffix or 'units'}): "
                f"{result.capital_bridge.diluted_shares / share_scale:,.1f}",
            ]
        )
        if result.share_repurchases is not None:
            repurchases = result.share_repurchases
            lines.extend(
                [
                    "",
                    "Share repurchase analysis",
                    f"{'Period':<12}{'Cash spent':>18}{'Purchase price':>18}"
                    f"{'Shares retired':>18}{'PV cash':>18}",
                    "-" * 84,
                ]
            )
            for period in repurchases.periods:
                lines.append(
                    f"{f'FY{period.fiscal_year}E':<12}"
                    f"{period.cash_spent / scale:>18,.1f}"
                    f"{period.purchase_price:>18,.2f}"
                    f"{period.shares_repurchased / share_scale:>18,.2f}"
                    f"{period.present_value_cash_spent / scale:>18,.1f}"
                )
            lines.extend(
                [
                    f"Total buyback cash ({amount_unit}): "
                    f"{repurchases.total_cash_spent / scale:,.1f}",
                    f"PV of buyback cash ({amount_unit}): "
                    f"{repurchases.present_value_cash_spent / scale:,.1f}",
                    f"Projected shares retired ({share_suffix or 'units'}): "
                    f"{repurchases.shares_repurchased / share_scale:,.2f}",
                    f"Remaining diluted shares ({share_suffix or 'units'}): "
                    f"{repurchases.ending_shares / share_scale:,.2f}",
                    f"Residual equity value ({amount_unit}): "
                    f"{repurchases.residual_equity_value / scale:,.1f}",
                    f"Buyback accretion / (dilution): "
                    f"{repurchases.accretion_percentage:+,.2f}%",
                    f"Repurchase discount rate: {repurchases.discount_rate:,.2f}% "
                    f"({repurchases.discount_rate_source})",
                    f"Repurchase-price growth: {repurchases.price_growth_rate:,.2f}%",
                    f"Purchase-price basis: {repurchases.purchase_price_source}",
                    f"Buyback source: {repurchases.source}",
                ]
            )
        if result.terminal_value_percentage is not None:
            lines.append(
                "Terminal PV / enterprise value: "
                f"{result.terminal_value_percentage:,.1f}%"
            )
        lines.extend(
            [
                "",
                f"Net debt source: {result.capital_bridge.net_debt_source}",
                "Non-operating investments source: "
                f"{result.capital_bridge.non_operating_assets_source}",
                f"Shares source: {result.capital_bridge.shares_source}",
                "Capital bridge dates: "
                f"debt={result.capital_bridge.debt_date or 'explicit/unknown'}, "
                f"cash={result.capital_bridge.cash_date or 'explicit/unknown'}, "
                f"shares={result.capital_bridge.shares_date or 'explicit/unknown'}, "
                "non-operating assets="
                f"{result.capital_bridge.non_operating_assets_date or 'none/explicit'}",
                f"Debt scope: {result.capital_bridge.debt_scope}",
            ]
        )
        if result.warnings:
            lines.extend(["", "WARNINGS"])
            lines.extend(f"- {warning}" for warning in result.warnings)
        lines.extend(["", "VALUATION CONCLUSION"])
        if result.share_repurchases is not None:
            lines.extend(
                [
                    f"Value per share without buybacks ({result.unit}): "
                    f"{result.value_per_share:,.2f}",
                    f"Final value per share after buybacks ({result.unit}): "
                    f"{result.share_repurchases.value_per_remaining_share:,.2f}",
                ]
            )
        else:
            lines.append(
                f"Final value per share ({result.unit}): {result.value_per_share:,.2f}"
            )
        return "\n".join(lines)

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

