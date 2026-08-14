import datetime
from decimal import Decimal

from edgarito.schemas.forecasting import FcffForecast
from edgarito.schemas.valuation.selection import RelativeValuationBasis
from edgarito.services.valuation.discounting import PresentValueService
from edgarito.services.valuation.models import (
    ComparableImpliedValuation,
    ComparableImpliedValuationCase,
    FcffDcfCapitalBridge,
    ResolvedMultiple,
)


class ComparableImpliedValuationService:
    """Convert peer, historical, and composite multiples into equity values."""

    def value(
        self,
        *,
        target_forecast: FcffForecast,
        capital_bridge: FcffDcfCapitalBridge,
        projected_shares: Decimal,
        resolved_multiple: ResolvedMultiple,
        valuation_date: datetime.date,
        horizon_years: Decimal,
        discount_rate: Decimal,
        current_price: Decimal | None = None,
        analyst_target_price: Decimal | None = None,
        intrinsic_value_per_share: Decimal | None = None,
    ) -> ComparableImpliedValuation:
        target_date = valuation_date + datetime.timedelta(
            days=int(horizon_years * Decimal(365))
        )
        observation = min(
            target_forecast.observations,
            key=lambda item: abs((item.period_end - target_date).days),
        )
        metric, label = self._forecast_metric(observation, resolved_multiple.basis)
        composite_cases = [
            self._case(
                label=label_name,
                multiple=multiple,
                metric=metric,
                bridge=capital_bridge,
                shares=projected_shares,
                discount_rate=discount_rate,
                horizon_years=horizon_years,
            )
            for label_name, multiple in (
                ("Lower", resolved_multiple.lower_bound),
                ("Resolved", resolved_multiple.point_estimate),
                ("Upper", resolved_multiple.upper_bound),
            )
        ]
        peer_anchor = resolved_multiple.peer_anchor
        peer_cases = self._independent_cases(
            label_prefix="Peer",
            multiples=(
                resolved_multiple.peer_anchor_percentile_25 or peer_anchor,
                peer_anchor,
                resolved_multiple.peer_anchor_percentile_75 or peer_anchor,
            ),
            metric=metric,
            bridge=capital_bridge,
            shares=projected_shares,
        )
        historical_anchor = resolved_multiple.historical_anchor
        historical_cases = self._independent_cases(
            label_prefix="Historical",
            multiples=(
                resolved_multiple.historical_percentile_25 or historical_anchor,
                historical_anchor,
                resolved_multiple.historical_percentile_75 or historical_anchor,
            ),
            metric=metric,
            bridge=capital_bridge,
            shares=projected_shares,
        )
        warnings = [
            *resolved_multiple.warnings,
            "Projected net debt and diluted shares are held flat because no "
            "capital-structure forecast was supplied",
        ]
        analyst_target_implied_multiple = None
        current_price_implied_multiple = None
        if current_price is not None:
            if current_price <= 0:
                raise ValueError("Current price must be positive")
            current_enterprise = (
                current_price * projected_shares
                + capital_bridge.net_debt
                - capital_bridge.non_operating_assets
            )
            current_price_implied_multiple = current_enterprise / metric
        if analyst_target_price is not None:
            if analyst_target_price <= 0:
                raise ValueError("Analyst target price must be positive")
            analyst_target_equity = analyst_target_price * projected_shares
            analyst_target_enterprise = (
                analyst_target_equity
                + capital_bridge.net_debt
                - capital_bridge.non_operating_assets
            )
            analyst_target_implied_multiple = analyst_target_enterprise / metric
        return ComparableImpliedValuation(
            provider=target_forecast.provider,
            company_id=target_forecast.company_id,
            company_name=target_forecast.company_name,
            ticker=target_forecast.ticker,
            valuation_date=valuation_date,
            target_date=target_date,
            horizon_years=horizon_years,
            currency=target_forecast.unit,
            basis=resolved_multiple.basis,
            forecast_metric=metric,
            forecast_metric_label=f"FY{observation.fiscal_year}E {label}",
            projected_net_debt=capital_bridge.net_debt,
            projected_diluted_shares=projected_shares,
            discount_rate=discount_rate,
            resolved_multiple=resolved_multiple,
            lower_case=composite_cases[0],
            point_case=composite_cases[1],
            upper_case=composite_cases[2],
            pure_peer_lower_case=peer_cases[0] if peer_cases else None,
            pure_peer_point_case=peer_cases[1] if peer_cases else None,
            pure_peer_upper_case=peer_cases[2] if peer_cases else None,
            historical_lower_case=historical_cases[0] if historical_cases else None,
            historical_point_case=historical_cases[1] if historical_cases else None,
            historical_upper_case=historical_cases[2] if historical_cases else None,
            composite_lower_case=composite_cases[0],
            composite_point_case=composite_cases[1],
            composite_upper_case=composite_cases[2],
            current_price=current_price,
            current_price_implied_multiple=current_price_implied_multiple,
            analyst_target_price=analyst_target_price,
            analyst_target_implied_multiple=analyst_target_implied_multiple,
            intrinsic_value_per_share=intrinsic_value_per_share,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _independent_cases(
        self,
        *,
        label_prefix,
        multiples,
        metric,
        bridge,
        shares,
    ):
        """Build target-date evidence values without intrinsic discounting."""

        if any(multiple is None or multiple <= 0 for multiple in multiples):
            return None
        return [
            self._case(
                label=label,
                multiple=multiple,
                metric=metric,
                bridge=bridge,
                shares=shares,
                discount_rate=None,
                horizon_years=None,
            )
            for label, multiple in zip(
                (
                    f"{label_prefix} lower",
                    f"{label_prefix} median",
                    f"{label_prefix} upper",
                ),
                multiples,
                strict=True,
            )
        ]

    @staticmethod
    def _forecast_metric(observation, basis):
        if basis == RelativeValuationBasis.EV_TO_EBITDA:
            return (
                observation.operating_income
                + observation.depreciation_and_amortization,
                "EBITDA",
            )
        if basis == RelativeValuationBasis.EV_TO_EBIT:
            return observation.operating_income, "EBIT"
        if basis == RelativeValuationBasis.EV_TO_REVENUE:
            return observation.revenue, "revenue"
        if basis == RelativeValuationBasis.EV_TO_FCF:
            return observation.fcff, "FCFF"
        raise ValueError(f"Unsupported implied valuation basis: {basis.value}")

    @staticmethod
    def _case(*, label, multiple, metric, bridge, shares, discount_rate, horizon_years):
        enterprise_value = metric * multiple
        equity_value = enterprise_value - bridge.net_debt + bridge.non_operating_assets
        value_per_share = equity_value / shares
        present_value = (
            value_per_share
            if discount_rate is None
            else value_per_share
            * PresentValueService.discount_factor(discount_rate, horizon_years)
        )
        return ComparableImpliedValuationCase(
            label=label,
            multiple=multiple,
            implied_enterprise_value=enterprise_value,
            implied_equity_value=equity_value,
            implied_value_per_share=value_per_share,
            present_value_per_share=present_value,
        )
