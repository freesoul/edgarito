import datetime
import math
from decimal import Decimal
from statistics import median

from edgarito.services.forecasting.models import FcffForecast
from edgarito.services.valuation.discounting import PresentValueService
from edgarito.services.valuation.models import (
    CompanyTradingMultiples,
    ComparableImpliedValuation,
    ComparableImpliedValuationCase,
    ComparableMultiplesReport,
    FcffDcfCapitalBridge,
    FcffDcfResult,
    HistoricalMultipleObservation,
    HistoricalMultipleSummary,
    MultipleConfidence,
    MultipleStatus,
    RelativeValuationBasis,
    ResolvedMultiple,
)
from edgarito.services.valuation.multiples import LtmMultiplesService


class HistoricalMultiplesService:
    """Recompute point-in-time LTM multiples at available quarterly dates."""

    def __init__(self, ltm_service: LtmMultiplesService | None = None):
        self._ltm_service = ltm_service or LtmMultiplesService()

    def compute(
        self,
        financials,
        market_data,
        basis: RelativeValuationBasis,
    ) -> HistoricalMultipleSummary:
        dates = sorted(
            {
                item.period_end
                for item in financials.observations
                if item.concept.value == "revenue"
                and item.granularity.value in {"quarterly", "annual"}
            }
        )
        observations_by_period = {}
        for observed_on in dates:
            try:
                snapshot = self._ltm_service.compute(
                    financials, market_data, as_of=observed_on
                )
            except ValueError:
                continue
            multiple = next(
                (
                    item
                    for item in snapshot.multiples
                    if item.basis == basis
                    and item.status == MultipleStatus.COMPUTED
                    and item.value is not None
                ),
                None,
            )
            if multiple is not None and multiple.value is not None:
                observations_by_period[snapshot.fundamentals.period_end] = (
                    HistoricalMultipleObservation(
                        observed_on=observed_on, value=multiple.value
                    )
                )

        observations = sorted(
            observations_by_period.values(), key=lambda item: item.observed_on
        )

        warnings = []
        if len(observations) < 4:
            warnings.append(
                "Fewer than four point-in-time observations are available; "
                "historical persistence will use a conservative fallback"
            )
        if not observations:
            return HistoricalMultipleSummary(basis=basis, warnings=tuple(warnings))
        values = sorted(item.value for item in observations)
        mean = sum(values, Decimal(0)) / Decimal(len(values))
        variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / Decimal(
            len(values)
        )
        volatility = variance.sqrt() / mean if mean > 0 else None
        trend = (
            (observations[-1].value / observations[0].value - Decimal(1))
            if len(observations) > 1
            else None
        )
        return HistoricalMultipleSummary(
            basis=basis,
            observations=tuple(observations),
            median=median(values),
            percentile_25=self._percentile(values, Decimal("0.25")),
            percentile_75=self._percentile(values, Decimal("0.75")),
            minimum=values[0],
            maximum=values[-1],
            current=observations[-1].value,
            volatility=volatility,
            trend=trend,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _percentile(values: list[Decimal], percentile: Decimal) -> Decimal:
        if len(values) == 1:
            return values[0]
        position = percentile * Decimal(len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - Decimal(lower)
        return values[lower] + (values[upper] - values[lower]) * weight


class MultipleResolver:
    """Resolve a market multiple as fundamental value plus persistent premium."""

    _SUPPORTED_BASES = {
        RelativeValuationBasis.EV_TO_EBITDA,
        RelativeValuationBasis.EV_TO_EBIT,
        RelativeValuationBasis.EV_TO_REVENUE,
        RelativeValuationBasis.EV_TO_FCF,
    }

    def resolve(
        self,
        *,
        basis: RelativeValuationBasis,
        target: CompanyTradingMultiples,
        target_history: HistoricalMultipleSummary | None,
        peer_histories: tuple[HistoricalMultipleSummary, ...] = (),
        peer_report: ComparableMultiplesReport,
        target_forecast: FcffForecast,
        intrinsic_valuation: FcffDcfResult,
        horizon_years: Decimal,
        policy,
    ) -> ResolvedMultiple:
        if basis not in self._SUPPORTED_BASES:
            raise ValueError(
                f"Automatic implied valuation does not yet support {basis.value}; "
                "use an enterprise-value basis"
            )
        fundamental = self._fundamental_anchor(
            basis, target_forecast, intrinsic_valuation
        )
        effective_history = target_history if policy.use_target_history else None
        target_current = self._target_multiple(target, basis)
        peer_summary = next(
            (item for item in peer_report.summaries if item.basis == basis), None
        )
        peer_anchor = peer_summary.median if peer_summary is not None else None
        historical_anchor = (
            effective_history.median
            if effective_history is not None
            else None
        )

        warnings = list(
            effective_history.warnings if effective_history is not None else ()
        )
        market_values = []
        if historical_anchor is not None:
            market_values.append(historical_anchor)
        elif policy.use_target_history and target_current is not None:
            market_values.append(target_current)
            warnings.append(
                "Target history is insufficient; current target multiple is used "
                "only as a low-confidence market anchor"
            )
        if policy.use_peer_median and peer_anchor is not None:
            market_values.append(peer_anchor)
        if not market_values:
            market_anchor = fundamental
            warnings.append(
                "No usable history or peer anchor is available; the relative "
                "estimate collapses to the DCF-consistent fundamental multiple"
            )
        else:
            market_anchor = median(market_values)

        (
            historical_persistence,
            historical_peer_premium,
            premium_history_sample_size,
            premium_mean_reversion_beta,
        ) = self._historical_persistence(
            effective_history, peer_histories, policy, warnings
        )
        fundamental_support, support_count = self._fundamental_support(
            target, peer_report.peers, warnings
        )
        horizon_retention = (
            Decimal(
                str(
                    math.exp(-float(policy.annual_premium_decay) * float(horizon_years))
                )
            )
            if policy.forecast_premium_mean_reversion
            else Decimal(1)
        )
        persistence = min(
            Decimal(1),
            historical_persistence * fundamental_support * horizon_retention,
        )
        observed_premium = market_anchor - fundamental
        if historical_peer_premium is None:
            historical_peer_premium = (
                (historical_anchor / peer_anchor - Decimal(1)) * Decimal(100)
                if historical_anchor is not None
                and peer_anchor is not None
                and peer_anchor > 0
                else None
            )
        point = fundamental + persistence * observed_premium
        range_step = policy.persistence_range_width
        lower_persistence = max(Decimal(0), persistence - range_step)
        upper_persistence = min(Decimal(1), persistence + range_step)
        lower = fundamental + lower_persistence * observed_premium
        upper = fundamental + upper_persistence * observed_premium
        lower, upper = min(lower, upper), max(lower, upper)
        lower = max(Decimal("0.01"), lower)
        point = max(Decimal("0.01"), point)
        upper = max(point, upper)

        sample_size = peer_summary.sample_size if peer_summary is not None else 0
        history_size = (
            len(effective_history.observations) if effective_history is not None else 0
        )
        if (
            history_size >= 8
            and sample_size >= policy.minimum_peer_sample
            and support_count >= 3
        ):
            confidence = MultipleConfidence.HIGH
        elif history_size >= 4 or sample_size >= policy.minimum_peer_sample:
            confidence = MultipleConfidence.MEDIUM
        else:
            confidence = MultipleConfidence.LOW
        if sample_size < policy.minimum_peer_sample:
            warnings.append(
                f"Only {sample_size} peer observations support {basis.value}; "
                f"policy requests at least {policy.minimum_peer_sample}"
            )

        return ResolvedMultiple(
            basis=basis,
            point_estimate=point,
            lower_bound=lower,
            upper_bound=upper,
            fundamental_anchor=fundamental,
            peer_anchor=peer_anchor,
            historical_anchor=historical_anchor,
            historical_percentile_25=(
                effective_history.percentile_25
                if effective_history is not None
                else None
            ),
            historical_percentile_75=(
                effective_history.percentile_75
                if effective_history is not None
                else None
            ),
            historical_volatility=(
                effective_history.volatility
                if effective_history is not None
                else None
            ),
            historical_trend=(
                effective_history.trend if effective_history is not None else None
            ),
            historical_sample_size=history_size,
            current_target_anchor=target_current,
            market_anchor=market_anchor,
            observed_premium=observed_premium,
            historical_peer_premium=historical_peer_premium,
            premium_history_sample_size=premium_history_sample_size,
            premium_mean_reversion_beta=premium_mean_reversion_beta,
            historical_persistence=historical_persistence,
            fundamental_support=fundamental_support,
            horizon_retention=horizon_retention,
            persistence_factor=persistence,
            sample_size=sample_size,
            confidence=confidence,
            methodology=(
                "fundamental anchor + (historical/peer market anchor - fundamental "
                "anchor) × historical persistence × economic support × horizon decay"
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _fundamental_anchor(basis, forecast, valuation):
        final = forecast.observations[-1]
        metric = {
            RelativeValuationBasis.EV_TO_EBITDA: (
                final.operating_income + final.depreciation_and_amortization
            ),
            RelativeValuationBasis.EV_TO_EBIT: final.operating_income,
            RelativeValuationBasis.EV_TO_REVENUE: final.revenue,
            RelativeValuationBasis.EV_TO_FCF: final.fcff,
        }[basis]
        if metric <= 0:
            raise ValueError("Fundamental multiple requires a positive terminal metric")
        return valuation.terminal_value.terminal_value / metric

    @staticmethod
    def _target_multiple(target, basis):
        multiple = next(
            (
                item
                for item in target.multiples
                if item.basis == basis
                and item.status == MultipleStatus.COMPUTED
                and item.value is not None
            ),
            None,
        )
        return multiple.value if multiple is not None else None

    @staticmethod
    def _historical_persistence(history, peer_histories, policy, warnings):
        premium_series = MultipleResolver._aligned_peer_premiums(
            history, peer_histories, policy
        )
        if len(premium_series) >= 4:
            premiums = [value for _, value in premium_series]
            beta = MultipleResolver._autoregressive_persistence(premiums)
            if beta is not None:
                if len(premiums) < 8:
                    warnings.append(
                        f"Premium mean reversion is estimated from only "
                        f"{len(premiums)} synchronized observations; treat the "
                        "persistence estimate as low precision"
                    )
                duration = min(Decimal(1), Decimal(len(premiums)) / Decimal(8))
                persistence = beta * (Decimal("0.5") + Decimal("0.5") * duration)
                return (
                    persistence,
                    median(premiums) * Decimal(100),
                    len(premiums),
                    beta,
                )
            warnings.append(
                "Historical peer-premium variation is too small to estimate "
                "mean reversion; target-multiple stability is used instead"
            )
        elif peer_histories:
            warnings.append(
                "Fewer than four synchronized target/peer observations are "
                "available; premium persistence uses target-multiple stability"
            )
        if history is None or len(history.observations) < 4:
            return policy.insufficient_history_persistence, None, len(premium_series), None
        values = sorted(item.value for item in history.observations)
        lower_percentile, upper_percentile = policy.winsorize_percentiles
        lower = HistoricalMultiplesService._percentile(
            values, lower_percentile / Decimal(100)
        )
        upper = HistoricalMultiplesService._percentile(
            values, upper_percentile / Decimal(100)
        )
        winsorized = [max(lower, min(upper, value)) for value in values]
        mean = sum(winsorized, Decimal(0)) / Decimal(len(winsorized))
        variance = sum(
            ((value - mean) ** 2 for value in winsorized), Decimal(0)
        ) / Decimal(len(winsorized))
        volatility = variance.sqrt() / mean if mean > 0 else Decimal(1)
        stability = max(Decimal(0), Decimal(1) - min(Decimal(1), volatility))
        duration = min(Decimal(1), Decimal(len(history.observations)) / Decimal(8))
        return stability * duration, None, len(premium_series), None

    @staticmethod
    def _aligned_peer_premiums(history, peer_histories, policy):
        if history is None or not peer_histories:
            return []
        aligned = []
        for target_observation in history.observations:
            peer_values = []
            for peer_history in peer_histories:
                candidates = [
                    item
                    for item in peer_history.observations
                    if item.observed_on <= target_observation.observed_on
                    and (target_observation.observed_on - item.observed_on).days <= 550
                ]
                if candidates:
                    peer_values.append(candidates[-1].value)
            required = min(2, policy.minimum_peer_sample)
            if len(peer_values) >= required:
                peer_median = median(peer_values)
                if peer_median > 0:
                    aligned.append(
                        (
                            target_observation.observed_on,
                            target_observation.value / peer_median - Decimal(1),
                        )
                    )
        return aligned

    @staticmethod
    def _autoregressive_persistence(values):
        """Estimate one-period AR(1) persistence with an intercept."""
        if len(values) < 4:
            return None
        lagged = values[:-1]
        current = values[1:]
        lagged_mean = sum(lagged, Decimal(0)) / Decimal(len(lagged))
        current_mean = sum(current, Decimal(0)) / Decimal(len(current))
        denominator = sum(
            ((value - lagged_mean) ** 2 for value in lagged), Decimal(0)
        )
        if denominator == 0:
            return None
        numerator = sum(
            (
                (lagged_value - lagged_mean) * (current_value - current_mean)
                for lagged_value, current_value in zip(lagged, current, strict=True)
            ),
            Decimal(0),
        )
        return max(Decimal(0), min(Decimal(1), numerator / denominator))

    @staticmethod
    def _fundamental_support(target, peers, warnings):
        scores = []
        target_fundamentals = target.fundamentals
        peer_fundamentals = [peer.fundamentals for peer in peers]

        def compare(
            target_value, peer_values, *, lower_is_better=False, allow_signed=False
        ):
            usable = [
                value
                for value in peer_values
                if value is not None and (allow_signed or value > 0)
            ]
            if (
                target_value is None
                or (not allow_signed and target_value <= 0)
                or not usable
            ):
                return
            anchor = median(usable)
            scale = max(abs(anchor), Decimal(1) if allow_signed else Decimal("0.01"))
            relative = (target_value - anchor) / scale
            if lower_is_better:
                relative = -relative
            scores.append(
                max(Decimal(0), min(Decimal(1), Decimal("0.75") + relative / 2))
            )

        target_margin = (
            target_fundamentals.ebitda / target_fundamentals.revenue
            if target_fundamentals.ebitda is not None
            and target_fundamentals.revenue is not None
            and target_fundamentals.revenue > 0
            else None
        )
        peer_margins = [
            item.ebitda / item.revenue
            if item.ebitda is not None and item.revenue is not None and item.revenue > 0
            else None
            for item in peer_fundamentals
        ]
        compare(target_margin, peer_margins)
        target_conversion = (
            target_fundamentals.free_cash_flow / target_fundamentals.ebitda
            if target_fundamentals.free_cash_flow is not None
            and target_fundamentals.ebitda is not None
            and target_fundamentals.ebitda > 0
            else None
        )
        peer_conversions = [
            item.free_cash_flow / item.ebitda
            if item.free_cash_flow is not None
            and item.ebitda is not None
            and item.ebitda > 0
            else None
            for item in peer_fundamentals
        ]
        compare(target_conversion, peer_conversions)
        compare(
            target_fundamentals.revenue_growth,
            [item.revenue_growth for item in peer_fundamentals],
            allow_signed=True,
        )
        target_leverage = (
            target_fundamentals.gross_debt / target_fundamentals.ebitda
            if target_fundamentals.gross_debt is not None
            and target_fundamentals.ebitda is not None
            and target_fundamentals.ebitda > 0
            else None
        )
        peer_leverage = [
            item.gross_debt / item.ebitda
            if item.gross_debt is not None
            and item.ebitda is not None
            and item.ebitda > 0
            else None
            for item in peer_fundamentals
        ]
        compare(target_leverage, peer_leverage, lower_is_better=True)
        target_capital_intensity = (
            target_fundamentals.capital_expenditures / target_fundamentals.revenue
            if target_fundamentals.capital_expenditures is not None
            and target_fundamentals.revenue is not None
            and target_fundamentals.revenue > 0
            else None
        )
        peer_capital_intensity = [
            item.capital_expenditures / item.revenue
            if item.capital_expenditures is not None
            and item.revenue is not None
            and item.revenue > 0
            else None
            for item in peer_fundamentals
        ]
        compare(
            target_capital_intensity,
            peer_capital_intensity,
            lower_is_better=True,
        )

        def roic(fundamentals):
            invested_capital = (
                (fundamentals.gross_debt or Decimal(0))
                + (fundamentals.book_equity or Decimal(0))
                - (fundamentals.cash_and_equivalents or Decimal(0))
            )
            if fundamentals.operating_income is None or invested_capital <= 0:
                return None
            return fundamentals.operating_income / invested_capital

        compare(
            roic(target_fundamentals),
            [roic(item) for item in peer_fundamentals],
        )
        if not scores:
            warnings.append(
                "Peer economics are insufficient to support a historical premium; "
                "premium persistence is capped at 50%"
            )
            return Decimal("0.5"), 0
        return sum(scores, Decimal(0)) / Decimal(len(scores)), len(scores)


class ComparableImpliedValuationService:
    """Convert a resolved forward multiple into an independent equity valuation."""

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
        cases = [
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
            lower_case=cases[0],
            point_case=cases[1],
            upper_case=cases[2],
            current_price=current_price,
            current_price_implied_multiple=current_price_implied_multiple,
            analyst_target_price=analyst_target_price,
            analyst_target_implied_multiple=analyst_target_implied_multiple,
            intrinsic_value_per_share=intrinsic_value_per_share,
            warnings=tuple(dict.fromkeys(warnings)),
        )

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
        present_value = value_per_share * PresentValueService.discount_factor(
            discount_rate, horizon_years
        )
        return ComparableImpliedValuationCase(
            label=label,
            multiple=multiple,
            implied_enterprise_value=enterprise_value,
            implied_equity_value=equity_value,
            implied_value_per_share=value_per_share,
            present_value_per_share=present_value,
        )


__all__ = [
    "ComparableImpliedValuationService",
    "HistoricalMultiplesService",
    "MultipleResolver",
]
