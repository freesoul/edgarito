import datetime
import math
from dataclasses import dataclass
from decimal import Decimal
from statistics import median

from edgarito.services.forecasting.fcff import FcffForecastService
from edgarito.services.forecasting.models import FcffForecast, FcffForecastParameters
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
    PeerMultipleSummary,
    RelativeValuationBasis,
    ResolvedMultiple,
)
from edgarito.services.valuation.multiples import LtmMultiplesService


@dataclass(frozen=True)
class _PremiumForecast:
    long_run_premium: Decimal | None
    percentile_25: Decimal | None
    percentile_75: Decimal | None
    raw_phi: Decimal | None
    shrunk_phi: Decimal
    statistical_premium: Decimal
    sample_size: int
    history_weight: Decimal
    observation_interval_years: Decimal | None


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
        revenue_observations = [
            item
            for item in financials.observations
            if item.concept.value == "revenue"
            and item.granularity.value in {"quarterly", "annual"}
        ]
        dates = sorted(
            {self._ltm_service.availability_date(item) for item in revenue_observations}
        )
        observations_by_period = {}
        warnings = []
        if any(
            item.provider.casefold() == "yahoo" and item.filed is None
            for item in revenue_observations
        ):
            warnings.append(
                "Yahoo statements do not expose filing dates; historical snapshots "
                "use conservative 45-day quarterly and 90-day annual availability "
                "lags"
            )
        for observed_on in dates:
            try:
                snapshot = self._ltm_service.compute(
                    financials,
                    market_data,
                    as_of=observed_on,
                    point_in_time=True,
                )
            except ValueError as exc:
                warnings.append(
                    f"{observed_on.isoformat()}: historical {basis.value} "
                    f"snapshot failed: {exc}"
                )
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
                observations_by_period.setdefault(
                    snapshot.fundamentals.period_end,
                    HistoricalMultipleObservation(
                        observed_on=observed_on,
                        value=multiple.value,
                        fundamentals_period_end=snapshot.fundamentals.period_end,
                        price_date=snapshot.price_date,
                    ),
                )
            else:
                selected = next(
                    (item for item in snapshot.multiples if item.basis == basis), None
                )
                reason = (
                    selected.reason
                    if selected is not None and selected.reason
                    else "the requested multiple was not computed"
                )
                context = "; ".join(snapshot.warnings)
                warnings.append(
                    f"{observed_on.isoformat()}: historical {basis.value} "
                    f"unavailable: {reason}" + (f" ({context})" if context else "")
                )

        observations = sorted(
            observations_by_period.values(), key=lambda item: item.observed_on
        )

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


class ForwardPeerMultiplesService:
    """Build same-horizon peer multiples from each peer's generic forecast."""

    def __init__(self, forecast_service: FcffForecastService | None = None):
        self._forecast_service = forecast_service or FcffForecastService()

    def build(
        self,
        report: ComparableMultiplesReport,
        financials_by_ticker,
        basis: RelativeValuationBasis,
        valuation_date: datetime.date,
        horizon_years: Decimal,
    ) -> ComparableMultiplesReport:
        target_date = valuation_date + datetime.timedelta(
            days=int(horizon_years * Decimal(365))
        )
        values = []
        warnings = list(report.warnings)
        for peer in report.peers:
            financials = financials_by_ticker.get(peer.ticker)
            if financials is None or peer.enterprise_value is None:
                warnings.append(
                    f"{peer.ticker} forward multiple unavailable: financials or "
                    "current enterprise value is missing"
                )
                continue
            latest_annual_end = max(
                (
                    item.period_end
                    for item in financials.observations
                    if item.granularity.value == "annual"
                    and item.concept.value == "revenue"
                ),
                default=None,
            )
            if latest_annual_end is None:
                warnings.append(
                    f"{peer.ticker} forward multiple unavailable: no annual revenue"
                )
                continue
            forecast_years = max(
                1,
                math.ceil(max(1, (target_date - latest_annual_end).days) / 365),
            )
            try:
                forecast = self._forecast_service.forecast(
                    financials,
                    FcffForecastParameters(forecast_years=forecast_years),
                )
                metric = MultipleResolver._forecast_metric_at_date(
                    basis, forecast, target_date
                )
                if metric <= 0:
                    raise ValueError("forecast metric is zero or negative")
                values.append(peer.enterprise_value / metric)
            except ValueError as exc:
                warnings.append(
                    f"{peer.ticker} forward {basis.value} unavailable: {exc}"
                )
        forward_summaries = [
            item for item in report.forward_summaries if item.basis != basis
        ]
        if values:
            ordered = sorted(values)
            forward_summaries.append(
                PeerMultipleSummary(
                    basis=basis,
                    median=median(ordered),
                    minimum=ordered[0],
                    maximum=ordered[-1],
                    percentile_25=HistoricalMultiplesService._percentile(
                        ordered, Decimal("0.25")
                    ),
                    percentile_75=HistoricalMultiplesService._percentile(
                        ordered, Decimal("0.75")
                    ),
                    sample_size=len(ordered),
                )
            )
        return report.model_copy(
            update={
                "forward_summaries": forward_summaries,
                "warnings": list(dict.fromkeys(warnings)),
            }
        )


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
            basis,
            target_forecast,
            intrinsic_valuation,
            horizon_years,
        )
        effective_history = target_history if policy.use_target_history else None
        ltm_target_multiple = self._target_multiple(target, basis)
        ltm_peer_summary = next(
            (item for item in peer_report.summaries if item.basis == basis), None
        )
        forward_peer_summary = next(
            (item for item in peer_report.forward_summaries if item.basis == basis),
            None,
        )
        usable_forward_peer_summary = (
            forward_peer_summary
            if forward_peer_summary is not None
            and forward_peer_summary.sample_size >= policy.minimum_peer_sample
            else None
        )
        peer_summary = usable_forward_peer_summary or ltm_peer_summary
        peer_anchor = peer_summary.median if peer_summary is not None else None
        historical_anchor = (
            effective_history.median if effective_history is not None else None
        )

        warnings = list(
            effective_history.warnings if effective_history is not None else ()
        )
        if forward_peer_summary is not None and usable_forward_peer_summary is None:
            warnings.append(
                f"Only {forward_peer_summary.sample_size} reliable forward peer "
                f"multiples are available; policy requires "
                f"{policy.minimum_peer_sample}, so the resolver falls back to LTM"
            )
        if policy.use_peer_median and peer_anchor is not None:
            market_anchor = peer_anchor
            if usable_forward_peer_summary is not None:
                peer_anchor_source = "forward"
                warnings.append(
                    "Historical premium observations are reconstructed from LTM "
                    "multiples; they are used as relative evidence against the "
                    "forward baseline, not as forward consensus observations"
                )
                target_current = self._target_forward_multiple(
                    target,
                    basis,
                    target_forecast,
                    intrinsic_valuation,
                    horizon_years,
                )
                if target_current is None:
                    target_current = ltm_target_multiple
                    warnings.append(
                        "Forward peer multiples are available, but the target forward "
                        "market multiple could not be reconstructed; current target "
                        "LTM is used with reduced comparability"
                    )
            else:
                peer_anchor_source = "current_ltm_fallback"
                target_current = ltm_target_multiple
                warnings.append(
                    "Peer baseline uses current LTM multiples because reliable peer "
                    "forward forecasts are not available"
                )
        else:
            market_anchor = fundamental
            peer_anchor_source = "dcf_fallback"
            target_current = ltm_target_multiple
            warnings.append(
                "No usable peer baseline is available; the DCF-implied forward "
                "multiple is used as the base case"
            )
        observed_premium = (
            target_current / market_anchor - Decimal(1)
            if target_current is not None and market_anchor > 0
            else None
        )
        if observed_premium is None:
            observed_premium = Decimal(0)
            warnings.append(
                "No current or historical target premium is available; the "
                "resolved multiple remains at the base forward multiple"
            )
        fundamental_premium = fundamental / market_anchor - Decimal(1)
        premium_forecast = self._statistical_premium_forecast(
            effective_history,
            peer_histories,
            observed_premium,
            horizon_years,
            policy,
            warnings,
        )
        fundamental_support, _support_count = self._fundamental_support(
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
        quality_weight = (
            fundamental_support
            if premium_forecast.statistical_premium > fundamental_premium
            else Decimal(1)
        )
        persistence = min(
            Decimal(1),
            premium_forecast.history_weight * quality_weight * horizon_retention,
        )
        resolved_premium = fundamental_premium + persistence * (
            premium_forecast.statistical_premium - fundamental_premium
        )
        point = market_anchor * (Decimal(1) + resolved_premium)
        lower, upper = self._evidence_range(
            fundamental_anchor=fundamental,
            point=point,
            base_anchor=market_anchor,
            peer_summary=peer_summary,
            premium_forecast=premium_forecast,
        )

        sample_size = peer_summary.sample_size if peer_summary is not None else 0
        history_size = (
            len(effective_history.observations) if effective_history is not None else 0
        )
        peer_confidence = self._sample_confidence(
            sample_size, policy.minimum_peer_sample
        )
        target_history_confidence = self._history_confidence(history_size)
        premium_persistence_confidence = self._persistence_confidence(
            premium_forecast.sample_size, premium_forecast.raw_phi
        )
        if peer_anchor_source == "current_ltm_fallback":
            peer_confidence = self._lower_confidence(peer_confidence)
        confidence = min(
            peer_confidence,
            target_history_confidence,
            premium_persistence_confidence,
            key=self._confidence_rank,
        )
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
            fundamental_premium=fundamental_premium,
            peer_anchor=peer_anchor,
            peer_anchor_source=peer_anchor_source,
            peer_anchor_percentile_25=(
                peer_summary.percentile_25 if peer_summary is not None else None
            ),
            peer_anchor_percentile_75=(
                peer_summary.percentile_75 if peer_summary is not None else None
            ),
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
                effective_history.volatility if effective_history is not None else None
            ),
            historical_trend=(
                effective_history.trend if effective_history is not None else None
            ),
            historical_sample_size=history_size,
            current_target_anchor=target_current,
            market_anchor=market_anchor,
            observed_premium=observed_premium,
            resolved_premium=resolved_premium,
            historical_peer_premium=premium_forecast.long_run_premium,
            historical_peer_premium_25=premium_forecast.percentile_25,
            historical_peer_premium_75=premium_forecast.percentile_75,
            premium_history_sample_size=premium_forecast.sample_size,
            premium_mean_reversion_beta=premium_forecast.raw_phi,
            shrunk_premium_persistence=premium_forecast.shrunk_phi,
            statistical_premium=premium_forecast.statistical_premium,
            premium_history_weight=premium_forecast.history_weight,
            premium_observation_interval_years=(
                premium_forecast.observation_interval_years
            ),
            historical_persistence=premium_forecast.shrunk_phi,
            fundamental_support=fundamental_support,
            horizon_retention=horizon_retention,
            persistence_factor=persistence,
            sample_size=sample_size,
            peer_confidence=peer_confidence,
            target_history_confidence=target_history_confidence,
            premium_persistence_confidence=premium_persistence_confidence,
            confidence=confidence,
            methodology=(
                "peer/base multiple × (1 + DCF-implied premium + evidence weight × "
                "(statistically forecast premium - DCF-implied premium))"
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _fundamental_anchor(basis, forecast, valuation, horizon_years):
        target_date = valuation.valuation_date + datetime.timedelta(
            days=int(horizon_years * Decimal(365))
        )
        metric = MultipleResolver._forecast_metric_at_date(basis, forecast, target_date)
        if metric <= 0:
            raise ValueError("Fundamental multiple requires a positive forward metric")
        future_cash_flows = sum(
            (
                item.amount
                * PresentValueService.discount_factor(
                    valuation.parameters.wacc,
                    item.period - horizon_years,
                )
                for item in valuation.explicit_forecast_present_value.cash_flows
                if item.period > horizon_years
            ),
            Decimal(0),
        )
        terminal = valuation.terminal_present_value
        if terminal.period < horizon_years:
            raise ValueError(
                "Relative valuation horizon extends beyond the DCF terminal date"
            )
        enterprise_value_at_horizon = future_cash_flows + terminal.amount * (
            PresentValueService.discount_factor(
                valuation.parameters.wacc,
                terminal.period - horizon_years,
            )
        )
        return enterprise_value_at_horizon / metric

    @staticmethod
    def _forecast_metric_at_date(basis, forecast, target_date):
        target = min(
            forecast.observations,
            key=lambda item: abs((item.period_end - target_date).days),
        )
        return {
            RelativeValuationBasis.EV_TO_EBITDA: (
                target.operating_income + target.depreciation_and_amortization
            ),
            RelativeValuationBasis.EV_TO_EBIT: target.operating_income,
            RelativeValuationBasis.EV_TO_REVENUE: target.revenue,
            RelativeValuationBasis.EV_TO_FCF: target.fcff,
        }[basis]

    @staticmethod
    def _target_forward_multiple(target, basis, forecast, valuation, horizon_years):
        if target.enterprise_value is None or target.enterprise_value <= 0:
            return None
        target_date = valuation.valuation_date + datetime.timedelta(
            days=int(horizon_years * Decimal(365))
        )
        metric = MultipleResolver._forecast_metric_at_date(basis, forecast, target_date)
        return target.enterprise_value / metric if metric > 0 else None

    @staticmethod
    def _confidence_rank(confidence):
        return {
            MultipleConfidence.LOW: 0,
            MultipleConfidence.MEDIUM: 1,
            MultipleConfidence.HIGH: 2,
        }[confidence]

    @staticmethod
    def _sample_confidence(sample_size, requested):
        if sample_size >= max(8, requested * 2):
            return MultipleConfidence.HIGH
        if sample_size >= requested:
            return MultipleConfidence.MEDIUM
        return MultipleConfidence.LOW

    @staticmethod
    def _lower_confidence(confidence):
        return {
            MultipleConfidence.HIGH: MultipleConfidence.MEDIUM,
            MultipleConfidence.MEDIUM: MultipleConfidence.LOW,
            MultipleConfidence.LOW: MultipleConfidence.LOW,
        }[confidence]

    @staticmethod
    def _history_confidence(sample_size):
        if sample_size >= 8:
            return MultipleConfidence.HIGH
        if sample_size >= 4:
            return MultipleConfidence.MEDIUM
        return MultipleConfidence.LOW

    @staticmethod
    def _persistence_confidence(sample_size, beta):
        if beta is None or sample_size < 8:
            return MultipleConfidence.LOW
        if sample_size >= 16:
            return MultipleConfidence.HIGH
        return MultipleConfidence.MEDIUM

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
    def _statistical_premium_forecast(
        history,
        peer_histories,
        current_premium,
        horizon_years,
        policy,
        warnings,
    ):
        premium_series = MultipleResolver._aligned_peer_premiums(
            history, peer_histories, policy
        )
        premiums = [value for _, value in premium_series]
        sample_size = len(premiums)
        prior_phi = (
            policy.insufficient_history_persistence
            if policy.insufficient_history_persistence is not None
            else policy.premium_persistence_prior
        )
        if not premiums:
            warnings.append(
                "No synchronized peer history is available; the statistical premium "
                "has zero blending weight and the resolver falls back to the "
                "DCF-implied premium"
            )
            return _PremiumForecast(
                long_run_premium=None,
                percentile_25=None,
                percentile_75=None,
                raw_phi=None,
                shrunk_phi=prior_phi,
                statistical_premium=current_premium,
                sample_size=0,
                history_weight=Decimal(0),
                observation_interval_years=None,
            )

        sorted_premiums = sorted(premiums)
        long_run_premium = median(sorted_premiums)
        percentile_25 = HistoricalMultiplesService._percentile(
            sorted_premiums, Decimal("0.25")
        )
        percentile_75 = HistoricalMultiplesService._percentile(
            sorted_premiums, Decimal("0.75")
        )
        full_sample = Decimal(policy.full_premium_history_observations)
        history_weight = min(Decimal(1), Decimal(sample_size) / full_sample)
        raw_phi = None
        if len(premium_series) >= 4:
            lower_percentile, upper_percentile = policy.winsorize_percentiles
            winsor_lower = HistoricalMultiplesService._percentile(
                sorted_premiums, lower_percentile / Decimal(100)
            )
            winsor_upper = HistoricalMultiplesService._percentile(
                sorted_premiums, upper_percentile / Decimal(100)
            )
            winsorized = [
                max(winsor_lower, min(winsor_upper, value)) for value in premiums
            ]
            raw_phi = MultipleResolver._autoregressive_persistence(winsorized)
        if sample_size < 8:
            warnings.append(
                f"Premium mean reversion is estimated from only {sample_size} "
                "synchronized observations; raw AR(1) phi is low precision and "
                "is shrunk toward the configured prior"
            )
        if raw_phi is None:
            warnings.append(
                "Historical peer-premium variation or sample size is insufficient "
                "to estimate AR(1) phi; the configured prior is used"
            )
        phi_evidence_weight = min(Decimal(1), Decimal(sample_size) / full_sample)
        estimated_phi = raw_phi if raw_phi is not None else prior_phi
        shrunk_phi = max(
            Decimal(0),
            min(
                Decimal(1),
                phi_evidence_weight * estimated_phi
                + (Decimal(1) - phi_evidence_weight) * prior_phi,
            ),
        )
        interval_years = MultipleResolver._median_observation_interval(premium_series)
        horizon_periods = (
            horizon_years / interval_years
            if interval_years is not None
            else horizon_years
        )
        if policy.forecast_premium_mean_reversion:
            deviation_retention = Decimal(
                str(float(shrunk_phi) ** float(horizon_periods))
            )
            statistical_premium = long_run_premium + deviation_retention * (
                current_premium - long_run_premium
            )
        else:
            statistical_premium = current_premium
        return _PremiumForecast(
            long_run_premium=long_run_premium,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            raw_phi=raw_phi,
            shrunk_phi=shrunk_phi,
            statistical_premium=statistical_premium,
            sample_size=sample_size,
            history_weight=history_weight,
            observation_interval_years=interval_years,
        )

    @staticmethod
    def _median_observation_interval(premium_series):
        if len(premium_series) < 2:
            return None
        gaps = sorted(
            Decimal((current[0] - previous[0]).days) / Decimal(365)
            for previous, current in zip(
                premium_series, premium_series[1:], strict=False
            )
            if current[0] > previous[0]
        )
        return median(gaps) if gaps else None

    @staticmethod
    def _evidence_range(
        *,
        fundamental_anchor,
        point,
        base_anchor,
        peer_summary,
        premium_forecast,
    ):
        fundamental_premium = fundamental_anchor / base_anchor - Decimal(1)
        resolved_premium = point / base_anchor - Decimal(1)
        premium_candidates = [fundamental_premium, resolved_premium]
        if premium_forecast.sample_size >= 4:
            premium_candidates.append(premium_forecast.statistical_premium)
            if premium_forecast.percentile_25 is not None:
                premium_candidates.append(premium_forecast.percentile_25)
            if premium_forecast.percentile_75 is not None:
                premium_candidates.append(premium_forecast.percentile_75)
        base_low = (
            peer_summary.percentile_25
            if peer_summary is not None
            and peer_summary.sample_size >= 4
            and peer_summary.percentile_25 is not None
            else base_anchor
        )
        base_high = (
            peer_summary.percentile_75
            if peer_summary is not None
            and peer_summary.sample_size >= 4
            and peer_summary.percentile_75 is not None
            else base_anchor
        )
        lower = min(
            fundamental_anchor,
            point,
            base_anchor * (Decimal(1) + min(premium_candidates)),
            base_low * (Decimal(1) + resolved_premium),
        )
        upper = max(
            fundamental_anchor,
            point,
            base_anchor * (Decimal(1) + max(premium_candidates)),
            base_high * (Decimal(1) + resolved_premium),
        )
        return max(Decimal("0.01"), lower), max(point, upper)

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
                    and (target_observation.observed_on - item.observed_on).days <= 120
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
        denominator = sum(((value - lagged_mean) ** 2 for value in lagged), Decimal(0))
        if denominator == 0:
            return None
        numerator = sum(
            (
                (lagged_value - lagged_mean) * (current_value - current_mean)
                for lagged_value, current_value in zip(lagged, current, strict=True)
            ),
            Decimal(0),
        )
        return max(Decimal(-1), min(Decimal(1), numerator / denominator))

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
            scale = max(
                abs(target_value) + abs(anchor),
                Decimal(1) if allow_signed else Decimal("0.01"),
            )
            relative = (target_value - anchor) / scale
            if lower_is_better:
                relative = -relative
            scores.append(max(Decimal(0), min(Decimal(1), Decimal("0.5") + relative)))

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
            (
                target_fundamentals.gross_debt
                - (target_fundamentals.cash_and_equivalents or Decimal(0))
            )
            / target_fundamentals.ebitda
            if target_fundamentals.gross_debt is not None
            and target_fundamentals.ebitda is not None
            and target_fundamentals.ebitda > 0
            else None
        )
        peer_leverage = [
            (item.gross_debt - (item.cash_and_equivalents or Decimal(0))) / item.ebitda
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
                "Peer economics are insufficient for cross-sectional premium "
                "support; no statistical premium above the DCF anchor is retained"
            )
            return Decimal(0), 0
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
    "ForwardPeerMultiplesService",
    "HistoricalMultiplesService",
    "MultipleResolver",
]
