import datetime
import math
from dataclasses import dataclass
from decimal import Decimal
from statistics import median

from edgarito.services.forecasting.models import FcffForecast
from edgarito.services.valuation.discounting import PresentValueService
from edgarito.services.valuation.historical_multiples import HistoricalMultiplesService
from edgarito.services.valuation.models import (
    CompanyTradingMultiples,
    ComparableMultiplesReport,
    FcffDcfResult,
    HistoricalMultipleSummary,
    MultipleConfidence,
    MultipleStatus,
    RelativeValuationBasis,
    ResolvedMultiple,
)


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


class MultipleResolver:
    """Resolve a market multiple as fundamental value plus persistent premium."""

    _FORWARD_HISTORY_SUPPORT_CAP = Decimal("0.25")
    _FORWARD_EVIDENCE_WEIGHT_CAP = Decimal("0.75")
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
        forward_evidence_available = policy.use_peer_median and (
            usable_forward_peer_summary is not None
        )
        target_forward_multiple = (
            self._target_forward_multiple(
                target,
                basis,
                target_forecast,
                intrinsic_valuation,
                horizon_years,
            )
            if forward_evidence_available
            else None
        )
        synchronized_forward_premium = (
            target_forward_multiple / usable_forward_peer_summary.median - Decimal(1)
            if target_forward_multiple is not None
            and usable_forward_peer_summary is not None
            and usable_forward_peer_summary.median > 0
            else None
        )
        peer_summary = (
            usable_forward_peer_summary or ltm_peer_summary
            if policy.use_peer_median
            else None
        )
        peer_anchor = peer_summary.median if peer_summary is not None else None
        historical_anchor, historical_percentile_25, historical_percentile_75 = (
            self._historical_multiple_statistics(effective_history)
        )

        warnings = list(
            effective_history.warnings if effective_history is not None else ()
        )
        if (
            policy.use_peer_median
            and forward_peer_summary is not None
            and usable_forward_peer_summary is None
        ):
            warnings.append(
                f"Only {forward_peer_summary.sample_size} reliable forward peer "
                f"multiples are available; policy requires "
                f"{policy.minimum_peer_sample}, so the resolver falls back to LTM"
            )
        if policy.use_peer_median and peer_anchor is not None:
            market_anchor = peer_anchor
            if usable_forward_peer_summary is not None:
                peer_anchor_source = "forward"
                target_current = target_forward_multiple
                if target_current is None:
                    target_current = ltm_target_multiple
                    warnings.append(
                        "Forward peer multiples are available, but the target forward "
                        "market multiple could not be reconstructed; current target "
                        "LTM is used with reduced comparability"
                    )
                elif synchronized_forward_premium is not None:
                    warnings.append(
                        "Synchronized forward target/peer premium at the requested "
                        "horizon is the primary premium evidence; historical premium "
                        "observations are reconstructed from LTM multiples and remain "
                        "supporting evidence for uncertainty and audit only. LTM "
                        "contributions to the forward "
                        "evidence range are capped at ±25 percentage points around "
                        "the synchronized forward premium"
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
        historical_current_premium = observed_premium
        if synchronized_forward_premium is not None:
            ltm_peer_anchor = (
                ltm_peer_summary.median
                if ltm_peer_summary is not None and ltm_peer_summary.median > 0
                else None
            )
            if ltm_target_multiple is not None and ltm_peer_anchor is not None:
                historical_current_premium = (
                    ltm_target_multiple / ltm_peer_anchor - Decimal(1)
                )
        premium_forecast = self._statistical_premium_forecast(
            effective_history,
            peer_histories,
            historical_current_premium,
            horizon_years,
            policy,
            warnings,
            forward_evidence_active=synchronized_forward_premium is not None,
        )
        primary_premium = (
            synchronized_forward_premium
            if synchronized_forward_premium is not None
            else premium_forecast.statistical_premium
        )
        premium_evidence_source = (
            "forward_synchronized"
            if synchronized_forward_premium is not None
            else "ltm_history"
            if premium_forecast.sample_size
            else "current_ltm"
            if peer_anchor_source == "current_ltm_fallback"
            else "dcf_fallback"
        )
        fundamental_support, _support_count = self._fundamental_support(
            target, peer_report.peers, warnings
        )
        forward_evidence_weight = (
            min(
                MultipleResolver._FORWARD_EVIDENCE_WEIGHT_CAP,
                # A current forward market multiple is evidence, not a target-date
                # valuation; retain a DCF leg even when the peer sample is large.
                Decimal(usable_forward_peer_summary.sample_size)
                / Decimal(max(8, policy.minimum_peer_sample * 2)),
            )
            if synchronized_forward_premium is not None
            and usable_forward_peer_summary is not None
            else Decimal(0)
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
            fundamental_support if primary_premium > fundamental_premium else Decimal(1)
        )
        evidence_weight = (
            forward_evidence_weight
            if synchronized_forward_premium is not None
            else premium_forecast.history_weight
        )
        persistence = min(
            Decimal(1), evidence_weight * quality_weight * horizon_retention
        )
        if synchronized_forward_premium is not None and persistence == 0:
            warnings.append(
                "Synchronized forward premium is available but fundamental support "
                "is insufficient; the DCF-implied premium remains the resolved "
                "anchor"
            )
        resolved_premium = fundamental_premium + persistence * (
            primary_premium - fundamental_premium
        )
        point = market_anchor * (Decimal(1) + resolved_premium)
        lower, upper = self._evidence_range(
            fundamental_anchor=fundamental,
            point=point,
            base_anchor=market_anchor,
            peer_summary=peer_summary,
            premium_forecast=premium_forecast,
            primary_premium=synchronized_forward_premium,
            minimum_premium_history_observations=(
                policy.minimum_premium_history_observations
            ),
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
        if synchronized_forward_premium is not None:
            confidence = (
                MultipleConfidence.LOW
                if persistence == 0
                else min(
                    peer_confidence,
                    MultipleConfidence.MEDIUM,
                    key=self._confidence_rank,
                )
            )
        else:
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
            historical_percentile_25=historical_percentile_25,
            historical_percentile_75=historical_percentile_75,
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
            forward_synchronized_premium=synchronized_forward_premium,
            forward_evidence_weight=forward_evidence_weight,
            premium_evidence_source=premium_evidence_source,
            historical_peer_premium=premium_forecast.long_run_premium,
            historical_peer_premium_25=premium_forecast.percentile_25,
            historical_peer_premium_75=premium_forecast.percentile_75,
            premium_history_sample_size=premium_forecast.sample_size,
            premium_mean_reversion_beta=premium_forecast.raw_phi,
            shrunk_premium_persistence=premium_forecast.shrunk_phi,
            statistical_premium=(
                premium_forecast.statistical_premium
                if premium_forecast.sample_size or synchronized_forward_premium is None
                else None
            ),
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
                (
                    "Composite diagnostic: peer forward multiple × (1 + DCF-implied premium + forward "
                    "evidence weight × (synchronized forward target/peer premium - "
                    "DCF-implied premium)); forward evidence weight is based on the "
                    "forward peer sample and independent of LTM history, which is "
                    "supporting evidence only"
                )
                if synchronized_forward_premium is not None
                else "Composite diagnostic: peer/base multiple × (1 + DCF-implied premium + evidence weight × "
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
    def _historical_multiple_statistics(history):
        """Return historical median and percentile bounds, including raw summaries.

        ``HistoricalMultiplesService`` populates these summary fields, but callers
        can also provide a lightweight ``HistoricalMultipleSummary`` containing
        only observations.  Supporting both forms keeps the implied-value layer
        independent from the DCF while preserving the existing resolver API.
        """

        if history is None:
            return None, None, None
        values = sorted(item.value for item in history.observations if item.value > 0)
        if not values:
            return history.median, history.percentile_25, history.percentile_75
        anchor = history.median or median(values)
        percentile_25 = history.percentile_25 or HistoricalMultiplesService._percentile(
            values, Decimal("0.25")
        )
        percentile_75 = history.percentile_75 or HistoricalMultiplesService._percentile(
            values, Decimal("0.75")
        )
        return anchor, percentile_25, percentile_75

    @staticmethod
    def _statistical_premium_forecast(
        history,
        peer_histories,
        current_premium,
        horizon_years,
        policy,
        warnings,
        forward_evidence_active=False,
    ):
        premium_series = MultipleResolver._aligned_peer_premiums(
            history, peer_histories, policy
        )
        premiums = [value for _, value in premium_series]
        sample_size = len(premiums)
        minimum_history = policy.minimum_premium_history_observations
        prior_phi = (
            policy.insufficient_history_persistence
            if policy.insufficient_history_persistence is not None
            else policy.premium_persistence_prior
        )
        if not premiums:
            warnings.append(
                "Insufficient synchronized premium history for AR(1) or "
                "premium-persistence blending: 0 observations are available and "
                f"the configured minimum is {minimum_history}; blending is disabled"
            )
            if forward_evidence_active:
                warnings.append(
                    "No synchronized LTM peer history is available; the forward "
                    "synchronized premium remains primary and no LTM persistence "
                    "support is applied"
                )
            else:
                warnings.append(
                    "No synchronized peer history is available; the statistical "
                    "premium has zero blending weight and the resolver falls back "
                    "to the DCF-implied premium"
                )
            return _PremiumForecast(
                long_run_premium=None,
                percentile_25=None,
                percentile_75=None,
                raw_phi=None,
                shrunk_phi=Decimal(0),
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
        if sample_size < minimum_history:
            warnings.append(
                "Insufficient synchronized premium history for AR(1) or "
                "premium-persistence blending: "
                f"{sample_size} observations are available and the configured "
                f"minimum is {minimum_history}; blending is disabled"
            )
            return _PremiumForecast(
                long_run_premium=long_run_premium,
                percentile_25=percentile_25,
                percentile_75=percentile_75,
                raw_phi=None,
                shrunk_phi=Decimal(0),
                statistical_premium=long_run_premium,
                sample_size=sample_size,
                history_weight=Decimal(0),
                observation_interval_years=MultipleResolver._median_observation_interval(
                    premium_series
                ),
            )

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
        primary_premium=None,
        minimum_premium_history_observations=8,
    ):
        historical_evidence_active = (
            premium_forecast.sample_size >= minimum_premium_history_observations
        )
        fundamental_premium = fundamental_anchor / base_anchor - Decimal(1)
        resolved_premium = point / base_anchor - Decimal(1)
        premium_candidates = [fundamental_premium, resolved_premium]
        if primary_premium is None:
            if historical_evidence_active:
                premium_candidates.append(premium_forecast.statistical_premium)
                if premium_forecast.percentile_25 is not None:
                    premium_candidates.append(premium_forecast.percentile_25)
                if premium_forecast.percentile_75 is not None:
                    premium_candidates.append(premium_forecast.percentile_75)
        elif historical_evidence_active:
            for supporting_premium in (
                premium_forecast.statistical_premium,
                premium_forecast.percentile_25,
                premium_forecast.percentile_75,
            ):
                if supporting_premium is not None:
                    premium_candidates.append(
                        MultipleResolver._bounded_forward_history_premium(
                            supporting_premium, primary_premium
                        )
                    )
        if primary_premium is not None and primary_premium not in premium_candidates:
            premium_candidates.append(primary_premium)
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
    def _bounded_forward_history_premium(
        supporting_premium: Decimal, forward_premium: Decimal
    ) -> Decimal:
        delta = max(
            -MultipleResolver._FORWARD_HISTORY_SUPPORT_CAP,
            min(
                MultipleResolver._FORWARD_HISTORY_SUPPORT_CAP,
                supporting_premium - forward_premium,
            ),
        )
        return forward_premium + delta

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
