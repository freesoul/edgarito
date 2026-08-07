from decimal import Decimal
from statistics import median

from edgarito.services.valuation.models import (
    HistoricalMultipleObservation,
    HistoricalMultipleSummary,
    MultipleStatus,
    RelativeValuationBasis,
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
            if (
                multiple is not None
                and multiple.value is not None
                and multiple.value > 0
            ):
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
                    "the computed multiple was non-positive"
                    if multiple is not None
                    and multiple.value is not None
                    and multiple.value <= 0
                    else selected.reason
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
