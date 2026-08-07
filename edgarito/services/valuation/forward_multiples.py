import datetime
import math
from decimal import Decimal
from statistics import median

from edgarito.services.financial_observation_availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting.fcff import FcffForecastService
from edgarito.services.forecasting.models import FcffForecastParameters
from edgarito.services.valuation.historical_multiples import HistoricalMultiplesService
from edgarito.services.valuation.models import (
    ComparableMultiplesReport,
    PeerMultipleSummary,
    RelativeValuationBasis,
)
from edgarito.services.valuation.multiple_resolver import MultipleResolver


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
                    as_of=valuation_date,
                    availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
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
