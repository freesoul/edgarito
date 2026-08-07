import datetime
from enum import Enum

from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import FinancialObservation


class ObservationAvailabilityMode(str, Enum):
    """The evidence standard used to decide whether an observation is known."""

    CURRENT_SNAPSHOT = "current_snapshot"
    POINT_IN_TIME = "point_in_time"


class FinancialObservationAvailabilityService:
    """Apply one provider-neutral observation-availability policy.

    A current snapshot proves only what was present when that snapshot was
    retrieved. Point-in-time reconstruction instead requires an actual filing
    date or a conservative provider estimate when publication metadata is absent.
    """

    _YAHOO_QUARTERLY_LAG_DAYS = 45
    _YAHOO_ANNUAL_LAG_DAYS = 90

    def available_on(
        self,
        observation: FinancialObservation,
        *,
        mode: ObservationAvailabilityMode,
        snapshot_retrieved_at: datetime.datetime | None = None,
    ) -> datetime.date:
        mode = ObservationAvailabilityMode(mode)
        if observation.filed is not None:
            return observation.filed
        if mode == ObservationAvailabilityMode.POINT_IN_TIME:
            if observation.provider.casefold() == "yahoo":
                lag_days = (
                    self._YAHOO_ANNUAL_LAG_DAYS
                    if observation.granularity == Granularity.ANNUAL
                    else self._YAHOO_QUARTERLY_LAG_DAYS
                )
                return observation.period_end + datetime.timedelta(days=lag_days)
            return observation.period_end
        if snapshot_retrieved_at is not None:
            return max(observation.period_end, snapshot_retrieved_at.date())
        return observation.period_end

    def is_available(
        self,
        observation: FinancialObservation,
        *,
        as_of: datetime.date,
        mode: ObservationAvailabilityMode,
        snapshot_retrieved_at: datetime.datetime | None = None,
    ) -> bool:
        # A provider snapshot cannot make a future reporting period current.
        if observation.period_end > as_of:
            return False
        return (
            self.available_on(
                observation,
                mode=mode,
                snapshot_retrieved_at=snapshot_retrieved_at,
            )
            <= as_of
        )


__all__ = [
    "FinancialObservationAvailabilityService",
    "ObservationAvailabilityMode",
]
