"""Provider-neutral discovery seam for operating evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from edgarito.schemas.operating import (
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)


@dataclass(frozen=True)
class OperatingForecastDiscoveryResult:
    segments: tuple[OperatingSegment, ...] = ()
    definitions: tuple[OperatingDriverDefinition, ...] = ()
    observations: tuple[OperatingDriverObservation, ...] = ()
    management_constraints: Any = ()
    historical_revenue: Mapping[Any, Any] | None = None


class OperatingForecastDiscovery(Protocol):
    def discover(self, *args: Any, **kwargs: Any) -> OperatingForecastDiscoveryResult:
        """Return normalized operating evidence for the integration seam."""


__all__ = ["OperatingForecastDiscovery", "OperatingForecastDiscoveryResult"]
