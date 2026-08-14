"""Compatibility facade for the forward-estimate resolver."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edgarito.services.forecasting.forward_estimates import (
        ForwardEstimateResolver,
        ForwardEstimateService,
        ForwardRevenueConsensusService,
        ForwardRevenueEstimateResolver,
        ForwardRevenueEstimateService,
    )

__all__ = [
    "ForwardEstimateResolver",
    "ForwardEstimateService",
    "ForwardRevenueConsensusService",
    "ForwardRevenueEstimateResolver",
    "ForwardRevenueEstimateService",
]


_LAZY_EXPORTS = {
    name: ("edgarito.services.forecasting.forward_estimates", name) for name in __all__
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
