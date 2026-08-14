from importlib import import_module

__all__ = [
    "AdaptiveMultistageFcffForecastService",
    "AdaptiveMultistagePlan",
    "ForwardEstimateResolver",
    "ForwardEstimateService",
    "ForwardRevenueConsensusService",
    "ForwardRevenueEstimateResolver",
    "ForwardRevenueEstimateService",
    "ForwardGrowthEvidence",
    "ForwardGrowthOutlook",
    "FcffForecast",
    "FcffForecastDcfStub",
    "FcffForecastDriver",
    "FcffForecastObservation",
    "FcffForecastParameters",
    "FcffForecastYtdAnchor",
    "FcffForecastService",
    "ForecastAssumptionSource",
    "ForecastValue",
    "ForecastSeedType",
    "FreeCashFlowForecast",
    "FreeCashFlowForecastObservation",
    "FreeCashFlowForecastParameters",
    "MonetaryForecastConstraint",
    "FreeCashFlowForecastService",
    "SimplifiedFcfForecast",
    "SimplifiedFcfForecastObservation",
    "SimplifiedFcfForecastParameters",
    "SimplifiedFcfForecastService",
]


_LAZY_EXPORTS = {
    name: ("edgarito.services.forecasting.models", name)
    for name in {
        "AdaptiveMultistagePlan",
        "FcffForecast",
        "FcffForecastDcfStub",
        "FcffForecastDriver",
        "FcffForecastObservation",
        "FcffForecastParameters",
        "FcffForecastYtdAnchor",
        "ForecastAssumptionSource",
        "ForecastSeedType",
        "ForecastValue",
        "ForwardGrowthEvidence",
        "ForwardGrowthOutlook",
        "FreeCashFlowForecast",
        "FreeCashFlowForecastObservation",
        "FreeCashFlowForecastParameters",
        "MonetaryForecastConstraint",
        "SimplifiedFcfForecast",
        "SimplifiedFcfForecastObservation",
        "SimplifiedFcfForecastParameters",
    }
}
_LAZY_EXPORTS.update(
    {
        "FcffForecastService": (
            "edgarito.services.forecasting.fcff",
            "FcffForecastService",
        ),
        "FreeCashFlowForecastService": (
            "edgarito.services.forecasting.fcff",
            "FreeCashFlowForecastService",
        ),
        "SimplifiedFcfForecastService": (
            "edgarito.services.forecasting.free_cash_flow",
            "SimplifiedFcfForecastService",
        ),
        "AdaptiveMultistageFcffForecastService": (
            "edgarito.services.forecasting.multistage",
            "AdaptiveMultistageFcffForecastService",
        ),
    }
)
_LAZY_EXPORTS.update(
    {
        name: ("edgarito.services.forecasting.forward_estimates", name)
        for name in {
            "ForwardEstimateResolver",
            "ForwardEstimateService",
            "ForwardRevenueConsensusService",
            "ForwardRevenueEstimateResolver",
            "ForwardRevenueEstimateService",
        }
    }
)


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
