from importlib import import_module

__all__ = [
    "AdaptiveMultistageFcffForecastService",
    "DriverBasedForecastIncompleteError",
    "FcffForecastOrchestrationService",
    "FcffForecastPlanService",
    "ForwardRevenueEstimateService",
    "FcffForecastService",
    "ForecastOrchestrationResult",
    "ForecastOrchestrationService",
    "ForecastPlanService",
    "IncompleteFcffForecastMethodError",
    "SimplifiedFcfForecastService",
]


_LAZY_EXPORTS = {
    "FcffForecastService": (
        "edgarito.services.forecasting._fcff.service",
        "FcffForecastService",
    ),
    "SimplifiedFcfForecastService": (
        "edgarito.services.forecasting.free_cash_flow",
        "SimplifiedFcfForecastService",
    ),
    "AdaptiveMultistageFcffForecastService": (
        "edgarito.services.forecasting.multistage",
        "AdaptiveMultistageFcffForecastService",
    ),
    "ForwardRevenueEstimateService": (
        "edgarito.services.forecasting.forward_estimates",
        "ForwardRevenueEstimateService",
    ),
    "FcffForecastPlanService": (
        "edgarito.services.forecasting.plan",
        "FcffForecastPlanService",
    ),
    "ForecastPlanService": (
        "edgarito.services.forecasting.plan",
        "ForecastPlanService",
    ),
    "FcffForecastOrchestrationService": (
        "edgarito.services.forecasting.orchestration",
        "FcffForecastOrchestrationService",
    ),
    "ForecastOrchestrationService": (
        "edgarito.services.forecasting.orchestration",
        "ForecastOrchestrationService",
    ),
    "ForecastOrchestrationResult": (
        "edgarito.services.forecasting.orchestration",
        "ForecastOrchestrationResult",
    ),
    "DriverBasedForecastIncompleteError": (
        "edgarito.services.forecasting.orchestration",
        "DriverBasedForecastIncompleteError",
    ),
    "IncompleteFcffForecastMethodError": (
        "edgarito.services.forecasting.orchestration",
        "IncompleteFcffForecastMethodError",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
