from importlib import import_module

__all__ = [
    "AdaptiveMultistageFcffForecastService",
    "DriverBasedFcffForecastResult",
    "DriverBasedFcffForecastService",
    "DriverBasedForecastReadiness",
    "ForecastContextBuild",
    "build_forecast_context",
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
    "ForecastReasoner",
    "ForecastReasoningInput",
    "ForecastReasoningResponse",
    "ForecastReasoningValidator",
    "ForecastReasoningCompiler",
    "ReasonedDriverBasedForecastService",
    "ForecastReasonedDriverBasedForecastService",
    "ForecastReasoningInputValidationError",
    "ForecastReasoningInputError",
    "InvalidForecastReasoningInputError",
    "ForecastReasoningResult",
    "ReasonedDriverBasedForecastResult",
    "ReasonedForecastAssumption",
    "ForecastReasoningValueBasis",
    "ForecastReasoningCache",
    "EvidenceCatalog",
    "EvidenceRecord",
    "EvidenceCatalogExclusion",
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
    "DriverBasedFcffForecastService": (
        "edgarito.services.forecasting._fcff.driver_based",
        "DriverBasedFcffForecastService",
    ),
    "DriverBasedFcffForecastResult": (
        "edgarito.services.forecasting._fcff.driver_based",
        "DriverBasedFcffForecastResult",
    ),
    "DriverBasedForecastReadiness": (
        "edgarito.services.forecasting._fcff.driver_based",
        "DriverBasedForecastReadiness",
    ),
    "ForecastContextBuild": (
        "edgarito.services.forecasting._fcff.context",
        "ForecastContextBuild",
    ),
    "build_forecast_context": (
        "edgarito.services.forecasting._fcff.context",
        "build_forecast_context",
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
    "ForecastReasoner": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoner",
    ),
    "ForecastReasoningInput": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoningInput",
    ),
    "ForecastReasoningResponse": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoningResponse",
    ),
    "ForecastReasoningValidator": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoningValidator",
    ),
    "ForecastReasoningCompiler": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoningCompiler",
    ),
    "ReasonedDriverBasedForecastService": (
        "edgarito.services.forecasting.reasoning",
        "ReasonedDriverBasedForecastService",
    ),
    "ForecastReasonedDriverBasedForecastService": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasonedDriverBasedForecastService",
    ),
    "ForecastReasoningInputValidationError": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoningInputValidationError",
    ),
    "ForecastReasoningInputError": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoningInputError",
    ),
    "InvalidForecastReasoningInputError": (
        "edgarito.services.forecasting.reasoning",
        "InvalidForecastReasoningInputError",
    ),
    "ForecastReasoningResult": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoningResult",
    ),
    "ReasonedDriverBasedForecastResult": (
        "edgarito.services.forecasting.reasoning",
        "ReasonedDriverBasedForecastResult",
    ),
    "ReasonedForecastAssumption": (
        "edgarito.services.forecasting.reasoning",
        "ReasonedForecastAssumption",
    ),
    "ForecastReasoningValueBasis": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoningValueBasis",
    ),
    "ForecastReasoningCache": (
        "edgarito.services.forecasting.reasoning",
        "ForecastReasoningCache",
    ),
    "EvidenceCatalog": (
        "edgarito.services.forecasting.reasoning",
        "EvidenceCatalog",
    ),
    "EvidenceRecord": (
        "edgarito.services.forecasting.reasoning",
        "EvidenceRecord",
    ),
    "EvidenceCatalogExclusion": (
        "edgarito.services.forecasting.reasoning",
        "EvidenceCatalogExclusion",
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
