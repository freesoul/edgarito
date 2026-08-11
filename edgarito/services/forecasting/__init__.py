from edgarito.services.forecasting.fcff import (
    FcffForecastService,
    FreeCashFlowForecastService,
)
from edgarito.services.forecasting.free_cash_flow import SimplifiedFcfForecastService
from edgarito.services.forecasting.models import (
    AdaptiveMultistagePlan,
    FcffForecast,
    FcffForecastDcfStub,
    FcffForecastDriver,
    FcffForecastObservation,
    FcffForecastParameters,
    FcffForecastYtdAnchor,
    ForecastAssumptionSource,
    ForecastSeedType,
    ForecastValue,
    ForwardGrowthEvidence,
    ForwardGrowthOutlook,
    FreeCashFlowForecast,
    FreeCashFlowForecastObservation,
    FreeCashFlowForecastParameters,
    MonetaryForecastConstraint,
    SimplifiedFcfForecast,
    SimplifiedFcfForecastObservation,
    SimplifiedFcfForecastParameters,
)
from edgarito.services.forecasting.multistage import (
    AdaptiveMultistageFcffForecastService,
)

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


def __getattr__(name):
    """Load the optional provider resolver without creating an import cycle."""

    if name in {
        "ForwardEstimateResolver",
        "ForwardEstimateService",
        "ForwardRevenueConsensusService",
        "ForwardRevenueEstimateResolver",
        "ForwardRevenueEstimateService",
    }:
        from edgarito.services.forward_estimates import (
            ForwardEstimateResolver,
            ForwardEstimateService,
            ForwardRevenueConsensusService,
            ForwardRevenueEstimateResolver,
            ForwardRevenueEstimateService,
        )

        return {
            "ForwardEstimateResolver": ForwardEstimateResolver,
            "ForwardEstimateService": ForwardEstimateService,
            "ForwardRevenueConsensusService": ForwardRevenueConsensusService,
            "ForwardRevenueEstimateResolver": ForwardRevenueEstimateResolver,
            "ForwardRevenueEstimateService": ForwardRevenueEstimateService,
        }[name]
    raise AttributeError(name)
