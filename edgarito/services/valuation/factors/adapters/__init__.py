"""Opt-in adapters into the provider-neutral factor contracts."""

from edgarito.services.valuation.factors.adapters.economic import (
    EconomicLeafFactorAdapter,
)
from edgarito.services.valuation.factors.adapters.financials import (
    FinancialAvailabilityPolicy,
    FinancialEvidenceAvailabilityPolicy,
    FinancialObservationFactorAdapter,
    FinancialsFactorAdapter,
    InvestmentProgramFactorAdapter,
    ManagementGuidanceAdapter,
    ManagementGuidanceFactorAdapter,
    NormalizedFinancialFactorAdapter,
    ObservationAvailabilityMode,
    OperatingInvestmentProgramFactorAdapter,
)
from edgarito.services.valuation.factors.adapters.reasoning import (
    FactorAugmentedReasoningInput,
    FactorReasoningContext,
    FactorReasoningContextItem,
    ResolvedFactorReasoningAdapter,
)
from edgarito.services.valuation.factors.adapters.research import (
    ExistingResearchEvidenceAdapter,
    ExistingResearchFactorAdapter,
    ResearchEvidenceAdapter,
    ResearchEvidenceFactorAdapter,
    ResearchFactorAdapter,
)

__all__ = [
    "EconomicLeafFactorAdapter",
    "ExistingResearchEvidenceAdapter",
    "ExistingResearchFactorAdapter",
    "FactorAugmentedReasoningInput",
    "FactorReasoningContext",
    "FactorReasoningContextItem",
    "FinancialAvailabilityPolicy",
    "FinancialEvidenceAvailabilityPolicy",
    "FinancialObservationFactorAdapter",
    "FinancialsFactorAdapter",
    "InvestmentProgramFactorAdapter",
    "ManagementGuidanceAdapter",
    "ManagementGuidanceFactorAdapter",
    "NormalizedFinancialFactorAdapter",
    "ObservationAvailabilityMode",
    "OperatingInvestmentProgramFactorAdapter",
    "ResearchEvidenceFactorAdapter",
    "ResearchEvidenceAdapter",
    "ResearchFactorAdapter",
    "ResolvedFactorReasoningAdapter",
]
