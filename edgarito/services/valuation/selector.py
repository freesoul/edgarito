from dataclasses import dataclass, field

from edgarito.services.valuation.models import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    DataReadiness,
    EconomicTrait,
    ForecastProfile,
    ModelRole,
    ModelSuitability,
    RelativeValuationBasis,
    ValuationInput,
    ValuationModel,
    ValuationProfile,
    ValuationSelection,
)


@dataclass
class _Assessment:
    model: ValuationModel
    score: int
    forecast_profile: ForecastProfile | None = None
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    hard_rejections: list[str] = field(default_factory=list)
    required_inputs: set[ValuationInput] = field(default_factory=set)
    relative_bases: tuple[RelativeValuationBasis, ...] = ()


class ValuationModelSelector:
    """Rank valuation models by economic fit and report data blockers."""

    _BASE_SCORES = {
        BusinessArchetype.GENERAL_OPERATING: {
            ValuationModel.FCFF_DCF: 90,
            ValuationModel.EQUITY_DCF: 45,
            ValuationModel.RESIDUAL_INCOME: 30,
            ValuationModel.NAV_SOTP: 20,
            ValuationModel.COMPARABLE_MULTIPLES: 70,
        },
        BusinessArchetype.FINANCIAL_INTERMEDIARY: {
            ValuationModel.FCFF_DCF: 0,
            ValuationModel.EQUITY_DCF: 75,
            ValuationModel.RESIDUAL_INCOME: 95,
            ValuationModel.NAV_SOTP: 30,
            ValuationModel.COMPARABLE_MULTIPLES: 80,
        },
        BusinessArchetype.ASSET_MANAGER: {
            ValuationModel.FCFF_DCF: 45,
            ValuationModel.EQUITY_DCF: 85,
            ValuationModel.RESIDUAL_INCOME: 60,
            ValuationModel.NAV_SOTP: 35,
            ValuationModel.COMPARABLE_MULTIPLES: 80,
        },
        BusinessArchetype.REIT_PROPERTY: {
            ValuationModel.FCFF_DCF: 25,
            ValuationModel.EQUITY_DCF: 60,
            ValuationModel.RESIDUAL_INCOME: 20,
            ValuationModel.NAV_SOTP: 95,
            ValuationModel.COMPARABLE_MULTIPLES: 85,
        },
        BusinessArchetype.RESOURCE_PRODUCER: {
            ValuationModel.FCFF_DCF: 55,
            ValuationModel.EQUITY_DCF: 30,
            ValuationModel.RESIDUAL_INCOME: 25,
            ValuationModel.NAV_SOTP: 95,
            ValuationModel.COMPARABLE_MULTIPLES: 75,
        },
        BusinessArchetype.PROJECT_PIPELINE: {
            ValuationModel.FCFF_DCF: 35,
            ValuationModel.EQUITY_DCF: 20,
            ValuationModel.RESIDUAL_INCOME: 20,
            ValuationModel.NAV_SOTP: 85,
            ValuationModel.COMPARABLE_MULTIPLES: 60,
        },
        BusinessArchetype.HOLDING_COMPANY: {
            ValuationModel.FCFF_DCF: 20,
            ValuationModel.EQUITY_DCF: 45,
            ValuationModel.RESIDUAL_INCOME: 35,
            ValuationModel.NAV_SOTP: 95,
            ValuationModel.COMPARABLE_MULTIPLES: 55,
        },
        BusinessArchetype.CONGLOMERATE: {
            ValuationModel.FCFF_DCF: 50,
            ValuationModel.EQUITY_DCF: 30,
            ValuationModel.RESIDUAL_INCOME: 25,
            ValuationModel.NAV_SOTP: 95,
            ValuationModel.COMPARABLE_MULTIPLES: 60,
        },
    }

    def select(self, profile: ValuationProfile) -> ValuationSelection:
        assessments = {
            model: _Assessment(
                model=model,
                score=self._BASE_SCORES[profile.business_archetype][model],
            )
            for model in ValuationModel
        }
        self._assess_fcff(profile, assessments[ValuationModel.FCFF_DCF])
        self._assess_equity_dcf(profile, assessments[ValuationModel.EQUITY_DCF])
        self._assess_residual_income(
            profile, assessments[ValuationModel.RESIDUAL_INCOME]
        )
        self._assess_nav(profile, assessments[ValuationModel.NAV_SOTP])
        self._assess_multiples(
            profile, assessments[ValuationModel.COMPARABLE_MULTIPLES]
        )

        intrinsic = [
            assessment
            for assessment in assessments.values()
            if assessment.model != ValuationModel.COMPARABLE_MULTIPLES
            and not assessment.hard_rejections
        ]
        primary_model = max(intrinsic, key=lambda item: item.score).model

        models = [
            self._result(
                assessment,
                profile,
                primary_model,
            )
            for assessment in assessments.values()
        ]
        role_order = {
            ModelRole.PRIMARY: 0,
            ModelRole.CONDITIONAL: 1,
            ModelRole.CROSSCHECK: 2,
            ModelRole.NOT_RECOMMENDED: 3,
        }
        models.sort(key=lambda item: (role_order[item.role], -item.suitability_score))
        return ValuationSelection(profile=profile, models=models)

    @staticmethod
    def _assess_fcff(profile: ValuationProfile, assessment: _Assessment) -> None:
        assessment.required_inputs.update(
            {
                ValuationInput.FCFF_FORECAST,
                ValuationInput.WACC,
                ValuationInput.TERMINAL_GROWTH,
                ValuationInput.NET_DEBT,
                ValuationInput.DILUTED_SHARES,
            }
        )
        assessment.forecast_profile = ForecastProfile.STANDARD
        if profile.business_archetype == BusinessArchetype.FINANCIAL_INTERMEDIARY:
            assessment.hard_rejections.append(
                "Debt and regulatory capital are operating inputs for financial "
                "intermediaries, so enterprise FCFF is not separable reliably"
            )
            return
        if profile.business_archetype == BusinessArchetype.REIT_PROPERTY:
            assessment.limitations.append(
                "Standard FCFF does not capture property NAV and accounting "
                "depreciation can distort REIT operating economics"
            )
        if profile.business_archetype == BusinessArchetype.RESOURCE_PRODUCER:
            assessment.forecast_profile = ForecastProfile.ASSET_LEVEL
            assessment.limitations.append(
                "A perpetual corporate growth model must not replace reserve "
                "depletion and commodity-price scenarios"
            )
        elif profile.business_archetype == BusinessArchetype.PROJECT_PIPELINE:
            assessment.forecast_profile = ForecastProfile.PRODUCT_PIPELINE
            assessment.limitations.append(
                "Pipeline projects require probability-weighted project cash flows"
            )
        elif EconomicTrait.BACKLOG_DRIVEN in profile.economic_traits:
            assessment.forecast_profile = ForecastProfile.BACKLOG_DRIVEN
            assessment.reasons.append(
                "Backlog conversion should drive near-term revenue assumptions"
            )
        elif profile.cyclicality == Cyclicality.HIGH:
            assessment.forecast_profile = ForecastProfile.NORMALIZED_CYCLE
            assessment.reasons.append(
                "High cyclicality requires normalized margins across a full cycle"
            )
        elif profile.lifecycle in {
            CompanyLifecycle.GROWTH,
            CompanyLifecycle.UNPROFITABLE_GROWTH,
        }:
            assessment.forecast_profile = ForecastProfile.REVENUE_TO_MARGIN
            assessment.reasons.append(
                "Revenue growth and margin convergence are more informative than "
                "current free cash flow"
            )
        else:
            assessment.reasons.append(
                "Operating cash flows and financing can be valued separately"
            )
        if EconomicTrait.LEASE_INTENSIVE in profile.economic_traits:
            assessment.limitations.append(
                "Lease liabilities and lease expenses must be treated consistently"
            )
        if EconomicTrait.FINANCING_SUBSIDIARY in profile.economic_traits:
            assessment.limitations.append(
                "Separate the financing subsidiary from industrial operations"
            )
        if ValuationInput.FCF_HISTORY in profile.available_inputs:
            assessment.limitations.append(
                "Historical operating cash flow minus capex is not FCFF; an EBIT, "
                "tax, D&A and working-capital bridge is still required"
            )

    @staticmethod
    def _assess_equity_dcf(profile: ValuationProfile, assessment: _Assessment) -> None:
        assessment.forecast_profile = ForecastProfile.DIVIDEND_OR_FCFE
        assessment.required_inputs.update(
            {
                ValuationInput.EQUITY_CASH_FLOW_FORECAST,
                ValuationInput.COST_OF_EQUITY,
                ValuationInput.TERMINAL_GROWTH,
                ValuationInput.DILUTED_SHARES,
            }
        )
        if profile.business_archetype in {
            BusinessArchetype.FINANCIAL_INTERMEDIARY,
            BusinessArchetype.ASSET_MANAGER,
        }:
            assessment.reasons.append(
                "Equity cash flows are more meaningful than enterprise cash flows"
            )
        if EconomicTrait.DIVIDEND_PAYER in profile.economic_traits:
            assessment.score += 10
            assessment.reasons.append("The company has an established distribution")
        else:
            assessment.limitations.append(
                "No established dividend or distributable-equity-cash-flow policy "
                "has been supplied"
            )
        if EconomicTrait.STABLE_PAYOUT in profile.economic_traits:
            assessment.score += 5
            assessment.reasons.append("Payout policy is identified as stable")

    @staticmethod
    def _assess_residual_income(
        profile: ValuationProfile, assessment: _Assessment
    ) -> None:
        assessment.forecast_profile = ForecastProfile.EXCESS_RETURN
        assessment.required_inputs.update(
            {
                ValuationInput.BOOK_EQUITY,
                ValuationInput.FORECAST_ROE,
                ValuationInput.COST_OF_EQUITY,
                ValuationInput.DILUTED_SHARES,
            }
        )
        if profile.business_archetype == BusinessArchetype.FINANCIAL_INTERMEDIARY:
            assessment.reasons.append(
                "Book equity, ROE and regulatory capital drive financial-firm value"
            )
            if ValuationInput.TANGIBLE_BOOK_EQUITY not in profile.available_inputs:
                assessment.limitations.append(
                    "Tangible common equity is preferred for bank and insurer analysis"
                )
        if profile.latest_book_equity is not None and profile.latest_book_equity <= 0:
            assessment.hard_rejections.append(
                "Residual income requires economically meaningful positive book equity"
            )
        if EconomicTrait.BOOK_VALUE_UNRELIABLE in profile.economic_traits:
            assessment.hard_rejections.append(
                "Book value has been marked unreliable for economic valuation"
            )

    @staticmethod
    def _assess_nav(profile: ValuationProfile, assessment: _Assessment) -> None:
        assessment.forecast_profile = ForecastProfile.ASSET_LEVEL
        assessment.required_inputs.update(
            {
                ValuationInput.ASSET_LEVEL_VALUES,
                ValuationInput.NET_DEBT,
                ValuationInput.DILUTED_SHARES,
            }
        )
        if profile.business_archetype == BusinessArchetype.REIT_PROPERTY:
            assessment.reasons.append(
                "Current property values are more meaningful than depreciated book cost"
            )
            assessment.required_inputs.add(ValuationInput.AFFO)
        elif profile.business_archetype == BusinessArchetype.RESOURCE_PRODUCER:
            assessment.reasons.append(
                "Finite reserves and extraction economics drive intrinsic value"
            )
            assessment.required_inputs.add(ValuationInput.RESERVE_DATA)
        elif profile.business_archetype == BusinessArchetype.PROJECT_PIPELINE:
            assessment.forecast_profile = ForecastProfile.PRODUCT_PIPELINE
            assessment.reasons.append(
                "Projects should be valued separately with probability-adjusted cash flows"
            )
            assessment.required_inputs.add(ValuationInput.PIPELINE_DATA)
        elif profile.business_archetype in {
            BusinessArchetype.HOLDING_COMPANY,
            BusinessArchetype.CONGLOMERATE,
        }:
            assessment.forecast_profile = ForecastProfile.SEGMENT_LEVEL
            assessment.reasons.append(
                "Economically distinct holdings or divisions require separate values"
            )
            assessment.required_inputs.discard(ValuationInput.ASSET_LEVEL_VALUES)
            assessment.required_inputs.add(ValuationInput.SEGMENT_VALUES)
        else:
            assessment.limitations.append(
                "Balance-sheet asset totals alone do not establish current asset values"
            )

    @staticmethod
    def _assess_multiples(profile: ValuationProfile, assessment: _Assessment) -> None:
        assessment.required_inputs.update(
            {
                ValuationInput.PEER_SET,
                ValuationInput.PEER_VALUATION_DATA,
                ValuationInput.TARGET_MULTIPLE_METRICS,
            }
        )
        assessment.reasons.append(
            "Relative valuation is useful as a market-pricing cross-check"
        )
        if profile.business_archetype == BusinessArchetype.FINANCIAL_INTERMEDIARY:
            assessment.relative_bases = (
                RelativeValuationBasis.PRICE_TO_TANGIBLE_BOOK,
                RelativeValuationBasis.PRICE_TO_BOOK,
                RelativeValuationBasis.PE,
            )
        elif profile.business_archetype == BusinessArchetype.REIT_PROPERTY:
            assessment.relative_bases = (
                RelativeValuationBasis.PRICE_TO_AFFO,
                RelativeValuationBasis.PRICE_TO_NAV,
                RelativeValuationBasis.DIVIDEND_YIELD,
            )
        elif profile.business_archetype == BusinessArchetype.RESOURCE_PRODUCER:
            assessment.relative_bases = (
                RelativeValuationBasis.PRICE_TO_NAV,
                RelativeValuationBasis.EV_TO_EBITDA,
                RelativeValuationBasis.EV_TO_FCF,
            )
        elif profile.lifecycle in {
            CompanyLifecycle.GROWTH,
            CompanyLifecycle.UNPROFITABLE_GROWTH,
        }:
            assessment.relative_bases = (
                RelativeValuationBasis.EV_TO_REVENUE,
                RelativeValuationBasis.EV_TO_FCF,
            )
        elif profile.cyclicality == Cyclicality.HIGH:
            assessment.relative_bases = (
                RelativeValuationBasis.EV_TO_EBITDA,
                RelativeValuationBasis.PE,
            )
            assessment.limitations.append(
                "Multiples must use normalized full-cycle denominators"
            )
        else:
            assessment.relative_bases = (
                RelativeValuationBasis.EV_TO_EBIT,
                RelativeValuationBasis.EV_TO_FCF,
                RelativeValuationBasis.PE,
            )
        if profile.peer_count is not None and profile.peer_count < 5:
            assessment.limitations.append(
                "Fewer than five candidate peers were supplied; comparability is weak"
            )

    def _result(
        self,
        assessment: _Assessment,
        profile: ValuationProfile,
        primary_model: ValuationModel,
    ) -> ModelSuitability:
        score = max(0, min(100, assessment.score))
        missing = assessment.required_inputs - profile.available_inputs
        readiness = self._readiness(assessment, profile, missing)
        if assessment.hard_rejections:
            role = ModelRole.NOT_RECOMMENDED
        elif assessment.model == ValuationModel.COMPARABLE_MULTIPLES:
            role = ModelRole.CROSSCHECK if score >= 50 else ModelRole.NOT_RECOMMENDED
        elif assessment.model == primary_model:
            role = ModelRole.PRIMARY
        elif score >= 60:
            role = ModelRole.CONDITIONAL
        else:
            role = ModelRole.NOT_RECOMMENDED
        return ModelSuitability(
            model=assessment.model,
            role=role,
            suitability_score=score,
            data_readiness=readiness,
            forecast_profile=assessment.forecast_profile,
            reasons=assessment.reasons,
            limitations=assessment.limitations,
            hard_rejections=assessment.hard_rejections,
            missing_inputs=missing,
            relative_bases=assessment.relative_bases,
        )

    @staticmethod
    def _readiness(
        assessment: _Assessment,
        profile: ValuationProfile,
        missing: set[ValuationInput],
    ) -> DataReadiness:
        if assessment.hard_rejections:
            return DataReadiness.NOT_APPLICABLE
        if not missing:
            return DataReadiness.READY
        supplied = assessment.required_inputs & profile.available_inputs
        return DataReadiness.PARTIAL if supplied else DataReadiness.BLOCKED
