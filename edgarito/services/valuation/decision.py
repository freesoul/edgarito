from __future__ import annotations

import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.services.financial_observation_availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting import (
    AdaptiveMultistageFcffForecastService,
    FcffForecast,
    FcffForecastParameters,
    FcffForecastService,
    ForwardGrowthEvidence,
)
from edgarito.services.valuation.decision_models import (
    DecisionScenario,
    DecisionValuationResult,
    IntrinsicScenarioCase,
    PriceComparison,
    RelativeScenarioCase,
    RelativeScenarioTimeBasis,
    ReverseDcfSolution,
    ReverseDcfStatus,
    ReverseDcfVariable,
    ScenarioAssumption,
    SensitivityCell,
    SensitivityTable,
    ValuationAssessment,
    ValuationAssessmentBand,
)
from edgarito.services.valuation.fcff_dcf import FcffDcfService
from edgarito.services.valuation.models import (
    ComparableImpliedValuation,
    ComparableImpliedValuationCase,
    FcffDcfCapitalBridge,
    FcffDcfResult,
    ShareRepurchaseParameters,
    TerminalValueMethod,
)

if TYPE_CHECKING:
    from edgarito.config.valuation import MultistageValuationConfiguration

PERCENT = Decimal(100)


def _decision_value_per_share(result: FcffDcfResult) -> Decimal:
    if result.share_repurchases is not None:
        return result.share_repurchases.value_per_remaining_share
    return result.value_per_share


@dataclass(frozen=True)
class DecisionScenarioPolicy:
    revenue_growth_delta: Decimal = Decimal("2")
    operating_margin_delta: Decimal = Decimal("2")
    bear_wacc_delta: Decimal = Decimal("0.75")
    bull_wacc_delta: Decimal = Decimal("0.50")
    terminal_growth_delta: Decimal = Decimal("0.25")
    terminal_roic_spread_change: Decimal = Decimal("0.25")
    fair_value_band: Decimal = Decimal("5")
    sensitivity_size: int = 5


@dataclass(frozen=True)
class IntrinsicDecisionContext:
    financials: NormalizedCompanyFinancials
    requested_parameters: FcffForecastParameters
    seed_forecast: FcffForecast
    base_forecast: FcffForecast
    base_result: FcffDcfResult
    capital_bridge: FcffDcfCapitalBridge
    terminal_roic: Decimal
    multistage_configuration: MultistageValuationConfiguration
    use_multistage: bool
    valuation_date: datetime.date
    availability_mode: ObservationAvailabilityMode = (
        ObservationAvailabilityMode.POINT_IN_TIME
    )
    normalized_tax_rate: Decimal | None = None
    share_repurchase_parameters: ShareRepurchaseParameters | None = None
    forward_evidence: ForwardGrowthEvidence | None = None
    flexible_revenue_growth: bool = True
    flexible_operating_margin: bool = True
    flexible_terminal_roic: bool = True
    flexible_wacc: bool = True
    flexible_terminal_growth: bool = True

    @property
    def base_wacc(self) -> Decimal:
        return self.base_result.parameters.wacc

    @property
    def base_terminal_growth(self) -> Decimal:
        growth = self.base_result.parameters.perpetual_growth_rate
        if growth is None:
            raise ValueError("Decision analysis requires a perpetuity-growth DCF")
        return growth

    @property
    def base_revenue_growth(self) -> Decimal:
        configured = self.requested_parameters.revenue_growth
        if configured is not None:
            return configured[0]
        return self.base_forecast.observations[0].revenue_growth

    @property
    def base_operating_margin(self) -> Decimal:
        configured = self.requested_parameters.operating_margin
        if configured is not None:
            return configured[0]
        return self.base_forecast.observations[0].operating_margin


@dataclass(frozen=True)
class _Evaluation:
    forecast: FcffForecast
    result: FcffDcfResult
    terminal_roic: Decimal
    terminal_growth: Decimal
    wacc: Decimal


class IntrinsicDecisionEngine:
    """Revalue the existing FCFF model under explicit, auditable changes."""

    def __init__(self, context: IntrinsicDecisionContext):
        if (
            context.base_result.parameters.terminal_method
            != TerminalValueMethod.PERPETUITY_GROWTH
        ):
            raise ValueError("Decision analysis currently requires perpetuity growth")
        self.context = context
        self._forecast_service = FcffForecastService()

    def evaluate(
        self,
        *,
        revenue_growth: Decimal | None = None,
        operating_margin: Decimal | None = None,
        terminal_roic: Decimal | None = None,
        terminal_growth: Decimal | None = None,
        wacc: Decimal | None = None,
        preserve_projection_structure: bool = False,
    ) -> _Evaluation:
        context = self.context
        selected_roic = (
            terminal_roic if terminal_roic is not None else context.terminal_roic
        )
        selected_growth = (
            terminal_growth
            if terminal_growth is not None
            else context.base_terminal_growth
        )
        selected_wacc = wacc if wacc is not None else context.base_wacc
        if selected_wacc <= selected_growth:
            raise ValueError("WACC must exceed terminal growth")
        if selected_roic <= selected_growth:
            raise ValueError("Terminal ROIC must exceed terminal growth")

        parameters = context.requested_parameters.model_copy(
            update={
                "revenue_growth": (
                    self._shifted_path("revenue_growth", revenue_growth)
                    if revenue_growth is not None
                    else context.requested_parameters.revenue_growth
                ),
                "operating_margin": (
                    self._shifted_path("operating_margin", operating_margin)
                    if operating_margin is not None
                    else context.requested_parameters.operating_margin
                ),
            }
        )
        forecast = self._forecast_service.forecast(
            context.financials,
            parameters,
            as_of=context.valuation_date,
            availability_mode=context.availability_mode,
        )
        if context.use_multistage:
            configuration = context.multistage_configuration.model_copy(
                update={"terminal_return_on_invested_capital": selected_roic}
            )
            fixed_plan = (
                context.base_result.multistage_plan
                if preserve_projection_structure
                else None
            )
            adaptive_seed = (
                context.base_forecast if preserve_projection_structure else forecast
            )
            forecast, plan = AdaptiveMultistageFcffForecastService(
                self._forecast_service
            ).forecast(
                context.financials,
                adaptive_seed,
                parameters,
                selected_growth,
                configuration,
                normalized_tax_rate=context.normalized_tax_rate,
                fixed_plan=fixed_plan,
                as_of=context.valuation_date,
                availability_mode=context.availability_mode,
                forward_evidence=context.forward_evidence,
            )
        else:
            plan = None

        dcf_parameters = context.base_result.parameters.model_copy(
            update={
                "wacc": selected_wacc,
                "wacc_source": "decision-analysis override",
                "perpetual_growth_rate": selected_growth,
                "perpetual_growth_source": "decision-analysis override",
            }
        )
        result = FcffDcfService().value(
            forecast,
            dcf_parameters,
            context.capital_bridge,
            multistage_plan=plan,
            valuation_date=context.valuation_date,
            share_repurchase_parameters=context.share_repurchase_parameters,
        )
        return _Evaluation(
            forecast=forecast,
            result=result,
            terminal_roic=selected_roic,
            terminal_growth=selected_growth,
            wacc=selected_wacc,
        )

    def _shifted_path(self, driver: str, target: Decimal) -> tuple[Decimal, ...]:
        base_values = tuple(
            getattr(item, driver)
            for item in self.context.seed_forecast.observations[
                : self.context.requested_parameters.forecast_years
            ]
        )
        shift = target - base_values[0]
        return tuple(value + shift for value in base_values)


class ScenarioValuationService:
    def __init__(self, policy: DecisionScenarioPolicy | None = None):
        self.policy = policy or DecisionScenarioPolicy()
        self._warnings: tuple[str, ...] = ()

    @property
    def warnings(self) -> tuple[str, ...]:
        """Warnings produced by the most recent scenario bundle."""

        return self._warnings

    def build(
        self, engine: IntrinsicDecisionEngine
    ) -> tuple[IntrinsicScenarioCase, ...]:
        context = engine.context
        self._warnings = ()
        base = self._base_case(context)
        bear = self._variant(engine, DecisionScenario.BEAR)
        bull = self._variant(engine, DecisionScenario.BULL)

        bear, bull = self._enforce_strict_ordering(
            bear,
            base,
            bull,
        )
        cases = (bear, base, bull)
        self._warnings = tuple(
            dict.fromkeys(warning for case in cases for warning in case.warnings)
        )
        return cases

    def _enforce_strict_ordering(self, bear, base, bull):
        """Reject, rather than repair, a non-monotonic stress bundle.

        Scenario values are not envelopes.  Selecting Base as a neutral bound
        would make a failed Bull stress look like a valid valuation and would
        also hide interactions between the economic drivers.  Each offending
        independently revalued case is therefore made explicitly unavailable.
        """

        base_value = base.value_per_share
        assert base_value is not None

        if bear.available:
            if not any(item.changed for item in bear.assumptions):
                bear = self._unavailable(
                    bear,
                    "explicit assumptions left the Bear stress unchanged from the "
                    "independently calculated Base; no genuine Bear revaluation "
                    "was available",
                )
            elif (
                bear.value_per_share is not None and bear.value_per_share >= base_value
            ):
                bear = self._unavailable(
                    bear,
                    "non-monotonic Bear stress: its independent revaluation is "
                    f"{bear.value_per_share:,.6f} per share, not strictly below "
                    f"Base at {base_value:,.6f}",
                )

        if bull.available:
            if not any(item.changed for item in bull.assumptions):
                bull = self._unavailable(
                    bull,
                    "explicit assumptions left the Bull stress unchanged from the "
                    "independently calculated Base; no genuine Bull revaluation "
                    "was available",
                )
            elif (
                bull.value_per_share is not None and bull.value_per_share <= base_value
            ):
                bull = self._unavailable(
                    bull,
                    "non-monotonic Bull stress: its independent revaluation is "
                    f"{bull.value_per_share:,.6f} per share, not strictly above "
                    f"Base at {base_value:,.6f}",
                )
        return bear, bull

    @staticmethod
    def _unavailable(case, reason):
        warning = f"{case.scenario.value.title()} scenario unavailable: {reason}."
        return case.model_copy(
            update={
                "value_per_share": None,
                "available": False,
                "invalid_reason": reason,
                "methodology": (
                    f"{case.methodology}; explicit invalid/unavailable output: {reason}"
                ),
                "warnings": tuple(dict.fromkeys([*case.warnings, warning])),
            }
        )

    def _base_case(self, context):
        values = self._base_values(context)
        return IntrinsicScenarioCase(
            scenario=DecisionScenario.BASE,
            value_per_share=_decision_value_per_share(context.base_result),
            assumptions=self._assumptions(
                context,
                values,
                DecisionScenario.BASE,
            ),
            methodology="Existing resolved FCFF DCF assumptions without modification",
        )

    def _variant(self, engine, scenario):
        context = engine.context
        values = self._base_values(context)
        try:
            values = self._scenario_values(context, scenario)
            return self._case(
                engine,
                scenario,
                values,
                methodology=(
                    "Independent FCFF scenario revaluation using one coherent "
                    f"{scenario.value} driver bundle; the independently selected "
                    "Base multistage stage topology is reused where available"
                ),
            )
        except ValueError as exc:
            return self._invalid_from_values(
                context,
                scenario,
                values,
                f"The stressed assumptions could not be revalued coherently: {exc}",
            )

    def _invalid_from_values(self, context, scenario, values, reason):
        return IntrinsicScenarioCase(
            scenario=scenario,
            value_per_share=None,
            assumptions=self._assumptions(context, values, scenario),
            methodology=(
                "Independent FCFF scenario revaluation was unavailable; the stressed "
                "bundle was not replaced with Base"
            ),
            available=False,
            invalid_reason=reason,
            warnings=(f"{scenario.value.title()} scenario unavailable: {reason}.",),
        )

    def _case(self, engine, scenario, values, *, methodology):
        context = engine.context
        evaluation = engine.evaluate(
            revenue_growth=(
                values["revenue_growth"] if context.flexible_revenue_growth else None
            ),
            operating_margin=(
                values["operating_margin"]
                if context.flexible_operating_margin
                else None
            ),
            terminal_roic=values["terminal_roic"],
            terminal_growth=values["terminal_growth"],
            wacc=values["wacc"],
            preserve_projection_structure=(
                context.use_multistage
                and context.base_result.multistage_plan is not None
            ),
        )
        return IntrinsicScenarioCase(
            scenario=scenario,
            value_per_share=_decision_value_per_share(evaluation.result),
            assumptions=self._assumptions(context, values, scenario),
            methodology=methodology,
        )

    def _scenario_values(self, context, scenario):
        base = self._base_values(context)
        direction = Decimal(-1) if scenario == DecisionScenario.BEAR else Decimal(1)
        growth = base["revenue_growth"]
        margin = base["operating_margin"]
        wacc = base["wacc"]
        terminal_growth = base["terminal_growth"]
        terminal_roic = base["terminal_roic"]
        if context.flexible_revenue_growth:
            growth += direction * self.policy.revenue_growth_delta
        if context.flexible_operating_margin:
            margin += direction * self.policy.operating_margin_delta
        if context.flexible_wacc:
            wacc += (
                self.policy.bear_wacc_delta
                if scenario == DecisionScenario.BEAR
                else -self.policy.bull_wacc_delta
            )
        if context.flexible_terminal_growth:
            terminal_growth += direction * self.policy.terminal_growth_delta
        if self._terminal_roic_is_economic(context) and context.flexible_terminal_roic:
            spread = max(terminal_roic - base["wacc"], Decimal("2"))
            terminal_roic += (
                direction * spread * self.policy.terminal_roic_spread_change
            )
            terminal_roic = min(Decimal(60), terminal_roic)
        if context.flexible_terminal_growth:
            terminal_growth = min(
                max(Decimal(0), terminal_growth),
                wacc - Decimal("0.5"),
                terminal_roic - Decimal("0.5"),
            )
        elif wacc <= terminal_growth:
            if context.flexible_wacc:
                wacc = terminal_growth + Decimal("0.01")
            else:
                raise ValueError("Explicit WACC must exceed explicit terminal growth")
        if self._terminal_roic_is_economic(context) and context.flexible_terminal_roic:
            terminal_roic = max(terminal_roic, terminal_growth + Decimal("0.5"))
        elif terminal_roic <= terminal_growth:
            if context.flexible_terminal_growth:
                terminal_growth = terminal_roic - Decimal("0.5")
            else:
                raise ValueError(
                    "Explicit terminal ROIC must exceed explicit terminal growth"
                )
        return {
            "revenue_growth": growth,
            "operating_margin": margin,
            "terminal_roic": terminal_roic,
            "wacc": wacc,
            "terminal_growth": terminal_growth,
        }

    @staticmethod
    def _terminal_roic_is_economic(context):
        return (
            context.use_multistage
            and context.multistage_configuration.fade_reinvestment_to_terminal
        )

    @staticmethod
    def _base_values(context):
        return {
            "revenue_growth": context.base_revenue_growth,
            "operating_margin": context.base_operating_margin,
            "terminal_roic": context.terminal_roic,
            "wacc": context.base_wacc,
            "terminal_growth": context.base_terminal_growth,
        }

    @staticmethod
    def _assumptions(context, values, scenario):
        base = ScenarioValuationService._base_values(context)
        flexible = {
            "revenue_growth": context.flexible_revenue_growth,
            "operating_margin": context.flexible_operating_margin,
            "terminal_roic": (
                context.flexible_terminal_roic
                and ScenarioValuationService._terminal_roic_is_economic(context)
            ),
            "wacc": context.flexible_wacc,
            "terminal_growth": context.flexible_terminal_growth,
        }
        labels = {
            "revenue_growth": "Initial revenue growth",
            "operating_margin": "Initial operating margin",
            "terminal_roic": "Terminal ROIC",
            "wacc": "WACC",
            "terminal_growth": "Terminal growth",
        }
        return tuple(
            ScenarioAssumption(
                name=labels[name],
                value=values[name],
                base_value=base[name],
                changed=values[name] != base[name],
                source=(
                    "base resolved assumption"
                    if scenario == DecisionScenario.BASE
                    else "scenario uncertainty policy"
                    if flexible[name]
                    else "preserved explicit profile/CLI override"
                ),
                methodology=(
                    "Unchanged base case"
                    if scenario == DecisionScenario.BASE
                    else "Symmetric operating uncertainty or bounded terminal-rate policy"
                    if flexible[name]
                    else "Explicit assumptions are not displaced by automatic scenarios"
                ),
            )
            for name in (
                "revenue_growth",
                "operating_margin",
                "terminal_roic",
                "wacc",
                "terminal_growth",
            )
        )


class SensitivityAnalysisService:
    def wacc_terminal_growth(
        self, engine: IntrinsicDecisionEngine, size: int = 5
    ) -> SensitivityTable:
        if size < 3 or size % 2 == 0:
            raise ValueError(
                "Sensitivity size must be an odd integer of at least three"
            )
        context = engine.context
        wacc_values = self._centered(context.base_wacc, Decimal("0.5"), size)
        growth_values = self._centered(
            context.base_terminal_growth, Decimal("0.25"), size
        )
        rows = []
        for wacc in wacc_values:
            cells = []
            for growth in growth_values:
                if growth >= wacc:
                    cells.append(
                        SensitivityCell(
                            row_value=wacc,
                            column_value=growth,
                            invalid_reason="terminal growth must be below WACC",
                        )
                    )
                    continue
                try:
                    evaluation = engine.evaluate(
                        wacc=wacc,
                        terminal_growth=growth,
                        preserve_projection_structure=True,
                    )
                    cells.append(
                        SensitivityCell(
                            row_value=wacc,
                            column_value=growth,
                            value_per_share=_decision_value_per_share(
                                evaluation.result
                            ),
                        )
                    )
                except ValueError as exc:
                    cells.append(
                        SensitivityCell(
                            row_value=wacc,
                            column_value=growth,
                            invalid_reason=str(exc),
                        )
                    )
            rows.append(tuple(cells))
        return SensitivityTable(
            name="Value/share sensitivity",
            row_label="WACC",
            column_label="Terminal growth",
            row_values=wacc_values,
            column_values=growth_values,
            cells=tuple(rows),
            methodology=(
                "Each cell fully reforecasts sustainable terminal reinvestment for "
                "the selected terminal growth and then rediscounts FCFF at the "
                "selected WACC"
            ),
        )

    @staticmethod
    def _centered(center, step, size):
        midpoint = size // 2
        return tuple(center + Decimal(index - midpoint) * step for index in range(size))


class BoundedRootFinder:
    def solve(
        self,
        function,
        lower: Decimal,
        upper: Decimal,
        *,
        preferred: Decimal,
        value_tolerance: Decimal = Decimal("0.01"),
        input_tolerance: Decimal = Decimal("0.0001"),
        scan_intervals: int = 40,
    ) -> tuple[Decimal, Decimal] | None:
        points = [
            lower + (upper - lower) * Decimal(index) / Decimal(scan_intervals)
            for index in range(scan_intervals + 1)
        ]
        observations = []
        for point in points:
            try:
                value = function(point)
            except (ArithmeticError, ValueError):
                continue
            if value.is_finite():
                observations.append((point, value))
                if abs(value) <= value_tolerance:
                    return point, value
        brackets = [
            (left, right)
            for left, right in zip(observations, observations[1:], strict=False)
            if left[1] == 0
            or right[1] == 0
            or (left[1] < 0 < right[1])
            or (right[1] < 0 < left[1])
        ]
        if not brackets:
            return None
        left, right = min(
            brackets,
            key=lambda pair: abs((pair[0][0] + pair[1][0]) / Decimal(2) - preferred),
        )
        for _ in range(100):
            midpoint = (left[0] + right[0]) / Decimal(2)
            value = function(midpoint)
            if abs(value) <= value_tolerance or right[0] - left[0] <= input_tolerance:
                return midpoint, value
            if (left[1] <= 0 <= value) or (value <= 0 <= left[1]):
                right = (midpoint, value)
            else:
                left = (midpoint, value)
        midpoint = (left[0] + right[0]) / Decimal(2)
        return midpoint, function(midpoint)


class ReverseDcfService:
    def __init__(self, root_finder: BoundedRootFinder | None = None):
        self.root_finder = root_finder or BoundedRootFinder()

    def solve_all(
        self, engine: IntrinsicDecisionEngine, current_price: Decimal
    ) -> tuple[ReverseDcfSolution, ...]:
        context = engine.context
        bounds = {
            ReverseDcfVariable.REVENUE_GROWTH: (
                max(Decimal("-20"), context.base_revenue_growth - Decimal("20")),
                min(Decimal("50"), context.base_revenue_growth + Decimal("30")),
            ),
            ReverseDcfVariable.OPERATING_MARGIN: (
                max(Decimal("-20"), context.base_operating_margin - Decimal("25")),
                min(Decimal("80"), context.base_operating_margin + Decimal("25")),
            ),
            ReverseDcfVariable.TERMINAL_ROIC: (
                max(context.base_terminal_growth + Decimal("0.5"), Decimal("3")),
                Decimal("60"),
            ),
            ReverseDcfVariable.TERMINAL_GROWTH: (
                Decimal(0),
                min(Decimal("5"), context.base_wacc - Decimal("0.5")),
            ),
            ReverseDcfVariable.WACC: (
                max(Decimal("3"), context.base_terminal_growth + Decimal("0.5")),
                Decimal("20"),
            ),
        }
        base_values = {
            ReverseDcfVariable.REVENUE_GROWTH: context.base_revenue_growth,
            ReverseDcfVariable.OPERATING_MARGIN: context.base_operating_margin,
            ReverseDcfVariable.TERMINAL_ROIC: context.terminal_roic,
            ReverseDcfVariable.TERMINAL_GROWTH: context.base_terminal_growth,
            ReverseDcfVariable.WACC: context.base_wacc,
        }
        return tuple(
            self._solve(
                engine,
                current_price,
                variable,
                base_values[variable],
                *bounds[variable],
            )
            for variable in ReverseDcfVariable
        )

    def _solve(self, engine, target_price, variable, base, lower, upper):
        def objective(value):
            arguments = {variable.value: value}
            if variable == ReverseDcfVariable.TERMINAL_GROWTH:
                arguments["preserve_projection_structure"] = True
            result = engine.evaluate(**arguments).result
            return _decision_value_per_share(result) - target_price

        solved = self.root_finder.solve(
            objective,
            lower,
            upper,
            preferred=base,
        )
        methodology = (
            "Bounded scan plus bisection root-finding; this variable changes alone "
            "while every other base-case assumption is held constant"
        )
        if solved is None:
            return ReverseDcfSolution(
                variable=variable,
                status=ReverseDcfStatus.NO_SOLUTION,
                base_value=base,
                lower_bound=lower,
                upper_bound=upper,
                target_price=target_price,
                methodology=methodology,
                explanation=(
                    "No economically valid solution brackets the market price within "
                    "the configured search range"
                ),
            )
        implied, residual = solved
        achieved = target_price + residual
        return ReverseDcfSolution(
            variable=variable,
            status=ReverseDcfStatus.SOLVED,
            base_value=base,
            implied_value=implied,
            lower_bound=lower,
            upper_bound=upper,
            achieved_price=achieved,
            target_price=target_price,
            methodology=methodology,
            explanation=(
                "Independent market-implied assumption; it is not a forecast and is "
                "not combined with the other reverse-DCF solutions"
            ),
        )


class DecisionValuationService:
    def __init__(self, policy: DecisionScenarioPolicy | None = None):
        self.policy = policy or DecisionScenarioPolicy()

    def build(
        self,
        context: IntrinsicDecisionContext,
        current_price: Decimal,
        relative: ComparableImpliedValuation | None = None,
        *,
        include_sensitivity: bool = True,
        include_reverse_dcf: bool = True,
    ) -> DecisionValuationResult:
        if current_price <= 0:
            raise ValueError("Decision analysis requires a positive current price")
        engine = IntrinsicDecisionEngine(context)
        scenario_service = ScenarioValuationService(self.policy)
        intrinsic = scenario_service.build(engine)
        relative_scenarios = self._relative_scenarios(relative, current_price)
        sensitivities = (
            (
                SensitivityAnalysisService().wacc_terminal_growth(
                    engine, self.policy.sensitivity_size
                ),
            )
            if include_sensitivity
            else ()
        )
        comparisons = self._comparisons(current_price, intrinsic, relative)
        target_date_relative = self._uses_target_date_relative(relative)
        assessment = self._assessment(
            current_price,
            intrinsic,
            relative_scenarios if not target_date_relative else (),
            target_date_relative=target_date_relative,
        )
        reverse = (
            ReverseDcfService().solve_all(engine, current_price)
            if include_reverse_dcf
            else ()
        )
        scenario_warnings = scenario_service.warnings
        return DecisionValuationResult(
            ticker=context.base_result.ticker,
            company_name=context.base_result.company_name,
            currency=context.base_result.unit,
            current_price=current_price,
            intrinsic_scenarios=intrinsic,
            relative_scenarios=relative_scenarios,
            sensitivity_tables=sensitivities,
            price_comparisons=comparisons,
            assessment=assessment,
            reverse_dcf=reverse,
            methodology=(
                "Scenario values are independent FCFF revaluations; relative ranges "
                "remain separate; target-date peer/historical values are not used as "
                "present-day comparisons or combined assessment inputs; no scenario or "
                "reverse-DCF assumption is calibrated to force agreement between models"
            ),
            warnings=scenario_warnings,
        )

    @staticmethod
    def _relative_scenarios(relative, current_price=None):
        if relative is None:
            return ()
        cases = DecisionValuationService._relative_case_group(relative)
        independent_peer = DecisionValuationService._uses_target_date_relative(relative)
        time_basis = (
            RelativeScenarioTimeBasis.TARGET_DATE
            if independent_peer
            else RelativeScenarioTimeBasis.PRESENT_DAY
        )
        target_date = relative.target_date if independent_peer else None
        horizon_years = relative.horizon_years if independent_peer else None
        return tuple(
            RelativeScenarioCase(
                scenario=scenario,
                value_per_share=DecisionValuationService._relative_case_value(
                    case, independent_peer
                ),
                multiple=case.multiple,
                methodology=(
                    "Independent pure peer target-date multiple range; no intrinsic "
                    "WACC discounting is applied"
                    if independent_peer
                    else "Existing evidence-constrained relative multiple range, "
                    "discounted to a present-value equivalent"
                ),
                time_basis=time_basis,
                target_date=target_date,
                horizon_years=horizon_years,
                horizon_upside_downside=(
                    (
                        DecisionValuationService._relative_case_value(
                            case, independent_peer
                        )
                        / current_price
                        - Decimal(1)
                    )
                    * PERCENT
                    if independent_peer and current_price is not None
                    else None
                ),
            )
            for scenario, case in (
                (DecisionScenario.BEAR, cases[0]),
                (DecisionScenario.BASE, cases[1]),
                (DecisionScenario.BULL, cases[2]),
            )
        )

    @staticmethod
    def _relative_case_group(relative):
        if (
            isinstance(relative, ComparableImpliedValuation)
            and relative.pure_peer_lower_case is not None
            and relative.pure_peer_point_case is not None
            and relative.pure_peer_upper_case is not None
        ):
            return (
                relative.pure_peer_lower_case,
                relative.pure_peer_point_case,
                relative.pure_peer_upper_case,
            )
        return relative.lower_case, relative.point_case, relative.upper_case

    @staticmethod
    def _uses_target_date_relative(relative):
        return (
            isinstance(relative, ComparableImpliedValuation)
            and relative.pure_peer_lower_case is not None
            and relative.pure_peer_point_case is not None
            and relative.pure_peer_upper_case is not None
        )

    @staticmethod
    def _relative_case_value(case, independent_peer):
        if independent_peer and isinstance(case, ComparableImpliedValuationCase):
            return case.target_date_value_per_share
        return case.present_value_per_share

    @staticmethod
    def _comparisons(current_price, intrinsic, relative):
        cases = [
            (case.scenario.value.title(), "intrinsic", case.value_per_share)
            for case in intrinsic
            if case.available and case.value_per_share is not None
        ]
        if relative is not None:
            relative_cases = DecisionValuationService._relative_case_group(relative)
            independent_peer = DecisionValuationService._uses_target_date_relative(
                relative
            )
            if not independent_peer:
                cases.append(
                    (
                        "Relative",
                        "relative",
                        DecisionValuationService._relative_case_value(
                            relative_cases[1], independent_peer
                        ),
                    )
                )
        return tuple(
            PriceComparison(
                label=label,
                model=model,
                value_per_share=value,
                upside_downside=(value / current_price - Decimal(1)) * PERCENT,
                margin_of_safety=(
                    (Decimal(1) - current_price / value) * PERCENT
                    if value > 0
                    else None
                ),
            )
            for label, model, value in cases
        )

    def _assessment(self, price, intrinsic, relative, *, target_date_relative=False):
        available_intrinsic = tuple(
            case
            for case in intrinsic
            if case.available and case.value_per_share is not None
        )
        base_case = intrinsic[1]
        assert base_case.value_per_share is not None
        if len(available_intrinsic) == 3:
            intrinsic_band = self._band(
                price,
                intrinsic[0].value_per_share,
                intrinsic[1].value_per_share,
                intrinsic[2].value_per_share,
            )
            intrinsic_rationale = None
        else:
            intrinsic_band = self._single_value_band(price, base_case.value_per_share)
            unavailable = ", ".join(
                case.scenario.value.title() for case in intrinsic if not case.available
            )
            intrinsic_rationale = (
                "Only the independently calculated Base intrinsic value was used "
                f"because {unavailable} scenario evidence was unavailable"
            )
        if not relative:
            return ValuationAssessment(
                intrinsic=intrinsic_band,
                overall=intrinsic_band.value,
                rationale=(
                    *(  # Keep unavailable scenario evidence explicit in the audit.
                        (intrinsic_rationale,)
                        if intrinsic_rationale is not None
                        else ()
                    ),
                    (
                        "Only present-day intrinsic DCF scenario evidence was used; "
                        "target-date peer/historical values are reported separately "
                        "with horizon upside and excluded from margin-of-safety and "
                        "combined assessment"
                        if target_date_relative
                        else "Only intrinsic scenario evidence was available"
                    ),
                ),
            )
        relative_band = self._band(
            price,
            relative[0].value_per_share,
            relative[1].value_per_share,
            relative[2].value_per_share,
        )
        ranks = list(ValuationAssessmentBand)
        left = ranks.index(intrinsic_band)
        right = ranks.index(relative_band)
        dispersion_value = (
            abs(intrinsic[1].value_per_share / relative[1].value_per_share - Decimal(1))
            * PERCENT
        )
        dispersion = (
            "low"
            if dispersion_value < Decimal(10)
            else "moderate"
            if dispersion_value < Decimal(25)
            else "high"
        )
        if left == right:
            overall = intrinsic_band.value
            rationale = "Intrinsic and relative classifications agree"
        elif abs(left - right) == 1:
            overall = (
                f"{ranks[min(left, right)].value}-to-{ranks[max(left, right)].value}"
            )
            rationale = "Adjacent model classifications are preserved as a range"
        else:
            overall = (
                f"models disagree: intrinsic {intrinsic_band.value}; "
                f"relative {relative_band.value}"
            )
            rationale = "Material model disagreement is not averaged away"
        return ValuationAssessment(
            intrinsic=intrinsic_band,
            relative=relative_band,
            overall=overall,
            model_dispersion=dispersion,
            rationale=(
                *((intrinsic_rationale,) if intrinsic_rationale is not None else ()),
                rationale,
            ),
        )

    def _single_value_band(self, price, value):
        fair_fraction = self.policy.fair_value_band / PERCENT
        if price < value * (Decimal(1) - fair_fraction):
            return ValuationAssessmentBand.CHEAP
        if price <= value * (Decimal(1) + fair_fraction):
            return ValuationAssessmentBand.FAIR
        return ValuationAssessmentBand.EXPENSIVE

    def _band(self, price, bear, base, bull):
        fair_fraction = self.policy.fair_value_band / PERCENT
        if price <= bear:
            return ValuationAssessmentBand.STRONGLY_CHEAP
        if price < base * (Decimal(1) - fair_fraction):
            return ValuationAssessmentBand.CHEAP
        if price <= base * (Decimal(1) + fair_fraction):
            return ValuationAssessmentBand.FAIR
        if price <= bull:
            return ValuationAssessmentBand.EXPENSIVE
        return ValuationAssessmentBand.STRONGLY_EXPENSIVE


__all__ = [
    "BoundedRootFinder",
    "DecisionScenarioPolicy",
    "DecisionValuationService",
    "IntrinsicDecisionContext",
    "IntrinsicDecisionEngine",
    "ReverseDcfService",
    "ScenarioValuationService",
    "SensitivityAnalysisService",
]
