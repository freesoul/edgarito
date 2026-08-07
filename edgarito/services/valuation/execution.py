"""Selector-driven execution for independent valuation models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from decimal import Decimal
from statistics import median
from typing import Any

from edgarito.schemas.valuation.intrinsic import (
    ExecutedValuation,
    ForecastSummaryPoint,
    InputProvenance,
    IntrinsicValuationResult,
    ModelWarning,
    ResolvedModelAssumption,
    SkippedValuation,
    ValuationConfidence,
    ValuationDispersion,
    ValuationRunResult,
    WarningSeverity,
)
from edgarito.services.valuation.models import (
    DataReadiness,
    FcffDcfResult,
    ModelRole,
    ValuationInput,
    ValuationModel,
    ValuationProfile,
    ValuationSelection,
)

ValuationRunner = Callable[
    [],
    IntrinsicValuationResult[Any] | tuple[IntrinsicValuationResult[Any], ...],
]


class ValuationInputCatalog:
    """Combine normalized, forecast, profile, and specialized readiness inputs."""

    @staticmethod
    def combine(
        profile: ValuationProfile,
        *input_sources: Iterable[ValuationInput],
    ) -> ValuationProfile:
        available = set(profile.available_inputs)
        for source in input_sources:
            available.update(source)
        return profile.model_copy(update={"available_inputs": available})


class ValuationExecutor:
    """Execute every ready selected model independently; never blend results."""

    def execute(
        self,
        *,
        selection: ValuationSelection,
        runners: dict[ValuationModel, ValuationRunner],
        requested_models: set[ValuationModel] | None = None,
        relative_cross_checks: tuple[Any, ...] = (),
    ) -> ValuationRunResult:
        executed: list[ExecutedValuation] = []
        skipped: list[SkippedValuation] = []
        for suitability in selection.models:
            if suitability.model == ValuationModel.COMPARABLE_MULTIPLES:
                continue
            if (
                requested_models is not None
                and suitability.model not in requested_models
            ):
                continue
            reasons = tuple(
                suitability.hard_rejections
                + suitability.limitations
                + suitability.reasons
            )
            if suitability.data_readiness != DataReadiness.READY:
                skipped.append(
                    SkippedValuation(
                        model=suitability.model,
                        role=suitability.role,
                        readiness=suitability.data_readiness,
                        missing_inputs=frozenset(
                            item.value for item in suitability.missing_inputs
                        ),
                        reasons=reasons,
                    )
                )
                continue
            if (
                suitability.role == ModelRole.NOT_RECOMMENDED
                and requested_models is None
            ):
                skipped.append(
                    SkippedValuation(
                        model=suitability.model,
                        role=suitability.role,
                        readiness=DataReadiness.NOT_APPLICABLE,
                        missing_inputs=frozenset(),
                        reasons=reasons or ("Model is economically inappropriate",),
                    )
                )
                continue
            runner = runners.get(suitability.model)
            if runner is None:
                skipped.append(
                    SkippedValuation(
                        model=suitability.model,
                        role=suitability.role,
                        readiness=DataReadiness.BLOCKED,
                        missing_inputs=frozenset(),
                        reasons=("No executable input adapter was resolved",),
                    )
                )
                continue
            try:
                output = runner()
            except ValueError as exc:
                skipped.append(
                    SkippedValuation(
                        model=suitability.model,
                        role=suitability.role,
                        readiness=DataReadiness.BLOCKED,
                        missing_inputs=frozenset(),
                        reasons=(str(exc),),
                    )
                )
                continue
            results = output if isinstance(output, tuple) else (output,)
            if not results:
                skipped.append(
                    SkippedValuation(
                        model=suitability.model,
                        role=suitability.role,
                        readiness=DataReadiness.BLOCKED,
                        reasons=("Valuation adapter returned no scenarios",),
                    )
                )
                continue
            for result in results:
                if result.model != suitability.model:
                    raise ValueError(
                        f"Runner for {suitability.model.value} returned {result.model.value}"
                    )
                executed.append(
                    ExecutedValuation(
                        role=suitability.role,
                        suitability=suitability,
                        result=result,
                    )
                )
        return ValuationRunResult(
            economic_profile=selection.profile,
            selection=selection,
            executed_models=tuple(executed),
            skipped_models=tuple(skipped),
            relative_cross_checks=relative_cross_checks,
            dispersion=self._dispersion(executed),
        )

    @staticmethod
    def _dispersion(
        results: list[ExecutedValuation],
    ) -> ValuationDispersion | None:
        if len(results) < 2:
            return None
        values = sorted(item.result.value_per_share for item in results)
        midpoint = Decimal(str(median(values)))
        spread = (
            Decimal(0)
            if midpoint == 0
            else (values[-1] - values[0]) / abs(midpoint) * Decimal(100)
        )
        return ValuationDispersion(
            minimum_value_per_share=values[0],
            maximum_value_per_share=values[-1],
            median_value_per_share=midpoint,
            range_as_percent_of_median=spread,
        )


def wrap_fcff_result(
    result: FcffDcfResult,
    *,
    confidence: ValuationConfidence = ValuationConfidence.MEDIUM,
) -> IntrinsicValuationResult[FcffDcfResult]:
    """Expose the existing detailed FCFF result through the common contract."""
    assumptions = [
        ResolvedModelAssumption(
            name="WACC",
            value=result.parameters.wacc,
            unit="percent",
            source=result.parameters.wacc_source,
        )
    ]
    if result.parameters.perpetual_growth_rate is not None:
        assumptions.append(
            ResolvedModelAssumption(
                name="Terminal growth",
                value=result.parameters.perpetual_growth_rate,
                unit="percent",
                source=result.parameters.perpetual_growth_source or "resolved",
            )
        )
    provenance: list[InputProvenance] = [
        InputProvenance(
            field="net_debt",
            source=result.capital_bridge.net_debt_source,
            observed_on=result.capital_bridge.debt_date,
        ),
        InputProvenance(
            field="diluted_shares",
            source=result.capital_bridge.shares_source,
            observed_on=result.capital_bridge.shares_date,
        ),
    ]
    common = IntrinsicValuationResult[FcffDcfResult](
        model=ValuationModel.FCFF_DCF,
        adapter="enterprise FCFF DCF",
        company_id=result.company_id,
        company_name=result.company_name,
        ticker=result.ticker,
        valuation_date=result.valuation_date,
        currency=result.unit,
        equity_value=result.equity_value,
        diluted_shares=result.capital_bridge.diluted_shares,
        value_per_share=result.value_per_share,
        assumptions=tuple(assumptions),
        forecast_summary=tuple(
            ForecastSummaryPoint(
                label=item.label or f"Period {item.period}",
                period=item.period,
                amount=item.amount,
                present_value=item.present_value,
                unit=result.unit,
            )
            for item in result.explicit_forecast_present_value.cash_flows
        ),
        confidence=confidence,
        warnings=tuple(
            ModelWarning(
                code=f"fcff_{index}",
                severity=WarningSeverity.MEDIUM,
                summary=warning,
            )
            for index, warning in enumerate(result.warnings, 1)
        ),
        provenance=tuple(provenance),
        details=result,
    )
    return common
