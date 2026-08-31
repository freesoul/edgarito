"""Async isolated backtest runner with explicit route and leakage boundaries."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from edgarito.schemas.forecasting import FcffForecastParameters
from edgarito.services.financials.availability import ObservationAvailabilityMode
from edgarito.services.forecasting.reasoning.contracts import ForecastReasoningInput
from edgarito.services.forecasting.reasoning.evidence import (
    build_evidence_catalog,
    content_hash,
)
from edgarito.services.forecasting.reasoning.reasoner import build_reasoning_content
from edgarito.services.forecasting.reasoning.service import (
    ReasonedDriverBasedForecastService,
)

from .calibration import aggregate_calibration
from .comparison import compare_forecasts
from .complexity import complexity_diagnostics
from .contracts import (
    ActualOutcomeData,
    BaselineComparison,
    ForecastBacktestCase,
    ForecastBacktestResult,
    RouteIdentity,
    _stable_value,
)
from .leakage import (
    LeakageError,
    audit_actual_outcomes,
    cutoff_financials,
    enforce_leakage,
)
from .scoring import _forecast_rows, score_assumptions, score_financials


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _canonical_forecast(value: Any) -> Any:
    for candidate in (
        _field(value, "canonical_forecast"),
        _field(value, "forecast"),
        _field(value, "driver_result"),
        value,
    ):
        if candidate is not None and _forecast_rows(candidate):
            return candidate
    return value


def _accepts_kwargs(signature: inspect.Signature) -> bool:
    return any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values())


def _filtered_kwargs(
    signature: inspect.Signature,
    values: dict[str, Any],
    *,
    preserve: set[str] | None = None,
) -> dict[str, Any]:
    if _accepts_kwargs(signature):
        # ``evidence`` is the one canonical evaluation argument.  Avoid
        # sending a fan-out of aliases to collaborators that intentionally
        # accept arbitrary keywords; this keeps baseline calls exact.
        return {
            key: value
            for key, value in values.items()
            if key
            in {
                "as_of",
                "availability_mode",
                "evidence",
                "method",
                *(preserve or set()),
            }
        }
    return {key: value for key, value in values.items() if key in signature.parameters}


async def _invoke_reasoned(runner: Any, financials: Any, input_value: Any, parameters: Any, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(runner)
        positional = tuple(
            item
            for item in signature.parameters.values()
            if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        has_var_positional = any(
            item.kind == inspect.Parameter.VAR_POSITIONAL
            for item in signature.parameters.values()
        )
    except (TypeError, ValueError):
        positional = (None, None, None)
        signature = None
        has_var_positional = False
    if has_var_positional or len(positional) >= 3:
        args = (financials, input_value, parameters)
    elif len(positional) >= 2:
        second = positional[1].name.casefold()
        if "param" in second:
            args = (financials, parameters)
            kwargs = {**kwargs, "reasoning_input": input_value, "input": input_value}
        else:
            args = (financials, input_value)
            kwargs = {**kwargs, "parameters": parameters}
    else:
        args = (financials,)
        kwargs = {**kwargs, "reasoning_input": input_value, "parameters": parameters}
    if signature is not None:
        kwargs = _filtered_kwargs(
            signature,
            kwargs,
            preserve=(
                {"reasoning_input", "input", "parameters"}
                if len(positional) < 3 and not has_var_positional
                else set()
            ),
        )
    raw = runner(*args, **kwargs)
    return await raw if inspect.isawaitable(raw) else raw


async def _invoke_baseline(runner: Any, financials: Any, parameters: Any, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(runner)
        positional = tuple(
            item
            for item in signature.parameters.values()
            if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    except (TypeError, ValueError):
        signature = None
        positional = (None, None)
        has_var_positional = False
    else:
        has_var_positional = any(
            item.kind == inspect.Parameter.VAR_POSITIONAL
            for item in signature.parameters.values()
        )
    if has_var_positional or len(positional) >= 2:
        args = (financials, parameters)
    else:
        args = (financials,)
        kwargs = {**kwargs, "parameters": parameters}
    if signature is not None:
        kwargs = _filtered_kwargs(
            signature,
            kwargs,
            preserve={"parameters"} if not has_var_positional and len(positional) < 2 else set(),
        )
    raw = runner(*args, **kwargs)
    return await raw if inspect.isawaitable(raw) else raw


class ForecastBacktestRunner:
    """Run reasoned forecasts and optional baselines against frozen outcomes."""

    def __init__(
        self,
        reasoned_service: Any | None = None,
        *,
        reasoned_forecast_service: Any | None = None,
        reasoner: Any | None = None,
        compiler: Any | None = None,
        driver_service: Any | None = None,
        normalized_baseline: Any | None = None,
        normalized_service: Any | None = None,
        hybrid_baseline: Any | None = None,
        hybrid_service: Any | None = None,
        leakage_auditor: Any | None = None,
    ) -> None:
        self.reasoned_service = reasoned_service or reasoned_forecast_service
        self.reasoner = reasoner
        self.compiler = compiler
        self.driver_service = driver_service
        self.normalized_baseline = normalized_baseline or normalized_service
        if self.normalized_baseline is None:
            from edgarito.services.forecasting._fcff.service import FcffForecastService

            self.normalized_baseline = FcffForecastService()
        self.hybrid_baseline = hybrid_baseline or hybrid_service
        self.leakage_auditor = leakage_auditor

    def _reasoned_runner(self) -> Any:
        if self.reasoned_service is not None:
            return getattr(self.reasoned_service, "forecast", None) or getattr(self.reasoned_service, "run", None) or self.reasoned_service
        service = ReasonedDriverBasedForecastService(
            reasoner=self.reasoner,
            compiler=self.compiler,
            driver_service=self.driver_service,
        )
        return service.forecast

    @staticmethod
    def _build_input(case: ForecastBacktestCase) -> ForecastReasoningInput:
        if case.reasoning_input is not None:
            return case.reasoning_input
        evidence = case.operating_evidence or case.evidence_snapshot
        values: dict[str, Any] = {}
        if isinstance(evidence, Mapping):
            values = dict(evidence)
        elif evidence is not None:
            for name in (
                "segments", "definitions", "observations", "management_guidance",
                "management_constraints", "investment_programs", "research_evidence",
                "evidence_consensus", "historical_facts",
            ):
                if hasattr(evidence, name):
                    values[name] = getattr(evidence, name)
        return ForecastReasoningInput.from_artifacts(
            case.point_in_time_financials,
            as_of=case.as_of,
            forecast_years=case.fiscal_years,
            **values,
        )

    async def run(
        self,
        case: ForecastBacktestCase | Any,
        actual_outcomes: ActualOutcomeData | Any | None = None,
        *,
        actuals: ActualOutcomeData | Any | None = None,
    ) -> ForecastBacktestResult:
        """Run one case; outcome data is deliberately a separate argument."""

        case = case if isinstance(case, ForecastBacktestCase) else ForecastBacktestCase.model_validate(case)
        if actuals is not None:
            actual_outcomes = actuals
        if actual_outcomes is None:
            raise TypeError("Backtest run requires separate ActualOutcomeData")
        actuals_value = actual_outcomes if isinstance(actual_outcomes, ActualOutcomeData) else ActualOutcomeData.model_validate(actual_outcomes)
        if actuals_value.company and actuals_value.company.casefold() != case.company.casefold():
            raise ValueError("Actual outcome company does not match the backtest case")
        if actuals_value.ticker and actuals_value.ticker.casefold() != case.ticker.casefold():
            raise ValueError("Actual outcome ticker does not match the backtest case")
        input_value = self._build_input(case)
        if case.reasoning_input is None:
            case = case.model_copy(update={"reasoning_input": input_value})
        # This is intentionally the first service-adjacent operation.  No
        # reasoner, cache, compiler, or baseline is called before this gate.
        if self.leakage_auditor is None:
            leakage = enforce_leakage(
                case, availability_mode=ObservationAvailabilityMode.POINT_IN_TIME
            )
        else:
            validator = getattr(self.leakage_auditor, "validate", None) or getattr(
                self.leakage_auditor, "audit", None
            )
            if validator is None:
                raise TypeError("Injected leakage auditor must expose validate or audit")
            leakage = validator(case)
            if not leakage.valid:
                raise LeakageError(leakage)
        actual_audit = audit_actual_outcomes(actuals_value, as_of=case.as_of)
        financials = cutoff_financials(
            case.point_in_time_financials,
            as_of=case.as_of,
            availability_mode=ObservationAvailabilityMode.POINT_IN_TIME,
            availability_manifest=case.availability_manifest,
        )
        parameters = case.parameters or FcffForecastParameters(forecast_years=len(case.fiscal_years))
        evidence = (
            case.operating_evidence
            if case.operating_evidence is not None
            else case.evidence_snapshot
        )
        evidence_identity = content_hash(_stable_value(evidence))
        hybrid_evidence = (
            case.hybrid_evidence if case.hybrid_evidence is not None else evidence
        )
        hybrid_evidence_identity = content_hash(_stable_value(hybrid_evidence))
        if case.hybrid_evidence is not None and hybrid_evidence_identity != evidence_identity:
            raise ValueError(
                "hybrid evidence identity differs from the reasoned frozen evidence"
            )
        common_kwargs = {
            "as_of": case.as_of,
            "availability_mode": ObservationAvailabilityMode.POINT_IN_TIME,
            "evidence": evidence,
            "operating_evidence": evidence,
            "frozen_evidence": evidence,
            "evidence_snapshot": evidence,
        }
        reasoned = await _invoke_reasoned(
            self._reasoned_runner(), financials, input_value, parameters, common_kwargs
        )
        canonical = _canonical_forecast(reasoned)
        assumption_scores = score_assumptions(reasoned, actuals_value)
        reasoned_financial_scores = score_financials(
            canonical,
            actuals_value,
            method="reasoned",
            required_years=case.fiscal_years,
        )
        complexity = complexity_diagnostics(input_value, reasoned_result=reasoned)

        baseline_kwargs = {
            **common_kwargs,
        }
        normalized_comparison = await self._run_baseline(
            self.normalized_baseline,
            route="normalized",
            method="normalized",
            financials=financials,
            parameters=parameters,
            kwargs={**baseline_kwargs, "method": "normalized"},
            reasoned=canonical,
            actuals=actuals_value,
            required_years=case.fiscal_years,
            reasoned_evidence_identity=evidence_identity,
            baseline_evidence_identity=evidence_identity,
        )
        hybrid_comparison = await self._run_baseline(
            self.hybrid_baseline,
            route="hybrid",
            method="hybrid",
            financials=financials,
            parameters=parameters,
            kwargs={
                **baseline_kwargs,
                "method": "hybrid",
                "evidence": hybrid_evidence,
                "operating_evidence": hybrid_evidence,
                "frozen_evidence": hybrid_evidence,
                "evidence_snapshot": hybrid_evidence,
            },
            reasoned=canonical,
            actuals=actuals_value,
            required_years=case.fiscal_years,
            reasoned_evidence_identity=evidence_identity,
            baseline_evidence_identity=hybrid_evidence_identity,
        )
        per_method = {"reasoned": reasoned_financial_scores.scores}
        for comparison in (normalized_comparison, hybrid_comparison):
            if comparison is not None and comparison.financial_scores is not None:
                per_method[comparison.method] = comparison.financial_scores.scores
        financial_scores = reasoned_financial_scores.model_copy(update={"per_method": per_method})
        routes = (
            RouteIdentity(
                route="reasoned",
                collaborator=type(reasoned).__name__,
                evidence_identity=evidence_identity,
            ),
            *tuple(item.route for item in (normalized_comparison, hybrid_comparison) if item is not None),
        )
        diagnostics = tuple(
            dict.fromkeys(
                (
                    *getattr(reasoned, "warnings", ()),
                    *getattr(reasoned, "audit", ()),
                    f"actual_outcome_dates={len(actual_audit.dates)} audited separately",
                )
            )
        )
        report_identity = content_hash(
            _stable_value(
                {
                    "case_id": case.case_id,
                    "reasoned": reasoned,
                    "actuals": actuals_value,
                    "assumption_scores": assumption_scores,
                    "financial_scores": financial_scores,
                    "normalized": normalized_comparison,
                    "hybrid": hybrid_comparison,
                }
            )
        )
        return ForecastBacktestResult(
            case_id=case.case_id,
            ticker=case.ticker,
            company=case.company,
            as_of=case.as_of,
            fiscal_years=case.fiscal_years,
            leakage_audit=leakage,
            reasoned_result=reasoned,
            canonical_forecast=canonical,
            actual_outcomes=actuals_value,
            assumption_scores=assumption_scores,
            financial_scores=financial_scores,
            calibration=aggregate_calibration(assumption_scores),
            complexity=complexity,
            validation=getattr(reasoned, "validation", None),
            normalized_comparison=normalized_comparison,
            hybrid_comparison=hybrid_comparison,
            actual_outcome_audit=actual_audit,
            diagnostics=diagnostics,
            routes=routes,
            report_identity=report_identity,
        )

    async def _run_baseline(
        self,
        collaborator: Any | None,
        *,
        route: str,
        method: str,
        financials: Any,
        parameters: Any,
        kwargs: dict[str, Any],
        reasoned: Any,
        actuals: ActualOutcomeData,
        required_years: tuple[int, ...],
        reasoned_evidence_identity: str,
        baseline_evidence_identity: str,
    ) -> BaselineComparison:
        if collaborator is None:
            if route == "normalized":
                raise RuntimeError("normalized baseline collaborator is required")
            return BaselineComparison(
                route=RouteIdentity(
                    route=route,
                    available=False,
                    reason=f"no {route} baseline collaborator supplied",
                    evidence_identity=baseline_evidence_identity,
                ),
                method=method,
                unavailable_reason=f"no {route} baseline collaborator supplied",
            )
        runner = getattr(collaborator, "forecast", None) or getattr(collaborator, "run", None) or collaborator
        result = await _invoke_baseline(runner, financials, parameters, kwargs)
        baseline_forecast = _canonical_forecast(result)
        if not _forecast_rows(baseline_forecast):
            raise ValueError(f"{route} baseline returned empty forecast rows")
        return compare_forecasts(
            reasoned,
            baseline_forecast,
            actuals,
            route=route,
            method=method,
            collaborator=type(collaborator).__name__,
            required_years=required_years,
            reasoned_evidence_identity=reasoned_evidence_identity,
            baseline_evidence_identity=baseline_evidence_identity,
        )

    async def run_case(self, *args: Any, **kwargs: Any) -> ForecastBacktestResult:
        return await self.run(*args, **kwargs)


def build_evaluation_evidence_catalog(case: ForecastBacktestCase | Any):
    """Build the information catalog only after a strict leakage gate."""

    case = case if isinstance(case, ForecastBacktestCase) else ForecastBacktestCase.model_validate(case)
    enforce_leakage(case)
    if case.reasoning_input is None:
        raise ValueError("Evidence catalog requires a ForecastReasoningInput")
    return build_evidence_catalog(case.reasoning_input)


def evaluation_information_set_hash(case: ForecastBacktestCase | Any) -> str:
    case = case if isinstance(case, ForecastBacktestCase) else ForecastBacktestCase.model_validate(case)
    enforce_leakage(case)
    return content_hash(
        _stable_value(
            {
                "case_id": case.case_id,
                "reasoning_input": case.reasoning_input,
                "evidence_snapshot": case.evidence_snapshot,
                "operating_evidence": case.operating_evidence,
                "hybrid_evidence": case.hybrid_evidence,
            }
        )
    )


def build_evaluation_reasoning_content(case: ForecastBacktestCase | Any) -> str:
    case = case if isinstance(case, ForecastBacktestCase) else ForecastBacktestCase.model_validate(case)
    catalog = build_evaluation_evidence_catalog(case)
    if case.reasoning_input is None:
        raise ValueError("Reasoning content requires a ForecastReasoningInput")
    return build_reasoning_content(case.reasoning_input, catalog)


run_backtest = ForecastBacktestRunner
BacktestRunner = ForecastBacktestRunner


__all__ = [
    "ForecastBacktestRunner",
    "BacktestRunner",
    "run_backtest",
    "build_evaluation_evidence_catalog",
    "evaluation_information_set_hash",
    "build_evaluation_reasoning_content",
]
