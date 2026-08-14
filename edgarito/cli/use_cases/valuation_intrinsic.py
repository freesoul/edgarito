"""Intrinsic-model construction and execution use case."""

from __future__ import annotations

import inspect

from edgarito.cli.presentation.console import IndependentValuationModelsConsolePresenter
from edgarito.cli.use_cases.context import ValuationDependencyContext
from edgarito.services.valuation.execution import ValuationExecutor
from edgarito.services.valuation.models import BusinessArchetype, ValuationModel


def _resolve(dependencies, name: str, default):
    return ValuationDependencyContext(dependencies).resolve(name, default)


def execute_intrinsic_models(
    *,
    selection,
    runners,
    requested_models,
    verbose: bool,
    dependencies=None,
) -> None:
    """Execute prepared intrinsic runners and render their model comparison."""

    executor_type = _resolve(dependencies, "ValuationExecutor", ValuationExecutor)
    presenter_type = _resolve(
        dependencies,
        "IndependentValuationModelsConsolePresenter",
        IndependentValuationModelsConsolePresenter,
    )
    run = executor_type().execute(
        selection=selection,
        runners=runners,
        requested_models=requested_models,
    )
    print(presenter_type().render(run, verbose=verbose))


def build_intrinsic_runners(
    *,
    selected_model,
    economic_profile,
    profile,
    context,
    annual,
    cost_of_equity,
    terminal_growth,
    terminal_roe_override,
    args,
    runner_factory,
    dependencies=None,
):
    """Construct the selected intrinsic model runner set."""

    requested_models = None
    runners = {}
    if selected_model == "auto-specialized":
        candidates = {
            BusinessArchetype.FINANCIAL_INTERMEDIARY: (
                "residual-income",
                "fcfe-dcf",
                "ddm",
            ),
            BusinessArchetype.REIT_PROPERTY: (
                "property-nav",
                "reit-affo",
                "ddm",
            ),
            BusinessArchetype.RESOURCE_PRODUCER: ("resource-nav",),
            BusinessArchetype.PROJECT_PIPELINE: ("pipeline-rnpv",),
            BusinessArchetype.HOLDING_COMPANY: ("nav",),
            BusinessArchetype.CONGLOMERATE: ("nav",),
        }.get(economic_profile.business_archetype, ())
        for candidate in candidates:
            try:
                model, runner = _call_runner(
                    runner_factory,
                    {
                        "selected_model": candidate,
                        "profile": profile,
                        "context": context,
                        "annual": annual,
                        "cost_of_equity": cost_of_equity,
                        "terminal_growth": terminal_growth,
                        "terminal_roe_override": terminal_roe_override,
                        "args": args,
                        "dependencies": dependencies,
                    },
                )
            except ValueError:
                continue
            runners.setdefault(model, runner)
    else:
        try:
            model, runner = _call_runner(
                runner_factory,
                {
                    "selected_model": selected_model,
                    "profile": profile,
                    "context": context,
                    "annual": annual,
                    "cost_of_equity": cost_of_equity,
                    "terminal_growth": terminal_growth,
                    "terminal_roe_override": terminal_roe_override,
                    "args": args,
                    "dependencies": dependencies,
                },
            )
        except ValueError as exc:
            model = {
                "fcfe-dcf": ValuationModel.EQUITY_DCF,
                "reit-affo": ValuationModel.EQUITY_DCF,
                "ddm": ValuationModel.DIVIDEND_DISCOUNT,
                "residual-income": ValuationModel.RESIDUAL_INCOME,
                "nav": ValuationModel.NAV_SOTP,
                "property-nav": ValuationModel.NAV_SOTP,
                "resource-nav": ValuationModel.NAV_SOTP,
                "pipeline-rnpv": ValuationModel.NAV_SOTP,
            }[selected_model]
            message = str(exc)

            def runner(message=message):
                raise ValueError(message)

        runners[model] = runner
        requested_models = {model}
    return runners, requested_models


def _call_runner(runner_factory, kwargs):
    try:
        signature = inspect.signature(runner_factory)
    except (TypeError, ValueError):
        return runner_factory(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return runner_factory(**kwargs)
    return runner_factory(
        **{
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
    )


__all__ = ["build_intrinsic_runners", "execute_intrinsic_models"]
