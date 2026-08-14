"""Command-line compatibility facade.

Command implementations live in
:mod:`edgarito.cli.use_cases.application`.  The facade keeps the historical
helper names available and takes a snapshot of facade-level overrides for each
call.  The snapshot is passed down explicitly; the application module is never
patched to make the old monkeypatch seams work.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable

from edgarito.cli.parser import build_parser as _build_parser
from edgarito.cli.use_cases import application as _application
from edgarito.cli.use_cases.context import DependencySnapshot

build_parser = _build_parser


# Keep every function defined by the former facade available at the same path.
_FUNCTION_NAMES = (
    "_valuation_step",
    "_run_financials",
    "_run_sec_inventory",
    "_run_metrics",
    "_run_export",
    "_run_red_flags",
    "_load_selected_valuation_profile",
    "_resolve_depreciable_asset_life_configuration",
    "_run_forecast",
    "_run_profile_intrinsic_valuation",
    "_profile_model_runner",
    "_equity_relative_valuation",
    "_run_valuation",
    "_run_valuation_models",
    "_run_classification",
    "_run_comparables",
    "_run_specialized_inputs",
    "_retrieve_classification",
    "_retrieve_financials",
    "_retrieve_forward_estimates",
    "_retrieve_operating_evidence",
    "_operating_quality_audit",
    "_retain_operating_audit_metadata",
    "_default_operating_vocabulary_audit",
    "_operating_evidence_provider",
    "_call_with_supported_kwargs",
    "_merge_forward_growth_evidence",
    "_materialize_forward_revenue_anchors",
    "_financial_snapshot_warnings",
    "_granularity",
    "_market_for_args",
    "_fcff_parameters",
    "_management_guidance_overlay",
    "_forward_growth_evidence",
    "_retrieve_automatic_assumption_inputs",
    "_apply_classification_overrides",
    "_resolve_wacc",
    "_validate_limit",
    "_parse_mappings",
    "_parse_provider_symbols",
    "main",
)


_IMPLEMENTATIONS = {name: getattr(_application, name) for name in _FUNCTION_NAMES}


# Re-export the application's imported collaborators as well as its constants.
# The old ``__main__`` module was also an integration seam for callers that
# supplied provider clients, credentials, and presenters.
for _name, _value in vars(_application).items():
    if not _name.startswith("__") and _name not in _FUNCTION_NAMES:
        globals().setdefault(_name, _value)


_COMPATIBILITY_NAMES = tuple(
    name for name in vars(_application) if not name.startswith("__")
)

# Values captured before installing facade wrappers are the application
# defaults for collaborators.  Function names get their wrapper as the facade
# default below; a wrapper is therefore not mistaken for a caller override.
_INITIAL_FACADE_DEFAULTS = {
    name: globals()[name]
    for name in _COMPATIBILITY_NAMES
    if name in globals()
}


def _dependency_snapshot() -> DependencySnapshot:
    """Capture facade overrides once for the current invocation."""

    namespace = globals()
    overrides = {
        name: namespace[name]
        for name, default in _FACADE_DEFAULTS.items()
        if name in namespace and namespace[name] is not default
    }
    return DependencySnapshot(overrides)


def _compatibility_wrapper(name: str) -> Callable[..., Any]:
    implementation = _IMPLEMENTATIONS[name]

    if inspect.iscoroutinefunction(implementation):

        @wraps(implementation)
        async def invoke(*args: Any, **kwargs: Any) -> Any:
            if name == "_profile_model_runner":
                kwargs.setdefault("dependencies", _dependency_snapshot())
            else:
                kwargs.setdefault("context", _dependency_snapshot())
            return await implementation(*args, **kwargs)

    else:

        @wraps(implementation)
        def invoke(*args: Any, **kwargs: Any) -> Any:
            if name == "_profile_model_runner":
                kwargs.setdefault("dependencies", _dependency_snapshot())
            else:
                kwargs.setdefault("context", _dependency_snapshot())
            return implementation(*args, **kwargs)

    invoke.__module__ = __name__
    return invoke


for _name in _FUNCTION_NAMES:
    globals()[_name] = _compatibility_wrapper(_name)


_FACADE_DEFAULTS = {
    **_INITIAL_FACADE_DEFAULTS,
    **{name: globals()[name] for name in _FUNCTION_NAMES},
}


def __getattr__(name: str) -> Any:
    """Resolve future application exports without breaking the facade."""

    return getattr(_application, name)
