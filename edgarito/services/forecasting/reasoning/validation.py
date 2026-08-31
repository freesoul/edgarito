"""Deterministic post-validation of ForecastReasoner proposals."""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from edgarito.schemas.forecasting import ForecastScope
from edgarito.schemas.operating import OperatingArchetype, operating_units_compatible
from edgarito.services.forecasting.reasoning.contracts import (
    ForecastReasoningInput,
    ForecastReasoningResponse,
    ForecastReasoningValidationIssue,
    ProposedModelingDecision,
    ReasonedForecastAssumption,
    canonical_driver_id,
)
from edgarito.services.forecasting.reasoning.evidence import (
    EvidenceCatalog,
    EvidenceCatalogItem,
    build_evidence_catalog,
)
from edgarito.services.operating.registry import FORMULA_REGISTRY

_SUPPORTED_ARCHETYPES = frozenset(
    {
        OperatingArchetype.VOLUME_PRICE,
        OperatingArchetype.SUBSCRIBERS_ARPU,
        OperatingArchetype.CAPACITY_UTILIZATION_PRICE,
        OperatingArchetype.TRANSACTIONS_TAKE_RATE,
        OperatingArchetype.BACKLOG_CONVERSION,
        OperatingArchetype.STORE_COUNT_SALES_PER_STORE,
        OperatingArchetype.GENERIC_SEGMENT_GROWTH,
    }
)
_ARCHETYPE_INPUTS = {
    OperatingArchetype.VOLUME_PRICE: ("volume", "price"),
    OperatingArchetype.SUBSCRIBERS_ARPU: ("subscribers", "arpu"),
    OperatingArchetype.CAPACITY_UTILIZATION_PRICE: (
        "capacity",
        "utilization",
        "price",
    ),
    OperatingArchetype.TRANSACTIONS_TAKE_RATE: ("transactions", "take_rate"),
    OperatingArchetype.BACKLOG_CONVERSION: ("backlog", "conversion_rate"),
    OperatingArchetype.STORE_COUNT_SALES_PER_STORE: (
        "store_count",
        "sales_per_store",
    ),
    OperatingArchetype.GENERIC_SEGMENT_GROWTH: ("growth",),
}
_ALLOWED_ARCHETYPE_OPTIONAL_INPUTS: dict[OperatingArchetype, frozenset[str]] = {
    archetype: frozenset() for archetype in _SUPPORTED_ARCHETYPES
}
_DERIVED_METRICS = frozenset(
    {"gross_profit", "ebit", "tax", "nopat", "delta_nwc", "fcff"}
)
_FINANCIAL_METRICS = frozenset(
    {
        "revenue",
        "gross_margin",
        "r_and_d",
        "sg_and_a",
        "other_operating_items",
        "tax_rate",
        "depreciation_and_amortization",
        "depreciation_to_revenue",
        "capex",
        "capex_to_revenue",
        "operating_working_capital",
        "operating_working_capital_to_revenue",
    }
)
_SUPPORTED_DECISION_STRATEGIES = frozenset(
    {"driver", "consolidated", "explicit", "ratio", "residual", "ignore"}
)
_NON_PATH_DECISION_STRATEGIES = frozenset({"driver", "consolidated", "ignore"})
_SUPPORTED_METHODS = frozenset(
    {
        "evidence",
        "evidence_precedence",
        "consensus",
        "management_guidance",
        "historical_trend",
        "stable",
        "base_case",
        "conservative",
        "explicit",
        "absolute",
        "ratio",
        "percentage_points",
        "driver",
        "model_assumption",
        "volume_price",
        "subscribers_arpu",
        "capacity_utilization_price",
        "transactions_take_rate",
        "backlog_conversion",
        "store_count_sales_per_store",
        "generic_segment_growth",
    }
)
_RATE_UNITS = frozenset(
    {
        "%",
        "percent",
        "percentage",
        "percentage_points",
        "percentage_point",
        "pp",
    }
)
_FRACTION_UNITS = frozenset({"fraction", "ratio", "decimal"})


class ForecastReasoningValidationResult(BaseModel):
    """Accepted and rejected records; rejected records are never discarded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted_assumptions: tuple[ReasonedForecastAssumption, ...] = ()
    rejected_assumptions: tuple[ForecastReasoningValidationIssue, ...] = ()
    accepted_decisions: tuple[ProposedModelingDecision, ...] = ()
    rejected_decisions: tuple[ForecastReasoningValidationIssue, ...] = ()
    input_issues: tuple[ForecastReasoningValidationIssue, ...] = ()
    unresolved_items: tuple[Any, ...] = ()
    warnings: tuple[str, ...] = ()
    overall_confidence: str = "medium"

    @property
    def accepted(self) -> tuple[ReasonedForecastAssumption, ...]:
        return self.accepted_assumptions

    @property
    def rejected(self) -> tuple[ForecastReasoningValidationIssue, ...]:
        return self.rejected_assumptions

    @property
    def is_valid(self) -> bool:
        return (
            not self.rejected_assumptions
            and not self.rejected_decisions
            and not self.input_issues
        )

    @property
    def passed(self) -> bool:
        return self.is_valid


class ForecastReasoningValidator:
    """Validate proposals against immutable input and its evidence catalog."""

    def validate(
        self,
        response: ForecastReasoningResponse | Any,
        input_value: ForecastReasoningInput,
        catalog: EvidenceCatalog | None = None,
    ) -> ForecastReasoningValidationResult:
        response = (
            response
            if isinstance(response, ForecastReasoningResponse)
            else ForecastReasoningResponse.model_validate(response)
        )
        input_value = (
            input_value
            if isinstance(input_value, ForecastReasoningInput)
            else ForecastReasoningInput.model_validate(input_value)
        )
        catalog = catalog or build_evidence_catalog(input_value)
        rejected: list[ForecastReasoningValidationIssue] = []
        accepted: list[ReasonedForecastAssumption] = []
        by_key: dict[tuple[str, str, str], list[ReasonedForecastAssumption]] = (
            defaultdict(list)
        )
        for assumption in response.assumptions:
            by_key[assumption.target_key].append(assumption)
        duplicate_keys = {key for key, values in by_key.items() if len(values) > 1}
        assumption_id_counts: dict[str, int] = defaultdict(int)
        for assumption in response.assumptions:
            assumption_id_counts[assumption.assumption_id] += 1
        for assumption in response.assumptions:
            reasons = self._assumption_issues(assumption, input_value, catalog)
            if assumption.target_key in duplicate_keys:
                reasons.append(("DUPLICATE_TARGET", "Duplicate assumption target"))
            if assumption_id_counts[assumption.assumption_id] > 1:
                reasons.append(("DUPLICATE_ASSUMPTION_ID", "Duplicate assumption ID"))
            if reasons:
                rejected.extend(
                    ForecastReasoningValidationIssue(
                        assumption_id=assumption.assumption_id,
                        code=code,
                        reason=reason,
                    )
                    for code, reason in reasons
                )
            else:
                accepted.append(assumption)

        accepted_decisions: list[ProposedModelingDecision] = []
        rejected_decisions: list[ForecastReasoningValidationIssue] = []
        decision_keys: set[tuple[str, str, str]] = set()
        for decision in response.modeling_decisions:
            reasons = self._decision_issues(decision, input_value)
            target = _metric_key(decision.metric) or f"driver:{decision.driver_id}"
            key = (decision.scope.value, decision.scope_id, target)
            if key in decision_keys:
                reasons.append(
                    ("DUPLICATE_DECISION", "Duplicate modeling decision target")
                )
            decision_keys.add(key)
            if reasons:
                rejected_decisions.extend(
                    ForecastReasoningValidationIssue(
                        decision_id=decision.decision_id,
                        code=code,
                        reason=reason,
                    )
                    for code, reason in reasons
                )
            else:
                accepted_decisions.append(decision)

        warnings = list(response.warnings)
        warnings.extend(
            f"duplicate explicit evidence ID retained for audit: {evidence_id}"
            for evidence_id in catalog.duplicate_explicit_ids
        )
        input_issues = tuple(
            ForecastReasoningValidationIssue(
                code=code,
                reason=reason,
            )
            for definition in input_value.definitions
            for code, reason in self._definition_issues(definition)
        )
        input_issues += tuple(
            ForecastReasoningValidationIssue(
                code="DUPLICATE_EXPLICIT_EVIDENCE_ID",
                reason=f"Explicit evidence ID is duplicated: {evidence_id}",
            )
            for evidence_id in catalog.duplicate_explicit_ids
        )
        if input_issues:
            warnings.extend(issue.reason for issue in input_issues)
        warnings.extend(self._consensus_warnings(response.assumptions, catalog))
        warnings = list(dict.fromkeys(warnings))
        confidence = response.overall_confidence
        if any(item.confidence == "low" for item in accepted):
            confidence = "low"
        elif (
            any(item.confidence == "medium" for item in accepted)
            and confidence == "high"
        ):
            confidence = "medium"
        return ForecastReasoningValidationResult(
            accepted_assumptions=tuple(accepted),
            rejected_assumptions=tuple(rejected),
            accepted_decisions=tuple(accepted_decisions),
            rejected_decisions=tuple(rejected_decisions),
            input_issues=input_issues,
            unresolved_items=response.unresolved_items,
            warnings=tuple(warnings),
            overall_confidence=confidence,
        )

    validate_response = validate
    validate_proposal = validate
    post_validate = validate
    check = validate

    def _assumption_issues(
        self,
        assumption: ReasonedForecastAssumption,
        input_value: ForecastReasoningInput,
        catalog: EvidenceCatalog,
    ) -> list[tuple[str, str]]:
        issues: list[tuple[str, str]] = []
        if assumption.fiscal_years != input_value.forecast_years:
            issues.append(
                (
                    "HORIZON_MISMATCH",
                    "Assumption fiscal years do not exactly match input horizon",
                )
            )
        method = assumption.method.casefold().replace("-", "_").replace(" ", "_")
        if method not in _SUPPORTED_METHODS:
            issues.append(
                (
                    "UNSUPPORTED_METHOD",
                    f"Unsupported modeling method: {assumption.method}",
                )
            )
        if (
            assumption.scope == ForecastScope.COMPANY
            and assumption.scope_id != "company"
        ):
            issues.append(
                ("SCOPE_MISMATCH", "Company assumptions must use scope_id='company'")
            )
        if assumption.scope == ForecastScope.SEGMENT and assumption.scope_id not in {
            item.segment_id for item in input_value.segments
        }:
            issues.append(
                ("UNKNOWN_SEGMENT", f"Unknown segment scope: {assumption.scope_id}")
            )

        if assumption.target_type == "operating_driver":
            issues.extend(self._driver_issues(assumption, input_value))
        else:
            issues.extend(self._metric_issues(assumption, input_value))
        issues.extend(self._citation_issues(assumption, input_value, catalog))
        issues.extend(self._sanity_issues(assumption))
        return _unique_issues(issues)

    def _driver_issues(
        self,
        assumption: ReasonedForecastAssumption,
        input_value: ForecastReasoningInput,
    ) -> list[tuple[str, str]]:
        driver = canonical_driver_id(assumption.driver_id)
        definitions = [
            item
            for item in input_value.definitions
            if item.segment_id == assumption.scope_id
        ]
        if assumption.scope != ForecastScope.SEGMENT:
            return [
                ("DRIVER_SCOPE", "Operating driver assumptions must be segment-scoped")
            ]
        if driver in {"revenue", "segment_revenue"}:
            return [
                (
                    "DIRECT_SEGMENT_REVENUE",
                    "Direct segment revenue AI assumptions are forbidden",
                )
            ]
        if not definitions:
            return [
                (
                    "UNKNOWN_DRIVER",
                    f"No accepted driver definition for segment {assumption.scope_id}",
                )
            ]
        matching = [item for item in definitions if driver in item.required_inputs]
        definition_issues = [
            issue
            for definition in definitions
            for issue in self._definition_issues(definition)
        ]
        if not matching:
            if driver in {"revenue", "segment_revenue"}:
                return [
                    (
                        "DIRECT_SEGMENT_REVENUE",
                        "Direct segment revenue AI assumptions are forbidden",
                    )
                ]
            return [
                *definition_issues,
                *[
                    (
                        "DRIVER_NOT_REQUIRED",
                        f"Driver {driver!r} is not a required input of an accepted definition",
                    )
                ],
            ]
        issues: list[tuple[str, str]] = list(definition_issues)
        for definition in matching:
            if (
                definition.archetype not in _SUPPORTED_ARCHETYPES
                or definition.archetype not in FORMULA_REGISTRY
            ):
                issues.append(
                    (
                        "UNSUPPORTED_ARCHETYPE",
                        f"Unsupported operating archetype: {definition.archetype}",
                    )
                )
            expected_formula = definition.archetype.value
            if (
                definition.formula_id != expected_formula
                or FORMULA_REGISTRY.get(definition.formula_id) is None
            ):
                issues.append(
                    (
                        "UNSUPPORTED_FORMULA",
                        f"Formula is not registry-backed: {definition.formula_id}",
                    )
                )
            expected_unit = definition.units.get(driver)
            if expected_unit and _unit_style(expected_unit) != _unit_style(
                assumption.unit
            ):
                issues.append(
                    (
                        "UNIT_SCALE_MISMATCH",
                        f"{assumption.unit!r} does not preserve driver unit semantics {expected_unit!r}",
                    )
                )
            elif expected_unit and not operating_units_compatible(
                expected_unit, assumption.unit
            ):
                issues.append(
                    (
                        "UNIT_MISMATCH",
                        f"{assumption.unit!r} is incompatible with driver unit {expected_unit!r}",
                    )
                )
            if expected_unit and _currency_codes(expected_unit) != _currency_codes(
                assumption.unit
            ):
                if _currency_codes(expected_unit) or _currency_codes(assumption.unit):
                    issues.append(
                        (
                            "CURRENCY_MISMATCH",
                            f"{assumption.unit!r} does not preserve driver currency {expected_unit!r}",
                        )
                    )
        return issues

    @staticmethod
    def _definition_issues(definition) -> list[tuple[str, str]]:
        archetype = definition.archetype
        if archetype not in _SUPPORTED_ARCHETYPES or archetype not in FORMULA_REGISTRY:
            return [
                (
                    "UNSUPPORTED_ARCHETYPE",
                    f"Unsupported operating archetype: {archetype}",
                )
            ]
        known = set(_ARCHETYPE_INPUTS[archetype])
        actual_inputs = set(definition.input_metrics)
        actual_required = set(definition.required_inputs)
        optional = set(definition.optional_inputs)
        issues: list[tuple[str, str]] = []
        if actual_inputs != known:
            issues.append(
                (
                    "MALFORMED_DEFINITION_INPUTS",
                    f"{archetype.value} input_metrics must be exactly {tuple(_ARCHETYPE_INPUTS[archetype])}",
                )
            )
        allowed_optional = _ALLOWED_ARCHETYPE_OPTIONAL_INPUTS[archetype]
        if not optional.issubset(allowed_optional):
            issues.append(
                (
                    "UNSUPPORTED_OPTIONAL_INPUT",
                    f"{archetype.value} has unsupported optional inputs: {tuple(sorted(optional - allowed_optional))}",
                )
            )
        if actual_required != known - optional:
            issues.append(
                (
                    "MALFORMED_REQUIRED_INPUTS",
                    f"{archetype.value} required_inputs do not match the registry contract",
                )
            )
        if (
            definition.formula_id != archetype.value
            or FORMULA_REGISTRY.get(definition.formula_id) is None
        ):
            issues.append(
                (
                    "UNSUPPORTED_FORMULA",
                    f"Formula is not registry-backed or does not match archetype: {definition.formula_id}",
                )
            )
        if definition.output_metric != "revenue":
            issues.append(
                ("UNSUPPORTED_OUTPUT", "Operating archetypes must output revenue")
            )
        return issues

    def _metric_issues(
        self,
        assumption: ReasonedForecastAssumption,
        input_value: ForecastReasoningInput,
    ) -> list[tuple[str, str]]:
        metric = _metric_key(assumption.metric)
        if metric in _DERIVED_METRICS:
            return [("DERIVED_TARGET", f"Derived metric target is forbidden: {metric}")]
        if metric not in _FINANCIAL_METRICS:
            return [
                (
                    "UNSUPPORTED_METRIC",
                    f"Unsupported financial metric: {assumption.metric}",
                )
            ]
        if assumption.scope == ForecastScope.SEGMENT:
            if metric not in {"gross_margin", "r_and_d", "sg_and_a"}:
                return [
                    (
                        "UNSUPPORTED_SCOPE",
                        f"Metric {metric} is not supported at segment scope",
                    )
                ]
            if metric == "revenue":
                return [
                    (
                        "DIRECT_SEGMENT_REVENUE",
                        "Direct segment revenue AI assumptions are forbidden",
                    )
                ]
        basis = assumption.basis.value
        issues: list[tuple[str, str]] = []
        if metric == "gross_margin" and basis != "percentage_points":
            issues.append(
                ("BASIS_MISMATCH", f"{metric} requires percentage-point basis")
            )
        if metric == "tax_rate" and basis != "percentage_points":
            issues.append(
                ("BASIS_MISMATCH", f"{metric} requires percentage-point basis")
            )
        if metric in {
            "r_and_d",
            "sg_and_a",
            "depreciation_and_amortization",
            "capex",
            "operating_working_capital",
        } and basis not in {"absolute", "percent_of_revenue"}:
            issues.append(
                (
                    "BASIS_MISMATCH",
                    f"{metric} requires absolute or percent-of-revenue basis",
                )
            )
        if (
            metric
            in {
                "depreciation_to_revenue",
                "capex_to_revenue",
                "operating_working_capital_to_revenue",
            }
            and basis != "percent_of_revenue"
        ):
            issues.append(
                ("BASIS_MISMATCH", f"{metric} requires percent-of-revenue basis")
            )
        if metric in {"other_operating_items", "revenue"} and basis != "absolute":
            issues.append(("BASIS_MISMATCH", f"{metric} requires absolute basis"))
        expected_rate = metric in {
            "gross_margin",
            "tax_rate",
            "depreciation_to_revenue",
            "capex_to_revenue",
            "operating_working_capital_to_revenue",
        } or basis in {"percentage_points", "percent_of_revenue"}
        if expected_rate and not _is_explicit_percentage_unit(assumption.unit):
            issues.append(
                (
                    "UNIT_SCALE_MISMATCH",
                    f"{metric} requires an explicit percent/percentage-point unit, not a ratio/fraction/decimal unit",
                )
            )
        if not expected_rate and not operating_units_compatible(
            input_value.unit, assumption.unit
        ):
            issues.append(
                (
                    "UNIT_MISMATCH",
                    f"{assumption.unit!r} is incompatible with company unit {input_value.unit!r}",
                )
            )
        if not expected_rate and _currency_codes(input_value.unit) != _currency_codes(
            assumption.unit
        ):
            if _currency_codes(input_value.unit) or _currency_codes(assumption.unit):
                issues.append(
                    (
                        "CURRENCY_MISMATCH",
                        f"{assumption.unit!r} is incompatible with company currency {input_value.unit!r}",
                    )
                )
        return issues

    def _citation_issues(
        self,
        assumption: ReasonedForecastAssumption,
        input_value: ForecastReasoningInput,
        catalog: EvidenceCatalog,
    ) -> list[tuple[str, str]]:
        if assumption.evidence_based and not assumption.evidence_ids:
            return [
                (
                    "MISSING_CITATION",
                    "Evidence-based assumptions require catalog evidence IDs",
                )
            ]
        issues: list[tuple[str, str]] = []
        for evidence_id in assumption.evidence_ids:
            item = catalog.get(evidence_id)
            if item is None:
                exclusion = catalog.exclusion(evidence_id)
                issues.append(
                    (
                        "EXCLUDED_EVIDENCE" if exclusion else "UNKNOWN_EVIDENCE",
                        exclusion.reason
                        if exclusion
                        else f"Evidence ID does not exist: {evidence_id}",
                    )
                )
                continue
            issues.extend(self._evidence_scope_issues(item, assumption, input_value))
            issues.extend(self._target_evidence_issues(item, assumption))
            if item.dispersion is not None and item.dispersion > 0:
                width = max(assumption.high) - min(assumption.low)
                if assumption.confidence == "high":
                    issues.append(
                        (
                            "CONSENSUS_DISPERSION_CONFIDENCE",
                            "High consensus dispersion requires lower assumption confidence",
                        )
                    )
                if width < item.dispersion:
                    issues.append(
                        (
                            "CONSENSUS_DISPERSION_RANGE",
                            "Assumption uncertainty must be widened to cover consensus dispersion",
                        )
                    )
        return issues

    @staticmethod
    def _evidence_scope_issues(
        item: EvidenceCatalogItem,
        assumption: ReasonedForecastAssumption,
        input_value: ForecastReasoningInput,
    ) -> list[tuple[str, str]]:
        context = item.context_map
        issues: list[tuple[str, str]] = []
        inferred_segment = (
            item.scope_id
            if item.category in {"OP", "MANUAL"}
            and item.scope_id not in {None, "company"}
            else None
        )
        evidence_segment = (
            item.scope_id
            if item.scope == "segment"
            else context.get("segment")
            or context.get("segment_name")
            or inferred_segment
        )
        evidence_scope = item.scope or context.get("scope")
        if (
            assumption.scope == ForecastScope.COMPANY
            and (evidence_segment or evidence_scope == "segment")
            and not (item.is_total and item.exhaustive)
        ):
            issues.append(
                (
                    "EVIDENCE_SCOPE_MISMATCH",
                    f"Segment-specific evidence {item.evidence_id} is not consolidated/exhaustive",
                )
            )
        if assumption.scope == ForecastScope.SEGMENT and evidence_scope in {
            "company",
            "consolidated",
        }:
            issues.append(
                (
                    "EVIDENCE_SCOPE_MISMATCH",
                    f"Company-level evidence {item.evidence_id} cannot support a segment driver",
                )
            )
        if (
            assumption.scope == ForecastScope.SEGMENT
            and evidence_segment
            and evidence_segment != assumption.scope_id
        ):
            issues.append(
                (
                    "EVIDENCE_SCOPE_MISMATCH",
                    f"Evidence {item.evidence_id} belongs to another segment",
                )
            )
        evidence_company = context.get("company")
        if evidence_company and evidence_company.casefold() not in {
            input_value.company_id.casefold(),
            (input_value.company_name or "").casefold(),
        }:
            issues.append(
                (
                    "EVIDENCE_COMPANY_MISMATCH",
                    f"Evidence {item.evidence_id} belongs to another company",
                )
            )
        if item.source_date and item.source_date > input_value.as_of:
            issues.append(
                (
                    "EVIDENCE_AS_OF_MISMATCH",
                    f"Evidence {item.evidence_id} was not available as of input date",
                )
            )
        if (
            item.fiscal_year is not None
            and item.category in {"MGMT", "MARKET"}
            and item.fiscal_year not in input_value.forecast_years
        ):
            issues.append(
                (
                    "EVIDENCE_PERIOD_MISMATCH",
                    f"Evidence {item.evidence_id} has an incompatible fiscal year",
                )
            )
        evidence_currencies = _currency_codes(item.currency) or _currency_codes(
            item.unit
        )
        target_currencies = _currency_codes(input_value.unit) or _currency_codes(
            assumption.unit
        )
        if (
            evidence_currencies
            and target_currencies
            and evidence_currencies.isdisjoint(target_currencies)
        ):
            issues.append(
                (
                    "EVIDENCE_CURRENCY_MISMATCH",
                    f"Evidence {item.evidence_id} has incompatible currency",
                )
            )
        if (
            item.category != "MARKET"
            and item.unit
            and assumption.unit
            and item.unit.casefold() not in {"unit", "unspecified"}
            and assumption.unit.casefold() not in {"unit", "unspecified"}
            and not operating_units_compatible(item.unit, assumption.unit)
        ):
            issues.append(
                (
                    "EVIDENCE_UNIT_MISMATCH",
                    f"Evidence {item.evidence_id} has an incompatible unit",
                )
            )
        return issues

    @staticmethod
    def _target_evidence_issues(
        item: EvidenceCatalogItem,
        assumption: ReasonedForecastAssumption,
    ) -> list[tuple[str, str]]:
        target = _metric_key(
            assumption.driver_id
            if assumption.target_type == "operating_driver"
            else assumption.metric
        )
        evidence_metric = _metric_key(item.metric or item.context_map.get("metric"))
        evidence_driver = _metric_key(item.driver_id)
        payload = item.payload_type.casefold()
        if item.category in {"OP", "MANUAL"}:
            if "operatingdriverdefinition" in payload:
                required = {
                    entry.strip().casefold()
                    for entry in item.context_map.get("required_inputs", "").split(",")
                    if entry.strip()
                }
                if target not in required:
                    return [
                        (
                            "EVIDENCE_TARGET_MISMATCH",
                            f"Evidence {item.evidence_id} does not support target {target}",
                        )
                    ]
            elif (
                evidence_driver
                and evidence_driver != target
                and evidence_metric != target
            ):
                return [
                    (
                        "EVIDENCE_TARGET_MISMATCH",
                        f"Evidence {item.evidence_id} is for {evidence_driver}, not {target}",
                    )
                ]
            elif not evidence_driver and not evidence_metric:
                return [
                    (
                        "EVIDENCE_TARGET_MISMATCH",
                        f"Evidence {item.evidence_id} has no target metric",
                    )
                ]
            return []
        if item.category == "HIST":
            return (
                []
                if evidence_metric == target
                else [
                    (
                        "EVIDENCE_TARGET_MISMATCH",
                        f"Historical evidence {item.evidence_id} is for {evidence_metric}, not {target}",
                    )
                ]
            )
        if item.category == "MGMT":
            management_metric = {
                "revenue_growth": "growth",
                "operating_income": "ebit",
                "effective_tax_rate": "tax_rate",
            }.get(evidence_metric, evidence_metric)
            return (
                []
                if management_metric == target
                else [
                    (
                        "EVIDENCE_TARGET_MISMATCH",
                        f"Management evidence {item.evidence_id} is for {management_metric}, not {target}",
                    )
                ]
            )
        if item.category == "MARKET":
            if "pricing" in payload:
                compatible = {"price", "arpu"}
            elif "productioncapacity" in payload or "capacity" in payload:
                compatible = {"capacity"}
            elif "marketgrowth" in payload or evidence_metric == "market_growth":
                compatible = {"growth", "growth_rate", "volume"}
            elif "marketshare" in payload or evidence_metric == "market_share":
                compatible = {"growth", "growth_rate", "volume"}
            elif "marketsize" in payload or evidence_metric == "market_size":
                compatible = {"volume", "capacity", "backlog"}
            else:
                compatible = set()
            return (
                []
                if target in compatible
                else [
                    (
                        "EVIDENCE_TARGET_MISMATCH",
                        f"Market evidence {item.evidence_id} is not compatible with {target}",
                    )
                ]
            )
        return [
            (
                "EVIDENCE_TARGET_MISMATCH",
                f"Evidence {item.evidence_id} is not compatible with {target}",
            )
        ]

    @staticmethod
    def _sanity_issues(assumption: ReasonedForecastAssumption) -> list[tuple[str, str]]:
        metric = _metric_key(assumption.metric)
        issues: list[tuple[str, str]] = []
        values = (*assumption.low, *assumption.base, *assumption.high)
        if any(not item.is_finite() for item in values):
            issues.append(("NONFINITE", "Assumption contains a non-finite value"))
        if metric in {"gross_margin"} and any(
            item < -100 or item > 100 for item in values
        ):
            issues.append(
                (
                    "MARGIN_UNSANE",
                    "Gross margin must be between -100 and 100 percentage points",
                )
            )
        if metric == "tax_rate" and any(item < 0 or item > 100 for item in values):
            issues.append(
                ("RATE_UNSANE", "Tax rate must be between 0 and 100 percentage points")
            )
        ratio_metric = metric in {
            "depreciation_to_revenue",
            "capex_to_revenue",
            "operating_working_capital_to_revenue",
        } or (
            metric
            in {
                "r_and_d",
                "sg_and_a",
                "depreciation_and_amortization",
                "capex",
                "operating_working_capital",
            }
            and assumption.basis.value == "percent_of_revenue"
        )
        if ratio_metric and (
            any(item < -500 or item > 500 for item in values)
            if metric
            in {"operating_working_capital", "operating_working_capital_to_revenue"}
            else any(item < 0 or item > 500 for item in values)
        ):
            issues.append(
                (
                    "RATIO_UNSANE",
                    f"{metric} ratio is outside its supported percentage bounds",
                )
            )
        if metric in {"r_and_d", "sg_and_a"} and any(item < 0 for item in values):
            issues.append(("AMOUNT_UNSANE", f"{metric} values cannot be negative"))
        if metric == "revenue" and any(item <= 0 for item in values):
            issues.append(("REVENUE_UNSANE", "Revenue assumptions must be positive"))
        if (
            metric
            in {
                "r_and_d",
                "sg_and_a",
                "depreciation_and_amortization",
                "capex",
                "operating_working_capital",
            }
            and assumption.basis.value == "absolute"
            and any(item < 0 for item in values)
        ):
            issues.append(
                ("AMOUNT_UNSANE", f"{metric} absolute values cannot be negative")
            )
        if assumption.target_type == "operating_driver":
            driver = canonical_driver_id(assumption.driver_id)
            style = _unit_style(assumption.unit)
            if driver in {"utilization", "take_rate", "conversion", "conversion_rate"}:
                lower, upper = (0, 100) if style == "percent" else (0, 1)
                if any(item < lower or item > upper for item in values):
                    issues.append(
                        (
                            "DRIVER_RATE_UNSANE",
                            f"{driver} must be between {lower} and {upper} for its declared unit",
                        )
                    )
            if driver in {"growth", "growth_rate"}:
                lower = Decimal("-100") if style == "percent" else Decimal("-1")
                if any(item <= lower for item in values):
                    issues.append(
                        (
                            "DRIVER_GROWTH_UNSANE",
                            f"Growth must be greater than {lower} for its declared unit",
                        )
                    )
            # Counts, price and capacity inputs are non-negative in the
            # registry-backed archetypes. Growth is the sole signed input.
            if driver != "growth" and any(item < 0 for item in values):
                issues.append(
                    (
                        "DRIVER_VALUE_UNSANE",
                        "Operating driver values cannot be negative",
                    )
                )
        return issues

    def _decision_issues(
        self,
        decision: ProposedModelingDecision,
        input_value: ForecastReasoningInput,
    ) -> list[tuple[str, str]]:
        issues: list[tuple[str, str]] = []
        strategy = decision.strategy.casefold().replace("-", "_").replace(" ", "_")
        if strategy not in _SUPPORTED_DECISION_STRATEGIES:
            issues.append(
                (
                    "UNSUPPORTED_DECISION",
                    f"Unsupported modeling decision strategy: {decision.strategy}",
                )
            )
        if decision.scope == ForecastScope.COMPANY and decision.scope_id != "company":
            issues.append(
                ("SCOPE_MISMATCH", "Company decisions must use scope_id='company'")
            )
        if decision.scope == ForecastScope.SEGMENT and decision.scope_id not in {
            item.segment_id for item in input_value.segments
        }:
            issues.append(
                ("UNKNOWN_SEGMENT", f"Unknown decision segment: {decision.scope_id}")
            )
        metric = _metric_key(decision.metric)
        if metric in _DERIVED_METRICS or metric == "revenue":
            issues.append(
                (
                    "UNSAFE_DECISION_TARGET",
                    f"Unsafe decision target is forbidden: {metric}",
                )
            )
        if decision.driver_id and canonical_driver_id(
            decision.driver_id
        ) in _DERIVED_METRICS | {
            "revenue",
            "segment_revenue",
        }:
            issues.append(
                (
                    "UNSAFE_DECISION_TARGET",
                    f"Unsafe decision driver is forbidden: {decision.driver_id}",
                )
            )
        if metric and metric not in _FINANCIAL_METRICS:
            issues.append(
                (
                    "UNSUPPORTED_METRIC",
                    f"Unsupported decision metric: {decision.metric}",
                )
            )
        if decision.driver_id:
            definitions = [
                item
                for item in input_value.definitions
                if item.segment_id == decision.scope_id
            ]
            decision_driver = canonical_driver_id(decision.driver_id)
            if decision.scope != ForecastScope.SEGMENT or not any(
                decision_driver in item.required_inputs for item in definitions
            ):
                issues.append(
                    (
                        "UNKNOWN_DRIVER",
                        f"Decision driver is not a required input of an accepted definition: {decision.driver_id}",
                    )
                )
        if (
            decision.fiscal_years
            and decision.fiscal_years != input_value.forecast_years
        ):
            issues.append(
                ("HORIZON_MISMATCH", "Decision fiscal years do not match input horizon")
            )
        if strategy not in _NON_PATH_DECISION_STRATEGIES:
            issues.append(
                (
                    "DECISION_REQUIRES_ASSUMPTION",
                    "Numeric/path strategies must be represented by a ReasonedForecastAssumption",
                )
            )
        return _unique_issues(issues)

    @staticmethod
    def _consensus_warnings(
        assumptions: tuple[ReasonedForecastAssumption, ...], catalog: EvidenceCatalog
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        for assumption in assumptions:
            for evidence_id in assumption.evidence_ids:
                item = catalog.get(evidence_id)
                dispersion = getattr(item, "dispersion", None) if item else None
                if dispersion is not None and dispersion > 0:
                    warnings.append(
                        f"{evidence_id}: consensus dispersion={dispersion}; widen uncertainty and lower confidence"
                    )
        return tuple(dict.fromkeys(warnings))


def _metric_key(value: Any) -> str:
    if value is None:
        return ""
    normalized = (
        str(getattr(value, "value", value))
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )
    try:
        from edgarito.config.operating import OPERATING_VOCABULARY

        normalized = OPERATING_VOCABULARY.metric_aliases.get(normalized, normalized)
    except (ImportError, AttributeError):
        pass
    return {
        "research_and_development": "r_and_d",
        "research_and_development_expense": "r_and_d",
        "selling_general_and_administrative": "sg_and_a",
        "selling_general_and_administrative_expense": "sg_and_a",
        "depreciation": "depreciation_and_amortization",
        "capital_expenditures": "capex",
        "capital_expenditure": "capex",
        "owc": "operating_working_capital",
        "effective_tax_rate": "tax_rate",
        "tax_rate_percent": "tax_rate",
        "gross_margin_percent": "gross_margin",
        "gross_margin_percentage": "gross_margin",
        "research_development": "r_and_d",
        "sg_and_a_expense": "sg_and_a",
    }.get(normalized, normalized)


def _is_rate_unit(value: str) -> bool:
    return _is_explicit_percentage_unit(value)


def _is_explicit_percentage_unit(value: str) -> bool:
    normalized = value.casefold().replace("_", " ").strip()
    return (
        normalized in {item.replace("_", " ") for item in _RATE_UNITS}
        or "%" in normalized
        or "percent" in normalized
        or "percentage" in normalized
    )


def _unit_style(value: str) -> str:
    normalized = value.casefold().replace("-", "_").replace(" ", "_").strip()
    if _is_explicit_percentage_unit(value):
        return "percent"
    if normalized in _FRACTION_UNITS:
        return normalized
    if normalized in {"rate", "per_rate"}:
        return "ambiguous_rate"
    return "dimension"


def _currency_codes(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(
        re.findall(r"(?<![A-Za-z])([A-Za-z]{3})(?![A-Za-z])", value.upper())
    )


def _unique_issues(issues: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return list(dict.fromkeys(issues))


DeterministicForecastReasoningValidator = ForecastReasoningValidator
ForecastReasoningPostValidator = ForecastReasoningValidator


__all__ = [
    "ForecastReasoningValidationResult",
    "ForecastReasoningValidator",
    "DeterministicForecastReasoningValidator",
    "ForecastReasoningPostValidator",
]
