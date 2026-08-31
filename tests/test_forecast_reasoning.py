"""Frozen fake-provider coverage for opt-in ForecastReasoner v1."""

import asyncio
import datetime
import subprocess
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest

from edgarito.schemas.forecasting import ForecastOverride
from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.services.forecasting.reasoning import (
    ForecastReasoner,
    ForecastReasoningCache,
    ForecastReasoningCompiler,
    ForecastReasoningInput,
    ForecastReasoningInputValidationError,
    ForecastReasoningResponse,
    ForecastReasoningValidator,
    ProposedModelingDecision,
    ReasonedDriverBasedForecastService,
    ReasonedForecastAssumption,
    build_evidence_catalog,
    build_reasoning_content,
    build_reasoning_prompt,
)
from edgarito.services.forecasting.reasoning import reasoner as reasoner_module
from edgarito.services.operating._forecast.selection import (
    _find_input_observation,
    _select_observations,
)
from edgarito.services.operating.registry import FORMULA_REGISTRY
from edgarito.services.research.consensus import reconcile_evidence
from edgarito.services.research.contracts import (
    MarketGrowthEvidence,
    MarketSizeEvidence,
)

D = Decimal
LOW_PATH = (D("1"), D("1"))
BASE_PATH = (D("2"), D("2"))
HIGH_PATH = (D("3"), D("3"))


def _definition(segment_id="cloud", archetype=OperatingArchetype.VOLUME_PRICE):
    inputs = {
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
    }[archetype]
    units = {
        item: "percent"
        if item in {"utilization", "take_rate", "conversion_rate", "growth"}
        else "units"
        for item in inputs
    }
    if "price" in units:
        units["price"] = "USD/unit"
    if "arpu" in units:
        units["arpu"] = "USD/user"
    if "sales_per_store" in units:
        units["sales_per_store"] = "USD/store"
    return OperatingDriverDefinition(
        driver_id=f"{segment_id}-revenue",
        archetype=archetype,
        segment_id=segment_id,
        output_metric="revenue",
        input_metrics=inputs,
        units=units,
        formula_id=archetype.value,
        required_inputs=inputs,
    )


def _input(
    *,
    manual_overrides=(),
    manual_forward=(),
    definitions=None,
    observations=(),
    research_evidence=(),
):
    return ForecastReasoningInput(
        company_id="fixture-company",
        company_name="Fixture Company",
        ticker="FIX",
        unit="USD",
        as_of=datetime.date(2025, 1, 1),
        forecast_years=(2025, 2026),
        segments=(OperatingSegment(segment_id="cloud", name="Cloud", currency="USD"),),
        definitions=(definitions or (_definition(),)),
        observations=observations,
        research_evidence=research_evidence,
        manual_overrides=manual_overrides,
        manual_forward_driver_observations=manual_forward,
    )


def _assumption(
    assumption_id="a",
    *,
    driver_id=None,
    metric=None,
    unit="units",
    basis="absolute",
    low=LOW_PATH,
    base=BASE_PATH,
    high=HIGH_PATH,
    evidence_based=False,
    model_assumption=True,
    method="volume_price",
    confidence="medium",
    scope="segment",
    scope_id="cloud",
):
    return ReasonedForecastAssumption(
        assumption_id=assumption_id,
        scope=scope,
        scope_id=scope_id,
        target_type="operating_driver" if driver_id else "forecast_metric",
        driver_id=driver_id,
        metric=metric,
        unit=unit,
        basis=basis,
        fiscal_years=(2025, 2026),
        low=low,
        base=base,
        high=high,
        method=method,
        rationale="Frozen test assumption",
        confidence=confidence,
        evidence_based=evidence_based,
        model_assumption=model_assumption,
    )


def test_catalog_and_prompt_are_compact_stable_and_citable():
    value = _input(
        manual_forward=(
            OperatingDriverObservation(
                segment_id="cloud",
                driver_id="volume",
                fiscal_year=2025,
                value=2,
                unit="units",
                origin="first_party_observation",
                confidence="high",
                evidence={
                    "provider": "sec",
                    "supporting_text": "x" * 10_000,
                },
            ),
        )
    )
    reordered = value.model_copy(
        update={
            "segments": tuple(reversed(value.segments)),
            "definitions": tuple(reversed(value.definitions)),
        }
    )
    left = build_evidence_catalog(value)
    right = build_evidence_catalog(reordered)
    assert left.bundle_hash == right.bundle_hash
    assert left.evidence_ids == right.evidence_ids
    content = build_reasoning_content(value, left)
    assert "x" * 1000 not in content
    prompt = build_reasoning_prompt().casefold()
    assert "never browse" in prompt
    assert "fcff" in prompt and "gross profit" in prompt


def test_strict_shape_and_post_validation_reject_unsafe_targets():
    with pytest.raises(ValueError):
        _assumption(low=(1,), base=(1,), high=(1,))
    value = _input()
    response = ForecastReasoningResponse(
        assumptions=(
            _assumption(driver_id="not_required"),
            _assumption(
                assumption_id="derived",
                metric="fcff",
                scope="company",
                scope_id="company",
                unit="USD",
                method="explicit",
            ),
        )
    )
    result = ForecastReasoningValidator().validate(response, value)
    assert {item.code for item in result.rejected_assumptions} >= {
        "DRIVER_NOT_REQUIRED",
        "DERIVED_TARGET",
    }


def test_manual_gross_margin_and_driver_inputs_have_precedence():
    manual_override = ForecastOverride(
        scope="company",
        metric="gross_margin",
        strategy="explicit",
        explicit_path=(D("55"), D("55")),
        basis="percentage_points",
    )
    manual_driver = OperatingDriverObservation(
        segment_id="cloud",
        driver_id="volume",
        fiscal_year=2025,
        value=D("99"),
        unit="units",
        origin="first_party_observation",
        confidence="high",
    )
    value = _input(manual_overrides=(manual_override,), manual_forward=(manual_driver,))
    response = ForecastReasoningResponse(
        assumptions=(
            _assumption(driver_id="volume"),
            _assumption(
                assumption_id="gm",
                metric="gross_margin",
                scope="company",
                scope_id="company",
                unit="percent",
                basis="percentage_points",
                method="explicit",
            ),
        )
    )
    validation = ForecastReasoningValidator().validate(response, value)
    compilation = ForecastReasoningCompiler().compile(value, validation)
    assert compilation.collisions
    assert {item.code for item in compilation.collisions} == {
        "MANUAL_DRIVER_PRECEDENCE",
        "MANUAL_OVERRIDE_PRECEDENCE",
    }
    assert all(item.fiscal_year != 2025 for item in compilation.observations)
    assert compilation.overrides[0].explicit_path == (D("55"), D("55"))
    assert compilation.plan.requested.value == "driver_based"


@pytest.mark.parametrize(
    ("manual_metric", "manual_strategy", "manual_basis", "ai_metric", "ai_basis"),
    [
        (
            "capex_to_revenue",
            "ratio",
            "percent_of_revenue",
            "capex",
            "absolute",
        ),
        ("capex", "explicit", "absolute", "capex_to_revenue", "percent_of_revenue"),
    ],
)
def test_manual_capex_representation_has_precedence_across_target_aliases(
    manual_metric, manual_strategy, manual_basis, ai_metric, ai_basis
):
    manual = ForecastOverride(
        scope="company",
        metric=manual_metric,
        strategy=manual_strategy,
        explicit_path=(D("10"), D("10")),
        basis=manual_basis,
    )
    ai = _assumption(
        assumption_id="ai-capex",
        metric=ai_metric,
        unit="percent" if ai_basis == "percent_of_revenue" else "USD",
        basis=ai_basis,
        method="ratio" if ai_basis == "percent_of_revenue" else "explicit",
        scope="company",
        scope_id="company",
    )
    value = _input(manual_overrides=(manual,))
    validation = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(ai,)), value
    )

    compilation = ForecastReasoningCompiler().compile(value, validation)

    assert compilation.collisions[0].code == "MANUAL_OVERRIDE_PRECEDENCE"
    assert compilation.overrides == (manual,)
    assert compilation.plan.overrides == (manual,)
    assert compilation.overrides[0].strategy.value == manual_strategy
    assert compilation.overrides[0].basis.value == manual_basis


@pytest.mark.parametrize(
    ("metric", "unit", "basis", "strategy_metric"),
    [
        ("gross_margin", "percent", "percentage_points", "gross_margin"),
        ("r_and_d", "USD", "absolute", "r_and_d"),
        ("r_and_d", "percent", "percent_of_revenue", "r_and_d"),
        ("sg_and_a", "USD", "absolute", "sg_and_a"),
        ("other_operating_items", "USD", "absolute", "other_operating_items"),
        ("tax_rate", "percent", "percentage_points", "tax_rate"),
        (
            "depreciation_and_amortization",
            "USD",
            "absolute",
            "depreciation_and_amortization",
        ),
        (
            "depreciation_and_amortization",
            "percent",
            "percent_of_revenue",
            "depreciation_to_revenue",
        ),
        ("capex", "USD", "absolute", "capex"),
        ("capex", "percent", "percent_of_revenue", "capex_to_revenue"),
        ("operating_working_capital", "USD", "absolute", "operating_working_capital"),
        (
            "operating_working_capital",
            "percent",
            "percent_of_revenue",
            "operating_working_capital_to_revenue",
        ),
    ],
)
def test_supported_financial_targets_compile_to_existing_strategies(
    metric, unit, basis, strategy_metric
):
    value = _input()
    assumption = _assumption(
        metric=metric,
        scope="company",
        scope_id="company",
        unit=unit,
        basis=basis,
        method="ratio" if basis == "percent_of_revenue" else "explicit",
    )
    validation = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(assumption,)), value
    )
    assert validation.is_valid
    compilation = ForecastReasoningCompiler().compile(value, validation)
    override = compilation.overrides[0]
    assert override.metric.value == strategy_metric
    assert override.strategy.value == (
        "ratio" if basis == "percent_of_revenue" else "explicit"
    )
    assert override.basis.value == basis


def test_frozen_fake_openai_cache_identity_and_model_invalidation(
    tmp_path, monkeypatch
):
    value = _input()

    class FakeOpenAI:
        reasoning_effort = "low"

        def __init__(self, model):
            self.model = model
            self.calls = 0

        async def extract_structured(self, **kwargs):
            self.calls += 1
            assert kwargs["response_model"] is ForecastReasoningResponse
            return ForecastReasoningResponse()

    async def run():
        fake = FakeOpenAI("fake-a")
        cache = ForecastReasoningCache(tmp_path)
        first = ForecastReasoner(fake, cache=cache)
        await first.reason(value)
        await first.reason(value)
        assert fake.calls == 1
        second = ForecastReasoner(FakeOpenAI("fake-b"), cache=cache)
        await second.reason(value)
        assert second.client.calls == 1
        monkeypatch.setattr(reasoner_module, "PROMPT_VERSION", "changed-prompt")
        await ForecastReasoner(fake, cache=cache).reason(value)
        assert fake.calls == 2
        changed_evidence = value.model_copy(
            update={
                "observations": (
                    OperatingDriverObservation(
                        segment_id="cloud",
                        driver_id="volume",
                        fiscal_year=2025,
                        value=D("10"),
                        unit="units",
                        origin="first_party_observation",
                        confidence="medium",
                    ),
                )
            }
        )
        await ForecastReasoner(fake, cache=cache).reason(changed_evidence)
        assert fake.calls == 3
        monkeypatch.setattr(reasoner_module, "SCHEMA_VERSION", "changed-schema")
        await ForecastReasoner(fake, cache=cache).reason(value)
        assert fake.calls == 4

    asyncio.run(run())


def test_async_service_executes_deterministic_driver_path(tmp_path):
    from test_driver_based_fcff import _financials, _fixture

    segments, definitions, _ = _fixture()
    value = ForecastReasoningInput.from_artifacts(
        _financials(),
        as_of=datetime.date(2025, 1, 1),
        forecast_years=(2025, 2026),
        segments=segments,
        definitions=definitions,
    )
    assumptions = []
    for segment_id in ("cloud", "hardware"):
        assumptions.extend(
            (
                _assumption(
                    f"{segment_id}-volume",
                    driver_id="volume",
                    unit="units",
                    low=(70, 80),
                    base=(70, 80),
                    high=(70, 80),
                    scope_id=segment_id,
                ),
                _assumption(
                    f"{segment_id}-price",
                    driver_id="price",
                    unit="USD/unit",
                    low=(2, 2),
                    base=(2, 2),
                    high=(2, 2),
                    scope_id=segment_id,
                ),
            )
        )
    for metric, unit, basis, path in (
        ("gross_margin", "percent", "percentage_points", (60, 60)),
        ("r_and_d", "USD", "absolute", (20, 20)),
        ("sg_and_a", "USD", "absolute", (30, 30)),
        ("other_operating_items", "USD", "absolute", (0, 0)),
        ("tax_rate", "percent", "percentage_points", (20, 20)),
        ("depreciation_and_amortization", "USD", "absolute", (14, 16)),
        ("capex", "USD", "absolute", (18, 20)),
        ("operating_working_capital", "USD", "absolute", (42, 45)),
    ):
        assumptions.append(
            _assumption(
                metric,
                metric=metric,
                unit=unit,
                basis=basis,
                low=path,
                base=path,
                high=path,
                scope="company",
                scope_id="company",
                method="explicit",
            )
        )

    class FakeOpenAI:
        model = "frozen-fake"
        reasoning_effort = "low"

        async def extract_structured(self, **_kwargs):
            return ForecastReasoningResponse(assumptions=tuple(assumptions))

    async def run():
        return await ReasonedDriverBasedForecastService(
            reasoner=ForecastReasoner(
                FakeOpenAI(), cache=ForecastReasoningCache(tmp_path)
            )
        ).forecast(_financials(), value)

    result = asyncio.run(run())
    assert result.forecast.method == "driver_based_fcff"
    assert result.reasoning.compiled_plan.requested.value == "driver_based"
    assert result.reasoning.validation is not None
    assert all(
        item.origin == "model_assumption"
        for item in result.reasoning.compiled_observations
    )


@pytest.mark.parametrize(
    ("archetype", "driver_id"),
    [
        (OperatingArchetype.VOLUME_PRICE, "volume"),
        (OperatingArchetype.SUBSCRIBERS_ARPU, "subscribers"),
        (OperatingArchetype.CAPACITY_UTILIZATION_PRICE, "utilization"),
        (OperatingArchetype.TRANSACTIONS_TAKE_RATE, "take_rate"),
        (OperatingArchetype.BACKLOG_CONVERSION, "conversion_rate"),
        (OperatingArchetype.STORE_COUNT_SALES_PER_STORE, "store_count"),
        (OperatingArchetype.GENERIC_SEGMENT_GROWTH, "growth"),
    ],
)
def test_every_registry_archetype_accepts_only_its_canonical_inputs(
    archetype, driver_id
):
    definition = _definition(archetype=archetype)
    value = _input(definitions=(definition,))
    unit = definition.units[driver_id]
    assumption = _assumption(
        driver_id=driver_id,
        unit=unit,
        method=archetype.value,
        low=(D("1"), D("1")),
        base=(D("2"), D("2")),
        high=(D("3"), D("3")),
    )
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(assumption,)), value
    )
    assert result.is_valid


def test_malformed_definition_and_unsupported_formula_are_rejected():
    raw = _definition().model_dump()
    raw.update(
        input_metrics=("volume", "foo"),
        required_inputs=("volume", "foo"),
        units={"volume": "units", "foo": "units"},
        formula_id="invented_formula",
    )
    value = _input(definitions=(OperatingDriverDefinition.model_validate(raw),))
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(_assumption(driver_id="volume"),)), value
    )
    assert not result.is_valid
    assert {issue.code for issue in result.rejected_assumptions} >= {
        "MALFORMED_DEFINITION_INPUTS",
        "MALFORMED_REQUIRED_INPUTS",
        "UNSUPPORTED_FORMULA",
    }


def test_percent_and_fraction_driver_units_preserve_scale():
    definition = _definition(archetype=OperatingArchetype.CAPACITY_UTILIZATION_PRICE)
    value = _input(definitions=(definition,))
    percent = _assumption(
        driver_id="utilization",
        unit="percent",
        method="capacity_utilization_price",
    )
    fraction = percent.model_copy(update={"assumption_id": "fraction", "unit": "ratio"})
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(fraction,)), value
    )
    assert any(
        issue.code == "UNIT_SCALE_MISMATCH" for issue in result.rejected_assumptions
    )
    assert FORMULA_REGISTRY.evaluate(
        "capacity_utilization_price",
        {"capacity": D("100"), "utilization": D("20"), "price": D("2")},
        units={"utilization": "percent"},
    ) == D("40")
    assert FORMULA_REGISTRY.evaluate(
        "capacity_utilization_price",
        {"capacity": D("100"), "utilization": D("0.20"), "price": D("2")},
        units={"utilization": "fraction"},
    ) == D("40")


@pytest.mark.parametrize(
    ("metric", "basis"),
    [
        ("gross_margin", "percentage_points"),
        ("tax_rate", "percentage_points"),
        ("capex", "percent_of_revenue"),
    ],
)
def test_forecast_rate_metrics_reject_ambiguous_ratio_units(metric, basis):
    assumption = _assumption(
        metric=metric,
        scope="company",
        scope_id="company",
        unit="ratio",
        basis=basis,
        method="ratio",
    )
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(assumption,)), _input()
    )
    assert any(
        issue.code == "UNIT_SCALE_MISMATCH" for issue in result.rejected_assumptions
    )


def test_modeling_decisions_cannot_carry_numeric_paths_or_bypass_safety():
    with pytest.raises(ValueError):
        ProposedModelingDecision(
            decision_id="bad",
            scope="company",
            metric="capex",
            strategy="explicit",
            explicit_path=(1, 2),
            rationale="bad",
        )
    response = ForecastReasoningResponse(
        modeling_decisions=(
            ProposedModelingDecision(
                decision_id="fcff",
                scope="company",
                metric="fcff",
                strategy="driver",
                rationale="bad",
            ),
            ProposedModelingDecision(
                decision_id="capex",
                scope="company",
                metric="capex",
                strategy="explicit",
                rationale="requires an assumption",
            ),
        )
    )
    result = ForecastReasoningValidator().validate(response, _input())
    assert {issue.code for issue in result.rejected_decisions} >= {
        "UNSAFE_DECISION_TARGET",
        "DECISION_REQUIRES_ASSUMPTION",
    }
    compilation = ForecastReasoningCompiler().compile(_input(), result)
    assert not compilation.overrides


def test_modeling_decisions_are_not_merged_into_executable_plan():
    manual = ForecastOverride(
        scope="company",
        metric="capex",
        strategy="explicit",
        explicit_path=(D("10"), D("10")),
        basis="absolute",
    )
    decision = ProposedModelingDecision(
        decision_id="ignore-capex",
        scope="company",
        metric="capex",
        strategy="ignore",
        rationale="audit only",
    )
    value = _input(manual_overrides=(manual,))
    validation = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(modeling_decisions=(decision,)), value
    )
    assert validation.accepted_decisions == (decision,)
    compilation = ForecastReasoningCompiler().compile(value, validation)
    assert compilation.overrides[0].metric.value == "capex"
    assert compilation.overrides[0].explicit_path == (D("10"), D("10"))
    assert "not executed" in " ".join(compilation.warnings)


def test_invalid_supplied_definition_is_an_execution_gate(tmp_path):
    from test_driver_based_fcff import _financials

    raw = _definition().model_dump()
    raw.update(
        input_metrics=("volume", "foo"),
        required_inputs=("volume", "foo"),
        units={"volume": "units", "foo": "units"},
    )
    value = _input(definitions=(OperatingDriverDefinition.model_validate(raw),))

    class FakeOpenAI:
        model = "gate-test"
        reasoning_effort = "low"

        async def extract_structured(self, **_kwargs):
            return ForecastReasoningResponse()

    class DriverThatMustNotRun:
        def __init__(self):
            self.calls = 0

        def forecast(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(warnings=(), forecast=None, validation=None)

    driver = DriverThatMustNotRun()
    invalid_validation = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(), value
    )
    with pytest.raises(ForecastReasoningInputValidationError):
        ForecastReasoningCompiler().compile(value, invalid_validation)

    async def run():
        return await ReasonedDriverBasedForecastService(
            reasoner=ForecastReasoner(
                FakeOpenAI(), cache=ForecastReasoningCache(tmp_path)
            ),
            driver_service=driver,
        ).forecast(_financials(), value)

    with pytest.raises(ForecastReasoningInputValidationError) as error:
        asyncio.run(run())
    assert error.value.code == "INVALID_FORECAST_REASONING_INPUT"
    assert error.value.issues
    assert driver.calls == 0


@pytest.mark.parametrize("authoritative_driver", ["subscriber_count", "transactions"])
def test_executor_driver_aliases_block_canonical_reasoned_inputs(authoritative_driver):
    if authoritative_driver == "subscriber_count":
        definition = _definition(archetype=OperatingArchetype.SUBSCRIBERS_ARPU)
        canonical = "subscribers"
        unit = "units"
    else:
        definition = _definition(archetype=OperatingArchetype.TRANSACTIONS_TAKE_RATE)
        canonical = "transactions"
        unit = "units"
    authoritative = OperatingDriverObservation(
        segment_id="cloud",
        driver_id=authoritative_driver,
        fiscal_year=2025,
        value=D("10"),
        unit=unit,
        origin="first_party_observation",
        confidence="medium",
    )
    value = _input(definitions=(definition,), observations=(authoritative,))
    assumption = _assumption(
        driver_id=canonical,
        unit=unit,
        method=definition.archetype.value,
    )
    validation = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(assumption,)), value
    )
    compilation = ForecastReasoningCompiler().compile(value, validation)
    assert all(item.fiscal_year != 2025 for item in compilation.observations)
    assert compilation.collisions[0].code == "AUTHORITATIVE_DRIVER_PRECEDENCE"


def test_driver_rate_bounds_follow_declared_percent_or_fraction_scale():
    definition = _definition(archetype=OperatingArchetype.CAPACITY_UTILIZATION_PRICE)
    value = _input(definitions=(definition,))
    percent = _assumption(
        driver_id="utilization",
        unit="percent",
        method="capacity_utilization_price",
        low=(0, 0),
        base=(101, 20),
        high=(101, 20),
    )
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(percent,)), value
    )
    assert any(
        issue.code == "DRIVER_RATE_UNSANE" for issue in result.rejected_assumptions
    )

    fraction_definition = _definition(
        archetype=OperatingArchetype.CAPACITY_UTILIZATION_PRICE
    )
    fraction_definition = fraction_definition.model_copy(
        update={
            "units": {
                "capacity": "units",
                "utilization": "fraction",
                "price": "USD/unit",
            }
        }
    )
    fraction_value = _input(definitions=(fraction_definition,))
    fraction = _assumption(
        driver_id="utilization",
        unit="fraction",
        method="capacity_utilization_price",
        low=(0, 0),
        base=(D("1.1"), D("0.2")),
        high=(D("1.1"), D("0.2")),
    )
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(fraction,)), fraction_value
    )
    assert any(
        issue.code == "DRIVER_RATE_UNSANE" for issue in result.rejected_assumptions
    )


@pytest.mark.parametrize(
    ("metric", "basis", "unit", "path"),
    [
        ("r_and_d", "percent_of_revenue", "percent", (0, 501)),
        ("depreciation_and_amortization", "percent_of_revenue", "percent", (0, 501)),
        ("capex", "percent_of_revenue", "percent", (0, 501)),
        ("operating_working_capital", "percent_of_revenue", "percent", (-501, 0)),
    ],
)
def test_compiled_ratio_targets_are_bounded_before_compilation(
    metric, basis, unit, path
):
    assumption = _assumption(
        metric=metric,
        scope="company",
        scope_id="company",
        unit=unit,
        basis=basis,
        method="ratio",
        low=path,
        base=path,
        high=path,
    )
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(assumption,)), _input()
    )
    assert not result.is_valid
    assert any(issue.code == "RATIO_UNSANE" for issue in result.rejected_assumptions)


def test_authoritative_observation_beats_high_confidence_model_assumption():
    reported = OperatingDriverObservation(
        segment_id="cloud",
        driver_id="volume",
        fiscal_year=2025,
        value=99,
        unit="units",
        origin="first_party_observation",
        confidence="medium",
    )
    candidates = (
        reported,
        reported.model_copy(
            update={"value": 1, "origin": "model_assumption", "confidence": "high"}
        ),
    )
    selected = _select_observations(candidates)
    assert selected[("cloud", "volume", 2025)].value == D("99")
    value = _input(observations=(reported,))
    model = _assumption(driver_id="volume", method="volume_price", confidence="high")
    validation = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(model,)), value
    )
    compilation = ForecastReasoningCompiler().compile(value, validation)
    assert all(item.fiscal_year != 2025 for item in compilation.observations)
    assert compilation.collisions[0].code == "AUTHORITATIVE_DRIVER_PRECEDENCE"


def test_legacy_executor_keeps_canonical_first_alias_lookup():
    definition = _definition(archetype=OperatingArchetype.SUBSCRIBERS_ARPU)
    canonical = OperatingDriverObservation(
        segment_id="cloud",
        driver_id="subscribers",
        fiscal_year=2025,
        value=D("10"),
        unit="units",
        origin="first_party_observation",
        confidence="medium",
    )
    alias = canonical.model_copy(
        update={
            "driver_id": "subscriber_count",
            "value": D("99"),
            "confidence": "high",
        }
    )
    selected = _select_observations((canonical, alias))
    resolved = _find_input_observation(
        selected, "cloud", "subscribers", 2025, definition
    )
    assert resolved is not None
    assert resolved.value == D("10")


@pytest.mark.parametrize(
    ("metric", "unit", "basis", "origin"),
    [
        ("capex", "USD", "absolute", "first_party_observation"),
        ("tax_rate", "percent", "percentage_points", "first_party_observation"),
        ("gross_margin", "percent", "percentage_points", "first_party_observation"),
        ("r_and_d", "USD", "absolute", "management_guidance"),
    ],
)
def test_authoritative_metric_observations_block_full_ai_override_path(
    metric, unit, basis, origin
):
    observation = OperatingDriverObservation(
        segment_id="company",
        driver_id=metric,
        fiscal_year=2025,
        value=D("20"),
        unit=unit,
        scope="company",
        origin=origin,
        confidence="medium",
    )
    value = _input(observations=(observation,))
    assumption = _assumption(
        metric=metric,
        scope="company",
        scope_id="company",
        unit=unit,
        basis=basis,
        method="explicit",
        confidence="high",
    )
    validation = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(assumption,)), value
    )
    assert validation.is_valid
    compilation = ForecastReasoningCompiler().compile(value, validation)
    assert not compilation.overrides
    assert compilation.collisions[0].code == "AUTHORITATIVE_METRIC_PRECEDENCE"
    assert compilation.retained_ranges[assumption.assumption_id][0][1] == D("2")


@pytest.mark.parametrize(
    ("authoritative_metric", "ai_metric"),
    [
        ("capex", "capex_to_revenue"),
        ("depreciation_and_amortization", "depreciation_to_revenue"),
        ("operating_working_capital", "operating_working_capital_to_revenue"),
    ],
)
def test_authoritative_economic_amounts_block_ai_ratio_aliases_without_overlap(
    authoritative_metric, ai_metric
):
    observation = OperatingDriverObservation(
        segment_id="company",
        driver_id=authoritative_metric,
        fiscal_year=2025,
        value=D("20"),
        unit="USD",
        scope="company",
        origin="first_party_observation",
        confidence="medium",
    )
    assumption = _assumption(
        assumption_id=f"ai-{ai_metric}",
        metric=ai_metric,
        scope="company",
        scope_id="company",
        unit="percent",
        basis="percent_of_revenue",
        method="ratio",
    )
    value = _input(observations=(observation,))
    validation = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(assumption,)), value
    )

    compilation = ForecastReasoningCompiler().compile(value, validation)

    assert not compilation.overrides
    assert not compilation.plan.overrides
    assert compilation.collisions[0].code == "AUTHORITATIVE_METRIC_PRECEDENCE"


def test_citations_bind_target_scope_and_currency_codes():
    segment_observation = OperatingDriverObservation(
        segment_id="cloud",
        driver_id="volume",
        fiscal_year=2025,
        value=D("10"),
        unit="units",
        origin="first_party_observation",
        confidence="medium",
    )
    value = _input(observations=(segment_observation,))
    catalog = build_evidence_catalog(value)
    volume_id = next(item.evidence_id for item in catalog if item.driver_id == "volume")
    company_tax = _assumption(
        metric="tax_rate",
        scope="company",
        scope_id="company",
        unit="percent",
        basis="percentage_points",
        method="explicit",
        evidence_based=True,
        model_assumption=False,
    ).model_copy(update={"evidence_ids": (volume_id,)})
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(company_tax,)), value, catalog
    )
    assert any(
        issue.code == "EVIDENCE_TARGET_MISMATCH"
        for issue in result.rejected_assumptions
    )

    euro_observation = segment_observation.model_copy(
        update={
            "segment_id": "company",
            "driver_id": "capex",
            "scope": "company",
            "is_total": True,
            "value": D("10"),
            "unit": "EUR millions",
        }
    )
    euro_input = _input(observations=(euro_observation,))
    euro_catalog = build_evidence_catalog(euro_input)
    capex_id = next(
        item.evidence_id for item in euro_catalog if item.driver_id == "capex"
    )
    capex = _assumption(
        metric="capex",
        scope="company",
        scope_id="company",
        unit="USD millions",
        basis="absolute",
        method="explicit",
        evidence_based=True,
        model_assumption=False,
    ).model_copy(update={"evidence_ids": (capex_id,)})
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(capex,)), euro_input, euro_catalog
    )
    assert any(
        issue.code == "EVIDENCE_CURRENCY_MISMATCH"
        for issue in result.rejected_assumptions
    )


def test_market_growth_is_compatible_with_volume_target():
    research = MarketGrowthEvidence(
        evidence_id="growth-evidence",
        source_date=datetime.date(2024, 1, 1),
        source_type="independent_industry_estimate",
        low="2",
        base="4",
        high="6",
        provenance={"source": "industry"},
        unit="percent",
        market="cloud",
    )
    value = _input(research_evidence=(research,))
    assumption = _assumption(
        driver_id="volume",
        evidence_based=True,
        model_assumption=False,
    ).model_copy(update={"evidence_ids": ("growth-evidence",)})
    result = ForecastReasoningValidator().validate(
        ForecastReasoningResponse(assumptions=(assumption,)), value
    )
    assert result.is_valid


def test_future_evidence_is_excluded_before_openai_prompt(tmp_path):
    future = MarketSizeEvidence(
        evidence_id="future-market",
        source_date=datetime.date(2026, 1, 1),
        source_type="analyst_estimate",
        low="100",
        base="200",
        high="300",
        provenance={"source": "future-source"},
        unit="USD",
        market="future-market",
    )
    value = _input()
    value = value.model_copy(update={"research_evidence": (future,)})
    catalog = build_evidence_catalog(value)
    assert all(item.evidence_id != "future-market" for item in catalog.items)
    assert catalog.exclusion("future-market") is not None

    class FakeOpenAI:
        model = "future-test"
        reasoning_effort = "low"

        async def extract_structured(self, **kwargs):
            assert "future-market" not in kwargs["content"]
            assert "future-source" not in kwargs["content"]
            assert "200" not in kwargs["content"]
            return ForecastReasoningResponse()

    async def run():
        await ForecastReasoner(
            FakeOpenAI(), cache=ForecastReasoningCache(tmp_path)
        ).reason(value)

    asyncio.run(run())


def test_reasoned_service_merges_all_additive_arguments_before_prompt_and_execution(
    tmp_path,
):
    from edgarito.services.forecasting.reasoning.service import _merge_additive_input

    base = _input()
    mobile = OperatingSegment(segment_id="mobile", name="Mobile", currency="USD")
    mobile_definition = _definition(segment_id="mobile")
    mobile_observation = OperatingDriverObservation(
        segment_id="mobile",
        driver_id="volume",
        fiscal_year=2025,
        value=5,
        unit="units",
        origin="first_party_observation",
        confidence="medium",
    )
    management_observation = mobile_observation.model_copy(
        update={"origin": "management_guidance", "value": D("6")}
    )
    program = {
        "program_id": "mobile-expansion",
        "name": "Mobile expansion",
        "segment_id": "mobile",
        "unit": "capacity",
        "status": "planned",
    }
    research = MarketGrowthEvidence(
        evidence_id="merged-research",
        source_date=datetime.date(2024, 1, 1),
        source_type="independent_industry_estimate",
        low="2",
        base="4",
        high="6",
        provenance={"source": "industry"},
        unit="percent",
        market="mobile",
    )
    consensus = reconcile_evidence((research,))
    override = ForecastOverride(
        scope="company",
        metric="capex",
        strategy="explicit",
        explicit_path=(D("10"), D("10")),
        basis="absolute",
    )
    forward = mobile_observation.model_copy(update={"value": D("7")})

    merged = _merge_additive_input(
        base,
        segments=(mobile,),
        definitions=(mobile_definition,),
        observations=(mobile_observation,),
        management_guidance=(),
        management_constraints=(management_observation,),
        investment_programs=(program,),
        historical_facts=(),
        research_evidence=(research,),
        evidence_consensus=(consensus,),
        manual_overrides=(override,),
        manual_forward_driver_observations=(forward,),
    )
    assert {item.segment_id for item in merged.segments} == {"cloud", "mobile"}
    assert "merged-research" in build_evidence_catalog(merged).evidence_ids
    assert merged.manual_overrides == (override,)
    assert merged.manual_forward_driver_observations == (forward,)

    with pytest.raises(ValueError, match="Conflicting duplicate segments"):
        _merge_additive_input(
            base,
            segments=(
                OperatingSegment(
                    segment_id="cloud", name="Cloud", currency="EUR", source="other"
                ),
            ),
            definitions=(),
            observations=(),
            management_guidance=(),
            management_constraints=(),
            investment_programs=(),
            historical_facts=(),
            research_evidence=(),
            evidence_consensus=(),
            manual_overrides=(),
            manual_forward_driver_observations=(),
        )


def test_duplicate_explicit_research_ids_and_generated_collisions_are_audited():
    value = _input()
    first = MarketSizeEvidence(
        evidence_id="same-id",
        source_date=datetime.date(2024, 1, 1),
        source_type="analyst_estimate",
        low=1,
        base=2,
        high=3,
        provenance={"source": "one"},
        unit="USD",
        market="one",
    )
    second = first.model_copy(update={"market": "two"})
    value = value.model_copy(update={"research_evidence": (first, second)})
    catalog = build_evidence_catalog(value)
    assert catalog.duplicate_explicit_ids == ("same-id",)
    assert len(catalog.evidence_ids) == len(set(catalog.evidence_ids))
    assert catalog.get("same-id") is None

    generated = build_evidence_catalog(_input()).evidence_ids[0]
    explicit = first.model_copy(update={"evidence_id": generated})
    collided = build_evidence_catalog(
        value.model_copy(update={"research_evidence": (explicit,)})
    )
    assert len(collided.evidence_ids) == len(set(collided.evidence_ids))
    assert generated in collided.evidence_ids
    assert any(item.evidence_id != generated for item in collided.items)


def test_stable_consumer_response_does_not_over_model():
    value = _input()
    response = ForecastReasoningResponse(
        unresolved_items=(
            {
                "item_id": "missing-price",
                "category": "driver_input",
                "reason": "No stable pricing evidence",
            },
        ),
        warnings=("No unsupported model path proposed",),
    )
    validation = ForecastReasoningValidator().validate(response, value)
    compilation = ForecastReasoningCompiler().compile(value, validation)
    assert validation.is_valid
    assert not compilation.observations
    assert not compilation.overrides
    assert compilation.plan.requested.value == "driver_based"


def test_forecasting_public_exports_import_in_a_fresh_process():
    code = (
        "import edgarito.services.forecasting as f; "
        "[getattr(f, name) for name in f.__all__]"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
