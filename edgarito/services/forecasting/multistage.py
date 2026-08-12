from decimal import ROUND_CEILING, Decimal

from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.services.financial_observation_availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting.fcff import FcffForecastService
from edgarito.services.forecasting.models import (
    AdaptiveMultistagePlan,
    FcffForecast,
    FcffForecastDriver,
    FcffForecastParameters,
    ForecastAssumptionSource,
    ForecastSeedType,
    ForwardGrowthEvidence,
    ForwardGrowthOutlook,
)


class AdaptiveMultistageFcffForecastService:
    """Fade FCFF operating drivers into a sustainable perpetual-growth stage."""

    _REINVESTMENT_TOLERANCE = Decimal("1e-12")
    _REINVESTMENT_MAX_ITERATIONS = 50

    def __init__(self, base_service: FcffForecastService | None = None):
        self._base_service = base_service or FcffForecastService()

    def forecast(
        self,
        financials: NormalizedCompanyFinancials,
        seed_forecast: FcffForecast,
        requested_parameters: FcffForecastParameters,
        terminal_growth_rate: Decimal,
        configuration,
        *,
        normalized_tax_rate: Decimal | None = None,
        fixed_plan: AdaptiveMultistagePlan | None = None,
        as_of=None,
        availability_mode: ObservationAvailabilityMode = (
            ObservationAvailabilityMode.POINT_IN_TIME
        ),
        forward_evidence: ForwardGrowthEvidence | None = None,
        forward_growth: ForwardGrowthOutlook | None = None,
        forward_growth_outlook: ForwardGrowthOutlook | None = None,
    ) -> tuple[FcffForecast, AdaptiveMultistagePlan]:
        if not seed_forecast.observations:
            raise ValueError("Adaptive multistage forecasting requires a seed forecast")
        if forward_growth is not None and forward_growth_outlook is not None:
            raise ValueError(
                "Provide only one of forward_growth and forward_growth_outlook"
            )
        resolved_forward_growth = forward_growth or forward_growth_outlook
        if fixed_plan is None:
            if resolved_forward_growth is None:
                resolved_forward_growth = self.resolve_forward_growth(
                    financials,
                    seed_forecast,
                    requested_parameters,
                    terminal_growth_rate,
                    forward_evidence=forward_evidence,
                    convergence_tolerance=configuration.convergence_tolerance,
                    as_of=as_of,
                    availability_mode=availability_mode,
                )
            growth_path, plan = self._growth_path(
                seed_forecast,
                requested_parameters,
                terminal_growth_rate,
                configuration,
                forward_evidence,
                resolved_forward_growth,
            )
        else:
            if len(seed_forecast.observations) != fixed_plan.effective_years:
                raise ValueError(
                    "A fixed multistage plan requires a seed forecast with the "
                    "same effective horizon"
                )
            growth_path = self._fixed_growth_path(
                seed_forecast,
                requested_parameters,
                terminal_growth_rate,
                fixed_plan,
            )
            plan = fixed_plan.model_copy(
                update={
                    "initial_growth_rate": growth_path[0],
                    "terminal_growth_rate": terminal_growth_rate,
                }
            )
        material_capex_shock_indices = self._material_capex_shock_indices(
            financials,
            seed_forecast,
            requested_parameters,
            configuration,
            as_of=as_of,
            availability_mode=availability_mode,
        )
        if (
            material_capex_shock_indices
            and configuration.depreciable_asset_life_years is None
        ):
            raise ValueError(
                "Material absolute CAPEX shock detected; configure "
                "depreciable_asset_life_years so post-shock D&A can be rolled "
                "forward"
            )
        capex_anchor_index = self._capex_anchor_index(
            seed_forecast, requested_parameters
        )
        independent_capex_transition = bool(material_capex_shock_indices)
        if configuration.fade_reinvestment_to_terminal and independent_capex_transition:
            growth_path, plan = self._extend_for_capex_transition(
                growth_path,
                plan,
                requested_parameters,
                seed_forecast,
                configuration,
                terminal_growth_rate,
            )
        values = requested_parameters.model_dump()
        values["forecast_years"] = plan.effective_years
        values["revenue_growth"] = growth_path
        for driver in FcffForecastDriver:
            if driver == FcffForecastDriver.REVENUE_GROWTH:
                continue
            path = getattr(requested_parameters, driver.value)
            values[driver.value] = self._extend_explicit_path(
                path, plan.effective_years
            )
        tax_is_adaptive = (
            requested_parameters.tax_rate is None and normalized_tax_rate is not None
        )
        if tax_is_adaptive:
            values["tax_rate"] = self._converging_path(
                seed_forecast.observations[0].tax_rate,
                normalized_tax_rate,
                plan,
            )

        if configuration.fade_reinvestment_to_terminal:
            values, plan = self._apply_sustainable_reinvestment(
                financials,
                values,
                requested_parameters,
                plan,
                configuration,
                as_of,
                availability_mode,
                force_depreciation_rollforward=bool(material_capex_shock_indices),
                independent_capex_transition=independent_capex_transition,
                capex_anchor_index=capex_anchor_index,
            )
        elif material_capex_shock_indices:
            values = self._apply_shock_depreciation_rollforward(
                financials,
                values,
                configuration.depreciable_asset_life_years,
                as_of,
                availability_mode,
            )
        plan = plan.model_copy(
            update={
                "depreciable_asset_life_years": (
                    configuration.depreciable_asset_life_years
                )
            }
        )

        parameters = FcffForecastParameters.model_validate(values)
        forecast = self._base_service.forecast(
            financials,
            parameters,
            as_of=as_of,
            availability_mode=availability_mode,
        )
        forecast.method = "adaptive_multistage_fcff"
        if (
            requested_parameters.assumption_source_overrides.get(
                FcffForecastDriver.REVENUE_GROWTH
            )
            != ForecastAssumptionSource.MANAGEMENT_GUIDANCE
        ):
            forecast.assumption_sources[FcffForecastDriver.REVENUE_GROWTH] = (
                ForecastAssumptionSource.ADAPTIVE_MULTISTAGE
            )
        if tax_is_adaptive:
            forecast.assumption_sources[FcffForecastDriver.TAX_RATE] = (
                ForecastAssumptionSource.ADAPTIVE_MULTISTAGE
            )
        if configuration.fade_reinvestment_to_terminal:
            forecast.assumption_sources[FcffForecastDriver.CAPEX_TO_REVENUE] = (
                ForecastAssumptionSource.ADAPTIVE_MULTISTAGE
            )
        if configuration.depreciable_asset_life_years is not None and (
            requested_parameters.depreciation_to_revenue is None
            or bool(material_capex_shock_indices)
        ):
            forecast.assumption_sources[FcffForecastDriver.DEPRECIATION_TO_REVENUE] = (
                ForecastAssumptionSource.ADAPTIVE_MULTISTAGE
            )
        forecast.assumption_source_paths = self._source_paths(
            seed_forecast=seed_forecast,
            requested_parameters=requested_parameters,
            plan=plan,
            forward_growth=resolved_forward_growth,
            tax_is_adaptive=tax_is_adaptive,
            reinvestment_is_adaptive=configuration.fade_reinvestment_to_terminal,
            independent_capex_transition=independent_capex_transition,
            depreciation_is_adaptive=(
                configuration.depreciable_asset_life_years is not None
                and (
                    requested_parameters.depreciation_to_revenue is None
                    or bool(material_capex_shock_indices)
                )
            ),
        )
        for driver, source_path in forecast.assumption_source_paths.items():
            if not source_path:
                continue
            forecast.assumption_sources[driver] = (
                ForecastAssumptionSource.ADAPTIVE_MULTISTAGE
                if ForecastAssumptionSource.ADAPTIVE_MULTISTAGE in source_path
                else source_path[0]
            )
        if forecast.capex_constraints_applied:
            capex_path = forecast.assumption_source_paths.get(
                FcffForecastDriver.CAPEX_TO_REVENUE,
                (ForecastAssumptionSource.ADAPTIVE_MULTISTAGE,)
                * len(forecast.observations),
            )
            forecast.assumption_source_paths[FcffForecastDriver.CAPEX_TO_REVENUE] = (
                tuple(
                    ForecastAssumptionSource.MANAGEMENT_GUIDANCE
                    if observation.fiscal_year in forecast.capex_constraints_applied
                    else capex_path[index]
                    for index, observation in enumerate(forecast.observations)
                )
            )
            forecast.assumption_sources[FcffForecastDriver.CAPEX_TO_REVENUE] = (
                ForecastAssumptionSource.MANAGEMENT_GUIDANCE
            )
        forecast.adaptive_stages = self._stage_path(plan)
        if resolved_forward_growth is not None:
            forecast.current_growth_rate = resolved_forward_growth.current_growth
            forecast.warnings = tuple(
                dict.fromkeys((*forecast.warnings, *resolved_forward_growth.warnings))
            )
        return self._base_service.regenerate_cell_audits(forecast), plan

    def resolve_forward_growth(
        self,
        financials: NormalizedCompanyFinancials,
        seed_forecast: FcffForecast,
        requested_parameters: FcffForecastParameters,
        terminal_growth_rate: Decimal,
        *,
        forward_evidence: ForwardGrowthEvidence | None = None,
        convergence_tolerance: Decimal = Decimal("1"),
        as_of=None,
        availability_mode: ObservationAvailabilityMode = (
            ObservationAvailabilityMode.POINT_IN_TIME
        ),
    ) -> ForwardGrowthOutlook:
        """Resolve the forward growth regime without conflating it with YTD growth.

        The order is intentionally explicit: configured paths, quantitative
        management guidance, quantitative forward evidence, normalized annual
        history, and finally the current run-rate.  The last case is retained
        for continuity but is marked low confidence and cannot independently
        establish a stable state.
        """

        current_growth = seed_forecast.current_growth_rate
        if current_growth is None and seed_forecast.observations:
            current_growth = seed_forecast.observations[0].revenue_growth

        history_path = tuple(seed_forecast.normalized_historical_growth_path)
        normalized_historical = seed_forecast.normalized_historical_growth
        if not history_path:
            history_path = self._annual_growth_path(
                financials,
                requested_parameters.historical_window,
                as_of=as_of,
                availability_mode=availability_mode,
            )
        if normalized_historical is None and history_path:
            normalized_historical = sum(history_path, Decimal(0)) / Decimal(
                len(history_path)
            )

        growth_override = requested_parameters.assumption_source_overrides.get(
            FcffForecastDriver.REVENUE_GROWTH
        )
        configured_path = requested_parameters.revenue_growth
        if configured_path is not None:
            source = (
                ForecastAssumptionSource.MANAGEMENT_GUIDANCE.value
                if growth_override == ForecastAssumptionSource.MANAGEMENT_GUIDANCE
                else ForecastAssumptionSource.EXPLICIT.value
            )
            resolved_path = tuple(configured_path)
            if (
                growth_override == ForecastAssumptionSource.MANAGEMENT_GUIDANCE
                and forward_evidence is not None
                and forward_evidence.growth_path
            ):
                # The overlay may carry a full baseline-length path after
                # replacing only one guided year.  Prefer the quantitative
                # guidance records themselves so uninformed years do not get
                # mislabeled as a management-guidance prefix.
                resolved_path = forward_evidence.growth_path
            return self._make_outlook(
                path=resolved_path,
                source=source,
                confidence="high",
                current_growth=current_growth,
                history_path=history_path,
                terminal_growth_rate=terminal_growth_rate,
                convergence_tolerance=convergence_tolerance,
                forward_evidence=forward_evidence,
            )

        anchor_path, anchor_source, anchor_start = self._revenue_anchor_growth_path(
            seed_forecast, requested_parameters
        )
        if anchor_path:
            return self._make_outlook(
                # ``anchor_path`` is aligned to forecast positions through the
                # last explicit anchor.  Keep the leading gap values: dropping
                # them compacts a sparse FY2026/FY2028 pair into positions one
                # and two and starts the adaptive fade too early.
                path=anchor_path,
                anchor=anchor_path[-1],
                source=anchor_source.value,
                confidence="high",
                current_growth=current_growth,
                history_path=history_path,
                terminal_growth_rate=terminal_growth_rate,
                convergence_tolerance=convergence_tolerance,
                forward_evidence=forward_evidence,
            )

        if forward_evidence is not None and (
            forward_evidence.growth_path or forward_evidence.growth_anchor is not None
        ):
            return self._make_outlook(
                path=forward_evidence.growth_path,
                anchor=forward_evidence.growth_anchor,
                source=(
                    forward_evidence.source
                    or ForecastAssumptionSource.FORWARD_EVIDENCE.value
                ),
                confidence=forward_evidence.confidence or "medium",
                current_growth=current_growth,
                history_path=history_path,
                terminal_growth_rate=terminal_growth_rate,
                convergence_tolerance=convergence_tolerance,
                forward_evidence=forward_evidence,
            )

        if normalized_historical is not None:
            historical_confidence = self._historical_growth_confidence(history_path)
            return self._make_outlook(
                anchor=normalized_historical,
                source=ForecastAssumptionSource.NORMALIZED_HISTORICAL.value,
                confidence=historical_confidence,
                current_growth=current_growth,
                history_path=history_path,
                terminal_growth_rate=terminal_growth_rate,
                convergence_tolerance=convergence_tolerance,
                forward_evidence=forward_evidence,
            )

        if current_growth is None:
            raise ValueError(
                "Adaptive multistage forecasting requires current or normalized "
                "revenue growth evidence"
            )
        return self._make_outlook(
            anchor=current_growth,
            source=ForecastAssumptionSource.CURRENT_RUN_RATE.value,
            confidence="low",
            current_growth=current_growth,
            history_path=history_path,
            terminal_growth_rate=terminal_growth_rate,
            convergence_tolerance=convergence_tolerance,
            forward_evidence=forward_evidence,
        )

    @staticmethod
    def _historical_growth_confidence(path: tuple[Decimal, ...]) -> str:
        """Mark short or materially volatile history as weak forward evidence."""

        if len(path) < 3:
            return "low"
        if max(path) - min(path) > Decimal("15"):
            return "low"
        return "medium"

    @classmethod
    def _make_outlook(
        cls,
        *,
        path: tuple[Decimal, ...] = (),
        anchor: Decimal | None = None,
        source: str,
        confidence: str,
        current_growth: Decimal | None,
        history_path: tuple[Decimal, ...],
        terminal_growth_rate: Decimal,
        convergence_tolerance: Decimal,
        forward_evidence: ForwardGrowthEvidence | None,
    ) -> ForwardGrowthOutlook:
        if anchor is None and path:
            anchor = path[-1]
        if anchor is None:
            raise ValueError("Forward growth resolution produced no anchor")
        current_near_terminal = (
            current_growth is not None
            and abs(current_growth - terminal_growth_rate) <= convergence_tolerance
        )
        stable_state_supported = cls._stable_state_supported(
            anchor=anchor,
            path=path,
            history_path=history_path,
            terminal_growth_rate=terminal_growth_rate,
            convergence_tolerance=convergence_tolerance,
            source=source,
            forward_evidence=forward_evidence,
        )
        warnings: tuple[str, ...] = ()
        if (
            source == ForecastAssumptionSource.NORMALIZED_HISTORICAL.value
            and confidence == "low"
        ):
            warnings = (
                "LOW confidence: normalized historical growth is short or volatile; "
                "no management or forward estimate evidence was available",
            )
        if (
            source == ForecastAssumptionSource.CURRENT_RUN_RATE.value
            and not stable_state_supported
        ):
            warnings = (
                "LOW confidence: current run-rate is the only forward-growth "
                "evidence; stable-state eligibility is uncertain",
            )
        return ForwardGrowthOutlook(
            growth_path=path,
            historical_growth_path=history_path,
            management_guidance_path=(
                (
                    forward_evidence.guidance_growth_path
                    or (
                        forward_evidence.growth_path
                        if not forward_evidence.forward_revenue_estimates
                        else ()
                    )
                )
                if forward_evidence is not None and forward_evidence.guidance
                else ()
            ),
            guidance_growth_path_by_year=(
                forward_evidence.guidance_growth_path_by_year
                if forward_evidence is not None
                else ()
            ),
            forward_estimates_path=(
                (
                    forward_evidence.forward_estimate_growth_path
                    or (
                        forward_evidence.growth_path
                        if not forward_evidence.guidance
                        else ()
                    )
                )
                if forward_evidence is not None
                else ()
            ),
            growth_path_by_year=(
                forward_evidence.growth_path_by_year
                if forward_evidence is not None
                else ()
            ),
            forward_revenue_estimates=(
                forward_evidence.forward_revenue_estimates
                if forward_evidence is not None
                else ()
            ),
            forward_estimate_provider=(
                forward_evidence.forward_estimate_provider
                if forward_evidence is not None
                else None
            ),
            forward_estimate_years=(
                forward_evidence.forward_estimate_years
                if forward_evidence is not None
                else ()
            ),
            forward_estimate_growth_path=(
                forward_evidence.forward_estimate_growth_path
                if forward_evidence is not None
                else ()
            ),
            forward_estimate_diagnostics=(
                forward_evidence.forward_estimate_diagnostics
                if forward_evidence is not None
                else ()
            ),
            normalized_growth=anchor,
            source=source,
            confidence=confidence,
            current_growth=current_growth,
            stable_state_supported=stable_state_supported,
            current_growth_near_terminal=current_near_terminal,
            warnings=warnings,
        )

    @staticmethod
    def _stable_state_supported(
        *,
        anchor: Decimal,
        path: tuple[Decimal, ...],
        history_path: tuple[Decimal, ...],
        terminal_growth_rate: Decimal,
        convergence_tolerance: Decimal,
        source: str,
        forward_evidence: ForwardGrowthEvidence | None,
    ) -> bool:
        """Require history/forward support rather than current-rate proximity."""

        if source == ForecastAssumptionSource.CURRENT_RUN_RATE.value:
            return False
        lifecycle = (
            forward_evidence.lifecycle.casefold()
            if forward_evidence is not None
            else "unknown"
        )
        if lifecycle in {
            "growth",
            "unprofitable_growth",
            "pre_revenue",
            "distressed",
        }:
            return False
        positive_forward_signal = forward_evidence is not None and (
            forward_evidence.backlog
            or forward_evidence.guidance
            or forward_evidence.capacity
            or forward_evidence.growth_visibility > Decimal("0")
        )
        if positive_forward_signal:
            return False
        if abs(anchor - terminal_growth_rate) > convergence_tolerance:
            return False

        def stable_values(values: tuple[Decimal, ...]) -> bool:
            return bool(values) and (max(values) - min(values) <= convergence_tolerance)

        # Two or more consecutive annual growth observations provide the
        # minimum historical evidence.  A mature lifecycle classification can
        # supplement a single normalized growth estimate, but never a growth
        # lifecycle or a current run-rate-only fallback.
        historical_support = len(history_path) >= 2 and stable_values(history_path)
        forward_support = len(path) >= 2 and stable_values(path)
        mature_lifecycle_support = lifecycle in {"mature", "declining"} and bool(
            history_path
        )
        return historical_support or forward_support or mature_lifecycle_support

    def _annual_growth_path(
        self,
        financials: NormalizedCompanyFinancials,
        historical_window: int,
        *,
        as_of,
        availability_mode: ObservationAvailabilityMode,
    ) -> tuple[Decimal, ...]:
        if as_of is not None:
            financials = financials.model_copy(
                update={
                    "observations": [
                        item
                        for item in financials.observations
                        if self._base_service._availability_service.is_available(
                            item,
                            as_of=as_of,
                            mode=availability_mode,
                            snapshot_retrieved_at=financials.retrieved_at,
                        )
                    ]
                }
            )
        periods = self._base_service._complete_annual_periods(financials)
        return tuple(
            self._base_service._historical_values(
                FcffForecastDriver.REVENUE_GROWTH,
                periods[-(historical_window + 1) :],
            )
        )

    @staticmethod
    def _revenue_anchor_growth_path(
        seed_forecast: FcffForecast,
        requested_parameters: FcffForecastParameters,
    ) -> tuple[tuple[Decimal, ...], ForecastAssumptionSource, int | None]:
        if not requested_parameters.revenue_anchors:
            return (), ForecastAssumptionSource.EXPLICIT, None
        source_values = requested_parameters.revenue_anchor_sources
        has_management = any(
            value == ForecastAssumptionSource.MANAGEMENT_GUIDANCE
            for value in source_values.values()
        )
        has_explicit = any(
            value == ForecastAssumptionSource.EXPLICIT
            for value in source_values.values()
        )
        has_forward = any(
            value == ForecastAssumptionSource.FORWARD_EVIDENCE
            for value in source_values.values()
        )
        source = (
            ForecastAssumptionSource.MANAGEMENT_GUIDANCE
            if has_management and not has_explicit
            else ForecastAssumptionSource.FORWARD_EVIDENCE
            if has_forward and not has_explicit
            else ForecastAssumptionSource.EXPLICIT
        )
        previous_revenue = seed_forecast.base_revenue
        anchor_indexes = [
            index
            for index, observation in enumerate(seed_forecast.observations)
            if observation.fiscal_year in requested_parameters.revenue_anchors
        ]
        if not anchor_indexes:
            return (), source, None
        last_anchor_index = max(anchor_indexes)
        has_anchor = False
        first_anchor_index = None
        path = []
        for index, observation in enumerate(
            seed_forecast.observations[: last_anchor_index + 1]
        ):
            target = requested_parameters.revenue_anchors.get(observation.fiscal_year)
            if target is not None and previous_revenue > 0:
                path.append((target / previous_revenue - Decimal(1)) * Decimal(100))
                previous_revenue = target
                has_anchor = True
                if first_anchor_index is None:
                    first_anchor_index = index
            elif has_anchor:
                # Keep non-anchor fiscal years as placeholders in the aligned
                # prefix.  Compacting FY2026/FY2028 anchors into a two-value
                # path would incorrectly apply the FY2028 growth to FY2027 and
                # start the adaptive fade before the final explicit anchor.
                path.append(observation.revenue_growth)
                previous_revenue = previous_revenue * (
                    Decimal(1) + observation.revenue_growth / Decimal(100)
                )
            else:
                # Preserve the pre-anchor forecast position as well.  The
                # returned path is indexed like ``seed_forecast.observations``
                # through the final explicit anchor, not like the sparse set
                # of anchors alone.
                path.append(observation.revenue_growth)
                previous_revenue = observation.revenue

        # The prefix is fiscal-year aligned from the first forecast position;
        # an anchor may occur after an unanchored gap, but that gap still has to
        # occupy a position before the adaptive transition.
        return tuple(path), source, first_anchor_index

    @classmethod
    def _fixed_growth_path(
        cls,
        seed_forecast: FcffForecast,
        requested_parameters: FcffForecastParameters,
        terminal_growth_rate: Decimal,
        plan: AdaptiveMultistagePlan,
    ) -> tuple[Decimal, ...]:
        """Change growth economics without changing the selected stage topology.

        A fixed plan is used by decision scenarios and sensitivity cells.  The
        old implementation always copied the seed growth path, which meant a
        scenario's requested revenue-growth stress silently disappeared.  Use
        the requested path when one is supplied, while retaining the base path
        for callers that only vary terminal growth (the sensitivity workflow).
        Stage counts remain those selected for the independently calculated Base
        case, so a stress cannot reverse scenario ordering merely by creating a
        different lifecycle horizon.
        """

        effective_years = plan.effective_years
        stable_years = plan.stable_years
        prefix_years = plan.explicit_growth_prefix_years
        configured = requested_parameters.revenue_growth
        if configured is None:
            prefix = [
                observation.revenue_growth
                for observation in seed_forecast.observations[
                    : effective_years - stable_years
                ]
            ]
            return tuple([*prefix, *([terminal_growth_rate] * stable_years)])

        if len(configured) > 1:
            explicit_prefix = list(configured[:prefix_years])
            initial_growth = explicit_prefix[-1] if explicit_prefix else configured[0]
        else:
            explicit_prefix = [configured[0]] * prefix_years
            initial_growth = configured[0]

        current_growth_years = min(plan.current_growth_years, effective_years)
        remaining = effective_years - current_growth_years - prefix_years
        high_growth_years = min(plan.high_growth_years, remaining)
        remaining -= high_growth_years
        transition_years = min(plan.transition_years, remaining)
        stable_years = remaining - transition_years
        path = [
            *([configured[0]] * current_growth_years),
            *explicit_prefix,
            *([initial_growth] * high_growth_years),
        ]
        path.extend(
            cls._linear_transition(
                initial_growth,
                terminal_growth_rate,
                transition_years,
            )
        )
        path.extend([terminal_growth_rate] * stable_years)
        return tuple(path)

    @classmethod
    def _source_paths(
        cls,
        *,
        seed_forecast: FcffForecast,
        requested_parameters: FcffForecastParameters,
        plan: AdaptiveMultistagePlan,
        forward_growth: ForwardGrowthOutlook | None,
        tax_is_adaptive: bool,
        reinvestment_is_adaptive: bool,
        independent_capex_transition: bool,
        depreciation_is_adaptive: bool,
    ) -> dict[FcffForecastDriver, tuple[ForecastAssumptionSource, ...]]:
        years = plan.effective_years
        paths = {
            driver: tuple(
                cls._base_source(seed_forecast, requested_parameters, driver)
                for _ in range(years)
            )
            for driver in FcffForecastDriver
        }

        growth_base = cls._growth_source(
            seed_forecast,
            requested_parameters,
            plan,
            forward_growth,
        )
        growth_prefix = plan.explicit_growth_prefix_years
        growth_current = (
            (ForecastAssumptionSource.CURRENT_RUN_RATE,)
            if plan.current_growth_years
            else ()
        )
        paths[FcffForecastDriver.REVENUE_GROWTH] = (
            growth_current
            + (growth_base,) * min(growth_prefix, years - len(growth_current))
            + (ForecastAssumptionSource.ADAPTIVE_MULTISTAGE,)
            * max(0, years - len(growth_current) - growth_prefix)
        )

        if tax_is_adaptive:
            paths[FcffForecastDriver.TAX_RATE] = (
                ForecastAssumptionSource.ADAPTIVE_MULTISTAGE,
            ) * years
        if reinvestment_is_adaptive:
            capex_base = cls._base_source(
                seed_forecast,
                requested_parameters,
                FcffForecastDriver.CAPEX_TO_REVENUE,
            )
            if independent_capex_transition:
                capex_prefix = min(
                    cls._capex_anchor_index(seed_forecast, requested_parameters) + 1,
                    years,
                )
                if requested_parameters.capex_to_revenue is not None:
                    capex_prefix = max(
                        capex_prefix,
                        min(len(requested_parameters.capex_to_revenue), years),
                    )
            else:
                capex_path = requested_parameters.capex_to_revenue
                convergence_year = plan.effective_years - plan.stable_years
                if capex_path is not None and len(capex_path) > 1:
                    capex_prefix = min(len(capex_path), convergence_year)
                elif capex_path is not None and capex_base in {
                    ForecastAssumptionSource.EXPLICIT,
                    ForecastAssumptionSource.MANAGEMENT_GUIDANCE,
                }:
                    capex_prefix = min(
                        plan.explicit_growth_prefix_years + plan.high_growth_years,
                        years,
                    )
                else:
                    capex_prefix = 0
            paths[FcffForecastDriver.CAPEX_TO_REVENUE] = cls._prefix_adaptive_path(
                capex_base, capex_prefix, years
            )
        if depreciation_is_adaptive:
            paths[FcffForecastDriver.DEPRECIATION_TO_REVENUE] = (
                ForecastAssumptionSource.ADAPTIVE_MULTISTAGE,
            ) * years
        return paths

    @staticmethod
    def _growth_source(
        seed_forecast: FcffForecast,
        requested_parameters: FcffForecastParameters,
        plan: AdaptiveMultistagePlan,
        forward_growth: ForwardGrowthOutlook | None,
    ) -> ForecastAssumptionSource:
        if forward_growth is not None:
            source_alias = {
                "consensus": ForecastAssumptionSource.FORWARD_EVIDENCE,
                "forward_consensus": ForecastAssumptionSource.FORWARD_EVIDENCE,
                "profile": ForecastAssumptionSource.EXPLICIT,
                "cli": ForecastAssumptionSource.EXPLICIT,
                "management": ForecastAssumptionSource.MANAGEMENT_GUIDANCE,
                "normalized_history": ForecastAssumptionSource.NORMALIZED_HISTORICAL,
                "run_rate": ForecastAssumptionSource.CURRENT_RUN_RATE,
            }.get(forward_growth.source)
            if source_alias is None and "analyst_consensus" in forward_growth.source:
                source_alias = ForecastAssumptionSource.FORWARD_EVIDENCE
            if source_alias is not None:
                return source_alias
            try:
                return ForecastAssumptionSource(forward_growth.source)
            except ValueError:
                pass
        return AdaptiveMultistageFcffForecastService._base_source(
            seed_forecast,
            requested_parameters,
            FcffForecastDriver.REVENUE_GROWTH,
        )

    @staticmethod
    def _base_source(
        seed_forecast: FcffForecast,
        requested_parameters: FcffForecastParameters,
        driver: FcffForecastDriver,
    ) -> ForecastAssumptionSource:
        override = requested_parameters.assumption_source_overrides.get(driver)
        if override is not None:
            return override
        if getattr(requested_parameters, driver.value) is not None:
            return ForecastAssumptionSource.EXPLICIT
        source_path = seed_forecast.assumption_source_paths.get(driver)
        if source_path:
            return source_path[0]
        return seed_forecast.assumption_sources.get(
            driver, ForecastAssumptionSource.TRAILING_AVERAGE
        )

    @staticmethod
    def _prefix_adaptive_path(
        source: ForecastAssumptionSource,
        prefix_years: int,
        years: int,
    ) -> tuple[ForecastAssumptionSource, ...]:
        return (source,) * min(prefix_years, years) + (
            ForecastAssumptionSource.ADAPTIVE_MULTISTAGE,
        ) * max(0, years - prefix_years)

    @staticmethod
    def _stage_path(plan: AdaptiveMultistagePlan) -> tuple[str, ...]:
        return (
            ("current",) * plan.current_growth_years
            + ("explicit",) * plan.explicit_growth_prefix_years
            + ("near_term",) * plan.high_growth_years
            + ("transition",) * plan.transition_years
            + ("stable",) * plan.stable_years
        )

    def _material_capex_shock_indices(
        self,
        financials,
        seed_forecast: FcffForecast,
        requested_parameters: FcffForecastParameters,
        configuration,
        *,
        as_of,
        availability_mode,
    ) -> tuple[int, ...]:
        """Return forecast indexes whose absolute CAPEX guidance is material.

        The comparison is against the same forecast with monetary constraints
        removed.  This keeps the shock test focused on the amount-level guidance
        rather than on a ratio path that may already have been rewritten by an
        earlier forecast pass.
        """

        if not requested_parameters.capex_constraints:
            return ()
        unconstrained_parameters = requested_parameters.model_copy(
            update={"capex_constraints": {}}
        )
        unconstrained = self._base_service.forecast(
            financials,
            unconstrained_parameters,
            as_of=as_of,
            availability_mode=availability_mode,
        )
        baseline_by_year = {
            observation.fiscal_year: observation
            for observation in unconstrained.observations
        }
        threshold = configuration.material_capex_shock_threshold
        material: list[int] = []
        for index, observation in enumerate(seed_forecast.observations):
            if observation.fiscal_year not in requested_parameters.capex_constraints:
                continue
            baseline = baseline_by_year.get(observation.fiscal_year)
            if baseline is None:
                continue
            shock_percentage = self._capex_shock_percentage(
                baseline.capital_expenditures,
                observation.capital_expenditures,
            )
            if shock_percentage >= threshold:
                material.append(index)
        return tuple(material)

    @staticmethod
    def _capex_shock_percentage(
        baseline_amount: Decimal, guided_amount: Decimal
    ) -> Decimal:
        if baseline_amount == 0:
            return Decimal(0) if guided_amount == 0 else Decimal("100")
        return (
            abs(guided_amount - baseline_amount) / abs(baseline_amount) * Decimal(100)
        )

    @staticmethod
    def _capex_anchor_index(
        seed_forecast: FcffForecast,
        requested_parameters: FcffForecastParameters,
    ) -> int:
        constraint_indexes = [
            index
            for index, observation in enumerate(seed_forecast.observations)
            if observation.fiscal_year in requested_parameters.capex_constraints
        ]
        anchor = max(constraint_indexes, default=0)
        if (
            requested_parameters.capex_to_revenue is not None
            and len(requested_parameters.capex_to_revenue) > 1
        ):
            anchor = max(anchor, len(requested_parameters.capex_to_revenue) - 1)
        return anchor

    def _extend_for_capex_transition(
        self,
        growth_path: tuple[Decimal, ...],
        plan: AdaptiveMultistagePlan,
        requested_parameters: FcffForecastParameters,
        seed_forecast: FcffForecast,
        configuration,
        terminal_growth_rate: Decimal,
    ) -> tuple[tuple[Decimal, ...], AdaptiveMultistagePlan]:
        transition_years = configuration.capex_transition_years
        anchor_index = self._capex_anchor_index(seed_forecast, requested_parameters)
        required_years = anchor_index + transition_years + 1
        if required_years > 30:
            raise ValueError(
                "CAPEX transition exceeds the 30-year forecast limit; shorten "
                "the CAPEX transition horizon or the explicit forecast path"
            )
        if required_years <= plan.effective_years:
            return growth_path, plan.model_copy(
                update={"capex_transition_years": transition_years}
            )

        additional_years = required_years - plan.effective_years
        extended_growth_path = (
            *growth_path,
            *(terminal_growth_rate for _ in range(additional_years)),
        )
        return extended_growth_path, plan.model_copy(
            update={
                "effective_years": required_years,
                "stable_years": plan.stable_years + additional_years,
                "extended_to_stable": True,
                "capex_transition_years": transition_years,
            }
        )

    def _apply_shock_depreciation_rollforward(
        self,
        financials,
        values,
        life_years: int | None,
        as_of,
        availability_mode,
    ):
        if life_years is None:
            return values
        provisional = FcffForecastParameters.model_validate(values)
        forecast = self._base_service.forecast(
            financials,
            provisional,
            as_of=as_of,
            availability_mode=availability_mode,
        )
        values["depreciation_to_revenue"] = self._depreciation_rollforward(
            forecast, life_years
        )
        return values

    def _apply_sustainable_reinvestment(
        self,
        financials,
        values,
        requested_parameters,
        plan,
        configuration,
        as_of,
        availability_mode,
        *,
        force_depreciation_rollforward: bool = False,
        independent_capex_transition: bool = False,
        capex_anchor_index: int = 0,
    ):
        terminal_roic = configuration.terminal_return_on_invested_capital
        if terminal_roic is None:
            raise ValueError(
                "Sustainable reinvestment requires a resolved terminal ROIC; use "
                "TerminalRoicResolver before forecast arithmetic"
            )
        if terminal_roic <= plan.terminal_growth_rate:
            raise ValueError(
                "Terminal ROIC must exceed stable growth for a sustainable "
                "reinvestment forecast"
            )
        reinvestment_rate = plan.terminal_growth_rate / terminal_roic
        life = configuration.depreciable_asset_life_years
        rollforward_depreciation = life is not None and (
            requested_parameters.depreciation_to_revenue is None
            or force_depreciation_rollforward
        )

        # The terminal CAPEX target depends on D&A, while an asset-life
        # roll-forward makes D&A depend on prior CAPEX.  Iterate to a fixed
        # point instead of assuming three passes are enough for every horizon.
        target_capex_ratio = None
        previous_target = None
        capex_fade_anchor = None
        converged = False
        for _ in range(self._REINVESTMENT_MAX_ITERATIONS):
            provisional = FcffForecastParameters.model_validate(values)
            forecast = self._base_service.forecast(
                financials,
                provisional,
                as_of=as_of,
                availability_mode=availability_mode,
            )
            if rollforward_depreciation:
                values["depreciation_to_revenue"] = self._depreciation_rollforward(
                    forecast, life
                )
                provisional = FcffForecastParameters.model_validate(values)
                forecast = self._base_service.forecast(
                    financials,
                    provisional,
                    as_of=as_of,
                    availability_mode=availability_mode,
                )
            final = forecast.observations[-1]
            required_net_reinvestment = final.nopat * reinvestment_rate
            target_capex = (
                final.depreciation_and_amortization
                + required_net_reinvestment
                - final.change_in_operating_working_capital
            )
            target_capex_ratio = max(
                Decimal(0), target_capex / final.revenue * Decimal(100)
            )
            if independent_capex_transition:
                capex_path = self._capex_driver_path(
                    requested_parameters.capex_to_revenue,
                    forecast,
                    target_capex_ratio,
                    plan,
                    anchor_index=capex_anchor_index,
                )
            else:
                if capex_fade_anchor is None:
                    capex_fade_anchor = forecast.observations[0].capex_to_revenue
                capex_path = self._fade_driver_path(
                    requested_parameters.capex_to_revenue,
                    capex_fade_anchor,
                    target_capex_ratio,
                    plan,
                )
            path_delta = self._path_delta(values.get("capex_to_revenue"), capex_path)
            target_delta = (
                None
                if previous_target is None
                else abs(target_capex_ratio - previous_target)
            )
            values["capex_to_revenue"] = capex_path
            if (
                target_delta is not None
                and target_delta <= self._REINVESTMENT_TOLERANCE
                and path_delta <= self._REINVESTMENT_TOLERANCE
            ):
                converged = True
                break
            previous_target = target_capex_ratio

        if not converged:
            raise ValueError(
                "Sustainable reinvestment did not converge within "
                f"{self._REINVESTMENT_MAX_ITERATIONS} iterations"
            )

        updated_plan = plan.model_copy(
            update={
                "terminal_return_on_invested_capital": terminal_roic,
                "terminal_reinvestment_rate": reinvestment_rate * Decimal(100),
                "terminal_capex_to_revenue": target_capex_ratio,
                "depreciable_asset_life_years": life,
            }
        )
        return values, updated_plan

    def _capex_driver_path(
        self,
        explicit,
        forecast: FcffForecast,
        target: Decimal,
        plan: AdaptiveMultistagePlan,
        *,
        anchor_index: int,
    ) -> tuple[Decimal, ...]:
        """Build an independent CAPEX fade path around amount-level guidance."""

        years = plan.effective_years
        if explicit is None:
            path = [
                observation.capex_to_revenue
                for observation in forecast.observations[:years]
            ]
            if not path:
                path = [Decimal(0)]
            path.extend([path[-1]] * (years - len(path)))
        elif len(explicit) == 1:
            path = [explicit[0]] * years
        else:
            path = list(self._extend_explicit_path(explicit, years))

        constraint_indexes = [
            index
            for index, observation in enumerate(forecast.observations[:years])
            if observation.fiscal_year in forecast.parameters.capex_constraints
        ]
        for index in constraint_indexes:
            path[index] = forecast.observations[index].capex_to_revenue

        anchor_index = min(anchor_index, years - 1)
        start = path[anchor_index]

        transition_years = plan.capex_transition_years
        for offset in range(1, transition_years + 1):
            index = anchor_index + offset
            if index >= years:
                break
            path[index] = start + (target - start) * Decimal(offset) / Decimal(
                transition_years
            )
        endpoint = anchor_index + transition_years
        for index in range(max(anchor_index + 1, endpoint + 1), years):
            path[index] = target
        return tuple(path)

    @classmethod
    def _path_delta(
        cls,
        previous: tuple[Decimal, ...] | list[Decimal] | None,
        current: tuple[Decimal, ...],
    ) -> Decimal:
        if previous is None or len(previous) != len(current):
            return Decimal("Infinity")
        return max(
            (
                abs(before - after)
                for before, after in zip(previous, current, strict=True)
            ),
            default=Decimal(0),
        )

    @staticmethod
    def _fade_driver_path(explicit, initial, target, plan):
        years = plan.effective_years
        convergence_year = years - plan.stable_years
        if explicit is not None and len(explicit) > 1:
            prefix = list(explicit[:convergence_year])
        else:
            held_years = max(
                1,
                plan.current_growth_years
                + plan.explicit_growth_prefix_years
                + plan.high_growth_years,
            )
            prefix = [explicit[0] if explicit is not None else initial] * held_years
        remaining = convergence_year - len(prefix)
        if remaining <= 0:
            prefix = prefix[:convergence_year]
            if prefix:
                prefix[-1] = target
            return tuple([*prefix, *([target] * plan.stable_years)])
        start = prefix[-1]
        prefix.extend(
            AdaptiveMultistageFcffForecastService._linear_transition(
                start, target, remaining
            )
        )
        prefix.extend([target] * plan.stable_years)
        return tuple(prefix)

    @staticmethod
    def _depreciation_rollforward(forecast, life_years):
        life = Decimal(life_years)
        depreciable_assets = forecast.base_depreciation_and_amortization * life
        ratios = []
        for observation in forecast.observations:
            depreciation = depreciable_assets / life
            ratios.append(depreciation / observation.revenue * Decimal(100))
            depreciable_assets = max(
                Decimal(0),
                depreciable_assets - depreciation + observation.capital_expenditures,
            )
        return tuple(ratios)

    def _growth_path(
        self,
        seed_forecast,
        parameters,
        terminal_growth_rate,
        configuration,
        forward_evidence=None,
        forward_growth: ForwardGrowthOutlook | None = None,
    ) -> tuple[tuple[Decimal, ...], AdaptiveMultistagePlan]:
        if forward_growth is None:
            configured = parameters.revenue_growth
            fallback_anchor = (
                configured[-1]
                if configured is not None
                else (
                    seed_forecast.normalized_historical_growth
                    if seed_forecast.normalized_historical_growth is not None
                    else seed_forecast.observations[0].revenue_growth
                )
            )
            forward_growth = self._make_outlook(
                path=tuple(configured or ()),
                anchor=fallback_anchor,
                source=(
                    ForecastAssumptionSource.EXPLICIT.value
                    if configured is not None
                    else ForecastAssumptionSource.NORMALIZED_HISTORICAL.value
                ),
                confidence="high" if configured is not None else "medium",
                current_growth=(
                    seed_forecast.current_growth_rate
                    or seed_forecast.observations[0].revenue_growth
                ),
                history_path=tuple(seed_forecast.normalized_historical_growth_path),
                terminal_growth_rate=terminal_growth_rate,
                convergence_tolerance=configuration.convergence_tolerance,
                forward_evidence=forward_evidence,
            )

        explicit_prefix = tuple(forward_growth.growth_path[: parameters.forecast_years])
        initial_growth = forward_growth.anchor
        evidence_score = (
            forward_evidence.score if forward_evidence is not None else Decimal(0)
        )
        gap = abs(initial_growth - terminal_growth_rate)
        stable_ready = forward_growth.stable_state_supported and (
            gap <= configuration.convergence_tolerance
        )

        if stable_ready:
            high_growth_years = 0
            transition_years = 0
        else:
            high_growth_years = (
                min(
                    configuration.maximum_high_growth_years,
                    self._ceiling(gap / configuration.growth_gap_per_high_growth_year),
                )
                if gap >= configuration.convergence_tolerance
                else 0
            )
            if explicit_prefix:
                # A supplied forward path already defines the near-term
                # regime; do not add another synthetic hold year after it.
                high_growth_years = 0
            if forward_evidence is not None and not explicit_prefix:
                high_growth_years = min(
                    configuration.maximum_high_growth_years,
                    high_growth_years + self._ceiling(evidence_score * Decimal("3")),
                )
            transition_years = (
                self._ceiling(gap / configuration.max_annual_growth_fade)
                if gap >= configuration.convergence_tolerance
                else configuration.minimum_transition_years
            )
            if forward_evidence is not None:
                transition_years += self._ceiling(evidence_score * Decimal("5"))
            transition_years = max(
                configuration.minimum_transition_years,
                min(configuration.maximum_transition_years, transition_years),
            )

        current_growth_years = int(
            seed_forecast.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
            and not explicit_prefix
            and not parameters.revenue_anchors
        )
        convergence_year = (
            current_growth_years
            + len(explicit_prefix)
            + high_growth_years
            + transition_years
        )
        effective_years = parameters.forecast_years
        if configuration.extend_to_stable:
            # One full year after the fade is needed for FCFF growth itself to
            # reflect the stable driver set used by the terminal value.
            effective_years = max(effective_years, convergence_year + 1)
        if effective_years > 30:
            raise ValueError(
                "Adaptive multistage convergence exceeds the 30-year forecast "
                "limit; shorten the explicit growth path or increase the allowed fade"
            )

        full_path = []
        if current_growth_years:
            full_path.append(
                forward_growth.current_growth
                if forward_growth.current_growth is not None
                else initial_growth
            )
        full_path.extend(explicit_prefix)
        full_path.extend([initial_growth] * high_growth_years)
        full_path.extend(
            self._linear_transition(
                initial_growth, terminal_growth_rate, transition_years
            )
        )
        full_path.extend(
            [terminal_growth_rate] * max(0, effective_years - len(full_path))
        )
        path = tuple(full_path[:effective_years])

        current_used = min(current_growth_years, effective_years)
        remaining = effective_years - current_used
        explicit_used = min(len(explicit_prefix), remaining)
        remaining -= explicit_used
        high_used = min(high_growth_years, remaining)
        remaining -= high_used
        transition_used = min(transition_years, remaining)
        stable_years = remaining - transition_used
        plan = AdaptiveMultistagePlan(
            requested_years=parameters.forecast_years,
            effective_years=effective_years,
            high_growth_years=high_used,
            transition_years=transition_used,
            stable_years=stable_years,
            initial_growth_rate=initial_growth,
            terminal_growth_rate=terminal_growth_rate,
            max_annual_growth_fade=configuration.max_annual_growth_fade,
            extended_to_stable=effective_years > parameters.forecast_years,
            explicit_growth_prefix_years=explicit_used,
            forward_evidence_score=(
                forward_evidence.score if forward_evidence is not None else Decimal(0)
            ),
            forward_evidence_summary=(
                forward_evidence.summary if forward_evidence is not None else ()
            ),
            current_growth_years=current_used,
            current_growth_rate=forward_growth.current_growth,
            forward_growth_rate=forward_growth.anchor,
            forward_growth_path=forward_growth.growth_path,
            historical_growth_path=forward_growth.historical_growth_path,
            management_guidance_path=forward_growth.management_guidance_path,
            forward_estimates_path=forward_growth.forward_estimates_path,
            forward_growth_path_by_year=forward_growth.growth_path_by_year,
            guidance_growth_path_by_year=forward_growth.guidance_growth_path_by_year,
            forward_revenue_estimates=forward_growth.forward_revenue_estimates,
            forward_estimate_provider=forward_growth.forward_estimate_provider,
            forward_estimate_years=forward_growth.forward_estimate_years,
            forward_estimate_growth_path=forward_growth.forward_estimate_growth_path,
            forward_estimate_diagnostics=forward_growth.forward_estimate_diagnostics,
            forward_growth_source=forward_growth.source,
            forward_growth_confidence=forward_growth.confidence,
            stable_state_supported=forward_growth.stable_state_supported,
            current_growth_near_terminal=forward_growth.current_growth_near_terminal,
            warnings=forward_growth.warnings,
        )
        return path, plan

    @staticmethod
    def _converging_path(
        initial: Decimal,
        target: Decimal,
        plan: AdaptiveMultistagePlan,
    ) -> tuple[Decimal, ...]:
        prefix_years = (
            plan.current_growth_years
            + plan.explicit_growth_prefix_years
            + plan.high_growth_years
        )
        values = [initial] * prefix_years
        values.extend(
            AdaptiveMultistageFcffForecastService._linear_transition(
                initial, target, plan.transition_years
            )
        )
        values.extend([target] * plan.stable_years)
        return tuple(values)

    @staticmethod
    def _linear_transition(
        initial: Decimal,
        target: Decimal,
        years: int,
    ) -> list[Decimal]:
        if years == 0:
            return []
        return [
            initial + (target - initial) * Decimal(year) / Decimal(years)
            for year in range(1, years + 1)
        ]

    @staticmethod
    def _extend_explicit_path(
        path: tuple[Decimal, ...] | None,
        years: int,
    ) -> tuple[Decimal, ...] | None:
        if path is None or len(path) == 1:
            return path
        return tuple([*path, *([path[-1]] * (years - len(path)))])

    @staticmethod
    def _ceiling(value: Decimal) -> int:
        return int(value.to_integral_value(rounding=ROUND_CEILING))


__all__ = ["AdaptiveMultistageFcffForecastService"]
