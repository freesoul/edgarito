from decimal import ROUND_CEILING, Decimal

from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.services.forecasting.fcff import FcffForecastService
from edgarito.services.forecasting.models import (
    AdaptiveMultistagePlan,
    FcffForecast,
    FcffForecastDriver,
    FcffForecastParameters,
    ForecastAssumptionSource,
)


class AdaptiveMultistageFcffForecastService:
    """Fade FCFF operating drivers into a sustainable perpetual-growth stage."""

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
    ) -> tuple[FcffForecast, AdaptiveMultistagePlan]:
        if not seed_forecast.observations:
            raise ValueError("Adaptive multistage forecasting requires a seed forecast")
        growth_path, plan = self._growth_path(
            seed_forecast,
            requested_parameters,
            terminal_growth_rate,
            configuration,
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

        parameters = FcffForecastParameters.model_validate(values)
        forecast = self._base_service.forecast(financials, parameters)
        forecast.method = "adaptive_multistage_fcff"
        forecast.assumption_sources[FcffForecastDriver.REVENUE_GROWTH] = (
            ForecastAssumptionSource.ADAPTIVE_MULTISTAGE
        )
        if tax_is_adaptive:
            forecast.assumption_sources[FcffForecastDriver.TAX_RATE] = (
                ForecastAssumptionSource.ADAPTIVE_MULTISTAGE
            )
        return forecast, plan

    def _growth_path(
        self,
        seed_forecast,
        parameters,
        terminal_growth_rate,
        configuration,
    ) -> tuple[tuple[Decimal, ...], AdaptiveMultistagePlan]:
        configured = parameters.revenue_growth
        explicit_prefix: tuple[Decimal, ...] = ()
        if configured is not None and len(configured) > 1:
            explicit_prefix = tuple(configured)
            initial_growth = explicit_prefix[-1]
            high_growth_years = 0
        else:
            initial_growth = (
                configured[0]
                if configured is not None
                else seed_forecast.observations[0].revenue_growth
            )
            gap = abs(initial_growth - terminal_growth_rate)
            high_growth_years = (
                min(
                    configuration.maximum_high_growth_years,
                    self._ceiling(gap / configuration.growth_gap_per_high_growth_year),
                )
                if gap >= configuration.convergence_tolerance
                else 0
            )

        gap = abs(initial_growth - terminal_growth_rate)
        transition_years = 0
        if gap >= configuration.convergence_tolerance:
            transition_years = self._ceiling(gap / configuration.max_annual_growth_fade)
            transition_years = max(
                configuration.minimum_transition_years,
                min(configuration.maximum_transition_years, transition_years),
            )

        convergence_year = len(explicit_prefix) + high_growth_years + transition_years
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

        full_path = [*explicit_prefix, *([initial_growth] * high_growth_years)]
        full_path.extend(
            self._linear_transition(
                initial_growth, terminal_growth_rate, transition_years
            )
        )
        full_path.extend(
            [terminal_growth_rate] * max(0, effective_years - len(full_path))
        )
        path = tuple(full_path[:effective_years])

        explicit_used = min(len(explicit_prefix), effective_years)
        remaining = effective_years - explicit_used
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
        )
        return path, plan

    @staticmethod
    def _converging_path(
        initial: Decimal,
        target: Decimal,
        plan: AdaptiveMultistagePlan,
    ) -> tuple[Decimal, ...]:
        prefix_years = plan.explicit_growth_prefix_years + plan.high_growth_years
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
