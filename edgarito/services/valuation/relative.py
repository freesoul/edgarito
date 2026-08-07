"""Provider-neutral enterprise and equity relative valuation services."""

import datetime
from decimal import Decimal
from statistics import median

from edgarito.schemas.valuation.relative import (
    ForwardValuationMetric,
    ProviderNeutralRelativeCase,
    ProviderNeutralRelativeValuation,
    RelativeCapitalBridge,
    RelativeNumeratorBasis,
)
from edgarito.services.valuation.discounting import PresentValueService
from edgarito.services.valuation.models import (
    PeerMultipleSummary,
    RelativeValuationBasis,
    ResolvedMultiple,
)


class ProviderNeutralForwardMultiplesService:
    """Summarize peer multiples using the numerator required by each basis."""

    @staticmethod
    def summarize(
        basis: RelativeValuationBasis,
        observations: tuple[tuple[Decimal, ForwardValuationMetric], ...],
    ) -> PeerMultipleSummary:
        values = sorted(
            numerator / metric.amount
            for numerator, metric in observations
            if metric.basis == basis and numerator > 0 and metric.amount > 0
        )
        if not values:
            raise ValueError(f"No usable peer forward metrics for {basis.value}")
        return PeerMultipleSummary(
            basis=basis,
            median=median(values),
            minimum=values[0],
            maximum=values[-1],
            percentile_25=_percentile(values, Decimal("0.25")),
            percentile_75=_percentile(values, Decimal("0.75")),
            sample_size=len(values),
        )


class ProviderNeutralRelativeValuationService:
    """Apply resolved multiples with an EV bridge only for enterprise bases."""

    def value(
        self,
        *,
        valuation_date: datetime.date,
        horizon_years: Decimal,
        metric: ForwardValuationMetric,
        diluted_shares: Decimal,
        discount_rate: Decimal,
        resolved_multiple: ResolvedMultiple,
        capital_bridge: RelativeCapitalBridge | None = None,
        current_price: Decimal | None = None,
    ) -> ProviderNeutralRelativeValuation:
        if resolved_multiple.basis != metric.basis:
            raise ValueError("Resolved multiple and forward metric bases must match")
        if diluted_shares <= 0:
            raise ValueError("Relative valuation requires positive diluted shares")
        enterprise = metric.numerator_basis == RelativeNumeratorBasis.ENTERPRISE_VALUE
        if enterprise and capital_bridge is None:
            raise ValueError("Enterprise-value bases require an EV-to-equity bridge")
        if not enterprise and capital_bridge is not None:
            raise ValueError("Equity-value bases must not apply an EV bridge")
        cases = tuple(
            self._case(
                label=label,
                multiple=multiple,
                metric=metric,
                shares=diluted_shares,
                discount_rate=discount_rate,
                horizon_years=horizon_years,
                bridge=capital_bridge,
            )
            for label, multiple in (
                ("Lower", resolved_multiple.lower_bound),
                ("Resolved", resolved_multiple.point_estimate),
                ("Upper", resolved_multiple.upper_bound),
            )
        )
        current_multiple = None
        if current_price is not None:
            if current_price <= 0:
                raise ValueError("Current price must be positive")
            equity_value = current_price * diluted_shares
            numerator = equity_value
            if enterprise:
                assert capital_bridge is not None
                numerator += (
                    capital_bridge.net_debt - capital_bridge.non_operating_assets
                )
            current_multiple = numerator / metric.amount
        warnings = list(resolved_multiple.warnings)
        if enterprise:
            warnings.append(
                "Projected net debt and non-operating assets require an explicit horizon assumption",
            )
        return ProviderNeutralRelativeValuation(
            valuation_date=valuation_date,
            target_date=metric.target_date,
            horizon_years=horizon_years,
            currency=metric.currency,
            metric=metric,
            diluted_shares=diluted_shares,
            discount_rate=discount_rate,
            lower_case=cases[0],
            point_case=cases[1],
            upper_case=cases[2],
            confidence=resolved_multiple.confidence,
            current_price=current_price,
            current_price_implied_multiple=current_multiple,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _case(
        *,
        label: str,
        multiple: Decimal,
        metric: ForwardValuationMetric,
        shares: Decimal,
        discount_rate: Decimal,
        horizon_years: Decimal,
        bridge: RelativeCapitalBridge | None,
    ) -> ProviderNeutralRelativeCase:
        numerator_value = metric.amount * multiple
        equity_value = numerator_value
        if bridge is not None:
            equity_value = (
                numerator_value - bridge.net_debt + bridge.non_operating_assets
            )
        target_value = equity_value / shares
        present_value = target_value * PresentValueService.discount_factor(
            discount_rate, horizon_years
        )
        return ProviderNeutralRelativeCase(
            label=label,
            multiple=multiple,
            target_date_numerator_value=numerator_value,
            target_date_equity_value=equity_value,
            target_date_value_per_share=target_value,
            present_value_per_share=present_value,
        )


def _percentile(values: list[Decimal], percentile: Decimal) -> Decimal:
    if len(values) == 1:
        return values[0]
    position = percentile * Decimal(len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - Decimal(lower)
    return values[lower] + (values[upper] - values[lower]) * weight


EQUITY_RELATIVE_BASES = frozenset(
    {
        RelativeValuationBasis.PE,
        RelativeValuationBasis.PRICE_TO_BOOK,
        RelativeValuationBasis.PRICE_TO_TANGIBLE_BOOK,
        RelativeValuationBasis.PRICE_TO_AFFO,
        RelativeValuationBasis.PRICE_TO_NAV,
    }
)
