import datetime
from decimal import Decimal
from statistics import median

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.market import SecurityMarketData
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.valuation.models import (
    CompanyTradingMultiples,
    ComparableMultiplesReport,
    LtmFundamentals,
    MultipleStatus,
    PeerMultipleSummary,
    PeerUniverse,
    RelativeValuationBasis,
    TradingMultiple,
)

PeriodKey = tuple[int, FiscalPeriod]


class LtmMultiplesService:
    """Compute price and enterprise-value multiples from four fiscal quarters."""

    _FLOW_CONCEPTS = {
        FinancialConcept.REVENUE,
        FinancialConcept.OPERATING_INCOME,
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
        FinancialConcept.NET_INCOME,
        FinancialConcept.OPERATING_CASH_FLOW,
        FinancialConcept.CAPITAL_EXPENDITURES,
        FinancialConcept.DIVIDENDS_PAID,
    }
    _BALANCE_CONCEPTS = {
        FinancialConcept.STOCKHOLDERS_EQUITY,
        FinancialConcept.GOODWILL,
        FinancialConcept.INTANGIBLE_ASSETS_NET,
        FinancialConcept.CASH_AND_EQUIVALENTS,
        FinancialConcept.SHORT_TERM_DEBT,
        FinancialConcept.LONG_TERM_DEBT_CURRENT,
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
        FinancialConcept.SHARES_OUTSTANDING,
        FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
    }

    @classmethod
    def required_concepts(cls) -> set[FinancialConcept]:
        return cls._FLOW_CONCEPTS | cls._BALANCE_CONCEPTS

    def compute(
        self,
        financials: NormalizedCompanyFinancials,
        market_data: SecurityMarketData,
        as_of: datetime.date | None = None,
        *,
        point_in_time: bool = False,
    ) -> CompanyTradingMultiples:
        try:
            return self._compute_quarterly(
                financials, market_data, as_of, point_in_time=point_in_time
            )
        except ValueError as exc:
            if "four consecutive quarterly revenue periods" not in str(exc):
                raise
            return self._compute_latest_annual(
                financials, market_data, as_of, point_in_time=point_in_time
            )

    def _compute_quarterly(
        self,
        financials: NormalizedCompanyFinancials,
        market_data: SecurityMarketData,
        as_of: datetime.date | None = None,
        *,
        point_in_time: bool = False,
    ) -> CompanyTradingMultiples:
        ticker = financials.ticker or financials.company_id
        price = self._latest_price(market_data, as_of)
        by_period = self._quarterly_by_period(
            financials, as_of, point_in_time=point_in_time
        )
        period_keys = self._latest_ltm_keys(by_period)
        first_revenue = by_period[period_keys[0]][FinancialConcept.REVENUE]
        predecessor = next(
            (
                values[FinancialConcept.REVENUE]
                for key, values in by_period.items()
                if self._period_index(key) == self._period_index(period_keys[0]) - 1
                and FinancialConcept.REVENUE in values
            ),
            None,
        )
        period_start = (
            predecessor.period_end + datetime.timedelta(days=1)
            if predecessor is not None
            else first_revenue.period_start or first_revenue.period_end
        )
        period_end = max(
            by_period[key][FinancialConcept.REVENUE].period_end for key in period_keys
        )
        revenue_observations = [
            by_period[key][FinancialConcept.REVENUE] for key in period_keys
        ]
        currency = revenue_observations[0].unit
        if any(observation.unit != currency for observation in revenue_observations):
            raise ValueError("LTM revenue quarters must use one currency")

        flows = {
            concept: self._sum_flow(by_period, period_keys, concept, currency)
            for concept in self._FLOW_CONCEPTS
        }
        balances = {
            concept: self._latest_balance(by_period, period_keys[-1], concept, currency)
            for concept in self._BALANCE_CONCEPTS
        }
        operating_income = flows[FinancialConcept.OPERATING_INCOME]
        depreciation = flows[FinancialConcept.DEPRECIATION_AND_AMORTIZATION]
        ebitda = self._sum_optional(operating_income, depreciation)
        free_cash_flow = self._subtract_optional(
            flows[FinancialConcept.OPERATING_CASH_FLOW],
            flows[FinancialConcept.CAPITAL_EXPENDITURES],
        )
        gross_debt = self._gross_debt(balances)
        equity = balances[FinancialConcept.STOCKHOLDERS_EQUITY]
        tangible_book = self._tangible_book_equity(balances)
        shares, share_basis = self._shares(balances)

        warnings = []
        if market_data.currency != currency:
            warnings.append(
                f"Market currency {market_data.currency} differs from reporting "
                f"currency {currency}; FX alignment is required"
            )
        market_cap = price.close * shares if shares is not None else None
        if shares is None:
            warnings.append("No current or diluted share count is available")
        cash = balances[FinancialConcept.CASH_AND_EQUIVALENTS]
        enterprise_value = None
        comparable_market_cap = None
        if market_data.currency == currency:
            comparable_market_cap = market_cap
            enterprise_value = self._enterprise_value(market_cap, gross_debt, cash)
            if enterprise_value is None:
                warnings.append("Enterprise value requires market cap, debt, and cash")

        fundamentals = LtmFundamentals(
            period_start=period_start,
            period_end=period_end,
            currency=currency,
            revenue=flows[FinancialConcept.REVENUE],
            operating_income=operating_income,
            depreciation_and_amortization=depreciation,
            ebitda=ebitda,
            net_income=flows[FinancialConcept.NET_INCOME],
            free_cash_flow=free_cash_flow,
            capital_expenditures=flows[FinancialConcept.CAPITAL_EXPENDITURES],
            dividends_paid=flows[FinancialConcept.DIVIDENDS_PAID],
            book_equity=equity,
            tangible_book_equity=tangible_book,
            cash_and_equivalents=cash,
            gross_debt=gross_debt,
            shares=shares,
            share_basis=share_basis,
        )
        multiples = self._multiples(
            comparable_market_cap, enterprise_value, fundamentals
        )
        return CompanyTradingMultiples(
            provider=financials.provider,
            market_provider=market_data.provider,
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=ticker,
            price_date=price.observed_on,
            price=price.close,
            currency=market_data.currency,
            market_capitalization=market_cap,
            enterprise_value=enterprise_value,
            fundamentals=fundamentals,
            multiples=multiples,
            warnings=warnings,
        )

    def _compute_latest_annual(
        self,
        financials: NormalizedCompanyFinancials,
        market_data: SecurityMarketData,
        as_of: datetime.date | None,
        *,
        point_in_time: bool = False,
    ) -> CompanyTradingMultiples:
        price = self._latest_price(market_data, as_of)
        annual = [
            item
            for item in financials.observations
            if item.granularity == Granularity.ANNUAL
            and item.fiscal_period == FiscalPeriod.FY
            and (
                as_of is None
                or self._is_available(item, as_of, point_in_time=point_in_time)
            )
        ]
        revenue_periods = [
            item for item in annual if item.concept == FinancialConcept.REVENUE
        ]
        if not revenue_periods:
            raise ValueError(
                "LTM multiples require four consecutive quarterly revenue periods; "
                "latest-annual fallback is also unavailable"
            )
        revenue = max(revenue_periods, key=lambda item: item.period_end)
        prior_revenue = next(
            (
                item
                for item in revenue_periods
                if item.fiscal_year == revenue.fiscal_year - 1
                and item.unit == revenue.unit
            ),
            None,
        )
        revenue_growth = (
            (revenue.value / prior_revenue.value - Decimal(1)) * Decimal(100)
            if prior_revenue is not None and prior_revenue.value > 0
            else None
        )
        same_period = [
            item
            for item in annual
            if item.fiscal_year == revenue.fiscal_year
            and item.period_end == revenue.period_end
        ]
        by_concept = {}
        for item in same_period:
            current = by_concept.get(item.concept)
            if current is None or self.availability_date(item) > self.availability_date(
                current
            ):
                by_concept[item.concept] = item
        currency = revenue.unit

        def value(concept):
            item = by_concept.get(concept)
            return (
                item.value
                if item is not None and item.unit in {currency, "shares"}
                else None
            )

        operating_income = value(FinancialConcept.OPERATING_INCOME)
        depreciation = value(FinancialConcept.DEPRECIATION_AND_AMORTIZATION)
        ebitda = self._sum_optional(operating_income, depreciation)
        free_cash_flow = self._subtract_optional(
            value(FinancialConcept.OPERATING_CASH_FLOW),
            value(FinancialConcept.CAPITAL_EXPENDITURES),
        )
        balances = {concept: value(concept) for concept in self._BALANCE_CONCEPTS}
        gross_debt = self._gross_debt(balances)
        shares, share_basis = self._shares(balances)
        market_cap = price.close * shares if shares is not None else None
        comparable_market_cap = market_cap if market_data.currency == currency else None
        cash = balances[FinancialConcept.CASH_AND_EQUIVALENTS]
        enterprise_value = self._enterprise_value(
            comparable_market_cap, gross_debt, cash
        )
        warnings = [
            "Four consecutive quarters were unavailable; multiples use the latest "
            f"annual FY{revenue.fiscal_year} fundamentals"
        ]
        if market_data.currency != currency:
            warnings.append(
                f"Market currency {market_data.currency} differs from reporting "
                f"currency {currency}; FX alignment is required"
            )
        if shares is None:
            warnings.append("No current or diluted share count is available")
        if enterprise_value is None:
            warnings.append("Enterprise value requires market cap, debt, and cash")
        fundamentals = LtmFundamentals(
            period_start=revenue.period_start or revenue.period_end,
            period_end=revenue.period_end,
            currency=currency,
            revenue=revenue.value,
            revenue_growth=revenue_growth,
            operating_income=operating_income,
            depreciation_and_amortization=depreciation,
            ebitda=ebitda,
            net_income=value(FinancialConcept.NET_INCOME),
            free_cash_flow=free_cash_flow,
            capital_expenditures=value(FinancialConcept.CAPITAL_EXPENDITURES),
            dividends_paid=value(FinancialConcept.DIVIDENDS_PAID),
            book_equity=balances[FinancialConcept.STOCKHOLDERS_EQUITY],
            tangible_book_equity=self._tangible_book_equity(balances),
            cash_and_equivalents=cash,
            gross_debt=gross_debt,
            shares=shares,
            share_basis=share_basis,
        )
        return CompanyTradingMultiples(
            provider=financials.provider,
            market_provider=market_data.provider,
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker or financials.company_id,
            price_date=price.observed_on,
            price=price.close,
            currency=market_data.currency,
            market_capitalization=market_cap,
            enterprise_value=enterprise_value,
            fundamentals=fundamentals,
            multiples=self._multiples(
                comparable_market_cap, enterprise_value, fundamentals
            ),
            warnings=warnings,
        )

    @classmethod
    def _quarterly_by_period(
        cls,
        financials: NormalizedCompanyFinancials,
        as_of: datetime.date | None = None,
        *,
        point_in_time: bool = False,
    ) -> dict[PeriodKey, dict[FinancialConcept, FinancialObservation]]:
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]] = {}
        for observation in financials.observations:
            if (
                observation.granularity != Granularity.QUARTERLY
                or observation.fiscal_period == FiscalPeriod.FY
                or (
                    as_of is not None
                    and not cls._is_available(
                        observation, as_of, point_in_time=point_in_time
                    )
                )
            ):
                continue
            values = by_period.setdefault(observation.period_key, {})
            current = values.get(observation.concept)
            if current is None or cls.availability_date(
                observation
            ) > cls.availability_date(current):
                values[observation.concept] = observation
        return by_period

    @staticmethod
    def availability_date(observation: FinancialObservation) -> datetime.date:
        """Return the earliest defensible date for a historical snapshot.

        SEC-normalized observations carry their actual filing date. Yahoo's
        standardized statement tables do not expose filing dates, so use a
        conservative, provider-wide publication lag instead of pretending the
        period-end values were public on the balance-sheet date.
        """
        if observation.filed is not None:
            return observation.filed
        if observation.provider.casefold() == "yahoo":
            lag_days = 90 if observation.granularity == Granularity.ANNUAL else 45
            return observation.period_end + datetime.timedelta(days=lag_days)
        return observation.period_end

    @classmethod
    def _is_available(
        cls,
        observation: FinancialObservation,
        as_of: datetime.date,
        *,
        point_in_time: bool,
    ) -> bool:
        available_on = (
            cls.availability_date(observation)
            if point_in_time or observation.filed is not None
            else observation.period_end
        )
        return available_on <= as_of

    @classmethod
    def _latest_ltm_keys(
        cls,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
    ) -> tuple[PeriodKey, PeriodKey, PeriodKey, PeriodKey]:
        revenue_keys = sorted(
            (
                key
                for key, values in by_period.items()
                if FinancialConcept.REVENUE in values
            ),
            key=cls._period_index,
        )
        for end_index in range(len(revenue_keys) - 1, 2, -1):
            keys = revenue_keys[end_index - 3 : end_index + 1]
            indices = [cls._period_index(key) for key in keys]
            if all(
                current == previous + 1
                for previous, current in zip(indices, indices[1:], strict=False)
            ):
                return tuple(keys)  # type: ignore[return-value]
        raise ValueError(
            "LTM multiples require four consecutive quarterly revenue periods"
        )

    @staticmethod
    def _period_index(key: PeriodKey) -> int:
        fiscal_year, fiscal_period = key
        quarter = {
            FiscalPeriod.Q1: 0,
            FiscalPeriod.Q2: 1,
            FiscalPeriod.Q3: 2,
            FiscalPeriod.Q4: 3,
        }.get(fiscal_period)
        if quarter is None:
            raise ValueError("LTM periods must be fiscal quarters")
        return fiscal_year * 4 + quarter

    @staticmethod
    def _sum_flow(
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        period_keys: tuple[PeriodKey, ...],
        concept: FinancialConcept,
        currency: str,
    ) -> Decimal | None:
        observations = [by_period[key].get(concept) for key in period_keys]
        if any(item is None or item.unit != currency for item in observations):
            return None
        return sum(
            (item.value for item in observations if item is not None), Decimal(0)
        )

    @classmethod
    def _latest_balance(
        cls,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        end_key: PeriodKey,
        concept: FinancialConcept,
        currency: str,
    ) -> Decimal | None:
        candidates = [
            (key, values[concept])
            for key, values in by_period.items()
            if concept in values
            and cls._period_index(key) <= cls._period_index(end_key)
            and values[concept].unit in {currency, "shares"}
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: cls._period_index(item[0]))[1].value

    @staticmethod
    def _gross_debt(
        balances: dict[FinancialConcept, Decimal | None],
    ) -> Decimal | None:
        short_term = balances[FinancialConcept.SHORT_TERM_DEBT]
        current_long_term = balances[FinancialConcept.LONG_TERM_DEBT_CURRENT]
        noncurrent = balances[FinancialConcept.LONG_TERM_DEBT_NONCURRENT]
        if all(value is None for value in (short_term, current_long_term, noncurrent)):
            return None
        current = short_term if short_term is not None else current_long_term
        return sum(
            (value for value in (current, noncurrent) if value is not None),
            Decimal(0),
        )

    @staticmethod
    def _tangible_book_equity(
        balances: dict[FinancialConcept, Decimal | None],
    ) -> Decimal | None:
        values = (
            balances[FinancialConcept.STOCKHOLDERS_EQUITY],
            balances[FinancialConcept.GOODWILL],
            balances[FinancialConcept.INTANGIBLE_ASSETS_NET],
        )
        if any(value is None for value in values):
            return None
        equity, goodwill, intangibles = values
        assert equity is not None and goodwill is not None and intangibles is not None
        return equity - goodwill - intangibles

    @staticmethod
    def _shares(
        balances: dict[FinancialConcept, Decimal | None],
    ) -> tuple[Decimal | None, str | None]:
        shares = balances[FinancialConcept.SHARES_OUTSTANDING]
        if shares is not None:
            return shares, "shares_outstanding"
        diluted = balances[FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES]
        return (diluted, "latest_diluted_weighted_average") if diluted else (None, None)

    @staticmethod
    def _enterprise_value(
        market_cap: Decimal | None,
        gross_debt: Decimal | None,
        cash: Decimal | None,
    ) -> Decimal | None:
        if market_cap is None or gross_debt is None or cash is None:
            return None
        return market_cap + gross_debt - cash

    @staticmethod
    def _sum_optional(left: Decimal | None, right: Decimal | None) -> Decimal | None:
        return left + right if left is not None and right is not None else None

    @staticmethod
    def _subtract_optional(
        left: Decimal | None, right: Decimal | None
    ) -> Decimal | None:
        return left - right if left is not None and right is not None else None

    @staticmethod
    def _latest_price(market_data: SecurityMarketData, as_of: datetime.date | None):
        prices = [
            price
            for price in market_data.prices
            if as_of is None or price.observed_on <= as_of
        ]
        if not prices:
            raise ValueError("No market price is available on or before the as-of date")
        return max(prices, key=lambda price: price.observed_on)

    def _multiples(
        self,
        market_cap: Decimal | None,
        enterprise_value: Decimal | None,
        fundamentals: LtmFundamentals,
    ) -> list[TradingMultiple]:
        return [
            self._multiple(
                RelativeValuationBasis.PE, market_cap, fundamentals.net_income
            ),
            self._multiple(
                RelativeValuationBasis.PRICE_TO_BOOK,
                market_cap,
                fundamentals.book_equity,
            ),
            self._multiple(
                RelativeValuationBasis.PRICE_TO_TANGIBLE_BOOK,
                market_cap,
                fundamentals.tangible_book_equity,
            ),
            self._multiple(
                RelativeValuationBasis.EV_TO_REVENUE,
                enterprise_value,
                fundamentals.revenue,
            ),
            self._multiple(
                RelativeValuationBasis.EV_TO_EBIT,
                enterprise_value,
                fundamentals.operating_income,
            ),
            self._multiple(
                RelativeValuationBasis.EV_TO_EBITDA,
                enterprise_value,
                fundamentals.ebitda,
            ),
            self._multiple(
                RelativeValuationBasis.EV_TO_FCF,
                enterprise_value,
                fundamentals.free_cash_flow,
            ),
            self._multiple(
                RelativeValuationBasis.DIVIDEND_YIELD,
                fundamentals.dividends_paid,
                market_cap,
                unit="percent",
                scale=Decimal(100),
            ),
        ]

    @staticmethod
    def _multiple(
        basis: RelativeValuationBasis,
        numerator: Decimal | None,
        denominator: Decimal | None,
        *,
        unit: str = "multiple",
        scale: Decimal = Decimal(1),
    ) -> TradingMultiple:
        if numerator is None or denominator is None:
            return TradingMultiple(
                basis=basis,
                status=MultipleStatus.UNAVAILABLE,
                unit=unit,
                numerator=numerator,
                denominator=denominator,
                reason="Required numerator or denominator is unavailable",
            )
        if denominator <= 0:
            return TradingMultiple(
                basis=basis,
                status=MultipleStatus.NOT_MEANINGFUL,
                unit=unit,
                numerator=numerator,
                denominator=denominator,
                reason="The denominator is zero or negative",
            )
        return TradingMultiple(
            basis=basis,
            status=MultipleStatus.COMPUTED,
            value=numerator / denominator * scale,
            unit=unit,
            numerator=numerator,
            denominator=denominator,
        )


class ComparableMultiplesService:
    """Aggregate selected peer multiples without hiding sample size."""

    def build(
        self,
        universe: PeerUniverse,
        target: CompanyTradingMultiples,
        peers: list[CompanyTradingMultiples],
    ) -> ComparableMultiplesReport:
        selected = set(universe.selected_tickers)
        included = [peer for peer in peers if peer.ticker in selected]
        summaries = []
        for basis in RelativeValuationBasis:
            values = [
                multiple.value
                for peer in included
                for multiple in peer.multiples
                if multiple.basis == basis
                and multiple.status == MultipleStatus.COMPUTED
                and multiple.value is not None
            ]
            if values:
                summaries.append(
                    PeerMultipleSummary(
                        basis=basis,
                        median=median(values),
                        minimum=min(values),
                        maximum=max(values),
                        sample_size=len(values),
                    )
                )
        warnings = list(universe.warnings)
        if len(included) != len(selected):
            warnings.append(
                f"Multiples were computed for {len(included)} of "
                f"{len(selected)} selected peers"
            )
        return ComparableMultiplesReport(
            universe=universe,
            target=target,
            peers=included,
            summaries=summaries,
            warnings=warnings,
        )


__all__ = ["ComparableMultiplesService", "LtmMultiplesService"]
