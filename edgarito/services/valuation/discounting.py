from collections.abc import Iterable
from decimal import Decimal, InvalidOperation, localcontext

from edgarito.services.valuation.models import (
    CashFlow,
    CostOfEquityResult,
    DiscountedCashFlow,
    PresentValueResult,
    TerminalValueMethod,
    TerminalValueResult,
    WaccResult,
)

DecimalInput = Decimal | int | str
_ONE_HUNDRED = Decimal(100)


def _decimal(value: DecimalInput, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        converted = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not converted.is_finite():
        raise ValueError(f"{name} must be finite")
    return converted


class DiscountRateService:
    """Calculate reusable discount-rate components in percentage points."""

    @staticmethod
    def lever_beta(
        unlevered_beta: DecimalInput,
        market_value_debt: DecimalInput,
        market_value_equity: DecimalInput,
        normalized_tax_rate: DecimalInput,
    ) -> Decimal:
        beta = _decimal(unlevered_beta, "unlevered_beta")
        debt = _decimal(market_value_debt, "market_value_debt")
        equity = _decimal(market_value_equity, "market_value_equity")
        tax_rate = _decimal(normalized_tax_rate, "normalized_tax_rate")
        if debt < 0:
            raise ValueError("market_value_debt cannot be negative")
        if equity <= 0:
            raise ValueError("market_value_equity must be positive to lever beta")
        DiscountRateService._validate_tax_rate(tax_rate)
        return beta * (
            Decimal(1) + (Decimal(1) - tax_rate / _ONE_HUNDRED) * debt / equity
        )

    @staticmethod
    def cost_of_equity(
        risk_free_rate: DecimalInput,
        levered_beta: DecimalInput,
        equity_risk_premium: DecimalInput,
        country_risk_premium: DecimalInput = Decimal(0),
    ) -> CostOfEquityResult:
        risk_free = _decimal(risk_free_rate, "risk_free_rate")
        beta = _decimal(levered_beta, "levered_beta")
        equity_premium = _decimal(equity_risk_premium, "equity_risk_premium")
        country_premium = _decimal(country_risk_premium, "country_risk_premium")
        result = risk_free + beta * equity_premium + country_premium
        DiscountRateService._validate_discount_rate(result, "cost_of_equity")
        return CostOfEquityResult(
            risk_free_rate=risk_free,
            levered_beta=beta,
            equity_risk_premium=equity_premium,
            country_risk_premium=country_premium,
            cost_of_equity=result,
        )

    @staticmethod
    def after_tax_cost_of_debt(
        pretax_cost_of_debt: DecimalInput,
        normalized_tax_rate: DecimalInput,
    ) -> Decimal:
        pretax = _decimal(pretax_cost_of_debt, "pretax_cost_of_debt")
        tax_rate = _decimal(normalized_tax_rate, "normalized_tax_rate")
        DiscountRateService._validate_discount_rate(pretax, "pretax_cost_of_debt")
        DiscountRateService._validate_tax_rate(tax_rate)
        return pretax * (Decimal(1) - tax_rate / _ONE_HUNDRED)

    @classmethod
    def wacc(
        cls,
        cost_of_equity: DecimalInput,
        pretax_cost_of_debt: DecimalInput,
        normalized_tax_rate: DecimalInput,
        market_value_equity: DecimalInput,
        market_value_debt: DecimalInput,
    ) -> WaccResult:
        equity_cost = _decimal(cost_of_equity, "cost_of_equity")
        debt_cost = _decimal(pretax_cost_of_debt, "pretax_cost_of_debt")
        tax_rate = _decimal(normalized_tax_rate, "normalized_tax_rate")
        equity = _decimal(market_value_equity, "market_value_equity")
        debt = _decimal(market_value_debt, "market_value_debt")
        cls._validate_discount_rate(equity_cost, "cost_of_equity")
        cls._validate_discount_rate(debt_cost, "pretax_cost_of_debt")
        cls._validate_tax_rate(tax_rate)
        if equity < 0 or debt < 0:
            raise ValueError("Market values cannot be negative")
        total_capital = equity + debt
        if total_capital <= 0:
            raise ValueError("WACC requires positive total capital")
        with localcontext() as context:
            context.prec = 34
            equity_weight = equity / total_capital
            debt_weight = Decimal(1) - equity_weight
            after_tax_debt = cls.after_tax_cost_of_debt(debt_cost, tax_rate)
            result = equity_weight * equity_cost + debt_weight * after_tax_debt
        cls._validate_discount_rate(result, "wacc")
        return WaccResult(
            cost_of_equity=equity_cost,
            pretax_cost_of_debt=debt_cost,
            normalized_tax_rate=tax_rate,
            after_tax_cost_of_debt=after_tax_debt,
            market_value_equity=equity,
            market_value_debt=debt,
            equity_weight=equity_weight,
            debt_weight=debt_weight,
            wacc=result,
        )

    @staticmethod
    def _validate_tax_rate(rate: Decimal) -> None:
        if not Decimal(0) <= rate <= _ONE_HUNDRED:
            raise ValueError("normalized_tax_rate must be between 0% and 100%")

    @staticmethod
    def _validate_discount_rate(rate: Decimal, name: str) -> None:
        if rate <= Decimal("-100"):
            raise ValueError(f"{name} must be greater than -100%")


class PresentValueService:
    """Discount annual or fractional-period cash flows at one constant rate."""

    @staticmethod
    def discount_factor(
        discount_rate: DecimalInput,
        period: DecimalInput,
    ) -> Decimal:
        rate = _decimal(discount_rate, "discount_rate")
        elapsed = _decimal(period, "period")
        DiscountRateService._validate_discount_rate(rate, "discount_rate")
        if elapsed < 0:
            raise ValueError("period cannot be negative")
        with localcontext() as context:
            context.prec = 34
            base = Decimal(1) + rate / _ONE_HUNDRED
            return Decimal(1) / (base**elapsed)

    @classmethod
    def present_value(
        cls,
        amount: DecimalInput,
        discount_rate: DecimalInput,
        period: DecimalInput,
    ) -> Decimal:
        cash_flow = _decimal(amount, "amount")
        with localcontext() as context:
            context.prec = 34
            return cash_flow * cls.discount_factor(discount_rate, period)

    @classmethod
    def discount(
        cls,
        cash_flows: Iterable[CashFlow],
        discount_rate: DecimalInput,
        unit: str,
    ) -> PresentValueResult:
        rate = _decimal(discount_rate, "discount_rate")
        DiscountRateService._validate_discount_rate(rate, "discount_rate")
        discounted = []
        for cash_flow in cash_flows:
            factor = cls.discount_factor(rate, cash_flow.period)
            with localcontext() as context:
                context.prec = 34
                value = cash_flow.amount * factor
            discounted.append(
                DiscountedCashFlow(
                    amount=cash_flow.amount,
                    period=cash_flow.period,
                    discount_rate=rate,
                    discount_factor=factor,
                    present_value=value,
                    label=cash_flow.label,
                )
            )
        total = sum((item.present_value for item in discounted), Decimal(0))
        return PresentValueResult(
            discount_rate=rate,
            unit=unit,
            cash_flows=tuple(discounted),
            total_present_value=total,
        )


class TerminalValueService:
    """Calculate an undiscounted terminal value for later PV conversion."""

    @staticmethod
    def perpetuity_growth(
        final_cash_flow: DecimalInput,
        discount_rate: DecimalInput,
        perpetual_growth_rate: DecimalInput,
    ) -> TerminalValueResult:
        cash_flow = _decimal(final_cash_flow, "final_cash_flow")
        rate = _decimal(discount_rate, "discount_rate")
        growth = _decimal(perpetual_growth_rate, "perpetual_growth_rate")
        DiscountRateService._validate_discount_rate(rate, "discount_rate")
        if cash_flow < 0:
            raise ValueError("final_cash_flow cannot be negative")
        if growth <= Decimal("-100"):
            raise ValueError("perpetual_growth_rate must be greater than -100%")
        if rate <= growth:
            raise ValueError("discount_rate must exceed perpetual_growth_rate")
        with localcontext() as context:
            context.prec = 34
            terminal_value = (
                cash_flow
                * (Decimal(1) + growth / _ONE_HUNDRED)
                / ((rate - growth) / _ONE_HUNDRED)
            )
        return TerminalValueResult(
            method=TerminalValueMethod.PERPETUITY_GROWTH,
            terminal_value=terminal_value,
            final_cash_flow=cash_flow,
            discount_rate=rate,
            perpetual_growth_rate=growth,
            formula="final cash flow × (1 + growth) ÷ (discount rate − growth)",
        )

    @staticmethod
    def exit_multiple(
        terminal_metric: DecimalInput,
        exit_multiple: DecimalInput,
    ) -> TerminalValueResult:
        metric = _decimal(terminal_metric, "terminal_metric")
        multiple = _decimal(exit_multiple, "exit_multiple")
        if metric < 0 or multiple < 0:
            raise ValueError("Exit-multiple inputs cannot be negative")
        terminal_value = metric * multiple
        return TerminalValueResult(
            method=TerminalValueMethod.EXIT_MULTIPLE,
            terminal_value=terminal_value,
            terminal_metric=metric,
            exit_multiple=multiple,
            formula="terminal metric × exit multiple",
        )


__all__ = [
    "DiscountRateService",
    "PresentValueService",
    "TerminalValueService",
]
