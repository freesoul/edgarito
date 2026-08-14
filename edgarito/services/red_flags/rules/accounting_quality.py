from edgarito.config.red_flags import AccountingQualityConfiguration
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
)
from edgarito.schemas.red_flags import (
    RedFlag,
    RedFlagCategory,
    RedFlagWarning,
)
from edgarito.services.red_flags.rules.context import PeriodKey


class _AccountingQualityRules:
    def _check_accounting_quality(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        periods: list[PeriodKey],
        config: AccountingQualityConfiguration,
        flags: list[RedFlag],
        warnings: list[RedFlagWarning],
    ) -> None:
        if not config.enabled:
            return
        receivable_count = 0
        inventory_count = 0
        goodwill_asset_count = 0
        for index, period in enumerate(periods):
            previous = (
                by_period[periods[index - 1]]
                if index and self._consecutive_periods(periods[index - 1], period)
                else None
            )
            current = by_period[period]
            revenue = self._single(current, FinancialConcept.REVENUE)
            prior_revenue = (
                self._single(previous, FinancialConcept.REVENUE)
                if previous is not None
                else None
            )
            revenue_growth = self._growth(revenue, prior_revenue)
            if revenue_growth is not None:
                receivables = self._single(
                    current, FinancialConcept.ACCOUNTS_RECEIVABLE
                )
                prior_receivables = (
                    self._single(previous, FinancialConcept.ACCOUNTS_RECEIVABLE)
                    if previous is not None
                    else None
                )
                receivable_growth = self._growth(receivables, prior_receivables)
                if receivable_growth is not None:
                    receivable_count += 1
                    premium = receivable_growth.value - revenue_growth.value
                    if premium > config.maximum_receivables_growth_premium_pp:
                        self._add_flag(
                            flags,
                            code="receivables_growth_ahead_of_revenue",
                            category=RedFlagCategory.ACCOUNTING_QUALITY,
                            severity=config.severity,
                            message=(
                                f"Receivables growth exceeded revenue growth by {self._format(premium)} percentage points, "
                                f"above the configured {self._format(config.maximum_receivables_growth_premium_pp)}-point ceiling."
                            ),
                            evidence=self._evidence(
                                metric="receivables_growth_premium",
                                value=premium,
                                unit="percentage_points",
                                threshold=config.maximum_receivables_growth_premium_pp,
                                comparison=">",
                                formula="receivables growth - revenue growth",
                                period=period,
                                values=(receivable_growth, revenue_growth),
                            ),
                        )
                else:
                    self._period_warning(
                        warnings,
                        code="receivables_growth_period_unavailable",
                        category=RedFlagCategory.ACCOUNTING_QUALITY,
                        period=period,
                        message=(
                            "Receivables-versus-revenue growth could not be evaluated "
                            "for this period because compatible consecutive balances "
                            "were not reported."
                        ),
                        required=(
                            FinancialConcept.ACCOUNTS_RECEIVABLE,
                            FinancialConcept.REVENUE,
                        ),
                    )

                inventory = self._single(current, FinancialConcept.INVENTORY)
                prior_inventory = (
                    self._single(previous, FinancialConcept.INVENTORY)
                    if previous is not None
                    else None
                )
                inventory_growth = self._growth(inventory, prior_inventory)
                if inventory_growth is not None:
                    inventory_count += 1
                    premium = inventory_growth.value - revenue_growth.value
                    if premium > config.maximum_inventory_growth_premium_pp:
                        self._add_flag(
                            flags,
                            code="inventory_growth_ahead_of_revenue",
                            category=RedFlagCategory.ACCOUNTING_QUALITY,
                            severity=config.severity,
                            message=(
                                f"Inventory growth exceeded revenue growth by {self._format(premium)} percentage points, "
                                f"above the configured {self._format(config.maximum_inventory_growth_premium_pp)}-point ceiling."
                            ),
                            evidence=self._evidence(
                                metric="inventory_growth_premium",
                                value=premium,
                                unit="percentage_points",
                                threshold=config.maximum_inventory_growth_premium_pp,
                                comparison=">",
                                formula="inventory growth - revenue growth",
                                period=period,
                                values=(inventory_growth, revenue_growth),
                            ),
                        )
                else:
                    self._period_warning(
                        warnings,
                        code="inventory_growth_period_unavailable",
                        category=RedFlagCategory.ACCOUNTING_QUALITY,
                        period=period,
                        message=(
                            "Inventory-versus-revenue growth could not be evaluated "
                            "for this period because compatible consecutive balances "
                            "were not reported."
                        ),
                        required=(
                            FinancialConcept.INVENTORY,
                            FinancialConcept.REVENUE,
                        ),
                    )

            elif index:
                self._period_warning(
                    warnings,
                    code="receivables_growth_period_unavailable",
                    category=RedFlagCategory.ACCOUNTING_QUALITY,
                    period=period,
                    message=(
                        "Receivables-versus-revenue growth could not be evaluated for "
                        "this period because compatible consecutive revenue periods "
                        "were not reported."
                    ),
                    required=(
                        FinancialConcept.ACCOUNTS_RECEIVABLE,
                        FinancialConcept.REVENUE,
                    ),
                )
                self._period_warning(
                    warnings,
                    code="inventory_growth_period_unavailable",
                    category=RedFlagCategory.ACCOUNTING_QUALITY,
                    period=period,
                    message=(
                        "Inventory-versus-revenue growth could not be evaluated for "
                        "this period because compatible consecutive revenue periods "
                        "were not reported."
                    ),
                    required=(
                        FinancialConcept.INVENTORY,
                        FinancialConcept.REVENUE,
                    ),
                )

            goodwill = self._single(current, FinancialConcept.GOODWILL)
            assets = self._single(current, FinancialConcept.TOTAL_ASSETS)
            ratio = self._ratio(goodwill, assets, percentage=True)
            if ratio is not None:
                goodwill_asset_count += 1
                if ratio.value > config.maximum_goodwill_to_assets_pct:
                    self._add_flag(
                        flags,
                        code="goodwill_to_assets_high",
                        category=RedFlagCategory.ACCOUNTING_QUALITY,
                        severity=config.severity,
                        message=(
                            f"Goodwill was {self._format(ratio.value)}% of assets, above the configured "
                            f"{self._format(config.maximum_goodwill_to_assets_pct)}% ceiling."
                        ),
                        evidence=self._evidence(
                            metric="goodwill_to_assets",
                            value=ratio.value,
                            unit="%",
                            threshold=config.maximum_goodwill_to_assets_pct,
                            comparison=">",
                            formula="100 × goodwill / total assets",
                            period=period,
                            values=(ratio,),
                        ),
                    )
            else:
                self._period_warning(
                    warnings,
                    code="goodwill_to_assets_period_unavailable",
                    category=RedFlagCategory.ACCOUNTING_QUALITY,
                    period=period,
                    message=(
                        "Goodwill-to-assets could not be evaluated for this period "
                        "because compatible goodwill and total-assets balances were "
                        "not reported."
                    ),
                    required=(
                        FinancialConcept.GOODWILL,
                        FinancialConcept.TOTAL_ASSETS,
                    ),
                )
        if receivable_count == 0:
            self._warning(
                warnings,
                code="receivables_growth_unavailable",
                category=RedFlagCategory.ACCOUNTING_QUALITY,
                message="Receivables-versus-revenue growth was unavailable because compatible consecutive balances were not reported.",
                required=(
                    FinancialConcept.ACCOUNTS_RECEIVABLE,
                    FinancialConcept.REVENUE,
                ),
            )
        if inventory_count == 0:
            self._warning(
                warnings,
                code="inventory_growth_unavailable",
                category=RedFlagCategory.ACCOUNTING_QUALITY,
                message="Inventory-versus-revenue growth was unavailable because compatible consecutive balances were not reported.",
                required=(FinancialConcept.INVENTORY, FinancialConcept.REVENUE),
            )
        if goodwill_asset_count == 0:
            self._warning(
                warnings,
                code="goodwill_to_assets_unavailable",
                category=RedFlagCategory.ACCOUNTING_QUALITY,
                message="Goodwill-to-assets was unavailable because compatible goodwill and total-assets balances were not reported.",
                required=(FinancialConcept.GOODWILL, FinancialConcept.TOTAL_ASSETS),
            )
