"""CLI use cases for financial, classification, and inventory retrieval."""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Optional

from edgarito.cli.presentation.console import (
    ClassificationConsolePresenter,
    FinancialsConsolePresenter,
    MetricsConsolePresenter,
    RedFlagsConsolePresenter,
)
from edgarito.cli.use_cases.context import call_with_context, dependency
from edgarito.config.red_flags import RedFlagsProfileLoader
from edgarito.enums.granularity import Granularity
from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.export import CompanyAnalysisReportService, ExcelRenderer
from edgarito.services.financials.availability import (
    FinancialObservationAvailabilityService,
    ObservationAvailabilityMode,
)
from edgarito.services.guidance.documents import (
    GuidanceDocumentSelector,
    is_exhibit_document,
)
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.reconciliation.classification import CompanyClassificationService
from edgarito.services.reconciliation.financials import FinancialDataService
from edgarito.services.red_flags import InvestmentRedFlagsService
from edgarito.settings import (
    ALPHAVANTAGE_API_KEY,
    CLASSIFICATION_PROVIDER_CONFIGURATION,
    FMP_API_KEY,
    OPENFIGI_API_KEY,
    PROVIDER_CONFIGURATION,
)


def validate_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("--limit must be at least 1")


def parse_mappings(values: Optional[list[str]], option: str) -> dict[str, str]:
    mappings = {}
    for value in values or []:
        key, separator, mapped_value = value.partition("=")
        key = key.strip()
        mapped_value = mapped_value.strip()
        if not separator or not key or not mapped_value:
            raise ValueError(f"{option} must use NAME=SYMBOL syntax")
        normalized_key = key.lower()
        if normalized_key in mappings:
            raise ValueError(f"Duplicate {option} mapping for {key}")
        mappings[normalized_key] = mapped_value
    return mappings


def parse_provider_symbols(
    values: Optional[list[str]],
    *,
    context=None,
) -> dict[ProviderName, str]:
    raw_mappings = call_with_context(
        dependency(context, "_parse_mappings", parse_mappings),
        values,
        "--provider-symbol",
        context=context,
    )
    try:
        return {
            ProviderName(provider): symbol for provider, symbol in raw_mappings.items()
        }
    except ValueError as exc:
        choices = ", ".join(provider.value for provider in ProviderName)
        raise ValueError(
            f"Unknown provider in --provider-symbol; choose one of: {choices}"
        ) from exc


async def retrieve_classification(
    args: argparse.Namespace,
    *,
    provider: Optional[ProviderName],
    crosscheck: bool,
    context=None,
):
    cache_type = dependency(context, "FileSystemCache", FileSystemCache)
    service_type = dependency(
        context,
        "CompanyClassificationService",
        CompanyClassificationService,
    )
    mappings = dependency(context, "_parse_mappings", parse_mappings)
    provider_symbols = dependency(
        context,
        "_parse_provider_symbols",
        parse_provider_symbols,
    )
    async with service_type(
        cache=cache_type(Path(args.cache_dir)),
        provider_configuration=dependency(
            context,
            "CLASSIFICATION_PROVIDER_CONFIGURATION",
            CLASSIFICATION_PROVIDER_CONFIGURATION,
        ),
        alphavantage_api_key=dependency(
            context, "ALPHAVANTAGE_API_KEY", ALPHAVANTAGE_API_KEY
        ),
        fmp_api_key=dependency(context, "FMP_API_KEY", FMP_API_KEY),
        openfigi_api_key=dependency(context, "OPENFIGI_API_KEY", OPENFIGI_API_KEY),
    ) as service:
        classification = await service.retrieve(
            ticker=args.ticker,
            cik=args.cik,
            isin=args.isin,
            exchange=args.exchange,
            exchange_symbols=call_with_context(
                mappings,
                args.exchange_symbol,
                "--exchange-symbol",
                context=context,
            ),
            provider_symbols=call_with_context(
                provider_symbols,
                args.provider_symbol,
                context=context,
            ),
            provider=provider,
            use_cache=not args.refresh,
            make_cache=True,
            crosscheck=crosscheck,
        )
    return classification


async def retrieve_financials(
    args: argparse.Namespace,
    granularity: Optional[Granularity],
    concepts: Optional[set[FinancialConcept]],
    *,
    context=None,
) -> NormalizedCompanyFinancials:
    cache_type = dependency(context, "FileSystemCache", FileSystemCache)
    service_type = dependency(context, "FinancialDataService", FinancialDataService)
    mappings = dependency(context, "_parse_mappings", parse_mappings)
    provider_symbols = dependency(
        context,
        "_parse_provider_symbols",
        parse_provider_symbols,
    )
    cache = cache_type(Path(args.cache_dir))
    async with service_type(
        cache=cache,
        provider_configuration=dependency(
            context, "PROVIDER_CONFIGURATION", PROVIDER_CONFIGURATION
        ),
        user_agent=args.user_agent,
        alphavantage_api_key=dependency(
            context, "ALPHAVANTAGE_API_KEY", ALPHAVANTAGE_API_KEY
        ),
        fmp_api_key=dependency(context, "FMP_API_KEY", FMP_API_KEY),
        openfigi_api_key=dependency(context, "OPENFIGI_API_KEY", OPENFIGI_API_KEY),
    ) as service:
        return await service.retrieve(
            ticker=args.ticker,
            cik=args.cik,
            isin=args.isin,
            exchange=args.exchange,
            exchange_symbols=call_with_context(
                mappings,
                args.exchange_symbol,
                "--exchange-symbol",
                context=context,
            ),
            provider_symbols=call_with_context(
                provider_symbols,
                args.provider_symbol,
                context=context,
            ),
            market=Market(args.market),
            provider=ProviderName(args.provider) if args.provider else None,
            granularity=granularity,
            concepts=concepts,
            use_cache=not args.refresh,
            make_cache=True,
            crosscheck=args.crosscheck,
        )


def merge_normalized_financials(
    *financials: NormalizedCompanyFinancials,
    as_of: datetime.date | None = None,
    availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
    availability_service: FinancialObservationAvailabilityService | None = None,
) -> NormalizedCompanyFinancials:
    """Merge deterministic annual/quarterly normalized retrieval results.

    Observation identity includes the reporting granularity and period dates,
    so annual and quarterly facts are never collapsed into one another. Exact
    duplicates are resolved by filed date, accession, and input position while
    issuer metadata remains anchored to the first result.
    """

    normalized = tuple(
        item
        if isinstance(item, NormalizedCompanyFinancials)
        else NormalizedCompanyFinancials.model_validate(item)
        for item in financials
        if item is not None
    )
    if not normalized:
        raise ValueError("At least one normalized financial result is required")
    availability = availability_service or FinancialObservationAvailabilityService()
    selected: dict[tuple, tuple[tuple, object]] = {}
    for result_position, result in enumerate(normalized):
        for observation_position, observation in enumerate(result.observations):
            if as_of is not None and not availability.is_available(
                observation,
                as_of=as_of,
                mode=availability_mode,
                snapshot_retrieved_at=result.retrieved_at,
            ):
                continue
            key = (
                observation.concept.value,
                observation.granularity.value,
                observation.fiscal_year,
                observation.fiscal_period.value,
                observation.period_start,
                observation.period_end,
                observation.unit,
            )
            rank = (
                observation.filed is not None,
                observation.filed or datetime.date.min,
                observation.accession_number or "",
                -result_position,
                -observation_position,
            )
            previous = selected.get(key)
            if previous is None or rank > previous[0]:
                selected[key] = (rank, observation)
    observations = sorted(
        (item for _rank, item in selected.values()),
        key=lambda item: (
            item.period_end,
            item.granularity.value,
            item.fiscal_year,
            item.fiscal_period.value,
            item.concept.value,
            item.unit,
            item.provider,
            item.accession_number or "",
        ),
    )
    primary = normalized[0]
    retrieved_at = max(
        (item.retrieved_at for item in normalized if item.retrieved_at is not None),
        default=primary.retrieved_at,
    )
    return primary.model_copy(
        update={"observations": observations, "retrieved_at": retrieved_at}
    )


_merge_normalized_financials = merge_normalized_financials


async def run_financials(args: argparse.Namespace, *, context=None) -> int:
    call_with_context(
        dependency(context, "_validate_limit", validate_limit),
        args.limit,
        context=context,
    )
    granularity = call_with_context(
        dependency(context, "_granularity", granularity_for_period),
        args.period,
        context=context,
    )
    concepts = (
        {FinancialConcept(value) for value in args.concept} if args.concept else None
    )
    financials = await call_with_context(
        dependency(context, "_retrieve_financials", retrieve_financials),
        args,
        granularity,
        concepts,
        context=context,
    )
    presenter = dependency(context, "FinancialsConsolePresenter", FinancialsConsolePresenter)
    print(presenter().render(financials, limit=args.limit))
    return 0


async def run_sec_inventory(args: argparse.Namespace, *, context=None) -> int:
    cache_type = dependency(context, "FileSystemCache", FileSystemCache)
    edgar_type = dependency(context, "EdgarClient", EdgarClient)
    selector_type = dependency(context, "GuidanceDocumentSelector", GuidanceDocumentSelector)
    exhibit_check = dependency(context, "is_exhibit_document", is_exhibit_document)
    cache = cache_type(Path(args.cache_dir))
    async with edgar_type(cache, args.user_agent) as client:
        refresh = args.refresh or args.refresh_sec
        cik = args.cik or await client.get_cik(
            args.ticker, use_cache=not refresh, make_cache=True
        )
        filings = await client.get_raw_operating_filings(
            cik,
            as_of=datetime.date.today(),
            use_cache=not refresh,
            make_cache=True,
        )
        selector = selector_type()
        candidates = selector.select_operating_filings(filings, limit=24)
        attachment_count = 0
        exhibit_count = 0
        for filing in candidates:
            populated = await client.get_filing_documents(
                filing, use_cache=not refresh, make_cache=True
            )
            attachment_count += len(populated.documents)
            exhibit_count += sum(
                exhibit_check(document) for document in populated.documents
            )
        print("SEC OPERATING INVENTORY")
        print(f"Cache bypass: {'yes' if refresh else 'no'}")
        print(f"Raw filings: {len(filings)}")
        print(f"Operating candidates: {len(candidates)}")
        print(f"Attachments enumerated: {attachment_count}")
        print(f"EX-99.x found: {exhibit_count}")
        for item in filings:
            print(
                f"{item.filing_date.isoformat()} | {item.form} | "
                f"{item.accession_number} | {item.primary_document}"
            )
    return 0


async def run_metrics(args: argparse.Namespace, *, context=None) -> int:
    call_with_context(
        dependency(context, "_validate_limit", validate_limit),
        args.limit,
        context=context,
    )
    granularity = call_with_context(
        dependency(context, "_granularity", granularity_for_period),
        args.period,
        context=context,
    )
    metrics_service = dependency(context, "FinancialMetricsService", FinancialMetricsService)
    selected_metrics = (
        {FinancialMetric(value) for value in args.metric} if args.metric else None
    )
    concepts = metrics_service.required_concepts(selected_metrics)
    financials = await call_with_context(
        dependency(context, "_retrieve_financials", retrieve_financials),
        args,
        granularity,
        concepts,
        context=context,
    )
    metrics = metrics_service().calculate(
        financials,
        granularity=granularity,
        metrics=selected_metrics,
    )
    presenter = dependency(context, "MetricsConsolePresenter", MetricsConsolePresenter)
    print(presenter().render(metrics, limit=args.limit))
    return 0


async def run_export(args: argparse.Namespace, *, context=None) -> int:
    granularity = call_with_context(
        dependency(context, "_granularity", granularity_for_period),
        args.period,
        context=context,
    )
    financials = await call_with_context(
        dependency(context, "_retrieve_financials", retrieve_financials),
        args,
        granularity,
        None,
        context=context,
    )
    metrics_type = dependency(context, "FinancialMetricsService", FinancialMetricsService)
    metrics = metrics_type().calculate(
        financials,
        granularity=granularity,
    )
    report_service = dependency(
        context, "CompanyAnalysisReportService", CompanyAnalysisReportService
    )
    report = report_service().compose(
        financials=financials,
        metrics=metrics,
    )
    renderer = dependency(context, "ExcelRenderer", ExcelRenderer)
    output = renderer().render(report, args.output)
    print(f"Exported Excel workbook to {output}")
    return 0


async def run_red_flags(args: argparse.Namespace, *, context=None) -> int:
    profile_loader = dependency(context, "RedFlagsProfileLoader", RedFlagsProfileLoader)
    configuration = profile_loader.load(args.profile)
    granularity = Granularity(args.period)
    concepts = {
        concept
        for category in configuration.enabled_categories
        for concept in configuration.required_concepts(category)
    }
    financials = await call_with_context(
        dependency(context, "_retrieve_financials", retrieve_financials),
        args,
        granularity,
        concepts,
        context=context,
    )
    service_type = dependency(
        context, "InvestmentRedFlagsService", InvestmentRedFlagsService
    )
    report = service_type(configuration).analyze(
        financials,
        granularity=granularity,
    )
    presenter = dependency(context, "RedFlagsConsolePresenter", RedFlagsConsolePresenter)
    print(presenter().render(report, verbose=args.verbose))
    return 0


async def run_classification(args: argparse.Namespace, *, context=None) -> int:
    classification = await call_with_context(
        dependency(context, "_retrieve_classification", retrieve_classification),
        args,
        provider=ProviderName(args.provider) if args.provider else None,
        crosscheck=args.crosscheck,
        context=context,
    )
    presenter = dependency(
        context, "ClassificationConsolePresenter", ClassificationConsolePresenter
    )
    print(presenter().render(classification))
    return 0


def granularity_for_period(period: str) -> Optional[Granularity]:
    return None if period == "all" else Granularity(period)


# Historical private names remain available from the focused module as well as
# from the application facade.
_validate_limit = validate_limit
_parse_mappings = parse_mappings
_parse_provider_symbols = parse_provider_symbols
_retrieve_classification = retrieve_classification
_retrieve_financials = retrieve_financials
_run_classification = run_classification
_run_export = run_export
_run_financials = run_financials
_run_metrics = run_metrics
_run_red_flags = run_red_flags
_run_sec_inventory = run_sec_inventory
granularity = granularity_for_period
_granularity = granularity_for_period


__all__ = [
    "_granularity",
    "_parse_mappings",
    "_parse_provider_symbols",
    "_retrieve_classification",
    "_retrieve_financials",
    "_run_classification",
    "_run_export",
    "_run_financials",
    "_run_metrics",
    "_run_red_flags",
    "_run_sec_inventory",
    "_validate_limit",
    "granularity",
    "parse_mappings",
    "parse_provider_symbols",
    "retrieve_classification",
    "retrieve_financials",
    "run_classification",
    "run_export",
    "run_financials",
    "run_metrics",
    "run_red_flags",
    "run_sec_inventory",
    "validate_limit",
]
