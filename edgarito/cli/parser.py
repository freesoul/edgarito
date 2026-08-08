import argparse
import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from edgarito.enums.market import Market
from edgarito.enums.provider import ProviderName
from edgarito.schemas.normalization.financials import FinancialConcept
from edgarito.schemas.valuation.specialized import SpecializedInputType
from edgarito.services.metrics import FinancialMetric
from edgarito.services.valuation import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    EconomicTrait,
    ValuationInput,
)
from edgarito.settings import (
    CLASSIFICATION_PROVIDER_CONFIGURATION,
    EDGARITO_CACHE_DIR,
    EDGARITO_USER_AGENT,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgarito", description="Retrieve and analyze normalized financials"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    financials = subparsers.add_parser(
        "financials", help="Display normalized historical financials"
    )
    metrics = subparsers.add_parser(
        "metrics", help="Calculate metrics from normalized financials"
    )
    red_flags = subparsers.add_parser(
        "red-flags", help="Detect investment red flags in normalized financials"
    )
    forecast = subparsers.add_parser(
        "forecast", help="Project annual driver-based FCFF"
    )
    valuation = subparsers.add_parser(
        "valuation", help="Calculate an intrinsic or relative valuation"
    )
    valuation_models = subparsers.add_parser(
        "valuation-models",
        help="Rank suitable valuation models and report missing inputs",
    )
    classification = subparsers.add_parser(
        "classification", help="Retrieve normalized company sector and industry"
    )
    comparables = subparsers.add_parser(
        "comparables",
        help="Select peers and compute keyless Yahoo-backed LTM multiples",
    )
    specialized_inputs = subparsers.add_parser(
        "specialized-inputs",
        help="Extract REIT, resource, biotech, or SOTP valuation inputs",
    )

    for command_parser in (financials, metrics):
        _add_retrieval_arguments(command_parser)
    _add_retrieval_arguments(red_flags, include_period=False)
    red_flags.add_argument(
        "--period",
        choices=("annual", "quarterly"),
        default="annual",
        help="Period granularity to analyze (default: annual)",
    )
    _add_red_flags_profile_argument(red_flags)
    _add_retrieval_arguments(forecast, include_period=False)
    _add_retrieval_arguments(valuation, include_period=False)
    _add_retrieval_arguments(valuation_models, include_period=False)
    for command_parser in (
        forecast,
        valuation,
        valuation_models,
        comparables,
        specialized_inputs,
    ):
        _add_valuation_profile_argument(command_parser)

    financials.add_argument(
        "--concept",
        action="append",
        choices=[concept.value for concept in FinancialConcept],
        help="Limit output to a concept; repeat this option for multiple concepts",
    )
    metrics.add_argument(
        "--metric",
        action="append",
        choices=[metric.value for metric in FinancialMetric],
        help="Limit output to a metric; repeat this option for multiple metrics",
    )
    forecast.add_argument(
        "--forecast-method",
        "--method",
        choices=("fcff", "simplified"),
        help="Forecast method; overrides the selected profile",
    )
    forecast.add_argument(
        "--years",
        type=int,
        help="Number of annual forecast periods; overrides the selected profile",
    )
    forecast.add_argument(
        "--revenue-growth",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help=(
            "Revenue growth in percentage points; provide once for a constant "
            "rate or once per forecast year"
        ),
    )
    forecast.add_argument(
        "--operating-margin",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="EBIT margin; provide once or once per forecast year",
    )
    forecast.add_argument(
        "--tax-rate",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="Normalized operating tax rate; provide once or once per forecast year",
    )
    forecast.add_argument(
        "--depreciation-to-revenue",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="D&A as a percentage of revenue; provide once or per forecast year",
    )
    forecast.add_argument(
        "--capex-to-revenue",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="Capex as a percentage of revenue; provide once or per forecast year",
    )
    forecast.add_argument(
        "--operating-working-capital-to-revenue",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help=(
            "Operating working capital as a percentage of revenue; provide once "
            "or per forecast year"
        ),
    )
    forecast.add_argument(
        "--fcf-margin",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help=(
            "FCF margin for --forecast-method simplified; provide once or once "
            "per forecast year"
        ),
    )
    forecast.add_argument(
        "--historical-window",
        type=int,
        help="Annual periods used to infer omitted assumptions; overrides the profile",
    )
    valuation.add_argument(
        "--model",
        choices=(
            "auto",
            "fcff-dcf",
            "fcfe-dcf",
            "ddm",
            "residual-income",
            "nav",
            "reit-affo",
            "property-nav",
            "resource-nav",
            "pipeline-rnpv",
            "comparables",
            "both",
        ),
        default="auto",
        help=(
            "Valuation model; auto executes every ready suitable model independently. "
            "both remains an alias for FCFF plus comparables"
        ),
    )
    valuation.add_argument(
        "--years",
        type=int,
        help=(
            "Minimum annual projection horizon; adaptive valuation extends it "
            "when needed to reach the stable stage"
        ),
    )
    valuation.add_argument(
        "--peer",
        action="append",
        help="Candidate Yahoo peer symbol for relative valuation; repeat per peer",
    )
    valuation.add_argument(
        "--relative-basis",
        choices=(
            "ev_to_ebitda",
            "ev_to_ebit",
            "ev_to_revenue",
            "ev_to_fcf",
            "price_to_earnings",
            "price_to_book",
            "price_to_tangible_book",
            "price_to_affo",
            "price_to_nav",
        ),
        help="Forward multiple basis; overrides the relative-valuation policy",
    )
    valuation.add_argument(
        "--horizon-years",
        type=_decimal_value,
        metavar="YEARS",
        help="Forward target-price horizon in years",
    )
    valuation.add_argument(
        "--analyst-target-price",
        type=_decimal_value,
        metavar="PRICE",
        help="Show the forward multiple implied by an external analyst target",
    )
    valuation.add_argument(
        "--projection-method",
        choices=("adaptive", "constant"),
        help=(
            "FCFF projection strategy; defaults to adaptive multistage from the "
            "selected profile"
        ),
    )
    valuation.add_argument(
        "--revenue-growth",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="Revenue growth; provide once or once per forecast year",
    )
    valuation.add_argument(
        "--operating-margin",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="EBIT margin; provide once or once per forecast year",
    )
    valuation.add_argument(
        "--tax-rate",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="Operating tax rate; provide once or once per forecast year",
    )
    valuation.add_argument(
        "--depreciation-to-revenue",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="D&A as a percentage of revenue; provide once or per forecast year",
    )
    valuation.add_argument(
        "--capex-to-revenue",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="Capex as a percentage of revenue; provide once or per forecast year",
    )
    valuation.add_argument(
        "--operating-working-capital-to-revenue",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="Operating working capital / revenue; provide once or per year",
    )
    valuation.add_argument(
        "--historical-window",
        type=int,
        help="Annual periods used to infer omitted forecast assumptions",
    )
    valuation.add_argument(
        "--wacc",
        type=_percentage,
        metavar="PERCENT",
        help="WACC in percentage points; overrides the selected profile",
    )
    valuation.add_argument(
        "--cost-of-equity",
        type=_percentage,
        metavar="PERCENT",
        help="Equity discount rate; overrides profile and CAPM resolution",
    )
    valuation.add_argument(
        "--fcfe",
        type=_decimal_value,
        action="append",
        metavar="AMOUNT",
        help="Explicit annual FCFE; repeat once per forecast year",
    )
    valuation.add_argument(
        "--net-borrowing",
        type=_decimal_value,
        action="append",
        metavar="AMOUNT",
        help="Explicit annual net borrowing for a reconciled FCFE path",
    )
    valuation.add_argument(
        "--debt-financing-ratio",
        type=_decimal_value,
        metavar="RATIO",
        help="Share of reinvestment financed with debt, from 0 to 1",
    )
    valuation.add_argument(
        "--dividend",
        type=_decimal_value,
        action="append",
        metavar="AMOUNT",
        help="Explicit annual total common dividends; repeat per forecast year",
    )
    valuation.add_argument(
        "--forecast-roe",
        type=_percentage,
        action="append",
        metavar="PERCENT",
        help="Forecast ROE in percentage points; repeat per forecast year",
    )
    valuation.add_argument(
        "--payout-ratio",
        type=_decimal_value,
        action="append",
        metavar="RATIO",
        help="Dividend payout ratio from 0 to 1; repeat per forecast year",
    )
    valuation.add_argument(
        "--terminal-roe",
        type=_percentage,
        metavar="PERCENT",
        help="Sustainable terminal return on equity",
    )
    valuation.add_argument(
        "--excess-return-persistence",
        type=_decimal_value,
        metavar="RATIO",
        help="Annual persistence of residual excess ROE, from 0 to 1",
    )
    valuation.add_argument(
        "--cash-flow-timing",
        choices=("end_of_period", "mid_year"),
        help="Explicit FCFF discount timing; overrides the selected profile",
    )
    valuation.add_argument(
        "--terminal-method",
        choices=("perpetuity_growth", "exit_multiple"),
        help="Terminal-value method; overrides the selected profile",
    )
    valuation.add_argument(
        "--terminal-growth",
        type=_percentage,
        metavar="PERCENT",
        help="Perpetual growth in percentage points",
    )
    valuation.add_argument(
        "--terminal-roic",
        type=_percentage,
        metavar="PERCENT",
        help="Sustainable terminal ROIC; overrides automatic resolution and profile",
    )
    valuation.add_argument(
        "--exit-multiple",
        type=_decimal_value,
        metavar="MULTIPLE",
        help="Terminal exit multiple",
    )
    valuation.add_argument(
        "--exit-metric",
        choices=("ebitda", "ebit", "fcff", "revenue"),
        help="Terminal metric for an exit multiple",
    )
    valuation.add_argument(
        "--net-debt",
        type=_decimal_value,
        metavar="AMOUNT",
        help="Override normalized net debt in reporting currency",
    )
    valuation.add_argument(
        "--gross-debt",
        type=_decimal_value,
        metavar="AMOUNT",
        help="Manual gross debt; must be supplied with --cash",
    )
    valuation.add_argument(
        "--cash",
        type=_decimal_value,
        metavar="AMOUNT",
        help="Manual cash and equivalents; must be supplied with --gross-debt",
    )
    valuation.add_argument(
        "--shares",
        type=_decimal_value,
        metavar="COUNT",
        help="Override normalized diluted shares",
    )
    valuation.add_argument(
        "--non-operating-assets",
        type=_decimal_value,
        metavar="AMOUNT",
        help="Override short-term and other non-operating investments",
    )
    valuation.add_argument(
        "--buyback-cash",
        type=_decimal_value,
        action="append",
        metavar="AMOUNT",
        help="Annual repurchase cash in reporting currency; repeat per year",
    )
    valuation.add_argument(
        "--buyback-price",
        type=_decimal_value,
        metavar="PRICE",
        help=(
            "Share price at the valuation date for projecting repurchase execution; "
            "defaults to model-implied fair value"
        ),
    )
    valuation.add_argument(
        "--buyback-price-growth",
        type=_percentage,
        metavar="PERCENT",
        help="Annual growth in the modeled repurchase price",
    )
    valuation.add_argument(
        "--buyback-discount-rate",
        type=_percentage,
        metavar="PERCENT",
        help="Rate used to discount repurchase cash; defaults to cost of equity",
    )
    valuation.add_argument(
        "--no-buybacks",
        action="store_true",
        help="Disable a share-repurchase schedule supplied by the profile",
    )
    valuation.add_argument(
        "--scenarios",
        action="store_true",
        help="Show the detailed bear/base/bull assumptions and values",
    )
    valuation.add_argument(
        "--sensitivity",
        action="store_true",
        help="Show the WACC by terminal-growth value/share sensitivity table",
    )
    valuation.add_argument(
        "--reverse-dcf",
        action="store_true",
        help="Show all independently solved market-implied DCF assumptions",
    )
    valuation.add_argument(
        "--audit",
        action="store_true",
        help=(
            "Show full valuation diagnostics and provenance without enabling "
            "debug logging"
        ),
    )
    valuation.add_argument(
        "--financial-snapshot-max-age-hours",
        type=int,
        default=24,
        metavar="HOURS",
        help=(
            "Warn when a cached current financial snapshot is older than this "
            "threshold (default: 24); use --refresh to retrieve a new snapshot"
        ),
    )
    valuation_models.add_argument(
        "--classification-provider",
        choices=[
            provider.value
            for provider in CLASSIFICATION_PROVIDER_CONFIGURATION.available_providers
        ],
        help="Override the configured classification provider",
    )
    valuation_models.add_argument(
        "--business-type",
        choices=[item.value for item in BusinessArchetype],
        help="Override the inferred economic business type",
    )
    valuation_models.add_argument(
        "--lifecycle",
        choices=[item.value for item in CompanyLifecycle],
        help="Override the inferred company lifecycle",
    )
    valuation_models.add_argument(
        "--cyclicality",
        choices=[item.value for item in Cyclicality],
        help="Override inferred cyclicality",
    )
    valuation_models.add_argument(
        "--trait",
        action="append",
        choices=[item.value for item in EconomicTrait],
        help="Add a known economic trait; repeat for multiple traits",
    )
    valuation_models.add_argument(
        "--available-input",
        action="append",
        choices=[item.value for item in ValuationInput],
        help="Mark external valuation data as available; repeat as needed",
    )
    valuation_models.add_argument(
        "--peer-count",
        type=int,
        help="Number of genuinely comparable companies available",
    )
    _add_identifier_arguments(classification)
    classification.add_argument(
        "--provider",
        choices=[
            provider.value
            for provider in CLASSIFICATION_PROVIDER_CONFIGURATION.available_providers
        ],
        help="Override the configured classification provider",
    )
    classification.add_argument("--refresh", action="store_true")
    classification.add_argument("--crosscheck", action="store_true")
    classification.add_argument("--cache-dir", default=EDGARITO_CACHE_DIR)
    classification.add_argument("--verbose", action="store_true")
    comparables.add_argument("--ticker", required=True, help="Target Yahoo symbol")
    comparables.add_argument(
        "--peer",
        action="append",
        help=(
            "Candidate Yahoo symbol; repeat to override automatic provider-backed "
            "discovery"
        ),
    )
    comparables.add_argument("--max-peers", type=int, help="Maximum selected peers")
    comparables.add_argument(
        "--preferred-minimum",
        type=int,
        help="Preferred minimum selected peers",
    )
    comparables.add_argument(
        "--minimum-score",
        type=int,
        help="Minimum comparability score from 0 to 100",
    )
    sector_requirement = comparables.add_mutually_exclusive_group()
    sector_requirement.add_argument(
        "--allow-cross-sector",
        dest="require_same_sector",
        action="store_false",
        help="Do not hard-exclude candidates from a different sector",
    )
    sector_requirement.add_argument(
        "--require-same-sector",
        dest="require_same_sector",
        action="store_true",
        help="Hard-exclude candidates from a different sector",
    )
    comparables.set_defaults(require_same_sector=None)
    comparables.add_argument(
        "--as-of",
        type=datetime.date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Use the latest price on or before this date",
    )
    comparables.add_argument("--refresh", action="store_true")
    comparables.add_argument("--cache-dir", default=EDGARITO_CACHE_DIR)
    comparables.add_argument("--verbose", action="store_true")
    specialized_identifier = specialized_inputs.add_mutually_exclusive_group(
        required=True
    )
    specialized_identifier.add_argument("--ticker", help="US-listed SEC ticker")
    specialized_identifier.add_argument("--cik", type=int, help="SEC Central Index Key")
    specialized_inputs.add_argument(
        "--type",
        required=True,
        choices=[item.value for item in SpecializedInputType],
        help="Specialized valuation input profile",
    )
    specialized_inputs.add_argument(
        "--history",
        type=int,
        help="Number of latest reporting period ends; overrides the profile",
    )
    specialized_inputs.add_argument("--refresh", action="store_true")
    specialized_inputs.add_argument("--cache-dir", default=EDGARITO_CACHE_DIR)
    specialized_inputs.add_argument(
        "--user-agent",
        default=EDGARITO_USER_AGENT,
        help="SEC user agent in 'Name (email@example.com)' form",
    )
    specialized_inputs.add_argument("--verbose", action="store_true")
    return parser


def _add_retrieval_arguments(
    command_parser: argparse.ArgumentParser, *, include_period: bool = True
) -> None:
    _add_identifier_arguments(command_parser)

    command_parser.add_argument(
        "--market",
        choices=[market.value for market in Market],
        default=Market.US.value,
        help="Stock market configuration to use (default: us)",
    )
    command_parser.add_argument(
        "--provider",
        choices=[provider.value for provider in ProviderName],
        help="Override the configured default provider",
    )

    if include_period:
        command_parser.add_argument(
            "--period",
            choices=("annual", "quarterly", "all"),
            default="annual",
            help="Period granularity to display (default: annual)",
        )
        command_parser.add_argument(
            "--limit", type=int, default=5, help="Number of latest periods to display"
        )
    command_parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached provider snapshots"
    )
    command_parser.add_argument(
        "--crosscheck",
        action="store_true",
        help="Compare with the other configured providers and emit warnings",
    )
    command_parser.add_argument(
        "--cache-dir",
        default=EDGARITO_CACHE_DIR,
        help="Snapshot cache directory (default: cache)",
    )
    command_parser.add_argument(
        "--user-agent",
        default=EDGARITO_USER_AGENT,
        help=(
            "SEC user agent in 'Name (email@example.com)' form; "
            "or configure user_agent in .env"
        ),
    )
    command_parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Enable debug logging; valuation also shows full diagnostics and provenance"
        ),
    )


def _add_valuation_profile_argument(
    command_parser: argparse.ArgumentParser,
) -> None:
    command_parser.add_argument(
        "--profile",
        type=Path,
        metavar="PATH",
        help=(
            "Forecast/valuation JSON profile; valuation otherwise uses an existing "
            "configs/valuation/<ticker>.json or generates one from the default"
        ),
    )


def _add_red_flags_profile_argument(
    command_parser: argparse.ArgumentParser,
) -> None:
    command_parser.add_argument(
        "--profile",
        "--config",
        dest="profile",
        type=Path,
        metavar="PATH",
        help="Red-flags JSON profile; defaults to configs/red_flags/default.json",
    )


def _add_identifier_arguments(command_parser: argparse.ArgumentParser) -> None:
    identifier = command_parser.add_mutually_exclusive_group(required=True)
    identifier.add_argument("--ticker", help="Stock ticker, for example AAPL")
    identifier.add_argument("--cik", type=int, help="SEC Central Index Key")
    identifier.add_argument(
        "--isin", help="12-character ISIN, for example US0378331005"
    )
    command_parser.add_argument(
        "--exchange",
        help="Exchange used to disambiguate a ticker or identifier, for example XETRA",
    )
    command_parser.add_argument(
        "--exchange-symbol",
        action="append",
        metavar="EXCHANGE=SYMBOL",
        help="Map an exchange to its symbol; repeat for multiple exchanges",
    )
    command_parser.add_argument(
        "--provider-symbol",
        action="append",
        metavar="PROVIDER=SYMBOL",
        help="Map a provider to its symbol; repeat for multiple providers",
    )


def _percentage(value: str) -> Decimal:
    try:
        converted = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid percentage: {value!r}") from exc
    if not converted.is_finite():
        raise argparse.ArgumentTypeError(f"invalid percentage: {value!r}")
    return converted


def _decimal_value(value: str) -> Decimal:
    try:
        converted = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value!r}") from exc
    if not converted.is_finite():
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value!r}")
    return converted


__all__ = ["build_parser"]
