"""Typed loaders for operating terminology and extraction data tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from edgarito.config._loader import read_config_payload


class _OperatingConfigModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KpiTermConfiguration(_OperatingConfigModel):
    raw: str = Field(min_length=1)
    metric: str = Field(min_length=1)


class KeywordPatternConfiguration(_OperatingConfigModel):
    keyword: str = Field(min_length=1)
    pattern: str = Field(min_length=1)


class OperatingVocabularyConfiguration(_OperatingConfigModel):
    schema_version: Literal[1]
    global_terms: tuple[KpiTermConfiguration, ...]
    industry_terms: dict[str, tuple[KpiTermConfiguration, ...]]
    industry_aliases: dict[str, str]
    metric_aliases: dict[str, str]
    revenue_driver_priority: dict[str, int]


class UnitScaleConfiguration(_OperatingConfigModel):
    label: str = Field(min_length=1)
    multiplier: Decimal


class OperatingUnitsConfiguration(_OperatingConfigModel):
    schema_version: Literal[1]
    scale_aliases: tuple[UnitScaleConfiguration, ...]


class OperatingExtractionConfiguration(_OperatingConfigModel):
    schema_version: Literal[1, 2, 3]
    number_pattern: str = Field(min_length=1)
    year_pattern: str = Field(min_length=1)
    forward_pattern: str = Field(min_length=1)
    first_party_pattern: str = Field(min_length=1)
    third_party_terms: tuple[str, ...]
    revenue_driver_ids: tuple[str, ...]
    operating_audit_patterns: tuple[KeywordPatternConfiguration, ...]
    currency_unit_pattern: str = Field(min_length=1)
    count_unit_exclusions: tuple[str, ...]
    metric_aliases: dict[str, tuple[str, ...]]
    implied_revenue_driver_ids: tuple[str, ...]
    implied_volume_driver_ids: tuple[str, ...]
    implied_subscriber_driver_ids: tuple[str, ...]
    history_revenue_metric_ids: tuple[str, ...]
    history_volume_metric_ids: tuple[str, ...]
    history_count_metric_ids: tuple[str, ...]
    gross_margin_driver_ids: tuple[str, ...] = ()
    gross_profit_driver_ids: tuple[str, ...] = ()
    cost_of_revenue_driver_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatingVocabularyView:
    """Immutable, runtime-shaped vocabulary tables."""

    schema_version: int
    fingerprint: str
    source_path: Path
    global_terms: tuple[tuple[str, str], ...]
    industry_terms: Mapping[str, tuple[tuple[str, str], ...]]
    industry_aliases: Mapping[str, str]
    metric_aliases: Mapping[str, str]
    revenue_driver_priority: Mapping[str, int]

    @property
    def cache_version(self) -> str:
        return f"operating-vocabulary-v{self.schema_version}:{self.fingerprint}"


@dataclass(frozen=True)
class CompiledPattern:
    """A source-preserving regex rule compiled once for immutable consumers."""

    pattern: str
    regex: re.Pattern[str]
    weight: int | None = None
    keyword: str | None = None


@dataclass(frozen=True)
class OperatingExtractionView:
    """Immutable, compiled operating extraction tables."""

    schema_version: int
    fingerprint: str
    source_path: Path
    number_pattern: CompiledPattern
    year_pattern: CompiledPattern
    forward_pattern: CompiledPattern
    first_party_pattern: CompiledPattern
    third_party_terms: tuple[str, ...]
    revenue_driver_ids: frozenset[str]
    operating_audit_patterns: tuple[CompiledPattern, ...]
    currency_unit_pattern: CompiledPattern
    count_unit_exclusions: frozenset[str]
    metric_aliases: Mapping[str, frozenset[str]]
    implied_revenue_driver_ids: frozenset[str]
    implied_volume_driver_ids: frozenset[str]
    implied_subscriber_driver_ids: frozenset[str]
    history_revenue_metric_ids: frozenset[str]
    history_volume_metric_ids: frozenset[str]
    history_count_metric_ids: frozenset[str]
    gross_margin_driver_ids: frozenset[str] = frozenset()
    gross_profit_driver_ids: frozenset[str] = frozenset()
    cost_of_revenue_driver_ids: frozenset[str] = frozenset()

    @property
    def cache_version(self) -> str:
        return f"operating-extraction-v{self.schema_version}:{self.fingerprint}"


@dataclass(frozen=True)
class OperatingUnitsView:
    """Immutable, source-ordered operating unit scale aliases."""

    schema_version: int
    fingerprint: str
    source_path: Path
    scale_aliases: tuple[UnitScaleConfiguration, ...]

    @property
    def cache_version(self) -> str:
        return f"operating-units-v{self.schema_version}:{self.fingerprint}"


def _compile_pattern(
    pattern: str,
    *,
    weight=None,
    keyword=None,
    flags: int = re.IGNORECASE,
) -> CompiledPattern:
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid operating configuration regex {pattern!r}") from exc
    return CompiledPattern(
        pattern=pattern,
        regex=regex,
        weight=weight,
        keyword=keyword,
    )


class OperatingVocabularyLoader:
    """Load the packaged operating KPI vocabulary."""

    PATH = Path("configs/operating/vocabulary.json")
    DESCRIPTION = "Operating KPI vocabulary configuration"

    @classmethod
    def load(cls, path: str | Path | None = None) -> OperatingVocabularyConfiguration:
        return cls._read(path)[0]

    @classmethod
    def compiled(cls, path: str | Path | None = None) -> OperatingVocabularyView:
        if path is None:
            return cls._default_compiled()
        return cls._compile(*cls._read(path))

    @classmethod
    def default_path(cls) -> Path:
        return cls._read(None)[1]

    @classmethod
    def _read(
        cls, path: str | Path | None
    ) -> tuple[OperatingVocabularyConfiguration, Path, str, str]:
        payload, source, content, fingerprint = read_config_payload(
            cls.PATH,
            path,
            description=cls.DESCRIPTION,
        )
        try:
            model = OperatingVocabularyConfiguration.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid operating vocabulary {source}: {exc}") from exc
        return model, source, content, fingerprint

    @classmethod
    def _compile(
        cls,
        model: OperatingVocabularyConfiguration,
        source: Path,
        _content: str,
        fingerprint: str,
    ) -> OperatingVocabularyView:
        return OperatingVocabularyView(
            schema_version=model.schema_version,
            fingerprint=fingerprint,
            source_path=source,
            global_terms=tuple((item.raw, item.metric) for item in model.global_terms),
            industry_terms=MappingProxyType(
                {
                    key: tuple((item.raw, item.metric) for item in terms)
                    for key, terms in model.industry_terms.items()
                }
            ),
            industry_aliases=MappingProxyType(dict(model.industry_aliases)),
            metric_aliases=MappingProxyType(dict(model.metric_aliases)),
            revenue_driver_priority=MappingProxyType(
                dict(model.revenue_driver_priority)
            ),
        )

    @classmethod
    def _default_compiled(cls) -> OperatingVocabularyView:
        return cls._compile(*cls._read(None))


class OperatingUnitsLoader:
    """Load packaged operating-unit scale aliases."""

    PATH = Path("configs/operating/units.json")
    DESCRIPTION = "Operating unit configuration"

    @classmethod
    def load(cls, path: str | Path | None = None) -> OperatingUnitsConfiguration:
        return cls._read(path)[0]

    @classmethod
    def compiled(cls, path: str | Path | None = None) -> OperatingUnitsView:
        if path is None:
            return cls._default_compiled()
        return cls._compile(*cls._read(path))

    @classmethod
    def default_path(cls) -> Path:
        return cls._read(None)[1]

    @classmethod
    def _read(
        cls, path: str | Path | None
    ) -> tuple[OperatingUnitsConfiguration, Path, str, str]:
        payload, source, content, fingerprint = read_config_payload(
            cls.PATH,
            path,
            description=cls.DESCRIPTION,
        )
        try:
            model = OperatingUnitsConfiguration.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid operating units {source}: {exc}") from exc
        return model, source, content, fingerprint

    @classmethod
    def _compile(
        cls,
        model: OperatingUnitsConfiguration,
        source: Path,
        _content: str,
        fingerprint: str,
    ) -> OperatingUnitsView:
        return OperatingUnitsView(
            schema_version=model.schema_version,
            fingerprint=fingerprint,
            source_path=source,
            scale_aliases=model.scale_aliases,
        )

    @classmethod
    def _default_compiled(cls) -> OperatingUnitsView:
        return cls._compile(*cls._read(None))


class OperatingExtractionLoader:
    """Load and compile operating extraction term and regex tables."""

    PATH = Path("configs/operating/extraction.json")
    DESCRIPTION = "Operating extraction configuration"

    @classmethod
    def load(cls, path: str | Path | None = None) -> OperatingExtractionConfiguration:
        return cls._read(path)[0]

    @classmethod
    def compiled(cls, path: str | Path | None = None) -> OperatingExtractionView:
        if path is None:
            return cls._default_compiled()
        return cls._compile(*cls._read(path))

    @classmethod
    def default_path(cls) -> Path:
        return cls._read(None)[1]

    @classmethod
    def _read(
        cls, path: str | Path | None
    ) -> tuple[OperatingExtractionConfiguration, Path, str, str]:
        payload, source, content, fingerprint = read_config_payload(
            cls.PATH,
            path,
            description=cls.DESCRIPTION,
        )
        try:
            model = OperatingExtractionConfiguration.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid operating extraction {source}: {exc}") from exc
        return model, source, content, fingerprint

    @classmethod
    def _compile(
        cls,
        model: OperatingExtractionConfiguration,
        source: Path,
        _content: str,
        fingerprint: str,
    ) -> OperatingExtractionView:
        try:
            number_pattern = _compile_pattern(model.number_pattern, flags=0)
            year_pattern = _compile_pattern(model.year_pattern)
            forward_pattern = _compile_pattern(model.forward_pattern)
            first_party_pattern = _compile_pattern(model.first_party_pattern)
            currency_unit_pattern = _compile_pattern(
                model.currency_unit_pattern, flags=0
            )
            audit_patterns = tuple(
                _compile_pattern(
                    item.pattern,
                    keyword=getattr(item, "keyword", None),
                    weight=getattr(item, "weight", None),
                )
                for item in model.operating_audit_patterns
            )
        except ValueError:
            raise
        return OperatingExtractionView(
            schema_version=model.schema_version,
            fingerprint=fingerprint,
            source_path=source,
            number_pattern=number_pattern,
            year_pattern=year_pattern,
            forward_pattern=forward_pattern,
            first_party_pattern=first_party_pattern,
            third_party_terms=model.third_party_terms,
            revenue_driver_ids=frozenset(model.revenue_driver_ids),
            operating_audit_patterns=audit_patterns,
            currency_unit_pattern=currency_unit_pattern,
            count_unit_exclusions=frozenset(model.count_unit_exclusions),
            metric_aliases=MappingProxyType(
                {key: frozenset(values) for key, values in model.metric_aliases.items()}
            ),
            implied_revenue_driver_ids=frozenset(model.implied_revenue_driver_ids),
            implied_volume_driver_ids=frozenset(model.implied_volume_driver_ids),
            implied_subscriber_driver_ids=frozenset(
                model.implied_subscriber_driver_ids
            ),
            history_revenue_metric_ids=frozenset(model.history_revenue_metric_ids),
            history_volume_metric_ids=frozenset(model.history_volume_metric_ids),
            history_count_metric_ids=frozenset(model.history_count_metric_ids),
            gross_margin_driver_ids=frozenset(model.gross_margin_driver_ids),
            gross_profit_driver_ids=frozenset(model.gross_profit_driver_ids),
            cost_of_revenue_driver_ids=frozenset(model.cost_of_revenue_driver_ids),
        )

    @classmethod
    def _default_compiled(cls) -> OperatingExtractionView:
        return cls._compile(*cls._read(None))


OPERATING_VOCABULARY = OperatingVocabularyLoader.compiled()
OPERATING_UNITS = OperatingUnitsLoader.compiled()
OPERATING_EXTRACTION = OperatingExtractionLoader.compiled()
DEFAULT_OPERATING_VOCABULARY_PATH = OperatingVocabularyLoader.PATH
DEFAULT_OPERATING_UNITS_PATH = OperatingUnitsLoader.PATH
DEFAULT_OPERATING_EXTRACTION_PATH = OperatingExtractionLoader.PATH


__all__ = [
    "CompiledPattern",
    "DEFAULT_OPERATING_EXTRACTION_PATH",
    "DEFAULT_OPERATING_UNITS_PATH",
    "DEFAULT_OPERATING_VOCABULARY_PATH",
    "KpiTermConfiguration",
    "KeywordPatternConfiguration",
    "OperatingExtractionConfiguration",
    "OperatingExtractionLoader",
    "OperatingExtractionView",
    "OperatingUnitsConfiguration",
    "OperatingUnitsLoader",
    "OperatingUnitsView",
    "OperatingVocabularyConfiguration",
    "OperatingVocabularyLoader",
    "OperatingVocabularyView",
    "OPERATING_EXTRACTION",
    "OPERATING_UNITS",
    "OPERATING_VOCABULARY",
    "UnitScaleConfiguration",
]
