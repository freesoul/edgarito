"""Compatibility facade for operating-evidence discovery.

The implementation and discovery contracts live in
:mod:`edgarito.services.operating._discovery` while this module preserves the
long-standing public import path and aliases.  In particular, the former
module imported the schema contracts below into its namespace; keep those
imports available for users that used this module as a public facade.
"""

import datetime
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDocumentAudit,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingEvidenceAuditRecord,
    OperatingEvidenceRejection,
    OperatingInvestmentProgram,
    OperatingSegment,
)
from edgarito.schemas.operating_history import (
    OperatingEvidenceGap,
    OperatingHistoryAudit,
)
from edgarito.schemas.vocabulary import KpiVocabularyAudit
from edgarito.services.guidance.documents import (
    GuidanceDocumentSelector,
    clean_document_text,
    extract_operating_context,
    guidance_keyword_hits,
    is_exhibit_document,
    is_periodic_filing,
)
from edgarito.services.openai import OpenAIAuthenticationError
from edgarito.services.operating._discovery import service as _implementation
from edgarito.services.operating._discovery.service import (
    OperatingDriverDiscoveryService,
    OperatingEvidenceDiscovery,
    OperatingEvidenceDiscoveryService,
    OperatingForecastDiscovery,
    OperatingForecastDiscoveryResult,
    OperatingForecastDiscoveryService,
    OperatingIrFallback,
)
from edgarito.services.operating.extraction import (
    OperatingEvidenceExtractor,
    operating_keyword_hits,
)
from edgarito.services.operating.history import OperatingHistoryAssembler
from edgarito.services.operating.vocabulary import (
    KpiVocabularyProvider,
    normalize_industry_namespace,
)
from edgarito.services.providers.edgar import EdgarClient

# These are deliberate compatibility exports, not unused implementation
# imports.  They were all importable from this module before the split.
# ruff: noqa: F401

__all__ = [
    "OperatingDriverDiscoveryService",
    "OperatingEvidenceDiscovery",
    "OperatingEvidenceDiscoveryService",
    "OperatingForecastDiscovery",
    "OperatingForecastDiscoveryResult",
    "OperatingForecastDiscoveryService",
    "OperatingIrFallback",
]


def __getattr__(name: str):
    """Resolve legacy private helper imports from the moved implementation."""

    return getattr(_implementation, name)


def __dir__():
    return sorted({*globals(), *vars(_implementation)})
