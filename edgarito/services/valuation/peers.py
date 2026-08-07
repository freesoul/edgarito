import re
from decimal import Decimal

from edgarito.services.valuation.models import (
    BusinessArchetype,
    CompanyTradingMultiples,
    PeerCandidateAssessment,
    PeerDiscoveryResult,
    PeerSelectionParameters,
    PeerUniverse,
    ValuationProfile,
)


class PeerUniverseSelector:
    """Rank an explicit candidate universe by economic comparability."""

    _SPECIALIZED_ARCHETYPES = {
        BusinessArchetype.FINANCIAL_INTERMEDIARY,
        BusinessArchetype.ASSET_MANAGER,
        BusinessArchetype.REIT_PROPERTY,
        BusinessArchetype.RESOURCE_PRODUCER,
        BusinessArchetype.PROJECT_PIPELINE,
        BusinessArchetype.HOLDING_COMPANY,
        BusinessArchetype.CONGLOMERATE,
    }

    def select(
        self,
        target: ValuationProfile,
        candidates: list[ValuationProfile],
        parameters: PeerSelectionParameters | None = None,
        target_multiples: CompanyTradingMultiples | None = None,
        candidate_multiples: dict[str, CompanyTradingMultiples] | None = None,
        discovery: PeerDiscoveryResult | None = None,
    ) -> PeerUniverse:
        parameters = parameters or PeerSelectionParameters()
        if not target.ticker:
            raise ValueError("Peer selection requires a target ticker")

        candidate_multiples = candidate_multiples or {}
        assessments = [
            self._assess(
                target,
                candidate,
                parameters,
                target_multiples,
                candidate_multiples.get(candidate.ticker or candidate.company_id),
            )
            for candidate in candidates
        ]
        assessments.sort(key=lambda item: (-item.score, item.ticker))
        seen_entities: dict[str, str] = {}
        deduplicated = []
        for item in assessments:
            entity_key = self._entity_key(item.company_name)
            representative = seen_entities.get(entity_key)
            if entity_key and representative is not None:
                item = item.model_copy(
                    update={
                        "exclusions": [
                            *item.exclusions,
                            f"Duplicate listing of the same issuer as {representative}",
                        ]
                    }
                )
            elif entity_key:
                seen_entities[entity_key] = item.ticker
            deduplicated.append(item)
        assessments = deduplicated
        eligible = [
            item
            for item in assessments
            if not item.exclusions and item.score >= parameters.minimum_score
        ]
        selected_tickers = tuple(
            item.ticker for item in eligible[: parameters.max_peers]
        )
        selected = set(selected_tickers)
        assessments = [
            item.model_copy(update={"selected": item.ticker in selected})
            for item in assessments
        ]

        warnings = []
        if len(selected_tickers) < parameters.preferred_minimum:
            warnings.append(
                f"Only {len(selected_tickers)} peers met the selection criteria; "
                f"{parameters.preferred_minimum} are preferred"
            )
        if not candidates:
            warnings.append("No provider-supported candidate universe was available")
        if discovery is not None:
            warnings.extend(discovery.warnings)
        confidence = (
            "high"
            if len(selected_tickers) >= parameters.preferred_minimum
            else "medium"
            if len(selected_tickers) >= max(2, parameters.preferred_minimum // 2)
            else "low"
        )
        return PeerUniverse(
            target_ticker=target.ticker,
            target_company_id=target.company_id,
            parameters=parameters,
            candidates=assessments,
            selected_tickers=selected_tickers,
            discovery_source=(
                discovery.provider if discovery is not None else "manual override"
            ),
            discovery_methodology=(
                discovery.methodology
                if discovery is not None
                else "Explicit --peer candidate symbols override automatic discovery"
            ),
            discovery_confidence=confidence,
            warnings=warnings,
        )

    def _assess(
        self,
        target: ValuationProfile,
        candidate: ValuationProfile,
        parameters: PeerSelectionParameters,
        target_multiples: CompanyTradingMultiples | None = None,
        candidate_multiples: CompanyTradingMultiples | None = None,
    ) -> PeerCandidateAssessment:
        ticker = candidate.ticker or candidate.company_id
        exclusions = []
        reasons = []
        if self._same_company(target, candidate):
            exclusions.append("Candidate is the target company")
        if (
            parameters.require_same_sector
            and target.sector is not None
            and candidate.sector is not None
            and candidate.sector != target.sector
        ):
            exclusions.append("Sector differs from the target")
        if (
            target.business_archetype in self._SPECIALIZED_ARCHETYPES
            and candidate.business_archetype != target.business_archetype
        ):
            exclusions.append("Specialized business economics differ from the target")

        score = 0
        if target.sector is not None and candidate.sector == target.sector:
            score += 20
            reasons.append("Same sector")

        industry_score = self._industry_score(target.industry, candidate.industry)
        score += industry_score
        if industry_score == 30:
            reasons.append("Same normalized industry")
        elif industry_score:
            reasons.append("Industry descriptions substantially overlap")

        if candidate.business_archetype == target.business_archetype:
            score += 20
            reasons.append("Same business archetype")
        if candidate.lifecycle == target.lifecycle:
            score += 10
            reasons.append("Same lifecycle")
        if candidate.cyclicality == target.cyclicality:
            score += 8
            reasons.append("Same cyclicality")
        shared_traits = target.economic_traits & candidate.economic_traits
        if shared_traits:
            score += min(5, len(shared_traits) * 2)
            reasons.append(
                "Shared explicit economic traits: "
                + ", ".join(sorted(item.value for item in shared_traits))
            )
        if target.country and candidate.country == target.country:
            score += 5
            reasons.append("Same country")
        if target.exchange and candidate.exchange == target.exchange:
            score += 2
            reasons.append("Same exchange")

        comparable_currency = (
            not target.reporting_currency
            or not candidate.reporting_currency
            or target.reporting_currency == candidate.reporting_currency
        )
        size_score = (
            self._size_score(target.latest_revenue, candidate.latest_revenue)
            if comparable_currency
            else 0
        )
        score += size_score
        if size_score:
            reasons.append("Revenue scale is comparable")
        if (
            target.reporting_currency
            and candidate.reporting_currency
            and target.reporting_currency != candidate.reporting_currency
        ):
            reasons.append("Different reporting currency; multiples need FX alignment")

        economic_score = self._economic_similarity(
            target,
            candidate,
            target_multiples,
            candidate_multiples,
        )
        if economic_score is not None:
            score = round(score * 0.75 + economic_score * 0.25)
            reasons.append(
                "Observable margin, cash-conversion, leverage, and capital-"
                f"intensity similarity: {economic_score}/100"
            )

        return PeerCandidateAssessment(
            ticker=ticker,
            company_id=candidate.company_id,
            company_name=candidate.company_name,
            score=max(0, min(100, score)),
            reasons=reasons,
            exclusions=exclusions,
        )

    @staticmethod
    def _economic_similarity(
        target_profile, candidate_profile, target, candidate
    ) -> int | None:
        if target is None or candidate is None:
            return None
        left = target.fundamentals
        right = candidate.fundamentals

        def ratio(numerator, denominator):
            if numerator is None or denominator is None or denominator <= 0:
                return None
            return numerator / denominator

        pairs = [
            (ratio(left.ebitda, left.revenue), ratio(right.ebitda, right.revenue)),
            (
                ratio(left.free_cash_flow, left.revenue),
                ratio(right.free_cash_flow, right.revenue),
            ),
            (
                ratio(left.gross_debt, left.ebitda),
                ratio(right.gross_debt, right.ebitda),
            ),
            (
                ratio(left.capital_expenditures, left.revenue),
                ratio(right.capital_expenditures, right.revenue),
            ),
        ]
        if (
            target_profile.revenue_growth_rates
            and candidate_profile.revenue_growth_rates
        ):
            pairs.append(
                (
                    target_profile.revenue_growth_rates[-1] / Decimal(100),
                    candidate_profile.revenue_growth_rates[-1] / Decimal(100),
                )
            )
        similarities = []
        for target_value, candidate_value in pairs:
            if target_value is None or candidate_value is None:
                continue
            scale = max(abs(target_value), abs(candidate_value), Decimal("0.01"))
            similarities.append(
                max(
                    Decimal(0),
                    Decimal(1) - abs(target_value - candidate_value) / scale,
                )
            )
        if not similarities:
            return None
        average = sum(similarities, Decimal(0)) / Decimal(len(similarities))
        return round(float(average * Decimal(100)))

    @staticmethod
    def _same_company(target: ValuationProfile, candidate: ValuationProfile) -> bool:
        if target.company_id.isdigit() and candidate.company_id.isdigit():
            return int(target.company_id) == int(candidate.company_id)
        same_ticker = bool(
            target.ticker
            and candidate.ticker
            and target.ticker.casefold() == candidate.ticker.casefold()
        )
        same_name = PeerUniverseSelector._entity_key(
            target.company_name
        ) == PeerUniverseSelector._entity_key(candidate.company_name)
        return same_ticker or same_name

    @staticmethod
    def _entity_key(name: str) -> str:
        noise = {
            "a",
            "adr",
            "ag",
            "b",
            "cdr",
            "cedear",
            "class",
            "common",
            "corp",
            "corporation",
            "depositary",
            "inc",
            "limited",
            "ltd",
            "nv",
            "ord",
            "ordinary",
            "plc",
            "receipt",
            "sa",
            "se",
            "spa",
            "stock",
        }
        tokens = re.findall(r"[a-z0-9]+", name.casefold())
        return "".join(token for token in tokens if token not in noise)

    @classmethod
    def _industry_score(cls, target: str | None, candidate: str | None) -> int:
        target_tokens = cls._tokens(target)
        candidate_tokens = cls._tokens(candidate)
        if not target_tokens or not candidate_tokens:
            return 0
        if target_tokens == candidate_tokens:
            return 30
        overlap = len(target_tokens & candidate_tokens) / len(
            target_tokens | candidate_tokens
        )
        return min(20, int(Decimal(str(overlap)) * 20))

    @staticmethod
    def _tokens(value: str | None) -> set[str]:
        if not value:
            return set()
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.casefold())
            if token not in {"and", "the", "services", "other"}
        }

    @staticmethod
    def _size_score(target: Decimal | None, candidate: Decimal | None) -> int:
        if target is None or candidate is None or target <= 0 or candidate <= 0:
            return 0
        ratio = max(target, candidate) / min(target, candidate)
        if ratio <= Decimal("1.5"):
            return 15
        if ratio <= Decimal(3):
            return 10
        if ratio <= Decimal(10):
            return 5
        return 0


__all__ = ["PeerUniverseSelector"]
