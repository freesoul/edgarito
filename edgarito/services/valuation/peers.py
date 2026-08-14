import re
from decimal import Decimal
from statistics import median

from edgarito.schemas.valuation.selection import (
    BusinessArchetype,
    PeerEvidenceGroup,
    ValuationProfile,
)
from edgarito.services.valuation.issuer_identity import (
    issuer_identity_keys,
    normalize_issuer_name,
)
from edgarito.services.valuation.models import (
    CompanyTradingMultiples,
    PeerCandidateAssessment,
    PeerDiscoveryResult,
    PeerSelectionParameters,
    PeerUniverse,
)


class PeerUniverseSelector:
    """Rank an explicit candidate universe by economic comparability."""

    _MIN_ECONOMIC_SIMILARITY_DIMENSIONS = 3
    _SPECIALIZED_ARCHETYPES = {
        BusinessArchetype.FINANCIAL_INTERMEDIARY,
        BusinessArchetype.ASSET_MANAGER,
        BusinessArchetype.REIT_PROPERTY,
        BusinessArchetype.RESOURCE_PRODUCER,
        BusinessArchetype.PROJECT_PIPELINE,
        BusinessArchetype.HOLDING_COMPANY,
        BusinessArchetype.CONGLOMERATE,
    }
    _GENERAL_EVIDENCE_GROUP = PeerEvidenceGroup.GENERAL_OPERATING.value
    _GROWTH_LIFECYCLES = {
        "pre_revenue",
        "unprofitable_growth",
        "growth",
    }
    _ENERGY_STORAGE = re.compile(
        r"\b(energy storage|batter(?:y|ies)|battery storage|lithium[- ]ion|"
        r"fuel cell)\b"
    )
    _ELECTRIC_VEHICLE = re.compile(
        r"\b(electric vehicles?|electric cars?|evs?|ev manufacturers?|"
        r"electric mobility)\b"
    )
    _AUTO_OEM = re.compile(
        r"\b(auto(?:mobile)? manufacturers?|automotive manufacturers?|"
        r"motor vehicles?|vehicle manufacturers?|car manufacturers?|"
        r"automobiles?|"
        r"auto makers?|automotive)\b"
    )
    _TECHNOLOGY_PLATFORM = re.compile(
        r"\b(software|cloud computing|cloud services?|technology platform|"
        r"technology services?|information technology services?|"
        r"internet services?|online platforms?|application software|"
        r"infrastructure software)\b"
    )
    _STRUCTURAL_PRODUCT_TRAITS = {
        "regulated_capital",
        "lease_intensive",
        "backlog_driven",
        "book_value_unreliable",
        "financing_subsidiary",
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
        target_evidence_group = self._evidence_group(
            target, override=parameters.evidence_group
        )
        candidates_by_ticker = {
            candidate.ticker or candidate.company_id: candidate
            for candidate in candidates
        }
        assessments = [
            self._assess(
                target,
                candidate,
                parameters,
                target_multiples,
                candidate_multiples.get(candidate.ticker or candidate.company_id),
                target_evidence_group=target_evidence_group,
            )
            for candidate in candidates
        ]
        assessments.sort(key=lambda item: (-item.score, item.ticker))
        seen_entities: dict[str, str] = {}
        deduplicated = []
        for item in assessments:
            candidate = candidates_by_ticker.get(item.ticker)
            entity_keys = (
                self._issuer_keys(candidate) if candidate is not None else frozenset()
            )
            representative = next(
                (
                    seen_entities[entity_key]
                    for entity_key in entity_keys
                    if entity_key in seen_entities
                ),
                None,
            )
            if representative is not None:
                item = item.model_copy(
                    update={
                        "exclusions": [
                            *item.exclusions,
                            f"Duplicate listing of the same issuer as {representative}",
                        ]
                    }
                )
            else:
                for entity_key in entity_keys:
                    seen_entities[entity_key] = item.ticker
            deduplicated.append(item)
        assessments = deduplicated

        # A candidate can satisfy the score and economic gate while still being
        # outside the target's evidence group. Keep that candidate visible for
        # audit/context, but never let it enter the primary peer pool.
        contextual_count = 0
        for index, item in enumerate(assessments):
            if (
                not item.exclusions
                and item.score >= parameters.minimum_score
                and item.evidence_group != target_evidence_group
            ):
                contextual_count += 1
                assessments[index] = item.model_copy(
                    update={
                        "contextual": True,
                        "exclusions": [
                            *item.exclusions,
                            "Evidence group "
                            f"{item.evidence_group!r} differs from target primary "
                            f"group {target_evidence_group!r}; retained as "
                            "contextual/non-selected evidence",
                        ],
                    }
                )

        eligible = self._eligible(assessments, parameters, target_evidence_group)
        if len(eligible) < parameters.preferred_minimum:
            relaxed = []
            for item in assessments:
                candidate = candidates_by_ticker.get(item.ticker)
                market_cap_only = self._market_cap_only(item)
                same_industry = (
                    candidate is not None
                    and self._industry_score(target.industry, candidate.industry) == 30
                )
                same_sector = (
                    candidate is not None
                    and target.sector is not None
                    and candidate.sector == target.sector
                )
                if (
                    market_cap_only
                    and item.economic_gate_passed
                    and item.evidence_group == target_evidence_group
                ):
                    fallback_reason = (
                        "Exact-industry fallback retained despite market-cap "
                        "difference because strict selection produced too few peers"
                        if same_industry and same_sector
                        else "Evidence-gated fallback retained despite market-cap "
                        "difference because strict selection produced too few peers"
                    )
                    item = item.model_copy(
                        update={
                            "reasons": [*item.reasons, fallback_reason],
                            "exclusions": [],
                        }
                    )
                relaxed.append(item)
            assessments = relaxed
            eligible = self._eligible(assessments, parameters, target_evidence_group)
        selected_assessments = eligible[: parameters.max_peers]
        selected_tickers = tuple(item.ticker for item in selected_assessments)
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
        if contextual_count:
            warnings.append(
                f"{contextual_count} otherwise eligible candidate(s) fell outside "
                f"the target primary evidence group {target_evidence_group!r}; "
                "they were retained as contextual evidence and excluded from peer "
                "aggregation"
            )
        elif candidates and not any(
            item.evidence_group == target_evidence_group for item in assessments
        ):
            warnings.append(
                f"No candidates matched the target primary evidence group "
                f"{target_evidence_group!r}"
            )
        if discovery is not None:
            warnings.extend(discovery.warnings)
        economic_scores = [
            item.economic_similarity
            for item in selected_assessments
            if item.economic_similarity is not None
        ]
        if economic_scores and median(economic_scores) < 35:
            warnings.append(
                "Selected peers have weak median observable economic similarity "
                f"({median(economic_scores):.0f}/100)"
            )
        confidence = self._selection_confidence(
            selected_assessments,
            parameters.preferred_minimum,
            discovery.confidence if discovery is not None else "high",
        )
        if confidence == "low" and selected_tickers:
            warnings.append(
                "Peer evidence confidence is low; relative valuation should be "
                "skipped rather than presented with false precision"
            )
        return PeerUniverse(
            target_ticker=target.ticker,
            target_company_id=target.company_id,
            parameters=parameters,
            evidence_group=target_evidence_group,
            candidates=assessments,
            selected_tickers=selected_tickers,
            discovery_source=(
                discovery.provider if discovery is not None else "manual override"
            ),
            discovery_methodology=(
                discovery.methodology
                if discovery is not None
                else "Explicit --peer candidate symbols are discovery inputs only; "
                "industry/product-economics or observable-similarity gating and "
                "primary evidence-group selection remain required"
            ),
            discovery_confidence=confidence,
            warnings=warnings,
        )

    @staticmethod
    def _market_cap_only(item: PeerCandidateAssessment) -> bool:
        return len(item.exclusions) == 1 and item.exclusions[0].startswith(
            "Market capitalization is outside"
        )

    @staticmethod
    def _eligible(
        assessments: list[PeerCandidateAssessment],
        parameters: PeerSelectionParameters,
        target_evidence_group: str,
    ) -> list[PeerCandidateAssessment]:
        return [
            item
            for item in assessments
            if (
                not item.exclusions
                and item.economic_gate_passed
                and item.evidence_group == target_evidence_group
                and item.score >= parameters.minimum_score
            )
        ]

    def _assess(
        self,
        target: ValuationProfile,
        candidate: ValuationProfile,
        parameters: PeerSelectionParameters,
        target_multiples: CompanyTradingMultiples | None = None,
        candidate_multiples: CompanyTradingMultiples | None = None,
        *,
        target_evidence_group: str | None = None,
    ) -> PeerCandidateAssessment:
        ticker = candidate.ticker or candidate.company_id
        exclusions = []
        reasons = []
        target_evidence_group = target_evidence_group or self._evidence_group(target)
        candidate_evidence_group = self._evidence_group(candidate)
        if self._same_company(target, candidate):
            exclusions.append(
                "Candidate is the target company/issuer (cross-listing or ADR excluded)"
            )
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

        industry_score = self._industry_score(target.industry, candidate.industry)
        industry_match = industry_score == 30
        product_economics_match = self._product_economics_match(
            target,
            candidate,
            target_evidence_group,
            candidate_evidence_group,
            industry_match=industry_match,
        )
        economic_score = self._economic_similarity(
            target,
            candidate,
            target_multiples,
            candidate_multiples,
        )
        economic_match = (
            economic_score is not None
            and economic_score >= parameters.minimum_economic_similarity
        )
        economic_gate_passed = (
            industry_match or product_economics_match or economic_match
        )
        if not economic_gate_passed:
            if economic_score is None:
                detail = "observable economic similarity is unavailable"
            else:
                detail = (
                    "observable economic similarity is "
                    f"{economic_score}/100, below the configured "
                    f"{parameters.minimum_economic_similarity}/100 threshold"
                )
            exclusions.append("No industry/product-economics match and " + detail)
        if industry_match:
            reasons.append("Same normalized industry")
        elif product_economics_match:
            reasons.append("Product-economics match")
        if economic_match:
            reasons.append(
                "Observable economic similarity meets the configured "
                f"{parameters.minimum_economic_similarity}/100 gate"
            )

        market_cap_score, market_cap_reason, market_cap_exclusion = (
            self._market_cap_comparability(
                target_multiples,
                candidate_multiples,
                parameters,
            )
        )
        if market_cap_exclusion:
            exclusions.append(market_cap_exclusion)

        score = market_cap_score
        if market_cap_reason:
            reasons.append(market_cap_reason)
        if target.sector is not None and candidate.sector == target.sector:
            score += 20
            reasons.append("Same sector")

        score += industry_score
        if industry_score and not industry_match:
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

        if economic_score is not None:
            score = round(score * 0.75 + economic_score * 0.25)
            reasons.append(
                "Observable growth, operating margin, ROIC, cash-conversion, "
                f"leverage, and capital-intensity similarity: {economic_score}/100"
            )

        return PeerCandidateAssessment(
            ticker=ticker,
            company_id=candidate.company_id,
            company_name=candidate.company_name,
            score=max(0, min(100, score)),
            evidence_group=candidate_evidence_group,
            economic_gate_passed=economic_gate_passed,
            economic_similarity=economic_score,
            reasons=reasons,
            exclusions=exclusions,
        )

    @classmethod
    def _evidence_group(
        cls,
        profile: ValuationProfile,
        *,
        override: str | None = None,
    ) -> str:
        explicit = override or profile.evidence_group
        if explicit:
            return (
                explicit.value
                if isinstance(explicit, PeerEvidenceGroup)
                else str(explicit).strip().casefold()
            )

        industry = (profile.industry or "").casefold()
        lifecycle = getattr(profile.lifecycle, "value", profile.lifecycle)
        archetype = profile.business_archetype
        traits = {getattr(trait, "value", trait) for trait in profile.economic_traits}

        if cls._ENERGY_STORAGE.search(industry):
            return PeerEvidenceGroup.ENERGY_STORAGE.value
        if cls._ELECTRIC_VEHICLE.search(industry):
            return (
                PeerEvidenceGroup.EV_GROWTH.value
                if lifecycle in cls._GROWTH_LIFECYCLES
                else PeerEvidenceGroup.AUTO_OEM.value
            )
        if cls._AUTO_OEM.search(industry):
            if lifecycle in cls._GROWTH_LIFECYCLES:
                return PeerEvidenceGroup.EV_GROWTH.value
            return PeerEvidenceGroup.AUTO_OEM.value
        if (
            archetype == BusinessArchetype.GENERAL_OPERATING
            and cls._TECHNOLOGY_PLATFORM.search(industry)
        ):
            return PeerEvidenceGroup.TECHNOLOGY_PLATFORM.value

        if archetype in cls._SPECIALIZED_ARCHETYPES:
            return archetype.value
        if "regulated_capital" in traits:
            return "regulated_operating"
        if "lease_intensive" in traits:
            return "lease_intensive_operating"
        if "backlog_driven" in traits:
            return "backlog_driven_operating"
        return cls._GENERAL_EVIDENCE_GROUP

    @classmethod
    def _product_economics_match(
        cls,
        target: ValuationProfile,
        candidate: ValuationProfile,
        target_group: str,
        candidate_group: str,
        *,
        industry_match: bool,
    ) -> bool:
        if industry_match:
            return True
        if (
            target_group == candidate_group
            and target_group != cls._GENERAL_EVIDENCE_GROUP
        ):
            return True
        if (
            target.business_archetype == candidate.business_archetype
            and target.business_archetype in cls._SPECIALIZED_ARCHETYPES
        ):
            return True
        shared_traits = {
            getattr(trait, "value", trait)
            for trait in target.economic_traits & candidate.economic_traits
        }
        return bool(
            shared_traits & cls._STRUCTURAL_PRODUCT_TRAITS
            and target.lifecycle == candidate.lifecycle
            and target.business_archetype == candidate.business_archetype
        )

    @staticmethod
    def _selection_confidence(selected, preferred_minimum, provider_confidence):
        levels = {"low": 0, "medium": 1, "high": 2}
        count = len(selected)
        count_level = (
            "high"
            if count >= preferred_minimum
            else "medium"
            if count >= max(2, preferred_minimum // 2)
            else "low"
        )
        score_median = median(item.score for item in selected) if selected else 0
        score_level = (
            "high" if score_median >= 70 else "medium" if score_median >= 55 else "low"
        )
        economic = [
            item.economic_similarity
            for item in selected
            if item.economic_similarity is not None
        ]
        economic_median = median(economic) if economic else None
        economic_level = (
            "high"
            if economic_median is None or economic_median >= 55
            else "medium"
            if economic_median >= 35
            else "low"
        )
        selected_level = min(
            levels[count_level],
            levels[score_level],
            levels[economic_level],
            levels.get(provider_confidence, 0),
        )
        return ("low", "medium", "high")[selected_level]

    @classmethod
    def _economic_similarity(
        cls, target_profile, candidate_profile, target, candidate
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
            (
                ratio(left.operating_income, left.revenue),
                ratio(right.operating_income, right.revenue),
            ),
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
        target_growth = left.revenue_growth
        candidate_growth = right.revenue_growth
        if target_growth is None and target_profile.revenue_growth_rates:
            target_growth = target_profile.revenue_growth_rates[-1]
        if candidate_growth is None and candidate_profile.revenue_growth_rates:
            candidate_growth = candidate_profile.revenue_growth_rates[-1]
        if target_growth is not None and candidate_growth is not None:
            pairs.append(
                (
                    target_growth / Decimal(100),
                    candidate_growth / Decimal(100),
                )
            )
        if (
            left.return_on_invested_capital is not None
            and right.return_on_invested_capital is not None
        ):
            pairs.append(
                (
                    left.return_on_invested_capital / Decimal(100),
                    right.return_on_invested_capital / Decimal(100),
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
        if len(similarities) < cls._MIN_ECONOMIC_SIMILARITY_DIMENSIONS:
            return None
        average = sum(similarities, Decimal(0)) / Decimal(len(similarities))
        return round(float(average * Decimal(100)))

    @staticmethod
    def _market_cap_comparability(target, candidate, parameters):
        if target is None or candidate is None:
            return 0, None, None
        target_cap = target.market_capitalization
        candidate_cap = candidate.market_capitalization
        if (
            target_cap is None
            or candidate_cap is None
            or target_cap <= 0
            or candidate_cap <= 0
            or target.currency != candidate.currency
        ):
            return 0, None, None
        ratio = candidate_cap / target_cap
        if (
            ratio < parameters.minimum_market_cap_ratio
            or ratio > parameters.maximum_market_cap_ratio
        ):
            return (
                0,
                None,
                "Market capitalization is outside the configured "
                f"{parameters.minimum_market_cap_ratio}x-"
                f"{parameters.maximum_market_cap_ratio}x target range",
            )
        proximity = min(ratio, Decimal(1) / ratio)
        score = (
            10
            if proximity >= Decimal("0.67")
            else 7
            if proximity >= Decimal("0.4")
            else 4
        )
        return (
            score,
            "Market capitalization is within the configured "
            f"{parameters.minimum_market_cap_ratio}x-"
            f"{parameters.maximum_market_cap_ratio}x target range",
            None,
        )

    @staticmethod
    def _same_company(target: ValuationProfile, candidate: ValuationProfile) -> bool:
        return bool(
            PeerUniverseSelector._issuer_keys(target)
            & PeerUniverseSelector._issuer_keys(candidate)
        )

    @staticmethod
    def _issuer_keys(profile: ValuationProfile) -> frozenset[str]:
        return issuer_identity_keys(
            company_id=profile.company_id,
            company_name=profile.company_name,
            ticker=profile.ticker,
            identifiers=profile.identifiers,
        )

    @staticmethod
    def _entity_key(name: str, ticker: str | None = None) -> str:
        return normalize_issuer_name(name, ticker=ticker)

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
