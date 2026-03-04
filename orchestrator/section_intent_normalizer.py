"""
Section Intent Normalizer - Pre-Retrieval Semantic Normalization Layer

Resolves semantic mismatches between user query terminology and SPL section titles.
Example: User says "clinical pharmacology" but SPL section is titled "ACTION".

Design:
- Executes BEFORE hybrid retrieval
- Deterministic, O(N) over small synonym list (N < 200 terms)
- No embeddings, no fuzzy matching, no LLM
- Returns canonical intent label or None
- Never blocks retrieval, never filters candidates
- Provides boost metadata for post-retrieval score adjustment

Integration:
- Used by HybridRetriever to apply additive score boost on matching chunks
- Cross-encoder remains final authority on ranking
"""

import re
import logging
from typing import Optional, Dict, List

from config.section_intent_map import SECTION_INTENT_MAP, DEFAULT_SECTION_BOOST_WEIGHT

logger = logging.getLogger(__name__)


class SectionIntentNormalizer:
    """
    Pre-retrieval section intent detection and post-retrieval score boosting.

    Responsibilities:
    1. Detect canonical section intent from user query (pre-retrieval)
    2. Apply additive score boost to matching chunks (post-retrieval, pre-rerank)

    Does NOT:
    - Filter candidates
    - Override cross-encoder scoring
    - Use embeddings or LLM
    - Hardcode drug-specific logic
    """

    def __init__(self, boost_weight: Optional[float] = None):
        """
        Args:
            boost_weight: Additive score boost for section-matched chunks.
                          Defaults to DEFAULT_SECTION_BOOST_WEIGHT from config.
        """
        self.boost_weight = boost_weight if boost_weight is not None else DEFAULT_SECTION_BOOST_WEIGHT

        # Build reverse lookup: synonym_phrase -> canonical_group
        # Sort by phrase length descending to prioritize longer matches
        self._synonym_lookup: List[tuple] = []
        for canonical, synonyms in SECTION_INTENT_MAP.items():
            for phrase in synonyms:
                self._synonym_lookup.append((phrase.lower(), canonical))

        # Sort by phrase length descending (longest first to avoid substring collisions)
        self._synonym_lookup.sort(key=lambda x: len(x[0]), reverse=True)

        logger.info(
            f"SectionIntentNormalizer: Loaded {len(self._synonym_lookup)} synonym phrases "
            f"across {len(SECTION_INTENT_MAP)} canonical groups (boost_weight={self.boost_weight})"
        )

    def detect_intent(self, query: str) -> Optional[str]:
        """
        Detect canonical section intent from user query.

        Algorithm:
        1. Normalize query (lowercase, strip punctuation)
        2. Scan synonym list (longest-first) for substring match
        3. Return canonical group key or None

        Args:
            query: Raw user query string

        Returns:
            Canonical section intent key (e.g., "CLINICAL_PHARMACOLOGY") or None
        """
        normalized = self._normalize_query(query)

        for phrase, canonical in self._synonym_lookup:
            if phrase in normalized:
                logger.info(f"SectionIntentNormalizer: Detected intent '{canonical}' "
                            f"(matched phrase: '{phrase}') for query: '{query}'")
                return canonical

        return None

    def boost_candidates(
        self,
        candidates: List[Dict],
        detected_intent: str
    ) -> List[Dict]:
        """
        Apply additive score boost to candidates whose loinc_section
        matches any synonym in the detected canonical group.

        Operates on post-retrieval, pre-rerank candidates.
        Does NOT remove any candidates. Cross-encoder remains final authority.

        Args:
            candidates: List of chunk dicts with 'metadata.loinc_section' and 'score'
            detected_intent: Canonical group key from detect_intent()

        Returns:
            Same candidates list with 'score' adjusted where applicable
        """
        if not detected_intent or detected_intent not in SECTION_INTENT_MAP:
            return candidates

        # Get all synonyms for this canonical group (lowercase)
        target_synonyms = {s.lower() for s in SECTION_INTENT_MAP[detected_intent]}
        boosted_count = 0

        for chunk in candidates:
            chunk_section = self._get_chunk_section(chunk)
            if not chunk_section:
                continue

            chunk_section_lower = chunk_section.lower()

            # Check if chunk's section matches any synonym in the canonical group
            if chunk_section_lower in target_synonyms:
                chunk['score'] = chunk.get('score', 0.0) + self.boost_weight
                boosted_count += 1

        if boosted_count > 0:
            logger.info(
                f"SectionIntentNormalizer: Boosted {boosted_count}/{len(candidates)} "
                f"candidates for intent '{detected_intent}' (boost={self.boost_weight})"
            )

        return candidates

    def normalize_and_boost(
        self,
        query: str,
        candidates: List[Dict]
    ) -> tuple:
        """
        Combined convenience method: detect intent + apply boost.

        Args:
            query: Raw user query
            candidates: Retrieved chunks (pre-rerank)

        Returns:
            (boosted_candidates, audit_metadata)
        """
        detected_intent = self.detect_intent(query)

        audit = None
        if detected_intent:
            candidates = self.boost_candidates(candidates, detected_intent)
            audit = {
                "query": query,
                "detected_intent": detected_intent,
                "boost_applied": True,
                "boost_weight": self.boost_weight,
                "candidates_count": len(candidates),
            }

        return candidates, audit

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Lowercase and strip punctuation for deterministic matching."""
        normalized = query.lower()
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)
        normalized = ' '.join(normalized.split())
        return normalized

    @staticmethod
    def _get_chunk_section(chunk: Dict) -> Optional[str]:
        """
        Extract section name from chunk, handling both flat and nested metadata.
        Checks multiple possible field locations for scalability.
        """
        # Nested metadata (standard format from QdrantManager)
        metadata = chunk.get('metadata', {})
        section = metadata.get('loinc_section')
        if section:
            return section

        # Top-level fallback (some pipelines flatten metadata)
        return chunk.get('loinc_section')
