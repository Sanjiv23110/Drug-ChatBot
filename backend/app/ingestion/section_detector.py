"""
Production-Grade Section Detection Engine for Pharmaceutical Monographs

This module implements a deterministic, layout-aware section detection system
designed to handle 19,000+ pharmaceutical PDFs with inconsistent layouts.

Architecture:
    Layer 1: Header Candidate Detection (layout signals)
    Layer 2: Text Normalization (deterministic)
    Layer 3: Section Mapping (synonym-based)
    Layer 4: LLM Judge (fallback only)

Author: Medical RAG System
Version: 1.0.0
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import re
import logging
from openai import AzureOpenAI
import os

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION CATEGORY ENUM (STRICT - NO DEVIATIONS)
# ============================================================================

class SectionCategory(str, Enum):
    """
    Fixed enumeration of pharmaceutical monograph sections.
    
    All detected sections MUST map to exactly one of these categories.
    Unknown or unimportant sections → OTHER
    """
    INDICATIONS = "indications"
    DOSAGE = "dosage"
    CONTRAINDICATIONS = "contraindications"
    WARNINGS = "warnings"
    ADVERSE_EFFECTS = "adverse_effects"
    PHARMACOLOGY = "pharmacology"
    INTERACTIONS = "interactions"
    OVERDOSAGE = "overdosage"
    STRUCTURE = "structure"
    STORAGE = "storage"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    ADMINISTRATION = "administration"
    OTHER = "other"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HeaderCandidate:
    """
    Represents a potential section header detected from layout analysis.
    
    Attributes:
        block_id: Unique identifier for the text block
        text: Raw header text
        normalized_text: Cleaned and normalized header text
        font_size: Font size in points
        font_weight: Font weight (400=normal, 700=bold)
        is_all_caps: Whether text is in ALL CAPS
        is_title_case: Whether text is in Title Case
        has_vertical_whitespace: Whether surrounded by blank lines
        confidence: Detection confidence (0.0-1.0)
    """
    block_id: int
    text: str
    normalized_text: str
    font_size: float
    font_weight: int
    is_all_caps: bool
    is_title_case: bool
    has_vertical_whitespace: bool
    confidence: float


@dataclass
class SectionBoundary:
    """
    Represents a detected section with start/end boundaries.
    
    Attributes:
        category: Section category (enum)
        start_block_id: ID of first block in section
        end_block_id: ID of last block in section (exclusive)
        confidence: Classification confidence (0.0-1.0)
        detection_method: How section was detected ("deterministic" | "llm" | "fallback")
        original_header: Raw header text before normalization
    """
    category: SectionCategory
    start_block_id: int
    end_block_id: int
    confidence: float
    detection_method: str
    original_header: str


# ============================================================================
# SECTION SYNONYMS (DETERMINISTIC MAPPING)
# ============================================================================

SECTION_SYNONYMS: Dict[SectionCategory, List[str]] = {
    SectionCategory.INDICATIONS: [
        "indication",
        "indications",
        "uses",
        "therapeutic indications",
        "therapeutic uses",
        "what is used for",
    ],
    SectionCategory.DOSAGE: [
        "dosage",
        "dosage and administration",
        "dose",
        "dosing",
        "how to take",
        "administration",
    ],
    SectionCategory.CONTRAINDICATIONS: [
        "contraindications",
        "contraindication",
        "when not to use",
        "should not use",
    ],
    SectionCategory.WARNINGS: [
        "warnings",
        "warnings and precautions",
        "precautions",
        "cautions",
        "warnings precautions",
    ],
    SectionCategory.ADVERSE_EFFECTS: [
        "adverse reactions",
        "adverse effects",
        "side effects",
        "undesirable effects",
        "unwanted effects",
    ],
    SectionCategory.PHARMACOLOGY: [
        "pharmacology",
        "clinical pharmacology",
        "actions",
        "actions and clinical pharmacology",
        "mechanism of action",
        "pharmacodynamics",
        "pharmacokinetics",
    ],
    SectionCategory.INTERACTIONS: [
        "interactions",
        "drug interactions",
        "interaction",
        "drug drug interactions",
    ],
    SectionCategory.OVERDOSAGE: [
        "overdosage",
        "overdose",
        "symptoms and treatment of overdosage",
        "treatment of overdose",
    ],
    SectionCategory.STRUCTURE: [
        "structural formula",
        "chemical structure",
        "molecular structure",
    ],
    SectionCategory.STORAGE: [
        "storage",
        "storage and stability",
        "how supplied",
        "storage conditions",
    ],
    SectionCategory.PEDIATRICS: [
        "pediatrics",
        "pediatric use",
        "use in children",
    ],
    SectionCategory.GERIATRICS: [
        "geriatrics",
        "geriatric use",
        "use in elderly",
    ],
    SectionCategory.ADMINISTRATION: [
        "administration",
        "how to administer",
        "method of administration",
    ],
}


# ============================================================================
# SECTION DETECTOR (MAIN ENGINE)
# ============================================================================

class SectionDetector:
    """
    Production-grade section detection engine.
    
    Implements a 4-layer pipeline:
        1. Header Candidate Detection (layout signals)
        2. Text Normalization (deterministic)
        3. Section Mapping (synonym-based)
        4. LLM Judge (fallback only)
    
    Usage:
        detector = SectionDetector()
        sections = detector.detect_sections(docling_blocks)
    """
    
    def __init__(self, use_llm_fallback: bool = True):
        """
        Initialize section detector.
        
        Args:
            use_llm_fallback: Whether to use LLM for ambiguous headers
        """
        self.use_llm_fallback = use_llm_fallback
        
        # Initialize Azure OpenAI client if fallback enabled
        if use_llm_fallback:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-08-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        else:
            self.client = None
    
    # ========================================================================
    # LAYER 1: HEADER CANDIDATE DETECTION
    # ========================================================================
    
    def detect_header_candidates(
        self,
        blocks: List[Dict],
        page_median_font_weight: float = 400.0
    ) -> List[HeaderCandidate]:
        """
        Detect potential section headers using layout signals.
        
        A block is a header candidate if ≥2 conditions are met:
            1. Text is ALL CAPS or Title Case
            2. Text length < 80 characters
            3. No sentence-ending punctuation
            4. Surrounded by vertical whitespace
            5. Font weight ≥ median font weight of page
            6. Appears at beginning of a text block
        
        Args:
            blocks: List of Docling text blocks
            page_median_font_weight: Median font weight for the page
        
        Returns:
            List of header candidates
        """
        candidates = []
        
        for i, block in enumerate(blocks):
            text = block.get("text", "").strip()
            if not text:
                continue
            
            # Extract layout features
            font_size = block.get("font_size", 12.0)
            font_weight = block.get("font_weight", 400)
            
            # Signal 1: ALL CAPS or Title Case (STRONG SIGNAL - REQUIRED)
            is_all_caps = text.isupper() and len(text) > 2
            is_title_case = text.istitle()
            has_case_signal = is_all_caps or is_title_case
            
            # Signal 2: Short text (< 80 chars)
            is_short = len(text) < 80
            
            # Signal 3: No sentence-ending punctuation
            has_no_punctuation = not text.endswith(('.', '!', '?', ';', ':'))
            
            # Signal 4: Vertical whitespace (check previous/next blocks)
            prev_block = blocks[i-1] if i > 0 else None
            next_block = blocks[i+1] if i < len(blocks)-1 else None
            
            has_vertical_whitespace = (
                (prev_block is None or not prev_block.get("text", "").strip()) or
                (next_block is None or not next_block.get("text", "").strip())
            )
            
            # Signal 5: Font weight >= median
            is_bold = font_weight >= page_median_font_weight
            
            # Signal 6: Appears at block start (always true for Docling blocks)
            is_block_start = True
            
            # Count signals
            signals = [
                has_case_signal,  # REQUIRED
                is_short,
                has_no_punctuation,
                has_vertical_whitespace,
                is_bold,
                is_block_start
            ]
            
            signal_count = sum(signals)
            
            # STRICT CRITERIA:
            # 1. MUST have case signal (ALL CAPS or Title Case)
            # 2. MUST have at least 3 total signals
            # This prevents list items like "• Sinus bradycardia" from being detected
            if has_case_signal and signal_count >= 3:
                normalized = self.normalize_header_text(text)
                
                # Calculate confidence based on signal count
                confidence = min(0.9, signal_count / 6.0 + 0.3)
                
                candidate = HeaderCandidate(
                    block_id=i,
                    text=text,
                    normalized_text=normalized,
                    font_size=font_size,
                    font_weight=font_weight,
                    is_all_caps=is_all_caps,
                    is_title_case=is_title_case,
                    has_vertical_whitespace=has_vertical_whitespace,
                    confidence=confidence
                )
                
                candidates.append(candidate)
                
                logger.debug(
                    f"Header candidate detected: '{text}' "
                    f"(signals={signal_count}/6, conf={confidence:.2f})"
                )

        
        return candidates
    
    # ========================================================================
    # LAYER 2: TEXT NORMALIZATION
    # ========================================================================
    
    def normalize_header_text(self, text: str) -> str:
        """
        Normalize header text for deterministic matching.
        
        Steps:
            1. Convert to lowercase
            2. Replace "&" with "and"
            3. Remove all non-alphabetic characters except spaces
            4. Collapse multiple spaces
            5. Strip leading/trailing whitespace
        
        Args:
            text: Raw header text
        
        Returns:
            Normalized text
        
        Examples:
            "WARNINGS & PRECAUTIONS" → "warnings and precautions"
            "DOSAGE AND ADMINISTRATION" → "dosage and administration"
            "2 CONTRAINDICATIONS" → "contraindications"
        """
        # Step 1: Lowercase
        text = text.lower()
        
        # Step 2: Replace ampersand
        text = text.replace("&", "and")
        
        # Step 3: Remove non-alphabetic (keep spaces)
        text = re.sub(r"[^a-z ]", "", text)
        
        # Step 4: Collapse spaces
        text = re.sub(r"\s+", " ", text)
        
        # Step 5: Strip
        text = text.strip()
        
        return text
    
    # ========================================================================
    # LAYER 3: DETERMINISTIC SECTION MAPPING
    # ========================================================================
    
    def map_to_section(self, normalized_text: str) -> Tuple[Optional[SectionCategory], float, str]:
        """
        Map normalized header text to section category using synonym matching.
        
        Matching rules:
            1. Exact match → assign section (confidence 1.0)
            2. Substring match → assign section (confidence 0.8)
            3. If multiple matches → prefer longest phrase
            4. If no match → return None
        
        Args:
            normalized_text: Normalized header text
        
        Returns:
            Tuple of (category, confidence, method)
        
        Examples:
            "contraindications" → (CONTRAINDICATIONS, 1.0, "deterministic")
            "warnings and precautions" → (WARNINGS, 1.0, "deterministic")
            "actions" → (PHARMACOLOGY, 1.0, "deterministic")
        """
        best_match = None
        best_confidence = 0.0
        best_phrase_len = 0
        
        for category, synonyms in SECTION_SYNONYMS.items():
            for synonym in synonyms:
                # Exact match
                if normalized_text == synonym:
                    if len(synonym) > best_phrase_len:
                        best_match = category
                        best_confidence = 1.0
                        best_phrase_len = len(synonym)
                
                # Substring match
                elif synonym in normalized_text or normalized_text in synonym:
                    if len(synonym) > best_phrase_len:
                        best_match = category
                        best_confidence = 0.8
                        best_phrase_len = len(synonym)
        
        if best_match:
            logger.debug(
                f"Deterministic match: '{normalized_text}' → {best_match.value} "
                f"(conf={best_confidence:.2f})"
            )
            return best_match, best_confidence, "deterministic"
        
        return None, 0.0, "none"
    
    # ========================================================================
    # LAYER 4: LLM JUDGE (FALLBACK)
    # ========================================================================
    
    def llm_classify_section(self, header_text: str) -> Tuple[Optional[SectionCategory], float]:
        """
        Use LLM to classify ambiguous headers (fallback only).
        
        Args:
            header_text: Normalized header text
        
        Returns:
            Tuple of (category, confidence)
        """
        if not self.use_llm_fallback or not self.client:
            return None, 0.0
        
        # Build enum list for prompt
        enum_list = ", ".join([cat.value.upper() for cat in SectionCategory])
        
        prompt = f"""Given this header text from a pharmaceutical drug monograph:

"{header_text}"

Map it to ONE and ONLY ONE of the following section enums:

[{enum_list}]

Output ONLY the enum. No explanation."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # Try to match result to enum
            for category in SectionCategory:
                if category.value in result:
                    logger.info(
                        f"LLM classified: '{header_text}' → {category.value} (conf=0.6)"
                    )
                    return category, 0.6
            
            # If no match, return OTHER
            logger.warning(f"LLM returned unrecognized category: '{result}'")
            return SectionCategory.OTHER, 0.5
        
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return None, 0.0
    
    # ========================================================================
    # ORCHESTRATOR
    # ========================================================================
    
    def detect_sections(self, blocks: List[Dict]) -> List[SectionBoundary]:
        """
        Main orchestrator: Detect all sections in a document.
        
        Pipeline:
            1. Detect header candidates (layout signals)
            2. Normalize header text
            3. Map to section category (deterministic)
            4. Use LLM fallback if needed
            5. Assign section boundaries
        
        Args:
            blocks: List of Docling text blocks
        
        Returns:
            List of detected section boundaries
        """
        # Calculate median font weight for the page
        font_weights = [b.get("font_weight", 400) for b in blocks if b.get("text")]
        median_weight = sorted(font_weights)[len(font_weights) // 2] if font_weights else 400
        
        # Layer 1: Detect header candidates
        candidates = self.detect_header_candidates(blocks, median_weight)
        
        if not candidates:
            logger.warning("No header candidates detected")
            return []
        
        # Process each candidate
        sections = []
        
        for i, candidate in enumerate(candidates):
            # Layer 3: Deterministic mapping
            category, confidence, method = self.map_to_section(candidate.normalized_text)
            
            # Layer 4: LLM fallback if no deterministic match
            if category is None and self.use_llm_fallback:
                category, confidence = self.llm_classify_section(candidate.normalized_text)
                method = "llm" if category else "fallback"
            
            # If still no match, assign OTHER
            if category is None:
                category = SectionCategory.OTHER
                confidence = 0.3
                method = "fallback"
            
            # Determine section boundaries
            start_block = candidate.block_id
            end_block = candidates[i+1].block_id if i < len(candidates)-1 else len(blocks)
            
            section = SectionBoundary(
                category=category,
                start_block_id=start_block,
                end_block_id=end_block,
                confidence=confidence,
                detection_method=method,
                original_header=candidate.text
            )
            
            sections.append(section)
            
            logger.info(
                f"Section detected: {category.value} "
                f"(blocks {start_block}-{end_block}, conf={confidence:.2f}, method={method})"
            )
        
        return sections
