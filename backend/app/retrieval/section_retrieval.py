"""
Section-aware retrieval system for exhaustive coverage.

This module implements production-grade section detection and exhaustive retrieval
to guarantee 100% coverage of multi-page sections (e.g., adverse reactions spanning 3 pages).

Architecture:
    1. Detect if query is about a specific section
    2. If YES → Retrieve ALL chunks from that section (exhaustive)
    3. If NO → Use standard top-K retrieval

This solves the fundamental limitation of top-K retrieval for comprehensive lists.
"""
import logging
from typing import List, Dict, Optional, Tuple

# Comprehensive section pattern definitions
# Maps section names to their various representations in queries and PDFs
SECTION_PATTERNS = {
    'adverse_reactions': {
        'query_patterns': [
            'adverse reaction', 'side effect', 'adverse effect',
            'undesirable effect', 'reaction', 'adverse event',
            'adverse experience', 'unwanted effect', 'toxic effect',
            'harmful effect', 'negative effect'
        ],
        'pdf_section_names': [
            'ADVERSE REACTIONS', 'SIDE EFFECTS', 'ADVERSE EFFECTS',
            'UNDESIRABLE EFFECTS', 'ADVERSE EVENTS', 'REACTIONS'
        ]
    },
    'contraindications': {
        'query_patterns': [
            'contraindication', 'when not to use', 'should not use',
            'should not take', 'avoid', 'do not use', 'must not use',
            'not recommended', 'forbidden', 'prohibited'
        ],
        'pdf_section_names': [
            'CONTRAINDICATIONS', 'WHEN NOT TO USE', 'DO NOT USE',
            'WARNINGS AND PRECAUTIONS'
        ]
    },
    'warnings': {
        'query_patterns': [
            'warning', 'precaution', 'caution', 'alert',
            'important safety information', 'safety warning',
            'boxed warning', 'black box warning'
        ],
        'pdf_section_names': [
            'WARNINGS', 'PRECAUTIONS', 'WARNINGS AND PRECAUTIONS',
            'CAUTIONS', 'IMPORTANT SAFETY INFORMATION'
        ]
    },
    'dosage': {
        'query_patterns': [
            'dosage', 'dose', 'dosing', 'administration',
            'how to take', 'how to use', 'recommended dose',
            'dosage and administration', 'posology'
        ],
        'pdf_section_names': [
            'DOSAGE AND ADMINISTRATION', 'DOSAGE', 'DOSE',
            'ADMINISTRATION', 'POSOLOGY', 'HOW TO USE'
        ]
    },
    'drug_interactions': {
        'query_patterns': [
            'drug interaction', 'interaction', 'drug-drug interaction',
            'food interaction', 'drug-food interaction', 'interact with',
            'combination with', 'take with'
        ],
        'pdf_section_names': [
            'DRUG INTERACTIONS', 'INTERACTIONS', 'DRUG-DRUG INTERACTIONS',
            'FOOD INTERACTIONS', 'DRUG-FOOD INTERACTIONS'
        ]
    },
    'pharmacology': {
        'query_patterns': [
            'mechanism of action', 'how it works', 'how does it work',
            'pharmacology', 'pharmacodynamics', 'pharmacokinetics',
            'mode of action', 'action mechanism'
        ],
        'pdf_section_names': [
            'CLINICAL PHARMACOLOGY', 'MECHANISM OF ACTION',
            'PHARMACOLOGY', 'PHARMACODYNAMICS', 'PHARMACOKINETICS',
            'MODE OF ACTION'
        ]
    },
    'indications': {
        'query_patterns': [
            'indication', 'what is it for', 'what is it used for',
            'used to treat', 'treatment of', 'therapeutic use',
            'medical use', 'approved for'
        ],
        'pdf_section_names': [
            'INDICATIONS AND USAGE', 'INDICATIONS', 'THERAPEUTIC INDICATIONS',
            'USAGE', 'CLINICAL USE'
        ]
    },
    'overdose': {
        'query_patterns': [
            'overdose', 'too much', 'excessive dose', 'toxic dose',
            'poisoning', 'intoxication', 'over medication'
        ],
        'pdf_section_names': [
            'OVERDOSAGE', 'OVERDOSE', 'TOXICITY', 'POISONING'
        ]
    }
}


class SectionDetector:
    """
    Detects if a query is asking about a specific medical section.
    
    Uses pattern matching to identify section-specific queries that
    require exhaustive retrieval rather than top-K.
    """
    
    def __init__(self):
        """Initialize section detector."""
        self.section_patterns = SECTION_PATTERNS
        logging.info("SectionDetector initialized with %d section types", len(SECTION_PATTERNS))
    
    def detect_section(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Detect if query is about a specific section.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (section_name, confidence) if detected, None otherwise
            
        Example:
            "What are the adverse reactions of Gravol?"
            → ('adverse_reactions', 0.95)
        """
        query_lower = query.lower()
        
        best_match = None
        best_confidence = 0.0
        
        for section_name, patterns_dict in self.section_patterns.items():
            query_patterns = patterns_dict['query_patterns']
            
            # Check if any pattern matches
            for pattern in query_patterns:
                if pattern in query_lower:
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(pattern, query_lower)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = section_name
        
        # Only return if confidence is high enough
        if best_match and best_confidence >= 0.5:
            logging.info(
                f"Section detected: {best_match} (confidence: {best_confidence:.2f})"
            )
            return (best_match, best_confidence)
        
        return None
    
    def _calculate_confidence(self, pattern: str, query: str) -> float:
        """
        Calculate confidence score for pattern match.
        
        Higher confidence for:
        - Longer patterns
        - Exact phrase matches
        - Query focuses on the pattern
        
        Args:
            pattern: Matched pattern
            query: Full query text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence
        confidence = 0.7
        
        # Bonus for longer patterns (more specific)
        if len(pattern.split()) >= 2:
            confidence += 0.15
        
        # Bonus if pattern is a significant part of query
        pattern_words = len(pattern.split())
        query_words = len(query.split())
        
        if query_words > 0:
            ratio = pattern_words / query_words
            if ratio > 0.3:  # Pattern is >30% of query
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    def get_pdf_section_names(self, section_type: str) -> List[str]:
        """
        Get possible section names as they appear in PDFs.
        
        Args:
            section_type: Section type (e.g., 'adverse_reactions')
            
        Returns:
            List of section names to match in PDF metadata
        """
        if section_type in self.section_patterns:
            return self.section_patterns[section_type]['pdf_section_names']
        return []


class ExhaustiveSectionRetriever:
    """
    Retrieves ALL chunks from a specific section.
    
    Guarantees 100% coverage for section queries by retrieving
    every chunk belonging to the detected section, regardless of
    semantic similarity scores.
    """
    
    def __init__(self, metadata_store):
        """
        Initialize exhaustive retriever.
        
        Args:
            metadata_store: SQLiteMetadataStore instance
        """
        self.metadata_store = metadata_store
        self.section_detector = SectionDetector()
        logging.info("ExhaustiveSectionRetriever initialized")
    
    def retrieve_section_exhaustive(
        self,
        drug_name: str,
        section_type: str
    ) -> List[Dict]:
        """
        Retrieve ALL chunks from a specific section.
        
        This bypasses top-K retrieval and gets every chunk from the section,
        ensuring complete coverage of multi-page lists.
        
        Args:
            drug_name: Drug name (generic or brand)
            section_type: Section type (e.g., 'adverse_reactions')
            
        Returns:
            List of ALL chunks from section, sorted by page and position
            
        Example:
            retrieve_section_exhaustive('trimipramine', 'adverse_reactions')
            → Returns ALL 24 chunks from pages 6-8
        """
        # Get possible section names
        pdf_section_names = self.section_detector.get_pdf_section_names(section_type)
        
        if not pdf_section_names:
            logging.warning(f"Unknown section type: {section_type}")
            return []
        
        # Query database for ALL chunks matching section
        all_chunks = []
        
        for section_name in pdf_section_names:
            chunks = self.metadata_store.get_chunks_by_drug_and_section(
                drug_name=drug_name,
                section_name=section_name
            )
            all_chunks.extend(chunks)
        
        # Remove duplicates (same chunk matched multiple section names)
        unique_chunks = self._deduplicate_chunks(all_chunks)
        
        # Sort by page number and chunk index for proper order
        unique_chunks.sort(key=lambda x: (
            x.get('page_num', 0),
            x.get('chunk_index', 0)
        ))
        
        logging.info(
            f"Exhaustive retrieval: {len(unique_chunks)} chunks from "
            f"'{section_type}' section for {drug_name}"
        )
        
        return unique_chunks
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on chunk_id."""
        seen = set()
        unique = []
        
        for chunk in chunks:
            chunk_id = f"{chunk.get('file_path', '')}_{chunk.get('chunk_index', -1)}"
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk)
        
        return unique
