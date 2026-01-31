"""
Unit Tests for Section Detection Engine

Tests the 4-layer pipeline:
    - Header candidate detection
    - Text normalization
    - Deterministic section mapping
    - LLM fallback (mocked)
"""

import pytest
from app.ingestion.section_detector import (
    SectionDetector,
    SectionCategory,
    HeaderCandidate,
    SectionBoundary,
    SECTION_SYNONYMS
)


class TestTextNormalization:
    """Test Layer 2: Text Normalization"""
    
    def setup_method(self):
        self.detector = SectionDetector(use_llm_fallback=False)
    
    def test_normalize_warnings_and_precautions(self):
        """Test: WARNINGS & PRECAUTIONS → warnings and precautions"""
        result = self.detector.normalize_header_text("WARNINGS & PRECAUTIONS")
        assert result == "warnings and precautions"
    
    def test_normalize_dosage_and_administration(self):
        """Test: DOSAGE AND ADMINISTRATION → dosage and administration"""
        result = self.detector.normalize_header_text("DOSAGE AND ADMINISTRATION")
        assert result == "dosage and administration"
    
    def test_normalize_numbered_header(self):
        """Test: 2 CONTRAINDICATIONS → contraindications"""
        result = self.detector.normalize_header_text("2 CONTRAINDICATIONS")
        assert result == "contraindications"
    
    def test_normalize_actions_and_clinical_pharmacology(self):
        """Test: ACTIONS AND CLINICAL PHARMACOLOGY → actions and clinical pharmacology"""
        result = self.detector.normalize_header_text("ACTIONS AND CLINICAL PHARMACOLOGY")
        assert result == "actions and clinical pharmacology"
    
    def test_normalize_with_special_chars(self):
        """Test: Side Effects (Unwanted) → side effects unwanted"""
        result = self.detector.normalize_header_text("Side Effects (Unwanted)")
        assert result == "side effects unwanted"


class TestDeterministicMapping:
    """Test Layer 3: Deterministic Section Mapping"""
    
    def setup_method(self):
        self.detector = SectionDetector(use_llm_fallback=False)
    
    def test_exact_match_contraindications(self):
        """Test exact match: contraindications"""
        category, confidence, method = self.detector.map_to_section("contraindications")
        assert category == SectionCategory.CONTRAINDICATIONS
        assert confidence == 1.0
        assert method == "deterministic"
    
    def test_exact_match_warnings_and_precautions(self):
        """Test exact match: warnings and precautions"""
        category, confidence, method = self.detector.map_to_section("warnings and precautions")
        assert category == SectionCategory.WARNINGS
        assert confidence == 1.0
        assert method == "deterministic"
    
    def test_substring_match_actions(self):
        """Test substring match: actions → PHARMACOLOGY"""
        category, confidence, method = self.detector.map_to_section("actions")
        assert category == SectionCategory.PHARMACOLOGY
        assert confidence >= 0.8
        assert method == "deterministic"
    
    def test_no_match_returns_none(self):
        """Test no match returns None"""
        category, confidence, method = self.detector.map_to_section("random text here")
        assert category is None
        assert confidence == 0.0
        assert method == "none"
    
    def test_prefer_longest_phrase(self):
        """Test: prefer longest matching phrase"""
        # "dosage and administration" should match DOSAGE, not ADMINISTRATION
        category, confidence, method = self.detector.map_to_section("dosage and administration")
        assert category == SectionCategory.DOSAGE
        assert confidence == 1.0


class TestHeaderCandidateDetection:
    """Test Layer 1: Header Candidate Detection"""
    
    def setup_method(self):
        self.detector = SectionDetector(use_llm_fallback=False)
    
    def test_detect_all_caps_header(self):
        """Test: ALL CAPS text detected as header"""
        blocks = [
            {"text": "CONTRAINDICATIONS", "font_size": 14, "font_weight": 700},
            {"text": ""},  # Whitespace
            {"text": "Patients who are hypersensitive...", "font_weight": 400},
        ]
        
        candidates = self.detector.detect_header_candidates(blocks, page_median_font_weight=400)
        
        assert len(candidates) >= 1
        assert candidates[0].text == "CONTRAINDICATIONS"
        assert candidates[0].is_all_caps is True
    
    def test_detect_title_case_header(self):
        """Test: Title Case text detected as header"""
        blocks = [
            {"text": "Warnings And Precautions", "font_size": 12, "font_weight": 600},
            {"text": ""},
            {"text": "Use with caution...", "font_weight": 400},
        ]
        
        candidates = self.detector.detect_header_candidates(blocks, page_median_font_weight=400)
        
        assert len(candidates) >= 1
        assert candidates[0].text == "Warnings And Precautions"
        assert candidates[0].is_title_case is True
    
    def test_ignore_long_text(self):
        """Test: Long text (>80 chars) not detected as header"""
        blocks = [
            {
                "text": "This is a very long paragraph that should not be detected as a header because it exceeds 80 characters",
                "font_weight": 700
            },
        ]
        
        candidates = self.detector.detect_header_candidates(blocks, page_median_font_weight=400)
        
        # Should have low signal count (only bold, but too long)
        assert len(candidates) == 0 or candidates[0].confidence < 0.5
    
    def test_ignore_sentence_with_punctuation(self):
        """Test: Text ending with punctuation not detected as header"""
        blocks = [
            {"text": "This is a sentence.", "font_weight": 700},
        ]
        
        candidates = self.detector.detect_header_candidates(blocks, page_median_font_weight=400)
        
        # Should have low signal count
        assert len(candidates) == 0 or candidates[0].confidence < 0.5


class TestAPOMetoprololCase:
    """Test the specific APO-METOPROLOL failure case"""
    
    def setup_method(self):
        self.detector = SectionDetector(use_llm_fallback=False)
    
    def test_bold_text_not_detected_as_header(self):
        """
        Critical test: Ensure bold text is NOT detected as a section header.
        
        This is the root cause of Issue #2:
        - "APO-METOPROLOL is contraindicated in patients with:" was detected as header
        - This split the section, causing partial answers
        """
        blocks = [
            # TRUE header
            {
                "text": "2 CONTRAINDICATIONS",
                "font_size": 14,
                "font_weight": 700
            },
            {"text": ""},  # Whitespace
            # Content paragraph
            {
                "text": "Patients who are hypersensitive to this drug or to any ingredient in the formulation.",
                "font_weight": 400
            },
            # BOLD TEXT (should NOT be detected as header)
            {
                "text": "APO-METOPROLOL is contraindicated in patients with:",
                "font_weight": 600  # Bold but not a header
            },
            # List items (should belong to same section)
            {"text": "• Sinus bradycardia", "font_weight": 400},
            {"text": "• Sick sinus syndrome", "font_weight": 400},
            {"text": "• Second and third degree A-V block", "font_weight": 400},
        ]
        
        candidates = self.detector.detect_header_candidates(blocks, page_median_font_weight=400)
        
        # Should detect ONLY the first block as a header
        assert len(candidates) == 1, f"Expected 1 header, got {len(candidates)}"
        assert candidates[0].block_id == 0, "First block should be the header"
        assert candidates[0].text == "2 CONTRAINDICATIONS"
    
    def test_full_section_detection(self):
        """Test: Full section detection for APO-METOPROLOL contraindications"""
        blocks = [
            {"text": "2 CONTRAINDICATIONS", "font_size": 14, "font_weight": 700},
            {"text": ""},
            {"text": "Patients who are hypersensitive...", "font_weight": 400},
            {"text": "APO-METOPROLOL is contraindicated in patients with:", "font_weight": 600},
            {"text": "• Sinus bradycardia", "font_weight": 400},
            {"text": "• Sick sinus syndrome", "font_weight": 400},
            {"text": "3 WARNINGS", "font_size": 14, "font_weight": 700},  # Next section
        ]
        
        sections = self.detector.detect_sections(blocks)
        
        # Should detect 2 sections
        assert len(sections) == 2
        
        # First section: CONTRAINDICATIONS
        assert sections[0].category == SectionCategory.CONTRAINDICATIONS
        assert sections[0].start_block_id == 0
        assert sections[0].end_block_id == 6  # Includes all content until next header
        
        # Second section: WARNINGS
        assert sections[1].category == SectionCategory.WARNINGS
        assert sections[1].start_block_id == 6


class TestConfidenceScoring:
    """Test confidence scoring logic"""
    
    def setup_method(self):
        self.detector = SectionDetector(use_llm_fallback=False)
    
    def test_deterministic_exact_match_high_confidence(self):
        """Test: Deterministic exact match → confidence 1.0"""
        category, confidence, method = self.detector.map_to_section("contraindications")
        assert confidence == 1.0
        assert method == "deterministic"
    
    def test_deterministic_substring_match_medium_confidence(self):
        """Test: Deterministic substring match → confidence 0.8"""
        category, confidence, method = self.detector.map_to_section("actions")
        assert confidence >= 0.8
        assert method == "deterministic"
    
    def test_fallback_low_confidence(self):
        """Test: No match → fallback to OTHER with low confidence"""
        blocks = [
            {"text": "RANDOM SECTION", "font_weight": 700},
        ]
        
        sections = self.detector.detect_sections(blocks)
        
        if sections:
            assert sections[0].category == SectionCategory.OTHER
            assert sections[0].confidence < 0.5


class TestRealWorldHeaders:
    """Test real-world pharmaceutical headers"""
    
    def setup_method(self):
        self.detector = SectionDetector(use_llm_fallback=False)
    
    @pytest.mark.parametrize("header,expected_category", [
        ("INDICATIONS AND CLINICAL USE", SectionCategory.INDICATIONS),
        ("CONTRAINDICATIONS", SectionCategory.CONTRAINDICATIONS),
        ("WARNINGS AND PRECAUTIONS", SectionCategory.WARNINGS),
        ("ADVERSE REACTIONS", SectionCategory.ADVERSE_EFFECTS),
        ("DOSAGE AND ADMINISTRATION", SectionCategory.DOSAGE),
        ("OVERDOSAGE", SectionCategory.OVERDOSAGE),
        ("ACTION AND CLINICAL PHARMACOLOGY", SectionCategory.PHARMACOLOGY),
        ("DRUG INTERACTIONS", SectionCategory.INTERACTIONS),
        ("STORAGE AND STABILITY", SectionCategory.STORAGE),
    ])
    def test_real_world_headers(self, header, expected_category):
        """Test: Real pharmaceutical headers map correctly"""
        normalized = self.detector.normalize_header_text(header)
        category, confidence, method = self.detector.map_to_section(normalized)
        
        assert category == expected_category, f"Failed for header: {header}"
        assert confidence >= 0.8
        assert method == "deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
