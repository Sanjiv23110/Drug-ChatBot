"""
Unit Tests for SectionIntentNormalizer

Tests:
1. Intent detection (positive and negative cases)
2. Longest-match priority (substring collision avoidance)
3. Score boosting (correct chunks boosted, others untouched)
4. No-intent passthrough (no modification when intent is None)
5. Audit metadata generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from orchestrator.section_intent_normalizer import SectionIntentNormalizer


class TestSectionIntentDetection(unittest.TestCase):
    """Test intent detection from user queries."""

    def setUp(self):
        self.normalizer = SectionIntentNormalizer(boost_weight=0.15)

    def test_exact_synonym_match(self):
        """Detect intent when query contains exact synonym phrase."""
        result = self.normalizer.detect_intent("clinical pharmacology of Renese")
        self.assertEqual(result, "CLINICAL_PHARMACOLOGY")

    def test_overdose_variants(self):
        """Both 'overdose' and 'overdosage' map to OVERDOSAGE."""
        self.assertEqual(
            self.normalizer.detect_intent("how to treat overdose of dantrium"),
            "OVERDOSAGE"
        )
        self.assertEqual(
            self.normalizer.detect_intent("what is the overdosage information for dantrium"),
            "OVERDOSAGE"
        )

    def test_mechanism_maps_to_clinical_pharmacology(self):
        """'mechanism of action' and 'action' map to CLINICAL_PHARMACOLOGY."""
        self.assertEqual(
            self.normalizer.detect_intent("mechanism of action of lisinopril"),
            "CLINICAL_PHARMACOLOGY"
        )
        self.assertEqual(
            self.normalizer.detect_intent("what is the action of tolinase"),
            "CLINICAL_PHARMACOLOGY"
        )

    def test_adverse_reactions(self):
        """Adverse reaction synonym variants detected."""
        self.assertEqual(
            self.normalizer.detect_intent("what are the side effects of bactrim"),
            "ADVERSE_REACTIONS"
        )
        self.assertEqual(
            self.normalizer.detect_intent("adverse reactions of dantrium"),
            "ADVERSE_REACTIONS"
        )

    def test_no_intent_for_generic_query(self):
        """Return None when no section synonym is present."""
        result = self.normalizer.detect_intent("tell me about dantrium")
        self.assertIsNone(result)

    def test_no_intent_for_drug_only_query(self):
        """Return None for pure drug name queries."""
        result = self.normalizer.detect_intent("dantrium")
        self.assertIsNone(result)

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        self.assertEqual(
            self.normalizer.detect_intent("CLINICAL PHARMACOLOGY of Drug X"),
            "CLINICAL_PHARMACOLOGY"
        )

    def test_longest_match_priority(self):
        """Longer synonym 'mechanism of action' should match before 'mechanism'."""
        result = self.normalizer.detect_intent("explain the mechanism of action")
        self.assertEqual(result, "CLINICAL_PHARMACOLOGY")

    def test_longest_match_overdosage_vs_dosage(self):
        """'overdosage' should NOT match DOSAGE_AND_ADMINISTRATION."""
        result = self.normalizer.detect_intent("overdosage of dantrium")
        self.assertEqual(result, "OVERDOSAGE")


class TestSectionBoostScoring(unittest.TestCase):
    """Test additive score boosting on candidate chunks."""

    def setUp(self):
        self.normalizer = SectionIntentNormalizer(boost_weight=0.20)

    def _make_chunk(self, section_name, score=0.5):
        return {
            "raw_text": f"Sample text from {section_name}",
            "metadata": {"loinc_section": section_name},
            "score": score
        }

    def test_matching_chunks_boosted(self):
        """Chunks with matching section get score boost."""
        candidates = [
            self._make_chunk("OVERDOSE", 0.5),
            self._make_chunk("DOSAGE AND ADMINISTRATION", 0.6),
            self._make_chunk("OVERDOSAGE", 0.4),
        ]
        result = self.normalizer.boost_candidates(candidates, "OVERDOSAGE")

        # OVERDOSE and OVERDOSAGE are both synonyms of OVERDOSAGE
        self.assertAlmostEqual(result[0]["score"], 0.7)  # 0.5 + 0.2
        self.assertAlmostEqual(result[1]["score"], 0.6)  # unchanged
        self.assertAlmostEqual(result[2]["score"], 0.6)  # 0.4 + 0.2

    def test_no_candidates_removed(self):
        """Boost never removes candidates."""
        candidates = [
            self._make_chunk("DESCRIPTION", 0.3),
            self._make_chunk("WARNINGS", 0.4),
        ]
        result = self.normalizer.boost_candidates(candidates, "OVERDOSAGE")
        self.assertEqual(len(result), 2)  # same count

    def test_no_boost_without_intent(self):
        """No modification when intent is None."""
        candidates = [self._make_chunk("OVERDOSE", 0.5)]
        result = self.normalizer.boost_candidates(candidates, None)
        self.assertAlmostEqual(result[0]["score"], 0.5)  # unchanged

    def test_clinical_pharmacology_boost(self):
        """ACTION section boosted when intent is CLINICAL_PHARMACOLOGY."""
        candidates = [
            self._make_chunk("ACTION", 0.4),
            self._make_chunk("DESCRIPTION", 0.5),
        ]
        result = self.normalizer.boost_candidates(candidates, "CLINICAL_PHARMACOLOGY")
        self.assertAlmostEqual(result[0]["score"], 0.6)  # 0.4 + 0.2
        self.assertAlmostEqual(result[1]["score"], 0.5)  # unchanged


class TestNormalizeAndBoostCombined(unittest.TestCase):
    """Test the combined convenience method."""

    def setUp(self):
        self.normalizer = SectionIntentNormalizer(boost_weight=0.15)

    def test_full_pipeline_with_intent(self):
        """Full detection + boost pipeline with valid intent."""
        candidates = [
            {"raw_text": "text", "metadata": {"loinc_section": "OVERDOSE"}, "score": 0.5},
            {"raw_text": "text", "metadata": {"loinc_section": "DOSAGE"}, "score": 0.6},
        ]
        result, audit = self.normalizer.normalize_and_boost(
            "how to treat overdosage of dantrium",
            candidates
        )
        self.assertIsNotNone(audit)
        self.assertEqual(audit["detected_intent"], "OVERDOSAGE")
        self.assertTrue(audit["boost_applied"])
        self.assertAlmostEqual(result[0]["score"], 0.65)  # 0.5 + 0.15

    def test_full_pipeline_without_intent(self):
        """No audit when intent is not detected."""
        candidates = [
            {"raw_text": "text", "metadata": {"loinc_section": "OVERDOSE"}, "score": 0.5},
        ]
        result, audit = self.normalizer.normalize_and_boost(
            "tell me about dantrium",
            candidates
        )
        self.assertIsNone(audit)
        self.assertAlmostEqual(result[0]["score"], 0.5)  # unchanged


if __name__ == "__main__":
    unittest.main()
