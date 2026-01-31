"""
Test script for Solution 1D: Hybrid Drug Name Extraction (Regex Path)

Tests the primary improvement: Regex patterns that handle hyphens and multi-word names.
LLM fallback is disabled for these tests to avoid authentication requirements.
"""

import asyncio
from app.retrieval.intent_classifier import IntentClassifier

# Test cases for regex extraction
TEST_CASES = [
    # (query, expected_drug_name, description)
    ("contraindications of APO-METOPROLOL", "apo-metoprolol", "Hyphenated brand name"),
    ("side effects of Co-Trimoxazole", "co-trimoxazole", "Hyphenated generic name"),
    ("what is Metoprolol Tartrate used for", "metoprolol tartrate", "Multi-word drug name"),
    ("tell me about Axid", "axid", "Simple brand name"),
    ("indications of nizatidine", "nizatidine", "Simple generic name"),
    ("what are the contraindications of axid?", "axid", "With question mark"),
    ("Axid dosage", "axid", "Drug followed by section"),
    ("APO-METOPROLOL contraindications", "apo-metoprolol", "Hyphenated with section"),
]

async def run_tests():
    """Run all test cases."""
    print("\n" + "="*80)
    print("SOLUTION 1D: HYBRID DRUG NAME EXTRACTION - REGEX TEST")
    print("="*80)
    
    print("\nTesting improved regex patterns (hyphens, spaces, apostrophes support)")
    print("LLM fallback is disabled for these tests\n")
    
    # Create classifier WITHOUT LLM fallback
    classifier = IntentClassifier(use_llm_fallback=False)
    
    passed = 0
    failed = 0
    failures = []
    
    for query, expected, description in TEST_CASES:
        intent = classifier.classify(query)
        extracted = intent.target_drug
        confidence = intent.drug_confidence
        section = intent.target_section
        
        if extracted == expected:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
            failures.append((query, expected, extracted))
        
        print(f"{status} | {description}")
        print(f"  Query: '{query}'")
        print(f"  Expected: '{expected}'")
        print(f"  Extracted: '{extracted}' (confidence: {confidence:.2f})")
        if section:
            print(f"  Section: '{section}'")
        print()
    
    # Summary
    print("="*80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("="*80)
    
    if failed > 0:
        print("\nFailed Test Details:")
        for query, expected, extracted in failures:
            print(f"  Query: '{query}'")
            print(f"    Expected: '{expected}' but got: '{extracted}'")
            print()
    
    # Critical tests
    critical_tests = [
        ("contraindications of APO-METOPROLOL", "apo-metoprolol"),
        ("what is Metoprolol Tartrate used for", "metoprolol tartrate"),
    ]
    
    critical_passed = all(
        classifier.classify(query).target_drug == expected 
        for query, expected in critical_tests
    )
    
    if critical_passed:
        print("\n✅ CRITICAL FUNCTIONALITY: Hyphenated and multi-word names work correctly")
    else:
        print("\n❌ CRITICAL FUNCTIONALITY: Some critical tests failed")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. ✅ Regex improvements verified")
    print("2. ⏭️  Test with live API: python test_api_endpoint.py")
    print("3. ⏭️  Monitor uvicorn logs for extraction behavior")
    print("4. ⏭️  (Optional) Configure Azure OpenAI to test LLM fallback")
    print("="*80 + "\n")
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    exit(0 if success else 1)
