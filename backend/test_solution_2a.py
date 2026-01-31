"""
Test script for Solution 2A: Multi-Field OR Query

Tests that brand names and generic names work correctly for retrieval.
"""

import asyncio
import sys
sys.path.insert(0, 'C:\\G\\Maclens chatbot w api\\backend')

from app.retrieval.intent_classifier import IntentClassifier
from app.retrieval.retrieve import MonographRetriever

# Test cases for brand/generic name retrieval
TEST_CASES = [
    # (query, expected_drug_extracted, expected_to_find_results, description)
    ("what is Axid used for", "axid", True, "Brand name 'Axid' should match brand_name='axid' in DB"),
    ("indications of nizatidine", "nizatidine", True, "Generic name should match drug_name='nizatidine'"),
    ("contraindications of Axid", "axid", True, "Brand name with section query"),
    ("Axid dosage", "axid", True, "Brand name simple query"),
]

async def test_solution_2a():
    """Test Solution 2A multi-field OR query implementation."""
    print("\n" + "="*80)
    print("SOLUTION 2A: MULTI-FIELD OR QUERY TEST")
    print("="*80)
    
    print("\nTesting brand/generic name retrieval with OR query logic")
    print("Database has: drug_name='nizatidine', brand_name='axid'\n")
    
    classifier = IntentClassifier(use_llm_fallback=False)
    retriever = MonographRetriever()
    
    passed = 0
    failed = 0
    
    for query, expected_drug, should_find, description in TEST_CASES:
        print(f"\n{'='*80}")
        print(f"TEST: {description}")
        print(f"Query: '{query}'")
        print('='*80)
        
        # Step 1: Test intent extraction
        intent = classifier.classify(query)
        extracted_drug = intent.target_drug
        extracted_section = intent.target_section
        
        print(f"✓ Intent Extraction:")
        print(f"  Drug: '{extracted_drug}' (expected: '{expected_drug}')")
        print(f"  Section: '{extracted_section}'")
        
        if extracted_drug != expected_drug:
            print(f"  ❌ FAIL: Drug extraction mismatch")
            failed += 1
            continue
        
        # Step 2: Test retrieval
        try:
            result = await retriever.retrieve(intent)
            found_results = result.total_results > 0
            
            print(f"\n✓ Retrieval Execution:")
            print(f"  Path Used: {result.path_used}")
            print(f"  Results: {result.total_results} sections found")
            
            if found_results == should_find:
                print(f"\n✅ PASS: {'Found' if found_results else 'Did not find'} results as expected")
                passed += 1
                
                # Show snippet of retrieved data
                if found_results and result.sections:
                    section = result.sections[0]
                    text_preview = section.get('chunk_text', '')[:100]
                    print(f"  Sample: {text_preview}...")
            else:
                print(f"\n❌ FAIL: Expected to {'find' if should_find else 'not find'} results")
                print(f"  SQL Executed: {result.sql_executed[:200]}...")
                failed += 1
                
        except Exception as e:
            import traceback
            print(f"\n❌ EXCEPTION: {e}")
            print(traceback.format_exc())
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("="*80)
    
    if passed == len(TEST_CASES):
        print("\n✅ SOLUTION 2A WORKING CORRECTLY")
        print("   - Brand names retrieve successfully (e.g., 'Axid' → nizatidine)")
        print("   - Generic names retrieve successfully")
        print("   - OR query logic functional across all paths")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        print("\nPossible reasons:")
        print("- Drug not in database")
        print("- Section name mismatch (check normalization)")
        print("- Database connection issue")
    
    print("\n" + "="*80 + "\n")
    
    return passed == len(TEST_CASES)

if __name__ == "__main__":
    try:
        success = asyncio.run(test_solution_2a())
        exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
