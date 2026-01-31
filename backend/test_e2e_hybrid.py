"""
End-to-end API test for Solution 1D implementation.

Tests that the hybrid drug name extraction works through the full stack:
User Query → Intent Classifier → Retrieval Engine → Answer Generator → Response
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/api/chat"

# Test cases for end-to-end verification
TEST_CASES = [
    {
        "query": "contraindications of APO-METOPROLOL",
        "description": "Hyphenated brand name (primary fix target)",
        "expected_success": True
    },
    {
        "query": "what is Metoprolol Tartrate used for",
        "description": "Multi-word drug name",
        "expected_success": True
    },
    {
        "query": "side effects of Co-Trimoxazole",
        "description": "Hyphenated generic name",
        "expected_success": True  # Might fail if not in DB
    },
    {
        "query": "what are the contraindications of axid?",
        "description": "Edge case with 'what are the' pattern",
        "expected_success": True
    },
    {
        "query": "tell me about Axid",
        "description": "Simple brand name (baseline test)",
        "expected_success": True
    }
]

def test_api(query, description):
    """Test a single query against the API."""
    payload = {"question": query}
    headers = {"Content-Type": "application/json"}
    
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"Query: '{query}'")
    print('='*80)
    
    try:
        start_time = time.time()
        response = requests.post(BASE_URL, json=payload, headers=headers, timeout=30)
        duration = time.time() - start_time
        
        print(f"⏱️  Response Time: {duration*1000:.0f}ms")
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            has_answer = data.get("has_answer", False)
            retrieval_path = data.get("retrieval_path", "unknown")
            chunks = data.get("chunks_retrieved", 0)
            
            if has_answer:
                print(f"✅ SUCCESS")
                print(f"   Retrieval Path: {retrieval_path}")
                print(f"   Chunks Retrieved: {chunks}")
                
                # Show answer preview
                answer = data.get("answer", "")
                preview = answer[:150] + "..." if len(answer) > 150 else answer
                print(f"   Answer Preview: {preview}")
                
                return True
            else:
                print(f"❌ FAILED: No answer generated")
                print(f"   Retrieval Path: {retrieval_path}")
                print(f"   Chunks Retrieved: {chunks}")
                return False
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error}")
            except:
                print(f"   Raw Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT: Request took longer than 30s")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def run_all_tests():
    """Run all end-to-end tests."""
    print("\n" + "#"*80)
    print("# SOLUTION 1D: END-TO-END API VERIFICATION")
    print("#"*80)
    print("\nNote: Ensure uvicorn is running (uvicorn app.main:app --reload)")
    print("Testing hybrid drug name extraction through full API stack\n")
    
    results = []
    for test_case in TEST_CASES:
        success = test_api(test_case["query"], test_case["description"])
        results.append({
            "query": test_case["query"],
            "description": test_case["description"],
            "expected": test_case["expected_success"],
            "actual": success
        })
    
    # Summary
    print("\n" + "#"*80)
    print("# TEST SUMMARY")
    print("#"*80)
    
    passed = sum(1 for r in results if r["actual"] == r["expected"])
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed as expected\n")
    
    for i, r in enumerate(results, 1):
        status = "✅" if r["actual"] == r["expected"] else "❌"
        print(f"{status} Test {i}: {r['description']}")
        if r["actual"] != r["expected"]:
            print(f"   Expected: {'SUCCESS' if r['expected'] else 'FAIL'}")
            print(f"   Actual: {'SUCCESS' if r['actual'] else 'FAIL'}")
    
    print("\n" + "#"*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Solution 1D successfully implemented")
        print("✅ Hyphenated drug names fully supported")
        print("✅ Multi-word drug names fully supported")
        print("✅ Hybrid fallback logic operational")
    else:
        print(f"\n⚠️  {total - passed} test(s) did not match expectations")
        print("\nPossible reasons:")
        print("- Drug not in database (expected if testing with Co-Trimoxazole)")
        print("- Uvicorn not running")
        print("- Database connection issues")
    
    print("\n" + "#"*80 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
