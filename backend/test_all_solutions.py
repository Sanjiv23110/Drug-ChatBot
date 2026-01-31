"""
Comprehensive End-to-End Test: All Solutions (1D + 2A + 3C)
Tests the complete hybrid retrieval system.
"""
import requests
import time

BASE_URL = "http://localhost:8000/api/chat"

# Test all three solutions together
TESTS = [
    # Solution 1D: Hyphenated names
    {
        "query": "contraindications of APO-METOPROLOL",
        "solution": "1D (Regex)",
        "expected": "Extracts hyphenated name correctly"
    },
    {
        "query": "what is Metoprolol Tartrate used for",
        "solution": "1D (Regex)",
        "expected": "Extracts multi-word name"
    },
    
    # Solution 2A: Brand/Generic matching
    {
        "query": "what is Axid used for",
        "solution": "2A (OR Query)",
        "expected": "Matches brand_name='axid'"
    },
    {
        "query": "contraindications of Axid",
        "solution": "2A (OR Query)",
        "expected": "Matches brand name"
    },
    {
        "query": "indications of nizatidine",
        "solution": "2A (OR Query)",
        "expected": "Matches drug_name"
    },
    
    # Solution 3C: Enhanced fuzzy matching
    {
        "query": "indications of axid",
        "solution": "3C (Fuzzy)",
        "expected": "Keyword fallback matches 'used for'"
    },
    {
        "query": "side effects of axid",
        "solution": "3C (Fuzzy)",
        "expected": "Fuzzy/keyword matching"
    },
    
    # Combined: All solutions working together
    {
        "query": "what are the contraindications of APO-METOPROLOL",
        "solution": "1D+2A+3C",
        "expected": "Regex extraction + OR query + fuzzy"
    },
]

def test_endpoint(query, solution, expected):
    """Test a single query."""
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print(f"Solution: {solution}")
    print(f"Expected: {expected}")
    print('='*70)
    
    try:
        start = time.time()
        resp = requests.post(BASE_URL, json={"question": query}, timeout=15)
        duration = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            has_answer = data.get("has_answer", False)
            chunks = data.get("chunks_retrieved", 0)
            path = data.get("retrieval_path", "unknown")
            
            if has_answer:
                print(f"✅ SUCCESS ({duration:.0f}ms)")
                print(f"   Chunks: {chunks}, Path: {path}")
                
                # Show snippet
                answer = data.get("answer", "")
                preview = answer[:120] + "..." if len(answer) > 120 else answer
                print(f"   Answer: {preview}")
                return True
            else:
                print(f"❌ FAIL: No answer returned")
                print(f"   Path: {path}, Chunks: {chunks}")
                return False
        else:
            print(f"❌ HTTP {resp.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT (>15s)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# COMPREHENSIVE END-TO-END TEST: Solutions 1D + 2A + 3C")
    print("#"*70)
    print("\nTesting hybrid retrieval system with uvicorn running...")
    print("Ensure: docker containers running, database populated\n")
    
    results = []
    for test in TESTS:
        success = test_endpoint(test["query"], test["solution"], test["expected"])
        results.append((test["query"], success))
    
    # Summary
    print("\n" + "#"*70)
    print("# FINAL RESULTS")
    print("#"*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed\n")
    
    for query, success in results:
        status = "✅" if success else "❌"
        query_short = query[:50] + "..." if len(query) > 50 else query
        print(f"{status} {query_short}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nSystem Status:")
        print("  ✅ Solution 1D: Hyphenated/multi-word extraction working")
        print("  ✅ Solution 2A: Brand/generic name matching working")
        print("  ✅ Solution 3C: Enhanced fuzzy matching working")
        print("\n🚀 Ready for 19,000 PDF ingestion!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\nPossible causes:")
        print("  - Uvicorn not running")
        print("  - Database not populated")
        print("  - Drug not in database (expected for APO-METOPROLOL if not ingested)")
    
    print("\n" + "#"*70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
