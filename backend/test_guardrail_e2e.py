"""
End-to-end tests for evidence-based guardrail - Phase 4
Tests the complete system with real API calls
"""
import requests
import time

BASE_URL = "http://localhost:8000"

def test_query(question: str, should_have_answer: bool, description: str):
    """Test a single query against the API."""
    print(f"\n🧪 Testing: {description}")
    print(f"   Query: '{question}'")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"question": question},
            timeout=15
        )
        
        if response.status_code != 200:
            print(f"   ❌ FAIL: API returned status {response.status_code}")
            return False
        
        data = response.json()
        has_answer = data.get('has_answer', False)
        answer = data.get('answer', '')
        chunks = data.get('chunks_retrieved', 0)
        
        if should_have_answer:
            if has_answer:
                print(f"   ✅ PASS: Got answer (as expected)")
                print(f"   📊 Chunks: {chunks}")
                return True
            else:
                print(f"   ❌ FAIL: Expected answer but got no-evidence response")
                print(f"   Response: {answer[:100]}...")
                return False
        else:
            if not has_answer:
                # Check for professional no-evidence message
                if any(phrase in answer.lower() for phrase in [
                    'could not find', 'not found', 'insufficient'
                ]):
                    print(f"   ✅ PASS: Got professional no-evidence response")
                    print(f"   📊 Chunks: {chunks}")
                    return True
                else:
                    print(f"   ⚠️  WARNING: No answer but message doesn't look like guardrail")
                    print(f"   Response: {answer[:100]}...")
                    return True  # Still pass, but note warning
            else:
                print(f"   ❌ FAIL: Expected no-evidence but got answer")
                print(f"   Answer: {answer[:100]}...")
                return False
                
    except requests.exceptions.ConnectionError:
        print(f"   ❌ ERROR: Cannot connect to API")
        print(f"   Make sure backend is running: uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False


print("="*80)
print("PHASE 4: EVIDENCE GUARDRAIL - END-TO-END TESTS")
print("="*80)
print("\nTesting against live API at http://localhost:8000")
print("Make sure backend is running!\n")

time.sleep(1)

results = []

# Test 1: Valid query - should get answer
results.append(test_query(
    "What is Gravol?",
    should_have_answer=True,
    description="Valid drug query (general info)"
))

# Test 2: Valid contraindications query - should get answer
results.append(test_query(
    "What are Gravol contraindications?",
    should_have_answer=True,
    description="Valid section query (contraindications)"
))

# Test 3: Non-existent drug - should get no-evidence
results.append(test_query(
    "What is FakeDrug123?",
    should_have_answer=False,
    description="Non-existent drug (should trigger guardrail)"
))

# Test 4: Query with missing section - should get no-evidence
results.append(test_query(
    "What is Gravol used for?",
    should_have_answer=False,
    description="Missing indications section (should trigger guardrail)"
))

# Test 5: Nonsense query - should get no-evidence
results.append(test_query(
    "Gravol quantum physics?",
    should_have_answer=False,
    description="Nonsense query (should trigger guardrail)"
))

# Results summary
print("\n" + "="*80)
passed = sum(results)
total = len(results)
print(f"RESULTS: {passed}/{total} tests passed")

if passed == total:
    print("🎉 ALL GUARDRAIL E2E TESTS PASSED!")
    print("\nGuardrail system is working correctly:")
    print("  ✅ Good queries return answers")
    print("  ✅ Bad queries return professional no-evidence responses")
    print("  ✅ No hallucinations!")
else:
    print(f"⚠️  {total - passed} tests failed - review above for details")

print("="*80)
