"""
Test evidence validator - Phase 1
"""
from app.guardrails import has_sufficient_evidence

print("="*70)
print("PHASE 1: Evidence Validator Tests")
print("="*70)

# Test 1: No chunks
print("\n Test 1: No chunks (should reject)")
chunks = []
result = has_sufficient_evidence(chunks, "what is gravol")
assert result == False, "Should reject no chunks"
print("✅ PASS: No chunks rejected")

# Test 2: Good chunks with keyword match
print("\n✔ Test 2: Good chunks with keywords (should accept)")
chunks = [
    {'chunk_text': 'Gravol is used to treat nausea and vomiting', 'score': 0.85},
    {'chunk_text': 'Dimenhydrinate is the active ingredient in Gravol', 'score': 0.75}
]
result = has_sufficient_evidence(chunks, "what is gravol")
assert result == True, "Should accept good chunks"
print("✅ PASS: Good chunks accepted")

# Test 3: Chunks but no keyword match
print("\n✔ Test 3: Unrelated chunks (should reject)")
chunks = [
    {'chunk_text': 'Aspirin is a pain reliever', 'score': 0.45}
]
result = has_sufficient_evidence(chunks, "what is gravol")
assert result == False, "Should reject unrelated chunks"
print("✅ PASS: Unrelated chunks rejected")

# Test 4: Low score chunks
print("\n✔ Test 4: Low relevance score (should reject)")
chunks = [
    {'chunk_text': 'Gravol information here', 'score': 0.15}
]
result = has_sufficient_evidence(chunks, "what is gravol")
assert result == False, "Should reject low score chunks"
print("✅ PASS: Low score chunks rejected")

print("\n" + "="*70)
print("🎉 ALL PHASE 1 TESTS PASSED!")
print("="*70)
