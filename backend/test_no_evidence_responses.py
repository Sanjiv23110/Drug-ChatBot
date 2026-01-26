"""
Test no-evidence response templates - Phase 2
"""
from app.guardrails import get_no_evidence_response

print("="*70)
print("PHASE 2: No-Evidence Response Tests")
print("="*70)

# Test 1: Default response
print("\n✔ Test 1: Default response")
response = get_no_evidence_response("random query")
assert "could not find sufficient information" in response
assert "consult a healthcare professional" in response
print("✅ PASS: Default response contains expected text")

# Test 2: Drug not found
print("\n✔ Test 2: Drug not found response")
response = get_no_evidence_response(
    "what is xyz", 
    drug_name="XYZ", 
    reason='drug_not_found'
)
assert "XYZ" in response
assert "Verify the drug name" in response
print("✅ PASS: Drug not found response is contextual")

# Test 3: Section not found
print("\n✔ Test 3: Section not found response")
response = get_no_evidence_response(
    "gravol indications",
    drug_name="Gravol",
    section_type="indications",
    reason='section_not_found'
)
assert "Gravol" in response
assert "Indications" in response
assert "section may not be present" in response
print("✅ PASS: Section not found response is detailed")

# Test 4: Low confidence
print("\n✔ Test 4: Low confidence response")
response = get_no_evidence_response(
    "some query",
    reason='low_confidence'
)
assert "match confidence is low" in response
assert "Try rephrasing" in response
print("✅ PASS: Low confidence response is helpful")

print("\n" + "="*70)
print("🎉 ALL PHASE 2 TESTS PASSED!")
print("="*70)
print("\nAll response templates are professional and contextual!")
