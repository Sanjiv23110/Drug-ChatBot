"""
Verify SectionClassifier fix
"""
import sys
sys.path.insert(0, 'c:/G/solomindUS')
from orchestrator.qa_orchestrator import SectionClassifier

print("Testing SectionClassifier 'overdosage' vs 'dosage' priority...")

classifier = SectionClassifier()

# Case 1: "overdosage" query
query1 = "how to treat overdosage of dantrium?"
loinc1 = classifier.classify(query1)
print(f"Query: '{query1}' -> LOINC: {loinc1}")

expected_overdosage = "34088-5"
if loinc1 == expected_overdosage:
    print("✓ SUCCESS: Correctly mapped to OVERDOSAGE")
else:
    print(f"✗ FAIL: Mapped to {loinc1} (Likely DOSAGE 34068-7)")

# Case 2: "dosage" query
query2 = "what is the dosage of lisinopril?"
loinc2 = classifier.classify(query2)
print(f"Query: '{query2}' -> LOINC: {loinc2}")

expected_dosage = "34068-7"
if loinc2 == expected_dosage:
    print("✓ SUCCESS: Correctly mapped to DOSAGE")
else:
    print(f"✗ FAIL: Mapped to {loinc2}")

# Case 3: "drug interactions" vs "interactions" (priority check)
query3 = "are there drug interactions?"
loinc3 = classifier.classify(query3)
print(f"Query: '{query3}' -> LOINC: {loinc3}")
if loinc3 == "34073-7":
    print("✓ SUCCESS: Correctly mapped to INTERACTIONS")

