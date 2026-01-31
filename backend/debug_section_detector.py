"""Debug script to test APO-METOPROLOL case"""

from app.ingestion.section_detector import SectionDetector

# Simulate APO-METOPROLOL blocks
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
    # List items
    {"text": "• Sinus bradycardia", "font_weight": 400},
    {"text": "• Sick sinus syndrome", "font_weight": 400},
    {"text": "• Second and third degree A-V block", "font_weight": 400},
]

detector = SectionDetector(use_llm_fallback=False)

print("=== TESTING HEADER CANDIDATE DETECTION ===\n")
candidates = detector.detect_header_candidates(blocks, page_median_font_weight=400)

print(f"Total candidates detected: {len(candidates)}\n")

for i, candidate in enumerate(candidates):
    print(f"Candidate {i+1}:")
    print(f"  Block ID: {candidate.block_id}")
    print(f"  Text: '{candidate.text}'")
    print(f"  Normalized: '{candidate.normalized_text}'")
    print(f"  Confidence: {candidate.confidence:.2f}")
    print(f"  ALL CAPS: {candidate.is_all_caps}")
    print(f"  Title Case: {candidate.is_title_case}")
    print(f"  Vertical WS: {candidate.has_vertical_whitespace}")
    print()

print("\n=== EXPECTED ===")
print("Should detect ONLY block 0 ('2 CONTRAINDICATIONS') as header")
print("Block 3 ('APO-METOPROLOL is contraindicated...') should NOT be detected")

print("\n=== RESULT ===")
if len(candidates) == 1 and candidates[0].block_id == 0:
    print("✅ PASS: Only the true header was detected")
else:
    print(f"❌ FAIL: Expected 1 candidate at block 0, got {len(candidates)} candidates")
    if len(candidates) > 1:
        print(f"  Extra candidates: {[c.block_id for c in candidates[1:]]}")
