"""
Simple test to verify Section Detector integration works.

This script tests the validation method independently before integrating into ingest.py.
"""

from app.ingestion.section_detector import SectionDetector
from app.ingestion.layout_extractor import fallback_blocks_from_markdown

# Sample markdown from a pharmaceutical PDF
sample_markdown = """
# APO-METOPROLOL

## 2 CONTRAINDICATIONS

Patients who are hypersensitive to this drug or to any ingredient in the formulation.

**APO-METOPROLOL is contraindicated in patients with:**

• Sinus bradycardia
• Sick sinus syndrome
• Second and third degree A-V block

## 3 WARNINGS

Use with caution in patients with...
"""

print("=== Testing Section Detector Integration ===\n")

# Step 1: Create pseudo-blocks from markdown
print("Step 1: Extracting blocks from markdown...")
blocks = fallback_blocks_from_markdown(sample_markdown)
print(f"  Extracted {len(blocks)} blocks\n")

# Step 2: Run section detector
print("Step 2: Running SectionDetector...")
detector = SectionDetector(use_llm_fallback=False)
sections = detector.detect_sections(blocks)
print(f"  Detected {len(sections)} sections\n")

# Step 3: Analyze results
print("Step 3: Analysis\n")

for i, section in enumerate(sections, 1):
    print(f"Section {i}:")
    print(f"  Category: {section.category.value}")
    print(f"  Header: '{section.original_header}'")
    print(f"  Blocks: {section.start_block_id} - {section.end_block_id}")
    print(f"  Confidence: {section.confidence:.2f}")
    print(f"  Method: {section.detection_method}")
    print()

# Step 4: Verify APO-METOPROLOL case
print("Step 4: Verification\n")

contraindications_section = [s for s in sections if s.category.value == "contraindications"]
if contraindications_section:
    section = contraindications_section[0]
    block_count = section.end_block_id - section.start_block_id
    print(f"✅ CONTRAINDICATIONS section found")
    print(f"   Spans {block_count} blocks (should include bullet list)")
    print(f"   Confidence: {section.confidence:.2f}")
else:
    print("❌ CONTRAINDICATIONS section NOT found")

print("\n=== Test Complete ===")
