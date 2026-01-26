import sqlite3
import json

conn = sqlite3.connect('c:/G/Maclens chatbot w api/backend/data/metadata.db')
cursor = conn.execute('SELECT DISTINCT section_name FROM chunks WHERE section_name IS NOT NULL ORDER BY section_name')
sections = [row[0] for row in cursor.fetchall()]

print("="*80)
print(f"ALL {len(sections)} SECTION NAMES IN DATABASE:")
print("="*80)

# Group by base section (before ">")
grouped = {}
for section in sections:
    base = section.split(' > ')[0].strip()
    if base not in grouped:
        grouped[base] = []
    grouped[base].append(section)

# Print grouped
for base in sorted(grouped.keys()):
    print(f"\n{base}:")
    for variant in sorted(grouped[base]):
        if variant == base:
            print(f"  - {variant} [BASE]")
        else:
            print(f"  - {variant}")

print("\n" + "="*80)
print(f"Total base sections: {len(grouped)}")
print(f"Total variations: {len(sections)}")
print("="*80)

# Save to JSON for reference
with open('c:/G/Maclens chatbot w api/backend/section_mapping.json', 'w') as f:
    json.dump(grouped, f, indent=2)
print("\nSaved to section_mapping.json")
