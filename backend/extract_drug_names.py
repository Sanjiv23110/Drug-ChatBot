"""
Quick script to extract real drug names from the database
"""
import sqlite3
from pathlib import Path
import re

db_path = "data/metadata.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get first chunk from each file to extract drug name
cursor.execute("""
    SELECT DISTINCT file_path,
           (SELECT chunk_text FROM chunks c2 
            WHERE c2.file_path = chunks.file_path 
            LIMIT 1) as first_chunk
    FROM chunks
    LIMIT 50
""")

results = cursor.fetchall()
conn.close()

print(f"Found {len(results)} PDF files\n")
print("="*70)

drug_names = {}
for file_path, first_chunk in results:
    filename = Path(file_path).name
    
    # Try to extract drug name from first chunk
    drug_name = None
    
    if first_chunk:
        # Look for common patterns
        lines = first_chunk.split('\n')
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            # Pattern: "Pr<DRUGNAME" or just "<DRUGNAME" at start
            if line.strip().startswith('Pr') and len(line.strip()) > 2:
                drug_name = line.strip().replace('Pr', '').strip()
                if len(drug_name) > 3 and len(drug_name) < 50:
                    break
            
            # Pattern: Look for "Product Monograph" followed by drug name
            if 'PRODUCT MONOGRAPH' in line.upper() and i < len(lines) - 1:
                next_line = lines[i+1].strip()
                if len(next_line) > 3 and len(next_line) < 50:
                    drug_name = next_line
                    break
            
            # Pattern: All caps words that might be drug names
            if line.strip().isupper() and 3 < len(line.strip()) < 50:
                # Check if it looks like a drug name (not just section headers)
                if not any(keyword in line for keyword in ['PAGE', 'TABLE', 'PART', 'SECTION']):
                    drug_name = line.strip()
                    break
    
    if not drug_name:
        drug_name = filename  # Fallback to filename
    
    drug_names[filename] = drug_name
    print(f"{filename[:15]:.<25} → {drug_name}")

print("="*70)
print(f"\nTotal drugs found: {len(drug_names)}")
