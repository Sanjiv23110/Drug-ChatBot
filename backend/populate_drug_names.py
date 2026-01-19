"""
Quick script to populate drug name mappings for brand → generic resolution.

Run this once to enable the DrugNameResolver.
"""
import sqlite3

# Common Health Canada brand → generic mappings
mappings = [
    # The one you just discovered
    ("GRAVOL", "dimenhydrinate"),
    
    # Other common ones from your PDFs
    ("CeeNU", "lomustine"),
    ("TEVA-CHLORPROMAZINE", "chlorpromazine"),
    ("CORTROSYN", "cosyntropin"),
    ("MIOSTAT", "carbachol"),
    ("DELATESTRYL", "testosterone"),
    
    # Add more as you discover them
    # ("Brand Name", "generic_name"),
]

def populate_mappings():
    conn = sqlite3.connect("data/metadata.db")
    
    # Create table if not exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS drug_names (
            id INTEGER PRIMARY KEY,
            brand_name TEXT UNIQUE,
            generic_name TEXT
        )
    ''')
    
    # Insert mappings
    added = 0
    for brand, generic in mappings:
        try:
            conn.execute(
                "INSERT INTO drug_names (brand_name, generic_name) VALUES (?, ?)",
                (brand, generic)
            )
            added += 1
            print(f"✓ Added: {brand} → {generic}")
        except sqlite3.IntegrityError:
            print(f"⏭ Skipped: {brand} (already exists)")
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Successfully added {added} new mappings")
    print(f"Total mappings in database: {len(mappings)}")

if __name__ == "__main__":
    populate_mappings()
