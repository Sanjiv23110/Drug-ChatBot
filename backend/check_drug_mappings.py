import sqlite3

conn = sqlite3.connect('data/metadata.db')
cursor = conn.cursor()

# Check if drug_names table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables in database: {tables}")

if 'drug_names' in tables:
    cursor.execute("SELECT COUNT(*) FROM drug_names")
    count = cursor.fetchone()[0]
    print(f"\n✓ drug_names table exists with {count} mappings")
    
    cursor.execute("SELECT brand_name, generic_name FROM drug_names LIMIT 10")
    mappings = cursor.fetchall()
    print("\nSample mappings:")
    for brand, generic in mappings:
        print(f"  {brand} → {generic}")
        
    # Check specifically for Gravol
    cursor.execute("SELECT generic_name FROM drug_names WHERE UPPER(brand_name) = 'GRAVOL'")
    gravol = cursor.fetchone()
    if gravol:
        print(f"\n✓ Gravol mapping found: Gravol → {gravol[0]}")
    else:
        print("\n❌ Gravol mapping NOT found in database!")
else:
    print("\n❌ drug_names table does NOT exist!")
    print("   You need to run: python populate_drug_names.py")

conn.close()
