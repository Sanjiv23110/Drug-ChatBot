import sqlite3

conn = sqlite3.connect('data/metadata.db')
cursor = conn.execute("""
    SELECT chunk_text 
    FROM chunks 
    WHERE file_path LIKE '%00033792%' 
      AND section_name LIKE '%CONTRAINDIC%'
""")

chunk = cursor.fetchone()

if chunk:
    print("CHUNK TEXT FROM DATABASE:")
    print("="*60)
    print(chunk[0])
    print("="*60)
else:
    print("No CONTRAINDICATIONS chunk found in database!")

conn.close()
