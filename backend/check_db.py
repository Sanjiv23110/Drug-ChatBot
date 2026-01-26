import sqlite3

# Connect to database
conn = sqlite3.connect('data/metadata.db')

print("=== First 3 chunks from 00010672 to identify drug ===")
cursor = conn.execute("""
    SELECT chunk_text
    FROM chunks 
    WHERE file_path LIKE '%00010672%'
    ORDER BY page_num, char_start
    LIMIT 3
""")

for i, row in enumerate(cursor.fetchall(), 1):
    print(f"\n--- Chunk {i} ---")
    text = row[0][:400]  # First 400 chars
    print(text)
    print("...")

conn.close()
