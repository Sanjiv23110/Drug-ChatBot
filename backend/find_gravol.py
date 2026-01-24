import sqlite3

conn = sqlite3.connect('data/metadata.db')

# Find Gravol/dimenhydrinate file
print("Finding Gravol PDF...")
cursor = conn.execute("""
    SELECT DISTINCT file_path 
    FROM chunks 
    WHERE chunk_text LIKE '%gravol%' OR chunk_text LIKE '%dimenhydrinate%'
    LIMIT 1
""")
result = cursor.fetchone()
if result:
    pdf_path = result[0]
    print(f"Found: {pdf_path}")
    
    # Get all sections in this PDF
    print(f"\nSections in {pdf_path}:")
    cursor = conn.execute("""
        SELECT DISTINCT section_name 
        FROM chunks 
        WHERE file_path = ?
        ORDER BY section_name
    """, (pdf_path,))
    sections = cursor.fetchall()
    for s in sections:
        print(f"  - {s[0]}")
    
    # Check for ANY chunks with "indication" or "use"
    print(f"\nChecking for indication/use content...")
    cursor = conn.execute("""
        SELECT COUNT(*), MIN(chunk_text)
        FROM chunks 
        WHERE file_path = ?
        AND (chunk_text LIKE '%indication%' OR chunk_text LIKE '%use%')
    """, (pdf_path,))
    count, sample = cursor.fetchone()
    print(f"  Chunks mentioning indication/use: {count}")
    if sample:
        print(f"  Sample: {sample[:200]}...")
else:
    print("NOT FOUND! This is the problem!")
    
conn.close()
