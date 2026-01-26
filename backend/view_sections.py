import sqlite3

conn = sqlite3.connect('c:/G/Maclens chatbot w api/backend/data/metadata.db')
cursor = conn.execute('SELECT DISTINCT section_name FROM chunks WHERE section_name IS NOT NULL ORDER BY section_name')
sections = [row[0] for row in cursor.fetchall()]

print("="*60)
print("ALL SECTION NAMES IN DATABASE:")
print("="*60)
for i, section in enumerate(sections, 1):
    print(f"{i}. {section}")
print("="*60)
print(f"Total: {len(sections)} distinct sections")
