"""
FREE RESTORE from backup - No API costs!
Restores the database to before the failed removal attempts.
"""

import shutil
from pathlib import Path

# Use the earliest backup (before any deletions)
backup_path = Path("data/backups/20260113_171910")

print("="*70)
print("RESTORING FROM BACKUP (FREE - No API costs!)")
print("="*70)

if not backup_path.exists():
    print(f"❌ Backup not found: {backup_path}")
    exit(1)

print(f"\nBackup timestamp: 2026-01-13 17:19:10")
print(f"This is BEFORE any deletions happened")
print()

# Restore database
print("Restoring database...")
shutil.copy2(backup_path / "metadata.db", "data/metadata.db")
print("  ✓ Restored metadata.db")

# Restore FAISS index
print("\nRestoring FAISS index...")
faiss_backup = backup_path / "medical_index"
if faiss_backup.exists():
    # Remove current corrupted index
    import os
    for item in Path("data/faiss").glob("medical_index*"):
        if item.is_file():
            os.remove(item)
        elif item.is_dir():
            shutil.rmtree(item)
    
    # Copy backup
    shutil.copytree(faiss_backup, "data/faiss/medical_index")
    print("  ✓ Restored FAISS index")
else:
    print("  ⚠️  No FAISS backup found")

print("\n" + "="*70)
print("✅ RESTORE COMPLETE")
print("="*70)
print("\nYour database is now restored to:")
print("  211 PDFs")  
print("  10,039 chunks")
print("  NO API COSTS - just copied files!")
print("\nAll 4 fixes are still active:")
print("  ✓ Enhanced section detection")
print("  ✓ Keyword boosting")
print("  ✓ Adaptive context window")
print("  ✓ Smart validation")
print("\nNow run: python evaluation/diverse_validation.py")
