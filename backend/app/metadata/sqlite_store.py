"""
SQLite metadata store for chunk information.

Deterministic chunk IDs ensure FAISS index alignment.
"""
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class SQLiteMetadataStore:
    """Manages chunk metadata in SQLite with deterministic IDs."""
    
    def __init__(self, db_path: str = "data/metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        with sqlite3.connect(self.db_path) as conn:
            with open(schema_path) as f:
                conn.executescript(f.read())
    
    @staticmethod
    def generate_chunk_id(file_path: str, page_num: int, char_start: int) -> int:
        """
        Generate deterministic chunk ID.
        
        Uses hash of (file_path, page_num, char_start) to ensure:
        - Same chunk always gets same ID
        - ID is stable across re-ingestion
        - Fits in SQLite INTEGER (signed 64-bit)
        
        Args:
            file_path: Normalized file path
            page_num: Page number (0-indexed)
            char_start: Character offset in page
            
        Returns:
            Deterministic chunk ID as positive integer
        """
        # Normalize path
        normalized = str(Path(file_path).resolve())
        
        # Create stable identifier
        identifier = f"{normalized}:{page_num}:{char_start}"
        
        # Hash to 31-bit positive integer (avoid SQLite INTEGER overflow)
        hash_bytes = hashlib.sha256(identifier.encode()).digest()
        chunk_id = int.from_bytes(hash_bytes[:4], 'big') & 0x7FFFFFFF
        
        return chunk_id
    
    def insert_chunk(
        self,
        chunk_id: int,
        file_path: str,
        chunk_text: str,
        page_num: Optional[int] = None,
        section_name: Optional[str] = None,
        char_start: int = 0,
        char_end: int = 0
    ) -> bool:
        """
        Insert chunk metadata.
        
        Returns:
            True if inserted, False if ID collision (should retry with different ID)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO chunks 
                    (chunk_id, file_path, page_num, section_name, chunk_text, char_start, char_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, file_path, page_num, section_name, chunk_text, char_start, char_end)
                )
                return True
        except sqlite3.IntegrityError:
            # Collision - caller should handle
            return False
    
    def insert_chunks_batch(self, chunks: List[Dict]) -> int:
        """
        Batch insert chunks.
        
        Args:
            chunks: List of chunk dicts with keys:
                   chunk_id, file_path, chunk_text, page_num, section_name, char_start, char_end
        
        Returns:
            Number of successfully inserted chunks
        """
        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            for chunk in chunks:
                try:
                    conn.execute(
                        """
                        INSERT INTO chunks 
                        (chunk_id, file_path, page_num, section_name, chunk_text, char_start, char_end)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk['chunk_id'],
                            chunk['file_path'],
                            chunk.get('page_num'),
                            chunk.get('section_name'),
                            chunk['chunk_text'],
                            chunk.get('char_start', 0),
                            chunk.get('char_end', 0)
                        )
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue  # Skip collisions
        return inserted
    
    def get_chunk(self, chunk_id: int) -> Optional[Dict]:
        """Retrieve chunk metadata by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?",
                (chunk_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_chunks_by_ids(self, chunk_ids: List[int]) -> List[Dict]:
        """Retrieve multiple chunks by IDs."""
        if not chunk_ids:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ','.join('?' * len(chunk_ids))
            cursor = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_chunks_by_file(self, file_path: str) -> List[Dict]:
        """Retrieve all chunks from a specific file."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM chunks WHERE file_path = ? ORDER BY page_num, char_start",
                (file_path,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def count_chunks(self) -> int:
        """Get total chunk count."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
    
    def delete_file_chunks(self, file_path: str) -> int:
        """Delete all chunks from a file. Returns number deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE file_path = ?",
                (file_path,)
            )
            return cursor.rowcount
    
    # File record methods for fingerprinting
    
    def get_file_record(self, file_path: str) -> Optional[Dict]:
        """Get file record with hash and version."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM files WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def upsert_file_record(
        self,
        file_path: str,
        file_hash: str,
        ingestion_version: int,
        chunk_count: int
    ):
        """Insert or update file record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO files (file_path, file_hash, ingestion_version, chunk_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    ingestion_version = excluded.ingestion_version,
                    chunk_count = excluded.chunk_count,
                    ingestion_date = CURRENT_TIMESTAMP
                """,
                (file_path, file_hash, ingestion_version, chunk_count)
            )
    
    def should_ingest_file(
        self,
        file_path: str,
        file_hash: str,
        current_version: int
    ) -> bool:
        """
        Check if file needs (re)ingestion.
        
        Returns True if:
        - File not in DB
        - File hash changed (content updated)
        - Ingestion version upgraded
        """
        record = self.get_file_record(file_path)
        
        if not record:
            return True  # New file
        
        if record['file_hash'] != file_hash:
            return True  # Content changed
        
        if record['ingestion_version'] < current_version:
            return True  # Schema/logic upgraded
        
        return False  # Already ingested, skip
