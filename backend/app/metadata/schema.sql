-- SQLite schema for chunk metadata storage
-- Design principles:
-- - Deterministic chunk IDs for stable FAISS mapping
-- - File-level and section-level indexing
-- - No embedding data stored here (FAISS only)
-- - File fingerprinting to detect PDF changes

CREATE TABLE IF NOT EXISTS files (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    ingestion_version INTEGER NOT NULL DEFAULT 1,
    ingestion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    page_num INTEGER,
    section_name TEXT,
    chunk_text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_path) REFERENCES files(file_path) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id);

-- For section-aware retrieval
CREATE INDEX IF NOT EXISTS idx_section ON chunks(section_name) WHERE section_name IS NOT NULL;

-- Drug name mapping for brand → generic resolution (Phase 4)
CREATE TABLE IF NOT EXISTS drug_names (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generic_name TEXT NOT NULL,
    brand_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_brand_name ON drug_names(brand_name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_generic_name ON drug_names(generic_name COLLATE NOCASE);

