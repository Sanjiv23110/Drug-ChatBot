"""
FAISS index persistence manager.

Handles atomic saves/loads with metadata tracking.
"""
import faiss
import json
import pickle
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import tempfile
import shutil


class IndexManager:
    """Manages FAISS index persistence with atomic writes."""
    
    def __init__(self, index_dir: str = "data/faiss"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.json"
        self.chunk_ids_path = self.index_dir / "chunk_ids.pkl"
    
    def save(self, faiss_store, metadata: Optional[Dict] = None):
        """
        Save FAISS index with atomic write.
        
        Args:
            faiss_store: FAISSVectorStore instance
            metadata: Optional metadata dict
        """
        # Create metadata
        meta = metadata or {}
        meta.update({
            "dimension": faiss_store.dimension,
            "count": faiss_store.count(),
            "last_modified": datetime.utcnow().isoformat(),
            "index_type": type(faiss_store.index).__name__
        })
        
        # Use temp files for atomic writes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp_index:
            temp_index_path = tmp_index.name
            faiss.write_index(faiss_store.index, temp_index_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_meta:
            temp_meta_path = tmp_meta.name
            json.dump(meta, tmp_meta, indent=2)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', mode='wb') as tmp_ids:
            temp_ids_path = tmp_ids.name
            pickle.dump(faiss_store.chunk_ids, tmp_ids)
        
        # Atomic moves
        shutil.move(temp_index_path, self.index_path)
        shutil.move(temp_meta_path, self.metadata_path)
        shutil.move(temp_ids_path, self.chunk_ids_path)
    
    def load(self, dimension: int = 1536) -> Optional[tuple]:
        """
        Load FAISS index and chunk IDs.
        
        Args:
            dimension: Expected dimension for validation
            
        Returns:
            (faiss.Index, chunk_ids, metadata) or None if no index exists
        """
        if not self.index_path.exists():
            return None
        
        # Load index
        index = faiss.read_index(str(self.index_path))
        
        # Load chunk IDs
        with open(self.chunk_ids_path, 'rb') as f:
            chunk_ids = pickle.load(f)
        
        # Load metadata
        with open(self.metadata_path) as f:
            metadata = json.load(f)
        
        # Validate dimension
        if metadata.get('dimension') != dimension:
            raise ValueError(
                f"Index dimension {metadata.get('dimension')} != expected {dimension}"
            )
        
        # Validate count
        if index.ntotal != len(chunk_ids):
            raise ValueError(
                f"Index count {index.ntotal} != chunk_ids count {len(chunk_ids)}"
            )
        
        return index, chunk_ids, metadata
    
    def exists(self) -> bool:
        """Check if index exists on disk."""
        return self.index_path.exists()
    
    def get_metadata(self) -> Optional[Dict]:
        """Load metadata without loading full index."""
        if not self.metadata_path.exists():
            return None
        
        with open(self.metadata_path) as f:
            return json.load(f)
    
    def delete(self):
        """Delete all index files."""
        for path in [self.index_path, self.metadata_path, self.chunk_ids_path]:
            if path.exists():
                path.unlink()
