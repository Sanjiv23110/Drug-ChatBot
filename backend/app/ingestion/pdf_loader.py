"""
PDF loader with file fingerprinting.

Computes SHA256 hash for change detection and streams pages.
"""
import hashlib
from pathlib import Path
from typing import Generator, Dict, Tuple
import pypdf


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of PDF file.
    
    Used for change detection - same hash = same content.
    """
    sha256 = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def load_pdf(file_path: str) -> Tuple[str, Generator[Dict, None, None]]:
    """
    Load PDF with file fingerprinting.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        (file_hash, page_generator)
        
    Yields:
        Dictionary with:
        - page_num: 0-indexed page number
        - text: Extracted text
        - char_start: Character offset in full document
        - char_end: End offset
        
    Raises:
        FileNotFoundError: If PDF doesn't exist
        pypdf.errors.PdfReadError: If PDF is corrupted
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    # Compute hash first
    file_hash = compute_file_hash(str(file_path))
    
    def page_generator():
        """Generator that yields pages with metadata."""
        try:
            reader = pypdf.PdfReader(str(file_path))
            char_offset = 0
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    
                    if not text.strip():
                        # Skip empty pages
                        continue
                    
                    yield {
                        'page_num': page_num,
                        'text': text,
                        'char_start': char_offset,
                        'char_end': char_offset + len(text)
                    }
                    
                    char_offset += len(text)
                
                except Exception as e:
                    # Log but continue with other pages
                    print(f"Warning: Failed to extract page {page_num} from {file_path}: {e}")
                    continue
        
        except Exception as e:
            raise pypdf.errors.PdfReadError(f"Failed to read PDF {file_path}: {e}")
    
    return file_hash, page_generator()


def get_pdf_metadata(file_path: str) -> Dict:
    """
    Get PDF metadata without reading full content.
    
    Returns:
        Dictionary with file_path, file_hash, page_count, file_size
    """
    file_path = Path(file_path)
    
    file_hash = compute_file_hash(str(file_path))
    
    try:
        reader = pypdf.PdfReader(str(file_path))
        page_count = len(reader.pages)
    except:
        page_count = 0
    
    return {
        'file_path': str(file_path),
        'file_hash': file_hash,
        'page_count': page_count,
        'file_size': file_path.stat().st_size
    }
