"""
Utility functions for SHA-256 hashing.

Used for:
- Document deduplication (hash PDF bytes)
- Image classification caching (hash image bytes)
- Idempotent ingestion
"""
import hashlib
from pathlib import Path
from typing import Union


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Args:
        file_path: Path to file to hash
        
    Returns:
        Hex string of SHA-256 hash (64 characters)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Example:
        hash = compute_file_hash("drug_monograph.pdf")
        # Returns: "a1b2c3d4..."
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sha256 = hashlib.sha256()
    
    # Read in chunks for memory efficiency with large files
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """
    Compute SHA-256 hash of bytes.
    
    Args:
        data: Raw bytes to hash
        
    Returns:
        Hex string of SHA-256 hash (64 characters)
        
    Example:
        hash = compute_bytes_hash(image_bytes)
    """
    return hashlib.sha256(data).hexdigest()


def compute_string_hash(text: str) -> str:
    """
    Compute SHA-256 hash of a string.
    
    Args:
        text: String to hash (UTF-8 encoded)
        
    Returns:
        Hex string of SHA-256 hash (64 characters)
        
    Example:
        hash = compute_string_hash("unique identifier")
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def short_hash(full_hash: str, length: int = 8) -> str:
    """
    Return shortened version of hash for display.
    
    Args:
        full_hash: Full 64-character hash
        length: Number of characters to return
        
    Returns:
        First `length` characters of hash
        
    Example:
        short = short_hash("a1b2c3d4e5f6...", 8)
        # Returns: "a1b2c3d4"
    """
    return full_hash[:length]
