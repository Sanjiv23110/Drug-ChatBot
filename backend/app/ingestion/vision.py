"""
Chemical structure image detection using GPT-4o Vision.

Uses vision model to classify extracted images as:
- Chemical structure diagrams → TRUE
- Logos, icons, decorative graphics → FALSE

Includes SHA-256 caching to avoid repeated API calls.
"""
import os
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple

from openai import AzureOpenAI
from PIL import Image

from app.utils.hashing import compute_file_hash, compute_bytes_hash

logger = logging.getLogger(__name__)


class VisionClassifier:
    """
    GPT-4o Vision classifier for chemical structure detection.
    
    Features:
    - Binary classification (is/isn't chemical structure)
    - SHA-256 caching to avoid duplicate API calls
    - Graceful error handling
    """
    
    # Classification prompt
    CLASSIFICATION_PROMPT = """You are an expert chemist and image classifier.

Look at this image and determine if it is a chemical structure diagram.

A chemical structure diagram shows:
- Molecular bonds (lines connecting atoms)
- Atom labels (C, H, O, N, etc.)
- Ring structures (benzene rings, etc.)
- Stereochemistry notation

NOT chemical structures:
- Logos or brand images
- Graphs or charts
- Tables
- Photos of pills/medications
- Decorative icons

Answer with ONLY one word: TRUE or FALSE

Is this image a chemical structure diagram?"""

    def __init__(
        self,
        model: str = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the vision classifier.
        
        Args:
            model: Vision model to use (default: gpt-4o from env)
            cache_enabled: Whether to use hash-based caching
        """
        self.model = model or os.getenv("VISION_MODEL", "gpt-4o")
        self.cache_enabled = cache_enabled
        
        # In-memory cache (will be replaced with DB cache)
        self._cache: dict[str, bool] = {}
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        
        logger.info(f"VisionClassifier initialized with model: {self.model}")
    
    def classify_image(
        self,
        image_path: str,
        use_cache: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Classify an image as chemical structure or not.
        
        Args:
            image_path: Path to image file
            use_cache: Whether to check cache first
            
        Returns:
            Tuple of (is_chemical_structure: bool, image_hash: str)
            
        Raises:
            FileNotFoundError: If image doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Compute hash
        image_hash = compute_file_hash(image_path)
        
        # Check cache
        if use_cache and self.cache_enabled and image_hash in self._cache:
            logger.debug(f"Cache hit for image {path.name}")
            return self._cache[image_hash], image_hash
        
        # Load and encode image
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Convert to base64
            base64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
            
            # Determine mime type
            mime_type = self._get_mime_type(path.suffix)
            
            # Call vision model
            is_structure = self._call_vision_api(base64_image, mime_type)
            
            # Cache result
            if self.cache_enabled:
                self._cache[image_hash] = is_structure
            
            logger.info(
                f"Classified {path.name}: "
                f"{'CHEMICAL STRUCTURE' if is_structure else 'NOT structure'}"
            )
            
            return is_structure, image_hash
            
        except Exception as e:
            logger.error(f"Vision classification failed for {path.name}: {e}")
            return False, image_hash
    
    def _call_vision_api(self, base64_image: str, mime_type: str) -> bool:
        """
        Call the GPT-4o Vision API.
        
        Args:
            base64_image: Base64-encoded image
            mime_type: Image MIME type
            
        Returns:
            True if chemical structure, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.CLASSIFICATION_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "low"  # Use low detail to save tokens
                                }
                            }
                        ]
                    }
                ],
                max_tokens=10,  # Only need TRUE or FALSE
                temperature=0  # Deterministic
            )
            
            answer = response.choices[0].message.content.strip().upper()
            
            return answer == "TRUE"
            
        except Exception as e:
            logger.error(f"Vision API call failed: {e}")
            return False
    
    def _get_mime_type(self, suffix: str) -> str:
        """Get MIME type from file extension."""
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_types.get(suffix.lower(), "image/png")
    
    def classify_batch(
        self,
        image_paths: list[str],
        continue_on_error: bool = True
    ) -> list[Tuple[str, bool, str]]:
        """
        Classify multiple images.
        
        Args:
            image_paths: List of image file paths
            continue_on_error: Continue if one image fails
            
        Returns:
            List of (path, is_structure, hash) tuples
        """
        results = []
        
        for path in image_paths:
            try:
                is_structure, image_hash = self.classify_image(path)
                results.append((path, is_structure, image_hash))
            except Exception as e:
                if continue_on_error:
                    logger.warning(f"Skipping {path}: {e}")
                    results.append((path, False, ""))
                else:
                    raise
        
        structure_count = sum(1 for _, is_struct, _ in results if is_struct)
        logger.info(
            f"Batch classification complete: "
            f"{structure_count}/{len(results)} are chemical structures"
        )
        
        return results
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "structures_cached": sum(1 for v in self._cache.values() if v),
            "non_structures_cached": sum(1 for v in self._cache.values() if not v),
        }


def is_chemical_structure(image_path: str) -> bool:
    """
    Convenience function to check if an image is a chemical structure.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if chemical structure, False otherwise
    """
    classifier = VisionClassifier()
    result, _ = classifier.classify_image(image_path)
    return result
