"""
Enhanced Docling parser that extracts layout blocks for section detection.

This extends the existing DoclingParser to provide:
1. Raw text blocks with layout metadata (font size, weight, position)
2. Backward compatibility with existing markdown-based flow
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LayoutBlock:
    """
    Represents a text block with layout metadata from Docling.
    
    This is used by SectionDetector for layout-aware header detection.
    """
    block_id: int
    text: str
    font_size: float = 12.0
    font_weight: int = 400  # 400=normal, 700=bold
    page_number: int = 0
    bbox: Optional[Dict[str, float]] = None  # Bounding box {x, y, width, height}


def extract_layout_blocks_from_docling(docling_result) -> List[Dict]:
    """
    Extract layout blocks from Docling's raw document structure.
    
    Args:
        docling_result: The result object from DocumentConverter.convert()
    
    Returns:
        List of block dictionaries compatible with SectionDetector
    
    Example block:
        {
            "text": "CONTRAINDICATIONS",
            "font_size": 14.0,
            "font_weight": 700,
            "page_number": 2
        }
    """
    blocks = []
    
    try:
        # Access Docling's document structure
        if not hasattr(docling_result, 'document'):
            logger.warning("Docling result has no document attribute")
            return blocks
        
        doc = docling_result.document
        
        # Iterate through pages
        if hasattr(doc, 'pages'):
            for page_idx, page in enumerate(doc.pages):
                # Extract text items/cells from page
                if hasattr(page, 'cells'):
                    for cell_idx, cell in enumerate(page.cells):
                        text = getattr(cell, 'text', '').strip()
                        if not text:
                            continue
                        
                        # Extract font metadata
                        font_size = 12.0
                        font_weight = 400
                        
                        # Try to get font info from cell properties
                        if hasattr(cell, 'font'):
                            font_size = getattr(cell.font, 'size', 12.0)
                            font_weight = getattr(cell.font, 'weight', 400)
                        elif hasattr(cell, 'properties'):
                            props = cell.properties
                            font_size = props.get('font_size', 12.0)
                            font_weight = props.get('font_weight', 400)
                        
                        # Get bounding box if available
                        bbox = None
                        if hasattr(cell, 'bbox'):
                            bbox = {
                                'x': cell.bbox.x,
                                'y': cell.bbox.y,
                                'width': cell.bbox.width,
                                'height': cell.bbox.height
                            }
                        
                        block = {
                            "text": text,
                            "font_size": font_size,
                            "font_weight": font_weight,
                            "page_number": page_idx,
                            "bbox": bbox
                        }
                        
                        blocks.append(block)
                
                # Alternative: Try text elements
                elif hasattr(page, 'elements'):
                    for elem in page.elements:
                        if hasattr(elem, 'text'):
                            text = elem.text.strip()
                            if text:
                                blocks.append({
                                    "text": text,
                                    "font_size": getattr(elem, 'font_size', 12.0),
                                    "font_weight": getattr(elem, 'font_weight', 400),
                                    "page_number": page_idx
                                })
        
        logger.info(f"Extracted {len(blocks)} layout blocks from Docling")
        
    except Exception as e:
        logger.error(f"Failed to extract layout blocks: {e}")
        logger.warning("Falling back to markdown-based chunking")
    
    return blocks


def fallback_blocks_from_markdown(markdown_content: str) -> List[Dict]:
    """
    Fallback: Create pseudo-blocks from markdown headers.
    
    Used when Docling doesn't provide layout metadata.
    Simulates layout signals based on markdown syntax.
    
    Args:
        markdown_content: Markdown text from Docling
    
    Returns:
        List of block dictionaries
    """
    import re
    
    blocks = []
    lines = markdown_content.split('\n')
    
    for line in lines:
        text = line.strip()
        if not text:
            continue
        
        # Detect markdown headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', text)
        
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2)
            
            # Simulate font properties based on header level
            font_size = 18 - (level * 2)  # # = 16pt, ## = 14pt, etc.
            font_weight = 700  # Headers are bold
            
            blocks.append({
                "text": header_text,
                "font_size": font_size,
                "font_weight": font_weight
            })
        else:
            # Regular text
            blocks.append({
                "text": text,
                "font_size": 12.0,
                "font_weight": 400
            })
    
    logger.info(f"Created {len(blocks)} fallback blocks from markdown")
    return blocks
