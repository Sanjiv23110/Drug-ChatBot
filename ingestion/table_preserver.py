"""
Table Preserver - XSLT-based table structure preservation
Uses FDA's official spl.xsl stylesheet to convert tables to Markdown
"""

from lxml import etree
from typing import Tuple
import logging
import os

logger = logging.getLogger(__name__)


class TablePreserver:
    """
    Preserves table structure using XSLT transformation
    """
    
    def __init__(self, xsl_path: str):
        """
        Initialize with path to spl.xsl stylesheet
        
        Args:
            xsl_path: Absolute path to spl.xsl file
        """
        if not os.path.exists(xsl_path):
            raise FileNotFoundError(f"XSLT stylesheet not found: {xsl_path}")
        
        try:
            # Load XSLT stylesheet
            xsl_doc = etree.parse(xsl_path)
            self.transform = etree.XSLT(xsl_doc)
            logger.info(f"Loaded FDA XSLT stylesheet: {xsl_path}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to load FDA XSLT stylesheet: {e}")
    
    def preserve_table_structure(self, table_element) -> Tuple[str, bool]:
        """
        Transform table element to Markdown using XSLT
        
        Args:
            table_element: lxml Element containing table
            
        Returns:
            (markdown_text, is_table_verified)
        """
        try:
            # Apply XSLT transformation
            result = self.transform(table_element)
            markdown_text = str(result)
            
            # Verify it's actually a table (contains pipe characters)
            is_table = '|' in markdown_text or 'Table' in markdown_text
            
            return markdown_text, is_table
            
        except Exception as e:
            logger.warning(f"XSLT transformation failed: {e}")
            # Fallback to basic text extraction
            text = ''.join(table_element.itertext())
            return text, False
