"""
docling PDF parser wrapper.

Provides clean interface to IBM's docling library for:
- Converting PDFs to layout-aware Markdown
- Extracting images from PDFs
- Handling corrupt/malformed PDFs gracefully
"""
import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

from app.utils.hashing import compute_file_hash

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Represents an extracted image from a PDF."""
    image_path: str
    page_number: int
    image_bytes: Optional[bytes] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class ParsedDocument:
    """Result of parsing a PDF with docling."""
    # Source info
    file_path: str
    file_name: str
    document_hash: str
    
    # Content
    markdown_content: str
    raw_text: str
    
    # Extracted images
    images: List[ExtractedImage] = field(default_factory=list)
    
    # Metadata
    page_count: int = 0
    parse_success: bool = True
    error_message: Optional[str] = None


class DoclingParser:
    """
    Wrapper around IBM docling for PDF parsing.
    
    Features:
    - Layout-aware Markdown conversion
    - Image extraction
    - Error handling for corrupt PDFs
    - Hash computation for idempotency
    """
    
    def __init__(
        self,
        image_output_dir: str = "./data/images",
        extract_images: bool = True,
        ocr_enabled: bool = True
    ):
        """
        Initialize the docling parser.
        
        Args:
            image_output_dir: Directory to save extracted images
            extract_images: Whether to extract images from PDFs
            ocr_enabled: Enable OCR for scanned PDFs
        """
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        
        # Configure docling pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr_enabled
        pipeline_options.images_scale = 2.0  # Higher quality images
        pipeline_options.generate_picture_images = extract_images
        
        # Initialize converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        logger.info(f"DoclingParser initialized (OCR: {ocr_enabled}, Images: {extract_images})")
    
    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file and extract content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument with markdown, images, and metadata
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
        """
        path = Path(pdf_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Compute hash first (for idempotency check)
        document_hash = compute_file_hash(pdf_path)
        
        logger.info(f"Parsing PDF: {path.name} (hash: {document_hash[:8]}...)")
        
        try:
            # Convert with docling
            result = self.converter.convert(str(path))
            
            # Export to markdown
            markdown_content = result.document.export_to_markdown()
            
            # Get plain text
            raw_text = result.document.export_to_text() if hasattr(result.document, 'export_to_text') else markdown_content
            
            # Extract images
            images = []
            if self.extract_images:
                images = self._extract_images(result, path.stem, document_hash)
            
            # Get page count
            page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 0
            
            logger.info(
                f"Successfully parsed {path.name}: "
                f"{len(markdown_content)} chars, {len(images)} images, {page_count} pages"
            )
            
            return ParsedDocument(
                file_path=str(path.absolute()),
                file_name=path.name,
                document_hash=document_hash,
                markdown_content=markdown_content,
                raw_text=raw_text,
                images=images,
                page_count=page_count,
                parse_success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {path.name}: {e}")
            
            return ParsedDocument(
                file_path=str(path.absolute()),
                file_name=path.name,
                document_hash=document_hash,
                markdown_content="",
                raw_text="",
                images=[],
                page_count=0,
                parse_success=False,
                error_message=str(e)
            )
    
    def _extract_images(
        self,
        result,
        doc_name: str,
        doc_hash: str
    ) -> List[ExtractedImage]:
        """
        Extract images from docling result and save to disk.
        
        Args:
            result: docling conversion result
            doc_name: Document name for file naming
            doc_hash: Document hash for unique naming
            
        Returns:
            List of ExtractedImage objects
        """
        images = []
        
        try:
            # Access pictures from docling result
            if not hasattr(result.document, 'pictures'):
                return images
            
            for idx, picture in enumerate(result.document.pictures):
                try:
                    # Get image data
                    if hasattr(picture, 'image') and picture.image is not None:
                        image_data = picture.image
                        
                        # Generate unique filename
                        image_filename = f"{doc_name}_{doc_hash[:8]}_img{idx:03d}.png"
                        image_path = self.image_output_dir / image_filename
                        
                        # Save image
                        if hasattr(image_data, 'save'):
                            image_data.save(str(image_path))
                        elif isinstance(image_data, bytes):
                            with open(image_path, 'wb') as f:
                                f.write(image_data)
                        
                        # Get dimensions
                        width, height = None, None
                        if hasattr(image_data, 'size'):
                            width, height = image_data.size
                        
                        # Get page number
                        page_num = getattr(picture, 'page_no', 0)
                        
                        images.append(ExtractedImage(
                            image_path=str(image_path),
                            page_number=page_num,
                            width=width,
                            height=height
                        ))
                        
                        logger.debug(f"Extracted image: {image_filename}")
                        
                except Exception as img_error:
                    logger.warning(f"Failed to extract image {idx}: {img_error}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
        
        return images
    
    def parse_batch(
        self,
        pdf_paths: List[str],
        continue_on_error: bool = True
    ) -> List[ParsedDocument]:
        """
        Parse multiple PDFs.
        
        Args:
            pdf_paths: List of paths to PDF files
            continue_on_error: Continue processing if one PDF fails
            
        Returns:
            List of ParsedDocument results
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.parse(pdf_path)
                results.append(result)
            except Exception as e:
                if continue_on_error:
                    logger.error(f"Skipping {pdf_path}: {e}")
                    results.append(ParsedDocument(
                        file_path=pdf_path,
                        file_name=Path(pdf_path).name,
                        document_hash="",
                        markdown_content="",
                        raw_text="",
                        parse_success=False,
                        error_message=str(e)
                    ))
                else:
                    raise
        
        success_count = sum(1 for r in results if r.parse_success)
        logger.info(f"Batch parsing complete: {success_count}/{len(results)} successful")
        
        return results


def parse_directory(
    pdf_dir: str,
    image_output_dir: str = "./data/images"
) -> List[ParsedDocument]:
    """
    Convenience function to parse all PDFs in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        image_output_dir: Where to save extracted images
        
    Returns:
        List of ParsedDocument results
    """
    parser = DoclingParser(image_output_dir=image_output_dir)
    
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    
    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")
    
    return parser.parse_batch([str(p) for p in pdf_files])
