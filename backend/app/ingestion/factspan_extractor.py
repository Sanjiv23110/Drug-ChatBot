"""
FactSpan Extractor - Atomic Fact Unit Extraction

Parses drug monograph sections into retrievable fact spans:
- Sentences
- Bullet points
- Table rows
- Captions/footnotes

CRITICAL: All text is preserved VERBATIM - no modification, no normalization.
"""
import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSpan:
    """
    Container for an extracted fact span before DB insertion.
    """
    text:str  # VERBATIM text
    text_type: str  # 'sentence' | 'bullet' | 'table_row' | 'caption'
    char_offset: int
    sequence_num: int


class FactSpanExtractor:
    """
    Extract atomic fact units from section text.
    
    DESIGN PRINCIPLES:
    - Preserve exact text - no cleaning, no normalization
    - Handle common PDF artifacts (hyphenation, line breaks)
    - Extract structured content (bullets, tables) as-is
    - Enable sub-section retrieval for precise answers
    
    Usage:
        extractor = FactSpanExtractor()
        spans = extractor.extract(section_text="...")
        for span in spans:
            # span.text is verbatim from PDF
            db_span = FactSpan(text=span.text, text_type=span.text_type, ...)
    """
    
    def __init__(self, enable_sentence_split: bool = True):
        """
        Initialize extractor.
        
        Args:
            enable_sentence_split: Whether to split into sentences (uses spacy if available)
        """
        self.enable_sentence_split = enable_sentence_split
        self._nlp = None
        
        # Try to load spacy for sentence splitting
        if enable_sentence_split:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                logger.info("Loaded spacy for sentence splitting")
            except:
                logger.warning("spacy not available, using regex-based sentence splitting")
                self._nlp = None
    
    def extract(self, section_text: str, section_id: Optional[int] = None) -> List[ExtractedSpan]:
        """
        Extract all fact spans from section text.
        
        Args:
            section_text: Verbatim section content from PDF
            section_id: Optional section ID for logging
            
        Returns:
            List of ExtractedSpan objects
        """
        all_spans = []
        
        # 1. Extract sentences
        sentences = self._extract_sentences(section_text)
        all_spans.extend(sentences)
        
        # 2. Extract bullets
        bullets = self._extract_bullets(section_text)
        all_spans.extend(bullets)
        
        # 3. Extract table rows
        table_rows = self._extract_table_rows(section_text)
        all_spans.extend(table_rows)
        
        # 4. Extract captions
        captions = self._extract_captions(section_text)
        all_spans.extend(captions)
        
        logger.info(
            f"Extracted {len(all_spans)} fact spans: "
            f"{len(sentences)} sentences, {len(bullets)} bullets, "
            f"{len(table_rows)} table rows, {len(captions)} captions"
        )
        
        return all_spans
    
    def _extract_sentences(self, text: str) -> List[ExtractedSpan]:
        """
        Split text into sentences using spacy or regex fallback.
        
        CRITICAL: Preserves exact text including whitespace and punctuation.
        
        Args:
            text: Section content
            
        Returns:
            List of sentence spans
        """
        if not self.enable_sentence_split:
            return []
        
        sentences = []
        
        if self._nlp:
            # Use spacy for accurate sentence splitting
            doc = self._nlp(text)
            for i, sent in enumerate(doc.sents):
                sentences.append(ExtractedSpan(
                    text=sent.text,  # VERBATIM
                    text_type='sentence',
                    char_offset=sent.start_char,
                    sequence_num=i
                ))
        else:
            # Fallback: regex-based sentence splitting
            # Patterns: . ! ? followed by space/newline and capital letter
            pattern = r'([^.!?]+[.!?]+)'
            matches = re.finditer(pattern, text)
            
            for i, match in enumerate(matches):
                sentence_text = match.group(1).strip()
                if len(sentence_text) > 10:  # Filter very short fragments
                    sentences.append(ExtractedSpan(
                        text=sentence_text,
                        text_type='sentence',
                        char_offset=match.start(),
                        sequence_num=i
                    ))
        
        return sentences
    
    def _extract_bullets(self, text: str) -> List[ExtractedSpan]:
        """
        Extract bullet points from text.
        
        Patterns recognized:
        - • Bullet text
        - - Dash bullets
        - * Asterisk bullets
        - ◦ Hollow bullets
        - 1. Numbered lists
        - a) Lettered lists
        
        CRITICAL: Preserves exact text including bullet character.
        
        Args:
            text: Section content
            
        Returns:
            List of bullet spans
        """
        bullets = []
        
        # Comprehensive bullet patterns
        # Matches: • - * ◦ or numbered (1. 2.) or lettered (a) b))
        bullet_pattern = r'^[\s]*([•\-\*◦]|[\d]+\.|[a-z]\))\s+(.+)$'
        
        lines = text.split('\n')
        bullet_sequence = 0
        
        for line_num, line in enumerate(lines):
            match = re.match(bullet_pattern, line, re.MULTILINE)
            if match:
                # Preserve entire line including bullet character
                bullet_text = line.strip()
                
                if len(bullet_text) > 5:  # Filter very short items
                    bullets.append(ExtractedSpan(
                        text=bullet_text,  # VERBATIM with bullet
                        text_type='bullet',
                        char_offset=sum(len(l) + 1 for l in lines[:line_num]),  # Approximate offset
                        sequence_num=bullet_sequence
                    ))
                    bullet_sequence += 1
        
        return bullets
    
    def _extract_table_rows(self, text: str) -> List[ExtractedSpan]:
        """
        Extract table rows from markdown or pipe-delimited tables.
        
        Assumes tables are in markdown format from docling:
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | Value 1  | Value 2  | Value 3  |
        
        CRITICAL: Preserves exact table formatting.
        
        Args:
            text: Section content (may contain markdown tables)
            
        Returns:
            List of table row spans
        """
        table_rows = []
        
        lines = text.split('\n')
        in_table = False
        row_sequence = 0
        
        for line_num, line in enumerate(lines):
            # Check if line contains table delimiters
            if '|' in line:
                in_table = True
                
                # Skip header separator lines (|---|---|)
                if re.match(r'^\s*\|[\s\-:]+\|\s*$', line):
                    continue
                
                # Store verbatim table row
                row_text = line.strip()
                
                if len(row_text) > 5:  # Filter empty rows
                    table_rows.append(ExtractedSpan(
                        text=row_text,  # VERBATIM row with pipes
                        text_type='table_row',
                        char_offset=sum(len(l) + 1 for l in lines[:line_num]),
                        sequence_num=row_sequence
                    ))
                    row_sequence += 1
            
            elif in_table and not line.strip():
                # Empty line marks end of table
                in_table = False
                row_sequence = 0
        
        return table_rows
    
    def _extract_captions(self, text: str) -> List[ExtractedSpan]:
        """
        Extract figure/table captions and footnotes.
        
        Recognizes patterns:
        - "Figure X: Description"
        - "Table X: Description"
        - "Image X: Description"
        - "Note: ..." / "Footnote:"
        
        CRITICAL: Preserves exact caption text.
        
        Args:
            text: Section content
            
        Returns:
            List of caption spans
        """
        captions = []
        
        # Patterns for different caption types
        patterns = [
            # Figure/Table/Image captions
            r'^(Figure|Table|Image|Diagram|Chart|Graph)\s+\d+[:\.](.+?)(?=\n\n|\n[A-Z]|$)',
            # Notes and footnotes
            r'^(Note|Footnote|Warning|Caution)[:\s]+(.+?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        caption_sequence = 0
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                caption_text = match.group(0).strip()
                
                if len(caption_text) > 10:  # Filter very short captions
                    captions.append(ExtractedSpan(
                        text=caption_text,  # VERBATIM caption
                        text_type='caption',
                        char_offset=match.start(),
                        sequence_num=caption_sequence
                    ))
                    caption_sequence += 1
        
        return captions
    
    def extract_stats(self, spans: List[ExtractedSpan]) -> Dict:
        """
        Calculate extraction statistics.
        
        Args:
            spans: List of extracted spans
            
        Returns:
            Dict with counts and metrics
        """
        stats = {
            'total': len(spans),
            'sentences': sum(1 for s in spans if s.text_type == 'sentence'),
            'bullets': sum(1 for s in spans if s.text_type == 'bullet'),
            'table_rows': sum(1 for s in spans if s.text_type == 'table_row'),
            'captions': sum(1 for s in spans if s.text_type == 'caption'),
        }
        
        # Average text lengths by type
        for text_type in ['sentence', 'bullet', 'table_row', 'caption']:
            type_spans = [s for s in spans if s.text_type == text_type]
            if type_spans:
                avg_len = sum(len(s.text) for s in type_spans) / len(type_spans)
                stats[f'avg_{text_type}_length'] = round(avg_len, 2)
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Test extraction
    sample_text = """
    Nizatidine is a histamine H2-receptor antagonist. The elimination half-life is 1-2 hours.
    
    Dosage Forms:
    • Capsules 150 mg
    • Capsules 300 mg
    • Oral solution 15 mg/mL
    
    Table 1 - Pharmacokinetic Parameters
    | Parameter | Value | Unit |
    |-----------|-------|------|
    | Half-life | 1-2   | hours |
    | Cmax      | 700   | ng/mL |
    
    Figure 1: Chemical structure of nizatidine
    
    Note: Dosage adjustment required in renal impairment.
    """
    
    extractor = FactSpanExtractor()
    spans = extractor.extract(sample_text)
    
    print(f"Extracted {len(spans)} spans:\n")
    for span in spans:
        print(f"[{span.text_type}] {span.text[:80]}...")
    
    stats = extractor.extract_stats(spans)
    print(f"\nStatistics: {stats}")
