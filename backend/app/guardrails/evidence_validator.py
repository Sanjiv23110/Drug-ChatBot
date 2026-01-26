"""
Evidence validation guardrail for RAG responses.

Prevents hallucinations by ensuring sufficient evidence exists
before generating answers.
"""
import logging
from typing import List, Dict, Set

# Stop words to ignore in keyword matching
STOP_WORDS = {
    'what', 'is', 'the', 'of', 'for', 'a', 'an', 'and', 'or', 
    'but', 'in', 'on', 'at', 'to', 'from', 'with', 'by', 'are'
}


def has_sufficient_evidence(
    chunks: List[Dict], 
    query: str,
    min_chunks: int = 1,
    min_keyword_overlap: float = 0.3
) -> bool:
    """
    Validate if retrieved chunks contain sufficient evidence to answer query.
    
    Implements 3-check validation:
    1. Minimum number of chunks retrieved
    2. Relevance score check (if available)
    3. Keyword overlap with query
    
    Args:
        chunks: List of retrieved chunks (each with 'chunk_text' and optional 'score')
        query: Original user query
        min_chunks: Minimum chunks required (default: 1)
        min_keyword_overlap: Minimum keyword overlap ratio (default: 0.3)
        
    Returns:
        True if sufficient evidence exists, False otherwise
        
    Examples:
        >>> chunks = [{'chunk_text': 'Gravol is used for nausea', 'score': 0.85}]
        >>> has_sufficient_evidence(chunks, 'what is gravol')
        True
        
        >>> has_sufficient_evidence([], 'what is gravol')
        False
    """
    # Check 1: Minimum chunks threshold
    if not chunks or len(chunks) < min_chunks:
        logging.info(f"Evidence check FAILED: {len(chunks) if chunks else 0} chunks (need {min_chunks})")
        return False
    
    # Check 2: Relevance score (if available)
    # Note: FAISS scores are similarity scores (higher = better)
    if 'score' in chunks[0]:
        top_score = chunks[0]['score']
        # Very low score indicates poor match
        if top_score < 0.2:  # Adjust threshold as needed
            logging.info(f"Evidence check FAILED: Top score {top_score:.3f} too low")
            return False
    
    # Check 3: Keyword overlap
    query_keywords = extract_keywords(query)
    
    if not query_keywords:
        # If no meaningful keywords, accept (edge case)
        return True
    
    # Check keyword presence in top 3 chunks
    top_chunks_text = ' '.join([
        chunk.get('chunk_text', '').lower() 
        for chunk in chunks[:3]
    ])
    
    matching_keywords = sum(
        1 for keyword in query_keywords 
        if keyword in top_chunks_text
    )
    
    overlap_ratio = matching_keywords / len(query_keywords)
    
    if overlap_ratio < min_keyword_overlap:
        logging.info(
            f"Evidence check FAILED: Keyword overlap {overlap_ratio:.2%} "
            f"(need {min_keyword_overlap:.0%})"
        )
        return False
    
    # All checks passed
    logging.info(
        f"Evidence check PASSED: {len(chunks)} chunks, "
        f"{overlap_ratio:.0%} keyword overlap"
    )
    return True


def extract_keywords(query: str) -> Set[str]:
    """
    Extract meaningful keywords from query.
    
    Removes stop words and extracts content words.
    
    Args:
        query: User query string
        
    Returns:
        Set of lowercase keywords
    """
    # Lowercase and split
    words = query.lower().split()
    
    # Remove stop words and short words
    keywords = {
        word.strip('.,!?;:')
        for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }
    
    return keywords


# ============================================================================
# PHASE 2: No-Evidence Response Templates
# ============================================================================

# Response templates for no-evidence scenarios
NO_EVIDENCE_RESPONSES = {
    'default': (
        "I could not find sufficient information about this query in the available "
        "drug monographs.\n\n"
        "**Possible reasons:**\n"
        "• The information may not be in the indexed documents\n"
        "• The query may need rephrasing\n"
        "• The drug/topic may not be in our database\n\n"
        "**Please consult a healthcare professional or refer to additional sources "
        "for this information.**"
    ),
    
    'drug_not_found': (
        "I could not find information about **{drug}** in the available monographs.\n\n"
        "**Suggestions:**\n"
        "• Verify the drug name spelling\n"
        "• Try using the generic or brand name variant\n"
        "• Check if the drug is in our database\n\n"
        "**Please consult a healthcare professional for accurate information.**"
    ),
    
    'section_not_found': (
        "While information about **{drug}** exists in the database, "
        "I could not find specific information about **{topic}** in the available documentation.\n\n"
        "**This could mean:**\n"
        "• This section may not be present in the drug monograph\n"
        "• The information may be organized differently\n"
        "• Additional sources may be needed\n\n"
        "**Please consult a healthcare professional or the complete drug monograph "
        "for this information.**"
    ),
    
    'low_confidence': (
        "I found some information related to your query, but the match confidence is low. "
        "To ensure accuracy, I cannot provide a definitive answer.\n\n"
        "**Suggestions:**\n"
        "• Try rephrasing your question\n"
        "• Be more specific about what you're looking for\n"
        "• Consult the complete drug monograph\n\n"
        "**For medical accuracy, please consult a healthcare professional.**"
    )
}


def get_no_evidence_response(
    query: str,
    drug_name: str = None,
    section_type: str = None,
    reason: str = 'default'
) -> str:
    """
    Generate appropriate no-evidence response based on context.
    
    Args:
        query: Original user query
        drug_name: Detected drug name (if any)
        section_type: Detected section type (if any)
        reason: Response type ('default', 'drug_not_found', 'section_not_found', 'low_confidence')
        
    Returns:
        Formatted no-evidence message
        
    Examples:
        >>> get_no_evidence_response("what is xyz", reason='drug_not_found', drug_name='XYZ')
        "I could not find information about **XYZ** in the available monographs..."
    """
    # Determine response type based on context
    if reason == 'drug_not_found' and drug_name:
        template = NO_EVIDENCE_RESPONSES['drug_not_found']
        return template.format(drug=drug_name)
    
    elif reason == 'section_not_found' and drug_name and section_type:
        template = NO_EVIDENCE_RESPONSES['section_not_found']
        topic = section_type.replace('_', ' ').title()
        return template.format(drug=drug_name, topic=topic)
    
    elif reason == 'low_confidence':
        return NO_EVIDENCE_RESPONSES['low_confidence']
    
    else:
        return NO_EVIDENCE_RESPONSES['default']

