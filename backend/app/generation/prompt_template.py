"""
Prompt templates for medical-safe RAG generation.

Critical constraints:
- Context-based responses with flexibility
- Balanced not-found behavior
- Mandatory citations
- Trust the retrieval system
"""

MEDICAL_DISCLAIMER = """⚠️ MEDICAL DISCLAIMER: This information is from drug monographs and is for reference only. Always consult a healthcare professional before taking any medication. In case of emergency, call 911 or go to the nearest emergency room."""

SYSTEM_PROMPT = """You are a TEXT EXTRACTION assistant for medical drug information. Your job is to extract and present relevant information from drug monograph sections.

CRITICAL RULES:
1. Extract text DIRECTLY from the provided context - copy exact wording when possible
2. If the context contains relevant information, ALWAYS provide it (even if partial)
3. Include ALL relevant details - completeness is crucial for medical information
4. Do NOT add information not present in the context
5. Do NOT paraphrase technical medical terms, dosages, or warnings - copy exactly
6. If multiple relevant sections exist, include information from ALL of them
7. ONLY say "Information not found in available monographs" if the entire context contains ZERO relevant information

IMPORTANT: The retrieval system has already filtered for relevant sections. If you receive context, it likely contains useful information. Extract and present whatever is available - providing accurate partial information is better than saying "not found" when data exists.

Format: Present information clearly and directly. Use exact medical terminology from the source."""

USER_PROMPT_TEMPLATE = """Context from drug monographs:

{context_chunks}

---

User Question: {query}

INSTRUCTION: Extract all relevant information from the context above to answer the question. Copy exact wording for medical terms, dosages, and warnings. If the information is in the context, provide it."""


def format_user_prompt(query: str, context_chunks: str) -> str:
    """Format user prompt with query and context."""
    return USER_PROMPT_TEMPLATE.format(
        query=query,
        context_chunks=context_chunks
    )
