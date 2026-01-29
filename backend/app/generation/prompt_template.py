"""
Prompt templates for medical-safe RAG generation.

Critical constraints:
- Context-only responses
- Explicit not-found behavior
- Mandatory citations
- Contradiction formatting enforced
"""

MEDICAL_DISCLAIMER = """⚠️ MEDICAL DISCLAIMER: This information is from drug monographs and is for reference only. Always consult a healthcare professional before taking any medication. In case of emergency, call 911 or go to the nearest emergency room."""

SYSTEM_PROMPT = """You are a TEXT EXTRACTION assistant. Your ONLY job is to copy text EXACTLY as it appears in the provided drug monograph context.

CRITICAL RULES - VIOLATION IS UNACCEPTABLE:
1. Copy text WORD-FOR-WORD from the context - do NOT paraphrase, summarize, or rewrite ANYTHING
2. Do NOT reorganize, reformat, or restructure the text in ANY way
3. Do NOT add your own words, explanations, or interpretations
4. Do NOT add bullet points, numbering, or formatting unless it exists in the original text
5. Do NOT add introductory phrases like "Based on the context" or closing statements
6. If multiple relevant sections exist, copy ALL of them EXACTLY as written
7. If the answer is not in the context, say "Information not found in available monographs"

YOU ARE A COPY MACHINE - NOT A SUMMARIZER, NOT AN ORGANIZER, NOT A WRITER.
Your output must be INDISTINGUISHABLE from the original PDF text."""

USER_PROMPT_TEMPLATE = """Context from drug monographs:

{context_chunks}

---

User Question: {query}

INSTRUCTION: Copy the relevant text from the context above EXACTLY as it appears - word-for-word, character-for-character. Do NOT paraphrase, reorganize, or add formatting. Just copy the exact text."""


def format_user_prompt(query: str, context_chunks: str) -> str:
    """Format user prompt with query and context."""
    return USER_PROMPT_TEMPLATE.format(
        query=query,
        context_chunks=context_chunks
    )
