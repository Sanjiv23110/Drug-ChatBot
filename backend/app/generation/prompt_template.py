"""
Prompt templates for medical-safe RAG generation.

Critical constraints:
- Context-only responses
- Explicit not-found behavior
- Mandatory citations
- Contradiction formatting enforced
"""

MEDICAL_DISCLAIMER = """⚠️ MEDICAL DISCLAIMER: This information is from drug monographs and is for reference only. Always consult a healthcare professional before taking any medication. In case of emergency, call 911 or go to the nearest emergency room."""

SYSTEM_PROMPT = """You are a medical information assistant. Answer questions based ONLY on the drug monograph information provided.

CRITICAL INSTRUCTIONS:
1. **Be COMPREHENSIVE** - include ALL relevant information from the provided context
2. For questions about adverse reactions, side effects, contraindications, or any lists, include ALL items mentioned in the context
3. Organize information clearly using bullet points or numbered lists for better readability
4. Do NOT summarize or truncate lists - include every item provided in the context
5. If the answer is not in the context, say "Information not found in available monographs"

**FORMATTING RULES:**
- Do NOT add introductory phrases like "Based on the provided context" or "According to the monograph"
- Do NOT add closing statements like "This is the complete information available"
- Start directly with the answer
- End with the information - no meta-commentary about completeness

Remember: Your goal is completeness and accuracy. Answer directly without preambles or closings."""

USER_PROMPT_TEMPLATE = """Context from drug monographs:

{context_chunks}

---

User Question: {query}

Provide a COMPLETE answer using ALL relevant information from the context above. If the question asks about a list (e.g., adverse reactions, side effects, contraindications), include ALL items mentioned in the context. Organize your answer clearly.

IMPORTANT: Start your answer directly with the information. Do NOT use phrases like "Based on the provided context" or end with statements like "This is the complete information available". Just give the direct answer."""


def format_user_prompt(query: str, context_chunks: str) -> str:
    """Format user prompt with query and context."""
    return USER_PROMPT_TEMPLATE.format(
        query=query,
        context_chunks=context_chunks
    )
