"""
Attribute Evidence Aggregator

Orchestrates verbatim sentence selection for attribute queries.
Uses LLM purely for relevance classification (filtering), NEVER for generation.

CRITICAL:
- Input: Candidate FactSpans
- Output: Subset of FactSpans (indices only)
- NO paraphrasing, NO rewriting
"""
import os
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)

@dataclass
class CandidateSentence:
    index: int
    text: str
    section_name: str
    ids: int  # database id of fact span

class AttributeAggregator:
    # Strict JSON output instruction
    SYSTEM_PROMPT = """You are a medical evidence filter.
Your task is to identify which sentences contain FACTUAL properties about a specific medical attribute.

INPUT:
- Attribute Name
- User Question
- Numbered Candidate Sentences

OUTPUT:
- JSON object with key "relevant_sentence_indices" containing a list of integers.
- Select ONLY sentences that directly state facts answering the question/attribute.
- Do NOT select sentences that are unrelated context.
- If no sentences are relevant, return empty list [].

CRITICAL RULES:
1. You are a FILTER. You do NOT write text.
2. You output ONLY JSON.
3. You select sentences by their index ID.
"""
    
    USER_PROMPT_TEMPLATE = """ATTRIBUTE: {attribute}
QUESTION: {question}

CANDIDATE SENTENCES:
{sentences_text}

Select indices of sentences that answer the question about the attribute."""

    def __init__(self):
        self.client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        self.model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

    async def filter_sentences(
        self, 
        attribute: str, 
        question: str, 
        candidates: List[CandidateSentence]
    ) -> List[CandidateSentence]:
        """
        Use LLM to select relevant sentences by index.
        """
        if not candidates:
            return []

        # Format candidates for prompt
        # [1] "Sentence text..."
        sentences_text = "\n".join(
            f"[{c.index}] \"{c.text}\"" for c in candidates
        )
        
        user_content = self.USER_PROMPT_TEMPLATE.format(
            attribute=attribute,
            question=question,
            sentences_text=sentences_text
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            indices = data.get("relevant_sentence_indices", [])
            
            # Filter candidates based on returned indices
            selected = [c for c in candidates if c.index in indices]
            
            logger.info(
                f"Attribute Aggregation: Found {len(candidates)} candidates, "
                f"LLM selected {len(selected)}."
            )
            
            return selected

        except Exception as e:
            logger.error(f"Error in attribute aggregation LLM call: {e}")
            # Fail safe: return nothing rather than garbage
            return []

    def format_verbatim_response(self, selected_sentences: List[CandidateSentence]) -> str:
        """
        Format selected sentences into section-grouped verbatim blocks.
        
        Format:
        [SECTION: SECTION_NAME]
        <sentence>
        <sentence>
        """
        if not selected_sentences:
            return "NO_RESULT"

        # Group by section
        from collections import defaultdict
        grouped = defaultdict(list)
        for s in selected_sentences:
            grouped[s.section_name].append(s.text)
            
        output_parts = []
        for section, texts in grouped.items():
            header = f"[SECTION: {section.upper()}]"
            block = "\n".join(texts)
            output_parts.append(f"{header}\n{block}")
            
        return "\n\n".join(output_parts)
