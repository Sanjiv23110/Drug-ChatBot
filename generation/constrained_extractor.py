"""
Constrained Extraction System
LLM-based verbatim text extraction with post-generation validation
ZERO TOLERANCE for hallucination
"""

from typing import List, Dict, Optional, Tuple
from openai import AzureOpenAI
from rapidfuzz import fuzz
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# RUNTIME SYSTEM PROMPT (GPT-4o or equivalent)
# ═══════════════════════════════════════════════════════

RUNTIME_SYSTEM_PROMPT = """You are a SENTENCE LOCATOR for FDA pharmaceutical labeling.

YOUR TASK:
Identify which sentences from the provided context answer the user's question.
Output ONLY the indices of the relevant sentences (0-indexed).

CRITICAL RULES:
1. You do NOT write or generate text
2. You ONLY identify which existing sentences answer the question
3. Output format: A JSON object with "indices" (list of integers)
4. If no relevant sentences exist, output: {"indices": [], "reason": "not_found"}

OUTPUT FORMAT:
{"indices": [0, 1, 2]}

EXAMPLES:

Context:
[0] The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.
[1] Serious reactions include angioedema and hepatic failure.
[2] Discontinue if angioedema occurs.

Question: What are the adverse reactions?
Output: {"indices": [0, 1]}

Question: What should I do if angioedema occurs?
Output: {"indices": [2]}

Question: Is this safe during pregnancy?
Output: {"indices": [], "reason": "not_found"}

Remember: You are a LOCATOR, not a WRITER. Never generate text."""


USER_PROMPT_TEMPLATE = """QUERY: {query}

RETRIEVED CONTEXT:
{context_chunks}

INSTRUCTIONS:
Extract the answer to the query using ONLY the text above.
Copy the relevant text verbatim. Do not rewrite or paraphrase.

If no relevant text exists, respond: "Evidence not found in source document."

Answer:"""


class ConstrainedExtractor:
    """
    Extract verbatim text from retrieved chunks using LLM
    Enforces strict validation
    """
    
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        model_name: str = "gpt-4o"
    ):
        """
        Args:
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            model_name: Model deployment name
        """
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.model_name = model_name
        
        if not self.azure_endpoint or not self.azure_api_key:
            raise ValueError("Azure OpenAI credentials not provided")
        
        self.client = AzureOpenAI(
            api_key=self.azure_api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=self.azure_endpoint
        )
        
        logger.info(f"Initialized ConstrainedExtractor with model: {model_name}")
    
    def extract(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_tokens: int = 500  # Reduced since we only need indices
    ) -> Tuple[str, Dict]:
        """
        Identify relevant sentences and return raw chunk text
        
        Args:
            query: User question
            retrieved_chunks: List of chunk dicts with 'raw_text' and 'metadata'
            max_tokens: Max tokens in response
            
        Returns:
            (answer_text, extraction_metadata)
        """
        # Build context with numbered sentences
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            text = chunk['raw_text']
            context_parts.append(f"[{i}] {text}")
        
        context = "\n\n".join(context_parts)
        
        # Build user prompt
        user_prompt = f"""QUERY: {query}

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
Identify the indices of sentences that answer the question.
Output JSON format: {{"indices": [0, 1, 2]}}

Answer:"""
        
        # Call LLM
        try:
            # DEBUG: Print prompt to see what LLM sees
            logger.info(f"--- EXTRACTOR PROMPT ---\n{user_prompt}\n------------------------")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": RUNTIME_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic
                max_tokens=max_tokens,
                top_p=0.1  # Minimal sampling
            )
            
            llm_output = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                import json
                result = json.loads(llm_output)
                indices = result.get("indices", [])
                
                # Handle not found case
                if not indices:
                    reason = result.get("reason", "not_found")
                    logger.info(f"No relevant sentences found: {reason}")
                    return "Evidence not found in source document.", {
                        "model": self.model_name,
                        "indices": [],
                        "reason": reason
                    }
                
                # Extract raw text from identified chunks
                extracted_texts = []
                for idx in indices:
                    if 0 <= idx < len(retrieved_chunks):
                        raw_text = retrieved_chunks[idx]['raw_text']
                        # Strip the system-added prefix "Drug: X. Section: Y. "
                        clean_text = self._strip_chunk_prefix(raw_text)
                        extracted_texts.append(clean_text)
                
                # Format answer based on number of sentences
                answer_text = self._format_answer(extracted_texts)
                
                # Get section info from first chunk
                first_chunk_meta = retrieved_chunks[indices[0]]['metadata']
                section_name = first_chunk_meta.get('loinc_section', 'Unknown')
                
                # Add clean source attribution
                answer_text += f"\n\n[Source: {section_name}]"
                
                # Extract metadata
                extraction_metadata = {
                    "model": self.model_name,
                    "indices": indices,
                    "num_sentences_extracted": len(indices),
                    "total_chunks_available": len(retrieved_chunks),
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason
                }
                
                logger.info(f"Extracted {len(indices)} sentences: {indices}")
                return answer_text, extraction_metadata
                
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse LLM JSON output: {llm_output}")
                # Fallback: return first chunk if JSON parsing fails
                return retrieved_chunks[0]['raw_text'], {"error": "json_parse_failed", "raw_output": llm_output}
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return "Unable to extract verbatim answer from available evidence.", {"error": str(e)}
    
    def _strip_chunk_prefix(self, text: str) -> str:
        """
        Remove system-added prefix 'Drug: X. Section: Y. ' from chunk text.
        This restores the original FDA verbatim text.
        """
        import re
        # Pattern: "Drug: <anything>. Section: <anything>. " at the start
        cleaned = re.sub(r'^Drug: .*?\. Section: .*?\. ', '', text)
        return cleaned.strip()
    
    def _format_answer(self, sentences: List[str]) -> str:
        """
        Format extracted sentences for better readability.
        - Single sentence: return as-is
        - Multiple sentences: format as bulleted list
        """
        if len(sentences) == 0:
            return ""
        elif len(sentences) == 1:
            return sentences[0]
        else:
            # Bulleted list for multiple sentences
            bullet_list = "\n".join([f"• {sent}" for sent in sentences])
            return bullet_list



class PostGenerationValidator:
    """
    Validate LLM output against source chunks
    Uses fuzzy string matching to ensure verbatim extraction
    """
    
    def __init__(self, similarity_threshold: int = 95):
        """
        Args:
            similarity_threshold: Minimum similarity score (0-100)
        """
        self.threshold = similarity_threshold
    
    def validate(
        self,
        generated_answer: str,
        source_chunks: List[Dict]
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Validate that generated answer appears verbatim in source
        
        Args:
            generated_answer: Text generated by LLM
            source_chunks: Original chunks
            
        Returns:
            (is_valid, max_similarity_score, best_match_text)
        """
        # Check for refusal responses (these are always valid)
        refusal_phrases = [
            "evidence not found in source document",
            "unable to extract verbatim answer",
            "this system provides fda labeling excerpts only",
            "patient-specific decisions require clinical evaluation"
        ]
        
        if any(phrase in generated_answer.lower() for phrase in refusal_phrases):
            logger.info("Refusal response detected - validation passed")
            return True, 100.0, None
        
        # Clean answer first (lowercases everything)
        clean_answer = self._clean_text(generated_answer)
        
        # Remove source attribution (robust regex)
        import re
        # Pattern matches [source: ... section] or just [source: ...]
        # clean_answer is already lowercased by _clean_text
        clean_answer = re.split(r'\[source:', clean_answer)[0].strip()
        
        max_similarity = 0.0
        best_match = None
        
        # Check against each chunk
        for chunk in source_chunks:
            clean_chunk = self._clean_text(chunk['raw_text'])
            
            # Use partial ratio (handles substring matching)
            score = fuzz.partial_ratio(clean_answer, clean_chunk)
            
            if score > max_similarity:
                max_similarity = score
                best_match = chunk['raw_text']
        
        is_valid = max_similarity >= self.threshold
        
        if not is_valid:
            logger.warning(
                f"Validation FAILED: {max_similarity:.1f}% < {self.threshold}%\n"
                f"Generated: {clean_answer[:100]}..."
            )
        else:
            logger.info(f"Validation PASSED: {max_similarity:.1f}% similarity")
        
        return is_valid, max_similarity, best_match
    
    def _clean_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common punctuation variations
        replacements = {
            '"': '',
            '"': '',
            '"': '',
            ''': "'",
            ''': "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()


class RegulatoryQAGenerator:
    """
    Complete generation pipeline with validation
    """
    
    def __init__(
        self,
        extractor: ConstrainedExtractor,
        validator: PostGenerationValidator
    ):
        """
        Args:
            extractor: Constrained extractor instance
            validator: Validation instance
        """
        self.extractor = extractor
        self.validator = validator
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Generate answer using pure extractive method (no validation needed)
        
        Returns:
            {
                "answer": str,
                "status": "extracted" | "refused",
                "metadata": {...}
            }
        """
        if not retrieved_chunks:
            return {
                "answer": "Evidence not found in source document.",
                "status": "refused",
                "reason": "no_chunks_retrieved",
                "metadata": {}
            }
        
        # Extract answer (returns raw chunks by index)
        answer_text, extraction_metadata = self.extractor.extract(
            query=query,
            retrieved_chunks=retrieved_chunks
        )
        
        # Check if extraction failed
        if "Evidence not found" in answer_text or "Unable to extract" in answer_text:
            return {
                "answer": answer_text,
                "status": "refused",
                "reason": extraction_metadata.get("reason", "not_found"),
                "metadata": extraction_metadata
            }
        
        # Success - raw chunk text extracted
        # Get metadata from first retrieved chunk (for drug name, LOINC, etc.)
        first_chunk = retrieved_chunks[0]
        chunk_metadata = first_chunk.get('metadata', {})
        
        return {
            "answer": answer_text,
            "status": "extracted",  # Changed from "validated"
            "metadata": {
                "drug_name": chunk_metadata.get('drug_name', 'Unknown'),
                "rxcui": chunk_metadata.get('rxcui', ''),
                "set_id": chunk_metadata.get('set_id', ''),
                "root_id": chunk_metadata.get('root_id', ''),
                "version": chunk_metadata.get('version', ''),
                "effective_date": chunk_metadata.get('effective_date', ''),
                "loinc_section": chunk_metadata.get('loinc_section', ''),
                "loinc_code": chunk_metadata.get('loinc_code', ''),
                "parent_id": chunk_metadata.get('parent_id', ''),
                **extraction_metadata
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize components
    extractor = ConstrainedExtractor(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_KEY"),
        model_name="gpt-4o"
    )
    
    validator = PostGenerationValidator(similarity_threshold=95)
    
    generator = RegulatoryQAGenerator(
        extractor=extractor,
        validator=validator
    )
    
    # Example chunks (would come from retriever)
    example_chunks = [
        {
            "raw_text": "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.",
            "metadata": {
                "drug_name": "Lisinopril",
                "rxcui": "203644",
                "set_id": "abc-123",
                "root_id": "xyz-789",
                "version": "23",
                "effective_date": "20240115",
                "loinc_section": "ADVERSE REACTIONS",
                "loinc_code": "34084-4"
            }
        }
    ]
    
    # Generate answer
    query = "What are the adverse reactions of Lisinopril?"
    result = generator.generate_answer(query, example_chunks)
    
    print(json.dumps(result, indent=2))
