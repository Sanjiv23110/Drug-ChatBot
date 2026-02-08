"""
EXTRACTIVE-ONLY GENERATION SYSTEM (MANDATORY)
The LLM is an EXTRACTION ENGINE, NOT a writer.

CRITICAL REQUIREMENTS:
1. LLM outputs EXACT start/end sentences from context
2. NO paraphrasing, NO summarization, NO explanation
3. 98% validation threshold (NOT 95%)
4. If validation fails: Display RAW PARENT paragraph instead
"""

from typing import List, Dict, Optional, Tuple
from openai import AzureOpenAI
from rapidfuzz import fuzz
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# EXTRACTIVE SYSTEM PROMPT (MANDATORY - DO NOT MODIFY)
# ═══════════════════════════════════════════════════════

EXTRACTIVE_SYSTEM_PROMPT = """You are an EXTRACTION ENGINE for FDA pharmaceutical labeling.

CRITICAL RULES:
1. You are NOT a writer. You are a text locator.
2. Your ONLY job is to identify the EXACT sentences that answer the question.
3. You MUST output text VERBATIM - word-for-word from the context.
4. You MUST NOT paraphrase, summarize, explain, or rewrite.
5. If the answer is not present in the context, output: NOT_FOUND

EXTRACTION INSTRUCTIONS:
- Identify the exact start and end sentence that answers the question
- Copy the text EXACTLY as it appears
- Include ALL relevant sentences between start and end
- Preserve original punctuation, capitalization, and formatting
- If the text is a table, preserve the table structure

OUTPUT FORMAT:
[EXACT TEXT FROM CONTEXT]

If not found:
NOT_FOUND

EXAMPLES:

✓ CORRECT:
Context: "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough."
Question: What are the adverse reactions?
Output: The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.

✗ INCORRECT (paraphrased):
Output: Patients may experience dizziness, headaches, tiredness, or coughing.

✓ CORRECT (not found):
Context: "This drug is contraindicated in patients with severe renal impairment."
Question: What are the adverse reactions?
Output: NOT_FOUND

Remember: You are a LOCATOR, not a GENERATOR. Copy-paste only."""


USER_PROMPT_TEMPLATE = """QUESTION: {query}

CONTEXT (from FDA SPL):
{context}

INSTRUCTIONS:
Locate the EXACT text that answers the question.
Output the text VERBATIM.
If not found, output: NOT_FOUND

ANSWER:"""


class ExtractiveLLM:
    """
    LLM configured as EXTRACTION ENGINE ONLY
    NO generation, NO paraphrasing
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
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY") # Updated
        self.model_name = model_name or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o") # Updated default
        
        if not self.azure_endpoint or not self.azure_api_key:
            raise ValueError("Azure OpenAI credentials not provided")
        
        self.client = AzureOpenAI(
            api_key=self.azure_api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"), # Updated
            azure_endpoint=self.azure_endpoint
        )
        
        logger.info(f"Initialized ExtractiveLLM with model: {model_name}")
    
    def extract(
        self,
        query: str,
        parent_chunks: List[Dict]
    ) -> Tuple[str, Dict]:
        """
        Extract VERBATIM answer from parent chunks
        
        Args:
            query: User question
            parent_chunks: List of PARENT chunk dicts (SOURCE OF TRUTH)
            
        Returns:
            (extracted_text, metadata)
        """
        # Build context from PARENT chunks (VERBATIM text)
        context_parts = []
        for i, parent in enumerate(parent_chunks):
            section_name = parent['loinc_section']
            text = parent['raw_text']  # IMMUTABLE SOURCE OF TRUTH
            context_parts.append(f"[Paragraph {i+1} - {section_name}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Build user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            query=query,
            context=context
        )
        
        # Call LLM with EXTRACTIVE prompt
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EXTRACTIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # DETERMINISTIC
                max_tokens=2000,
                top_p=0.0  # NO sampling - pure extraction
            )
            
            extracted_text = response.choices[0].message.content.strip()
            
            metadata = {
                "model": self.model_name,
                "num_parents_used": len(parent_chunks),
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason
            }
            
            return extracted_text, metadata
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return "NOT_FOUND", {"error": str(e)}


class StrictValidator:
    """
    MANDATORY 98% validation threshold (NOT 95%)
    Uses RapidFuzz for deterministic verification
    """
    
    def __init__(self, similarity_threshold: int = 98):
        """
        Args:
            similarity_threshold: MUST be 98 (per specification)
        """
        if similarity_threshold != 98:
            logger.warning(f"Threshold {similarity_threshold} != 98. Using 98 as MANDATED.")
            similarity_threshold = 98
        
        self.threshold = similarity_threshold
        logger.info(f"Initialized StrictValidator with threshold: {self.threshold}%")
    
    def validate(
        self,
        extracted_text: str,
        parent_chunks: List[Dict]
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Validate extracted text against PARENT chunks (SOURCE OF TRUTH)
        
        Args:
            extracted_text: Text extracted by LLM
            parent_chunks: Original parent chunks (VERBATIM)
            
        Returns:
            (is_valid, max_similarity_score, best_match_text)
        """
        # Check for NOT_FOUND
        if extracted_text.strip().upper() == "NOT_FOUND":
            logger.info("LLM returned NOT_FOUND - validation passed")
            return True, 100.0, None
        
        # Clean extracted text
        clean_extracted = self._clean_text(extracted_text)
        
        max_similarity = 0.0
        best_match = None
        
        # Check against each PARENT chunk (SOURCE OF TRUTH)
        for parent in parent_chunks:
            clean_parent = self._clean_text(parent['raw_text'])
            
            # Use partial_ratio (handles substring matching)
            score = fuzz.partial_ratio(clean_extracted, clean_parent)
            
            if score > max_similarity:
                max_similarity = score
                best_match = parent['raw_text']
        
        is_valid = max_similarity >= self.threshold
        
        if not is_valid:
            logger.warning(
                f"VALIDATION FAILED: {max_similarity:.1f}% < {self.threshold}%\n"
                f"Extracted: {clean_extracted[:100]}..."
            )
        else:
            logger.info(f"Validation PASSED: {max_similarity:.1f}% >= {self.threshold}%")
        
        return is_valid, max_similarity, best_match
    
    def _clean_text(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = ' '.join(text.split())
        
        # Remove common variations
        replacements = {
            '"': '', '"': '', '"': '',
            ''': "'", ''': "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()


class ExtractiveQASystem:
    """
    Complete EXTRACTIVE QA system
    
    MANDATORY BEHAVIOR:
    1. LLM extracts VERBATIM text
    2. Validate at 98% threshold
    3. If validation fails: Display RAW PARENT paragraph instead
    """
    
    def __init__(
        self,
        extractor: ExtractiveLLM,
        validator: StrictValidator
    ):
        """
        Args:
            extractor: Extractive LLM instance
            validator: Strict validator (98% threshold)
        """
        self.extractor = extractor
        self.validator = validator
    
    def generate_answer(
        self,
        query: str,
        parent_chunks: List[Dict]
    ) -> Dict:
        """
        Generate answer using EXTRACTIVE approach
        
        CRITICAL: If validation fails, return RAW PARENT text instead
        
        Returns:
            {
                "answer": str,
                "status": "validated" | "rejected_fallback" | "not_found",
                "validation_score": float,
                "metadata": {...}
            }
        """
        if not parent_chunks:
            return {
                "answer": "NOT_FOUND",
                "status": "not_found",
                "reason": "no_parents_retrieved",
                "validation_score": 0.0,
                "metadata": {}
            }
        
        # Step 1: Extract answer using LLM
        extracted_text, extraction_metadata = self.extractor.extract(
            query=query,
            parent_chunks=parent_chunks
        )
        
        # Check if LLM returned NOT_FOUND
        if extracted_text.strip().upper() == "NOT_FOUND":
            return {
                "answer": "NOT_FOUND",
                "status": "not_found",
                "reason": "llm_not_found",
                "validation_score": 0.0,
                "metadata": extraction_metadata
            }
        
        # Step 2: Validate extracted text (98% threshold)
        is_valid, similarity_score, matched_text = self.validator.validate(
            extracted_text=extracted_text,
            parent_chunks=parent_chunks
        )
        
        # Step 3: Handle validation result
        if is_valid:
            # Validation PASSED - return extracted text
            first_parent = parent_chunks[0]
            
            return {
                "answer": extracted_text,
                "status": "validated",
                "validation_score": similarity_score,
                "metadata": {
                    "drug_name": first_parent['drug_name'],
                    "rxcui": first_parent['rxcui'],
                    "set_id": first_parent['set_id'],
                    "root_id": first_parent['root_id'],
                    "version": first_parent['version'],
                    "effective_date": first_parent['effective_date'],
                    "loinc_section": first_parent['loinc_section'],
                    "loinc_code": first_parent['loinc_code'],
                    **extraction_metadata
                }
            }
        else:
            # Validation FAILED - FALLBACK to RAW PARENT text
            logger.warning(
                f"Validation failed ({similarity_score:.1f}% < 98%). "
                f"Falling back to RAW PARENT text."
            )
            
            # Return FIRST parent chunk as fallback (VERBATIM)
            first_parent = parent_chunks[0]
            
            return {
                "answer": first_parent['raw_text'],  # RAW PARENT (VERBATIM)
                "status": "rejected_fallback",
                "reason": "validation_failed",
                "validation_score": similarity_score,
                "rejected_extraction": extracted_text,  # For debugging
                "metadata": {
                    "drug_name": first_parent['drug_name'],
                    "rxcui": first_parent['rxcui'],
                    "set_id": first_parent['set_id'],
                    "loinc_section": first_parent['loinc_section'],
                    "fallback_mode": True,
                    **extraction_metadata
                }
            }


# Example usage
if __name__ == "__main__":
    # Initialize components
    extractor = ExtractiveLLM(model_name="gpt-4o")
    validator = StrictValidator(similarity_threshold=98)
    qa_system = ExtractiveQASystem(extractor=extractor, validator=validator)
    
    # Example parent chunks
    example_parents = [
        {
            "parent_id": "LISINOPRIL_v23_34084-4_para_001",
            "raw_text": "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.",
            "drug_name": "Lisinopril",
            "rxcui": "203644",
            "set_id": "abc-123",
            "root_id": "xyz-789",
            "version": "23",
            "effective_date": "20240115",
            "loinc_section": "ADVERSE REACTIONS",
            "loinc_code": "34084-4"
        }
    ]
    
    # Generate answer
    query = "What are the adverse reactions of Lisinopril?"
    result = qa_system.generate_answer(query, example_parents)
    
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer']}")
    print(f"Validation: {result['validation_score']:.1f}%")
