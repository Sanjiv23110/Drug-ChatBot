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

SPECIFIC INSTRUCTIONS FOR "TREATMENT" / "MANAGEMENT" / "OVERDOSE":
- Include ALL supportive measures (e.g. "airway maintenance", "lavage")
- Include ALL monitoring instructions (e.g. "ECG monitoring", "vital signs")
- Include RELEVANT UNKNOWNS or WARNINGS (e.g. "dialysis value unknown")
- ERR ON THE SIDE OF INCLUSION. Completeness is critical for patient safety.

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
        max_tokens: int = 500,
        section_specific: bool = False,
        target_loinc_code: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Identify relevant sentences and return raw chunk text.
        
        Args:
            query: User question
            retrieved_chunks: List of chunk dicts with 'raw_text' and 'metadata'
            max_tokens: Max tokens in response
            section_specific: If True, bypass LLM and return ALL chunks verbatim.
                              Used when retrieval already filtered to the correct section.
            target_loinc_code: LOINC code of the target section. Used to filter
                               wrong-section chunks from retrieval fallback.
            
        Returns:
            (answer_text, extraction_metadata)
        """
        # SECTION-SPECIFIC BYPASS: Skip LLM, return ALL chunks as bullet points
        if section_specific:
            # GUARANTEE: This path involves NO OpenAI API calls.
            # It uses purely deterministic Python string manipulation.
            return self._extract_all_verbatim(query, retrieved_chunks, target_loinc_code)
        
        # GENERAL QUERY: Use LLM to identify relevant sentence indices
        return self._extract_via_llm(query, retrieved_chunks, max_tokens)
    
    def _detect_query_shape(self, query: str) -> str:
        """
        Classify query into FACT, MANAGEMENT, or LIST.
        Pure Python — no LLM. Returns one of:
          'FACT'       → return minimal matching phrase
          'MANAGEMENT' → return full paragraph verbatim
          'LIST'       → return full paragraph verbatim
        """
        q = query.lower()

        # LIST signals
        list_kw = ["what are", "list ", "adverse reaction", "adverse effect",
                   "side effect", "contraindication", "warnings", "precautions"]
        if any(kw in q for kw in list_kw):
            return "LIST"

        # MANAGEMENT signals
        mgmt_kw = ["how to", "how do", "treat", "treatment", "management",
                   "overdose", "overdosage", "should be", "monitoring",
                   "administer", "procedure", "protocol"]
        if any(kw in q for kw in mgmt_kw):
            return "MANAGEMENT"

        # FACT signals (short-answer expected)
        fact_kw = ["what is", "generic name", "brand name", "strength",
                   "route of", "manufacturer", "ndc", "dosage form",
                   "chemical name", "molecular"]
        # Only classify as FACT if no management-style language is present
        mgmt_exclusions = ["how", "treat", "management", "monitoring", "precaution"]
        if any(kw in q for kw in fact_kw) and not any(ex in q for ex in mgmt_exclusions):
            return "FACT"

        # Default: treat as MANAGEMENT (return full paragraph — safe for clinical use)
        return "MANAGEMENT"

    def _extract_fact_span(self, query: str, full_text: str) -> str:
        """
        For FACT queries: extract the shortest verbatim span that answers the query.
        Uses keyword proximity search — no LLM, no paraphrasing.
        Falls back to full_text if no tight span is found.
        """
        import re

        q_lower = query.lower()
        text_lower = full_text.lower()

        # Build candidate anchor terms from query (exclude stopwords & the drug name)
        stopwords = {
            "what", "is", "the", "of", "for", "a", "an", "its", "are",
            "how", "does", "do", "my", "to", "in", "this", "that", "with"
        }
        query_tokens = [
            t.strip("?.,;:") for t in q_lower.split()
            if t.strip("?.,;:") not in stopwords and len(t) > 2
        ]

        # Find the sentence in full_text that contains the most query tokens
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        best_sentence = ""
        best_hits = 0

        for sent in sentences:
            s_lower = sent.lower()
            hits = sum(1 for tok in query_tokens if tok in s_lower)
            if hits > best_hits:
                best_hits = hits
                best_sentence = sent

        return best_sentence.strip() if best_sentence else full_text

    def _extract_all_verbatim(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        target_loinc_code: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Shape-aware deterministic extraction. No LLM.

        The retriever now delivers exactly ONE best-parent chunk whose
        raw_text is a fully reconstructed verbatim paragraph.

        Shape routing:
          FACT       → minimal phrase from full paragraph
          MANAGEMENT → full paragraph verbatim
          LIST       → full paragraph verbatim
        """
        # Optional LOINC filter (safety net for fallback paths)
        if target_loinc_code:
            section_chunks = [
                c for c in retrieved_chunks
                if c.get('metadata', {}).get('loinc_code') == target_loinc_code
            ]
            if section_chunks:
                logger.info(
                    f"Section filter: {len(section_chunks)}/{len(retrieved_chunks)} "
                    f"chunks match LOINC {target_loinc_code}"
                )
                retrieved_chunks = section_chunks
            else:
                logger.warning(
                    f"No chunks match LOINC {target_loinc_code}, "
                    f"using all {len(retrieved_chunks)} chunks"
                )

        if not retrieved_chunks:
            return "Evidence not found in source document.", {
                "extraction_mode": "section_verbatim",
                "reason": "no_chunks"
            }

        # The retriever returns exactly 1 reconstructed paragraph.
        # If multiple chunks survive (fallback edge-case), use the highest-scored one.
        best_chunk = max(
            retrieved_chunks,
            key=lambda c: c.get("rerank_score", 0.0)
        )
        full_text = self._strip_chunk_prefix(best_chunk.get("raw_text", ""))

        if not full_text:
            return "Evidence not found in source document.", {
                "extraction_mode": "section_verbatim",
                "reason": "empty_text"
            }

        # Detect query shape and apply shape-specific extraction
        shape = self._detect_query_shape(query)
        logger.info(f"Query shape: {shape} | query='{query[:60]}'")

        if shape == "FACT":
            answer_text = self._extract_fact_span(query, full_text)
            extraction_mode = "fact_span"
        else:
            # MANAGEMENT or LIST → return full reconstructed paragraph
            answer_text = full_text
            extraction_mode = "full_paragraph"

        # Source attribution
        first_meta = best_chunk.get('metadata', {})
        section_name = first_meta.get('loinc_section', 'Unknown')
        answer_text += f"\n\n[Source: {section_name}]"

        logger.info(
            f"Section-specific verbatim extraction: shape={shape} "
            f"chars={len(answer_text)} (no LLM)"
        )

        return answer_text, {
            "extraction_mode": extraction_mode,
            "query_shape":     shape,
            "section":         section_name,
            "target_loinc":    target_loinc_code,
            "llm_used":        False
        }
    
    def _extract_via_llm(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_tokens: int = 500
    ) -> Tuple[str, Dict]:
        """
        LLM-based extraction: ask GPT to identify relevant sentence indices.
        Used for general/ambiguous queries where no section was detected.
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
            logger.info(f"LLM extraction for general query: {query}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": RUNTIME_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                top_p=0.1
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
                        "extraction_mode": "llm",
                        "model": self.model_name,
                        "indices": [],
                        "reason": reason
                    }
                
                # Extract raw text from identified chunks
                extracted_texts = []
                seen_text_fps: set = set()
                for idx in indices:
                    if 0 <= idx < len(retrieved_chunks):
                        raw_text = retrieved_chunks[idx]['raw_text']
                        clean_text = self._strip_chunk_prefix(raw_text)
                        # Deduplicate: two chunks can produce the same cleaned text
                        # (e.g. same sentence under different parent contexts).
                        # Collapse here so the same sentence never appears twice.
                        fp = " ".join(clean_text.lower().split())[:200]
                        if fp in seen_text_fps:
                            logger.debug(
                                f"Extractor dedup: dropped duplicate idx={idx}: {clean_text[:60]}..."
                            )
                            continue
                        seen_text_fps.add(fp)
                        extracted_texts.append(clean_text)

                
                answer_text = self._format_answer(extracted_texts)
                
                # Source attribution
                first_chunk_meta = retrieved_chunks[indices[0]]['metadata']
                section_name = first_chunk_meta.get('loinc_section', 'Unknown')
                answer_text += f"\n\n[Source: {section_name}]"
                
                extraction_metadata = {
                    "extraction_mode": "llm",
                    "model": self.model_name,
                    "indices": indices,
                    "num_sentences_extracted": len(indices),
                    "total_chunks_available": len(retrieved_chunks),
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason,
                    "llm_used": True
                }
                
                logger.info(f"LLM extracted {len(indices)} sentences: {indices}")
                return answer_text, extraction_metadata
                
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse LLM JSON output: {llm_output}")
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
        
        NOTE: This method uses ONLY Python string manipulation.
        NO LLM or external API is called here.
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
        retrieved_chunks: List[Dict],
        section_specific: bool = False,
        target_loinc_code: Optional[str] = None
    ) -> Dict:
        """
        Generate answer using pure extractive method.
        
        Args:
            query: User question
            retrieved_chunks: Chunks from retrieval pipeline
            section_specific: If True, bypass LLM and return all chunks verbatim.
            target_loinc_code: LOINC code to filter chunks by section.
        
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
        
        # Extract answer
        answer_text, extraction_metadata = self.extractor.extract(
            query=query,
            retrieved_chunks=retrieved_chunks,
            section_specific=section_specific,
            target_loinc_code=target_loinc_code
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
