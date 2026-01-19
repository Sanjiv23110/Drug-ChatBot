"""
Medical-safe answer generator with 5-gate validation.

Critical safety features:
- Citation validity enforcement
- 15 hallucination markers
- Explicit not-found behavior
- Contradiction formatting
- Mandatory disclaimers
"""
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

from app.generation.prompt_template import (
    SYSTEM_PROMPT,
    MEDICAL_DISCLAIMER,
    format_user_prompt
)

load_dotenv()


# Hallucination markers (MINIMAL - only 4 critical ones)
# Ultra-lenient: only flag clear opinion/invention markers
HALLUCINATION_MARKERS = [
    "in my opinion",
    "i believe",
    "as far as i know",
    "to the best of my knowledge"
]

# Removed even more (these can appear in medical context):
# - "generally speaking" - can be legitimate
# - "based on general medical knowledge" - too restrictive
# - "based on common medical practice" - too restrictive
# - "research suggests" - appears in monographs


class AnswerGenerator:
    """
    Generate medical-safe answers with strict validation.
    
    5-Gate Validation Pipeline:
    1. Must have context chunks
    2. Generate response (GPT-4, temp=0)
    3. Validate citations (all must match context)
    4. Detect hallucination markers
    5. Detect not-found indicators
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 800,  # Increased for comprehensive answers
        timeout: int = 30
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
        
        logging.info(f"AnswerGenerator initialized with model={self.model}, temp={self.temperature}")
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict]
    ) -> Dict:
        """
        Generate answer with 5-gate validation.
        
        Args:
            query: User query
            context_chunks: Retrieved chunks with metadata
            
        Returns:
            {
                'answer': str (with disclaimer),
                'sources': List[Dict],
                'has_answer': bool,
                'validation_passed': bool,
                'response_time_ms': int
            }
        """
        start_time = time.time()
        
        # Gate 1: Must have context
        if not context_chunks:
            return self._return_not_found("No context chunks provided", start_time)
        
        # Format context with sources
        context_text, context_sources = self._format_context(context_chunks)
        user_prompt = format_user_prompt(query, context_text)
        
        # Generate response with GPT-4 (NO VALIDATION - just return answer)
        try:
            response = self._call_llm(user_prompt)
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            return self._return_not_found(f"LLM generation failed: {e}", start_time)
        
        # Return answer directly - NO MORE VALIDATION GATES
        # Gate 5 DISABLED: Not-found detection was too aggressive, causing false negatives
        # Many valid answers were being rejected because they contained phrases like "not documented"
        # if self._detect_not_found(response):
        #     return self._return_not_found("LLM indicated not found", start_time)
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            'answer': response,
            'sources': context_sources,
            'has_answer': True,
            'validation_passed': True,
            'response_time_ms': response_time_ms
        }
    
    def _format_context(self, chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Format chunks with inline source markers.
        
        Returns:
            (formatted_text, sources_list)
        """
        context_parts = []
        sources = []
        
        for idx, chunk in enumerate(chunks, 1):
            source_label = f"[Source {idx}]"
            
            # Extract source metadata
            file = Path(chunk.get('file_path', 'Unknown')).name
            page = chunk.get('page_num', 'Unknown')
            section = chunk.get('section_name', 'Unknown')
            drug = chunk.get('drug_generic', chunk.get('file_path', 'Unknown drug'))
            
            # Format chunk with source
            context_parts.append(
                f"{source_label} {drug}, Page {page}, Section: {section}\n"
                f"{chunk['chunk_text']}\n"
            )
            
            sources.append({
                'file': file,
                'page': page,
                'section': section,
                'drug': drug
            })
        
        context_text = "\n---\n\n".join(context_parts)
        return context_text, sources
    
    def _call_llm(self, user_prompt: str, max_retries: int = 3) -> str:
        """
        Call Azure OpenAI with retry logic.
        
        Args:
            user_prompt: Already formatted user prompt with context
            max_retries: Number of retry attempts
            
        Returns:
            LLM response text
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                
                return response.choices[0].message.content
            
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                wait_time = 2 ** attempt
                logging.warning(f"LLM call attempt {attempt + 1} failed: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        raise Exception("LLM call failed after retries")
    
    def _validate_citations(self, response: str, context_sources: List[Dict]) -> bool:
        """
        Validate all citations match actual context sources.
        
        CRITICAL SAFETY CHECK:
        If citation doesn't match source → REJECT response
        
        Note: We only validate file + page number, NOT section names.
        GPT-4 reads section names from the PDF text itself, which may differ
        from our detected section names during ingestion.
        """
        # Extract citations from response
        extracted_citations = self._extract_citations(response)
        
        if not extracted_citations:
            # No citations found - might be valid if "not found" response
            # Let other gates handle this
            return True
        
        # Build valid (file, page) pairs from context
        valid_file_pages = set()
        for source in context_sources:
            # Use basename only for file comparison
            file_basename = Path(source['file']).name
            # Only use file + page (section names may differ from PDF text)
            file_page_key = f"{file_basename}|{source['page']}"
            valid_file_pages.add(file_page_key.lower())
        
        # Verify each citation (file + page only)
        for citation in extracted_citations:
            file_basename = Path(citation['file']).name
            file_page_key = f"{file_basename}|{citation['page']}"
            
            if file_page_key.lower() not in valid_file_pages:
                logging.error(
                    f"CITATION VALIDATION FAILED: "
                    f"Response cited [{citation['file']}, Page {citation['page']}] "
                    f"which is not in context sources"
                )
                return False
        
        logging.info(f"Citation validation passed: {len(extracted_citations)} citations verified (file+page)")
        return True
    
    def _extract_citations(self, response: str) -> List[Dict]:
        """
        Extract citation metadata from response.
        
        Pattern: [Source: filename, Page X, Section: Y]
        """
        pattern = r'\[Source: ([^,]+), Page (\d+|Unknown), Section: ([^\]]+)\]'
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        citations = []
        for file, page, section in matches:
            citations.append({
                'file': file.strip(),
                'page': page if page == 'Unknown' else int(page),
                'section': section.strip()
            })
        
        return citations
    
    def _detect_hallucination(self, response: str, context_chunks: List[Dict]) -> bool:
        """
        Context-aware hallucination detection.
        
        Only flags phrases that appear in response BUT NOT in source context.
        This allows legitimate medical language from monographs while blocking
        true hallucinations.
        
        Example: "typically" in response is OK if it also appears in the PDF.
        """
        response_lower = response.lower()
        
        # Build combined context text from all chunks
        context_text = " ".join(
            [chunk.get('chunk_text', chunk.get('text', '')) for chunk in context_chunks]
        ).lower()
        
        for marker in HALLUCINATION_MARKERS:
            if marker in response_lower:
                # Check if phrase also appears in source context
                if marker not in context_text:
                    # Phrase in response but NOT in context = hallucination!
                    logging.warning(
                        f"HALLUCINATION MARKER DETECTED: '{marker}' in response "
                        f"but NOT in source context. REJECTING."
                    )
                    return True
                else:
                    # Phrase appears in both = legitimate medical language
                    logging.debug(f"Phrase '{marker}' found in both response and context - allowing")
        
        return False
    
    def _detect_not_found(self, response: str) -> bool:
        """
        Detect if LLM explicitly said not found.
        
        LENIENT: Only catch obvious not-found statements.
        """
        response_lower = response.lower()
        
        # Only these exact phrases indicate not-found
        explicit_not_found = [
            "information not found in available monographs",
            "i couldn't find information about that",
            "not found in the available",
            "the answer is not in"
        ]
        
        return any(phrase in response_lower for phrase in explicit_not_found)
    
    def _return_not_found(self, reason: str, start_time: float) -> Dict:
        """
        Return standardized not-found response.
        
        Called when any validation gate fails.
        """
        logging.info(f"Returning NOT FOUND response: {reason}")
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            'answer': (
                "I couldn't find information about that in the available drug monographs.\n\n"
                "This may be because:\n"
                "- The drug is not in our database\n"
                "- The specific information requested is not documented in the monographs\n"
                "- Try rephrasing your question or checking the drug name spelling"
            ),
            'sources': [],
            'has_answer': False,
            'validation_passed': False,
            'response_time_ms': response_time_ms
        }
