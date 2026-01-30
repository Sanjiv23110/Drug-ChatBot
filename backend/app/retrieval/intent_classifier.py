"""
Query Intent Classifier for Medical Drug Queries.

Extracts structured intent from user queries:
- target_drug: Which drug is being asked about
- target_section: Which section type (DYNAMIC - learned from data)
- needs_image: Whether user wants a chemical structure image

Uses rule-based matching with LLM fallback for ambiguous queries.

DESIGN: NO HARDCODED SECTIONS - section detection uses dynamic patterns
and learns from the database.
"""
import os
import re
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

from openai import AzureOpenAI

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Structured intent extracted from a user query."""
    # Extracted entities
    target_drug: Optional[str] = None
    target_section: Optional[str] = None  # DYNAMIC - string, not enum
    
    # Flags
    needs_image: bool = False
    
    # Confidence scores
    drug_confidence: float = 0.0
    section_confidence: float = 0.0
    
    # Original query
    original_query: str = ""
    normalized_query: str = ""
    
    # Classification method
    method: str = "rule_based"  # "rule_based" or "llm"


class IntentClassifier:
    """
    Classify query intent for routing to appropriate retrieval path.
    
    Strategy:
    1. Rule-based extraction (fast, deterministic)
    2. LLM fallback for ambiguous queries
    
    DYNAMIC SECTIONS:
    - Uses keyword patterns to suggest section names
    - Section names are stored as-is (no enum restriction)
    - The retrieval engine uses fuzzy matching
    """
    
    # DYNAMIC section keyword patterns
    # Maps keywords to likely section name patterns (NOT fixed enums)
    SECTION_PATTERNS = {
        # Indications / Uses
        r"(used\s+for|treat|use|indication|therapeutic|medical\s+use)": "indications",
        
        # Dosage / Administration
        r"(dosage|dose|how\s+much|administration|administer|how\s+to\s+take|posology)": "dosage",
        
        # Contraindications
        r"(contraindication|when\s+not\s+to\s+use|should\s+not|avoid|do\s+not\s+use|forbidden)": "contraindications",
        
        # Warnings / Precautions
        r"(warning|precaution|caution|careful|alert|safety|risk|danger|boxed\s+warning)": "warnings",
        
        # Adverse effects / Side effects
        r"(side\s+effect|adverse\s+(effect|reaction)|reaction|undesirable|toxicity|harmful)": "adverse",
        
        # Pharmacology
        r"(pharmacology|pharmacokinetic|pharmacodynamic|mechanism|how\s+does.*work|half-life|absorption|metabolism|elimination)": "pharmacology",
        
        # Interactions
        r"(interaction|drug\s+interaction|food\s+interaction|combine\s+with|take\s+with|interact)": "interactions",
        
        # Structure
        r"(structure|chemical\s+structure|molecular|formula|structural)": "structure",
        
        # Storage
        r"(storage|store|shelf\s+life|expiry|stability|how\s+to\s+store|storage\s+condition)": "storage",
        
        # Overdosage
        r"(overdose|overdosage|too\s+much|excess\s+dose)": "overdosage",
        
        # Composition
        r"(composition|ingredients|contains|active\s+ingredient)": "composition",
        
        # Pregnancy / Nursing
        r"(pregnan|nursing|lactation|breastfeed|fetal|fetus)": "pregnancy",
        
        # Pediatric / Geriatric
        r"(pediatric|child|geriatric|elderly|age)": "special populations",
        
        # Renal / Hepatic impairment
        r"(renal|kidney|hepatic|liver|impairment)": "impairment",
    }
    
    # Image request patterns
    IMAGE_PATTERNS = [
        r"show\s+(?:me\s+)?(?:the\s+)?(?:chemical\s+)?structure",
        r"(?:chemical\s+)?structure\s+(?:of|for|diagram)",
        r"molecular\s+(?:structure|formula|diagram)",
        r"what\s+does\s+.+\s+look\s+like",
        r"image\s+of",
        r"picture\s+of",
        r"diagram"
    ]
    
    # LLM classification prompt (DYNAMIC - no enum restriction)
    LLM_CLASSIFICATION_PROMPT = """You are a medical query classifier.

Given this user query about a drug, extract:
1. DRUG: The drug name mentioned (or UNKNOWN)
2. SECTION: What section of the drug monograph they're asking about

Common section types include (but are NOT limited to):
- indications (what it's used for)
- dosage (how to take it)
- contraindications (when not to use)
- warnings (precautions)
- adverse effects (side effects)
- pharmacology (how it works)
- interactions (drug interactions)
- structure (chemical structure)
- storage (how to store)
- overdosage (overdose info)
- composition (ingredients)
- and any other section found in drug monographs

Query: {query}

Respond in this exact format:
DRUG: [drug name or UNKNOWN]
SECTION: [section type in lowercase, snake_case]"""

    def __init__(self, use_llm_fallback: bool = True):
        """
        Initialize the intent classifier.
        
        Args:
            use_llm_fallback: Whether to use LLM for ambiguous queries
        """
        self.use_llm_fallback = use_llm_fallback
        
        if use_llm_fallback:
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            )
            self.model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-agent")
        
        logger.info(f"IntentClassifier initialized (LLM fallback: {use_llm_fallback})")
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify query intent.
        
        Args:
            query: User's natural language query
            
        Returns:
            QueryIntent with extracted entities
        """
        # Normalize query
        normalized = query.lower().strip()
        
        # Initialize intent
        intent = QueryIntent(
            original_query=query,
            normalized_query=normalized
        )
        
        # Step 1: Check for image request
        intent.needs_image = self._detect_image_request(normalized)
        
        # Step 2: Extract drug name (rule-based)
        drug_name, drug_confidence = self._extract_drug_name(normalized)
        if drug_name:
            intent.target_drug = drug_name
            intent.drug_confidence = drug_confidence
        
        # Step 3: Detect section type (rule-based, DYNAMIC)
        section, section_confidence = self._detect_section(normalized)
        if section:
            intent.target_section = section
            intent.section_confidence = section_confidence
        
        # Step 4: LLM fallback if low confidence
        if self.use_llm_fallback:
            if not intent.target_drug or not intent.target_section:
                llm_intent = self._classify_with_llm(query)
                
                if not intent.target_drug and llm_intent.target_drug:
                    intent.target_drug = llm_intent.target_drug
                    intent.drug_confidence = 0.8
                
                if not intent.target_section and llm_intent.target_section:
                    intent.target_section = llm_intent.target_section
                    intent.section_confidence = 0.8
                
                intent.method = "llm" if not section else "hybrid"
        
        # If image requested, default section to structure
        if intent.needs_image and not intent.target_section:
            intent.target_section = "structure"
            intent.section_confidence = 1.0
        
        logger.info(
            f"Intent classified: drug={intent.target_drug}, "
            f"section={intent.target_section}, image={intent.needs_image}"
        )
        
        return intent
    
    def _detect_image_request(self, query: str) -> bool:
        """Detect if query is requesting an image/structure."""
        for pattern in self.IMAGE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _extract_drug_name(self, query: str) -> Tuple[Optional[str], float]:
        """
        Extract drug name from query using patterns.
        
        Returns:
            Tuple of (drug_name, confidence)
        """
        # Common patterns for drug name extraction
        patterns = [
            # "what is X used for"
            r"what\s+is\s+(\w+)\s+used\s+for",
            # "X contraindications"
            r"(\w+)\s+(?:contraindication|indication|dosage|side\s+effect|interaction)",
            # "side effects of X"
            r"(?:side\s+effect|contraindication|dosage|indication)s?\s+(?:of|for)\s+(\w+)",
            # "tell me about X"
            r"tell\s+me\s+about\s+(\w+)",
            # "information on/about X"
            r"information\s+(?:on|about)\s+(\w+)",
            # General "drug X" or "X drug"
            r"(\w+)\s+drug|drug\s+(\w+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Get the captured group (might be in different positions)
                groups = [g for g in match.groups() if g]
                if groups:
                    drug = groups[0].lower()
                    # Filter out common words
                    if drug not in ['the', 'a', 'an', 'this', 'that', 'what', 'how']:
                        return drug, 0.9
        
        # Fallback: look for capitalized words that might be drug names
        words = query.split()
        for word in words:
            if len(word) > 3 and word[0].isupper():
                return word.lower(), 0.5
        
        return None, 0.0
    
    def _detect_section(self, query: str) -> Tuple[Optional[str], float]:
        """
        Detect which section type the query is asking about.
        
        DYNAMIC: Returns a suggested section name string, not an enum.
        The retrieval engine uses fuzzy matching to find actual sections.
        
        Returns:
            Tuple of (section_name, confidence)
        """
        best_match = None
        best_score = 0.0
        
        for pattern, section_name in self.SECTION_PATTERNS.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Score based on match quality
                score = len(match.group()) / len(query) * 2  # Normalize
                score = min(score, 1.0)
                
                if score > best_score:
                    best_score = score
                    best_match = section_name
        
        if best_match:
            return best_match, max(0.5, best_score)
        
        return None, 0.0
    
    def _classify_with_llm(self, query: str) -> QueryIntent:
        """
        Use LLM for intent classification.
        
        DYNAMIC: No enum restriction - LLM can suggest any section name.
        
        Args:
            query: User query
            
        Returns:
            QueryIntent from LLM classification
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.LLM_CLASSIFICATION_PROMPT.format(query=query)
                    }
                ],
                max_tokens=50,
                temperature=0
            )
            
            result = response.choices[0].message.content
            
            intent = QueryIntent(original_query=query, method="llm")
            
            # Parse response
            for line in result.strip().split('\n'):
                if line.startswith('DRUG:'):
                    value = line[5:].strip().lower()
                    if value and value != 'unknown':
                        intent.target_drug = value
                elif line.startswith('SECTION:'):
                    value = line[8:].strip().lower()
                    if value and value != 'unknown':
                        # DYNAMIC: store section name as-is
                        # Replace spaces with underscores for consistency
                        intent.target_section = value.replace(' ', '_')
            
            return intent
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return QueryIntent(original_query=query)


# Convenience function
def classify_query(query: str) -> QueryIntent:
    """
    Convenience function to classify a query.
    
    Args:
        query: User query
        
    Returns:
        QueryIntent with extracted entities
    """
    classifier = IntentClassifier()
    return classifier.classify(query)
