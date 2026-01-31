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
    target_attribute: Optional[str] = None  # NEW: Specific attribute (half_life, bioavailability, etc.)
    
    # Flags
    needs_image: bool = False
    
    # Confidence scores
    drug_confidence: float = 0.0
    section_confidence: float = 0.0
    attribute_confidence: float = 0.0  # NEW
    
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
    
    # NEW: Deterministic Map for Attribute-Level Retrieval (Path A++)
    # Maps specific attributes to the sections where they are most likely found.
    # Used for surgical retrieval of facts (e.g., "half-life") without scanning full docs.
    ATTRIBUTE_MAP = {
        # Pharmacokinetics
        "half_life": {
            "keywords": ["half life", "half-life", "elimination half life", "t½", "t1/2"],
            "sections": ["pharmacology", "pharmacokinetics", "clinical pharmacology"]
        },
        "bioavailability": {
            "keywords": ["bioavailability", "systemic availability", "absolute bioavailability"],
            "sections": ["pharmacology", "pharmacokinetics"]
        },
        "absorption": {
            "keywords": ["absorption", "absorbed", "rate of absorption"],
            "sections": ["pharmacology", "pharmacokinetics"]
        },
        "distribution": {
            "keywords": ["distribution", "volume of distribution", "vd"],
            "sections": ["pharmacology", "pharmacokinetics"]
        },
        "protein_binding": {
            "keywords": ["protein binding", "plasma protein binding"],
            "sections": ["pharmacology", "pharmacokinetics"]
        },
        "metabolism": {
            "keywords": ["metabolism", "metabolized", "hepatic metabolism"],
            "sections": ["pharmacology", "pharmacokinetics"]
        },
        "elimination": {
            "keywords": ["elimination", "excretion", "renal excretion", "clearance"],
            "sections": ["pharmacology", "pharmacokinetics"]
        },
        "clearance": {
            "keywords": ["clearance", "cl", "systemic clearance"],
            "sections": ["pharmacology", "pharmacokinetics"]
        },
        "cmax": {
            "keywords": ["cmax", "maximum concentration", "peak plasma concentration"],
            "sections": ["pharmacology", "pharmacokinetics"]
        },
        "tmax": {
            "keywords": ["tmax", "time to peak concentration"],
            "sections": ["pharmacology", "pharmacokinetics"]
        },

        # Dosing & exposure
        "onset_of_action": {
            "keywords": ["onset of action", "onset"],
            "sections": ["pharmacology"]
        },
        "duration_of_action": {
            "keywords": ["duration of action", "duration"],
            "sections": ["pharmacology"]
        },

        # Safety / population attributes
        "pregnancy_risk": {
            "keywords": ["pregnancy", "pregnant", "teratogenic"],
            "sections": ["warnings", "precautions"]
        },
        "lactation": {
            "keywords": ["lactation", "breastfeeding", "nursing mothers"],
            "sections": ["warnings", "precautions"]
        },
        "renal_impairment": {
            "keywords": ["renal impairment", "kidney disease"],
            "sections": ["dosage", "warnings"]
        },
        "hepatic_impairment": {
            "keywords": ["hepatic impairment", "liver disease"],
            "sections": ["dosage", "warnings"]
        },

        # Composition
        "active_ingredient": {
            "keywords": ["active ingredient", "composition", "contains"],
            "sections": ["composition", "description"]
        }
    }
    
    # DYNAMIC section keyword patterns
    # Maps keywords to likely section name patterns (NOT fixed enums)
    SECTION_PATTERNS = {
        # Basic Information / Description (NEW - for generic queries)
        r"(proper\s+name|generic\s+name|brand\s+name|therapeutic\s+class|drug\s+class|classification)": "description",
        
        # Indications / Uses
        r"(used\s+for|treat|use|indication|therapeutic|medical\s+use)": "indications",
        
        # Dosage / Administration
        r"(dosage|dose|how\s+much|administration|administer|how\s+to\s+take|posology|missed\s+dose)": "dosage",
        
        # Contraindications
        r"(contraindication|when\s+not\s+to\s+use|should\s+not|avoid|do\s+not\s+use|forbidden)": "contraindications",
        
        # Warnings / Precautions
        r"(warning|precaution|caution|careful|alert|safety|risk|danger|boxed\s+warning|malignancy|laboratory\s+test|vitamin\s+deficiency)": "warnings",
        
        # Adverse effects / Side effects (ENHANCED - with subsections)
        r"(side\s+effect|adverse\s+(effect|reaction)|reaction|undesirable|toxicity|harmful|commonly\s+reported|frequency|serious|rare|hematologic|hypersensitivity|nervous\s+system)": "adverse",
        
        # Pharmacology
        r"(pharmacology|pharmacokinetic|pharmacodynamic|mechanism|how\s+does.*work|half-life|absorption|metabolism|elimination)": "pharmacology",
        
        # Interactions (ENHANCED - with subtypes)
        r"(interaction|drug\s+interaction|food\s+interaction|herbal\s+interaction|combine\s+with|take\s+with|interact|cytochrome|p450|aspirin|antacid)": "interactions",
        
        # Structure
        r"(structure|chemical\s+structure|molecular|formula|structural)": "structure",
        
        # Storage
        r"(storage|store|shelf\s+life|expiry|stability|how\s+to\s+store|storage\s+condition|disposal|dispose)": "storage",
        
        # Overdosage
        r"(overdose|overdosage|too\s+much|excess\s+dose|dialysis|management)": "overdosage",
        
        # Composition
        r"(composition|ingredients|contains|active\s+ingredient|dosage\s+form|strength|route\s+of\s+administration|manufacturer)": "composition",
        
        # Pregnancy / Nursing
        r"(pregnan|nursing|lactation|breastfeed|fetal|fetus)": "pregnancy",
        
        # Pediatric / Geriatric (ENHANCED)
        r"(pediatric|child|paediatric|geriatric|elderly|age|older\s+adult)": "special populations",
        
        # Renal / Hepatic impairment (ENHANCED)
        r"(renal|kidney|hepatic|liver|impairment|clearance|dose\s+adjust)": "warnings",
        
        # Patient Information (NEW)
        r"(patient\s+information|report\s+immediately|tell\s+your\s+doctor|side\s+effects\s+to\s+report|what\s+to\s+tell)": "patient information",
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
3. ATTRIBUTE: Specific medical attribute (e.g., half_life, bioavailability) if applicable, else UNKNOWN

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
SECTION: [section type in lowercase, snake_case]
ATTRIBUTE: [attribute key or UNKNOWN]"""

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
            
        # Step 2.5: Detect specific attribute (Path A++ Trigger)
        # This takes precedence over generic section detection for attributes
        attribute, attr_confidence = self._detect_attribute(normalized)
        if attribute:
            intent.target_attribute = attribute
            intent.attribute_confidence = attr_confidence
            logger.info(f"Path A++ Trigger detected: attribute={attribute}")
        
        # Step 3: Detect section type (rule-based, DYNAMIC)
        # If attribute detected, map it to sections? 
        # No, let RetrievalEngine handle the mapping. But we can hint.
        section, section_confidence = self._detect_section(normalized)
        if section:
            intent.target_section = section
            intent.section_confidence = section_confidence
        
        # Step 4: LLM fallback if low confidence
        if self.use_llm_fallback:
            if not intent.target_drug or (not intent.target_section and not intent.target_attribute):
                llm_intent = self._classify_with_llm(query)
                
                if not intent.target_drug and llm_intent.target_drug:
                    intent.target_drug = llm_intent.target_drug
                    intent.drug_confidence = 0.8
                
                if not intent.target_section and llm_intent.target_section:
                    intent.target_section = llm_intent.target_section
                    intent.section_confidence = 0.8
                    
                if not intent.target_attribute and llm_intent.target_attribute:
                    intent.target_attribute = llm_intent.target_attribute
                    intent.attribute_confidence = 0.8
                
                intent.method = "llm" if not section else "hybrid"
        
        # If image requested, default section to structure
        if intent.needs_image and not intent.target_section:
            intent.target_section = "structure"
            intent.section_confidence = 1.0
        
        logger.info(
            f"Intent classified: drug={intent.target_drug}, "
            f"section={intent.target_section}, attribute={intent.target_attribute}"
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
        Hybrid drug name extraction: Try regex first, fall back to LLM for complex cases.
        
        Handles:
        - Simple names: "axid", "nizatidine"
        - Hyphenated names: "APO-METOPROLOL", "Co-Trimoxazole"
        - Multi-word names: "Metoprolol Tartrate"
        - Complex names: "St. John's Wort"
        
        Returns:
            Tuple of (drug_name, confidence)
        """
        # Try regex-based extraction first (fast path)
        drug_name, confidence = self._extract_drug_name_regex(query)
        
        # If regex returns low confidence and LLM is available, use it as fallback
        if confidence < 0.7 and self.use_llm_fallback:
            logger.info(f"Regex confidence low ({confidence:.2f}), falling back to LLM")
            return self._extract_drug_name_with_llm(query)
        
        return drug_name, confidence
    
    def _extract_drug_name_regex(self, query: str) -> Tuple[Optional[str], float]:
        """
        Extract drug name from query using regex patterns.
        
        Improved patterns now support hyphens, spaces, and apostrophes in drug names.
        
        Returns:
            Tuple of (drug_name, confidence)
        """
        # Improved patterns with support for hyphens, spaces, apostrophes
        # Non-greedy matching with lookahead to prevent over-capture
        patterns = [
            # Fix for Issue #1: Explicitly handle "all/every/complete X of Y"
            r"(?:all|every|complete|list)\s+(?:of\s+)?(?:contraindication|indication|side\s+effect|dosage)s?\s+(?:of|for)\s+([\w\s'-]+?)(?:\s+(?:and|or)|\?|$)",
            
            # "what are the contraindications of X"
            r"what\s+are\s+the\s+(?:contraindication|indication|side\s+effect|dosage)s?\s+(?:of|for)\s+([\w\s'-]+?)(?:\s+(?:and|or)|\?|$)",
            # "what is X used for"
            r"what\s+is\s+([\w\s'-]+?)\s+(?:used\s+for|and|or|\?|$)",
            # "X contraindications"
            r"([\w\s'-]+?)\s+(?:contraindication|indication|dosage|side\s+effect|interaction)",
            # "side effects of X"
            r"(?:side\s+effect|contraindication|dosage|indication)s?\s+(?:of|for)\s+([\w\s'-]+?)(?:\s+(?:and|or)|\?|$)",
            # "tell me about X"
            r"tell\s+me\s+about\s+([\w\s'-]+?)(?:\s+(?:and|or)|\?|$)",
            # "information on/about X"
            r"information\s+(?:on|about)\s+([\w\s'-]+?)(?:\s+(?:and|or)|\?|$)",
            # General "drug X" or "X drug"
            r"([\w\s'-]+?)\s+drug|drug\s+([\w\s'-]+?)(?:\s|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Get the captured group (might be in different positions)
                groups = [g for g in match.groups() if g]
                if groups:
                    drug = groups[0].strip().lower()
                    # Filter out common words
                    # Fix for Issue #1: Added 'all', 'every', 'complete', 'list' to stopwords
                    if drug not in ['the', 'a', 'an', 'this', 'that', 'what', 'how', 'all', 'every', 'complete', 'list']:
                        logger.debug(f"Regex extracted drug: '{drug}' from query: '{query}'")
                        return drug, 0.9
        
        # Fallback: look for capitalized words that might be drug names
        words = query.split()
        for word in words:
            if len(word) > 3 and word[0].isupper():
                drug = word.lower()
                # Additional filter for capitalized words
                if drug in ['all', 'every', 'complete', 'list']:
                    continue
                    
                logger.debug(f"Regex fallback extracted: '{drug}' (low confidence)")
                return drug, 0.5
        
        return None, 0.0
    
    def _extract_drug_name_with_llm(self, query: str) -> Tuple[Optional[str], float]:
        """
        Use LLM to extract drug name for complex cases regex cannot handle.
        
        This is a fallback method for edge cases like:
        - Multi-word drug names with unusual punctuation
        - Ambiguous queries where drug name is unclear
        
        Returns:
            Tuple of (drug_name, confidence)
        """
        try:
            prompt = f"""Extract the drug name from this query. Return ONLY the drug name, nothing else.

Query: "{query}"

Drug name:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0
            )
            
            drug_name = response.choices[0].message.content.strip().lower()
            
            # Filter out common words that LLM might return
            if drug_name and drug_name not in ['the', 'a', 'an', 'what', 'how', 'when', 'unknown', 'none']:
                logger.info(f"LLM extracted drug: '{drug_name}' from query: '{query}'")
                return drug_name, 0.95  # High confidence for LLM extraction
            
            logger.warning(f"LLM returned invalid drug name: '{drug_name}'")
            return None, 0.0
            
        except Exception as e:
            logger.error(f"LLM drug extraction failed: {e}")
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
    
    def _detect_attribute(self, query: str) -> Tuple[Optional[str], float]:
        """
        Detect specific medical attribute from query.
        
        Uses ATTRIBUTE_MAP for precise keyword matching.
        
        Returns:
            Tuple of (attribute_key, confidence)
        """
        best_match = None
        best_score = 0.0
        
        for attribute_key, data in self.ATTRIBUTE_MAP.items():
            for keyword in data["keywords"]:
                # Check for keyword usage in query
                # Use regex word boundary to avoid partial matches
                # e.g. avoid matching "elimination" in "elimination half-life" if checking for "elimination"
                pattern = r'\b' + re.escape(keyword) + r'\b'
                match = re.search(pattern, query, re.IGNORECASE)
                
                if match:
                    # Found a match
                    # Score based on length ratio to prioritize specific terms
                    # e.g. "elimination half life" > "half life"
                    score = len(keyword) / len(query) * 2
                    score = min(score, 0.95)  # Cap at 0.95 (reserve 1.0 for perfect)
                    
                    if score > best_score:
                        best_score = score
                        best_match = attribute_key
        
        if best_match:
            return best_match, max(0.6, best_score)  # Minimum confidence 0.6 if found
        
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
                    if value and value not in ['unknown', 'none', 'n/a']:
                        intent.target_drug = value
                elif line.startswith('SECTION:'):
                    value = line[8:].strip().lower()
                    if value and value not in ['unknown', 'none', 'n/a']:
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
