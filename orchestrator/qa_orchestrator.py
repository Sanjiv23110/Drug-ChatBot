"""
Complete Regulatory QA Orchestrator
End-to-end pipeline from query to validated answer
"""

from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Classify user intent for routing
    Uses Hybrid Guardrails: Regex (fast) + LLM arbiter (ambiguous)
    """
    
    INTENT_TYPES = ["product_specific", "class_based", "comparative", "out_of_scope"]
    
    def __init__(self, drug_normalizer, azure_endpoint=None, azure_api_key=None, model_name="gpt-4o"):
        self.normalizer = drug_normalizer
        
        # Initialize Azure OpenAI client for LLM arbiter
        import os
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.model_name = model_name
        
        if self.azure_endpoint and self.azure_api_key:
            from openai import AzureOpenAI
            self.llm_client = AzureOpenAI(
                api_key=self.azure_api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=self.azure_endpoint
            )
            logger.info("IntentClassifier: LLM arbiter enabled")
        else:
            self.llm_client = None
            logger.warning("IntentClassifier: LLM arbiter disabled (no Azure credentials)")
    
    
    def classify(self, query: str) -> Tuple[str, Dict]:
        """
        Classify query intent using Hybrid Guardrails
        Tier 1: Regex (fast path for obvious cases)
        Tier 2: LLM arbiter (for ambiguous cases)
        
        Returns:
            (intent_type, metadata)
        """
        query_lower = query.lower()
        
        # TIER 1: REGEX FAST PATH
        # Check for OBVIOUSLY blocked patterns (high confidence)
        is_advice_certain = self._is_medical_advice_certain(query_lower)
        is_patient_certain = self._is_patient_specific_certain(query_lower)
        
        if is_advice_certain or is_patient_certain:
            reason = "medical_advice" if is_advice_certain else "patient_specific"
            refusal = self._get_refusal_text(reason)
            logger.info(f"Blocked by Tier 1 (Regex): {reason}")
            return "out_of_scope", {
                "reason": reason,
                "refusal": refusal,
                "guardrail_tier": "regex"
            }
        
        # Check for OBVIOUSLY allowed patterns (known drug + known section)
        if self._is_safe_labeling_query(query_lower):
            logger.info("Allowed by Tier 1 (Regex): safe labeling query")
            # Continue to normal routing (product_specific, class, etc.)
            return self._route_safe_query(query)
        
        # TIER 2: LLM ARBITER (ambiguous cases)
        if self.llm_client:
            logger.info("Tier 1 inconclusive, escalating to LLM arbiter...")
            llm_verdict = self._llm_arbiter(query)
            
            if llm_verdict == "ADVICE":
                logger.info("Blocked by Tier 2 (LLM): medical advice detected")
                return "out_of_scope", {
                    "reason": "medical_advice",
                    "refusal": self._get_refusal_text("medical_advice"),
                    "guardrail_tier": "llm"
                }
            elif llm_verdict == "LABELING":
                logger.info("Allowed by Tier 2 (LLM): labeling query")
                return self._route_safe_query(query)
        
        # Fallback: if no LLM or inconclusive, use legacy strict regex
        logger.warning("Falling back to legacy regex (strict)")
        if self._is_medical_advice(query_lower):
            return "out_of_scope", {
                "reason": "medical_advice",
                "refusal": self._get_refusal_text("medical_advice"),
                "guardrail_tier": "regex_fallback"
            }
        
        if self._is_patient_specific(query_lower):
            return "out_of_scope", {
                "reason": "patient_specific",
                "refusal": self._get_refusal_text("patient_specific"),
                "guardrail_tier": "regex_fallback"
            }
        
        # Not blocked, route normally
        return self._route_safe_query(query)
    
    def _route_safe_query(self, query: str) -> Tuple[str, Dict]:
        """Route query to appropriate handler after safety check"""
        # Check for comparative
        if self._is_comparative(query.lower()):
            return "comparative", {}
        
        # Check for class-based
        if self.normalizer.is_class_query(query):
            return "class_based", {}
        
        # Default: product-specific
        return "product_specific", {}
    
    def _llm_arbiter(self, query: str) -> str:
        """
        Use LLM to determine if query is LABELING or ADVICE
        Returns: "LABELING" or "ADVICE" or "UNCLEAR"
        """
        prompt = f"""You are a safety classifier for a medical information system.

Query: "{query}"

Is this query asking for:
A) General information about FDA drug labeling (dosage, contraindications, warnings, etc.)
B) Personal medical advice (should I take, is it safe for me specifically, treatment recommendations)

Rules:
- If the query uses "I", "me", "my" in a personal context (e.g., "Should I take..."), classify as ADVICE.
- If the query asks about general labeling info for healthcare professionals or general knowledge, classify as LABELING.
- Pharmacists asking about labeling for their patients = LABELING.

Respond with ONLY one word: LABELING or ADVICE"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical query safety classifier. Respond with only one word."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            verdict = response.choices[0].message.content.strip().upper()
            logger.info(f"LLM arbiter verdict: {verdict}")
            
            if "LABELING" in verdict:
                return "LABELING"
            elif "ADVICE" in verdict:
                return "ADVICE"
            else:
                logger.warning(f"LLM arbiter returned unclear verdict: {verdict}")
                return "UNCLEAR"
                
        except Exception as e:
            logger.error(f"LLM arbiter failed: {e}")
            return "UNCLEAR"
    
    def _is_medical_advice_certain(self, query: str) -> bool:
        """HIGH CONFIDENCE medical advice patterns (Tier 1 block)"""
        certain_patterns = [
            "should i take", "should i stop", "should i start",
            "can i take", "prescribe me", "recommend me",
            "what should i do", "is it safe for me to"
        ]
        return any(pattern in query for pattern in certain_patterns)
    
    def _is_patient_specific_certain(self, query: str) -> bool:
        """HIGH CONFIDENCE patient-specific patterns (Tier 1 block)"""
        import re
        certain_patterns = [
            r'\bfor me\b', r'\bin my case\b', r'\bi am taking\b',
            r'\bmy doctor said\b', r'\bmy symptoms\b'
        ]
        return any(re.search(pattern, query) for pattern in certain_patterns)
    
    def _is_safe_labeling_query(self, query: str) -> bool:
        """HIGH CONFIDENCE safe labeling query (Tier 1 allow)"""
        # Simple heuristic: contains a medical term AND no first-person pronouns
        labeling_terms = [
            'dosage', 'contraindication', 'warning', 'precaution',
            'adverse reaction', 'side effect', 'interaction',
            'indication', 'mechanism'
        ]
        
        first_person = ['i ', 'me ', 'my ', 'mine ', "i'm", "i've"]
        
        has_labeling_term = any(term in query for term in labeling_terms)
        has_first_person = any(fp in query for fp in first_person)
        
        return has_labeling_term and not has_first_person
    
    def _get_refusal_text(self, reason: str) -> str:
        """Get standardized refusal text"""
        refusals = {
            "medical_advice": "This system provides FDA labeling excerpts only, not medical advice. Consult a healthcare provider.",
            "patient_specific": "Patient-specific decisions require clinical evaluation. Consult a healthcare provider."
        }
        return refusals.get(reason, "Unable to process this query.")
    
    def _is_medical_advice(self, query: str) -> bool:
        """Detect medical advice requests (legacy fallback)"""
        advice_patterns = [
            "should i take", "should i use", "should i stop",
            "can i take", "is it safe for me", "what should i do",
            "which drug is better", "which medication",
            "suggest", "prescribe"
        ]
        return any(pattern in query for pattern in advice_patterns)
    
    def _is_patient_specific(self, query: str) -> bool:
        """Detect patient-specific questions"""
        import re
        patient_patterns = [
            r'\bfor my\b', r'\bfor me\b', r'\bin my case\b',
            r'\bi am taking\b', r'\bi have\b', r'\bmy doctor\b',
            r'\bcan i\b', r'\bshould i\b'
        ]
        return any(re.search(pattern, query) for pattern in patient_patterns)
    
    def _is_comparative(self, query: str) -> bool:
        """Detect comparative questions"""
        comparative_patterns = [
            r'compare .* (vs|versus|and)',
            r'difference between .* and',
            r'which is better'
        ]
        import re
        return any(re.search(pattern, query) for pattern in comparative_patterns)


class SectionClassifier:
    """
    Map natural language question to LOINC section
    """
    
    SECTION_MAPPING = {
        # Existing high-frequency mappings
        "adverse reactions": "34084-4",
        "side effects": "34084-4",
        "contraindications": "34070-3",
        "warnings": "43685-7",
        "precautions": "43685-7",
        "interactions": "34073-7",
        "drug interactions": "34073-7",
        "dosage": "34068-7",
        "dosing": "34068-7",
        "administration": "34068-7",
        "indications": "34067-9",
        "uses": "34067-9",
        "overdose": "34088-5",
        "laboratory tests": "34075-2",
        "mechanism": "43678-2",
        
        # Complete FDA LOINC Coverage (121 codes)
        "abuse": "34086-9",
        "accessories": "60555-0",
        "alarms": "69761-5",
        "animal pharmacology": "34091-9",
        "toxicology": "34091-9",
        "assembly": "60556-8",
        "installation": "60556-8",
        "boxed warning": "34066-1",
        "black box": "34066-1",
        "calibration": "60557-6",
        "carcinogenesis": "34083-6",
        "mutagenesis": "34083-6",
        "impairment of fertility": "34083-6",
        "clinical pharmacology": "34090-1",
        "cleaning": "60558-4",
        "disinfecting": "60558-4",
        "sterilization": "60558-4",
        "clinical studies": "34092-7",
        "clinical trials": "90374-0",
        "compatible accessories": "69760-7",
        "components": "60559-2",
        "controlled substance": "34085-1",
        "dependence": "34087-7",
        "description": "34089-3",
        "diagram": "69758-1",
        "disposal": "69763-1",
        "waste handling": "69763-1",
        "dosage forms": "43678-2",
        "strengths": "43678-2",
        "laboratory test interactions": "34074-5",
        "drug abuse": "42227-9",
        "environmental warning": "50742-6",
        "reproductive potential": "77291-3",
        "food safety": "50743-4",
        "general precautions": "34072-9",
        "geriatric": "34082-8",
        "elderly": "34082-8",
        "guaranteed analysis": "50740-0",
        "health care provider letter": "71744-7",
        "health claim": "69719-3",
        "hepatic impairment": "88829-7",
        "liver": "88829-7",
        "how supplied": "34069-5",
        "immunogenicity": "88830-5",
        "inactive ingredient": "51727-6",
        "information for owners": "50744-2",
        "caregivers": "50744-2",
        "information for patients": "34076-0",
        "patient information": "34076-0",
        "instructions for use": "59845-8",
        "intended use": "60560-0",
        "labor": "34079-4",
        "delivery": "34079-4",
        "lactation": "77290-5",
        "breastfeeding": "77290-5",
        "mechanism of action": "43679-0",
        "microbiology": "49489-8",
        "nonclinical toxicology": "43680-8",
        "nonteratogenic": "34078-6",
        "nursing mothers": "34080-2",
        "other safety information": "60561-8",
        "overdosage": "34088-5",
        "otc active ingredient": "55106-9",
        "otc ask doctor": "50569-3",
        "otc do not use": "50570-1",
        "otc keep out of reach": "50565-1",
        "otc pregnancy": "53414-9",
        "otc purpose": "55105-1",
        "otc questions": "53413-1",
        "otc stop use": "50566-9",
        "otc when using": "50567-7",
        "package label": "51945-4",
        "principal display panel": "51945-4",
        "patient counseling": "88436-1",
        "patient medication information": "68498-5",
        "pediatric": "34081-0",
        "children": "34081-0",
        "pharmacodynamics": "43681-6",
        "pharmacogenomics": "66106-6",
        "pharmacokinetics": "43682-4",
        "postmarketing": "90375-7",
        "post-marketing": "90375-7",
        "pregnancy": "42228-7",
        "recent major changes": "43683-2",
        "references": "34093-5",
        "residue warning": "53412-3",
        "rems": "100382-1",
        "rems administrative": "87523-7",
        "rems applicant": "87526-0",
        "rems communication": "82344-3",
        "rems elements": "82348-4",
        "rems safe use": "82345-0",
        "rems goals": "82349-2",
        "rems implementation": "82350-0",
        "rems material": "82346-8",
        "rems medication guide": "82598-4",
        "rems participant": "87525-2",
        "rems requirements": "87524-5",
        "rems summary": "82347-6",
        "rems timetable": "82352-6",
        "renal impairment": "88828-9",
        "kidney": "88828-9",
        "risks": "69759-9",
        "route": "60562-6",
        "method of administration": "60562-6",
        "frequency": "60562-6",
        "safe handling": "50741-8",
        "spl indexing": "48779-3",
        "spl product data": "48780-1",
        "medguide": "42231-1",
        "medication guide": "42231-1",
        "patient package insert": "42230-3",
        "spl unclassified": "42229-5",
        "statement of identity": "69718-5",
        "storage": "44425-7",
        "handling": "44425-7",
        "safety and effectiveness": "60563-4",
        "teratogenic": "34077-8",
        "troubleshooting": "69762-3",
        "specific populations": "43684-0",
        "user safety warnings": "54433-8",
        "veterinary indications": "50745-9",
        "warnings and precautions": "43685-7"
    }
    
    def classify(self, query: str) -> Optional[str]:
        """
        Identify LOINC section code from query
        
        Returns:
            LOINC code or None
        """
        query_lower = query.lower()
        
        for keyword, loinc in self.SECTION_MAPPING.items():
            if keyword in query_lower:
                logger.info(f"Mapped '{keyword}' → LOINC {loinc}")
                return loinc
        
        logger.warning("Could not map query to specific section")
        return None


class RegulatoryQAOrchestrator:
    """
    Complete end-to-end QA system
    Orchestrates all components: normalization, retrieval, generation, validation
    """
    
    def __init__(
        self,
        drug_normalizer,
        retriever,
        generator,
        audit_log_path: str = "pharma_qa_audit.jsonl"
    ):
        """
        Args:
            drug_normalizer: DrugNormalizer instance
            retriever: HybridRetriever instance
            generator: RegulatoryQAGenerator instance
            audit_log_path: Path to audit log file
        """
        self.normalizer = drug_normalizer
        self.retriever = retriever
        self.generator = generator
        self.audit_log_path = audit_log_path
        
        self.intent_classifier = IntentClassifier(drug_normalizer)
        self.section_classifier = SectionClassifier()
        
        logger.info("Initialized RegulatoryQAOrchestrator")
    
    def query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Process complete query from start to finish
        
        Returns:
            {
                "answer": str,
                "status": str,
                "metadata": {...},
                "timestamp": str
            }
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Step 1: Intent Classification
        intent, intent_metadata = self.intent_classifier.classify(query)
        logger.info(f"Intent: {intent}")
        
        if intent == "out_of_scope":
            return self._create_refusal_response(
                query=query,
                reason=intent_metadata['reason'],
                refusal_text=intent_metadata['refusal'],
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id
            )
        
        # Step 2: Route to appropriate handler
        if intent == "product_specific":
            result = self._handle_product_specific(query)
        elif intent == "class_based":
            result = self._handle_class_based(query)
        elif intent == "comparative":
            result = self._handle_comparative(query)
        else:
            result = self._create_refusal_response(
                query=query,
                reason="unknown_intent",
                refusal_text="Unable to process this query.",
                timestamp=timestamp
            )
        
        # Add timestamp
        result['timestamp'] = timestamp
        
        # Step 3: Audit logging
        self._log_query(
            query=query,
            result=result,
            intent=intent,
            user_id=user_id,
            session_id=session_id
        )
        
        return result
    
    def _handle_product_specific(self, query: str) -> Dict:
        """Handle product-specific query"""
        
        # Extract drug name (simple approach - can be improved with NER)
        drug_name = self._extract_drug_name(query)
        
        if not drug_name:
            return self._create_refusal_response(
                query=query,
                reason="drug_not_found",
                refusal_text="Could not identify drug name in query."
            )
        
        # Normalize drug name to RxCUI
        drug_info = self.normalizer.normalize_drug_name(drug_name)
        
        if not drug_info:
            return self._create_refusal_response(
                query=query,
                reason="drug_not_found",
                refusal_text=f"No FDA-approved labeling found for '{drug_name}'."
            )
        
        rxcui = drug_info['rxcui']
        
        # Classify section
        loinc_code = self.section_classifier.classify(query)
        
        if not loinc_code:
            # If no specific section, retrieve from all sections
            filter_conditions = {"rxcui": rxcui}
        else:
            filter_conditions = {
                "rxcui": rxcui,
                "loinc_code": loinc_code
            }
        
        # Retrieve chunks
        retrieved_chunks, retrieval_metadata = self.retriever.retrieve(
            query=query,
            filter_conditions=filter_conditions,
            retrieval_limit=50,
            rerank_top_k=15
        )
        
        if not retrieved_chunks:
            section_name = SectionClassifier.SECTION_MAPPING.get(loinc_code, "requested section")
            return self._create_refusal_response(
                query=query,
                reason="section_not_found",
                refusal_text=f"The {section_name} section was not found for {drug_name}."
            )
        
        # Generate answer
        result = self.generator.generate_answer(query, retrieved_chunks)
        
        return result
    
    def _handle_class_based(self, query: str) -> Dict:
        """Handle class-based query (e.g., 'ACE inhibitors')"""
        
        # Extract class term
        class_term = self._extract_class_term(query)
        
        # KEY FIX: If query mentions a specific drug (e.g., "diuretics like Renese"), 
        # treat as product-specific despite class term.
        drug_name = self._extract_drug_name(query)
        if drug_name:
            logger.info(f"Class query '{query}' contains drug '{drug_name}' -> Rerouting to product handler")
            return self._handle_product_specific(query)
        
        if not class_term:
            return self._create_refusal_response(
                query=query,
                reason="class_not_found",
                refusal_text="Could not identify drug class in query."
            )
        
        # Expand class to RxCUIs
        rxcui_list = self.normalizer.expand_class_to_drugs(class_term)
        
        if not rxcui_list:
            return self._create_refusal_response(
                query=query,
                reason="class_not_found",
                refusal_text=f"No drugs found for class '{class_term}'."
            )
        
        logger.info(f"Expanded '{class_term}' to {len(rxcui_list)} drugs")
        
        # Classify section
        loinc_code = self.section_classifier.classify(query)
        
        filter_conditions = {
            "rxcui": rxcui_list,  # Match any of these
        }
        
        if loinc_code:
            filter_conditions["loinc_code"] = loinc_code
        
        # Retrieve chunks (more candidates for class queries)
        retrieved_chunks, retrieval_metadata = self.retriever.retrieve(
            query=query,
            filter_conditions=filter_conditions,
            retrieval_limit=100,
            rerank_top_k=10
        )
        
        if not retrieved_chunks:
            return self._create_refusal_response(
                query=query,
                reason="no_evidence",
                refusal_text=f"No evidence found for {class_term}."
            )
        
        # Generate answer
        result = self.generator.generate_answer(query, retrieved_chunks)
        
        return result
    
    def _handle_comparative(self, query: str) -> Dict:
        """Handle comparative queries"""
        # Comparative queries are complex and may require special handling
        # For now, refuse
        return self._create_refusal_response(
            query=query,
            reason="comparative_not_supported",
            refusal_text="Comparative queries are not currently supported. Please ask about one drug at a time."
        )
    
    def _extract_drug_name(self, query: str) -> Optional[str]:
        """
        Simple drug name extraction
        In production, use NER (scispacy, BioBERT NER, etc.)
        """
        import string
        
        # Common English words AND medical terminology to skip (stopwords)
        # These words appear in queries but are NOT drug names
        stopwords = {
            # Basic English stopwords
            'what', 'are', 'the', 'is', 'of', 'for', 'in', 'to', 'and', 'or',
            'with', 'about', 'from', 'that', 'this', 'these', 'those', 'can',
            'should', 'would', 'could', 'will', 'may', 'might', 'must', 'have',
            'has', 'had', 'do', 'does', 'did', 'be', 'been', 'being', 'was',
            'were', 'am', 'is', 'are', 'a', 'an', 'the', 'some', 'any', 'all',
            'tell', 'me', 'you', 'your', 'my', 'their', 'his', 'her', 'its', 'our',
            'how', 'why', 'when', 'where', 'which', 'who', 'whom', 'whose',
            'take', 'taken', 'taking', 'give', 'given', 'giving', 'use', 'used', 'using',
            
            # Drug label section terms
            'most', 'common', 'adverse', 'reactions', 'side', 'effects', 'dosage',
            'dose', 'indication', 'indications', 'contraindication', 'contraindications',
            'warning', 'warnings', 'precaution', 'precautions', 'interaction', 'interactions',
            'types', 'type', 'indicated', 'treat', 'treatment', 'therapy',
            'available', 'strengths', 'special', 'cardiovascular', 'mortality',
            'laboratory', 'tests', 'inactive', 'ingredients', 'active',
            'biological', 'half', 'life', 'mechanism', 'action', 'clinical',
            'pharmacology', 'pharmacokinetics', 'overdosage', 'overdose',
            'storage', 'handling', 'description', 'supplied', 'packaging',
            
            # Body systems / medical terms (NOT drug names)
            'gastrointestinal', 'cardiovascular', 'respiratory', 'renal', 'hepatic',
            'nervous', 'central', 'peripheral', 'dermatologic', 'hematologic',
            'endocrine', 'metabolic', 'musculoskeletal', 'genitourinary',
            'electrolyte', 'electrolytes', 'imbalance', 'imbalances',
            
            # Patient/condition terms
            'patients', 'patient', 'pediatric', 'geriatric', 'adult', 'adults',
            'children', 'elderly', 'pregnant', 'pregnancy', 'nursing', 'mothers',
            'diabetic', 'diabetics', 'hypertensive', 'renal', 'hepatic',
            'disease', 'disorder', 'condition', 'conditions', 'susceptible',
            'predispose', 'predisposes', 'monitored', 'safe', 'safety',
            'recommended', 'recommend', 'recommendation', 'recommendations',
            'suggested', 'suggest', 'suggestion', 'suggestions',
            'prescribed', 'prescribe', 'prescription', 'prescriptions',
            'administered', 'administer', 'administration', 'administrations',
            'inject', 'injected', 'injection', 'injections',
            'daily', 'weekly', 'monthly', 'yearly', 'once', 'twice', 'thrice',
            'times', 'day', 'week', 'month', 'year', 'hour', 'hours', 'minute', 'minutes',
            
            # Common verbs/adjectives
            'can', 'could', 'should', 'would', 'will', 'may', 'might', 'must',
            'known', 'unknown', 'possible', 'potential', 'likely', 'unlikely',
            'severe', 'mild', 'moderate', 'serious', 'life-threatening',
            'fatal', 'death', 'die', 'dying', 'kill', 'killing', 'kills',
            'cause', 'causes', 'causing', 'caused', 'result', 'results', 'resulting',
            'associated', 'associate', 'association', 'associations',
            'related', 'relate', 'relation', 'relations', 'relationship', 'relationships',
            'include', 'includes', 'including', 'included', 'contain', 'contains', 'containing',
            'consist', 'consists', 'consisting', 'composed', 'composes', 'composing',
            'made', 'make', 'makes', 'making', 'form', 'forms', 'forming',
            'table', 'tables', 'chart', 'charts', 'list', 'lists', 'listed',
            'section', 'sections', 'part', 'parts', 'page', 'pages', 'paragraph',
            'text', 'texts', 'sentence', 'sentences', 'word', 'words', 'phrase',
            'information', 'info', 'detail', 'details', 'detailed', 'summary',
            'summarize', 'summarized', 'explain', 'explained', 'explanation',
            'describe', 'described', 'description', 'descriptions', 'define',
            
            # Common query patterns
            'affect', 'affects', 'requirements', 'routinely', 'diuretics',
            'tablets', 'capsules', 'injection', 'solution', 'oral'
        }
        
        # Very simple approach: capitalize words and check with RxNorm
        # Robust N-gram approach (Sliding Window)
        # Try 3-grams, then 2-grams, then 1-gram to catch multi-word drugs like "Meperidine Hydrochloride"
        import string
        words = query.split()
        num_words = len(words)
        max_gram = 3
        
        for gram_len in range(max_gram, 0, -1):
            for i in range(num_words - gram_len + 1):
                phrase_list = words[i : i + gram_len]
                phrase = " ".join(phrase_list)
                
                clean_phrase = phrase.strip(string.punctuation)
                if not clean_phrase: continue
                if clean_phrase.lower() in stopwords: continue
                
                # Check RxNorm
                rxcui = self.normalizer.rxnorm.get_rxcui_from_name(clean_phrase)
                if rxcui:
                    return clean_phrase
                
                if not clean_phrase[0].isupper():
                    rxcui = self.normalizer.rxnorm.get_rxcui_from_name(clean_phrase.capitalize())
                    if rxcui:
                        return clean_phrase.capitalize()
                        
                if not clean_phrase.islower():
                    rxcui = self.normalizer.rxnorm.get_rxcui_from_name(clean_phrase.lower())
                    if rxcui:
                        return clean_phrase.lower()
        return None
    
    def _extract_class_term(self, query: str) -> Optional[str]:
        """Extract drug class term from query"""
        # Simple pattern matching
        import re
        
        patterns = [
            r'(ACE inhibitors?)',
            r'(beta blockers?)',
            r'(statins?)',
            r'(diuretics?)',
            r'(ARBs?)',
            r'(SSRIs?)',
            r'(NSAIDs?)',
            r'(PPIs?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _create_refusal_response(
        self,
        query: str,
        reason: str,
        refusal_text: str,
        timestamp: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """Create standardized refusal response"""
        return {
            "answer": refusal_text,
            "status": "refused",
            "reason": reason,
            "metadata": {},
            "validation_score": 0.0,
            "timestamp": timestamp or datetime.utcnow().isoformat()
        }
    
    def _log_query(
        self,
        query: str,
        result: Dict,
        intent: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Write query to audit log"""
        audit_entry = {
            "timestamp": result.get('timestamp'),
            "query": query,
            "intent": intent,
            "answer": result.get('answer'),
            "status": result.get('status'),
            "validation_score": result.get('validation_score', 0.0),
            "metadata": result.get('metadata', {}),
            "user_id": user_id,
            "session_id": session_id
        }
        
        try:
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")


# Example usage
if __name__ == "__main__":
    from normalization.rxnorm_integration import DrugNormalizer
    from retrieval.hybrid_retriever import HybridRetriever, DenseEmbedder, CrossEncoderReranker
    from generation.constrained_extractor import ConstrainedExtractor, PostGenerationValidator, RegulatoryQAGenerator
    from vector_db.qdrant_manager import QdrantManager
    
    # Initialize all components
    logger.info("Initializing components...")
    
    normalizer = DrugNormalizer()
    
    dense_embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
    vector_db = QdrantManager(host="localhost", port=6333)
    
    retriever = HybridRetriever(
        dense_embedder=dense_embedder,
        sparse_embedder=None,
        reranker=reranker,
        vector_db_manager=vector_db
    )
    
    extractor = ConstrainedExtractor(model_name="gpt-4o")
    validator = PostGenerationValidator(similarity_threshold=75)
    generator = RegulatoryQAGenerator(extractor=extractor, validator=validator)
    
    # Create orchestrator
    orchestrator = RegulatoryQAOrchestrator(
        drug_normalizer=normalizer,
        retriever=retriever,
        generator=generator
    )
    
    # Example queries
    queries = [
        "What are the adverse reactions of Lisinopril?",
        "What are contraindications of ACE inhibitors?",
        "Should I take Lisinopril for my high blood pressure?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result = orchestrator.query(query)
        
        print(f"Status: {result['status']}")
        print(f"Answer: {result['answer']}")
        if result.get('metadata'):
            print(f"Source: {result['metadata'].get('loinc_section')}")
            print(f"Validation: {result.get('validation_score', 0):.1f}%")
