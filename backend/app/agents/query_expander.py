"""
Query expansion agent for comprehensive retrieval coverage.

This agent expands user queries into multiple related variants to retrieve
information from different angles, significantly improving coverage of PDF content.

Example:
    Input: "Gravol contraindications"
    Output: [
        "Gravol contraindications",
        "When should Gravol not be used",
        "Gravol safety warnings",
        "Gravol precautions",
        "Who cannot take Gravol"
    ]
    
Each variant retrieves different chunks, combined for comprehensive coverage.
"""
import logging
from typing import List, Optional

class QueryExpander:
    """
   Expands queries to retrieve from multiple angles.
    
    Component 2 of Path 1 Enhanced Retrieval.
    Target: Improve coverage from 85% to 90%
    """
    
    def __init__(self, drug_resolver=None):
        """
        Initialize query expander.
        
        Args:
            drug_resolver: Optional drug name resolver for extracting drug names
        """
        self.drug_resolver = drug_resolver
        logging.info("QueryExpander initialized")
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate multiple query variants for comprehensive retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            List of query variants (including original), max 5
        """
        query_lower = query.lower()
        variants = [query]  # Always include original
        
        # Extract drug name
        drug = self._extract_drug_name(query)
        
        # Detect query type and expand accordingly
        if self._is_contraindication_query(query_lower):
            variants.extend(self._expand_contraindications(drug, query_lower))
        
        elif self._is_side_effect_query(query_lower):
            variants.extend(self._expand_side_effects(drug, query_lower))
        
        elif self._is_dosage_query(query_lower):
            variants.extend(self._expand_dosage(drug, query_lower))
        
        elif self._is_interaction_query(query_lower):
            variants.extend(self._expand_interactions(drug, query_lower))
        
        elif self._is_general_info_query(query_lower):
            variants.extend(self._expand_general(drug, query_lower))
        
        elif self._is_mechanism_query(query_lower):
            variants.extend(self._expand_mechanism(drug, query_lower))
        
        # Remove duplicates while preserving order
        variants = list(dict.fromkeys(variants))
        
        # Limit to 5 variants to control costs
        variants = variants[:5]
        
        if len(variants) > 1:
            logging.info(f"Expanded query into {len(variants)} variants")
        else:
            logging.debug(f"Query not expanded (no matching patterns)")
        
        return variants
    
    def _is_contraindication_query(self, query_lower: str) -> bool:
        """Detect contraindication queries."""
        patterns = ['contraindication', 'not use', 'should not', 'cannot take', 
                   'avoid', 'warning', 'precaution']
        return any(p in query_lower for p in patterns)
    
    def _is_side_effect_query(self, query_lower: str) -> bool:
        """Detect side effect queries."""
        patterns = ['side effect', 'adverse', 'reaction', 'undesirable']
        return any(p in query_lower for p in patterns)
    
    def _is_dosage_query(self, query_lower: str) -> bool:
        """Detect dosage queries."""
        patterns = ['dosage', 'dose', 'how much', 'how many', 'administration']
        return any(p in query_lower for p in patterns)
    
    def _is_interaction_query(self, query_lower: str) -> bool:
        """Detect drug interaction queries."""
        patterns = ['interaction', 'combine with', 'take with', 'drug-drug']
        return any(p in query_lower for p in patterns)
    
    def _is_general_info_query(self, query_lower: str) -> bool:
        """Detect general information queries."""
        patterns = [
            'what is', 'tell me about', 'information', 'describe',
            'used for', 'use of', 'indication', 'purpose'
        ]
        return any(p in query_lower for p in patterns)
    
    def _is_mechanism_query(self, query_lower: str) -> bool:
        """Detect mechanism of action queries."""
        patterns = ['mechanism', 'how does', 'how it works', 'pharmacology', 'action']
        return any(p in query_lower for p in patterns)
    
    def _expand_contraindications(self, drug: Optional[str], query_lower: str) -> List[str]:
        """
        Expand contraindication queries.
        
        Example:
            "Gravol contraindications" →
            ["When should Gravol not be used",
             "Gravol safety warnings",
             "Gravol precautions",
             "Who cannot take Gravol"]
        """
        if not drug:
            return []
        
        variants = [
            f"When should {drug} not be used",
            f"{drug} safety warnings",
            f"{drug} precautions",
            f"Who cannot take {drug}"
        ]
        
        # Add population-specific if detected
        if any(word in query_lower for word in ['pregnan', 'child', 'elderly', 'liver', 'kidney']):
            variants.append(f"{drug} warnings and contraindications")
        
        return variants
    
    def _expand_side_effects(self, drug: Optional[str], query_lower: str) -> List[str]:
        """
        Expand side effect queries.
        
        Example:
            "Gravol side effects" →
            ["Gravol adverse reactions",
             "Gravol adverse effects",
             "Gravol undesirable effects",
             "What are side effects of Gravol"]
        """
        if not drug:
            return []
        
        return [
            f"{drug} adverse reactions",
            f"{drug} adverse effects",
            f"{drug} undesirable effects",
            f"What are side effects of {drug}"
        ]
    
    def _expand_dosage(self, drug: Optional[str], query_lower: str) -> List[str]:
        """
        Expand dosage queries.
        
        Example:
            "Gravol dosage" →
            ["Gravol dose",
             "How much Gravol to take",
             "Gravol administration",
             "Gravol dosing"]
        """
        if not drug:
            return []
        
        return [
            f"{drug} dose",
            f"How much {drug} to take",
            f"{drug} administration",
            f"{drug} dosing"
        ]
    
    def _expand_interactions(self, drug: Optional[str], query_lower: str) -> List[str]:
        """
        Expand drug interaction queries.
        
        Example:
            "Gravol interactions" →
            ["Gravol drug interactions",
             "What drugs interact with Gravol",
             "Gravol drug-drug interactions",
             "Can I take Gravol with"]
        """
        if not drug:
            return []
        
        return [
            f"{drug} drug interactions",
            f"What drugs interact with {drug}",
            f"{drug} drug-drug interactions",
            f"Can I take {drug} with"
        ]
    
    def _expand_general(self, drug: Optional[str], query_lower: str) -> List[str]:
        """
        Expand general information queries.
        
        Example:
            "What is Gravol" →
            ["Gravol information",
             "Gravol description",
             "Gravol indications",
             "Tell me about Gravol"]
            
            "What is Gravol used for" →
            ["Gravol indications",
             "Gravol use",
             "Gravol purpose",
             "What is Gravol for"]
        """
        if not drug:
            return []
        
        variants = [
            f"{drug} information",
            f"{drug} description",
            f"{drug} indications",
            f"Tell me about {drug}"
        ]
        
        # If asking about use/purpose, add those variants
        if any(p in query_lower for p in ['used for', 'use of', 'purpose', 'indication']):
            variants.extend([
                f"{drug} use",
                f"{drug} purpose",
                f"What is {drug} for",
                f"{drug} therapeutic use"
            ])
        
        return variants
    
    def _expand_mechanism(self, drug: Optional[str], query_lower: str) -> List[str]:
        """
        Expand mechanism of action queries.
        
        Example:
            "How does Gravol work" →
            ["Gravol mechanism of action",
             "Gravol pharmacology",
             "Gravol action mechanism",
             "Gravol mode of action"]
        """
        if not drug:
            return []
        
        return [
            f"{drug} mechanism of action",
            f"{drug} pharmacology",
            f"{drug} action mechanism",
            f"{drug} mode of action"
        ]
    
    def _extract_drug_name(self, query: str) -> Optional[str]:
        """
        Extract drug name from query.
        
        Uses drug resolver if available, otherwise simple heuristic.
        
        Args:
            query: User query
            
        Returns:
            Drug name if found, None otherwise
        """
        if self.drug_resolver:
            # Use resolver's extraction if available
            try:
                drugs = self.drug_resolver.extract_drug_names(query)
                if drugs:
                    return drugs[0]  # Return first drug found
            except Exception as e:
                logging.debug(f"Drug resolver failed: {e}")
        
        # Fallback: simple heuristic (capitalized words > 3 chars)
        words = query.split()
        for word in words:
            # Remove punctuation
            word_clean = word.strip('.,!?;:')
            if word_clean and word_clean[0].isupper() and len(word_clean) > 3:
                # Likely a drug name
                return word_clean
        
        return None
