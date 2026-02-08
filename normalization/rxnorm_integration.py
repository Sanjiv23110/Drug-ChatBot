"""
RxNorm and RxClass Integration
Semantic normalization and drug class expansion for regulatory QA system
"""

import requests
from typing import List, Dict, Optional
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RxNormClient:
    """
    Interface to RxNorm REST API (free, no authentication required)
    https://rxnav.nlm.nih.gov/
    """
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def __init__(self):
        self.session = requests.Session()
        logger.info("Initialized RxNorm client")

    def _get_rxcui_approximate(self, drug_name: str) -> Optional[str]:
        """Fallback validation using approximate match"""
        url = f"{self.BASE_URL}/approximateTerm.json"
        params = {"term": drug_name, "maxEntries": 1}
        
        try:
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if 'approximateGroup' in data and 'candidate' in data['approximateGroup']:
                candidates = data['approximateGroup']['candidate']
                if candidates:
                    rxcui = candidates[0].get('rxcui')
                    logger.info(f"Resolved '{drug_name}' via approximate match → RxCUI: {rxcui}")
                    return rxcui
            
            return None
            
        except Exception as e:
            logger.warning(f"RxNorm approximate search error for '{drug_name}': {e}")
            return None

    
    @lru_cache(maxsize=10000)
    def get_rxcui_from_name(self, drug_name: str) -> Optional[str]:
        """
        Convert drug name to RxCUI
        
        Args:
            drug_name: Drug name (generic or brand)
            
        Returns:
            RxCUI string or None if not found
        """
        url = f"{self.BASE_URL}/rxcui.json"
        params = {"name": drug_name}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                rxcui = data['idGroup']['rxnormId'][0]
                logger.info(f"Resolved '{drug_name}' → RxCUI: {rxcui}")
                return rxcui
            
            logger.warning(f"No exact RxCUI found for '{drug_name}', trying approximate...")
            return self._get_rxcui_approximate(drug_name)
            
        except Exception as e:
            logger.error(f"RxNorm API error for '{drug_name}': {e}")
            return None
    
    @lru_cache(maxsize=5000)
    def get_rxcui_from_ndc(self, ndc: str) -> Optional[str]:
        """
        Convert NDC code to RxCUI
        
        Args:
            ndc: NDC code (10 or 11 digits)
            
        Returns:
            RxCUI string or None
        """
        # Normalize NDC (remove hyphens)
        ndc_normalized = ndc.replace('-', '')
        
        url = f"{self.BASE_URL}/ndcstatus.json"
        params = {"ndc": ndc_normalized}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'ndcStatus' in data and 'rxcui' in data['ndcStatus']:
                rxcui = data['ndcStatus']['rxcui']
                logger.info(f"Resolved NDC '{ndc}' → RxCUI: {rxcui}")
                return rxcui
            
            logger.warning(f"No RxCUI found for NDC '{ndc}'")
            return None
            
        except Exception as e:
            logger.error(f"RxNorm API error for NDC '{ndc}': {e}")
            return None
    
    @lru_cache(maxsize=10000)
    def get_drug_name(self, rxcui: str) -> Optional[str]:
        """
        Get drug name from RxCUI
        
        Args:
            rxcui: RxCUI identifier
            
        Returns:
            Drug name or None
        """
        url = f"{self.BASE_URL}/rxcui/{rxcui}/property.json"
        params = {"propName": "RxNorm Name"}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'propConceptGroup' in data:
                properties = data['propConceptGroup'].get('propConcept', [])
                if properties:
                    name = properties[0].get('propValue')
                    return name
            
            return None
            
        except Exception as e:
            logger.error(f"RxNorm API error for RxCUI '{rxcui}': {e}")
            return None
    
    @lru_cache(maxsize=5000)
    def get_related_drugs(self, rxcui: str, relation: str = "SCD") -> List[str]:
        """
        Get related RxCUIs (e.g., different formulations)
        
        Args:
            rxcui: RxCUI identifier
            relation: Relation type (SCD, SCDG, IN, etc.)
            
        Returns:
            List of related RxCUIs
        """
        url = f"{self.BASE_URL}/rxcui/{rxcui}/related.json"
        params = {"rela": relation}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            related = []
            if 'relatedGroup' in data:
                for group in data['relatedGroup'].get('conceptGroup', []):
                    for concept in group.get('conceptProperties', []):
                        related.append(concept['rxcui'])
            
            return related
            
        except Exception as e:
            logger.error(f"RxNorm API error for related drugs '{rxcui}': {e}")
            return []


class RxClassClient:
    """
    Interface to RxClass API for drug class information
    Enables class-based queries (e.g., "ACE inhibitors")
    """
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    def __init__(self):
        self.session = requests.Session()
        logger.info("Initialized RxClass client")
    
    @lru_cache(maxsize=10000)
    def get_drug_classes(self, rxcui: str) -> List[Dict[str, str]]:
        """
        Get pharmacologic and therapeutic classes for a drug
        
        Args:
            rxcui: RxCUI identifier
            
        Returns:
            List of class dictionaries:
            [
                {"class_name": "ACE Inhibitors", "class_type": "PE"},
                {"class_name": "Cardiovascular Agents", "class_type": "EPC"}
            ]
        """
        url = f"{self.BASE_URL}/rxclass/class/byRxcui.json"
        params = {"rxcui": rxcui}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            classes = []
            if 'rxclassDrugInfoList' in data:
                for item in data['rxclassDrugInfoList'].get('rxclassDrugInfo', []):
                    class_info = item.get('rxclassMinConceptItem', {})
                    classes.append({
                        'class_name': class_info.get('className', ''),
                        'class_type': class_info.get('classType', ''),
                        'class_id': class_info.get('classId', '')
                    })
            
            logger.info(f"Retrieved {len(classes)} classes for RxCUI {rxcui}")
            return classes
            
        except Exception as e:
            logger.error(f"RxClass API error for RxCUI '{rxcui}': {e}")
            return []
    
    @lru_cache(maxsize=1000)
    def get_drugs_in_class(self, class_name: str, class_type: str = "MESHPA") -> List[str]:
        """
        Get all drugs in a therapeutic/pharmacologic class
        
        Args:
            class_name: Class name (e.g., "ACE Inhibitors")
            class_type: Class type (MESHPA, PE, EPC, etc.)
            
        Returns:
            List of RxCUIs in this class
        """
        # First, find the class ID
        url = f"{self.BASE_URL}/rxclass/class/byName.json"
        params = {"className": class_name, "classTypes": class_type}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'rxclassMinConceptList' not in data:
                logger.warning(f"Class '{class_name}' not found")
                return []
            
            # Get first matching class
            classes = data['rxclassMinConceptList'].get('rxclassMinConcept', [])
            if not classes:
                return []
            
            class_id = classes[0]['classId']
            
            # Get drugs in this class
            url2 = f"{self.BASE_URL}/rxclass/classMembers.json"
            params2 = {"classId": class_id, "relaSource": "MESHPA"}
            
            response2 = self.session.get(url2, params=params2, timeout=10)
            response2.raise_for_status()
            data2 = response2.json()
            
            rxcuis = []
            if 'drugMemberGroup' in data2:
                for member in data2['drugMemberGroup'].get('drugMember', []):
                    rxcui = member.get('minConcept', {}).get('rxcui')
                    if rxcui:
                        rxcuis.append(rxcui)
            
            logger.info(f"Found {len(rxcuis)} drugs in class '{class_name}'")
            return rxcuis
            
        except Exception as e:
            logger.error(f"RxClass API error for class '{class_name}': {e}")
            return []
    
    def expand_class_query(self, class_term: str) -> List[str]:
        """
        Expand a class term to list of RxCUIs
        Tries multiple class types for robustness
        
        Args:
            class_term: e.g., "ACE inhibitors", "statins", "beta blockers"
            
        Returns:
            List of RxCUIs
        """
        # Try different class type sources
        class_types = ["MESHPA", "PE", "EPC"]
        
        all_rxcuis = []
        
        for class_type in class_types:
            rxcuis = self.get_drugs_in_class(class_term, class_type)
            all_rxcuis.extend(rxcuis)
        
        # Remove duplicates
        unique_rxcuis = list(set(all_rxcuis))
        
        return unique_rxcuis


class DrugNormalizer:
    """
    High-level interface for drug normalization and expansion
    Combines RxNorm and RxClass functionality
    """
    
    def __init__(self):
        self.rxnorm = RxNormClient()
        self.rxclass = RxClassClient()
    
    def normalize_drug_name(self, drug_name: str) -> Optional[Dict]:
        """
        Normalize drug name to RxCUI and get classes
        
        Returns:
            {
                "drug_name": str,
                "rxcui": str,
                "classes": [...]
            }
        """
        rxcui = self.rxnorm.get_rxcui_from_name(drug_name)
        
        if not rxcui:
            return None
        
        classes = self.rxclass.get_drug_classes(rxcui)
        
        return {
            "drug_name": drug_name,
            "rxcui": rxcui,
            "classes": classes
        }
    
    def normalize_ndc(self, ndc: str) -> Optional[Dict]:
        """
        Normalize NDC code to drug information
        
        Returns:
            {
                "ndc": str,
                "rxcui": str,
                "drug_name": str,
                "classes": [...]
            }
        """
        rxcui = self.rxnorm.get_rxcui_from_ndc(ndc)
        
        if not rxcui:
            return None
        
        drug_name = self.rxnorm.get_drug_name(rxcui)
        classes = self.rxclass.get_drug_classes(rxcui)
        
        return {
            "ndc": ndc,
            "rxcui": rxcui,
            "drug_name": drug_name,
            "classes": classes
        }
    
    def expand_class_to_drugs(self, class_term: str) -> List[str]:
        """
        Expand drug class to list of RxCUIs
        
        Args:
            class_term: e.g., "ACE inhibitors"
            
        Returns:
            List of RxCUIs
        """
        return self.rxclass.expand_class_query(class_term)
    
    def is_class_query(self, query: str) -> bool:
        """
        Detect if query mentions a drug class
        
        Returns:
            True if class-based query
        """
        class_indicators = [
            'inhibitor', 'blocker', 'agonist', 'antagonist',
            'antibiotic', 'statin', 'diuretic', 'analgesic',
            'antidepressant', 'antipsychotic', 'anticoagulant',
            'beta blocker', 'ace inhibitor', 'arb', 'ssri',
            'nsaid', 'ppi', 'sulfonylurea'
        ]
        
        query_lower = query.lower()
        
        return any(indicator in query_lower for indicator in class_indicators)


# Example usage
if __name__ == "__main__":
    normalizer = DrugNormalizer()
    
    # Example 1: Normalize drug name
    print("=== Example 1: Normalize Drug Name ===")
    lisinopril = normalizer.normalize_drug_name("Lisinopril")
    if lisinopril:
        print(f"Drug: {lisinopril['drug_name']}")
        print(f"RxCUI: {lisinopril['rxcui']}")
        print(f"Classes: {lisinopril['classes']}")
    
    # Example 2: Normalize NDC
    print("\n=== Example 2: Normalize NDC ===")
    ndc_info = normalizer.normalize_ndc("0378-0172-93")
    if ndc_info:
        print(f"NDC: {ndc_info['ndc']}")
        print(f"Drug: {ndc_info['drug_name']}")
        print(f"RxCUI: {ndc_info['rxcui']}")
    
    # Example 3: Expand class query
    print("\n=== Example 3: Expand Class Query ===")
    ace_inhibitors = normalizer.expand_class_to_drugs("ACE Inhibitors")
    print(f"Found {len(ace_inhibitors)} ACE inhibitors")
    print(f"First 5: {ace_inhibitors[:5]}")
    
    # Example 4: Detect class query
    print("\n=== Example 4: Detect Class Query ===")
    queries = [
        "What are adverse reactions of Lisinopril?",
        "What are adverse reactions of ACE inhibitors?",
        "What are adverse reactions of beta blockers?"
    ]
    for q in queries:
        is_class = normalizer.is_class_query(q)
        print(f"'{q}' → Class query: {is_class}")
