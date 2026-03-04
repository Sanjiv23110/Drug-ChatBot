
import string
import logging
from normalization.rxnorm_integration import DrugNormalizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replicating the logic from qa_orchestrator.py _extract_drug_name
# Note: I'm not copying the full stopword list to keep it brief, but copying the logic structure
stopwords = {
    'what', 'is', 'the', 'recommended', 'dosage', 'for', 'of', 'in', 'to', 'and', 'or',
    'with', 'about', 'from', 'that', 'this', 'these', 'those', 'can',
    'should', 'would', 'could', 'will', 'may', 'might', 'must', 'have',
    'has', 'had', 'do', 'does', 'did', 'be', 'been', 'being', 'was',
    'were', 'am', 'is', 'are', 'a', 'an', 'the', 'some', 'any', 'all',
    'tell', 'me', 'you', 'your', 'my', 'their', 'his', 'her', 'its', 'our',
    'how', 'why', 'when', 'where', 'which', 'who', 'whom', 'whose',
    'take', 'taken', 'taking', 'give', 'given', 'giving', 'use', 'used', 'using',
    'most', 'common', 'adverse', 'reactions', 'side', 'effects', 'dosage',
    'dose', 'indication', 'indications', 'contraindication', 'contraindications',
    'warning', 'warnings', 'precaution', 'precautions', 'interaction', 'interactions',
    'types', 'type', 'indicated', 'treat', 'treatment', 'therapy',
    'available', 'strengths', 'special', 'cardiovascular', 'mortality',
    'laboratory', 'tests', 'inactive', 'ingredients', 'active',
    'biological', 'half', 'life', 'mechanism', 'action', 'clinical',
    'pharmacology', 'pharmacokinetics', 'overdosage', 'overdose',
    'storage', 'handling', 'description', 'supplied', 'packaging',
    'gastrointestinal', 'cardiovascular', 'respiratory', 'renal', 'hepatic',
    'nervous', 'central', 'peripheral', 'dermatologic', 'hematologic',
    'endocrine', 'metabolic', 'musculoskeletal', 'genitourinary',
    'electrolyte', 'electrolytes', 'imbalance', 'imbalances',
    'patients', 'patient', 'pediatric', 'geriatric', 'adult', 'adults',
    'children', 'elderly', 'pregnant', 'pregnancy', 'nursing', 'mothers',
    'diabetic', 'diabetics', 'hypertensive', 'renal', 'hepatic',
    'disease', 'disorder', 'condition', 'conditions', 'susceptible',
    'predispose', 'predisposes', 'monitored', 'safe', 'safety',
}

def extract_drug_name(query, normalizer):
    words = query.split()
    print(f"\nScanning query: '{query}'")
    
    for word in words:
        clean_word = word.strip(string.punctuation)
        if clean_word.lower() in stopwords:
            # print(f"  Skipping stopword: {clean_word}")
            continue
            
        if len(clean_word) > 3:
            # print(f"  Testing word: {clean_word}")
            
            # Try as is
            rxcui = normalizer.rxnorm.get_rxcui_from_name(clean_word)
            if rxcui:
                print(f"  MATCH FOUND (As Is): {clean_word} -> {rxcui}")
                return clean_word
            
            # Try capitalized
            if not clean_word[0].isupper():
                cap_word = clean_word.capitalize()
                rxcui = normalizer.rxnorm.get_rxcui_from_name(cap_word)
                if rxcui:
                    print(f"  MATCH FOUND (Capitalized): {cap_word} -> {rxcui}")
                    return cap_word
                    
            # Try lower
            if not clean_word.islower():
                lower_word = clean_word.lower()
                rxcui = normalizer.rxnorm.get_rxcui_from_name(lower_word)
                if rxcui:
                    print(f"  MATCH FOUND (Lower): {lower_word} -> {rxcui}")
                    return lower_word

    print("  NO MATCH FOUND")
    return None

if __name__ == "__main__":
    try:
        normalizer = DrugNormalizer()
        
        queries = [
            "What is the recommended dosage for Meperidine Hydrochloride?",
            "dosage for Meperidine", 
            "Meperidine Hydrochloride dosage",
            "recommended dosage for Hypaque Sodium"
        ]
        
        for q in queries:
            print("-" * 40)
            extract_drug_name(q, normalizer)
            
    except Exception as e:
        print(f"Error: {e}")
