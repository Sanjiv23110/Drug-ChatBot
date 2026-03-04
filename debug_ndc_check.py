
import logging
from normalization.rxnorm_integration import DrugNormalizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    normalizer = DrugNormalizer()
    
    # NDCs from the Meperidine Hydrochloride file (9C4F...xml)
    ndcs = [
        "0054-8596-11",
        "0054-4596-25",
        "0054-3545-63"
    ]
    
    print("\n--- Checking NDCs for Meperidine Hydrochloride ---")
    for ndc in ndcs:
        rxcui = normalizer.rxnorm.get_rxcui_from_ndc(ndc)
        print(f"NDC {ndc} -> RxCUI {rxcui}")
        
        if rxcui:
            name = normalizer.rxnorm.get_drug_name(rxcui)
            print(f"RxCUI {rxcui} -> Name: {name}")
    
    print("\n--- Checking RxCUI for 'Meperidine Hydrochloride' string ---")
    rxcui_hydro = normalizer.rxnorm.get_rxcui_from_name("Meperidine Hydrochloride")
    print(f"String 'Meperidine Hydrochloride' -> RxCUI {rxcui_hydro}")
