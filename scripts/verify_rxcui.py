import sys
import os
import logging

# Set up path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalization.rxnorm_integration import DrugNormalizer

logging.basicConfig(level=logging.WARNING)

def check(name):
    n = DrugNormalizer()
    print(f"Checking '{name}'...")
    rxcui = n.rxnorm.get_rxcui_from_name(name)
    print(f"Result: {rxcui}")

if __name__ == "__main__":
    check("Renese")
    check("renese")
    check("Tolinase")
    check("tolinase")
