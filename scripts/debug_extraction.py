import sys
import os
import logging

# Set up path to include root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalization.rxnorm_integration import DrugNormalizer

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_extraction(query):
    print(f"\nTesting query: '{query}'")
    
    # Simulate the EXACT logic currently in qa_orchestrator.py
    words = query.split()
    found = None
    
    # We need to initialize this outside the loop in the real code, but here is fine
    normalizer = DrugNormalizer()
    
    print(f"Tokenized: {words}")
    
    for word in words:
        if len(word) > 3:
            # Check the word exactly as it is (with punctuation)
            rxcui = normalizer.rxnorm.get_rxcui_from_name(word)
            if rxcui:
                print(f"  [SUCCESS] '{word}' -> RxCUI: {rxcui}")
                found = word
                break
            else:
                print(f"  [FAIL] '{word}' -> No RxCUI")
                
            # Now let's try with stripped punctuation to prove the fix
            import string
            stripped_word = word.strip(string.punctuation)
            if stripped_word != word:
                print(f"    (Debug) Checking stripped: '{stripped_word}'")
                rxcui_stripped = normalizer.rxnorm.get_rxcui_from_name(stripped_word)
                if rxcui_stripped:
                     print(f"    (Debug) [WOULD SUCCEED] '{stripped_word}' -> RxCUI: {rxcui_stripped}")

    if found:
        print(f"Result: Extracted '{found}'")
    else:
        print("Result: FAILED to extract drug name")

if __name__ == "__main__":
    queries = [
        "What is Tolinase indicated for?",
        "What are the adverse reactions of Tolinase?",
        "Renese dosage"
    ]
    for q in queries:
        test_extraction(q)
