
import os

target_file = r"c:/G/solomindUS/orchestrator/qa_orchestrator.py"

new_logic = """        # Robust N-gram approach (Sliding Window)
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
        return None"""

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# The original code at 553 maps to index 552 (0-based)
# But verifying content first
start_idx = 552 # line 553
# check if line starts with '        words = query.split()'
if 'words = query.split()' not in lines[start_idx]:
    print(f"Warning: Line {start_idx+1} does not look right: {lines[start_idx]}")
    # Search for it
    for i, line in enumerate(lines):
        if 'words = query.split()' in line and 'capitalize words' in lines[i-1]:
            start_idx = i
            print(f"Found start at line {i+1}")
            break

# End index: where 'return None' is
# Originally line 582 -> index 581
end_idx = 581
# Verify
if 'return None' not in lines[end_idx]:
    print(f"Warning: Line {end_idx+1} is not 'return None': {lines[end_idx]}")
    # Search forward
    for i in range(start_idx, len(lines)):
        if 'return None' in lines[i] and lines[i].strip() == 'return None':
            end_idx = i
            print(f"Found end at line {i+1}")
            break

print(f"Replacing lines {start_idx+1} to {end_idx+1}")

# Keep indentation similar?
# The new_logic string is already indented with 8 spaces.

new_lines = lines[:start_idx] + [new_logic + "\n"] + lines[end_idx+1:]

with open(target_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Patch applied successfully.")
