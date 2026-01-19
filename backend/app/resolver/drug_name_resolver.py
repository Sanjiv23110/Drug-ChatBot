"""
Drug name resolver with CRITICAL guardrails.

Guardrails:
1. NON-DESTRUCTIVE: If resolver fails, system behaves exactly like Phase 3
2. BOUNDED FUZZY: Length normalization prevents absurd matches
3. MULTI-DRUG GUARD: >1 drug detected → disable filtering

Scope lock:
- ONLY brand → generic resolution
- NO ingestion changes
- NO FAISS changes
- NO ML models
"""
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict
from functools import lru_cache


class DrugNameResolver:
    """
    Resolve brand names to generic names with strict safety guarantees.
    
    Critical invariant:
        If resolution fails → return None (no fallback, no partial guesses)
    """
    
    def __init__(self, db_path: str = "data/metadata.db"):
        self.db_path = Path(db_path)
        self._load_mappings()
    
    def _load_mappings(self):
        """
        Load drug name mappings from database.
        
        If table doesn't exist or is empty, log warning and continue.
        System will behave like Phase 3 (no drug filtering).
        """
        if not self.db_path.exists():
            logging.warning(
                f"Drug database not found at {self.db_path}. "
                "Resolver disabled (will behave like Phase 3)."
            )
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM drug_names")
                count = cursor.fetchone()[0]
                
                if count == 0:
                    logging.warning(
                        "Drug names table is empty. "
                        "Add mappings to enable brand name resolution."
                    )
                else:
                    logging.info(f"Loaded {count} drug name mappings")
        
        except sqlite3.OperationalError:
            logging.warning(
                "Drug names table does not exist. "
                "Run schema migration to enable resolver."
            )
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Compute edit distance between two strings.
        
        Used for typo tolerance: "Gravl" → "Gravol"
        """
        if len(s1) < len(s2):
            return DrugNameResolver.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @lru_cache(maxsize=128)
    def _fuzzy_match(self, query_name: str, max_distance: int = 2) -> Optional[str]:
        """
        Fuzzy match brand name with BOUNDED matching.
        
        Guardrail 2: Length normalization prevents absurd matches.
        
        Args:
            query_name: Brand name to search (case-insensitive)
            max_distance: Maximum edit distance (default 2)
            
        Returns:
            Generic name if match found, None otherwise
        """
        if not self.db_path.exists():
            return None
        
        query_lower = query_name.lower().strip()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT brand_name, generic_name FROM drug_names"
                )
                
                best_match = None
                best_distance = max_distance + 1
                
                for row in cursor:
                    brand = row['brand_name'].lower()
                    
                    # Guardrail 2: Length normalization
                    # Skip if length difference > 2 (prevents absurd matches)
                    if abs(len(query_lower) - len(brand)) > 2:
                        continue
                    
                    distance = self.levenshtein_distance(query_lower, brand)
                    
                    if distance <= max_distance and distance < best_distance:
                        best_match = row['generic_name']
                        best_distance = distance
                
                return best_match
        
        except sqlite3.OperationalError:
            return None
    
    @lru_cache(maxsize=128)
    def resolve(self, query: str) -> Optional[str]:
        """
        Resolve brand name to generic name.
        
        Guardrail 1: NON-DESTRUCTIVE
        If resolution fails → return None (no fallback, no partial guesses)
        
        Args:
            query: User query text
            
        Returns:
            Generic name if found, None otherwise
            
        Examples:
            "Gravol" → "dimenhydrinate"
            "Tylenol" → "acetaminophen"
            "Unknown Drug" → None
            "Gravl" (typo) → "dimenhydrinate"
        """
        if not self.db_path.exists():
            return None  # Guardrail 1: fail gracefully
        
        # Extract potential drug names (simple word extraction)
        words = query.split()
        
        for word in words:
            # Clean word (remove punctuation)
            clean_word = ''.join(c for c in word if c.isalnum())
            
            if len(clean_word) < 3:
                continue  # Skip short words
            
            # Try exact match first (case-insensitive)
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT generic_name FROM drug_names WHERE brand_name = ? COLLATE NOCASE",
                        (clean_word,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        logging.info(f"Exact match: '{clean_word}' → '{row['generic_name']}'")
                        return row['generic_name']
            
            except sqlite3.OperationalError:
                return None  # Guardrail 1: fail gracefully
        
        # Try fuzzy matching on longest word
        if words:
            longest_word = max(words, key=len)
            clean_longest = ''.join(c for c in longest_word if c.isalnum())
            
            if len(clean_longest) >= 3:
                match = self._fuzzy_match(clean_longest)
                if match:
                    logging.info(f"Fuzzy match: '{clean_longest}' → '{match}'")
                    return match
        
        # Guardrail 1: If no match, return None (no fallback)
        return None
    
    def extract_drug_names(self, query: str) -> List[str]:
        """
        Extract ALL drug names from query.
        
        Guardrail 3: Multi-drug detection
        
        Args:
            query: User query text
            
        Returns:
            List of generic names found (may be empty, may be >1)
            
        Examples:
            "Gravol side effects" → ["dimenhydrinate"]
            "Can I take Tylenol and Advil?" → ["acetaminophen", "ibuprofen"]
            "Random query" → []
        """
        if not self.db_path.exists():
            return []
        
        words = query.split()
        found_drugs = []
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            
            if len(clean_word) < 3:
                continue
            
            # Try exact match
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT generic_name FROM drug_names WHERE brand_name = ? COLLATE NOCASE",
                        (clean_word,)
                    )
                    row = cursor.fetchone()
                    
                    if row and row['generic_name'] not in found_drugs:
                        found_drugs.append(row['generic_name'])
            
            except sqlite3.OperationalError:
                continue
        
        return found_drugs
    
    def add_mapping(self, brand_name: str, generic_name: str):
        """
        Add a brand → generic mapping.
        
        Args:
            brand_name: Brand name (e.g., "Gravol")
            generic_name: Generic name (e.g., "dimenhydrinate")
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO drug_names (brand_name, generic_name) VALUES (?, ?)",
                (brand_name, generic_name)
            )
        
        # Clear cache
        self.resolve.cache_clear()
        self._fuzzy_match.cache_clear()
        
        logging.info(f"Added mapping: {brand_name} → {generic_name}")
