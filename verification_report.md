# Retrieval Pipeline Refactor - Verification Report

## Changes Implemented

### Section-First Reranking Strategy
- **Broad Initial Retrieval (K=75)**: Retrieval now fetches top 75 chunks purely based on `rxcui` (drug match), ignoring section filters initially.
- **In-Memory Filtering**: If a `loinc_code` is detected (e.g. for "overdosage"), `HybridRetriever` filters the 75 candidates in-memory to isolate the target section subset.
- **Targeted Reranking**: If the subset exists, only the subset is reranked. This prevents non-section content (e.g. "Dosage") from displacing target section content (e.g. "Overdose").
- **Fallback**: If the subset is empty (or no section detected), full reranking occurs.

### Observability
- Added structured JSON logs for retrieval stats:
  - `total_candidates_retrieved`
  - `section_subset_count`
  - `rerank_scope_size`
  - `final_candidate_count`

## Verification Steps

1. Run the following queries:
   - "how to treat overdosage of dantrium?" (Should use `LOINC 34088-5`)
   - "how to treat overdose of dantrium?" (Should use `LOINC 34088-5`)

2. Check the backend logs for:
   - `Detected LOINC: 34088-5`
   - `Section-First Strategy: Reranking X chunks from LOINC 34088-5`
   - `Section-specific verbatim extraction: X sentences returned`

## Expected Outcome
Both queries should return identical, complete results from the Overdosage section, regardless of "overdose" vs "overdosage" terminology.
