"""
Verification test: parent-fetch pipeline fix for duplicate answers.
Run from c:\G\solomindUS:  python tests\test_parent_fetch.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "x")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "x")

from dotenv import load_dotenv
load_dotenv()

from retrieval.hybrid_retriever import DenseEmbedder, CrossEncoderReranker, HybridRetriever
from vector_db.qdrant_manager import QdrantManager

print("Loading models (this takes ~30s)...")
de = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
rr = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
vdb = QdrantManager(host="127.0.0.1", port=6333, collection_name="spl_children")
ret = HybridRetriever(de, None, rr, vdb)

QUERY = "how many times serum levels should be determined in acute mania for lithane"
print(f"\nQuery: {QUERY}")

chunks, _ = ret.retrieve(QUERY, filter_conditions=None, retrieval_limit=75, rerank_top_k=15)

print(f"\nResults: {len(chunks)} parent paragraph(s) returned\n")
raw_texts = [c.get("raw_text", "") for c in chunks]
duplicates = [t for t in raw_texts if raw_texts.count(t) > 1]

for i, c in enumerate(chunks[:5]):
    meta = c.get("metadata", {})
    print(f"--- Result {i+1} ---")
    print(f"  parent_id : {meta.get('parent_id')}")
    print(f"  loinc_sec : {meta.get('loinc_section')}")
    print(f"  raw_text  : {c.get('raw_text', '')[:300]}")
    print()

print("=" * 50)
print(f"Duplicate raw_texts found: {len(duplicates)}")
if duplicates:
    print("FAIL - duplicates still present!")
    for d in set(duplicates):
        print(f"  DUPE: {d[:100]}")
    sys.exit(1)
else:
    print("PASS - zero duplicates. Parent-fetch pipeline working correctly.")
    sys.exit(0)
