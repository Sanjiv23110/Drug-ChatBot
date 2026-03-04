import ast, sys
sys.path.insert(0, r"C:\G\solomindUS")

for f in ['retrieval/hybrid_retriever.py', 'orchestrator/qa_orchestrator.py']:
    ast.parse(open(f).read())
    print(f"Syntax OK: {f}")

from orchestrator.qa_orchestrator import SectionClassifier
sc = SectionClassifier()
tests = [
    ("what is the generic name of dantrium",        "34089-3"),
    ("whats the generic name of the dantrium",      "34089-3"),
    ("indication and usage of dantrium",            "34067-9"),
    ("whats elimination half-life of dantrium",     "43682-4"),
    ("adverse reactions of dantrium",               "34084-4"),
    ("mechanism of action of renese",               "43678-2"),
    ("clinical pharmacology of dantrium",           "34090-1"),
]
print()
all_pass = True
for q, expected in tests:
    got = sc.classify(q)
    status = "PASS" if got == expected else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"[{status}] \"{q}\" -> {got} (expected {expected})")
print()
print("ALL PASS" if all_pass else "SOME FAILED")
