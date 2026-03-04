"""Verification test for the parent-level deterministic extraction refactor."""
import requests

def test(q):
    r = requests.post("http://localhost:8000/chat", json={"query": q}, timeout=60)
    return r.json()

r1 = test("how to treat overdose of dantrium?")
r2 = test("how to treat overdosage of dantrium?")
r3 = test("what is the generic name of dantrium?")

a1 = r1.get("answer", "")
a2 = r2.get("answer", "")
a3 = r3.get("answer", "")

with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write("=== TEST 1: how to treat overdose of dantrium? ===\n")
    f.write(a1 + "\n\n")
    f.write("=== TEST 2: how to treat overdosage of dantrium? ===\n")
    f.write(a2 + "\n\n")
    f.write("=== IDENTICAL? " + str(a1 == a2) + " ===\n\n")
    f.write("=== TEST 3: what is the generic name of dantrium? ===\n")
    f.write(a3 + "\n\n")

    tox = ["animal", "rats", "dogs", "toxicity", "subacute", "hepatic changes"]
    f.write("ANIMAL_TOX_IN_OVERDOSE: " + str(any(k in a1.lower() for k in tox)) + "\n")
    f.write("ANIMAL_TOX_IN_OVERDOSAGE: " + str(any(k in a2.lower() for k in tox)) + "\n")
    f.write("META1: " + str(r1.get("metadata", {})) + "\n")
    f.write("META2: " + str(r2.get("metadata", {})) + "\n")
    f.write("META3: " + str(r3.get("metadata", {})) + "\n")

print("Written to test_results.txt")
