import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.db.models import MonographSection, SectionMapping
    from app.db.fact_span_model import FactSpan
    print("SUCCESS: Models imported without error.")
except Exception as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
