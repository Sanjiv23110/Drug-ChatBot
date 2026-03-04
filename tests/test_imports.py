"""
Test critical imports
"""
import sys
print(f"Python: {sys.version}")

try:
    import scipy
    print(f"✅ scipy: {scipy.__version__}")
except ImportError as e:
    print(f"❌ scipy: {e}")

try:
    import sklearn
    print(f"✅ sklearn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ sklearn: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print(f"✅ sentence_transformers loaded")
except ImportError as e:
    print(f"❌ sentence_transformers: {e}")

try:
    import rapidfuzz
    print(f"✅ rapidfuzz: {rapidfuzz.__version__}")
except ImportError as e:
    print(f"❌ rapidfuzz: {e}")
