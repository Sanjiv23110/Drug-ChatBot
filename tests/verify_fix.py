"""
Reproduction Script for EntityValidator Fix
Verifies that EntityValidator works with both QdrantManager and HierarchicalQdrantManager
"""

import sys
sys.path.insert(0, 'c:/G/solomindUS')

from orchestrator.entity_validator import EntityValidator
from unittest.mock import Mock

print("="*80)
print("VERIFYING ENTITY VALIDATOR FIX")
print("="*80)

# 1. Test with HierarchicalQdrantManager (has child_collection)
print("\n[TEST 1] HierarchicalQdrantManager (Original behavior)")
mock_hierarchical_db = Mock()
mock_hierarchical_db.child_collection = "spl_children"
mock_hierarchical_db.client.scroll.return_value = ([
    Mock(payload={'drug_name': 'Drug A'})
], None)

try:
    validator1 = EntityValidator(mock_hierarchical_db)
    print("✓ SUCCESS: Initialized with HierarchicalQdrantManager")
    print(f"  Collection used: {mock_hierarchical_db.child_collection}")
    print(f"  Drugs loaded: {len(validator1.drug_names)}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# 2. Test with QdrantManager (NO child_collection, only collection_name)
print("\n[TEST 2] QdrantManager (Backend Server behavior)")
mock_simple_db = Mock()
# Explicitly delete child_collection ensures usage of getattr/hasattr check
del mock_simple_db.child_collection 
mock_simple_db.collection_name = "spl_children_simple"
mock_simple_db.client.scroll.return_value = ([
    Mock(payload={'drug_name': 'Drug B'})
], None)

try:
    validator2 = EntityValidator(mock_simple_db)
    print("✓ SUCCESS: Initialized with QdrantManager")
    print(f"  Collection used: {mock_simple_db.collection_name}")
    print(f"  Drugs loaded: {len(validator2.drug_names)}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
