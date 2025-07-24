#!/usr/bin/env python3
"""
Simple test script for validation module.
Avoids importing the full hirag package to prevent tiktoken network issues.
"""

import sys
import os

# Add the hirag directory to the path
hirag_dir = os.path.join(os.path.dirname(__file__), 'hirag')
sys.path.insert(0, hirag_dir)

# Import directly from the validation module
from _validation import validate

def test_validation():
    print("Testing validation module...")
    
    # Test entity validation - valid case
    result1 = validate('record_attributes_for_entity', ['"entity"', 'name', 'type', 'desc', 'true'])
    print(f'Entity validation (valid): {result1}')
    assert result1 == True
    
    # Test entity validation - too short
    result2 = validate('record_attributes_for_entity', ['"entity"', 'name', 'type'])
    print(f'Entity validation (too short): {result2}')
    assert result2 == False
    
    # Test entity validation - wrong type
    result3 = validate('record_attributes_for_entity', ['"relationship"', 'name', 'type', 'desc', 'true'])
    print(f'Entity validation (wrong type): {result3}')
    assert result3 == False
    
    # Test relationship validation - valid case
    result4 = validate('record_attributes_for_relationship', ['"relationship"', 'src', 'tgt', 'desc', '1.0'])
    print(f'Relationship validation (valid): {result4}')
    assert result4 == True
    
    # Test relationship validation - too short
    result5 = validate('record_attributes_for_relationship', ['"relationship"', 'src'])
    print(f'Relationship validation (too short): {result5}')
    assert result5 == False
    
    # Test clustering entity validation - valid case (only 4 elements needed)
    result6 = validate('record_attributes_for_entity_clustering', ['"entity"', 'name', 'type', 'desc'])
    print(f'Entity clustering validation (valid): {result6}')
    assert result6 == True
    
    # Test clustering entity validation - too short
    result7 = validate('record_attributes_for_entity_clustering', ['"entity"', 'name'])
    print(f'Entity clustering validation (too short): {result7}')
    assert result7 == False
    
    # Test with non-list input
    result8 = validate('record_attributes_for_entity', 'not a list')
    print(f'Entity validation (not a list): {result8}')
    assert result8 == False
    
    # Test with empty list
    result9 = validate('record_attributes_for_entity', [])
    print(f'Entity validation (empty list): {result9}')
    assert result9 == False
    
    # Test with None values
    result10 = validate('record_attributes_for_entity', [None, 'name', 'type', 'desc', 'true'])
    print(f'Entity validation (None first element): {result10}')
    assert result10 == False
    
    # Test invalid validation type
    try:
        validate('invalid_type', ['"entity"', 'name'])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f'Invalid validation type error: {e}')
    
    print("All validation tests passed!")

if __name__ == "__main__":
    test_validation()