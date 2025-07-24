"""
Validation utilities for HiRAG data structures.

This module provides centralized validation functions to replace naive manual validations
throughout the codebase with more robust and maintainable validation logic.
"""

from typing import Any, List


def validate(validation_type: str, data: Any) -> bool:
    """
    Centralized validation function for various data types used in HiRAG.
    
    Args:
        validation_type: Type of validation to perform
        data: Data to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    if validation_type == "record_attributes_for_entity":
        return _validate_entity_record_attributes(data)
    elif validation_type == "record_attributes_for_relationship":
        return _validate_relationship_record_attributes(data)
    elif validation_type == "record_attributes_for_entity_clustering":
        return _validate_entity_record_attributes_clustering(data)
    elif validation_type == "record_attributes_for_relationship_clustering":
        return _validate_relationship_record_attributes_clustering(data)
    else:
        raise ValueError(f"Unknown validation type: {validation_type}")


def _validate_entity_record_attributes(record_attributes: Any) -> bool:
    """
    Validate record attributes for entity extraction.
    
    Expected format: ["entity", name, type, desc, is_temporary]
    Minimum length: 5 elements
    First element must be '"entity"'
    
    Args:
        record_attributes: List of record attributes to validate
        
    Returns:
        bool: True if valid entity record attributes, False otherwise
    """
    if not isinstance(record_attributes, list):
        return False
    
    if len(record_attributes) < 5:
        return False
        
    # Handle potential None values or empty list
    if not record_attributes or record_attributes[0] is None:
        return False
        
    if record_attributes[0] != '"entity"':
        return False
        
    return True


def _validate_relationship_record_attributes(record_attributes: Any) -> bool:
    """
    Validate record attributes for relationship extraction.
    
    Expected format: ["relationship", source, target, description, weight]
    Minimum length: 5 elements
    First element must be '"relationship"'
    
    Args:
        record_attributes: List of record attributes to validate
        
    Returns:
        bool: True if valid relationship record attributes, False otherwise
    """
    if not isinstance(record_attributes, list):
        return False
    
    if len(record_attributes) < 5:
        return False
        
    # Handle potential None values or empty list
    if not record_attributes or record_attributes[0] is None:
        return False
        
    if record_attributes[0] != '"relationship"':
        return False
        
    return True


def _validate_entity_record_attributes_clustering(record_attributes: Any) -> bool:
    """
    Validate record attributes for entity extraction in clustering context.
    
    Expected format: ["entity", name, type, desc] (without is_temporary)
    Minimum length: 4 elements
    First element must be '"entity"'
    
    Args:
        record_attributes: List of record attributes to validate
        
    Returns:
        bool: True if valid entity record attributes for clustering, False otherwise
    """
    if not isinstance(record_attributes, list):
        return False
    
    if len(record_attributes) < 4:
        return False
        
    # Handle potential None values or empty list
    if not record_attributes or record_attributes[0] is None:
        return False
        
    if record_attributes[0] != '"entity"':
        return False
        
    return True


def _validate_relationship_record_attributes_clustering(record_attributes: Any) -> bool:
    """
    Validate record attributes for relationship extraction in clustering context.
    
    Expected format: ["relationship", source, target, description, weight]
    Minimum length: 5 elements
    First element must be '"relationship"'
    
    Args:
        record_attributes: List of record attributes to validate
        
    Returns:
        bool: True if valid relationship record attributes for clustering, False otherwise
    """
    if not isinstance(record_attributes, list):
        return False
    
    if len(record_attributes) < 5:
        return False
        
    # Handle potential None values or empty list
    if not record_attributes or record_attributes[0] is None:
        return False
        
    if record_attributes[0] != '"relationship"':
        return False
        
    return True