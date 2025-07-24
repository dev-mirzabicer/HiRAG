#!/usr/bin/env python3
"""
Demonstration script for HiRAG error handling capabilities.
Shows how the configurable error handling system works in practice.
"""

import sys
import os
import logging

# Add the hirag directory to the path
hirag_dir = os.path.join(os.path.dirname(__file__), 'hirag')
sys.path.insert(0, hirag_dir)

from _error_handling import (
    ErrorConfig, 
    HiRAGErrorHandler, 
    ErrorSeverity,
    error_handler,
    get_error_handler,
    set_error_handler
)
from _utils import convert_response_to_json, truncate_list_by_token_size

def setup_logging():
    """Set up comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def demo_basic_error_handling():
    """Demonstrate basic error handling configuration."""
    print("\n🔧 DEMO: Basic Error Handling Configuration")
    print("=" * 50)
    
    # Create a custom error configuration
    config = ErrorConfig()
    config.should_break["demo_context"] = False  # Don't break on demo errors
    config.should_break["critical_demo"] = True   # Break on critical demo errors
    
    handler = HiRAGErrorHandler(config)
    
    # Simulate a non-critical error
    test_error = ValueError("This is a demo error")
    should_continue = handler.handle_operation_error(
        error=test_error,
        context="demo_context",
        operation_details="Demo operation",
        reraise=False
    )
    print(f"Non-critical error handling result: {'Continue' if should_continue else 'Stop'}")
    
    # Simulate a critical error  
    critical_error = RuntimeError("This is a critical demo error")
    should_continue = handler.handle_operation_error(
        error=critical_error,
        context="critical_demo",
        operation_details="Critical demo operation",
        reraise=False
    )
    print(f"Critical error handling result: {'Continue' if should_continue else 'Stop'}")
    
    # Show error summary
    print(f"Error summary: {handler.get_error_summary()}")

def demo_json_parsing_with_error_handling():
    """Demonstrate JSON parsing with improved error handling."""
    print("\n📄 DEMO: JSON Parsing with Error Handling")
    print("=" * 50)
    
    test_cases = [
        '{"valid": "json", "data": true}',  # Valid JSON
        '{"incomplete": "json"',            # Invalid JSON
        'not json at all',                  # Not JSON
        '',                                 # Empty string
        '{"nested": {"data": {"key": "value"}}}',  # Nested JSON
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test_case[:30]}{'...' if len(test_case) > 30 else ''}")
        result = convert_response_to_json(test_case)
        print(f"Result: {result}")

@error_handler("demo_operations", reraise_on_break=False, default_return_value="FALLBACK")
def demo_function_with_errors(should_fail=False):
    """Demo function that can be configured to fail."""
    if should_fail:
        raise RuntimeError("Demo function configured to fail")
    return "SUCCESS"

def demo_function_decorator():
    """Demonstrate the error handler decorator."""
    print("\n🎯 DEMO: Function Error Decorator")
    print("=" * 50)
    
    # Configure error handling to continue on demo_operations errors
    config = ErrorConfig()
    config.should_break["demo_operations"] = False
    set_error_handler(HiRAGErrorHandler(config))
    
    # Test successful execution
    result1 = demo_function_with_errors(should_fail=False)
    print(f"Successful execution result: {result1}")
    
    # Test failed execution with fallback
    result2 = demo_function_with_errors(should_fail=True)
    print(f"Failed execution result (with fallback): {result2}")

def demo_pipeline_simulation():
    """Simulate a data processing pipeline with configurable error handling."""
    print("\n🔄 DEMO: Pipeline Simulation with Error Handling")
    print("=" * 50)
    
    # Configure error handling for different pipeline stages
    config = ErrorConfig()
    config.should_break.update({
        "data_ingestion": False,     # Continue on data ingestion errors
        "data_processing": False,    # Continue on processing errors  
        "data_validation": True,     # Stop on validation errors
        "data_storage": True,        # Stop on storage errors
    })
    
    handler = HiRAGErrorHandler(config)
    set_error_handler(handler)
    
    # Simulate pipeline stages
    pipeline_stages = [
        ("data_ingestion", "Loading data from source", ValueError("Connection timeout")),
        ("data_processing", "Processing data chunks", RuntimeError("Memory allocation failed")),
        ("data_validation", "Validating processed data", ValueError("Invalid data format")),
        ("data_storage", "Storing results", IOError("Disk full")),
    ]
    
    print("Simulating pipeline execution:")
    for stage, description, error in pipeline_stages:
        print(f"\n🔧 Stage: {stage}")
        print(f"   Operation: {description}")
        
        should_continue = handler.handle_operation_error(
            error=error,
            context=stage,
            operation_details=description,
            severity=ErrorSeverity.ERROR,
            reraise=False
        )
        
        if should_continue:
            print(f"   Result: ✅ Continuing to next stage")
        else:
            print(f"   Result: ❌ Pipeline stopped at {stage}")
            break
    
    print(f"\nPipeline error summary: {handler.get_error_summary()}")

def demo_configurable_breaking():
    """Demonstrate how to configure which errors break the pipeline."""
    print("\n⚙️  DEMO: Configurable Pipeline Breaking")
    print("=" * 50)
    
    # Show how the same error can be handled differently based on configuration
    test_error = ConnectionError("Network connection failed")
    
    # Configuration 1: Break on network errors
    config1 = ErrorConfig()
    config1.should_break["network_operations"] = True
    handler1 = HiRAGErrorHandler(config1)
    
    should_continue1 = handler1.handle_operation_error(
        error=test_error,
        context="network_operations", 
        operation_details="API call",
        reraise=False
    )
    print(f"Config 1 (break=True): {'Continue' if should_continue1 else 'Stop'}")
    
    # Configuration 2: Continue on network errors
    config2 = ErrorConfig()
    config2.should_break["network_operations"] = False
    handler2 = HiRAGErrorHandler(config2)
    
    should_continue2 = handler2.handle_operation_error(
        error=test_error,
        context="network_operations",
        operation_details="API call", 
        reraise=False
    )
    print(f"Config 2 (break=False): {'Continue' if should_continue2 else 'Stop'}")

def demo_error_severity_levels():
    """Demonstrate different error severity levels."""
    print("\n📊 DEMO: Error Severity Levels")
    print("=" * 50)
    
    handler = HiRAGErrorHandler()
    test_error = RuntimeError("Demo error for severity testing")
    
    severities = [ErrorSeverity.INFO, ErrorSeverity.WARNING, ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
    
    for severity in severities:
        print(f"\nLogging error with severity: {severity.value}")
        handler.log_error(
            error=test_error,
            context="severity_demo",
            severity=severity,
            operation_details=f"Demo operation at {severity.value} level"
        )

def main():
    """Run all demonstrations."""
    setup_logging()
    
    print("🎯 HiRAG Error Handling System Demonstration")
    print("=" * 60)
    print("This demo shows the comprehensive error handling capabilities")
    print("implemented for the HiRAG data ingestion pipeline.")
    
    try:
        demo_basic_error_handling()
        demo_json_parsing_with_error_handling() 
        demo_function_decorator()
        demo_pipeline_simulation()
        demo_configurable_breaking()
        demo_error_severity_levels()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("\nKey features demonstrated:")
        print("✅ Configurable error handling (should_break)")
        print("✅ Comprehensive error logging with context")
        print("✅ Automatic error recovery with fallback values")
        print("✅ Pipeline stage-specific error handling")
        print("✅ Error severity levels and traceability")
        print("✅ Function decorator for automatic error handling")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()