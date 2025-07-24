#!/usr/bin/env python3
"""
Demo script showcasing the advanced error handling features in HiRAG.

This demo shows:
1. Context manager syntax for error handling
2. Different retry strategies
3. LLM-informed error recovery
4. Granular error configuration
"""

import asyncio
import logging
from hirag import (
    HiRAG, 
    ErrorConfig, 
    RetryStrategy, 
    RetryConfig,
    ErrorSeverity,
    error_handling_context,
    with_error_handling
)

# Set up logging to see error handling in action
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_context_manager_error_handling():
    """Demo using context managers for error handling."""
    print("\n=== Context Manager Error Handling Demo ===")
    
    # Configure HiRAG with custom error handling
    error_config = ErrorConfig()
    error_config.should_break.update({
        "entity_extraction": False,  # Continue on entity extraction errors
        "demo_operation": False,     # Continue on demo operation errors
    })
    
    hirag = HiRAG(error_config=error_config)
    
    # Demo 1: Basic context manager usage
    print("\n1. Basic context manager with error recovery:")
    with error_handling_context("demo_operation", reraise_on_break=False, default_return_value="recovered") as ctx:
        # This will intentionally fail
        try:
            raise ValueError("Simulated error for demo")
        except ValueError as e:
            print(f"Error caught and handled: {e}")
            result = ctx.result  # Get the default return value
            print(f"Result after error handling: {result}")

    # Demo 2: Context manager with function execution
    print("\n2. Context manager with function execution:")
    
    def failing_function(x, y):
        if x < 0:
            raise ValueError("Negative numbers not allowed")
        return x + y
    
    with with_error_handling("demo_operation", RetryStrategy.DIRECT, max_retries=2) as ctx:
        # This will succeed
        result = ctx.execute(failing_function, 5, 10)
        print(f"Success result: {result}")
        
    with with_error_handling("demo_operation", RetryStrategy.DIRECT, max_retries=2) as ctx:
        # This will fail and use default value
        result = ctx.execute(failing_function, -1, 10)
        print(f"Fallback result: {result}")


async def demo_llm_informed_retry():
    """Demo LLM-informed retry mechanism."""
    print("\n=== LLM-Informed Retry Demo ===")
    
    # Configure error handling with LLM-informed retries
    error_config = ErrorConfig()
    error_config.retry_config["entity_extraction"] = RetryConfig(
        max_retries=2,
        strategy=RetryStrategy.LLM_INFORMED,
        llm_error_prompt="entiti_continue_extraction",
        include_error_context=True
    )
    
    hirag = HiRAG(error_config=error_config)
    
    print("Configured HiRAG with LLM-informed retry for entity extraction")
    print("In real usage, this would attempt to recover from entity extraction errors")
    print("by informing the LLM about the error and retrying with additional context.")


async def demo_different_retry_strategies():
    """Demo different retry strategies."""
    print("\n=== Different Retry Strategies Demo ===")
    
    # Custom retry function
    async def custom_retry_logic(func, args, kwargs, error, attempt):
        print(f"Custom retry attempt {attempt} after error: {error}")
        await asyncio.sleep(0.1)  # Short delay
        return await func(*args, **kwargs)
    
    strategies = [
        (RetryStrategy.NONE, "No retries"),
        (RetryStrategy.DIRECT, "Direct retry"),
        (RetryStrategy.EXPONENTIAL_BACKOFF, "Exponential backoff"),
        (RetryStrategy.LLM_INFORMED, "LLM-informed retry"),
    ]
    
    for strategy, description in strategies:
        print(f"\n{description}:")
        
        error_config = ErrorConfig()
        error_config.retry_config["demo_operation"] = RetryConfig(
            max_retries=2,
            strategy=strategy,
            backoff_factor=1.5
        )
        
        hirag = HiRAG(error_config=error_config)
        
        with with_error_handling("demo_operation") as ctx:
            try:
                # Simulate a function that might fail
                import random
                if random.random() < 0.7:  # 70% chance of failure
                    raise ConnectionError("Simulated network error")
                result = "Success!"
            except Exception as e:
                result = f"Handled: {e}"
            
            print(f"  Result: {result}")


async def demo_granular_configuration():
    """Demo granular error configuration."""
    print("\n=== Granular Error Configuration Demo ===")
    
    # Create HiRAG with detailed error configuration
    error_config = ErrorConfig()
    
    # Configure different behaviors for different operations
    error_config.should_break.update({
        "critical_operation": True,     # Stop pipeline on critical errors
        "optional_operation": False,    # Continue on optional operation errors
        "network_errors": False,        # Continue on network issues
        "validation_errors": True,      # Stop on validation failures
    })
    
    # Configure different retry strategies for different operations
    error_config.retry_config.update({
        "network_operation": RetryConfig(
            max_retries=5,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            backoff_factor=2.0
        ),
        "llm_operation": RetryConfig(
            max_retries=3,
            strategy=RetryStrategy.LLM_INFORMED,
            llm_error_prompt="entiti_continue_extraction"
        ),
        "parsing_operation": RetryConfig(
            max_retries=2,
            strategy=RetryStrategy.DIRECT
        )
    })
    
    hirag = HiRAG(error_config=error_config)
    
    print("Configured HiRAG with granular error handling:")
    print("- Critical operations will stop the pipeline")
    print("- Optional operations will continue with fallbacks")
    print("- Network operations will retry with exponential backoff")
    print("- LLM operations will use LLM-informed recovery")
    
    # Demo the configuration
    operations = [
        ("critical_operation", "This would stop the pipeline"),
        ("optional_operation", "This would continue with fallback"),
        ("network_operation", "This would retry with backoff"),
        ("llm_operation", "This would use LLM-informed retry")
    ]
    
    for op_type, description in operations:
        should_break = hirag.error_handler.should_break_pipeline(op_type)
        retry_config = hirag.error_handler.config.retry_config.get(op_type)
        
        print(f"\n{op_type}:")
        print(f"  Description: {description}")
        print(f"  Breaks pipeline: {should_break}")
        if retry_config:
            print(f"  Retry strategy: {retry_config.strategy.value}")
            print(f"  Max retries: {retry_config.max_retries}")


async def main():
    """Run all demos."""
    print("HiRAG Advanced Error Handling Demo")
    print("=" * 50)
    
    await demo_context_manager_error_handling()
    await demo_llm_informed_retry()
    await demo_different_retry_strategies() 
    await demo_granular_configuration()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the logs to see error handling in action.")


if __name__ == "__main__":
    asyncio.run(main())