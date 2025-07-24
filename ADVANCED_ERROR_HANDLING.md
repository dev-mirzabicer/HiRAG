# Advanced Error Handling in HiRAG

This document describes the enhanced error handling system implemented in HiRAG, which provides comprehensive, configurable, and granular error management with different retry strategies.

## Key Features

### 1. Context Manager Syntax

Use `with` statements for cleaner error handling:

```python
from hirag import error_handling_context, with_error_handling, RetryStrategy

# Basic context manager
with error_handling_context("entity_extraction", reraise_on_break=False) as ctx:
    # Your code here
    result = some_risky_operation()

# Convenience function with retry configuration
with with_error_handling("llm_operations", RetryStrategy.LLM_INFORMED, max_retries=3) as ctx:
    result = ctx.execute(my_function, arg1, arg2)
```

### 2. Multiple Retry Strategies

Choose from different retry approaches:

- **`RetryStrategy.NONE`** - No retries
- **`RetryStrategy.DIRECT`** - Simple retry of the same operation
- **`RetryStrategy.EXPONENTIAL_BACKOFF`** - Retry with increasing delays
- **`RetryStrategy.LLM_INFORMED`** - Inform LLM about the error and retry with additional context

### 3. LLM-Informed Error Recovery

When errors occur in LLM-dependent operations, the system can automatically:
- Inform the LLM about the specific error that occurred
- Use specialized prompts for error recovery (e.g., `entiti_continue_extraction`)
- Retry the operation with additional context

```python
from hirag import ErrorConfig, RetryConfig, RetryStrategy

error_config = ErrorConfig()
error_config.retry_config["entity_extraction"] = RetryConfig(
    max_retries=2,
    strategy=RetryStrategy.LLM_INFORMED,
    llm_error_prompt="entiti_continue_extraction",
    include_error_context=True
)

hirag = HiRAG(error_config=error_config)
```

### 4. Granular Configuration

Configure different behaviors for different operation types:

```python
from hirag import ErrorConfig, RetryConfig, RetryStrategy

error_config = ErrorConfig()

# Configure which operations should break the pipeline
error_config.should_break.update({
    "entity_extraction": False,     # Continue on entity extraction errors
    "text_chunking": True,          # Stop on text chunking errors
    "network_errors": False,        # Continue on network issues
    "validation_errors": True,      # Stop on validation failures
})

# Configure different retry strategies
error_config.retry_config.update({
    "network_operations": RetryConfig(
        max_retries=5,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_factor=2.0
    ),
    "llm_operations": RetryConfig(
        max_retries=3,
        strategy=RetryStrategy.LLM_INFORMED,
        llm_error_prompt="entiti_continue_extraction"
    ),
    "parsing_operations": RetryConfig(
        max_retries=2,
        strategy=RetryStrategy.DIRECT
    )
})

hirag = HiRAG(error_config=error_config)
```

## Usage Examples

### Example 1: Basic Error Handling with Context Manager

```python
from hirag import HiRAG, error_handling_context

hirag = HiRAG()

with error_handling_context("entity_extraction", default_return_value=[]) as ctx:
    entities = extract_entities_from_text(text)
    # If an error occurs, entities will be [] (the default_return_value)
```

### Example 2: Function Execution with Retry

```python
from hirag import with_error_handling, RetryStrategy

def parse_json_response(response_text):
    return json.loads(response_text)

with with_error_handling("parsing_operations", RetryStrategy.DIRECT, max_retries=2) as ctx:
    result = ctx.execute(parse_json_response, llm_response)
```

### Example 3: Async Operations with LLM-Informed Recovery

```python
from hirag import HiRAG, ErrorConfig, RetryConfig, RetryStrategy

async def extract_entities_with_llm(text):
    # LLM-based entity extraction
    pass

error_config = ErrorConfig()
error_config.retry_config["entity_extraction"] = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.LLM_INFORMED,
    llm_error_prompt="entiti_continue_extraction"
)

hirag = HiRAG(error_config=error_config)

# The error handler will automatically use LLM-informed recovery if extraction fails
entities = await hirag.error_handler.execute_with_retry(
    extract_entities_with_llm,
    args=(text,),
    context="entity_extraction"
)
```

### Example 4: Custom Retry Logic

```python
from hirag import ErrorConfig, RetryConfig, RetryStrategy

async def custom_retry_logic(func, args, kwargs, error, attempt):
    print(f"Custom retry attempt {attempt} after error: {error}")
    await asyncio.sleep(0.5 * attempt)  # Custom backoff
    return await func(*args, **kwargs)

error_config = ErrorConfig()
error_config.retry_config["custom_operation"] = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.CUSTOM,
    custom_retry_func=custom_retry_logic
)
```

## Integration with Existing Prompts

The system integrates seamlessly with HiRAG's existing prompt system:

- **`entiti_continue_extraction`** - Used when entity extraction fails
- **`entiti_if_loop_extraction`** - Used for parsing errors in entity extraction
- Custom prompts can be specified for any operation context

## Migration from Previous Error Handling

The new system is fully backward compatible. Existing code will continue to work with enhanced error handling automatically enabled. To opt into advanced features:

1. **Replace manual try/catch blocks** with context managers
2. **Configure retry strategies** for your specific use cases  
3. **Enable LLM-informed recovery** for operations that can benefit from it

## Default Configurations

The system includes sensible defaults:

**Pipeline-breaking operations (critical):**
- `text_chunking` - Required for pipeline to continue
- `storage_operations` - Data persistence critical
- `validation_errors` - Data integrity critical
- `configuration_errors` - System setup critical

**Non-breaking operations (recoverable):**
- `entity_extraction` - Can retry or skip individual entities
- `clustering` - Can use fallback algorithms
- `network_errors` - Can retry with backoff
- `parsing_errors` - Can use fallback parsing strategies

**Default retry strategies:**
- `llm_operations` - LLM-informed retry with `entiti_continue_extraction` prompt
- `entity_extraction` - LLM-informed retry 
- `network_errors` - Exponential backoff
- `api_rate_limits` - Exponential backoff with longer delays

This enhanced error handling system provides the modularity and granularity requested, with different retry mechanisms including LLM-informed recovery that leverages the existing prompt system.