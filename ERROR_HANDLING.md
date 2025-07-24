# HiRAG Error Handling & Logging Documentation

## Overview

HiRAG now includes a comprehensive error handling and logging system that provides:

- **Configurable pipeline behavior**: Control which errors break the pipeline vs. continue
- **Structured error logging**: All errors are logged with context and traceability
- **Automatic error recovery**: Sensible fallback values when operations fail
- **Granular control**: Different error handling for different pipeline stages

## Key Features

### 1. Configurable Error Handling

Use the `should_break` configuration to control which errors stop the pipeline:

```python
from hirag import HiRAG
from hirag._error_handling import ErrorConfig

# Create custom error configuration
error_config = ErrorConfig()
error_config.should_break.update({
    "entity_extraction": False,     # Continue on entity extraction errors
    "text_chunking": True,          # Stop on text chunking errors  
    "network_errors": False,        # Continue on network issues
    "validation_errors": True,      # Stop on validation failures
})

# Initialize HiRAG with error configuration
hirag = HiRAG(error_config=error_config)
```

### 2. Dynamic Configuration

Configure error handling after initialization:

```python
hirag = HiRAG()

# Configure which errors should break the pipeline
hirag.configure_error_handling({
    "entity_extraction": True,   # Now break on entity extraction errors
    "clustering": False,         # Continue on clustering errors
})

# Get error summary
error_summary = hirag.get_error_summary()
print(f"Error counts: {error_summary}")
```

### 3. Default Configuration

The system comes with sensible defaults:

```python
# Pipeline stages that BREAK on errors (critical):
- text_chunking: True
- storage_operations: True  
- validation_errors: True
- configuration_errors: True
- authentication_errors: True

# Pipeline stages that CONTINUE on errors (non-critical):
- document_processing: False
- entity_extraction: False
- graph_operations: False
- clustering: False
- community_reports: False
- vector_operations: False
- llm_operations: False
- embedding_operations: False
- network_errors: False
- api_rate_limits: False
- parsing_errors: False
```

## Pipeline Stage Error Handling

### Document Processing
- **Context**: `document_processing`
- **Default**: Continue on errors
- **Includes**: Document hashing, deduplication, basic validation

### Text Chunking
- **Context**: `text_chunking`  
- **Default**: Break on errors (critical for pipeline)
- **Includes**: Token encoding, chunk generation, chunk validation

### Entity Extraction
- **Context**: `entity_extraction`
- **Default**: Continue on errors
- **Includes**: LLM-based entity extraction, entity validation, entity processing

### Graph Operations
- **Context**: `graph_operations`
- **Default**: Continue on errors
- **Includes**: Graph construction, node/edge operations, graph queries

### Clustering
- **Context**: `clustering`
- **Default**: Continue on errors
- **Includes**: Community detection, hierarchical clustering

### Community Reports
- **Context**: `community_reports`
- **Default**: Continue on errors
- **Includes**: Community report generation, report formatting

### Vector Operations
- **Context**: `vector_operations`
- **Default**: Continue on errors
- **Includes**: Embedding generation, vector database operations, similarity search

### Storage Operations
- **Context**: `storage_operations`
- **Default**: Break on errors (critical)
- **Includes**: File I/O, database operations, data persistence

### LLM Operations
- **Context**: `llm_operations`
- **Default**: Continue on errors
- **Includes**: API calls to language models, response processing

## Error Types

### Network Errors
- **Type**: `network_errors`
- **Default**: Continue
- **Examples**: Connection timeouts, DNS failures, API unreachable

### API Rate Limits
- **Type**: `api_rate_limits`
- **Default**: Continue (with retry)
- **Examples**: OpenAI rate limiting, API quota exceeded

### Parsing Errors
- **Type**: `parsing_errors`
- **Default**: Continue
- **Examples**: JSON parsing failures, malformed responses

### Validation Errors
- **Type**: `validation_errors`
- **Default**: Break (critical)
- **Examples**: Invalid entity format, schema validation failures

### Configuration Errors
- **Type**: `configuration_errors`
- **Default**: Break (critical)
- **Examples**: Invalid settings, missing required configuration

### Authentication Errors
- **Type**: `authentication_errors`
- **Default**: Break (critical)
- **Examples**: Invalid API keys, authorization failures

## Error Logging

All errors are logged with comprehensive context:

```
[CONTEXT] ErrorType: Error message | Operation: operation_details | Context: key=value | Error count for context: N
```

Example:
```
[ENTITY_EXTRACTION] ValueError: Invalid entity format | Operation: Entity extraction from chunk-123 | Context: chunk_id=chunk-123 | Error count for entity_extraction: 1
```

### Log Levels

- **INFO**: Operational information, non-error events
- **WARNING**: Recoverable errors, fallback actions taken
- **ERROR**: Errors that occurred but pipeline continues
- **CRITICAL**: Errors that stop the pipeline

### Stack Traces

Stack traces are logged for ERROR and CRITICAL levels by default. Configure this:

```python
error_config = ErrorConfig()
error_config.log_stack_traces.update({
    "warning": True,   # Also log stack traces for warnings
    "error": False,    # Don't log stack traces for errors
})
```

## Function-Level Error Handling

Use the `@error_handler` decorator for automatic error handling:

```python
from hirag._error_handling import error_handler, ErrorSeverity

@error_handler(
    context="custom_operation",
    severity=ErrorSeverity.WARNING,
    reraise_on_break=False,
    default_return_value="fallback"
)
async def my_custom_function():
    # Function that might fail
    raise ValueError("Something went wrong")
    
# Usage
result = await my_custom_function()  # Returns "fallback" if error occurs
```

## Best Practices

### 1. Configure Based on Criticality

Set `should_break=True` for errors that make the pipeline meaningless:

```python
# Critical errors that should stop the pipeline
error_config.should_break.update({
    "storage_operations": True,     # Can't continue without storage
    "text_chunking": True,          # Can't continue without chunks
    "validation_errors": True,      # Invalid data is problematic
})
```

### 2. Allow Retries for Transient Errors

Set `should_break=False` for transient errors:

```python
# Non-critical errors that should allow continuation
error_config.should_break.update({
    "network_errors": False,        # Retry or skip individual operations
    "api_rate_limits": False,       # Can retry or use cached results
    "parsing_errors": False,        # Can use fallback parsing
})
```

### 3. Monitor Error Patterns

Regularly check error summaries to identify issues:

```python
# After processing
error_summary = hirag.get_error_summary()
if error_summary.get("network_errors", 0) > 10:
    print("High number of network errors - check connectivity")
```

### 4. Use Appropriate Severity Levels

- Use `ErrorSeverity.INFO` for debugging information
- Use `ErrorSeverity.WARNING` for recoverable issues
- Use `ErrorSeverity.ERROR` for significant problems
- Use `ErrorSeverity.CRITICAL` for pipeline-stopping issues

## Example: Complete Error Handling Setup

```python
from hirag import HiRAG
from hirag._error_handling import ErrorConfig

# Create custom configuration
error_config = ErrorConfig()

# Configure for development (more lenient)
error_config.should_break.update({
    "entity_extraction": False,      # Continue on extraction errors during development
    "clustering": False,             # Continue on clustering errors
    "community_reports": False,      # Continue on report generation errors
    "network_errors": False,         # Continue on network issues
    "api_rate_limits": False,        # Continue on rate limits
})

# Configure for production (more strict)
# error_config.should_break.update({
#     "entity_extraction": True,      # Stop on extraction errors in production
#     "validation_errors": True,      # Always stop on validation errors
#     "storage_operations": True,     # Always stop on storage errors
# })

# Initialize HiRAG
hirag = HiRAG(
    error_config=error_config,
    enable_comprehensive_error_logging=True
)

# Process documents
try:
    hirag.insert([
        "Document 1 content...",
        "Document 2 content...",
    ])
    
    # Check for errors
    errors = hirag.get_error_summary()
    if errors:
        print(f"Processing completed with errors: {errors}")
    else:
        print("Processing completed successfully!")
        
except Exception as e:
    print(f"Critical error stopped processing: {e}")
```

## Migration Guide

If you're upgrading from an older version of HiRAG:

1. **No breaking changes**: Existing code will work with default error handling
2. **Optional configuration**: Add error configuration if you want custom behavior
3. **Enhanced logging**: You'll see more detailed error messages automatically
4. **Error summaries**: Use `get_error_summary()` to monitor pipeline health

The error handling system is designed to be backward-compatible while providing enhanced control and visibility for users who need it.