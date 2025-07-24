"""
Error handling and logging utilities for HiRAG.
Provides configurable error handling for different pipeline stages.
"""

import logging
import traceback
from typing import Dict, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps


class ErrorSeverity(Enum):
    """Error severity levels for HiRAG operations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorConfig:
    """Configuration for error handling behavior in different pipeline stages."""
    
    # Pipeline stage configurations - whether errors should break the pipeline
    should_break: Dict[str, bool] = field(default_factory=lambda: {
        # Core pipeline stages
        "document_processing": False,     # Document hashing, deduplication
        "text_chunking": True,           # Text chunking operations
        "entity_extraction": False,       # Entity extraction from text
        "graph_operations": False,        # Graph construction and manipulation
        "clustering": False,              # Community clustering
        "community_reports": False,       # Community report generation
        "vector_operations": False,       # Vector database operations
        "storage_operations": True,       # Critical storage failures
        "llm_operations": False,          # LLM API calls
        "embedding_operations": False,    # Embedding generation
        
        # Specific error types
        "network_errors": False,          # Network connectivity issues
        "api_rate_limits": False,         # API rate limiting
        "parsing_errors": False,          # JSON/data parsing failures
        "validation_errors": True,        # Data validation failures
        "configuration_errors": True,     # Invalid configuration
        "authentication_errors": True,    # Auth failures
    })
    
    # Retry configurations for different operation types
    retry_config: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "llm_operations": {"max_retries": 3, "backoff_factor": 2},
        "embedding_operations": {"max_retries": 3, "backoff_factor": 2},
        "vector_operations": {"max_retries": 2, "backoff_factor": 1.5},
        "network_errors": {"max_retries": 5, "backoff_factor": 2},
        "api_rate_limits": {"max_retries": 5, "backoff_factor": 3},
    })
    
    # Whether to log full stack traces for different severity levels
    log_stack_traces: Dict[str, bool] = field(default_factory=lambda: {
        "info": False,
        "warning": False,
        "error": True,
        "critical": True,
    })


class HiRAGErrorHandler:
    """Centralized error handling for HiRAG operations."""
    
    def __init__(self, config: Optional[ErrorConfig] = None, logger_name: str = "HiRAG"):
        self.config = config or ErrorConfig()
        self.logger = logging.getLogger(logger_name)
        self._error_counts = {}
        
    def should_break_pipeline(self, context: str, error_type: Optional[str] = None) -> bool:
        """
        Determine if an error should break the pipeline based on configuration.
        
        Args:
            context: The pipeline stage or context where the error occurred
            error_type: Optional specific error type for more granular control
            
        Returns:
            True if the pipeline should stop, False if it should continue
        """
        # Check specific error type first if provided
        if error_type and error_type in self.config.should_break:
            return self.config.should_break[error_type]
            
        # Fall back to context-based configuration
        return self.config.should_break.get(context, True)  # Default to breaking on unknown contexts
    
    def log_error(
        self, 
        error: Exception, 
        context: str, 
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        additional_info: Optional[Dict[str, Any]] = None,
        operation_details: Optional[str] = None
    ) -> None:
        """
        Log an error with proper context and traceability.
        
        Args:
            error: The exception that occurred
            context: The pipeline stage or context where the error occurred
            severity: The severity level of the error
            additional_info: Additional context information
            operation_details: Specific details about the operation that failed
        """
        # Increment error count for this context
        self._error_counts[context] = self._error_counts.get(context, 0) + 1
        
        # Build error message with context
        error_msg = f"[{context.upper()}] {type(error).__name__}: {str(error)}"
        
        if operation_details:
            error_msg += f" | Operation: {operation_details}"
            
        if additional_info:
            info_str = ", ".join([f"{k}={v}" for k, v in additional_info.items()])
            error_msg += f" | Context: {info_str}"
            
        error_msg += f" | Error count for {context}: {self._error_counts[context]}"
        
        # Log based on severity
        log_func = getattr(self.logger, severity.value)
        
        if self.config.log_stack_traces.get(severity.value, False):
            error_msg += f"\nStack trace:\n{traceback.format_exc()}"
            
        log_func(error_msg)
        
    def handle_operation_error(
        self,
        error: Exception,
        context: str,
        operation_details: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        error_type: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        reraise: bool = True
    ) -> bool:
        """
        Handle an error that occurred during a pipeline operation.
        
        Args:
            error: The exception that occurred
            context: The pipeline stage or context where the error occurred  
            operation_details: Specific details about the operation that failed
            severity: The severity level of the error
            error_type: Optional specific error type for configuration lookup
            additional_info: Additional context information
            reraise: Whether to reraise the exception if pipeline should break
            
        Returns:
            True if the operation should continue, False if it should stop
            
        Raises:
            The original exception if configured to break pipeline and reraise=True
        """
        # Log the error with full context
        self.log_error(error, context, severity, additional_info, operation_details)
        
        # Determine if we should break the pipeline
        should_break = self.should_break_pipeline(context, error_type)
        
        if should_break:
            self.logger.critical(f"PIPELINE STOPPED: Critical error in {context}")
            if reraise:
                raise error
            return False
        else:
            self.logger.info(f"PIPELINE CONTINUING: Non-critical error in {context}")
            return True
            
    def get_error_summary(self) -> Dict[str, int]:
        """Get a summary of error counts by context."""
        return self._error_counts.copy()


def error_handler(
    context: str,
    error_handler_instance: Optional[HiRAGErrorHandler] = None,
    operation_details: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    error_type: Optional[str] = None,
    reraise_on_break: bool = True,
    default_return_value: Any = None
):
    """
    Decorator for adding error handling to functions.
    
    Args:
        context: The pipeline context for error handling configuration
        error_handler_instance: Optional error handler instance to use
        operation_details: Description of the operation being performed  
        severity: Default severity level for errors
        error_type: Optional specific error type for configuration
        reraise_on_break: Whether to reraise exceptions when pipeline should break
        default_return_value: Value to return if error occurs and pipeline continues
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = error_handler_instance or _get_default_error_handler()
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                should_continue = handler.handle_operation_error(
                    error=e,
                    context=context,
                    operation_details=operation_details or func.__name__,
                    severity=severity,
                    error_type=error_type,
                    reraise=reraise_on_break
                )
                if should_continue:
                    return default_return_value
                # If we get here, reraise_on_break was False but pipeline should break
                return default_return_value
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            handler = error_handler_instance or _get_default_error_handler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                should_continue = handler.handle_operation_error(
                    error=e,
                    context=context,
                    operation_details=operation_details or func.__name__,
                    severity=severity,
                    error_type=error_type,
                    reraise=reraise_on_break
                )
                if should_continue:
                    return default_return_value
                # If we get here, reraise_on_break was False but pipeline should break
                return default_return_value
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Global error handler instance
_global_error_handler: Optional[HiRAGErrorHandler] = None


def get_error_handler() -> HiRAGErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = HiRAGErrorHandler()
    return _global_error_handler


def _get_default_error_handler() -> HiRAGErrorHandler:
    """Internal function to get default error handler."""
    return get_error_handler()


def set_error_handler(handler: HiRAGErrorHandler) -> None:
    """Set the global error handler instance."""
    global _global_error_handler
    _global_error_handler = handler


def configure_error_handling(config: ErrorConfig) -> None:
    """Configure the global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = HiRAGErrorHandler(config)
    else:
        _global_error_handler.config = config