"""
Error handling and logging utilities for HiRAG.
Provides configurable error handling for different pipeline stages.
"""

import logging
import traceback
import asyncio
import time
from typing import Dict, Optional, Callable, Any, Union, List, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from contextlib import asynccontextmanager, contextmanager

if TYPE_CHECKING:
    from ._llm import LLMFunction


class ErrorSeverity(Enum):
    """Error severity levels for HiRAG operations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Different retry strategies for error handling."""
    NONE = "none"                    # No retries
    DIRECT = "direct"               # Direct retry of the same operation
    LLM_INFORMED = "llm_informed"   # Inform LLM about error and retry
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Retry with exponential backoff
    CUSTOM = "custom"               # Custom retry logic


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    backoff_factor: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    custom_retry_func: Optional[Callable] = None
    llm_error_prompt: Optional[str] = None  # Prompt template for LLM-informed retries
    include_error_context: bool = True      # Whether to include error details in LLM prompts


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
    retry_config: Dict[str, RetryConfig] = field(default_factory=lambda: {
        "llm_operations": RetryConfig(
            max_retries=3, 
            backoff_factor=2,
            strategy=RetryStrategy.LLM_INFORMED,
            llm_error_prompt="entiti_continue_extraction"
        ),
        "embedding_operations": RetryConfig(max_retries=3, backoff_factor=2),
        "vector_operations": RetryConfig(max_retries=2, backoff_factor=1.5),
        "network_errors": RetryConfig(max_retries=5, backoff_factor=2),
        "api_rate_limits": RetryConfig(max_retries=5, backoff_factor=3),
        "entity_extraction": RetryConfig(
            max_retries=2,
            strategy=RetryStrategy.LLM_INFORMED,
            llm_error_prompt="entiti_continue_extraction"
        ),
        "parsing_errors": RetryConfig(
            max_retries=2,
            strategy=RetryStrategy.LLM_INFORMED,
            llm_error_prompt="entiti_if_loop_extraction"
        ),
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
        self._llm_func: Optional['LLMFunction'] = None
        self._prompts: Optional[Dict[str, str]] = None
        
    def set_llm_function(self, llm_func: 'LLMFunction') -> None:
        """Set the LLM function for LLM-informed retries."""
        self._llm_func = llm_func
        
    def set_prompts(self, prompts: Dict[str, str]) -> None:
        """Set prompts dictionary for LLM-informed retries."""
        self._prompts = prompts
        
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

    async def _execute_llm_informed_retry(
        self, 
        func: Callable, 
        args: tuple, 
        kwargs: dict,
        error: Exception,
        context: str,
        retry_config: RetryConfig
    ) -> Any:
        """Execute an LLM-informed retry attempt."""
        if not self._llm_func or not self._prompts:
            self.logger.warning(f"LLM-informed retry requested for {context} but LLM function or prompts not set")
            # Fall back to direct retry
            return await self._execute_direct_retry(func, args, kwargs, error, context, retry_config)
        
        prompt_key = retry_config.llm_error_prompt
        if not prompt_key or prompt_key not in self._prompts:
            self.logger.warning(f"LLM prompt '{prompt_key}' not found for context {context}")
            return await self._execute_direct_retry(func, args, kwargs, error, context, retry_config)
        
        try:
            # Prepare LLM prompt with error context
            prompt = self._prompts[prompt_key]
            if retry_config.include_error_context:
                error_context = f"\n\nPrevious error encountered: {type(error).__name__}: {str(error)}\n"
                prompt = error_context + prompt
            
            # Try to extract relevant parameters for LLM call
            # This is a simplified approach - in practice, this would need more sophisticated parameter extraction
            llm_kwargs = {}
            if 'input_text' in kwargs:
                llm_kwargs['input_text'] = kwargs['input_text']
            elif len(args) > 0 and isinstance(args[0], str):
                llm_kwargs['input_text'] = args[0]
            
            # Call LLM with error-informed prompt
            self.logger.info(f"Attempting LLM-informed retry for {context} using prompt '{prompt_key}'")
            return await self._llm_func(prompt, **llm_kwargs)
            
        except Exception as llm_error:
            self.logger.warning(f"LLM-informed retry failed for {context}: {llm_error}")
            # Fall back to direct retry
            return await self._execute_direct_retry(func, args, kwargs, error, context, retry_config)

    async def _execute_direct_retry(
        self, 
        func: Callable, 
        args: tuple, 
        kwargs: dict,
        error: Exception,
        context: str,
        retry_config: RetryConfig
    ) -> Any:
        """Execute a direct retry of the original function."""
        self.logger.info(f"Attempting direct retry for {context}")
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    async def _execute_retry_with_backoff(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        error: Exception,
        context: str,
        retry_config: RetryConfig,
        attempt: int
    ) -> Any:
        """Execute retry with exponential backoff."""
        delay = retry_config.backoff_factor ** attempt
        self.logger.info(f"Waiting {delay:.2f}s before retry attempt {attempt + 1} for {context}")
        await asyncio.sleep(delay)
        
        if retry_config.strategy == RetryStrategy.LLM_INFORMED:
            return await self._execute_llm_informed_retry(func, args, kwargs, error, context, retry_config)
        else:
            return await self._execute_direct_retry(func, args, kwargs, error, context, retry_config)

    async def execute_with_retry(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        context: str = "unknown",
        error_type: Optional[str] = None
    ) -> Any:
        """
        Execute a function with retry logic based on configuration.
        
        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            context: Context for error handling configuration
            error_type: Specific error type for configuration lookup
            
        Returns:
            Function result or raises exception if all retries failed
        """
        kwargs = kwargs or {}
        
        # Get retry configuration
        retry_config = self.config.retry_config.get(context)
        if not retry_config:
            retry_config = self.config.retry_config.get(error_type)
        if not retry_config:
            retry_config = RetryConfig()  # Use default
        
        last_error = None
        
        for attempt in range(retry_config.max_retries + 1):
            try:
                if attempt == 0:
                    # First attempt - direct execution
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                else:
                    # Retry attempts
                    if retry_config.strategy == RetryStrategy.NONE:
                        break
                    elif retry_config.strategy == RetryStrategy.CUSTOM and retry_config.custom_retry_func:
                        return await retry_config.custom_retry_func(func, args, kwargs, last_error, attempt)
                    elif retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                        return await self._execute_retry_with_backoff(func, args, kwargs, last_error, context, retry_config, attempt - 1)
                    elif retry_config.strategy == RetryStrategy.LLM_INFORMED:
                        return await self._execute_llm_informed_retry(func, args, kwargs, last_error, context, retry_config)
                    else:  # RetryStrategy.DIRECT
                        return await self._execute_direct_retry(func, args, kwargs, last_error, context, retry_config)
                        
            except Exception as e:
                last_error = e
                self.log_error(
                    e, 
                    context, 
                    operation_details=f"Attempt {attempt + 1}/{retry_config.max_retries + 1}"
                )
                
                if attempt == retry_config.max_retries:
                    # Final attempt failed
                    break
                    
        # All retries failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError(f"All retry attempts failed for {context}")
    
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


class ErrorHandlingContext:
    """Context manager for error handling blocks."""
    
    def __init__(
        self,
        context: str,
        error_handler_instance: Optional[HiRAGErrorHandler] = None,
        error_type: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        reraise_on_break: bool = True,
        default_return_value: Any = None,
        retry_enabled: bool = True
    ):
        self.context = context
        self.error_handler = error_handler_instance or get_error_handler()
        self.error_type = error_type
        self.severity = severity
        self.reraise_on_break = reraise_on_break
        self.default_return_value = default_return_value
        self.retry_enabled = retry_enabled
        self.result = None
        self.exception_occurred = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exception_occurred = True
            should_continue = self.error_handler.handle_operation_error(
                error=exc_val,
                context=self.context,
                severity=self.severity,
                error_type=self.error_type,
                reraise=self.reraise_on_break
            )
            if should_continue:
                # Suppress the exception and set default return value
                self.result = self.default_return_value
                return True
        return False

    async def execute_async(self, coro_func: Callable, *args, **kwargs):
        """Execute an async function within the error handling context with retry support."""
        if self.retry_enabled:
            try:
                self.result = await self.error_handler.execute_with_retry(
                    coro_func, args, kwargs, self.context, self.error_type
                )
                return self.result
            except Exception as e:
                self.exception_occurred = True
                should_continue = self.error_handler.handle_operation_error(
                    error=e,
                    context=self.context,
                    severity=self.severity,
                    error_type=self.error_type,
                    reraise=self.reraise_on_break
                )
                if should_continue:
                    self.result = self.default_return_value
                    return self.result
                else:
                    raise
        else:
            # No retry, direct execution
            try:
                self.result = await coro_func(*args, **kwargs)
                return self.result
            except Exception as e:
                self.exception_occurred = True
                should_continue = self.error_handler.handle_operation_error(
                    error=e,
                    context=self.context,
                    severity=self.severity,
                    error_type=self.error_type,
                    reraise=self.reraise_on_break
                )
                if should_continue:
                    self.result = self.default_return_value
                    return self.result
                else:
                    raise

    def execute(self, func: Callable, *args, **kwargs):
        """Execute a sync function within the error handling context with retry support."""
        if self.retry_enabled:
            try:
                # Convert to async for retry mechanism
                async def async_wrapper():
                    return func(*args, **kwargs)
                
                # Run the async wrapper
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                self.result = loop.run_until_complete(
                    self.error_handler.execute_with_retry(
                        async_wrapper, (), {}, self.context, self.error_type
                    )
                )
                return self.result
            except Exception as e:
                self.exception_occurred = True
                should_continue = self.error_handler.handle_operation_error(
                    error=e,
                    context=self.context,
                    severity=self.severity,
                    error_type=self.error_type,
                    reraise=self.reraise_on_break
                )
                if should_continue:
                    self.result = self.default_return_value
                    return self.result
                else:
                    raise
        else:
            # No retry, direct execution
            try:
                self.result = func(*args, **kwargs)
                return self.result
            except Exception as e:
                self.exception_occurred = True
                should_continue = self.error_handler.handle_operation_error(
                    error=e,
                    context=self.context,
                    severity=self.severity,
                    error_type=self.error_type,
                    reraise=self.reraise_on_break
                )
                if should_continue:
                    self.result = self.default_return_value
                    return self.result
                else:
                    raise


@asynccontextmanager
async def async_error_handler(
    context: str,
    error_handler_instance: Optional[HiRAGErrorHandler] = None,
    error_type: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    reraise_on_break: bool = True,
    default_return_value: Any = None,
    retry_enabled: bool = True
):
    """Async context manager for error handling blocks."""
    handler_ctx = ErrorHandlingContext(
        context=context,
        error_handler_instance=error_handler_instance,
        error_type=error_type,
        severity=severity,
        reraise_on_break=reraise_on_break,
        default_return_value=default_return_value,
        retry_enabled=retry_enabled
    )
    
    try:
        yield handler_ctx
    except Exception as e:
        handler_ctx.exception_occurred = True
        should_continue = handler_ctx.error_handler.handle_operation_error(
            error=e,
            context=context,
            severity=severity,
            error_type=error_type,
            reraise=reraise_on_break
        )
        if should_continue:
            handler_ctx.result = default_return_value
        else:
            raise


@contextmanager
def error_handling_context(
    context: str,
    error_handler_instance: Optional[HiRAGErrorHandler] = None,
    error_type: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    reraise_on_break: bool = True,
    default_return_value: Any = None,
    retry_enabled: bool = True
):
    """Synchronous context manager for error handling blocks."""
    handler_ctx = ErrorHandlingContext(
        context=context,
        error_handler_instance=error_handler_instance,
        error_type=error_type,
        severity=severity,
        reraise_on_break=reraise_on_break,
        default_return_value=default_return_value,
        retry_enabled=retry_enabled
    )
    
    try:
        yield handler_ctx
    except Exception as e:
        handler_ctx.exception_occurred = True
        should_continue = handler_ctx.error_handler.handle_operation_error(
            error=e,
            context=context,
            severity=severity,
            error_type=error_type,
            reraise=reraise_on_break
        )
        if should_continue:
            handler_ctx.result = default_return_value
        else:
            raise


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


def configure_llm_integration(llm_func: 'LLMFunction', prompts: Dict[str, str]) -> None:
    """Configure LLM integration for error handling."""
    handler = get_error_handler()
    handler.set_llm_function(llm_func)
    handler.set_prompts(prompts)


# Convenience functions for creating error handling contexts
def create_error_context(
    context: str,
    **kwargs
) -> ErrorHandlingContext:
    """Create an error handling context with the given parameters."""
    return ErrorHandlingContext(context=context, **kwargs)


def with_error_handling(
    context: str,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    max_retries: int = 3,
    **kwargs
) -> ErrorHandlingContext:
    """
    Convenience function to create error handling context with common retry settings.
    
    Usage:
        with with_error_handling("entity_extraction", RetryStrategy.LLM_INFORMED) as ctx:
            result = ctx.execute(my_function, arg1, arg2)
    """
    handler = get_error_handler()
    
    # Update retry configuration for this context
    if context not in handler.config.retry_config:
        handler.config.retry_config[context] = RetryConfig()
    
    handler.config.retry_config[context].strategy = retry_strategy
    handler.config.retry_config[context].max_retries = max_retries
    
    return ErrorHandlingContext(context=context, **kwargs)