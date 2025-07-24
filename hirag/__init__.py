from .hirag import HiRAG, QueryParam
from ._error_handling import (
    ErrorConfig, 
    HiRAGErrorHandler, 
    ErrorSeverity, 
    RetryStrategy,
    RetryConfig,
    ErrorHandlingContext,
    error_handling_context,
    async_error_handler,
    with_error_handling,
    configure_llm_integration
)

__version__ = "0.1.0"
__author__ = "Haoyu Huang"
__url__ = "https://github.com/hhy-huang/HiRAG"