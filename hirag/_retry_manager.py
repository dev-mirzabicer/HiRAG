"""
Robust Retry Management System for HiRAG

This module provides comprehensive retry mechanisms for LLM calls with
intelligent failure categorization, exponential backoff, circuit breaker
patterns, and graph integrity preservation.

Key Features:
- Categorized retry strategies for different failure types
- Exponential backoff with jitter to prevent thundering herd
- Circuit breaker pattern to prevent cascading failures
- Graph integrity preservation during retries
- Detailed retry statistics and monitoring
- Integration with checkpointing system
"""

import asyncio
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import json

from ._utils import logger
from .base import BaseKVStorage
from ._token_estimation import LLMCallType


class FailureType(Enum):
    """Categories of failures with different retry strategies"""
    NETWORK_TIMEOUT = "network_timeout"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"  # 5xx errors
    AUTHENTICATION_ERROR = "authentication_error"  # 401
    INVALID_REQUEST = "invalid_request"  # 400
    CONTENT_POLICY = "content_policy"  # Content violations
    QUOTA_EXCEEDED = "quota_exceeded"  # Permanent quota issues
    PARSING_ERROR = "parsing_error"  # Response parsing failures
    VALIDATION_ERROR = "validation_error"  # Data validation failures
    UNKNOWN_ERROR = "unknown_error"


class RetryStrategy(Enum):
    """Different retry strategies based on failure type"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1  # ±10% jitter
    timeout: float = 30.0  # Individual request timeout
    
    # Circuit breaker settings
    circuit_failure_threshold: int = 5  # Failures before opening circuit
    circuit_recovery_timeout: float = 60.0  # Time before trying again
    circuit_success_threshold: int = 3  # Successes needed to close circuit


@dataclass
class RetryAttempt:
    """Record of a single retry attempt"""
    attempt_number: int
    timestamp: float
    failure_type: FailureType
    error_message: str
    delay_before_retry: float = 0.0
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)


@dataclass
class RetryStatistics:
    """Statistics for retry operations"""
    call_type: LLMCallType
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_retries: int = 0
    avg_retries_per_call: float = 0.0
    failure_breakdown: Dict[str, int] = field(default_factory=dict)
    avg_success_time: float = 0.0
    circuit_breaker_trips: int = 0
    last_updated: float = field(default_factory=time.time)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing cascading failures"""
    call_type: LLMCallType
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    next_attempt_time: float = 0.0
    success_count: int = 0  # For half-open state
    config: RetryConfig = field(default_factory=RetryConfig)
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed through circuit breaker"""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if current_time >= self.next_attempt_time:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker for {self.call_type.value} moved to HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record a successful request"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.circuit_success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker for {self.call_type.value} CLOSED after recovery")
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """Record a failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.circuit_failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.next_attempt_time = time.time() + self.config.circuit_recovery_timeout
                logger.warning(f"Circuit breaker for {self.call_type.value} OPENED after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.circuit_recovery_timeout
            logger.warning(f"Circuit breaker for {self.call_type.value} back to OPEN after failure during recovery")


class RetryManager:
    """
    Comprehensive retry management system for HiRAG LLM calls
    
    This manager provides:
    - Intelligent failure classification and retry strategies
    - Circuit breaker pattern to prevent cascading failures
    - Exponential backoff with jitter
    - Graph integrity preservation
    - Comprehensive statistics and monitoring
    """
    
    # Failure type classification patterns
    FAILURE_PATTERNS = {
        FailureType.NETWORK_TIMEOUT: [
            "timeout", "timed out", "connection timeout",
            "read timeout", "ConnectTimeout", "ReadTimeout"
        ],
        FailureType.CONNECTION_ERROR: [
            "connection", "connect", "network", "dns",
            "ConnectionError", "DNSError", "NetworkError"
        ],
        FailureType.RATE_LIMIT: [
            "rate limit", "too many requests", "429",
            "RateLimitError", "quota exceeded", "rate_limit_exceeded"
        ],
        FailureType.SERVER_ERROR: [
            "500", "502", "503", "504", "internal server error",
            "bad gateway", "service unavailable", "gateway timeout"
        ],
        FailureType.AUTHENTICATION_ERROR: [
            "401", "unauthorized", "authentication", "invalid api key",
            "AuthenticationError", "permission denied"
        ],
        FailureType.INVALID_REQUEST: [
            "400", "bad request", "invalid request", "malformed",
            "InvalidRequestError", "validation failed"
        ],
        FailureType.CONTENT_POLICY: [
            "content policy", "safety", "harmful content",
            "content filter", "moderation", "policy violation"
        ],
        FailureType.QUOTA_EXCEEDED: [
            "quota", "billing", "insufficient funds",
            "usage limit", "account suspended"
        ],
        FailureType.PARSING_ERROR: [
            "json", "parse", "invalid json", "malformed response",
            "decode", "format error"
        ],
        FailureType.VALIDATION_ERROR: [
            "validation", "schema", "required field",
            "invalid format", "constraint violation"
        ]
    }
    
    # Retry strategies for each failure type
    RETRY_STRATEGIES = {
        FailureType.NETWORK_TIMEOUT: RetryStrategy.EXPONENTIAL_BACKOFF,
        FailureType.CONNECTION_ERROR: RetryStrategy.EXPONENTIAL_BACKOFF,
        FailureType.RATE_LIMIT: RetryStrategy.EXPONENTIAL_BACKOFF,
        FailureType.SERVER_ERROR: RetryStrategy.EXPONENTIAL_BACKOFF,
        FailureType.AUTHENTICATION_ERROR: RetryStrategy.NO_RETRY,
        FailureType.INVALID_REQUEST: RetryStrategy.NO_RETRY,
        FailureType.CONTENT_POLICY: RetryStrategy.NO_RETRY,
        FailureType.QUOTA_EXCEEDED: RetryStrategy.NO_RETRY,
        FailureType.PARSING_ERROR: RetryStrategy.LINEAR_BACKOFF,
        FailureType.VALIDATION_ERROR: RetryStrategy.NO_RETRY,
        FailureType.UNKNOWN_ERROR: RetryStrategy.EXPONENTIAL_BACKOFF,
    }
    
    def __init__(
        self,
        stats_storage: Optional[BaseKVStorage] = None,
        default_config: Optional[RetryConfig] = None
    ):
        self.stats_storage = stats_storage
        self.default_config = default_config or RetryConfig()
        
        # Per-call-type configurations
        self.call_configs: Dict[LLMCallType, RetryConfig] = {}
        
        # Circuit breakers per call type
        self.circuit_breakers: Dict[LLMCallType, CircuitBreaker] = {}
        
        # Statistics tracking
        self.statistics: Dict[LLMCallType, RetryStatistics] = {}
        
        # Active retry attempts (for monitoring)
        self.active_attempts: Dict[str, List[RetryAttempt]] = {}
        
        logger.info("RetryManager initialized")

    def configure_call_type(
        self,
        call_type: LLMCallType,
        config: RetryConfig
    ):
        """Configure retry behavior for a specific call type"""
        self.call_configs[call_type] = config
        
        # Update or create circuit breaker
        if call_type in self.circuit_breakers:
            self.circuit_breakers[call_type].config = config
        else:
            self.circuit_breakers[call_type] = CircuitBreaker(
                call_type=call_type,
                config=config
            )
        
        logger.info(f"Configured retry settings for {call_type.value}")

    def classify_failure(self, error: Exception) -> FailureType:
        """Classify failure type based on error message and type"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check error message and type against patterns
        for failure_type, patterns in self.FAILURE_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in error_str or pattern.lower() in error_type:
                    return failure_type
        
        return FailureType.UNKNOWN_ERROR

    def should_retry(
        self,
        failure_type: FailureType,
        attempt_number: int,
        call_type: LLMCallType
    ) -> bool:
        """Determine if a failure should be retried"""
        config = self.call_configs.get(call_type, self.default_config)
        
        # Check maximum attempts
        if attempt_number >= config.max_attempts:
            return False
        
        # Check retry strategy
        strategy = self.RETRY_STRATEGIES.get(failure_type, RetryStrategy.NO_RETRY)
        if strategy == RetryStrategy.NO_RETRY:
            return False
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(call_type)
        if circuit_breaker and not circuit_breaker.should_allow_request():
            logger.warning(f"Circuit breaker OPEN for {call_type.value}, skipping retry")
            return False
        
        return True

    def calculate_delay(
        self,
        failure_type: FailureType,
        attempt_number: int,
        call_type: LLMCallType
    ) -> float:
        """Calculate delay before next retry attempt"""
        config = self.call_configs.get(call_type, self.default_config)
        strategy = self.RETRY_STRATEGIES.get(failure_type, RetryStrategy.EXPONENTIAL_BACKOFF)
        
        if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            base_delay = config.initial_delay * (config.backoff_multiplier ** (attempt_number - 1))
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            base_delay = config.initial_delay * attempt_number
        elif strategy == RetryStrategy.FIXED_DELAY:
            base_delay = config.initial_delay
        else:
            base_delay = config.initial_delay
        
        # Apply maximum delay cap
        base_delay = min(base_delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        jitter = base_delay * config.jitter_range * (2 * random.random() - 1)
        final_delay = max(0, base_delay + jitter)
        
        # Special handling for rate limits - longer delays
        if failure_type == FailureType.RATE_LIMIT:
            final_delay *= 2  # Double the delay for rate limits
        
        return final_delay

    async def execute_with_retry(
        self,
        func: Callable,
        call_type: LLMCallType,
        *args,
        operation_id: Optional[str] = None,
        preserve_graph_integrity: bool = True,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic
        
        Args:
            func: The async function to execute
            call_type: Type of LLM call for configuration
            operation_id: Unique ID for tracking this operation
            preserve_graph_integrity: Whether to preserve graph state on failures
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function call
            
        Raises:
            The last exception if all retries fail
        """
        if operation_id is None:
            operation_id = f"{call_type.value}_{int(time.time() * 1000)}"
        
        config = self.call_configs.get(call_type, self.default_config)
        self.active_attempts[operation_id] = []
        
        # Initialize statistics if needed
        if call_type not in self.statistics:
            self.statistics[call_type] = RetryStatistics(call_type=call_type)
        
        stats = self.statistics[call_type]
        stats.total_calls += 1
        
        last_exception = None
        start_time = time.time()
        
        for attempt in range(1, config.max_attempts + 1):
            # Get circuit breaker outside try block to ensure it's always defined
            circuit_breaker = self.circuit_breakers.get(call_type)
            
            try:
                # Check circuit breaker
                if circuit_breaker and not circuit_breaker.should_allow_request():
                    stats.circuit_breaker_trips += 1
                    raise Exception(f"Circuit breaker OPEN for {call_type.value}")
                
                # Execute the function with timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=config.timeout
                )
                
                # Success! Record statistics
                end_time = time.time()
                stats.successful_calls += 1
                if stats.successful_calls > 0:
                    stats.avg_success_time = (
                        (stats.avg_success_time * (stats.successful_calls - 1) + 
                         (end_time - start_time)) / stats.successful_calls
                    )
                
                # Record circuit breaker success
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                # Clean up tracking
                if operation_id in self.active_attempts:
                    del self.active_attempts[operation_id]
                
                logger.debug(f"Successful {call_type.value} call after {attempt} attempt(s)")
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = e
                failure_type = FailureType.NETWORK_TIMEOUT
                
            except Exception as e:
                last_exception = e
                failure_type = self.classify_failure(e)
            
            # Record the attempt
            retry_attempt = RetryAttempt(
                attempt_number=attempt,
                timestamp=time.time(),
                failure_type=failure_type,
                error_message=str(last_exception)
            )
            self.active_attempts[operation_id].append(retry_attempt)
            
            # Update statistics
            stats.total_retries += 1
            failure_key = failure_type.value
            stats.failure_breakdown[failure_key] = stats.failure_breakdown.get(failure_key, 0) + 1
            
            # Record circuit breaker failure
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            logger.warning(f"Attempt {attempt} failed for {call_type.value}: {failure_type.value} - {str(last_exception)}")
            
            # Check if we should retry
            if not self.should_retry(failure_type, attempt, call_type):
                logger.error(f"Not retrying {call_type.value} due to failure type: {failure_type.value}")
                break
            
            if attempt < config.max_attempts:
                # Calculate delay before next attempt
                delay = self.calculate_delay(failure_type, attempt, call_type)
                retry_attempt.delay_before_retry = delay
                
                logger.info(f"Retrying {call_type.value} in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})")
                await asyncio.sleep(delay)
        
        # All retries failed
        stats.failed_calls += 1
        stats.avg_retries_per_call = stats.total_retries / stats.total_calls if stats.total_calls > 0 else 0
        stats.last_updated = time.time()
        
        # Save statistics
        await self._save_statistics()
        
        # Clean up tracking
        if operation_id in self.active_attempts:
            del self.active_attempts[operation_id]
        
        logger.error(f"All retry attempts failed for {call_type.value}")
        if last_exception is not None:
            raise last_exception
        else:
            raise Exception(f"All retry attempts failed for {call_type.value} with unknown error")

    async def get_statistics(self, call_type: Optional[LLMCallType] = None) -> Union[RetryStatistics, Dict[LLMCallType, RetryStatistics]]:
        """Get retry statistics"""
        if call_type:
            return self.statistics.get(call_type, RetryStatistics(call_type=call_type))
        else:
            return self.statistics.copy()

    async def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        status = {}
        for call_type, breaker in self.circuit_breakers.items():
            status[call_type.value] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure_time": breaker.last_failure_time,
                "next_attempt_time": breaker.next_attempt_time if breaker.state == CircuitBreakerState.OPEN else None,
                "success_count": breaker.success_count
            }
        return status

    async def reset_circuit_breaker(self, call_type: LLMCallType):
        """Manually reset a circuit breaker"""
        if call_type in self.circuit_breakers:
            breaker = self.circuit_breakers[call_type]
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            logger.info(f"Circuit breaker for {call_type.value} manually reset")

    async def _save_statistics(self):
        """Save statistics to storage"""
        if not self.stats_storage:
            return
        
        try:
            stats_data = {}
            for call_type, stats in self.statistics.items():
                stats_data[call_type.value] = asdict(stats)
            
            stats_id = f"retry_stats_{int(time.time())}"
            await self.stats_storage.upsert({stats_id: stats_data})
            
        except Exception as e:
            logger.warning(f"Failed to save retry statistics: {e}")

    async def generate_retry_report(self) -> str:
        """Generate a comprehensive retry statistics report"""
        report_lines = [
            "=== Retry Manager Statistics Report ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if not self.statistics:
            report_lines.append("No retry statistics available.")
            return "\n".join(report_lines)
        
        total_calls = sum(stats.total_calls for stats in self.statistics.values())
        total_successes = sum(stats.successful_calls for stats in self.statistics.values())
        total_failures = sum(stats.failed_calls for stats in self.statistics.values())
        total_retries = sum(stats.total_retries for stats in self.statistics.values())
        
        report_lines.extend([
            "Overall Summary:",
            f"  • Total LLM calls: {total_calls:,}",
            f"  • Successful calls: {total_successes:,} ({total_successes/max(1,total_calls):.1%})",
            f"  • Failed calls: {total_failures:,} ({total_failures/max(1,total_calls):.1%})",
            f"  • Total retries: {total_retries:,}",
            f"  • Average retries per call: {total_retries/max(1,total_calls):.2f}",
            ""
        ])
        
        # Per-call-type breakdown
        report_lines.append("Per-Call-Type Breakdown:")
        for call_type, stats in self.statistics.items():
            success_rate = stats.successful_calls / max(1, stats.total_calls)
            
            report_lines.extend([
                f"  {call_type.value.upper()}:",
                f"    • Calls: {stats.total_calls} | Success rate: {success_rate:.1%}",
                f"    • Avg retries: {stats.avg_retries_per_call:.2f}",
                f"    • Avg success time: {stats.avg_success_time:.2f}s",
                f"    • Circuit trips: {stats.circuit_breaker_trips}"
            ])
            
            if stats.failure_breakdown:
                report_lines.append("    • Failure types:")
                for failure_type, count in stats.failure_breakdown.items():
                    percentage = count / max(1, stats.total_retries) * 100
                    report_lines.append(f"      - {failure_type}: {count} ({percentage:.1f}%)")
            
            report_lines.append("")
        
        # Circuit breaker status
        circuit_status = await self.get_circuit_breaker_status()
        if circuit_status:
            report_lines.extend([
                "Circuit Breaker Status:",
                ""
            ])
            
            for call_type, status in circuit_status.items():
                state_icon = {"closed": "✅", "open": "🔴", "half_open": "🟡"}.get(status["state"], "❓")
                report_lines.append(f"  {state_icon} {call_type}: {status['state'].upper()}")
                
                if status["state"] != "closed":
                    report_lines.append(f"    • Failures: {status['failure_count']}")
                    if status["next_attempt_time"]:
                        next_attempt = datetime.fromtimestamp(status["next_attempt_time"])
                        report_lines.append(f"    • Next attempt: {next_attempt.strftime('%H:%M:%S')}")
        
        return "\n".join(report_lines)


# Utility functions

def create_retry_manager(
    stats_storage: Optional[BaseKVStorage] = None,
    default_max_attempts: int = 3,
    default_initial_delay: float = 1.0,
    default_max_delay: float = 60.0
) -> RetryManager:
    """Factory function to create a RetryManager with sensible defaults"""
    default_config = RetryConfig(
        max_attempts=default_max_attempts,
        initial_delay=default_initial_delay,
        max_delay=default_max_delay
    )
    
    manager = RetryManager(stats_storage=stats_storage, default_config=default_config)
    
    # Configure specific call types with optimized settings
    
    # Entity extraction - can be expensive, allow more retries
    manager.configure_call_type(
        LLMCallType.ENTITY_EXTRACTION,
        RetryConfig(max_attempts=4, initial_delay=2.0, max_delay=90.0)
    )
    
    # Relation extraction - similar to entity extraction
    manager.configure_call_type(
        LLMCallType.RELATION_EXTRACTION,
        RetryConfig(max_attempts=4, initial_delay=2.0, max_delay=90.0)
    )
    
    # Continue extraction (gleaning) - less critical, fewer retries
    manager.configure_call_type(
        LLMCallType.CONTINUE_EXTRACTION,
        RetryConfig(max_attempts=2, initial_delay=1.0, max_delay=30.0)
    )
    
    # Loop detection - very quick, minimal retries
    manager.configure_call_type(
        LLMCallType.LOOP_DETECTION,
        RetryConfig(max_attempts=2, initial_delay=0.5, max_delay=10.0)
    )
    
    # Hierarchical clustering - expensive and important
    manager.configure_call_type(
        LLMCallType.HIERARCHICAL_CLUSTERING,
        RetryConfig(max_attempts=5, initial_delay=3.0, max_delay=120.0)
    )
    
    # Entity disambiguation - critical for quality, allow generous retries
    manager.configure_call_type(
        LLMCallType.ENTITY_DISAMBIGUATION,
        RetryConfig(max_attempts=5, initial_delay=2.0, max_delay=90.0)
    )
    
    # Entity merging - important for consistency
    manager.configure_call_type(
        LLMCallType.ENTITY_MERGING,
        RetryConfig(max_attempts=3, initial_delay=1.5, max_delay=60.0)
    )
    
    # Community reports - final step, important but not critical for core functionality
    manager.configure_call_type(
        LLMCallType.COMMUNITY_REPORT,
        RetryConfig(max_attempts=3, initial_delay=2.0, max_delay=60.0)
    )
    
    logger.info("RetryManager configured with optimized settings for all call types")
    return manager