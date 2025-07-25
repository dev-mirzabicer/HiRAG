"""
Advanced Rate Limiting System for HiRAG

This module provides sophisticated rate limiting capabilities for LLM API calls
with per-model configuration, token bucket algorithms, and intelligent backpressure.

Key Features:
- Token bucket algorithm for smooth rate limiting
- Per-model and per-provider configuration
- Multiple concurrent limits (RPM, TPM, RPD, etc.)
- Smart waiting and backpressure handling
- Integration with retry mechanisms
- Comprehensive monitoring and statistics
- Adaptive rate adjustment based on API responses
"""

import asyncio
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
from collections import deque
import math

from ._utils import logger
from .base import BaseKVStorage
from ._token_estimation import LLMCallType


class RateLimitType(Enum):
    """Types of rate limits"""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    TOKENS_PER_MINUTE = "tokens_per_minute"
    TOKENS_PER_HOUR = "tokens_per_hour"
    TOKENS_PER_DAY = "tokens_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"


@dataclass
class RateLimitConfig:
    """Configuration for a specific rate limit"""
    limit_type: RateLimitType
    limit_value: int  # Maximum allowed in the time window
    time_window_seconds: float  # Time window for the limit
    burst_allowance: float = 1.2  # Allow 20% burst above steady rate
    
    @property
    def steady_rate_per_second(self) -> float:
        """Calculate the steady rate per second"""
        return self.limit_value / self.time_window_seconds
    
    @property
    def burst_capacity(self) -> int:
        """Calculate burst capacity"""
        return int(self.limit_value * self.burst_allowance)


@dataclass
class ModelRateConfig:
    """Complete rate limiting configuration for a model"""
    model_name: str
    provider: str = "openai"  # openai, azure, gemini, etc.
    
    # Rate limits
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    requests_per_day: int = 86400
    tokens_per_minute: int = 150000
    tokens_per_hour: int = 9000000
    tokens_per_day: int = 216000000
    concurrent_requests: int = 10
    
    # Advanced settings
    adaptive_rate_adjustment: bool = True
    backpressure_threshold: float = 0.8  # Start backing off when 80% of limit used
    priority_multiplier: float = 1.0  # Priority multiplier for this model
    
    def to_rate_limit_configs(self) -> List[RateLimitConfig]:
        """Convert to list of RateLimitConfig objects"""
        return [
            RateLimitConfig(RateLimitType.REQUESTS_PER_MINUTE, self.requests_per_minute, 60.0),
            RateLimitConfig(RateLimitType.REQUESTS_PER_HOUR, self.requests_per_hour, 3600.0),
            RateLimitConfig(RateLimitType.REQUESTS_PER_DAY, self.requests_per_day, 86400.0),
            RateLimitConfig(RateLimitType.TOKENS_PER_MINUTE, self.tokens_per_minute, 60.0),
            RateLimitConfig(RateLimitType.TOKENS_PER_HOUR, self.tokens_per_hour, 3600.0),
            RateLimitConfig(RateLimitType.TOKENS_PER_DAY, self.tokens_per_day, 86400.0),
            RateLimitConfig(RateLimitType.CONCURRENT_REQUESTS, self.concurrent_requests, 1.0),
        ]


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int  # Maximum tokens in bucket
    tokens: float  # Current tokens in bucket
    refill_rate: float  # Tokens added per second
    last_refill: float = field(default_factory=time.time)
    
    def refill(self):
        """Refill the bucket based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        self.last_refill = now
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
    
    def can_consume(self, tokens_needed: int) -> bool:
        """Check if we can consume the specified number of tokens"""
        self.refill()
        return self.tokens >= tokens_needed
    
    def consume(self, tokens_needed: int) -> bool:
        """Try to consume tokens from the bucket"""
        self.refill()
        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            return True
        return False
    
    def time_until_available(self, tokens_needed: int) -> float:
        """Calculate time until enough tokens are available"""
        self.refill()
        if self.tokens >= tokens_needed:
            return 0.0
        
        tokens_deficit = tokens_needed - self.tokens
        if self.refill_rate <= 0:
            return float('inf')
        
        return tokens_deficit / self.refill_rate
    
    @property
    def utilization(self) -> float:
        """Get current utilization (0.0 = empty, 1.0 = full)"""
        self.refill()
        return (self.capacity - self.tokens) / self.capacity if self.capacity > 0 else 0.0


@dataclass
class RequestRecord:
    """Record of a single request for tracking"""
    timestamp: float
    tokens_used: int
    model_name: str
    call_type: LLMCallType
    wait_time: float = 0.0
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)


@dataclass
class RateLimitStatistics:
    """Statistics for rate limiting"""
    model_name: str
    total_requests: int = 0
    total_tokens_used: int = 0
    total_wait_time: float = 0.0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    rate_limit_hits: int = 0
    backpressure_activations: int = 0
    current_utilization: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class RateLimiter:
    """
    Advanced rate limiter for HiRAG LLM calls
    
    This limiter provides:
    - Token bucket algorithm for smooth rate limiting
    - Multiple concurrent limits per model
    - Adaptive rate adjustment based on API responses
    - Smart waiting with backpressure
    - Comprehensive statistics and monitoring
    """
    
    # Default configurations for common models
    DEFAULT_CONFIGS = {
        "gpt-4o": ModelRateConfig(
            model_name="gpt-4o",
            provider="openai",
            requests_per_minute=500,
            tokens_per_minute=30000,
            concurrent_requests=20
        ),
        "gpt-4o-mini": ModelRateConfig(
            model_name="gpt-4o-mini",
            provider="openai",
            requests_per_minute=1000,
            tokens_per_minute=200000,
            concurrent_requests=50
        ),
        "gpt-3.5-turbo": ModelRateConfig(
            model_name="gpt-3.5-turbo",
            provider="openai",
            requests_per_minute=3500,
            tokens_per_minute=160000,
            concurrent_requests=100
        ),
        "gemini-1.5-pro": ModelRateConfig(
            model_name="gemini-1.5-pro",
            provider="gemini",
            requests_per_minute=300,
            tokens_per_minute=32000,
            concurrent_requests=15
        ),
        "gemini-1.5-flash": ModelRateConfig(
            model_name="gemini-1.5-flash",
            provider="gemini",
            requests_per_minute=1000,
            tokens_per_minute=1000000,
            concurrent_requests=100
        ),
        "claude-3-5-sonnet": ModelRateConfig(
            model_name="claude-3-5-sonnet",
            provider="anthropic",
            requests_per_minute=200,
            tokens_per_minute=40000,
            concurrent_requests=10
        ),
    }
    
    def __init__(
        self,
        stats_storage: Optional[BaseKVStorage] = None,
        enable_adaptive_adjustment: bool = True
    ):
        self.stats_storage = stats_storage
        self.enable_adaptive_adjustment = enable_adaptive_adjustment
        
        # Model configurations
        self.model_configs: Dict[str, ModelRateConfig] = {}
        
        # Token buckets for each limit type per model
        self.token_buckets: Dict[str, Dict[RateLimitType, TokenBucket]] = {}
        
        # Concurrent request tracking
        self.active_requests: Dict[str, int] = {}  # model -> count
        
        # Request history for statistics
        self.request_history: Dict[str, deque] = {}  # model -> deque of RequestRecord
        
        # Statistics
        self.statistics: Dict[str, RateLimitStatistics] = {}
        
        # Adaptive adjustment tracking
        self.recent_rate_limit_responses: Dict[str, deque] = {}  # model -> timestamps
        
        # Priority queue for waiting requests
        self.waiting_requests: List[Tuple[float, str, asyncio.Event]] = []  # (priority, model, event)
        
        logger.info("RateLimiter initialized")

    def configure_model(self, model_name: str, config: ModelRateConfig):
        """Configure rate limiting for a specific model"""
        self.model_configs[model_name] = config
        
        # Initialize token buckets
        buckets = {}
        for limit_config in config.to_rate_limit_configs():
            if limit_config.limit_type == RateLimitType.CONCURRENT_REQUESTS:
                # Concurrent requests don't use token buckets
                continue
            
            bucket = TokenBucket(
                capacity=limit_config.burst_capacity,
                tokens=float(limit_config.burst_capacity),
                refill_rate=limit_config.steady_rate_per_second
            )
            buckets[limit_config.limit_type] = bucket
        
        self.token_buckets[model_name] = buckets
        
        # Initialize tracking structures
        self.active_requests[model_name] = 0
        self.request_history[model_name] = deque(maxlen=1000)  # Keep last 1000 requests
        self.statistics[model_name] = RateLimitStatistics(model_name=model_name)
        self.recent_rate_limit_responses[model_name] = deque(maxlen=10)  # Track recent rate limits
        
        logger.info(f"Configured rate limiting for model: {model_name}")

    def get_or_create_default_config(self, model_name: str) -> ModelRateConfig:
        """Get existing config or create default for a model"""
        if model_name in self.model_configs:
            return self.model_configs[model_name]
        
        # Try to find a default config
        if model_name in self.DEFAULT_CONFIGS:
            config = self.DEFAULT_CONFIGS[model_name]
        else:
            # Create conservative default
            config = ModelRateConfig(
                model_name=model_name,
                requests_per_minute=60,
                tokens_per_minute=10000,
                concurrent_requests=5
            )
            logger.warning(f"Using conservative default rate limits for unknown model: {model_name}")
        
        self.configure_model(model_name, config)
        return config

    async def acquire(
        self,
        model_name: str,
        call_type: LLMCallType,
        estimated_tokens: int,
        priority: float = 1.0
    ) -> float:
        """
        Acquire permission to make a request
        
        Args:
            model_name: Name of the model
            call_type: Type of LLM call
            estimated_tokens: Estimated tokens for the request
            priority: Priority multiplier (higher = more important)
            
        Returns:
            Wait time in seconds before the request was allowed
        """
        start_time = time.time()
        config = self.get_or_create_default_config(model_name)
        
        # Apply priority multiplier
        effective_priority = priority * config.priority_multiplier
        
        while True:
            # Check all rate limits
            max_wait_time = 0.0
            limiting_factor = None
            
            # Check concurrent requests
            if self.active_requests[model_name] >= config.concurrent_requests:
                # Wait for a concurrent slot to become available
                logger.debug(f"Waiting for concurrent request slot for {model_name}")
                await asyncio.sleep(0.1)  # Small delay before rechecking
                continue
            
            # Check token bucket limits
            buckets = self.token_buckets.get(model_name, {})
            for limit_type, bucket in buckets.items():
                if limit_type in [RateLimitType.REQUESTS_PER_MINUTE, RateLimitType.REQUESTS_PER_HOUR, RateLimitType.REQUESTS_PER_DAY]:
                    # For request-based limits, we need 1 token
                    tokens_needed = 1
                else:
                    # For token-based limits, we need the estimated tokens
                    tokens_needed = estimated_tokens
                
                wait_time = bucket.time_until_available(tokens_needed)
                if wait_time > max_wait_time:
                    max_wait_time = wait_time
                    limiting_factor = limit_type
            
            # Check backpressure threshold
            if limiting_factor and self._should_apply_backpressure(model_name, limiting_factor):
                backpressure_delay = self._calculate_backpressure_delay(model_name, effective_priority)
                max_wait_time = max(max_wait_time, backpressure_delay)
                
                stats = self.statistics[model_name]
                stats.backpressure_activations += 1
                
                logger.debug(f"Applying backpressure for {model_name}: {backpressure_delay:.2f}s")
            
            if max_wait_time <= 0:
                # We can proceed immediately
                break
            
            # Apply priority-based wait time adjustment
            adjusted_wait_time = max_wait_time / max(0.1, effective_priority)
            
            logger.debug(f"Rate limit hit for {model_name} ({limiting_factor.value if limiting_factor else 'concurrent'}): waiting {adjusted_wait_time:.2f}s")
            
            # Track rate limit hit
            stats = self.statistics[model_name]
            stats.rate_limit_hits += 1
            
            await asyncio.sleep(min(adjusted_wait_time, 5.0))  # Cap individual waits at 5 seconds
        
        # Consume tokens from buckets
        buckets = self.token_buckets.get(model_name, {})
        for limit_type, bucket in buckets.items():
            if limit_type in [RateLimitType.REQUESTS_PER_MINUTE, RateLimitType.REQUESTS_PER_HOUR, RateLimitType.REQUESTS_PER_DAY]:
                bucket.consume(1)
            else:
                bucket.consume(estimated_tokens)
        
        # Increment concurrent requests
        self.active_requests[model_name] += 1
        
        # Calculate total wait time
        total_wait_time = time.time() - start_time
        
        # Update statistics
        stats = self.statistics[model_name]
        stats.total_requests += 1
        stats.total_wait_time += total_wait_time
        stats.avg_wait_time = stats.total_wait_time / stats.total_requests
        stats.max_wait_time = max(stats.max_wait_time, total_wait_time)
        
        # Update utilization stats
        stats.current_utilization = self._calculate_current_utilization(model_name)
        
        logger.debug(f"Acquired rate limit permission for {model_name} after {total_wait_time:.2f}s wait")
        return total_wait_time

    async def release(
        self,
        model_name: str,
        call_type: LLMCallType,
        actual_tokens_used: int,
        success: bool = True,
        rate_limited: bool = False
    ):
        """
        Release a request and update statistics
        
        Args:
            model_name: Name of the model
            call_type: Type of LLM call
            actual_tokens_used: Actual tokens used in the request
            success: Whether the request was successful
            rate_limited: Whether the request was rate limited by the API
        """
        # Decrement concurrent requests
        if model_name in self.active_requests:
            self.active_requests[model_name] = max(0, self.active_requests[model_name] - 1)
        
        # Record the request
        record = RequestRecord(
            timestamp=time.time(),
            tokens_used=actual_tokens_used,
            model_name=model_name,
            call_type=call_type
        )
        
        if model_name in self.request_history:
            self.request_history[model_name].append(record)
        
        # Update statistics
        if model_name in self.statistics:
            stats = self.statistics[model_name]
            stats.total_tokens_used += actual_tokens_used
            stats.last_updated = time.time()
        
        # Handle adaptive rate adjustment
        if rate_limited and self.enable_adaptive_adjustment:
            await self._handle_rate_limit_response(model_name)
        
        logger.debug(f"Released rate limit for {model_name} (tokens: {actual_tokens_used}, success: {success})")

    def _should_apply_backpressure(self, model_name: str, limit_type: RateLimitType) -> bool:
        """Check if backpressure should be applied"""
        config = self.model_configs.get(model_name)
        if not config or not config.adaptive_rate_adjustment:
            return False
        
        buckets = self.token_buckets.get(model_name, {})
        bucket = buckets.get(limit_type)
        if not bucket:
            return False
        
        return bucket.utilization > config.backpressure_threshold

    def _calculate_backpressure_delay(self, model_name: str, priority: float) -> float:
        """Calculate backpressure delay based on current utilization"""
        config = self.model_configs.get(model_name)
        if not config:
            return 0.0
        
        # Find the highest utilization across all buckets
        max_utilization = 0.0
        buckets = self.token_buckets.get(model_name, {})
        for bucket in buckets.values():
            max_utilization = max(max_utilization, bucket.utilization)
        
        if max_utilization <= config.backpressure_threshold:
            return 0.0
        
        # Calculate delay based on excess utilization
        excess_utilization = max_utilization - config.backpressure_threshold
        base_delay = excess_utilization * 5.0  # Up to 5 seconds for full utilization
        
        # Apply priority adjustment
        priority_adjusted_delay = base_delay / max(0.1, priority)
        
        return min(priority_adjusted_delay, 10.0)  # Cap at 10 seconds

    def _calculate_current_utilization(self, model_name: str) -> Dict[str, float]:
        """Calculate current utilization for all rate limits"""
        utilization = {}
        buckets = self.token_buckets.get(model_name, {})
        
        for limit_type, bucket in buckets.items():
            utilization[limit_type.value] = bucket.utilization
        
        # Add concurrent requests utilization
        config = self.model_configs.get(model_name)
        if config:
            concurrent_util = self.active_requests.get(model_name, 0) / max(1, config.concurrent_requests)
            utilization[RateLimitType.CONCURRENT_REQUESTS.value] = concurrent_util
        
        return utilization

    async def _handle_rate_limit_response(self, model_name: str):
        """Handle a rate limit response from the API for adaptive adjustment"""
        current_time = time.time()
        
        # Record the rate limit response
        if model_name in self.recent_rate_limit_responses:
            self.recent_rate_limit_responses[model_name].append(current_time)
        
        # Check if we're getting rate limited frequently
        recent_responses = self.recent_rate_limit_responses.get(model_name, deque())
        recent_count = sum(1 for timestamp in recent_responses if current_time - timestamp < 300)  # Last 5 minutes
        
        if recent_count >= 3:
            # We're getting rate limited frequently, reduce our limits
            await self._reduce_rate_limits(model_name)
            logger.warning(f"Reducing rate limits for {model_name} due to frequent API rate limiting")

    async def _reduce_rate_limits(self, model_name: str, reduction_factor: float = 0.8):
        """Reduce rate limits for a model due to frequent API rate limiting"""
        config = self.model_configs.get(model_name)
        if not config:
            return
        
        # Create new config with reduced limits
        new_config = ModelRateConfig(
            model_name=model_name,
            provider=config.provider,
            requests_per_minute=int(config.requests_per_minute * reduction_factor),
            requests_per_hour=int(config.requests_per_hour * reduction_factor),
            requests_per_day=int(config.requests_per_day * reduction_factor),
            tokens_per_minute=int(config.tokens_per_minute * reduction_factor),
            tokens_per_hour=int(config.tokens_per_hour * reduction_factor),
            tokens_per_day=int(config.tokens_per_day * reduction_factor),
            concurrent_requests=max(1, int(config.concurrent_requests * reduction_factor)),
            adaptive_rate_adjustment=config.adaptive_rate_adjustment,
            backpressure_threshold=config.backpressure_threshold,
            priority_multiplier=config.priority_multiplier
        )
        
        # Reconfigure with new limits
        self.configure_model(model_name, new_config)

    async def get_statistics(self, model_name: Optional[str] = None) -> Dict[str, RateLimitStatistics]:
        """Get rate limiting statistics"""
        if model_name:
            return {model_name: self.statistics.get(model_name, RateLimitStatistics(model_name=model_name))}
        else:
            return self.statistics.copy()

    async def get_current_utilization(self, model_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get current utilization for all models or a specific model"""
        if model_name:
            return {model_name: self._calculate_current_utilization(model_name)}
        else:
            return {name: self._calculate_current_utilization(name) for name in self.model_configs.keys()}

    async def reset_model_limits(self, model_name: str):
        """Reset rate limits for a model to default configuration"""
        if model_name in self.DEFAULT_CONFIGS:
            default_config = self.DEFAULT_CONFIGS[model_name]
            self.configure_model(model_name, default_config)
            logger.info(f"Reset rate limits for {model_name} to defaults")
        else:
            logger.warning(f"No default configuration available for {model_name}")

    async def generate_rate_limit_report(self) -> str:
        """Generate a comprehensive rate limiting report"""
        report_lines = [
            "=== Rate Limiting Statistics Report ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if not self.statistics:
            report_lines.append("No rate limiting statistics available.")
            return "\n".join(report_lines)
        
        # Overall summary
        total_requests = sum(stats.total_requests for stats in self.statistics.values())
        total_tokens = sum(stats.total_tokens_used for stats in self.statistics.values())
        total_wait_time = sum(stats.total_wait_time for stats in self.statistics.values())
        total_rate_limits = sum(stats.rate_limit_hits for stats in self.statistics.values())
        
        report_lines.extend([
            "Overall Summary:",
            f"  • Total requests: {total_requests:,}",
            f"  • Total tokens used: {total_tokens:,}",
            f"  • Total wait time: {total_wait_time:.2f}s",
            f"  • Rate limit hits: {total_rate_limits:,}",
            f"  • Average wait per request: {total_wait_time/max(1,total_requests):.3f}s",
            ""
        ])
        
        # Per-model breakdown
        report_lines.append("Per-Model Breakdown:")
        for model_name, stats in self.statistics.items():
            config = self.model_configs.get(model_name)
            utilization = self._calculate_current_utilization(model_name)
            
            report_lines.extend([
                f"  {model_name.upper()}:",
                f"    • Requests: {stats.total_requests:,} | Tokens: {stats.total_tokens_used:,}",
                f"    • Avg wait: {stats.avg_wait_time:.3f}s | Max wait: {stats.max_wait_time:.2f}s",
                f"    • Rate limit hits: {stats.rate_limit_hits} | Backpressure: {stats.backpressure_activations}",
                f"    • Active requests: {self.active_requests.get(model_name, 0)}"
            ])
            
            if config:
                report_lines.append(f"    • Concurrent limit: {config.concurrent_requests}")
            
            # Current utilization
            if utilization:
                report_lines.append("    • Current utilization:")
                for limit_type, util in utilization.items():
                    if util > 0:
                        report_lines.append(f"      - {limit_type}: {util:.1%}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


# Utility functions

def create_rate_limiter(
    stats_storage: Optional[BaseKVStorage] = None,
    enable_adaptive_adjustment: bool = True
) -> RateLimiter:
    """Factory function to create a RateLimiter with default configurations"""
    limiter = RateLimiter(
        stats_storage=stats_storage,
        enable_adaptive_adjustment=enable_adaptive_adjustment
    )
    
    # Configure all default models
    for model_name, config in RateLimiter.DEFAULT_CONFIGS.items():
        limiter.configure_model(model_name, config)
    
    logger.info("RateLimiter created with all default model configurations")
    return limiter


def get_conservative_limits() -> Dict[str, ModelRateConfig]:
    """Get conservative rate limit configurations for production use"""
    conservative_configs = {}
    
    for model_name, default_config in RateLimiter.DEFAULT_CONFIGS.items():
        # Reduce all limits by 20% for safety margin
        conservative_config = ModelRateConfig(
            model_name=model_name,
            provider=default_config.provider,
            requests_per_minute=int(default_config.requests_per_minute * 0.8),
            requests_per_hour=int(default_config.requests_per_hour * 0.8),
            requests_per_day=int(default_config.requests_per_day * 0.8),
            tokens_per_minute=int(default_config.tokens_per_minute * 0.8),
            tokens_per_hour=int(default_config.tokens_per_hour * 0.8),
            tokens_per_day=int(default_config.tokens_per_day * 0.8),
            concurrent_requests=max(1, int(default_config.concurrent_requests * 0.8)),
            adaptive_rate_adjustment=True,
            backpressure_threshold=0.7,  # More conservative threshold
            priority_multiplier=default_config.priority_multiplier
        )
        conservative_configs[model_name] = conservative_config
    
    return conservative_configs