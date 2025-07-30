"""
Advanced Progress Tracking and Visual Dashboard for HiRAG

This module provides comprehensive progress tracking and visualization capabilities
for the HiRAG ingestion pipeline, including real-time dashboards, ETA calculations,
and detailed metrics monitoring.

Key Features:
- Real-time progress tracking with ETA calculations
- Rich terminal UI and web-based dashboard options
- Token usage and cost tracking
- Rate limiting status monitoring
- Retry and error statistics
- Checkpoint integration
- Export capabilities for external monitoring
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, TYPE_CHECKING
from enum import Enum
import threading
from collections import deque, defaultdict
import math

try:
    from rich.console import Console  # type: ignore
    from rich.table import Table  # type: ignore
    from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn  # type: ignore
    from rich.live import Live  # type: ignore
    from rich.layout import Layout  # type: ignore
    from rich.panel import Panel  # type: ignore
    from rich.text import Text  # type: ignore
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    if TYPE_CHECKING:
        from rich.console import Console  # type: ignore
        from rich.table import Table  # type: ignore
        from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn  # type: ignore
        from rich.live import Live  # type: ignore
        from rich.layout import Layout  # type: ignore
        from rich.panel import Panel  # type: ignore
        from rich.text import Text  # type: ignore

from ._utils import logger
from .base import BaseKVStorage
from ._token_estimation import LLMCallType, PipelineEstimate
from ._checkpointing import CheckpointStage, PipelineCheckpoint
from ._rate_limiting import RateLimiter, RateLimitStatistics
from ._retry_manager import RetryManager, RetryStatistics


class DashboardType(Enum):
    """Types of dashboards available"""
    TERMINAL = "terminal"
    WEB = "web"
    JSON_EXPORT = "json_export"
    CONSOLE_LOG = "console_log"


@dataclass
class ProgressMetrics:
    """Core progress metrics for tracking"""
    total_chunks: int = 0
    processed_chunks: int = 0
    total_entities: int = 0
    extracted_entities: int = 0
    total_relations: int = 0
    extracted_relations: int = 0
    total_communities: int = 0
    generated_communities: int = 0
    
    # Token metrics
    estimated_total_tokens: int = 0
    consumed_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    
    # Time metrics
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    estimated_completion_time: Optional[float] = None
    
    # Current operation
    current_stage: str = "initialization"
    current_operation: str = ""
    
    @property
    def overall_progress(self) -> float:
        """Calculate overall progress (0.0 to 1.0)"""
        if self.total_chunks == 0:
            return 0.0
        
        # Weight different stages
        chunk_progress = self.processed_chunks / self.total_chunks * 0.4
        entity_progress = (self.extracted_entities / max(1, self.total_entities)) * 0.3 if self.total_entities > 0 else 0
        relation_progress = (self.extracted_relations / max(1, self.total_relations)) * 0.2 if self.total_relations > 0 else 0
        community_progress = (self.generated_communities / max(1, self.total_communities)) * 0.1 if self.total_communities > 0 else 0
        
        return min(1.0, chunk_progress + entity_progress + relation_progress + community_progress)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimate time to completion in seconds"""
        progress = self.overall_progress
        if progress <= 0:
            return None
        
        elapsed = self.elapsed_time
        estimated_total_time = elapsed / progress
        return max(0, estimated_total_time - elapsed)
    
    @property
    def processing_rate(self) -> float:
        """Get processing rate (chunks per second)"""
        elapsed = self.elapsed_time
        if elapsed <= 0:
            return 0.0
        return self.processed_chunks / elapsed


@dataclass
class LiveStatistics:
    """Live statistics for the dashboard"""
    # Rate limiting stats
    rate_limit_hits: int = 0
    current_rate_utilization: Dict[str, float] = field(default_factory=dict)
    
    # Retry stats  
    total_retries: int = 0
    failure_breakdown: Dict[str, int] = field(default_factory=dict)
    circuit_breaker_status: Dict[str, str] = field(default_factory=dict)
    
    # Performance stats
    avg_response_time: float = 0.0
    requests_per_minute: float = 0.0
    tokens_per_minute: float = 0.0
    
    # Error tracking
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=10))
    error_rate: float = 0.0


class ProgressTracker:
    """
    Main progress tracking system for HiRAG pipeline
    
    This tracker provides:
    - Real-time progress monitoring
    - ETA calculations
    - Integration with all pipeline components
    - Multiple dashboard options
    - Export capabilities
    """
    
    def __init__(
        self,
        dashboard_type: DashboardType = DashboardType.TERMINAL,
        storage: Optional[BaseKVStorage] = None,
        update_interval: float = 1.0,
        enable_web_dashboard: bool = False,
        web_port: int = 8080
    ):
        self.dashboard_type = dashboard_type
        self.storage = storage
        self.update_interval = update_interval
        self.enable_web_dashboard = enable_web_dashboard
        self.web_port = web_port
        
        # Core tracking data
        self.metrics = ProgressMetrics()
        self.live_stats = LiveStatistics()
        self.pipeline_estimate: Optional[PipelineEstimate] = None
        self.current_checkpoint: Optional[PipelineCheckpoint] = None
        
        # Component references
        self.rate_limiter: Optional[RateLimiter] = None
        self.retry_manager: Optional[RetryManager] = None
        
        # Dashboard components
        self.console: Optional['Console'] = Console() if RICH_AVAILABLE else None
        self.rich_progress: Optional['Progress'] = None
        self.main_task_id: Optional['TaskID'] = None
        self.live_display: Optional['Live'] = None
        
        # Background update task
        self._update_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Event tracking
        self.recent_events = deque(maxlen=100)
        
        logger.info(f"ProgressTracker initialized with {dashboard_type.value} dashboard")

    def set_pipeline_estimate(self, estimate: PipelineEstimate):
        """Set the initial pipeline estimate"""
        self.pipeline_estimate = estimate
        
        # Update metrics with estimates
        self.metrics.total_chunks = estimate.num_chunks
        self.metrics.estimated_total_tokens = estimate.total_input_tokens + estimate.total_output_tokens
        self.metrics.estimated_cost_usd = estimate.estimated_cost_usd
        
        # Extract entity and community estimates from metadata
        metadata = estimate.metadata
        self.metrics.total_entities = metadata.get("estimated_entities", 0)
        self.metrics.total_communities = metadata.get("estimated_communities", 0)
        
        logger.info(f"Pipeline estimate set: {estimate.num_chunks} chunks, {self.metrics.total_entities} entities")

    def set_components(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        retry_manager: Optional[RetryManager] = None
    ):
        """Set component references for monitoring"""
        self.rate_limiter = rate_limiter
        self.retry_manager = retry_manager

    def set_checkpoint(self, checkpoint: PipelineCheckpoint):
        """Update with current checkpoint"""
        self.current_checkpoint = checkpoint
        
        # Update metrics from checkpoint
        self.metrics.processed_chunks = len(checkpoint.processed_chunks)
        self.metrics.extracted_entities = len(checkpoint.extracted_entities)
        self.metrics.extracted_relations = len(checkpoint.extracted_relations)
        self.metrics.generated_communities = len(checkpoint.completed_communities)
        self.metrics.current_stage = checkpoint.current_stage.value
        
        # Update overall progress
        if hasattr(checkpoint, 'overall_progress'):
            # Use checkpoint's calculated progress if available
            pass
        
        self._add_event(f"Checkpoint updated: {checkpoint.current_stage.value}")

    async def start_tracking(self):
        """Start the progress tracking system"""
        self._running = True
        
        if self.dashboard_type == DashboardType.TERMINAL and RICH_AVAILABLE:
            await self._start_rich_dashboard()
        elif self.dashboard_type == DashboardType.WEB and self.enable_web_dashboard:
            await self._start_web_dashboard()
        
        # Start background update task
        self._update_task = asyncio.create_task(self._update_loop())
        
        logger.info("Progress tracking started")

    async def stop_tracking(self):
        """Stop the progress tracking system"""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self.live_display:
            self.live_display.stop()
        
        logger.info("Progress tracking stopped")

    def update_progress(
        self,
        stage: Optional[CheckpointStage] = None,
        operation: Optional[str] = None,
        chunks_processed: Optional[int] = None,
        entities_extracted: Optional[int] = None,
        relations_extracted: Optional[int] = None,
        communities_generated: Optional[int] = None,
        tokens_consumed: Optional[int] = None,
        cost_incurred: Optional[float] = None
    ):
        """Update progress metrics"""
        current_time = time.time()
        
        if stage:
            self.metrics.current_stage = stage.value
        if operation:
            self.metrics.current_operation = operation
        if chunks_processed is not None:
            self.metrics.processed_chunks = chunks_processed
        if entities_extracted is not None:
            self.metrics.extracted_entities = entities_extracted
        if relations_extracted is not None:
            self.metrics.extracted_relations = relations_extracted
        if communities_generated is not None:
            self.metrics.generated_communities = communities_generated
        if tokens_consumed is not None:
            self.metrics.consumed_tokens = tokens_consumed
        if cost_incurred is not None:
            self.metrics.actual_cost_usd += cost_incurred
        
        self.metrics.last_update = current_time
        
        # Update ETA
        self._calculate_eta()

    def add_error(self, error: Exception, context: str = ""):
        """Add an error to tracking"""
        error_info = {
            "timestamp": time.time(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context
        }
        
        self.live_stats.recent_errors.append(error_info)
        self._add_event(f"Error: {error_info['type']} in {context}")

    def _add_event(self, message: str):
        """Add an event to the recent events log"""
        event = {
            "timestamp": time.time(),
            "message": message
        }
        self.recent_events.append(event)

    def _calculate_eta(self):
        """Calculate estimated time to completion"""
        progress = self.metrics.overall_progress
        if progress <= 0.01:  # Less than 1% progress
            self.metrics.estimated_completion_time = None
            return
        
        elapsed = self.metrics.elapsed_time
        if elapsed <= 0:
            return
        
        # Use multiple estimation methods and average them
        estimates = []
        
        # Method 1: Linear extrapolation from overall progress
        if progress > 0:
            total_estimated_time = elapsed / progress
            estimates.append(total_estimated_time - elapsed)
        
        # Method 2: Based on chunk processing rate
        if self.metrics.processed_chunks > 0 and self.metrics.total_chunks > 0:
            chunks_remaining = self.metrics.total_chunks - self.metrics.processed_chunks
            chunk_rate = self.metrics.processing_rate
            if chunk_rate > 0:
                estimates.append(chunks_remaining / chunk_rate)
        
        # Method 3: Based on token consumption rate
        if self.metrics.consumed_tokens > 0 and self.metrics.estimated_total_tokens > 0:
            token_rate = self.metrics.consumed_tokens / elapsed
            tokens_remaining = self.metrics.estimated_total_tokens - self.metrics.consumed_tokens
            if token_rate > 0:
                estimates.append(tokens_remaining / token_rate)
        
        if estimates:
            # Use weighted average, giving more weight to overall progress
            weights = [3, 2, 1][:len(estimates)]
            weighted_avg = sum(est * weight for est, weight in zip(estimates, weights)) / sum(weights)
            self.metrics.estimated_completion_time = time.time() + weighted_avg

    async def _update_loop(self):
        """Background loop to update statistics"""
        while self._running:
            try:
                await self._update_live_statistics()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in progress update loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _update_live_statistics(self):
        """Update live statistics from components"""
        try:
            # Update rate limiting statistics
            if self.rate_limiter:
                rate_stats = await self.rate_limiter.get_statistics()
                utilization = await self.rate_limiter.get_current_utilization()
                
                total_rate_hits = sum(stats.rate_limit_hits for stats in rate_stats.values())
                self.live_stats.rate_limit_hits = total_rate_hits
                
                # Flatten utilization data
                self.live_stats.current_rate_utilization = {}
                for model, model_util in utilization.items():
                    for limit_type, util_value in model_util.items():
                        key = f"{model}_{limit_type}"
                        self.live_stats.current_rate_utilization[key] = util_value
            
            # Update retry statistics
            if self.retry_manager:
                retry_stats = await self.retry_manager.get_statistics()
                circuit_status = await self.retry_manager.get_circuit_breaker_status()
                
                # Handle both Dict[LLMCallType, RetryStatistics] and single RetryStatistics
                if isinstance(retry_stats, dict):
                    stats_list = list(retry_stats.values())
                else:
                    stats_list = [retry_stats]
                
                total_retries = sum(stats.total_retries for stats in stats_list)
                self.live_stats.total_retries = total_retries
                
                # Aggregate failure breakdown
                failure_breakdown = defaultdict(int)
                for stats in stats_list:
                    for failure_type, count in stats.failure_breakdown.items():
                        failure_breakdown[failure_type] += count
                self.live_stats.failure_breakdown = dict(failure_breakdown)
                
                # Circuit breaker status
                self.live_stats.circuit_breaker_status = {
                    call_type: status["state"] for call_type, status in circuit_status.items()
                }
            
            # Calculate performance metrics
            elapsed = self.metrics.elapsed_time
            if elapsed > 0:
                self.live_stats.requests_per_minute = (self.metrics.processed_chunks * 60) / elapsed
                self.live_stats.tokens_per_minute = (self.metrics.consumed_tokens * 60) / elapsed
            
            # Calculate error rate
            recent_error_count = len([
                event for event in self.recent_events 
                if "Error:" in event["message"] and time.time() - event["timestamp"] < 300
            ])
            total_recent_operations = max(1, self.metrics.processed_chunks)
            self.live_stats.error_rate = recent_error_count / total_recent_operations
            
        except Exception as e:
            logger.error(f"Error updating live statistics: {e}")

    async def _start_rich_dashboard(self):
        """Start the Rich terminal dashboard"""
        if not RICH_AVAILABLE:
            logger.warning("Rich library not available, falling back to console logging")
            self.dashboard_type = DashboardType.CONSOLE_LOG
            return
        
        try:
            # Create progress bars
            self.rich_progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            )
            
            # Create main progress task
            if self.rich_progress is not None:
                self.main_task_id = self.rich_progress.add_task(
                    "Overall Progress", 
                    total=100
                )
            
            # Create layout for rich dashboard
            layout = self._create_rich_layout()
            
            # Start live display
            self.live_display = Live(
                layout, 
                console=self.console, 
                refresh_per_second=2,
                screen=True
            )
            if self.live_display is not None:
                self.live_display.start()
            
        except Exception as e:
            logger.error(f"Error starting Rich dashboard: {e}")
            self.dashboard_type = DashboardType.CONSOLE_LOG

    def _create_rich_layout(self) -> 'Layout':
        """Create the Rich dashboard layout"""
        if not RICH_AVAILABLE:
            raise RuntimeError("Rich library not available")
        
        from rich.layout import Layout  # type: ignore
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=8)
        )
        
        layout["main"].split_row(
            Layout(name="progress", ratio=2),
            Layout(name="stats", ratio=1)
        )
        
        # Update layout with current data
        self._update_rich_layout(layout)
        
        return layout

    def _update_rich_layout(self, layout: 'Layout'):
        """Update the Rich dashboard layout with current data"""
        if not RICH_AVAILABLE:
            return
            
        try:
            from rich.text import Text  # type: ignore
            from rich.panel import Panel  # type: ignore
            from rich.table import Table  # type: ignore
            
            # Header
            header_text = Text()
            header_text.append("HiRAG Pipeline Progress Dashboard", style="bold magenta")
            header_text.append(f" | Started: {datetime.fromtimestamp(self.metrics.start_time).strftime('%H:%M:%S')}")
            layout["header"].update(Panel(header_text, title="HiRAG Dashboard"))
            
            # Progress section
            progress_table = Table(show_header=True, header_style="bold blue")
            progress_table.add_column("Metric", style="cyan")
            progress_table.add_column("Progress", justify="right")
            progress_table.add_column("Count", justify="right")
            
            # Add progress rows
            progress_table.add_row(
                "Chunks", 
                f"{self.metrics.overall_progress:.1%}",
                f"{self.metrics.processed_chunks}/{self.metrics.total_chunks}"
            )
            progress_table.add_row(
                "Entities",
                f"{(self.metrics.extracted_entities/max(1,self.metrics.total_entities)):.1%}" if self.metrics.total_entities > 0 else "N/A",
                f"{self.metrics.extracted_entities}/{self.metrics.total_entities}"
            )
            progress_table.add_row(
                "Relations",
                f"{(self.metrics.extracted_relations/max(1,self.metrics.total_relations)):.1%}" if self.metrics.total_relations > 0 else "N/A",
                f"{self.metrics.extracted_relations}/{self.metrics.total_relations}"
            )
            progress_table.add_row(
                "Communities",
                f"{(self.metrics.generated_communities/max(1,self.metrics.total_communities)):.1%}" if self.metrics.total_communities > 0 else "N/A",
                f"{self.metrics.generated_communities}/{self.metrics.total_communities}"
            )
            
            layout["progress"].update(Panel(progress_table, title="Pipeline Progress"))
            
            # Statistics section
            stats_table = Table(show_header=True, header_style="bold green")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", justify="right")
            
            # Time metrics
            elapsed_str = str(timedelta(seconds=int(self.metrics.elapsed_time)))
            stats_table.add_row("Elapsed Time", elapsed_str)
            
            if self.metrics.eta_seconds:
                eta_str = str(timedelta(seconds=int(self.metrics.eta_seconds)))
                stats_table.add_row("ETA", eta_str)
            
            # Token and cost metrics
            stats_table.add_row("Tokens Used", f"{self.metrics.consumed_tokens:,}")
            stats_table.add_row("Est. Cost", f"${self.metrics.estimated_cost_usd:.4f}")
            stats_table.add_row("Actual Cost", f"${self.metrics.actual_cost_usd:.4f}")
            
            # Performance metrics
            stats_table.add_row("Processing Rate", f"{self.metrics.processing_rate:.2f} chunks/s")
            stats_table.add_row("Rate Limit Hits", str(self.live_stats.rate_limit_hits))
            stats_table.add_row("Total Retries", str(self.live_stats.total_retries))
            
            layout["stats"].update(Panel(stats_table, title="Statistics"))
            
            # Footer with recent events
            recent_events_text = Text()
            for event in list(self.recent_events)[-5:]:  # Last 5 events
                timestamp = datetime.fromtimestamp(event["timestamp"]).strftime('%H:%M:%S')
                recent_events_text.append(f"[{timestamp}] {event['message']}\n", style="dim")
            
            layout["footer"].update(Panel(recent_events_text, title="Recent Events"))
            
        except Exception as e:
            logger.error(f"Error updating Rich layout: {e}")

    async def _start_web_dashboard(self):
        """Start the web-based dashboard"""
        # This would implement a web server (e.g., using FastAPI)
        # For now, we'll just log that it's not implemented
        logger.warning("Web dashboard not implemented yet, falling back to terminal")
        self.dashboard_type = DashboardType.TERMINAL
        await self._start_rich_dashboard()

    async def export_progress_data(self) -> Dict[str, Any]:
        """Export current progress data for external consumption"""
        return {
            "timestamp": time.time(),
            "metrics": asdict(self.metrics),
            "live_stats": asdict(self.live_stats),
            "pipeline_estimate": asdict(self.pipeline_estimate) if self.pipeline_estimate else None,
            "checkpoint_info": {
                "current_stage": self.current_checkpoint.current_stage.value if self.current_checkpoint else None,
                "overall_progress": self.current_checkpoint.overall_progress if self.current_checkpoint else None
            },
            "recent_events": list(self.recent_events)[-10:]  # Last 10 events
        }

    async def generate_progress_report(self) -> str:
        """Generate a comprehensive progress report"""
        report_lines = [
            "=== HiRAG Pipeline Progress Report ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session started: {datetime.fromtimestamp(self.metrics.start_time).strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Overall progress
        elapsed_str = str(timedelta(seconds=int(self.metrics.elapsed_time)))
        eta_str = str(timedelta(seconds=int(self.metrics.eta_seconds))) if self.metrics.eta_seconds else "Unknown"
        
        report_lines.extend([
            "Overall Progress:",
            f"  • Completion: {self.metrics.overall_progress:.1%}",
            f"  • Elapsed time: {elapsed_str}",
            f"  • ETA: {eta_str}",
            f"  • Processing rate: {self.metrics.processing_rate:.2f} chunks/second",
            ""
        ])
        
        # Detailed progress
        report_lines.extend([
            "Detailed Progress:",
            f"  • Chunks: {self.metrics.processed_chunks}/{self.metrics.total_chunks} ({self.metrics.processed_chunks/max(1,self.metrics.total_chunks):.1%})",
            f"  • Entities: {self.metrics.extracted_entities}/{self.metrics.total_entities}",
            f"  • Relations: {self.metrics.extracted_relations}/{self.metrics.total_relations}",
            f"  • Communities: {self.metrics.generated_communities}/{self.metrics.total_communities}",
            ""
        ])
        
        # Token and cost information
        token_efficiency = (self.metrics.consumed_tokens / max(1, self.metrics.estimated_total_tokens)) * 100
        cost_efficiency = (self.metrics.actual_cost_usd / max(0.001, self.metrics.estimated_cost_usd)) * 100
        
        report_lines.extend([
            "Resource Usage:",
            f"  • Tokens used: {self.metrics.consumed_tokens:,} / {self.metrics.estimated_total_tokens:,} ({token_efficiency:.1f}%)",
            f"  • Estimated cost: ${self.metrics.estimated_cost_usd:.4f}",
            f"  • Actual cost: ${self.metrics.actual_cost_usd:.4f} ({cost_efficiency:.1f}%)",
            ""
        ])
        
        # Error and retry statistics
        if self.live_stats.total_retries > 0 or self.live_stats.rate_limit_hits > 0:
            report_lines.extend([
                "Error Statistics:",
                f"  • Total retries: {self.live_stats.total_retries}",
                f"  • Rate limit hits: {self.live_stats.rate_limit_hits}",
                f"  • Error rate: {self.live_stats.error_rate:.1%}",
                ""
            ])
            
            if self.live_stats.failure_breakdown:
                report_lines.append("  • Failure breakdown:")
                for failure_type, count in self.live_stats.failure_breakdown.items():
                    percentage = (count / max(1, self.live_stats.total_retries)) * 100
                    report_lines.append(f"    - {failure_type}: {count} ({percentage:.1f}%)")
                report_lines.append("")
        
        # Current status
        report_lines.extend([
            "Current Status:",
            f"  • Stage: {self.metrics.current_stage}",
            f"  • Operation: {self.metrics.current_operation or 'N/A'}",
            ""
        ])
        
        return "\n".join(report_lines)


# Utility functions

def create_progress_tracker(
    dashboard_type: DashboardType = DashboardType.TERMINAL,
    storage: Optional[BaseKVStorage] = None,
    update_interval: float = 1.0
) -> ProgressTracker:
    """Factory function to create a ProgressTracker"""
    return ProgressTracker(
        dashboard_type=dashboard_type,
        storage=storage,
        update_interval=update_interval
    )


# Context manager for easy usage
class ProgressContext:
    """Context manager for progress tracking"""
    
    def __init__(
        self,
        tracker: ProgressTracker,
        pipeline_estimate: Optional[PipelineEstimate] = None
    ):
        self.tracker = tracker
        self.pipeline_estimate = pipeline_estimate
    
    async def __aenter__(self):
        if self.pipeline_estimate:
            self.tracker.set_pipeline_estimate(self.pipeline_estimate)
        
        await self.tracker.start_tracking()
        return self.tracker
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.tracker.stop_tracking()
        
        if exc_type:
            self.tracker.add_error(exc_val, "Pipeline execution")
        
        # Generate final report
        final_report = await self.tracker.generate_progress_report()
        logger.info(f"Final progress report:\n{final_report}")


def progress_context(
    dashboard_type: DashboardType = DashboardType.TERMINAL,
    pipeline_estimate: Optional[PipelineEstimate] = None,
    **kwargs
) -> ProgressContext:
    """Create a progress tracking context manager"""
    tracker = create_progress_tracker(dashboard_type=dashboard_type, **kwargs)
    return ProgressContext(tracker, pipeline_estimate)