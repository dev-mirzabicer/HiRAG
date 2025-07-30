"""
Estimation Database for HiRAG Token Usage Learning

This module provides a specialized database for storing and analyzing actual
token usage data to continuously improve estimation accuracy over time.

Key Features:
- Store actual vs estimated token usage
- Analyze patterns and trends in token consumption
- Provide statistical insights for estimate refinement
- Export usage data for external analysis
- Automatic cleanup of old data
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics

from ._utils import logger, compute_mdhash_id
from .base import BaseKVStorage
from ._token_estimation import LLMCallType, TokenEstimate


@dataclass
class UsageRecord:
    """Single record of actual token usage"""

    call_type: str
    actual_input_tokens: int
    actual_output_tokens: int
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    model_name: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def input_accuracy(self) -> float:
        """Calculate input token estimation accuracy (0-1, 1 = perfect)"""
        if self.estimated_input_tokens == 0:
            return 0.0
        return 1.0 - abs(self.actual_input_tokens - self.estimated_input_tokens) / max(
            self.actual_input_tokens, 1
        )

    @property
    def output_accuracy(self) -> float:
        """Calculate output token estimation accuracy (0-1, 1 = perfect)"""
        if self.estimated_output_tokens == 0:
            return 0.0
        return 1.0 - abs(
            self.actual_output_tokens - self.estimated_output_tokens
        ) / max(self.actual_output_tokens, 1)

    @property
    def total_accuracy(self) -> float:
        """Calculate overall estimation accuracy"""
        return (self.input_accuracy + self.output_accuracy) / 2


@dataclass
class UsageStatistics:
    """Statistical analysis of usage patterns"""

    call_type: str
    total_calls: int
    avg_input_tokens: float
    avg_output_tokens: float
    median_input_tokens: float
    median_output_tokens: float
    std_input_tokens: float
    std_output_tokens: float
    min_input_tokens: int
    max_input_tokens: int
    min_output_tokens: int
    max_output_tokens: int
    avg_accuracy: float
    total_cost_usd: float = 0.0
    time_period_days: int = 0


class EstimationDatabase:
    """
    Specialized database for storing and analyzing token usage patterns

    This database helps the token estimation system learn from actual usage
    to continuously improve prediction accuracy.
    """

    def __init__(
        self, storage: BaseKVStorage, max_records: int = 10000, cleanup_days: int = 90
    ):
        self.storage = storage
        self.max_records = max_records
        self.cleanup_days = cleanup_days
        self._stats_cache = {}
        self._cache_expires = 0

        logger.info(
            f"EstimationDatabase initialized with max_records={max_records}, cleanup_days={cleanup_days}"
        )

    async def record_usage(
        self,
        call_type: LLMCallType,
        actual_input_tokens: int,
        actual_output_tokens: int,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        model_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a single token usage instance

        Returns:
            Record ID for future reference
        """
        record = UsageRecord(
            call_type=call_type.value,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            model_name=model_name,
            metadata=metadata or {},
        )

        record_id = compute_mdhash_id(
            f"{record.call_type}_{record.timestamp}_{actual_input_tokens}_{actual_output_tokens}",
            prefix="usage-",
        )

        await self.storage.upsert({record_id: asdict(record)})

        # Invalidate stats cache
        self._cache_expires = 0

        logger.debug(
            f"Recorded usage: {call_type.value} {actual_input_tokens}→{actual_output_tokens}"
        )
        return record_id

    async def get_usage_records(
        self,
        call_type: Optional[LLMCallType] = None,
        model_name: Optional[str] = None,
        days_back: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[UsageRecord]:
        """
        Retrieve usage records with optional filtering

        Args:
            call_type: Filter by specific call type
            model_name: Filter by model name
            days_back: Only include records from the last N days
            limit: Maximum number of records to return

        Returns:
            List of UsageRecord objects
        """
        all_records = await self.storage.get_all()

        # Convert to UsageRecord objects
        records = []
        for record_data in all_records.values():
            try:
                record = UsageRecord(**record_data)
                records.append(record)
            except Exception as e:
                logger.warning(f"Skipping invalid usage record: {e}")
                continue

        # Apply filters
        if call_type:
            records = [r for r in records if r.call_type == call_type.value]

        if model_name:
            records = [r for r in records if r.model_name == model_name]

        if days_back is not None:
            cutoff_time = time.time() - (days_back * 24 * 3600)
            records = [r for r in records if r.timestamp >= cutoff_time]

        # Sort by timestamp (newest first)
        records.sort(key=lambda r: r.timestamp, reverse=True)

        if limit:
            records = records[:limit]

        return records

    async def get_statistics(
        self,
        call_type: Optional[LLMCallType] = None,
        days_back: int = 30,
        use_cache: bool = True,
    ) -> Dict[str, UsageStatistics]:
        """
        Get comprehensive usage statistics

        Args:
            call_type: Specific call type to analyze (None for all)
            days_back: Time window for analysis
            use_cache: Whether to use cached results

        Returns:
            Dictionary mapping call types to their statistics
        """
        cache_key = f"{call_type}_{days_back}"

        # Check cache
        if (
            use_cache
            and time.time() < self._cache_expires
            and cache_key in self._stats_cache
        ):
            return self._stats_cache[cache_key]

        records = await self.get_usage_records(call_type=call_type, days_back=days_back)

        if not records:
            return {}

        # Group by call type
        by_call_type = defaultdict(list)
        for record in records:
            by_call_type[record.call_type].append(record)

        statistics_dict = {}

        for call_type_str, type_records in by_call_type.items():
            if not type_records:
                continue

            input_tokens = [r.actual_input_tokens for r in type_records]
            output_tokens = [r.actual_output_tokens for r in type_records]
            accuracies = [
                r.total_accuracy for r in type_records if r.total_accuracy > 0
            ]

            stats = UsageStatistics(
                call_type=call_type_str,
                total_calls=len(type_records),
                avg_input_tokens=statistics.mean(input_tokens),
                avg_output_tokens=statistics.mean(output_tokens),
                median_input_tokens=statistics.median(input_tokens),
                median_output_tokens=statistics.median(output_tokens),
                std_input_tokens=statistics.stdev(input_tokens)
                if len(input_tokens) > 1
                else 0,
                std_output_tokens=statistics.stdev(output_tokens)
                if len(output_tokens) > 1
                else 0,
                min_input_tokens=min(input_tokens),
                max_input_tokens=max(input_tokens),
                min_output_tokens=min(output_tokens),
                max_output_tokens=max(output_tokens),
                avg_accuracy=statistics.mean(accuracies) if accuracies else 0.0,
                time_period_days=days_back,
            )

            statistics_dict[call_type_str] = stats

        # Cache results
        self._stats_cache[cache_key] = statistics_dict
        self._cache_expires = time.time() + 300  # Cache for 5 minutes

        return statistics_dict

    async def generate_improvement_suggestions(
        self, days_back: int = 30
    ) -> Dict[str, List[str]]:
        """
        Analyze usage patterns and generate suggestions for improving estimates

        Returns:
            Dictionary mapping call types to lists of improvement suggestions
        """
        stats = await self.get_statistics(days_back=days_back)
        suggestions = {}

        for call_type, stat in stats.items():
            call_suggestions = []

            # Check accuracy
            if stat.avg_accuracy < 0.7:
                call_suggestions.append(
                    f"Low estimation accuracy ({stat.avg_accuracy:.2f}). Consider refining estimation parameters."
                )

            # Check variance
            if stat.std_output_tokens > stat.avg_output_tokens * 0.5:
                call_suggestions.append(
                    f"High output variance (std={stat.std_output_tokens:.0f}). "
                    f"Consider context-dependent estimation."
                )

            # Check for outliers
            if stat.max_output_tokens > stat.avg_output_tokens * 3:
                call_suggestions.append(
                    f"Detected output outliers (max={stat.max_output_tokens}, avg={stat.avg_output_tokens:.0f}). "
                    f"Investigate edge cases."
                )

            # Sample size recommendations
            if stat.total_calls < 20:
                call_suggestions.append(
                    f"Limited sample size ({stat.total_calls} calls). "
                    f"Collect more data for reliable statistics."
                )

            if call_suggestions:
                suggestions[call_type] = call_suggestions

        return suggestions

    async def export_usage_data(
        self,
        output_format: str = "json",
        call_type: Optional[LLMCallType] = None,
        days_back: Optional[int] = None,
    ) -> str:
        """
        Export usage data for external analysis

        Args:
            output_format: "json" or "csv"
            call_type: Filter by call type
            days_back: Time window for export

        Returns:
            Formatted string containing the data
        """
        records = await self.get_usage_records(call_type=call_type, days_back=days_back)

        if output_format.lower() == "csv":
            return self._export_as_csv(records)
        else:
            return self._export_as_json(records)

    def _export_as_json(self, records: List[UsageRecord]) -> str:
        """Export records as JSON"""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_records": len(records),
            "records": [asdict(record) for record in records],
        }
        return json.dumps(data, indent=2)

    def _export_as_csv(self, records: List[UsageRecord]) -> str:
        """Export records as CSV"""
        if not records:
            return "No records to export"

        # CSV header
        lines = [
            "timestamp,call_type,model_name,actual_input_tokens,actual_output_tokens,"
            "estimated_input_tokens,estimated_output_tokens,input_accuracy,output_accuracy,total_accuracy"
        ]

        # CSV data
        for record in records:
            lines.append(
                f"{datetime.fromtimestamp(record.timestamp).isoformat()},"
                f"{record.call_type},{record.model_name},"
                f"{record.actual_input_tokens},{record.actual_output_tokens},"
                f"{record.estimated_input_tokens},{record.estimated_output_tokens},"
                f"{record.input_accuracy:.3f},{record.output_accuracy:.3f},{record.total_accuracy:.3f}"
            )

        return "\n".join(lines)

    async def cleanup_old_records(self) -> int:
        """
        Remove old records to maintain database size

        Returns:
            Number of records removed
        """
        cutoff_time = time.time() - (self.cleanup_days * 24 * 3600)
        all_records = await self.storage.get_all()

        to_remove = []
        for record_id, record_data in all_records.items():
            try:
                record_timestamp = record_data.get("timestamp", 0)
                if record_timestamp < cutoff_time:
                    to_remove.append(record_id)
            except Exception as e:
                logger.warning(f"Error checking record {record_id}: {e}")
                to_remove.append(record_id)  # Remove corrupted records

        # Also enforce max_records limit
        if len(all_records) - len(to_remove) > self.max_records:
            # Sort remaining records by timestamp and remove oldest
            remaining_records = [
                (record_id, record_data.get("timestamp", 0))
                for record_id, record_data in all_records.items()
                if record_id not in to_remove
            ]
            remaining_records.sort(key=lambda x: x[1])  # Sort by timestamp

            excess_count = len(remaining_records) - self.max_records
            for record_id, _ in remaining_records[:excess_count]:
                to_remove.append(record_id)

        # Remove records
        for record_id in to_remove:
            await self.storage.delete(record_id)

        # Invalidate cache after cleanup
        self._cache_expires = 0

        logger.info(f"Cleaned up {len(to_remove)} old usage records")
        return len(to_remove)

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for the monitoring dashboard

        Returns:
            Dictionary with dashboard metrics
        """
        # Get statistics for different time windows
        stats_7d = await self.get_statistics(days_back=7)
        stats_30d = await self.get_statistics(days_back=30)

        # Get recent records for trending
        recent_records = await self.get_usage_records(days_back=7, limit=100)

        # Calculate trends
        trends = {}
        if len(recent_records) >= 10:
            mid_point = len(recent_records) // 2
            recent_half = recent_records[:mid_point]
            older_half = recent_records[mid_point:]

            for call_type in {r.call_type for r in recent_records}:
                recent_avg = statistics.mean(
                    [
                        r.actual_output_tokens
                        for r in recent_half
                        if r.call_type == call_type
                    ]
                    or [0]
                )
                older_avg = statistics.mean(
                    [
                        r.actual_output_tokens
                        for r in older_half
                        if r.call_type == call_type
                    ]
                    or [0]
                )

                if older_avg > 0:
                    trend = (recent_avg - older_avg) / older_avg
                    trends[call_type] = trend

        # Get improvement suggestions
        suggestions = await self.generate_improvement_suggestions()

        return {
            "last_updated": datetime.now().isoformat(),
            "statistics_7d": {k: asdict(v) for k, v in stats_7d.items()},
            "statistics_30d": {k: asdict(v) for k, v in stats_30d.items()},
            "trends": trends,
            "improvement_suggestions": suggestions,
            "total_records": len(await self.storage.get_all()),
            "recent_activity": len(recent_records),
        }

    async def generate_analysis_report(self, days_back: int = 30) -> str:
        """
        Generate a comprehensive analysis report

        Args:
            days_back: Time window for analysis

        Returns:
            Formatted report string
        """
        stats = await self.get_statistics(days_back=days_back)
        suggestions = await self.generate_improvement_suggestions(days_back)

        report_lines = [
            f"=== Token Usage Analysis Report ===",
            f"Analysis Period: Last {days_back} days",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Summary by Call Type:",
        ]

        total_calls = sum(stat.total_calls for stat in stats.values())
        total_input_tokens = sum(
            stat.avg_input_tokens * stat.total_calls for stat in stats.values()
        )
        total_output_tokens = sum(
            stat.avg_output_tokens * stat.total_calls for stat in stats.values()
        )

        report_lines.extend(
            [
                f"  • Total LLM calls: {total_calls:,}",
                f"  • Total input tokens: {total_input_tokens:,.0f}",
                f"  • Total output tokens: {total_output_tokens:,.0f}",
                f"  • Combined total: {total_input_tokens + total_output_tokens:,.0f}",
                "",
            ]
        )

        for call_type, stat in stats.items():
            report_lines.extend(
                [
                    f"{call_type.upper()}:",
                    f"  • Calls: {stat.total_calls}",
                    f"  • Avg input tokens: {stat.avg_input_tokens:.1f} (σ={stat.std_input_tokens:.1f})",
                    f"  • Avg output tokens: {stat.avg_output_tokens:.1f} (σ={stat.std_output_tokens:.1f})",
                    f"  • Range: {stat.min_output_tokens}-{stat.max_output_tokens} output tokens",
                    f"  • Estimation accuracy: {stat.avg_accuracy:.1%}",
                    "",
                ]
            )

        if suggestions:
            report_lines.extend(["Improvement Suggestions:", ""])

            for call_type, call_suggestions in suggestions.items():
                report_lines.append(f"{call_type.upper()}:")
                for suggestion in call_suggestions:
                    report_lines.append(f"  • {suggestion}")
                report_lines.append("")

        return "\n".join(report_lines)

    async def get_contextual_usage_records(
        self,
        days_back: int = 7,
        call_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get contextual usage records (enhanced records with rich metadata)

        Args:
            days_back: Number of days to look back
            call_type: Filter by specific call type
            limit: Maximum number of records to return

        Returns:
            List of contextual usage record dictionaries
        """
        all_records = await self.storage.get_all()
        contextual_records = []
        cutoff_time = time.time() - (days_back * 24 * 3600)

        for record_id, record_data in all_records.items():
            if (
                record_id.startswith("contextual-usage-")
                and record_data.get("timestamp", 0) >= cutoff_time
            ):
                if call_type is None or record_data.get("call_type") == call_type:
                    contextual_records.append(record_data)

        # Sort by timestamp (newest first)
        contextual_records.sort(key=lambda r: r.get("timestamp", 0), reverse=True)

        if limit:
            contextual_records = contextual_records[:limit]

        return contextual_records

    async def record_contextual_usage(
        self,
        call_type: str,
        actual_input_tokens: int,
        actual_output_tokens: int,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        model_name: str = "",
        chunk_size: Optional[int] = None,
        document_type: Optional[str] = None,
        success: bool = True,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record contextual usage with rich metadata

        Returns:
            Record ID for future reference
        """
        from ._utils import compute_mdhash_id

        # Calculate accuracies
        input_accuracy = None
        output_accuracy = None
        total_accuracy = None

        if estimated_input_tokens > 0:
            input_accuracy = 1.0 - abs(
                actual_input_tokens - estimated_input_tokens
            ) / max(actual_input_tokens, 1)

        if estimated_output_tokens > 0:
            output_accuracy = 1.0 - abs(
                actual_output_tokens - estimated_output_tokens
            ) / max(actual_output_tokens, 1)

        if input_accuracy is not None and output_accuracy is not None:
            total_accuracy = (input_accuracy + output_accuracy) / 2

        record = {
            "call_type": call_type,
            "actual_input_tokens": actual_input_tokens,
            "actual_output_tokens": actual_output_tokens,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "model_name": model_name,
            "timestamp": time.time(),
            "chunk_size": chunk_size,
            "document_type": document_type,
            "success": success,
            "latency_ms": latency_ms,
            "input_accuracy": input_accuracy,
            "output_accuracy": output_accuracy,
            "total_accuracy": total_accuracy,
            "metadata": metadata or {},
        }

        record_id = compute_mdhash_id(
            f"{call_type}_{actual_input_tokens}_{actual_output_tokens}_{time.time()}",
            prefix="contextual-usage-",
        )

        await self.storage.upsert({record_id: record})

        logger.debug(
            f"Recorded contextual usage: {call_type} {actual_input_tokens}→{actual_output_tokens}"
        )
        return record_id

    async def store_parameter(
        self,
        parameter_name: str,
        current_value: float,
        previous_value: Optional[float] = None,
        confidence: float = 1.0,
        sample_size: int = 1,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store a parameter value in the database

        Args:
            parameter_name: Name of the parameter
            current_value: Current parameter value
            previous_value: Previous value (if updating)
            confidence: Confidence in the value (0-1)
            sample_size: Number of samples the value is based on
            context: Context information

        Returns:
            Whether the storage was successful
        """
        try:
            parameter_record = {
                "current_value": current_value,
                "previous_value": previous_value,
                "confidence": confidence,
                "sample_size": sample_size,
                "last_updated": time.time(),
                "context": context or {},
            }

            await self.storage.upsert({f"param_{parameter_name}": parameter_record})

            logger.debug(f"Stored parameter {parameter_name}: {current_value}")
            return True

        except Exception as e:
            logger.error(f"Failed to store parameter {parameter_name}: {e}")
            return False

    async def get_parameter(self, parameter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a parameter record from the database

        Args:
            parameter_name: Name of the parameter

        Returns:
            Parameter record dictionary or None if not found
        """
        try:
            return await self.storage.get_by_id(f"param_{parameter_name}")
        except Exception as e:
            logger.error(f"Failed to get parameter {parameter_name}: {e}")
            return None

    async def store_parameter_update(
        self,
        parameter_name: str,
        old_value: float,
        new_value: float,
        confidence: float,
        sample_size: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a parameter update in the history

        Returns:
            Update record ID
        """
        from ._utils import compute_mdhash_id

        update_record = {
            "parameter_name": parameter_name,
            "old_value": old_value,
            "new_value": new_value,
            "confidence": confidence,
            "sample_size": sample_size,
            "timestamp": time.time(),
            "context": context or {},
        }

        update_id = compute_mdhash_id(
            f"param_update_{parameter_name}_{int(time.time())}", prefix="param-update-"
        )

        await self.storage.upsert({update_id: update_record})

        logger.debug(
            f"Stored parameter update: {parameter_name} {old_value}→{new_value}"
        )
        return update_id

    async def get_parameter_history(
        self, parameter_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get parameter update history

        Args:
            parameter_name: Name of the parameter
            limit: Maximum number of updates to return

        Returns:
            List of parameter update records
        """
        all_records = await self.storage.get_all()

        # Filter for parameter updates
        updates = []
        for record_id, record_data in all_records.items():
            if (
                record_id.startswith("param-update-")
                and record_data.get("parameter_name") == parameter_name
            ):
                updates.append(record_data)

        # Sort by timestamp (newest first) and limit
        updates.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return updates[:limit]

    async def get_learning_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get statistics about the learning system performance

        Args:
            days_back: Number of days to analyze

        Returns:
            Learning statistics dictionary
        """
        contextual_records = await self.get_contextual_usage_records(
            days_back=days_back
        )

        if not contextual_records:
            return {"error": "No contextual usage data available"}

        # Analyze accuracy trends
        accuracy_by_call_type = {}
        for record in contextual_records:
            call_type = record.get("call_type")
            total_accuracy = record.get("total_accuracy")

            if call_type and total_accuracy is not None:
                if call_type not in accuracy_by_call_type:
                    accuracy_by_call_type[call_type] = []
                accuracy_by_call_type[call_type].append(total_accuracy)

        # Calculate statistics
        statistics_summary = {}
        for call_type, accuracies in accuracy_by_call_type.items():
            if len(accuracies) > 0:
                statistics_summary[call_type] = {
                    "avg_accuracy": sum(accuracies) / len(accuracies),
                    "min_accuracy": min(accuracies),
                    "max_accuracy": max(accuracies),
                    "sample_count": len(accuracies),
                }

        # Get parameter update statistics
        all_records = await self.storage.get_all()
        parameter_updates = []
        cutoff_time = time.time() - (days_back * 24 * 3600)

        for record_id, record_data in all_records.items():
            if (
                record_id.startswith("param-update-")
                and record_data.get("timestamp", 0) >= cutoff_time
            ):
                parameter_updates.append(record_data)

        return {
            "analysis_period_days": days_back,
            "total_contextual_records": len(contextual_records),
            "accuracy_by_call_type": statistics_summary,
            "parameter_updates_count": len(parameter_updates),
            "learning_active": len(parameter_updates) > 0,
            "last_updated": datetime.now().isoformat(),
        }


# Utility functions


async def create_estimation_database(
    storage: BaseKVStorage, max_records: int = 10000, cleanup_days: int = 90
) -> EstimationDatabase:
    """Factory function to create an EstimationDatabase instance"""
    db = EstimationDatabase(storage, max_records, cleanup_days)

    # Perform initial cleanup
    await db.cleanup_old_records()

    return db
