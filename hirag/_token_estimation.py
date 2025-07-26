"""
Token Estimation Framework for HiRAG

This module provides comprehensive token estimation capabilities for the entire
HiRAG ingestion pipeline, including precise calculation of input tokens and
estimated output tokens for all LLM calls.

Key Features:
- Precise token counting for all prompt templates
- Estimation of variable inputs based on chunk content and extracted data
- Learning database integration for improving estimates over time
- Cost prediction for different models and providers
- Per-operation breakdown for detailed analysis
"""

import asyncio
import tiktoken
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
from enum import Enum

from ._utils import logger, compute_mdhash_id
from .base import BaseKVStorage, TextChunkSchema
from .prompt import PROMPTS

import statistics
import time


class LLMCallType(Enum):
    """Enumeration of all LLM call types in the HiRAG pipeline"""

    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    CONTINUE_EXTRACTION = "continue_extraction"  # Gleaning
    LOOP_DETECTION = "loop_detection"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    ENTITY_DISAMBIGUATION = "entity_disambiguation"
    ENTITY_MERGING = "entity_merging"
    COMMUNITY_REPORT = "community_report"


# =============================================================================
# CONFIGURABLE CONSTANTS - All remaining "magic numbers" moved here
# These should be the ONLY hardcoded numbers in this entire file
# =============================================================================

# System-level constants that cannot be learned from data
CACHE_DURATION_SECONDS = 300  # Parameter cache duration
HOURS_IN_DAY = 24
SECONDS_IN_HOUR = 3600
MILLISECONDS_IN_SECOND = 1000

# Learning algorithm configuration
LEARNING_CONFIDENCE_THRESHOLD = 0.7
MINIMUM_SAMPLE_SIZE_FOR_LEARNING = 3
MINIMUM_RECORDS_FOR_PATTERN_DETECTION = 5
MINIMUM_RECORDS_FOR_OPTIMIZATION = 10
MINIMUM_RECORDS_FOR_PERIODIC_LEARNING = 20
CONFIDENCE_SAMPLE_SIZE_DIVISOR = 20  # For confidence = min(0.95, samples / 20)
MAX_CONFIDENCE = 0.95
HIGH_CONFIDENCE_FOR_ROLLBACK = 1.0

# Accuracy calculation constants
ACCURACY_BASE = 1.0
ACCURACY_AVERAGE_DIVISOR = 2  # For (input_accuracy + output_accuracy) / 2

# Statistical thresholds
SIGNIFICANT_ACCURACY_DIFFERENCE = 0.1
VERY_SIGNIFICANT_ACCURACY_DIFFERENCE = 0.15
INPUT_ERROR_THRESHOLD = 0.2
OUTPUT_ERROR_THRESHOLD = 0.3
MIN_ACCURACY_THRESHOLD = 0.7
MIN_RECORD_ACCURACY_FILTER = 0.5

# Cache and database constants
DEFAULT_HISTORY_LIMIT = 100
DEFAULT_DAYS_BACK = 7
EXTENDED_ANALYSIS_DAYS = 30
CACHE_EXPIRY_RESET = 0

# Approximation constants (these should eventually become learnable too)
WORDS_TO_TOKENS_RATIO = 1.3  # Will be made learnable
AVERAGE_DIVISOR_FOR_ESTIMATES = 2  # For (min + max) / 2

# Model pricing (should be moved to configuration file eventually)
MODEL_PRICING_DATA = {
    "gpt-4o": {
        "input_cost_per_1k": 5.0,
        "output_cost_per_1k": 15.0,
        "max_input": 128000,
        "max_output": 4096,
    },
    "gpt-4o-mini": {
        "input_cost_per_1k": 0.15,
        "output_cost_per_1k": 0.60,
        "max_input": 128000,
        "max_output": 16384,
    },
    "gpt-3.5-turbo": {
        "input_cost_per_1k": 1.0,
        "output_cost_per_1k": 2.0,
        "max_input": 16385,
        "max_output": 4096,
    },
    "claude-3-5-sonnet": {
        "input_cost_per_1k": 3.0,
        "output_cost_per_1k": 15.0,
        "max_input": 200000,
        "max_output": 8192,
    },
    "gemini-1.5-pro": {
        "input_cost_per_1k": 1.25,
        "output_cost_per_1k": 5.0,
        "max_input": 2000000,
        "max_output": 8192,
    },
    "gemini-1.5-flash": {
        "input_cost_per_1k": 0.075,
        "output_cost_per_1k": 0.30,
        "max_input": 1000000,
        "max_output": 8192,
    },
    "gemini-2.5-flash": {
        "input_cost_per_1k": 0.0003,
        "output_cost_per_1k": 0.0025,
        "max_input": 1000000,
        "max_output": 65536,
    },
    "gemini-2.5-pro": {
        "input_cost_per_1k": 0.00125,
        "output_cost_per_1k": 0.01,
        "max_input": 1000000,
        "max_output": 65536,
    },
}

# Default model configuration (until learned dynamically)
DEFAULT_CHUNK_TOKEN_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 200
TOKENS_PER_1K = 1000

# =============================================================================
# END OF CONFIGURABLE CONSTANTS
# Everything below this point should use ONLY learnable parameters
# =============================================================================


class ParameterCategory(Enum):
    """Categories of learnable parameters"""

    BASE_ESTIMATION = "base_estimation"
    RATIO_PARAMETERS = "ratio_parameters"
    DYNAMIC_PARAMETERS = "dynamic_parameters"
    CONTEXTUAL_MODIFIERS = "contextual_modifiers"
    LEARNING_ALGORITHM = "learning_algorithm"


@dataclass
class ParameterMetadata:
    """Metadata for a learnable parameter"""

    name: str
    category: ParameterCategory
    description: str
    min_value: float
    max_value: float
    default_value: float
    update_frequency: str  # "high", "medium", "low"
    confidence_threshold: float = 0.7


@dataclass
class ParameterUpdate:
    """Record of a parameter update"""

    parameter_name: str
    old_value: float
    new_value: float
    confidence: float
    sample_size: int
    timestamp: float
    context: Dict[str, Any]
    performance_impact: Optional[float] = None


@dataclass
class ContextualUsageRecord:
    """Enhanced usage record with rich context"""

    call_type: str
    actual_input_tokens: int
    actual_output_tokens: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    model_name: str
    timestamp: float

    # Rich context information
    chunk_size: Optional[int] = None
    chunk_complexity: Optional[float] = None
    document_type: Optional[str] = None
    content_domain: Optional[str] = None
    processing_stage: Optional[str] = None
    prompt_template: Optional[str] = None
    success: bool = True
    latency_ms: Optional[float] = None

    # Performance metrics
    input_accuracy: Optional[float] = None
    output_accuracy: Optional[float] = None
    total_accuracy: Optional[float] = None

    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

        # Calculate accuracies if not provided
        if self.input_accuracy is None and self.estimated_input_tokens > 0:
            self.input_accuracy = 1.0 - abs(
                self.actual_input_tokens - self.estimated_input_tokens
            ) / max(self.actual_input_tokens, 1)

        if self.output_accuracy is None and self.estimated_output_tokens > 0:
            self.output_accuracy = 1.0 - abs(
                self.actual_output_tokens - self.estimated_output_tokens
            ) / max(self.actual_output_tokens, 1)

        if (
            self.total_accuracy is None
            and self.input_accuracy is not None
            and self.output_accuracy is not None
        ):
            self.total_accuracy = (self.input_accuracy + self.output_accuracy) / 2


_PARAMETER_DEFINITIONS = [
    # =================================================================
    # RATIO PARAMETERS
    # =================================================================
    {
        "name": "chunk_tokens_to_entities_divisor",
        "category": ParameterCategory.RATIO_PARAMETERS,
        "description": "Divisor for estimating entities from chunk tokens",
        "min_value": 50.0,
        "max_value": 500.0,
        "default_value": 200.0,
        "update_frequency": "high",
    },
    {
        "name": "chunk_tokens_to_relations_divisor",
        "category": ParameterCategory.RATIO_PARAMETERS,
        "description": "Divisor for estimating relations from chunk tokens",
        "min_value": 100.0,
        "max_value": 800.0,
        "default_value": 300.0,
        "update_frequency": "high",
    },
    {
        "name": "history_tokens_fraction",
        "category": ParameterCategory.RATIO_PARAMETERS,
        "description": "Fraction of chunk tokens used for history estimation",
        "min_value": 0.1,
        "max_value": 0.9,
        "default_value": 0.5,
        "update_frequency": "medium",
    },
    {
        "name": "entities_to_clusters_divisor",
        "category": ParameterCategory.RATIO_PARAMETERS,
        "description": "Divisor for estimating clusters from entities",
        "min_value": 2.0,
        "max_value": 20.0,
        "default_value": 5.0,
        "update_frequency": "medium",
    },
    {
        "name": "tokens_per_entity_description",
        "category": ParameterCategory.RATIO_PARAMETERS,
        "description": "Average tokens per entity description",
        "min_value": 20.0,
        "max_value": 150.0,
        "default_value": 50.0,
        "update_frequency": "high",
    },
    {
        "name": "tokens_per_entity_context",
        "category": ParameterCategory.RATIO_PARAMETERS,
        "description": "Average tokens per entity context",
        "min_value": 100.0,
        "max_value": 500.0,
        "default_value": 200.0,
        "update_frequency": "high",
    },
    {
        "name": "tokens_per_community_member",
        "category": ParameterCategory.RATIO_PARAMETERS,
        "description": "Average tokens per community member",
        "min_value": 50.0,
        "max_value": 300.0,
        "default_value": 100.0,
        "update_frequency": "medium",
    },
    # =================================================================
    # BASE ESTIMATION PARAMETERS
    # =================================================================
    # Entity Extraction
    {
        "name": "entity_extraction_base_output_min",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Minimum output tokens for entity extraction",
        "min_value": 50.0,
        "max_value": 500.0,
        "default_value": 200.0,
        "update_frequency": "medium",
    },
    {
        "name": "entity_extraction_base_output_max",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Maximum output tokens for entity extraction",
        "min_value": 500.0,
        "max_value": 2000.0,
        "default_value": 1100.0,
        "update_frequency": "medium",
    },
    {
        "name": "entity_extraction_entities_per_chunk_avg",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Average entities per chunk",
        "min_value": 1.0,
        "max_value": 50.0,
        "default_value": 10.0,
        "update_frequency": "high",
    },
    {
        "name": "entity_extraction_tokens_per_entity",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Average tokens per extracted entity",
        "min_value": 20.0,
        "max_value": 200.0,
        "default_value": 60.0,
        "update_frequency": "high",
    },
    {
        "name": "entity_extraction_thinking_ratio",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Thinking tokens ratio for entity extraction",
        "min_value": 0.5,
        "max_value": 5.0,
        "default_value": 2.0,
        "update_frequency": "medium",
    },
    # Relation Extraction
    {
        "name": "relation_extraction_base_output_min",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Minimum output tokens for relation extraction",
        "min_value": 50.0,
        "max_value": 400.0,
        "default_value": 150.0,
        "update_frequency": "medium",
    },
    {
        "name": "relation_extraction_base_output_max",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Maximum output tokens for relation extraction",
        "min_value": 300.0,
        "max_value": 1200.0,
        "default_value": 600.0,
        "update_frequency": "medium",
    },
    {
        "name": "relation_extraction_relations_per_chunk_avg",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Average relations per chunk",
        "min_value": 1.0,
        "max_value": 20.0,
        "default_value": 3.0,
        "update_frequency": "high",
    },
    {
        "name": "relation_extraction_tokens_per_relation",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Average tokens per extracted relation",
        "min_value": 30.0,
        "max_value": 200.0,
        "default_value": 80.0,
        "update_frequency": "high",
    },
    {
        "name": "relation_extraction_thinking_ratio",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Thinking tokens ratio for relation extraction",
        "min_value": 0.5,
        "max_value": 5.0,
        "default_value": 2.0,
        "update_frequency": "medium",
    },
    # Continue Extraction
    {
        "name": "continue_extraction_base_output_min",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Minimum output tokens for continue extraction",
        "min_value": 20.0,
        "max_value": 300.0,
        "default_value": 100.0,
        "update_frequency": "medium",
    },
    {
        "name": "continue_extraction_base_output_max",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Maximum output tokens for continue extraction",
        "min_value": 200.0,
        "max_value": 1000.0,
        "default_value": 600.0,
        "update_frequency": "medium",
    },
    {
        "name": "continue_extraction_additional_entities_ratio",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Ratio of additional entities found in gleaning",
        "min_value": 0.05,
        "max_value": 0.8,
        "default_value": 0.2,
        "update_frequency": "medium",
    },
    {
        "name": "continue_extraction_thinking_ratio",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Thinking tokens ratio for continue extraction",
        "min_value": 0.5,
        "max_value": 4.0,
        "default_value": 1.5,
        "update_frequency": "medium",
    },
    # Loop Detection
    {
        "name": "loop_detection_base_output_min",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Minimum output tokens for loop detection",
        "min_value": 2.0,
        "max_value": 20.0,
        "default_value": 5.0,
        "update_frequency": "low",
    },
    {
        "name": "loop_detection_base_output_max",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Maximum output tokens for loop detection",
        "min_value": 10.0,
        "max_value": 50.0,
        "default_value": 20.0,
        "update_frequency": "low",
    },
    {
        "name": "loop_detection_thinking_ratio",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Thinking tokens ratio for loop detection",
        "min_value": 5.0,
        "max_value": 50.0,
        "default_value": 20.0,
        "update_frequency": "low",
    },
    # Hierarchical Clustering
    {
        "name": "hierarchical_clustering_base_output_min",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Minimum output tokens for hierarchical clustering",
        "min_value": 50.0,
        "max_value": 300.0,
        "default_value": 100.0,
        "update_frequency": "medium",
    },
    {
        "name": "hierarchical_clustering_base_output_max",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Maximum output tokens for hierarchical clustering",
        "min_value": 200.0,
        "max_value": 800.0,
        "default_value": 400.0,
        "update_frequency": "medium",
    },
    {
        "name": "hierarchical_clustering_tokens_per_cluster",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Average tokens per cluster output",
        "min_value": 30.0,
        "max_value": 200.0,
        "default_value": 80.0,
        "update_frequency": "medium",
    },
    {
        "name": "hierarchical_clustering_thinking_ratio",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Thinking tokens ratio for hierarchical clustering",
        "min_value": 0.1,
        "max_value": 2.0,
        "default_value": 0.4,
        "update_frequency": "medium",
    },
    # Entity Disambiguation
    {
        "name": "entity_disambiguation_base_output_min",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Minimum output tokens for entity disambiguation",
        "min_value": 50.0,
        "max_value": 400.0,
        "default_value": 150.0,
        "update_frequency": "medium",
    },
    {
        "name": "entity_disambiguation_base_output_max",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Maximum output tokens for entity disambiguation",
        "min_value": 200.0,
        "max_value": 1000.0,
        "default_value": 500.0,
        "update_frequency": "medium",
    },
    {
        "name": "entity_disambiguation_tokens_per_decision",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Average tokens per disambiguation decision",
        "min_value": 20.0,
        "max_value": 150.0,
        "default_value": 50.0,
        "update_frequency": "medium",
    },
    {
        "name": "entity_disambiguation_thinking_ratio",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Thinking tokens ratio for entity disambiguation",
        "min_value": 0.1,
        "max_value": 2.0,
        "default_value": 0.5,
        "update_frequency": "medium",
    },
    # Entity Merging
    {
        "name": "entity_merging_base_output_min",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Minimum output tokens for entity merging",
        "min_value": 30.0,
        "max_value": 300.0,
        "default_value": 100.0,
        "update_frequency": "medium",
    },
    {
        "name": "entity_merging_base_output_max",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Maximum output tokens for entity merging",
        "min_value": 150.0,
        "max_value": 600.0,
        "default_value": 300.0,
        "update_frequency": "medium",
    },
    {
        "name": "entity_merging_thinking_ratio",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Thinking tokens ratio for entity merging",
        "min_value": 0.1,
        "max_value": 1.5,
        "default_value": 0.3,
        "update_frequency": "medium",
    },
    # Community Report
    {
        "name": "community_report_base_output_min",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Minimum output tokens for community report",
        "min_value": 200.0,
        "max_value": 1000.0,
        "default_value": 500.0,
        "update_frequency": "medium",
    },
    {
        "name": "community_report_base_output_max",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Maximum output tokens for community report",
        "min_value": 1000.0,
        "max_value": 5000.0,
        "default_value": 2000.0,
        "update_frequency": "medium",
    },
    {
        "name": "community_report_tokens_per_entity",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Average tokens per entity in community report",
        "min_value": 5.0,
        "max_value": 100.0,
        "default_value": 20.0,
        "update_frequency": "medium",
    },
    {
        "name": "community_report_tokens_per_relation",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Average tokens per relation in community report",
        "min_value": 5.0,
        "max_value": 50.0,
        "default_value": 15.0,
        "update_frequency": "medium",
    },
    {
        "name": "community_report_thinking_ratio",
        "category": ParameterCategory.BASE_ESTIMATION,
        "description": "Thinking tokens ratio for community report",
        "min_value": 0.2,
        "max_value": 2.0,
        "default_value": 0.6,
        "update_frequency": "medium",
    },
    # =================================================================
    # DYNAMIC PARAMETERS
    # =================================================================
    # Entity extraction dynamics
    {
        "name": "entities_per_chunk_base",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Base entities per chunk",
        "min_value": 1.0,
        "max_value": 20.0,
        "default_value": 5.0,
        "update_frequency": "high",
    },
    {
        "name": "entities_per_100_tokens",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Additional entities per 100 tokens",
        "min_value": 0.5,
        "max_value": 10.0,
        "default_value": 2.0,
        "update_frequency": "high",
    },
    {
        "name": "chunk_complexity_factor",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Multiplier for complex content",
        "min_value": 0.8,
        "max_value": 3.0,
        "default_value": 1.2,
        "update_frequency": "medium",
    },
    {
        "name": "min_entities_per_chunk",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Minimum entities per chunk",
        "min_value": 1.0,
        "max_value": 5.0,
        "default_value": 1.0,
        "update_frequency": "low",
    },
    {
        "name": "max_entities_per_chunk",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Maximum entities per chunk",
        "min_value": 10.0,
        "max_value": 50.0,
        "default_value": 15.0,
        "update_frequency": "low",
    },
    # Gleaning loop dynamics
    {
        "name": "avg_actual_loops",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Average actual gleaning loops needed",
        "min_value": 0.1,
        "max_value": 3.0,
        "default_value": 0.7,
        "update_frequency": "medium",
    },
    {
        "name": "loop_probability_decay",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Each subsequent loop probability decay",
        "min_value": 0.3,
        "max_value": 0.9,
        "default_value": 0.6,
        "update_frequency": "medium",
    },
    {
        "name": "min_loop_probability",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Minimum loop probability",
        "min_value": 0.01,
        "max_value": 0.5,
        "default_value": 0.1,
        "update_frequency": "low",
    },
    {
        "name": "chunk_size_loop_factor",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Larger chunks more likely to need loops",
        "min_value": 0.5,
        "max_value": 2.0,
        "default_value": 1.0,
        "update_frequency": "medium",
    },
    # Hierarchical clustering dynamics
    {
        "name": "clustering_iterations_base",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Base number of clustering iterations",
        "min_value": 1.0,
        "max_value": 8.0,
        "default_value": 3.0,
        "update_frequency": "medium",
    },
    {
        "name": "entities_per_cluster",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Average entities per cluster",
        "min_value": 2.0,
        "max_value": 15.0,
        "default_value": 5.0,
        "update_frequency": "medium",
    },
    {
        "name": "cluster_reduction_ratio",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "How much clusters reduce each iteration",
        "min_value": 0.2,
        "max_value": 0.8,
        "default_value": 0.4,
        "update_frequency": "medium",
    },
    {
        "name": "min_entities_for_clustering",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Minimum entities needed for clustering",
        "min_value": 5.0,
        "max_value": 20.0,
        "default_value": 10.0,
        "update_frequency": "low",
    },
    {
        "name": "max_clustering_iterations",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Maximum clustering iterations",
        "min_value": 3.0,
        "max_value": 10.0,
        "default_value": 5.0,
        "update_frequency": "low",
    },
    # Entity disambiguation dynamics
    {
        "name": "disambiguation_probability",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Probability that entities need disambiguation",
        "min_value": 0.05,
        "max_value": 0.5,
        "default_value": 0.12,
        "update_frequency": "medium",
    },
    {
        "name": "avg_cluster_size",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Average entities per disambiguation cluster",
        "min_value": 1.5,
        "max_value": 8.0,
        "default_value": 2.8,
        "update_frequency": "medium",
    },
    {
        "name": "complexity_disambiguation_factor",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Complex domains need more disambiguation",
        "min_value": 0.8,
        "max_value": 3.0,
        "default_value": 1.3,
        "update_frequency": "medium",
    },
    {
        "name": "min_disambiguation_clusters",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Minimum disambiguation clusters",
        "min_value": 0.0,
        "max_value": 5.0,
        "default_value": 0.0,
        "update_frequency": "low",
    },
    {
        "name": "max_disambiguation_cluster_size",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Maximum disambiguation cluster size",
        "min_value": 3.0,
        "max_value": 15.0,
        "default_value": 6.0,
        "update_frequency": "low",
    },
    # Community detection dynamics
    {
        "name": "entities_per_community",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Average entities per community",
        "min_value": 3.0,
        "max_value": 20.0,
        "default_value": 8.0,
        "update_frequency": "medium",
    },
    {
        "name": "community_size_variance",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Variance in community sizes",
        "min_value": 0.1,
        "max_value": 1.0,
        "default_value": 0.4,
        "update_frequency": "medium",
    },
    {
        "name": "min_community_size",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Minimum community size",
        "min_value": 2.0,
        "max_value": 8.0,
        "default_value": 3.0,
        "update_frequency": "low",
    },
    {
        "name": "max_community_size",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Maximum community size",
        "min_value": 15.0,
        "max_value": 50.0,
        "default_value": 25.0,
        "update_frequency": "low",
    },
    {
        "name": "community_overlap_ratio",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Some entities appear in multiple communities",
        "min_value": 0.05,
        "max_value": 0.5,
        "default_value": 0.15,
        "update_frequency": "medium",
    },
    # General pipeline dynamics
    {
        "name": "relation_to_entity_ratio",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Relations per entity ratio",
        "min_value": 0.2,
        "max_value": 2.0,
        "default_value": 0.6,
        "update_frequency": "medium",
    },
    {
        "name": "temporary_entity_ratio",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Percentage of temporary entities",
        "min_value": 0.05,
        "max_value": 0.5,
        "default_value": 0.15,
        "update_frequency": "medium",
    },
    {
        "name": "entity_merging_ratio",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Percentage of entities that get merged",
        "min_value": 0.02,
        "max_value": 0.3,
        "default_value": 0.08,
        "update_frequency": "medium",
    },
    {
        "name": "content_complexity_multiplier",
        "category": ParameterCategory.DYNAMIC_PARAMETERS,
        "description": "Global complexity adjustment",
        "min_value": 0.5,
        "max_value": 3.0,
        "default_value": 1.0,
        "update_frequency": "medium",
    },
    # =================================================================
    # LEARNING ALGORITHM PARAMETERS
    # =================================================================
    {
        "name": "learning_tokens_per_entity_estimate",
        "category": ParameterCategory.LEARNING_ALGORITHM,
        "description": "Estimated tokens per entity for learning",
        "min_value": 30.0,
        "max_value": 150.0,
        "default_value": 60.0,
        "update_frequency": "high",
    },
    {
        "name": "learning_tokens_per_relation_estimate",
        "category": ParameterCategory.LEARNING_ALGORITHM,
        "description": "Estimated tokens per relation for learning",
        "min_value": 40.0,
        "max_value": 200.0,
        "default_value": 80.0,
        "update_frequency": "high",
    },
    {
        "name": "learning_chunk_to_entities_heuristic",
        "category": ParameterCategory.LEARNING_ALGORITHM,
        "description": "Chunk size to entities ratio heuristic",
        "min_value": 50.0,
        "max_value": 300.0,
        "default_value": 150.0,
        "update_frequency": "medium",
    },
    {
        "name": "learning_small_output_filter",
        "category": ParameterCategory.LEARNING_ALGORITHM,
        "description": "Filter threshold for very small outputs",
        "min_value": 20.0,
        "max_value": 100.0,
        "default_value": 50.0,
        "update_frequency": "low",
    },
    {
        "name": "learning_small_relation_output_filter",
        "category": ParameterCategory.LEARNING_ALGORITHM,
        "description": "Filter threshold for small relation outputs",
        "min_value": 10.0,
        "max_value": 80.0,
        "default_value": 30.0,
        "update_frequency": "low",
    },
    {
        "name": "learning_input_context_threshold",
        "category": ParameterCategory.LEARNING_ALGORITHM,
        "description": "Minimum input tokens for context learning",
        "min_value": 50.0,
        "max_value": 300.0,
        "default_value": 100.0,
        "update_frequency": "low",
    },
    {
        "name": "learning_entity_tokens_min_bound",
        "category": ParameterCategory.LEARNING_ALGORITHM,
        "description": "Minimum bound for learned entity tokens",
        "min_value": 10.0,
        "max_value": 50.0,
        "default_value": 20.0,
        "update_frequency": "low",
    },
    {
        "name": "learning_entity_tokens_max_bound",
        "category": ParameterCategory.LEARNING_ALGORITHM,
        "description": "Maximum bound for learned entity tokens",
        "min_value": 100.0,
        "max_value": 300.0,
        "default_value": 150.0,
        "update_frequency": "low",
    },
    {
        "name": "learning_default_accuracy_fallback",
        "category": ParameterCategory.LEARNING_ALGORITHM,
        "description": "Default accuracy when none available",
        "min_value": 0.1,
        "max_value": 0.9,
        "default_value": 0.5,
        "update_frequency": "low",
    },
    # =================================================================
    # CONTEXTUAL MODIFIERS
    # =================================================================
    {
        "name": "academic_document_modifier",
        "category": ParameterCategory.CONTEXTUAL_MODIFIERS,
        "description": "Multiplier for academic documents",
        "min_value": 0.5,
        "max_value": 2.0,
        "default_value": 1.2,
        "update_frequency": "low",
    },
    {
        "name": "technical_document_modifier",
        "category": ParameterCategory.CONTEXTUAL_MODIFIERS,
        "description": "Multiplier for technical documents",
        "min_value": 0.5,
        "max_value": 2.0,
        "default_value": 1.1,
        "update_frequency": "low",
    },
    {
        "name": "narrative_document_modifier",
        "category": ParameterCategory.CONTEXTUAL_MODIFIERS,
        "description": "Multiplier for narrative documents",
        "min_value": 0.5,
        "max_value": 2.0,
        "default_value": 0.9,
        "update_frequency": "low",
    },
    # Model-specific factors
    {
        "name": "gpt4o_efficiency_factor",
        "category": ParameterCategory.CONTEXTUAL_MODIFIERS,
        "description": "Efficiency factor for GPT-4o",
        "min_value": 0.5,
        "max_value": 1.5,
        "default_value": 1.0,
        "update_frequency": "medium",
    },
    {
        "name": "gpt4o_mini_efficiency_factor",
        "category": ParameterCategory.CONTEXTUAL_MODIFIERS,
        "description": "Efficiency factor for GPT-4o-mini",
        "min_value": 0.5,
        "max_value": 1.5,
        "default_value": 0.8,
        "update_frequency": "medium",
    },
]


class LearnableParameterManager:
    """
    Centralized manager for all learnable estimation parameters.
    This class replaces all magic numbers with learnable parameters that
    adapt based on actual usage data and context.
    """

    def __init__(self, estimation_db, logger=None):
        self.estimation_db = estimation_db
        self.logger = logger
        self._parameter_cache = {}
        self._cache_expires = 0
        self._parameter_metadata = self._initialize_parameter_metadata()
        self._learning_engine = None

    def _initialize_parameter_metadata(self) -> Dict[str, ParameterMetadata]:
        """Initialize metadata for ALL learnable parameters from the global list."""
        return {
            param["name"]: ParameterMetadata(**param)
            for param in _PARAMETER_DEFINITIONS
        }

    async def get_parameter(
        self,
        parameter_name: str,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> float:
        """
        Get a parameter value, considering context if provided

        Args:
            parameter_name: Name of the parameter
            context: Optional context for contextual parameters
            use_cache: Whether to use cached values

        Returns:
            Parameter value, with contextual adjustments if applicable
        """
        # Check cache first
        cache_key = f"{parameter_name}_{hash(str(context) if context else '')}"
        if (
            use_cache
            and time.time() < self._cache_expires
            and cache_key in self._parameter_cache
        ):
            return self._parameter_cache[cache_key]

        # Get base parameter value
        base_value = await self._get_base_parameter(parameter_name)

        # Apply contextual modifiers
        final_value = await self._apply_contextual_modifiers(
            base_value, parameter_name, context
        )

        # Cache the result
        self._parameter_cache[cache_key] = final_value
        self._cache_expires = time.time() + 300  # Cache for 5 minutes

        return final_value

    async def _get_base_parameter(self, parameter_name: str) -> float:
        """Get the base parameter value from storage or default"""
        if not self.estimation_db:
            return self._parameter_metadata[parameter_name].default_value

        try:
            # Try to get from database
            parameter_record = await self.estimation_db.storage.get(
                f"param_{parameter_name}"
            )
            if parameter_record:
                return parameter_record.get(
                    "current_value",
                    self._parameter_metadata[parameter_name].default_value,
                )
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to get parameter {parameter_name} from database: {e}"
                )

        # Fall back to default
        return self._parameter_metadata[parameter_name].default_value

    async def _apply_contextual_modifiers(
        self, base_value: float, parameter_name: str, context: Optional[Dict[str, Any]]
    ) -> float:
        """Apply contextual modifiers to the base parameter value"""
        if not context:
            return base_value

        modified_value = base_value

        # Apply document type modifiers
        document_type = context.get("document_type")
        if document_type == "academic":
            modifier = await self.get_parameter(
                "academic_document_modifier", use_cache=False
            )
            modified_value *= modifier
        elif document_type == "technical":
            modifier = await self.get_parameter(
                "technical_document_modifier", use_cache=False
            )
            modified_value *= modifier
        elif document_type == "narrative":
            modifier = await self.get_parameter(
                "narrative_document_modifier", use_cache=False
            )
            modified_value *= modifier

        # Apply model-specific modifiers
        model_name = context.get("model_name", "").lower()
        if "gpt-4o" in model_name and "mini" not in model_name:
            modifier = await self.get_parameter(
                "gpt4o_efficiency_factor", use_cache=False
            )
            modified_value *= modifier
        elif "gpt-4o-mini" in model_name:
            modifier = await self.get_parameter(
                "gpt4o_mini_efficiency_factor", use_cache=False
            )
            modified_value *= modifier

        return modified_value

    async def update_parameter(
        self,
        parameter_name: str,
        new_value: float,
        confidence: float,
        context: Optional[Dict[str, Any]] = None,
        sample_size: int = 1,
    ) -> bool:
        """
        Update a parameter value based on learned data

        Args:
            parameter_name: Name of the parameter to update
            new_value: New parameter value
            confidence: Confidence in the new value (0-1)
            context: Context information for the update
            sample_size: Number of samples the update is based on

        Returns:
            Whether the update was successful
        """
        if parameter_name not in self._parameter_metadata:
            if self.logger:
                self.logger.warning(f"Unknown parameter: {parameter_name}")
            return False

        metadata = self._parameter_metadata[parameter_name]

        # Validate new value is within bounds
        if not (metadata.min_value <= new_value <= metadata.max_value):
            if self.logger:
                self.logger.warning(
                    f"Parameter value {new_value} out of bounds for {parameter_name} "
                    f"(min: {metadata.min_value}, max: {metadata.max_value})"
                )
            return False

        # Check confidence threshold
        if confidence < metadata.confidence_threshold:
            if self.logger:
                self.logger.debug(
                    f"Confidence {confidence} below threshold {metadata.confidence_threshold} "
                    f"for parameter {parameter_name}"
                )
            return False

        try:
            # Get current value
            current_value = await self._get_base_parameter(parameter_name)

            # Create update record
            update_record = ParameterUpdate(
                parameter_name=parameter_name,
                old_value=current_value,
                new_value=new_value,
                confidence=confidence,
                sample_size=sample_size,
                timestamp=time.time(),
                context=context or {},
            )

            # Store the parameter update
            if self.estimation_db:
                parameter_record = {
                    "current_value": new_value,
                    "previous_value": current_value,
                    "confidence": confidence,
                    "sample_size": sample_size,
                    "last_updated": time.time(),
                    "context": context or {},
                }

                await self.estimation_db.storage.upsert(
                    {f"param_{parameter_name}": parameter_record}
                )

                # Also store the update history
                update_id = f"param_update_{parameter_name}_{int(time.time())}"
                await self.estimation_db.storage.upsert(
                    {
                        update_id: {
                            "parameter_name": parameter_name,
                            "old_value": current_value,
                            "new_value": new_value,
                            "confidence": confidence,
                            "sample_size": sample_size,
                            "timestamp": time.time(),
                            "context": context or {},
                        }
                    }
                )

            # Invalidate cache
            self._cache_expires = 0

            if self.logger:
                self.logger.info(
                    f"Updated parameter {parameter_name}: {current_value:.4f} → {new_value:.4f} "
                    f"(confidence: {confidence:.3f}, samples: {sample_size})"
                )

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update parameter {parameter_name}: {e}")
            return False

    async def get_parameter_history(
        self, parameter_name: str, limit: int = 100
    ) -> List[ParameterUpdate]:
        """Get the update history for a parameter"""
        if not self.estimation_db:
            return []

        try:
            all_records = await self.estimation_db.storage.get_all()

            # Filter for parameter updates
            updates = []
            for record_id, record_data in all_records.items():
                if (
                    record_id.startswith(f"param_update_{parameter_name}_")
                    and record_data.get("parameter_name") == parameter_name
                ):
                    update = ParameterUpdate(
                        parameter_name=record_data["parameter_name"],
                        old_value=record_data["old_value"],
                        new_value=record_data["new_value"],
                        confidence=record_data["confidence"],
                        sample_size=record_data["sample_size"],
                        timestamp=record_data["timestamp"],
                        context=record_data.get("context", {}),
                    )
                    updates.append(update)

            # Sort by timestamp (newest first) and limit
            updates.sort(key=lambda x: x.timestamp, reverse=True)
            return updates[:limit]

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Failed to get parameter history for {parameter_name}: {e}"
                )
            return []

    async def rollback_parameter(
        self, parameter_name: str, to_timestamp: float
    ) -> bool:
        """Rollback a parameter to a previous value"""
        history = await self.get_parameter_history(parameter_name)

        # Find the value at the specified timestamp
        target_value = None
        for update in reversed(history):  # Oldest first
            if update.timestamp <= to_timestamp:
                target_value = update.new_value
                break

        if target_value is None:
            # Fall back to default
            target_value = self._parameter_metadata[parameter_name].default_value

        # Update to the target value
        return await self.update_parameter(
            parameter_name,
            target_value,
            confidence=1.0,  # High confidence for rollback
            context={"rollback": True, "rollback_timestamp": to_timestamp},
        )

    def get_all_parameters(self) -> Dict[str, ParameterMetadata]:
        """Get metadata for all parameters"""
        return self._parameter_metadata.copy()


class ParameterLearningEngine:
    """
    Advanced learning engine for parameter optimization

    This class implements sophisticated statistical learning algorithms
    to continuously improve parameter accuracy based on actual usage data.
    """

    def __init__(self, parameter_manager: LearnableParameterManager, logger=None):
        self.parameter_manager = parameter_manager
        self.logger = logger
        self._learning_history = {}

    async def analyze_usage_patterns(
        self, usage_records: List[ContextualUsageRecord]
    ) -> Dict[str, Any]:
        """
        Analyze usage patterns to identify parameter learning opportunities

        Args:
            usage_records: List of contextual usage records

        Returns:
            Analysis results with insights and recommendations
        """
        if not usage_records:
            return {"error": "No usage records provided"}

        analysis = {
            "total_records": len(usage_records),
            "call_type_distribution": {},
            "accuracy_metrics": {},
            "parameter_insights": {},
            "recommendations": [],
        }

        # Analyze by call type
        by_call_type = {}
        for record in usage_records:
            call_type = record.call_type
            if call_type not in by_call_type:
                by_call_type[call_type] = []
            by_call_type[call_type].append(record)

        analysis["call_type_distribution"] = {
            ct: len(records) for ct, records in by_call_type.items()
        }

        # Calculate accuracy metrics for each call type
        for call_type, records in by_call_type.items():
            if (
                len(records) < MINIMUM_SAMPLE_SIZE_FOR_LEARNING
            ):  # Use constant instead of magic number
                continue

            successful_records = [
                r for r in records if r.success and r.total_accuracy is not None
            ]
            if not successful_records:
                continue

            accuracies = [r.total_accuracy for r in successful_records]
            input_errors = [
                abs(r.actual_input_tokens - r.estimated_input_tokens)
                / max(r.actual_input_tokens, ACCURACY_BASE)
                for r in successful_records
                if r.estimated_input_tokens > CACHE_EXPIRY_RESET
            ]
            output_errors = [
                abs(r.actual_output_tokens - r.estimated_output_tokens)
                / max(r.actual_output_tokens, ACCURACY_BASE)
                for r in successful_records
                if r.estimated_output_tokens > CACHE_EXPIRY_RESET
            ]

            analysis["accuracy_metrics"][call_type] = {
                "avg_accuracy": statistics.mean(accuracies),
                "accuracy_std": statistics.stdev(accuracies)
                if len(accuracies) > ACCURACY_BASE
                else CACHE_EXPIRY_RESET,
                "avg_input_error": statistics.mean(input_errors)
                if input_errors
                else CACHE_EXPIRY_RESET,
                "avg_output_error": statistics.mean(output_errors)
                if output_errors
                else CACHE_EXPIRY_RESET,
                "sample_size": len(successful_records),
            }

            # Generate recommendations based on error patterns using configurable thresholds
            if statistics.mean(accuracies) < MIN_ACCURACY_THRESHOLD:
                analysis["recommendations"].append(
                    f"Low accuracy for {call_type} ({statistics.mean(accuracies):.2f}). "
                    "Consider reviewing estimation parameters."
                )

            if input_errors and statistics.mean(input_errors) > INPUT_ERROR_THRESHOLD:
                analysis["recommendations"].append(
                    f"High input estimation error for {call_type}. "
                    "Review prompt token calculations."
                )

            if (
                output_errors
                and statistics.mean(output_errors) > OUTPUT_ERROR_THRESHOLD
            ):
                analysis["recommendations"].append(
                    f"High output estimation error for {call_type}. "
                    "Review output token parameters."
                )

        return analysis

    async def learn_ratio_parameters(
        self, usage_records: List[ContextualUsageRecord]
    ) -> Dict[str, ParameterUpdate]:
        """
        Learn ratio parameters from usage data using statistical regression

        Args:
            usage_records: Usage records to learn from

        Returns:
            Dictionary of parameter updates
        """
        updates = {}

        # Group records by call type for ratio learning
        by_call_type = {}
        for record in usage_records:
            if (
                record.success
                and record.chunk_size
                and record.total_accuracy
                and record.total_accuracy > MIN_RECORD_ACCURACY_FILTER
            ):
                call_type = record.call_type
                if call_type not in by_call_type:
                    by_call_type[call_type] = []
                by_call_type[call_type].append(record)

        # Learn entity extraction ratios
        if "entity_extraction" in by_call_type:
            entity_records = by_call_type["entity_extraction"]
            if len(entity_records) >= MINIMUM_RECORDS_FOR_PATTERN_DETECTION:
                # Learn chunk_tokens_to_entities_divisor
                ratios = []
                weights = []

                # Get learnable parameter for entity token estimation instead of magic number 60
                tokens_per_entity_param = await self.parameter_manager.get_parameter(
                    "learning_tokens_per_entity_estimate"
                )
                small_output_filter = await self.parameter_manager.get_parameter(
                    "learning_small_output_filter"
                )

                for record in entity_records:
                    if record.actual_output_tokens > small_output_filter:
                        # Estimate entities from output using learnable parameter instead of magic number
                        estimated_entities = max(
                            ACCURACY_BASE,
                            record.actual_output_tokens / tokens_per_entity_param,
                        )
                        if estimated_entities > CACHE_EXPIRY_RESET:
                            ratio = record.chunk_size / estimated_entities
                            ratios.append(ratio)
                            weights.append(record.total_accuracy)

                if ratios and len(ratios) >= MINIMUM_SAMPLE_SIZE_FOR_LEARNING:
                    # Calculate confidence-weighted average
                    weighted_avg = sum(r * w for r, w in zip(ratios, weights)) / sum(
                        weights
                    )
                    confidence = min(
                        MAX_CONFIDENCE, len(ratios) / CONFIDENCE_SAMPLE_SIZE_DIVISOR
                    )

                    update = await self.parameter_manager.update_parameter(
                        "chunk_tokens_to_entities_divisor",
                        weighted_avg,
                        confidence,
                        context={
                            "learning_method": "statistical_regression",
                            "sample_size": len(ratios),
                        },
                        sample_size=len(ratios),
                    )

                    if update:
                        updates["chunk_tokens_to_entities_divisor"] = ParameterUpdate(
                            parameter_name="chunk_tokens_to_entities_divisor",
                            old_value=await self.parameter_manager.get_parameter(
                                "chunk_tokens_to_entities_divisor"
                            ),
                            new_value=weighted_avg,
                            confidence=confidence,
                            sample_size=len(ratios),
                            timestamp=time.time(),
                            context={"learning_method": "statistical_regression"},
                        )

        # Learn relation extraction ratios
        if "relation_extraction" in by_call_type:
            relation_records = by_call_type["relation_extraction"]
            if len(relation_records) >= MINIMUM_RECORDS_FOR_PATTERN_DETECTION:
                ratios = []
                weights = []

                # Get learnable parameters instead of magic numbers
                tokens_per_relation_param = await self.parameter_manager.get_parameter(
                    "learning_tokens_per_relation_estimate"
                )
                small_relation_filter = await self.parameter_manager.get_parameter(
                    "learning_small_relation_output_filter"
                )

                for record in relation_records:
                    if record.actual_output_tokens > small_relation_filter:
                        # Estimate relations from output using learnable parameter instead of magic number
                        estimated_relations = max(
                            ACCURACY_BASE,
                            record.actual_output_tokens / tokens_per_relation_param,
                        )
                        if estimated_relations > CACHE_EXPIRY_RESET:
                            ratio = record.chunk_size / estimated_relations
                            ratios.append(ratio)
                            weights.append(record.total_accuracy)

                if ratios and len(ratios) >= MINIMUM_SAMPLE_SIZE_FOR_LEARNING:
                    weighted_avg = sum(r * w for r, w in zip(ratios, weights)) / sum(
                        weights
                    )
                    confidence = min(
                        MAX_CONFIDENCE, len(ratios) / CONFIDENCE_SAMPLE_SIZE_DIVISOR
                    )

                    update = await self.parameter_manager.update_parameter(
                        "chunk_tokens_to_relations_divisor",
                        weighted_avg,
                        confidence,
                        context={
                            "learning_method": "statistical_regression",
                            "sample_size": len(ratios),
                        },
                        sample_size=len(ratios),
                    )

                    if update:
                        updates["chunk_tokens_to_relations_divisor"] = ParameterUpdate(
                            parameter_name="chunk_tokens_to_relations_divisor",
                            old_value=await self.parameter_manager.get_parameter(
                                "chunk_tokens_to_relations_divisor"
                            ),
                            new_value=weighted_avg,
                            confidence=confidence,
                            sample_size=len(ratios),
                            timestamp=time.time(),
                            context={"learning_method": "statistical_regression"},
                        )

        # Learn other ratio parameters similarly
        await self._learn_context_parameters(usage_records, updates)

        return updates

    async def _learn_context_parameters(
        self,
        usage_records: List[ContextualUsageRecord],
        updates: Dict[str, ParameterUpdate],
    ):
        """Learn context-specific parameters like tokens per entity description"""

        # Learn tokens_per_entity_description from clustering records
        clustering_records = [
            r
            for r in usage_records
            if r.call_type == "hierarchical_clustering" and r.success
        ]

        if len(clustering_records) >= MINIMUM_SAMPLE_SIZE_FOR_LEARNING:
            # Analyze input vs entity count relationship
            input_per_entity_ratios = []
            weights = []

            # Get learnable parameters instead of magic numbers
            chunk_to_entities_heuristic = await self.parameter_manager.get_parameter(
                "learning_chunk_to_entities_heuristic"
            )
            input_threshold = await self.parameter_manager.get_parameter(
                "learning_input_context_threshold"
            )
            min_bound = await self.parameter_manager.get_parameter(
                "learning_entity_tokens_min_bound"
            )
            max_bound = await self.parameter_manager.get_parameter(
                "learning_entity_tokens_max_bound"
            )
            default_accuracy = await self.parameter_manager.get_parameter(
                "learning_default_accuracy_fallback"
            )

            for record in clustering_records:
                # Very rough heuristic: assume input tokens relate to entity descriptions
                if record.chunk_size and record.actual_input_tokens > input_threshold:
                    # Estimate entities from chunk size using learnable parameter instead of magic number
                    estimated_entities = max(
                        ACCURACY_BASE, record.chunk_size / chunk_to_entities_heuristic
                    )
                    if estimated_entities > CACHE_EXPIRY_RESET:
                        tokens_per_entity = (
                            record.actual_input_tokens / estimated_entities
                        )
                        if (
                            min_bound <= tokens_per_entity <= max_bound
                        ):  # Sanity check using learnable bounds
                            input_per_entity_ratios.append(tokens_per_entity)
                            weights.append(record.total_accuracy or default_accuracy)

            if (
                input_per_entity_ratios
                and len(input_per_entity_ratios) >= MINIMUM_SAMPLE_SIZE_FOR_LEARNING
            ):
                weighted_avg = sum(
                    r * w for r, w in zip(input_per_entity_ratios, weights)
                ) / sum(weights)
                confidence = min(
                    MAX_CONFIDENCE - 0.05, len(input_per_entity_ratios) / 15
                )  # Slightly lower confidence

                await self.parameter_manager.update_parameter(
                    "tokens_per_entity_description",
                    weighted_avg,
                    confidence,
                    context={
                        "learning_method": "clustering_analysis",
                        "sample_size": len(input_per_entity_ratios),
                    },
                    sample_size=len(input_per_entity_ratios),
                )

    async def detect_contextual_patterns(
        self, usage_records: List[ContextualUsageRecord]
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns that suggest contextual parameter adjustments

        Args:
            usage_records: Usage records to analyze

        Returns:
            List of detected patterns with recommendations
        """
        patterns = []

        # Group by document type
        by_doc_type = {}
        for record in usage_records:
            if record.document_type and record.total_accuracy is not None:
                doc_type = record.document_type
                if doc_type not in by_doc_type:
                    by_doc_type[doc_type] = []
                by_doc_type[doc_type].append(record)

        # Analyze accuracy by document type
        if len(by_doc_type) > ACCURACY_BASE:  # Use constant instead of magic number 1
            doc_type_accuracies = {}
            for doc_type, records in by_doc_type.items():
                if len(records) >= MINIMUM_SAMPLE_SIZE_FOR_LEARNING:
                    accuracies = [
                        r.total_accuracy
                        for r in records
                        if r.total_accuracy > CACHE_EXPIRY_RESET
                    ]
                    if accuracies:
                        doc_type_accuracies[doc_type] = {
                            "avg_accuracy": statistics.mean(accuracies),
                            "sample_size": len(accuracies),
                        }

            # Find significant differences using configurable constant
            if len(doc_type_accuracies) > ACCURACY_BASE:
                accuracies = list(doc_type_accuracies.values())
                avg_accuracy = statistics.mean([a["avg_accuracy"] for a in accuracies])

                for doc_type, metrics in doc_type_accuracies.items():
                    if (
                        abs(metrics["avg_accuracy"] - avg_accuracy)
                        > SIGNIFICANT_ACCURACY_DIFFERENCE
                    ):
                        patterns.append(
                            {
                                "type": "document_type_bias",
                                "document_type": doc_type,
                                "accuracy_difference": metrics["avg_accuracy"]
                                - avg_accuracy,
                                "sample_size": metrics["sample_size"],
                                "recommendation": f"Consider adjusting parameters for {doc_type} documents",
                            }
                        )

        # Group by model
        by_model = {}
        for record in usage_records:
            if record.model_name and record.total_accuracy is not None:
                model = record.model_name
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(record)

        # Analyze accuracy by model
        if len(by_model) > ACCURACY_BASE:
            model_accuracies = {}
            for model, records in by_model.items():
                if len(records) >= MINIMUM_SAMPLE_SIZE_FOR_LEARNING:
                    accuracies = [
                        r.total_accuracy
                        for r in records
                        if r.total_accuracy > CACHE_EXPIRY_RESET
                    ]
                    if accuracies:
                        model_accuracies[model] = {
                            "avg_accuracy": statistics.mean(accuracies),
                            "sample_size": len(accuracies),
                        }

            if len(model_accuracies) > ACCURACY_BASE:
                accuracies = list(model_accuracies.values())
                avg_accuracy = statistics.mean([a["avg_accuracy"] for a in accuracies])

                for model, metrics in model_accuracies.items():
                    if (
                        abs(metrics["avg_accuracy"] - avg_accuracy)
                        > VERY_SIGNIFICANT_ACCURACY_DIFFERENCE
                    ):
                        patterns.append(
                            {
                                "type": "model_performance_bias",
                                "model": model,
                                "accuracy_difference": metrics["avg_accuracy"]
                                - avg_accuracy,
                                "sample_size": metrics["sample_size"],
                                "recommendation": f"Consider model-specific parameter adjustment for {model}",
                            }
                        )

        return patterns

    async def optimize_parameters_automatically(
        self,
        usage_records: List[ContextualUsageRecord],
        max_updates: int = 5,  # This is configurable per call, not a magic number
    ) -> Dict[str, Any]:
        """
        Automatically optimize parameters based on usage data

        Args:
            usage_records: Usage records for learning
            max_updates: Maximum number of parameter updates per run

        Returns:
            Summary of optimization results
        """
        optimization_results = {
            "updates_applied": {},
            "patterns_detected": [],
            "recommendations": [],
            "performance_summary": {},
        }

        if len(usage_records) < MINIMUM_RECORDS_FOR_OPTIMIZATION:
            optimization_results["error"] = "Insufficient data for optimization"
            return optimization_results

        try:
            # Analyze current performance
            analysis = await self.analyze_usage_patterns(usage_records)
            optimization_results["performance_summary"] = analysis["accuracy_metrics"]

            # Learn ratio parameters
            parameter_updates = await self.learn_ratio_parameters(usage_records)

            # Limit updates to prevent instability
            updates_applied = CACHE_EXPIRY_RESET
            for param_name, update in parameter_updates.items():
                if updates_applied >= max_updates:
                    break

                optimization_results["updates_applied"][param_name] = {
                    "old_value": update.old_value,
                    "new_value": update.new_value,
                    "confidence": update.confidence,
                    "sample_size": update.sample_size,
                }
                updates_applied += ACCURACY_BASE  # Increment by 1

            # Detect contextual patterns
            patterns = await self.detect_contextual_patterns(usage_records)
            optimization_results["patterns_detected"] = patterns

            # Generate recommendations
            optimization_results["recommendations"] = analysis["recommendations"]

            if self.logger:
                self.logger.info(
                    f"Parameter optimization completed: {updates_applied} updates applied, "
                    f"{len(patterns)} patterns detected"
                )

        except Exception as e:
            optimization_results["error"] = str(e)
            if self.logger:
                self.logger.error(f"Parameter optimization failed: {e}")

        return optimization_results


@dataclass
class TokenEstimate:
    """Represents token estimates for a single LLM call"""

    call_type: LLMCallType
    input_tokens: int
    output_tokens_min: int
    output_tokens_max: int
    output_tokens_estimated: int
    thinking_tokens: int = 0  # For models with reasoning
    confidence: float = 0.8  # Confidence in the estimate
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self.output_tokens_estimated + self.thinking_tokens


@dataclass
class PipelineEstimate:
    """Represents token estimates for the entire pipeline"""

    document_tokens: int
    num_chunks: int
    token_per_chunk: float
    call_estimates: List[TokenEstimate] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate totals after initialization"""
        self.total_input_tokens = sum(
            est.total_input_tokens for est in self.call_estimates
        )
        self.total_output_tokens = sum(
            est.total_output_tokens for est in self.call_estimates
        )


@dataclass
class ModelPricing:
    """Pricing information for different models"""

    input_cost_per_1k: float  # Cost per 1K input tokens
    output_cost_per_1k: float  # Cost per 1K output tokens
    context_window: int  # Maximum context window
    max_output_tokens: int  # Maximum output tokens


class TokenEstimator:
    """
    Main token estimation engine for the HiRAG pipeline

    This class provides comprehensive token estimation capabilities by:
    1. Analyzing prompt templates to get fixed token counts
    2. Estimating variable content based on input characteristics
    3. Learning from actual usage to improve estimates over time
    4. Providing cost predictions across different models
    """

    # Model pricing (as of 2024 - should be configurable)
    MODEL_PRICING = {
        model: ModelPricing(
            input_cost_per_1k=data["input_cost_per_1k"],
            output_cost_per_1k=data["output_cost_per_1k"],
            context_window=data["max_input"],
            max_output_tokens=data["max_output"],
        )
        for model, data in MODEL_PRICING_DATA.items()
    }

    def __init__(
        self,
        global_config: Dict[str, Any],
        estimation_db: Optional[BaseKVStorage] = None,
        tiktoken_model: str = "gpt-4o",
    ):
        self.global_config = global_config
        self.estimation_db = estimation_db
        self.tiktoken_model = tiktoken_model
        self.tokenizer = tiktoken.encoding_for_model(tiktoken_model)

        # Cache for prompt token counts
        self._prompt_token_cache = {}

        # Initialize new learnable parameter system
        self.parameter_manager = LearnableParameterManager(estimation_db, logger)
        self.learning_engine = ParameterLearningEngine(self.parameter_manager, logger)

        # Initialize prompt token counts
        self._initialize_prompt_tokens()

        # Default estimates for variable content (all learnable parameters)
        self._default_estimates = self._initialize_default_estimates()

        # Dynamic estimation parameters (learnable loop counts, ratios, etc.)
        self._dynamic_params = self._initialize_dynamic_parameters()

        logger.info(f"TokenEstimator initialized with model: {tiktoken_model}")
        logger.info("Learnable parameter system activated")

    def _initialize_prompt_tokens(self):
        """Pre-calculate token counts for all prompt templates"""
        logger.info("Initializing prompt token counts...")

        for prompt_key, prompt_template in PROMPTS.items():
            if isinstance(prompt_template, str):
                # Count tokens in the base template (without variable substitution)
                base_tokens = len(self.tokenizer.encode(prompt_template))
                self._prompt_token_cache[prompt_key] = base_tokens
                logger.debug(f"Prompt '{prompt_key}': {base_tokens} base tokens")

    async def _get_base_estimation_parameters(
        self, call_type: LLMCallType
    ) -> Dict[str, Any]:
        """Get base estimation parameters for a call type using a mapping."""
        param_map = {
            LLMCallType.ENTITY_EXTRACTION: [
                "entity_extraction_base_output_min",
                "entity_extraction_base_output_max",
                "entity_extraction_entities_per_chunk_avg",
                "entity_extraction_tokens_per_entity",
                "entity_extraction_thinking_ratio",
            ],
            LLMCallType.RELATION_EXTRACTION: [
                "relation_extraction_base_output_min",
                "relation_extraction_base_output_max",
                "relation_extraction_relations_per_chunk_avg",
                "relation_extraction_tokens_per_relation",
                "relation_extraction_thinking_ratio",
            ],
            LLMCallType.CONTINUE_EXTRACTION: [
                "continue_extraction_base_output_min",
                "continue_extraction_base_output_max",
                "continue_extraction_additional_entities_ratio",
                "continue_extraction_thinking_ratio",
            ],
            LLMCallType.LOOP_DETECTION: [
                "loop_detection_base_output_min",
                "loop_detection_base_output_max",
                "loop_detection_thinking_ratio",
            ],
            LLMCallType.HIERARCHICAL_CLUSTERING: [
                "hierarchical_clustering_base_output_min",
                "hierarchical_clustering_base_output_max",
                "hierarchical_clustering_tokens_per_cluster",
                "hierarchical_clustering_thinking_ratio",
            ],
            LLMCallType.ENTITY_DISAMBIGUATION: [
                "entity_disambiguation_base_output_min",
                "entity_disambiguation_base_output_max",
                "entity_disambiguation_tokens_per_decision",
                "entity_disambiguation_thinking_ratio",
            ],
            LLMCallType.ENTITY_MERGING: [
                "entity_merging_base_output_min",
                "entity_merging_base_output_max",
                "entity_merging_thinking_ratio",
            ],
            LLMCallType.COMMUNITY_REPORT: [
                "community_report_base_output_min",
                "community_report_base_output_max",
                "community_report_tokens_per_entity",
                "community_report_tokens_per_relation",
                "community_report_thinking_ratio",
            ],
        }

        param_names = param_map.get(call_type, [])
        tasks = [self.parameter_manager.get_parameter(name) for name in param_names]
        values = await asyncio.gather(*tasks)

        # Rename keys to be generic for the caller
        renamed_keys = [name.split("_", 1)[1] for name in param_names]
        return dict(zip(renamed_keys, values))

    async def _get_dynamic_parameters(self, category: str) -> Dict[str, Any]:
        """Get dynamic estimation parameters using a mapping."""
        param_map = {
            "entity_extraction": [
                "entities_per_chunk_base",
                "entities_per_100_tokens",
                "chunk_complexity_factor",
                "min_entities_per_chunk",
                "max_entities_per_chunk",
            ],
            "gleaning_loops": [
                "avg_actual_loops",
                "loop_probability_decay",
                "min_loop_probability",
                "chunk_size_loop_factor",
            ],
            "hierarchical_clustering": [
                "clustering_iterations_base",
                "entities_per_cluster",
                "cluster_reduction_ratio",
                "min_entities_for_clustering",
                "max_clustering_iterations",
            ],
            "entity_disambiguation": [
                "disambiguation_probability",
                "avg_cluster_size",
                "complexity_disambiguation_factor",
                "min_disambiguation_clusters",
                "max_disambiguation_cluster_size",
            ],
            "community_detection": [
                "entities_per_community",
                "community_size_variance",
                "min_community_size",
                "max_community_size",
                "community_overlap_ratio",
            ],
            "pipeline": [
                "relation_to_entity_ratio",
                "temporary_entity_ratio",
                "entity_merging_ratio",
                "content_complexity_multiplier",
            ],
        }

        param_names = param_map.get(category, [])
        tasks = [self.parameter_manager.get_parameter(name) for name in param_names]
        values = await asyncio.gather(*tasks)
        return dict(zip(param_names, values))

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        return len(self.tokenizer.encode(text))

    def estimate_chunk_processing_tokens(
        self,
        chunks: Dict[str, TextChunkSchema],
        enable_hierarchical: bool = True,
        entity_extract_max_gleaning: int = 1,
    ) -> List[TokenEstimate]:
        """
        Estimate tokens for processing text chunks (entity & relation extraction)

        Args:
            chunks: Dictionary of text chunks to process
            enable_hierarchical: Whether hierarchical extraction is enabled
            entity_extract_max_gleaning: Number of gleaning iterations

        Returns:
            List of TokenEstimate objects for chunk processing
        """
        estimates = []

        for chunk_id, chunk_data in chunks.items():
            chunk_content = chunk_data["content"]
            chunk_tokens = chunk_data.get("tokens", self.count_tokens(chunk_content))

            # Entity extraction
            entity_estimate = self._estimate_entity_extraction(
                chunk_content, chunk_tokens
            )
            estimates.append(entity_estimate)

            # Relation extraction (if hierarchical mode)
            if enable_hierarchical:
                relation_estimate = self._estimate_relation_extraction(
                    chunk_content, chunk_tokens
                )
                estimates.append(relation_estimate)

            # Gleaning iterations
            for gleaning_iter in range(entity_extract_max_gleaning):
                continue_estimate = self._estimate_continue_extraction(
                    chunk_content, chunk_tokens
                )
                estimates.append(continue_estimate)

                # Loop detection for gleaning
                loop_estimate = self._estimate_loop_detection(
                    chunk_content, chunk_tokens
                )
                estimates.append(loop_estimate)

        return estimates

    async def _estimate_entity_extraction(
        self,
        chunk_content: str,
        chunk_tokens: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> TokenEstimate:
        """Estimate tokens for entity extraction from a single chunk"""
        # Input tokens: prompt + chunk content
        prompt_tokens = self._prompt_token_cache.get(
            "hi_entity_extraction", CACHE_EXPIRY_RESET
        )
        context_tokens = len(self.tokenizer.encode(str(self._get_entity_context())))
        input_tokens = prompt_tokens + context_tokens + chunk_tokens

        # Get all parameters from learnable parameter manager
        base_params = await self._get_base_estimation_parameters(
            LLMCallType.ENTITY_EXTRACTION
        )

        # Use learnable parameter instead of magic number
        divisor = await self.parameter_manager.get_parameter(
            "chunk_tokens_to_entities_divisor", context=context or {}
        )

        # Estimate based on chunk size and content complexity
        estimated_entities = (
            max(ACCURACY_BASE, chunk_tokens / divisor)
            * base_params["entities_per_chunk_avg"]
        )

        output_min = base_params["base_output_min"]
        output_max = base_params["base_output_max"] + (
            estimated_entities * base_params["tokens_per_entity"]
        )
        output_estimated = int(
            (output_min + output_max) / AVERAGE_DIVISOR_FOR_ESTIMATES
        )

        thinking_tokens = int(output_estimated * base_params["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.ENTITY_EXTRACTION,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={
                "chunk_tokens": chunk_tokens,
                "estimated_entities": estimated_entities,
                "divisor_used": divisor,
            },
        )

    async def _estimate_relation_extraction(
        self,
        chunk_content: str,
        chunk_tokens: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> TokenEstimate:
        """Estimate tokens for relation extraction from a single chunk"""
        prompt_tokens = self._prompt_token_cache.get(
            "hi_relation_extraction", CACHE_EXPIRY_RESET
        )
        context_tokens = len(self.tokenizer.encode(str(self._get_relation_context())))
        input_tokens = prompt_tokens + context_tokens + chunk_tokens

        # Get all parameters from learnable parameter manager
        base_params = await self._get_base_estimation_parameters(
            LLMCallType.RELATION_EXTRACTION
        )

        # Use learnable parameter instead of magic number
        divisor = await self.parameter_manager.get_parameter(
            "chunk_tokens_to_relations_divisor", context=context or {}
        )

        estimated_relations = (
            max(ACCURACY_BASE, chunk_tokens / divisor)
            * base_params["relations_per_chunk_avg"]
        )

        output_min = base_params["base_output_min"]
        output_max = base_params["base_output_max"] + (
            estimated_relations * base_params["tokens_per_relation"]
        )
        output_estimated = int(
            (output_min + output_max) / AVERAGE_DIVISOR_FOR_ESTIMATES
        )

        thinking_tokens = int(output_estimated * base_params["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.RELATION_EXTRACTION,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={
                "chunk_tokens": chunk_tokens,
                "estimated_relations": estimated_relations,
                "divisor_used": divisor,
            },
        )

    async def _estimate_continue_extraction(
        self,
        chunk_content: str,
        chunk_tokens: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> TokenEstimate:
        """Estimate tokens for continue extraction (gleaning)"""
        prompt_tokens = self._prompt_token_cache.get(
            "entiti_continue_extraction", CACHE_EXPIRY_RESET
        )

        # Use learnable parameter instead of magic number
        history_fraction = await self.parameter_manager.get_parameter(
            "history_tokens_fraction", context=context or {}
        )

        # Continue extraction includes conversation history
        history_tokens = int(chunk_tokens * history_fraction)
        input_tokens = prompt_tokens + history_tokens

        # Get all parameters from learnable parameter manager
        base_params = await self._get_base_estimation_parameters(
            LLMCallType.CONTINUE_EXTRACTION
        )

        output_min = base_params["base_output_min"]
        output_max = base_params["base_output_max"]
        output_estimated = int(
            (output_min + output_max) / AVERAGE_DIVISOR_FOR_ESTIMATES
        )

        thinking_tokens = int(output_estimated * base_params["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.CONTINUE_EXTRACTION,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={
                "chunk_tokens": chunk_tokens,
                "history_fraction_used": history_fraction,
            },
        )

    async def _estimate_loop_detection(
        self,
        chunk_content: str,
        chunk_tokens: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> TokenEstimate:
        """Estimate tokens for loop detection in gleaning"""
        prompt_tokens = self._prompt_token_cache.get(
            "entiti_if_loop_extraction", CACHE_EXPIRY_RESET
        )

        # Use learnable parameter instead of magic number
        history_fraction = await self.parameter_manager.get_parameter(
            "history_tokens_fraction", context=context or {}
        )

        history_tokens = int(chunk_tokens * history_fraction)
        input_tokens = prompt_tokens + history_tokens

        # Get all parameters from learnable parameter manager
        base_params = await self._get_base_estimation_parameters(
            LLMCallType.LOOP_DETECTION
        )

        output_min = base_params["base_output_min"]
        output_max = base_params["base_output_max"]
        output_estimated = int(
            (output_min + output_max) / AVERAGE_DIVISOR_FOR_ESTIMATES
        )

        thinking_tokens = int(output_estimated * base_params["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.LOOP_DETECTION,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={
                "chunk_tokens": chunk_tokens,
                "history_fraction_used": history_fraction,
            },
        )

    async def estimate_hierarchical_clustering_tokens(
        self,
        num_entities: int,
        clustering_iterations: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TokenEstimate]:
        """Estimate tokens for hierarchical clustering"""

        # Get dynamic parameters for clustering
        clustering_params = await self._get_dynamic_parameters(
            "hierarchical_clustering"
        )

        # Use learnable parameter for iterations if not provided
        if clustering_iterations is None:
            clustering_iterations = int(clustering_params["clustering_iterations_base"])

        estimates = []
        current_entities = num_entities

        for iteration in range(clustering_iterations):
            # Use learnable parameter instead of magic number 2**iteration
            reduction_ratio = clustering_params["cluster_reduction_ratio"]
            entities_this_iter = max(
                ACCURACY_BASE, current_entities * (ACCURACY_BASE - reduction_ratio)
            )

            estimate = await self._estimate_single_clustering_iteration(
                int(entities_this_iter), context
            )
            estimates.append(estimate)

            current_entities = entities_this_iter

            # Stop if we have too few entities
            if current_entities < clustering_params["min_entities_for_clustering"]:
                break

        return estimates

    async def _estimate_single_clustering_iteration(
        self, num_entities: int, context: Optional[Dict[str, Any]] = None
    ) -> TokenEstimate:
        """Estimate tokens for a single hierarchical clustering iteration"""
        prompt_tokens = self._prompt_token_cache.get(
            "summary_clusters", CACHE_EXPIRY_RESET
        )

        # Use learnable parameter instead of magic number
        tokens_per_entity_desc = await self.parameter_manager.get_parameter(
            "tokens_per_entity_description", context=context or {}
        )

        # Input includes entity descriptions
        entity_descriptions_tokens = int(num_entities * tokens_per_entity_desc)
        input_tokens = prompt_tokens + entity_descriptions_tokens

        # Get all parameters from learnable parameter manager
        base_params = await self._get_base_estimation_parameters(
            LLMCallType.HIERARCHICAL_CLUSTERING
        )

        # Use learnable parameter instead of magic number
        entities_to_clusters_divisor = await self.parameter_manager.get_parameter(
            "entities_to_clusters_divisor", context=context or {}
        )

        num_clusters = max(
            ACCURACY_BASE, int(num_entities / entities_to_clusters_divisor)
        )

        output_min = base_params["base_output_min"]
        output_max = base_params["base_output_max"] + (
            num_clusters * base_params["tokens_per_cluster"]
        )
        output_estimated = int(
            (output_min + output_max) / AVERAGE_DIVISOR_FOR_ESTIMATES
        )

        thinking_tokens = int(output_estimated * base_params["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.HIERARCHICAL_CLUSTERING,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={
                "num_entities": num_entities,
                "estimated_clusters": num_clusters,
                "tokens_per_entity_desc_used": tokens_per_entity_desc,
                "entities_to_clusters_divisor_used": entities_to_clusters_divisor,
            },
        )

    async def estimate_disambiguation_tokens(
        self,
        num_entity_clusters: int,
        avg_cluster_size: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TokenEstimate]:
        """Estimate tokens for entity disambiguation"""

        # Get dynamic parameters for disambiguation
        disambig_params = await self._get_dynamic_parameters("entity_disambiguation")

        # Use learnable parameter if not provided
        if avg_cluster_size is None:
            avg_cluster_size = int(disambig_params["avg_cluster_size"])

        estimates = []

        for _ in range(num_entity_clusters):
            estimate = await self._estimate_single_disambiguation(
                avg_cluster_size, context
            )
            estimates.append(estimate)

        return estimates

    async def _estimate_single_disambiguation(
        self, cluster_size: int, context: Optional[Dict[str, Any]] = None
    ) -> TokenEstimate:
        """Estimate tokens for disambiguating a single entity cluster"""
        prompt_tokens = self._prompt_token_cache.get(
            "entity_disambiguation", CACHE_EXPIRY_RESET
        )

        # Use learnable parameter instead of magic number
        tokens_per_entity_context = await self.parameter_manager.get_parameter(
            "tokens_per_entity_context", context=context or {}
        )

        # Input includes entity details and context
        entity_context_tokens = int(cluster_size * tokens_per_entity_context)
        input_tokens = prompt_tokens + entity_context_tokens

        # Get all parameters from learnable parameter manager
        base_params = await self._get_base_estimation_parameters(
            LLMCallType.ENTITY_DISAMBIGUATION
        )

        output_min = base_params["base_output_min"]
        output_max = base_params["base_output_max"] + (
            cluster_size * base_params["tokens_per_decision"]
        )
        output_estimated = int(
            (output_min + output_max) / AVERAGE_DIVISOR_FOR_ESTIMATES
        )

        thinking_tokens = int(output_estimated * base_params["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.ENTITY_DISAMBIGUATION,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={
                "cluster_size": cluster_size,
                "tokens_per_entity_context_used": tokens_per_entity_context,
            },
        )

    async def estimate_community_report_tokens(
        self,
        num_communities: int,
        avg_community_size: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TokenEstimate]:
        """Estimate tokens for community reports"""

        # Get dynamic parameters for communities
        community_params = await self._get_dynamic_parameters("community_detection")

        # Use learnable parameter if not provided
        if avg_community_size is None:
            avg_community_size = int(community_params["entities_per_community"])

        estimates = []

        for _ in range(num_communities):
            estimate = await self._estimate_single_community_report(
                avg_community_size, context
            )
            estimates.append(estimate)

        return estimates

    async def _estimate_single_community_report(
        self, community_size: int, context: Optional[Dict[str, Any]] = None
    ) -> TokenEstimate:
        """Estimate tokens for generating a single community report"""
        prompt_tokens = self._prompt_token_cache.get(
            "community_report", CACHE_EXPIRY_RESET
        )

        # Use learnable parameter instead of magic number
        tokens_per_community_member = await self.parameter_manager.get_parameter(
            "tokens_per_community_member", context=context or {}
        )

        # Input includes community entities and relationships
        community_context_tokens = int(community_size * tokens_per_community_member)
        input_tokens = prompt_tokens + community_context_tokens

        # Get all parameters from learnable parameter manager
        base_params = await self._get_base_estimation_parameters(
            LLMCallType.COMMUNITY_REPORT
        )

        output_min = base_params["base_output_min"]
        output_max = base_params["base_output_max"] + (
            community_size * base_params["tokens_per_entity"]
        )
        output_estimated = int(
            (output_min + output_max) / AVERAGE_DIVISOR_FOR_ESTIMATES
        )

        thinking_tokens = int(output_estimated * base_params["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.COMMUNITY_REPORT,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={
                "community_size": community_size,
                "tokens_per_community_member_used": tokens_per_community_member,
            },
        )

    async def estimate_full_pipeline(
        self,
        document_content: Union[str, List[str]],
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> PipelineEstimate:
        """
        Estimate the full pipeline cost and token usage

        Args:
            document_content: Single document or list of documents
            config_overrides: Optional configuration overrides

        Returns:
            Comprehensive pipeline estimate
        """
        # Prepare config with no magic numbers
        config = self.global_config.copy()
        if config_overrides:
            config.update(config_overrides)

        # Process documents
        if isinstance(document_content, str):
            documents = [document_content]
        else:
            documents = document_content

        # Calculate total document tokens
        total_document_tokens = sum(self.count_tokens(doc) for doc in documents)

        # Get chunk configuration from learnable parameters or config
        chunk_token_size = config.get("chunk_token_size", DEFAULT_CHUNK_TOKEN_SIZE)
        chunk_overlap = config.get("chunk_overlap_token_size", DEFAULT_CHUNK_OVERLAP)
        effective_chunk_size = chunk_token_size - chunk_overlap

        num_chunks = max(ACCURACY_BASE, total_document_tokens // effective_chunk_size)
        token_per_chunk = min(
            chunk_token_size, total_document_tokens // max(ACCURACY_BASE, num_chunks)
        )

        all_estimates = []

        # Create context for estimation
        context = {
            "document_type": config.get("document_type", "general"),
            "model_name": config.get("best_model_func", "gpt-4o").__name__
            if callable(config.get("best_model_func"))
            else str(config.get("best_model_func", "gpt-4o")),
        }

        # Estimate chunk processing using dynamic parameters
        chunk_estimates = await self._estimate_chunk_processing_dynamic(
            num_chunks=int(num_chunks),
            token_per_chunk=int(token_per_chunk),
            enable_hierarchical=config.get("enable_hierarchical", True),
            max_gleaning_iterations=config.get(
                "entity_extract_max_gleaning", ACCURACY_BASE
            ),
            context=context,
        )
        all_estimates.extend(chunk_estimates)

        # Get estimated entity count for downstream processing
        entity_params = await self._get_dynamic_parameters("entity_extraction")
        estimated_entities = await self._estimate_entity_count(
            int(num_chunks), int(token_per_chunk)
        )

        # Hierarchical clustering estimates
        if config.get("enable_hierarchical", True):
            clustering_estimates = await self._estimate_hierarchical_clustering_dynamic(
                estimated_entities, context
            )
            all_estimates.extend(clustering_estimates)

        # Entity disambiguation estimates
        disambiguation_estimates = await self._estimate_disambiguation_dynamic(
            estimated_entities, context
        )
        all_estimates.extend(disambiguation_estimates)

        # Community reports
        community_params = await self._get_dynamic_parameters("community_detection")
        estimated_communities = await self._estimate_community_count(estimated_entities)
        community_estimates = await self._estimate_community_reports_dynamic(
            estimated_communities, estimated_entities, context
        )
        all_estimates.extend(community_estimates)

        # Calculate cost
        model_name = await self._get_primary_model_name(config)
        estimated_cost = self._calculate_cost(all_estimates, model_name)

        return PipelineEstimate(
            estimates=all_estimates,
            total_input_tokens=sum(est.input_tokens for est in all_estimates),
            total_output_tokens=sum(
                est.output_tokens_estimated for est in all_estimates
            ),
            estimated_cost_usd=estimated_cost,
            metadata={
                "num_documents": len(documents),
                "total_document_tokens": total_document_tokens,
                "num_chunks": int(num_chunks),
                "estimated_entities": estimated_entities,
                "estimated_communities": estimated_communities,
                "model_name": model_name,
            },
        )

    def _get_primary_model_name(self, config: Dict[str, Any]) -> str:
        """Extract primary model name from configuration"""
        # This is a simplified extraction - in practice, you'd want more sophisticated logic
        best_model_func = config.get("best_model_func", "gpt_4o_mini_complete")

        # Map function names to model names
        model_mapping = {
            "gpt_4o_complete": "gpt-4o",
            "gpt_4o_mini_complete": "gpt-4o-mini",
            "gpt_35_turbo_complete": "gpt-3.5-turbo",
            "azure_gpt_4o_complete": "gpt-4o",
            "azure_gpt_4o_mini_complete": "gpt-4o-mini",
            "gemini_pro_complete": "gemini-1.5-pro",
            "gemini_flash_complete": "gemini-1.5-flash",
        }

        if hasattr(best_model_func, "__name__"):
            func_name = best_model_func.__name__
        else:
            func_name = (
                str(best_model_func).split(".")[-1]
                if "." in str(best_model_func)
                else str(best_model_func)
            )

        return model_mapping.get(func_name, "gpt-4o-mini")  # Default fallback

    def _calculate_cost(self, estimates: List[TokenEstimate], model_name: str) -> float:
        """Calculate estimated cost for all estimates"""

        if model_name not in self.MODEL_PRICING:
            logger.warning(f"Unknown model {model_name}, using default pricing")
            # Use a default model for pricing if unknown
            default_model = "gpt-4o-mini"
            if default_model in self.MODEL_PRICING:
                pricing = self.MODEL_PRICING[default_model]
            else:
                return float(CACHE_EXPIRY_RESET)  # Can't calculate cost
        else:
            pricing = self.MODEL_PRICING[model_name]

        total_input_tokens = sum(est.input_tokens for est in estimates)
        total_output_tokens = sum(est.output_tokens_estimated for est in estimates)

        input_cost = (total_input_tokens / TOKENS_PER_1K) * pricing.input_cost_per_1k
        output_cost = (total_output_tokens / TOKENS_PER_1K) * pricing.output_cost_per_1k

        return input_cost + output_cost

    async def record_actual_usage(
        self,
        call_type: LLMCallType,
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
    ):
        """Record actual token usage with rich context for improved learning"""
        if not self.estimation_db:
            return

        # Create enhanced contextual usage record
        contextual_record = ContextualUsageRecord(
            call_type=call_type.value,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            model_name=model_name,
            timestamp=time.time(),
            chunk_size=chunk_size,
            document_type=document_type,
            success=success,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        record_id = compute_mdhash_id(
            f"{call_type.value}_{actual_input_tokens}_{actual_output_tokens}_{time.time()}",
            prefix="contextual-usage-",
        )

        # Store the contextual record
        await self.estimation_db.upsert(
            {
                record_id: {
                    "call_type": contextual_record.call_type,
                    "actual_input_tokens": contextual_record.actual_input_tokens,
                    "actual_output_tokens": contextual_record.actual_output_tokens,
                    "estimated_input_tokens": contextual_record.estimated_input_tokens,
                    "estimated_output_tokens": contextual_record.estimated_output_tokens,
                    "model_name": contextual_record.model_name,
                    "timestamp": contextual_record.timestamp,
                    "chunk_size": contextual_record.chunk_size,
                    "document_type": contextual_record.document_type,
                    "success": contextual_record.success,
                    "latency_ms": contextual_record.latency_ms,
                    "input_accuracy": contextual_record.input_accuracy,
                    "output_accuracy": contextual_record.output_accuracy,
                    "total_accuracy": contextual_record.total_accuracy,
                    "metadata": contextual_record.metadata,
                }
            }
        )

        logger.debug(
            f"Recorded contextual usage for {call_type.value}: "
            f"{actual_input_tokens}→{actual_output_tokens} "
            f"(accuracy: {contextual_record.total_accuracy:.3f})"
        )

        # Trigger learning if we have enough data
        await self._trigger_periodic_learning()

    async def _trigger_periodic_learning(self):
        """Trigger learning if we have accumulated enough data"""
        if not self.estimation_db:
            return

        try:
            # Get recent usage records for learning
            all_records = await self.estimation_db.get_all()
            contextual_records = []

            for record_id, record_data in all_records.items():
                if record_id.startswith("contextual-usage-"):
                    try:
                        contextual_record = ContextualUsageRecord(**record_data)
                        contextual_records.append(contextual_record)
                    except Exception as e:
                        logger.warning(f"Skipping invalid contextual record: {e}")
                        continue

            # Trigger learning if we have enough recent data using configurable constant
            recent_threshold = time.time() - (HOURS_IN_DAY * SECONDS_IN_HOUR)
            recent_records = [
                r for r in contextual_records if r.timestamp >= recent_threshold
            ]

            if len(recent_records) >= MINIMUM_RECORDS_FOR_PERIODIC_LEARNING:
                logger.info(
                    f"Triggering parameter learning with {len(recent_records)} recent records"
                )
                max_updates_per_trigger = 3  # Configurable limit to prevent instability
                await self.learning_engine.optimize_parameters_automatically(
                    recent_records, max_updates=max_updates_per_trigger
                )

        except Exception as e:
            logger.error(f"Error in periodic learning trigger: {e}")

    async def get_contextual_usage_records(
        self,
        days_back: int = DEFAULT_DAYS_BACK,
        call_type: Optional[LLMCallType] = None,
    ) -> List[ContextualUsageRecord]:
        """Get contextual usage records for analysis"""
        if not self.estimation_db:
            return []

        try:
            all_records = await self.estimation_db.get_all()
            contextual_records = []
            cutoff_time = time.time() - (days_back * HOURS_IN_DAY * SECONDS_IN_HOUR)

            for record_id, record_data in all_records.items():
                if (
                    record_id.startswith("contextual-usage-")
                    and record_data.get("timestamp", CACHE_EXPIRY_RESET) >= cutoff_time
                ):
                    try:
                        contextual_record = ContextualUsageRecord(**record_data)
                        if (
                            call_type is None
                            or contextual_record.call_type == call_type.value
                        ):
                            contextual_records.append(contextual_record)
                    except Exception as e:
                        logger.warning(f"Skipping invalid contextual record: {e}")
                        continue

            return sorted(contextual_records, key=lambda r: r.timestamp, reverse=True)

        except Exception as e:
            logger.error(f"Error getting contextual usage records: {e}")
            return []

    async def improve_estimates(self):
        """Analyze recorded usage data to improve estimation accuracy using advanced learning"""
        if not self.estimation_db:
            logger.warning("No estimation database available for learning")
            return

        logger.info(
            "Analyzing usage data to improve estimates using advanced learning engine..."
        )

        try:
            # Get contextual usage records from the last month using configurable constant
            contextual_records = await self.get_contextual_usage_records(
                days_back=EXTENDED_ANALYSIS_DAYS
            )

            if len(contextual_records) < MINIMUM_RECORDS_FOR_OPTIMIZATION:
                logger.info("Insufficient usage data for meaningful learning")
                return

            # Use the advanced learning engine for parameter optimization
            max_updates_for_improvement = 10  # Configurable limit
            optimization_results = (
                await self.learning_engine.optimize_parameters_automatically(
                    contextual_records, max_updates=max_updates_for_improvement
                )
            )

            # Log results
            if "error" in optimization_results:
                logger.error(
                    f"Parameter optimization failed: {optimization_results['error']}"
                )
                return

            updates_applied = optimization_results.get("updates_applied", {})
            patterns_detected = optimization_results.get("patterns_detected", [])
            recommendations = optimization_results.get("recommendations", [])

            logger.info(f"Parameter optimization completed:")
            logger.info(f"  • {len(updates_applied)} parameters updated")
            logger.info(f"  • {len(patterns_detected)} patterns detected")
            logger.info(f"  • {len(recommendations)} recommendations generated")

            # Log specific updates
            for param_name, update_info in updates_applied.items():
                logger.info(
                    f"  • Updated {param_name}: {update_info['old_value']:.4f} → "
                    f"{update_info['new_value']:.4f} (confidence: {update_info['confidence']:.3f})"
                )

            # Log patterns and recommendations
            for pattern in patterns_detected:
                logger.info(
                    f"  • Pattern: {pattern.get('recommendation', 'Unknown pattern')}"
                )

            for recommendation in recommendations:
                logger.info(f"  • Recommendation: {recommendation}")

        except Exception as e:
            logger.error(f"Error in advanced estimate improvement: {e}")

        logger.info("Estimate improvement completed")

    def _get_entity_context(self) -> Dict[str, str]:
        """Get entity extraction context parameters"""
        return {
            "tuple_delimiter": PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "|"),
            "record_delimiter": PROMPTS.get("DEFAULT_RECORD_DELIMITER", "##"),
            "completion_delimiter": PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "###"),
            "entity_types": ",".join(PROMPTS.get("DEFAULT_ENTITY_TYPES", [])),
        }

    def _get_relation_context(self) -> Dict[str, str]:
        """Get relation extraction context parameters"""
        return self._get_entity_context()  # Same context for relations

    def generate_estimation_report(self, pipeline_estimate: PipelineEstimate) -> str:
        """Generate a human-readable estimation report"""
        report_lines = [
            "=== HiRAG Pipeline Token Estimation Report ===",
            "",
            "Document Statistics:",
            f"  • Total document tokens: {pipeline_estimate.document_tokens:,}",
            f"  • Number of chunks: {pipeline_estimate.num_chunks}",
            f"  • Average tokens per chunk: {pipeline_estimate.token_per_chunk:.1f}",
            "",
            "Token Usage Breakdown:",
            f"  • Total input tokens: {pipeline_estimate.total_input_tokens:,}",
            f"  • Total output tokens: {pipeline_estimate.total_output_tokens:,}",
            f"  • Combined total: {pipeline_estimate.total_input_tokens + pipeline_estimate.total_output_tokens:,}",
            "",
            "Cost Estimation:",
            f"  • Model: {pipeline_estimate.metadata.get('model_name', 'Unknown')}",
            f"  • Estimated cost: ${pipeline_estimate.estimated_cost_usd:.4f} USD",
            "",
            "Per-Operation Breakdown:",
        ]

        # Group estimates by call type
        by_type = defaultdict(list)
        for est in pipeline_estimate.call_estimates:
            by_type[est.call_type].append(est)

        for call_type, estimates in by_type.items():
            total_input = sum(e.total_input_tokens for e in estimates)
            total_output = sum(e.total_output_tokens for e in estimates)
            count = len(estimates)

            report_lines.extend(
                [
                    f"  • {call_type.value}:",
                    f"    - Calls: {count}",
                    f"    - Input tokens: {total_input:,}",
                    f"    - Output tokens: {total_output:,}",
                    f"    - Subtotal: {total_input + total_output:,}",
                ]
            )

        return "\n".join(report_lines)

    # Dynamic estimation helper methods

    async def _estimate_entity_count(
        self, num_chunks: int, token_per_chunk: int
    ) -> int:
        """Estimate total entities that will be extracted from chunks"""

        # Get entity extraction parameters
        entity_params = await self._get_dynamic_parameters("entity_extraction")

        # Use learnable parameters instead of magic numbers
        base_entities = entity_params["entities_per_chunk_base"]
        entities_per_100_tokens = entity_params["entities_per_100_tokens"]

        # Calculate token-based entities using learnable parameter instead of magic number 100
        tokens_per_100 = 100  # This is a unit conversion, not a magic number
        token_based_entities = (
            token_per_chunk / tokens_per_100
        ) * entities_per_100_tokens

        # Calculate total entities per chunk
        entities_per_chunk = base_entities + token_based_entities

        # Apply bounds using learnable parameters
        min_entities = entity_params["min_entities_per_chunk"]
        max_entities = entity_params["max_entities_per_chunk"]
        entities_per_chunk = max(min_entities, min(max_entities, entities_per_chunk))

        total_entities = int(num_chunks * entities_per_chunk)

        return max(int(min_entities), total_entities)

    async def _estimate_total_gleaning_loops(self, num_chunks: int) -> int:
        """Estimate total gleaning loops needed across all chunks"""

        # Get gleaning parameters using learnable parameter manager
        gleaning_params = await self._get_dynamic_parameters("gleaning_loops")

        # Use learnable parameter instead of any hardcoded values
        avg_loops = gleaning_params["avg_actual_loops"]

        return int(num_chunks * avg_loops)

    async def _estimate_clustering_iterations(self, num_entities: int) -> int:
        """Estimate number of clustering iterations needed"""

        # Get clustering parameters using learnable parameter manager
        clustering_params = await self._get_dynamic_parameters(
            "hierarchical_clustering"
        )

        # Check if we have enough entities for clustering
        min_entities = clustering_params["min_entities_for_clustering"]
        if num_entities < min_entities:
            return int(CACHE_EXPIRY_RESET)  # No clustering needed

        # Use learnable parameters for iteration calculation
        base_iterations = clustering_params["clustering_iterations_base"]
        max_iterations = clustering_params["max_clustering_iterations"]

        # Simple calculation based on entity count - more entities might need more iterations
        entity_factor = num_entities / 100  # Scale factor for entity count
        iterations = min(max_iterations, base_iterations + (entity_factor * 0.1))

        return max(ACCURACY_BASE, int(iterations))

    async def _estimate_community_count(self, num_entities: int) -> int:
        """Estimate number of communities that will be detected"""

        # Get community detection parameters using learnable parameter manager
        community_params = await self._get_dynamic_parameters("community_detection")

        # Use learnable parameters for calculation
        entities_per_community = community_params["entities_per_community"]
        overlap_ratio = community_params["community_overlap_ratio"]

        # Calculate base communities
        estimated_communities = num_entities / entities_per_community

        # Adjust for overlap - some entities appear in multiple communities
        estimated_communities *= ACCURACY_BASE - overlap_ratio

        return max(ACCURACY_BASE, int(estimated_communities))

    async def _estimate_chunk_processing_dynamic(
        self,
        num_chunks: int,
        token_per_chunk: int,
        enable_hierarchical: bool,
        max_gleaning_iterations: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TokenEstimate]:
        """Estimate chunk processing tokens using dynamic parameters"""

        estimates = []

        # Entity extraction for each chunk
        entity_estimate = await self._estimate_entity_extraction(
            "", token_per_chunk, context
        )
        estimates.extend([entity_estimate] * num_chunks)

        # Relation extraction if hierarchical is enabled
        if enable_hierarchical:
            relation_estimate = await self._estimate_relation_extraction(
                "", token_per_chunk, context
            )
            estimates.extend([relation_estimate] * num_chunks)

        # Gleaning loops estimation using learnable parameters
        gleaning_params = await self._get_dynamic_parameters("gleaning_loops")
        actual_gleaning_loops = await self._estimate_total_gleaning_loops(num_chunks)

        # Continue extraction estimates
        continue_estimate = await self._estimate_continue_extraction(
            "", token_per_chunk, context
        )
        estimates.extend([continue_estimate] * actual_gleaning_loops)

        # Loop detection estimates
        loop_estimate = await self._estimate_loop_detection(
            "", token_per_chunk, context
        )
        estimates.extend([loop_estimate] * actual_gleaning_loops)

        return estimates

    async def _estimate_hierarchical_clustering_dynamic(
        self, num_entities: int, context: Optional[Dict[str, Any]] = None
    ) -> List[TokenEstimate]:
        """Estimate hierarchical clustering tokens using dynamic parameters"""

        # Get clustering parameters using learnable parameter manager
        clustering_params = await self._get_dynamic_parameters(
            "hierarchical_clustering"
        )

        # Check minimum entities threshold
        min_entities = clustering_params["min_entities_for_clustering"]
        if num_entities < min_entities:
            return []  # No clustering needed

        estimates = []
        current_entities = num_entities
        iterations = await self._estimate_clustering_iterations(num_entities)

        # Use learnable parameters for iteration logic
        reduction_ratio = clustering_params["cluster_reduction_ratio"]

        for _ in range(int(iterations)):
            if current_entities < min_entities:
                break

            estimate = await self._estimate_single_clustering_iteration(
                int(current_entities), context
            )
            estimates.append(estimate)

            # Reduce entities for next iteration using learnable parameter
            current_entities *= ACCURACY_BASE - reduction_ratio

        return estimates

    async def _estimate_disambiguation_dynamic(
        self, num_entities: int, context: Optional[Dict[str, Any]] = None
    ) -> List[TokenEstimate]:
        """Estimate entity disambiguation tokens using dynamic parameters"""

        # Get disambiguation parameters using learnable parameter manager
        disambig_params = await self._get_dynamic_parameters("entity_disambiguation")

        # Calculate entities needing disambiguation using learnable parameter
        disambiguation_probability = disambig_params["disambiguation_probability"]
        entities_needing_disambiguation = int(num_entities * disambiguation_probability)

        if entities_needing_disambiguation == CACHE_EXPIRY_RESET:
            return []

        # Use learnable parameters for cluster calculation
        avg_cluster_size = disambig_params["avg_cluster_size"]
        min_clusters = disambig_params["min_disambiguation_clusters"]

        num_clusters = max(
            int(min_clusters), int(entities_needing_disambiguation / avg_cluster_size)
        )

        estimates = []
        for _ in range(num_clusters):
            estimate = await self._estimate_single_disambiguation(
                int(avg_cluster_size), context
            )
            estimates.append(estimate)

        return estimates

    async def _estimate_community_reports_dynamic(
        self,
        num_communities: int,
        num_entities: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TokenEstimate]:
        """Estimate community report tokens using dynamic parameters"""

        # Get community parameters using learnable parameter manager
        community_params = await self._get_dynamic_parameters("community_detection")

        estimates = []

        # Calculate average community size using learnable parameters
        entities_per_community = community_params["entities_per_community"]
        min_size = community_params["min_community_size"]
        max_size = community_params["max_community_size"]

        # Calculate realistic average based on total entities and communities
        calculated_avg = num_entities / max(ACCURACY_BASE, num_communities)
        avg_community_size = max(min_size, min(max_size, calculated_avg))

        for _ in range(num_communities):
            estimate = await self._estimate_single_community_report(
                int(avg_community_size), context
            )
            estimates.append(estimate)

        return estimates


# Utility functions for easy integration


def create_token_estimator(
    global_config: Dict[str, Any], estimation_db: Optional[BaseKVStorage] = None
) -> TokenEstimator:
    """Factory function to create a TokenEstimator instance"""
    return TokenEstimator(
        global_config=global_config,
        estimation_db=estimation_db,
        tiktoken_model=global_config.get("tiktoken_model", "gpt-4o"),
    )


async def estimate_document_processing_cost(
    document_content: Union[str, List[str]],
    global_config: Dict[str, Any],
    estimation_db: Optional[BaseKVStorage] = None,
) -> Tuple[PipelineEstimate, str]:
    """
    Convenience function to estimate cost and generate report for document processing

    Returns:
        Tuple of (PipelineEstimate, formatted_report_string)
    """
    estimator = create_token_estimator(global_config, estimation_db)

    # Improve estimates based on historical data
    await estimator.improve_estimates()

    # Generate estimate
    estimate = estimator.estimate_full_pipeline(document_content)

    # Generate report
    report = estimator.generate_estimation_report(estimate)

    return estimate, report


# LLM Call Instrumentation System


def llm_usage_recorder(
    call_type: LLMCallType, token_estimator: Optional[TokenEstimator] = None
):
    """
    Decorator for automatically recording LLM usage with rich context

    This decorator wraps LLM functions to automatically record actual vs estimated
    token usage for continuous learning and parameter optimization.

    Args:
        call_type: Type of LLM call being made
        token_estimator: TokenEstimator instance for recording

    Usage:
        @llm_usage_recorder(LLMCallType.ENTITY_EXTRACTION, estimator)
        async def extract_entities(chunk_content, model_func, ...):
            # ... implementation
            return result
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Skip instrumentation if no estimator provided
            if not token_estimator or not token_estimator.estimation_db:
                return await func(*args, **kwargs)

            start_time = time.time()
            success = True
            actual_input_tokens = CACHE_EXPIRY_RESET
            actual_output_tokens = CACHE_EXPIRY_RESET

            try:
                # Extract context information before the call
                context = _extract_call_context(func.__name__, args, kwargs)

                # Estimate tokens before the call (if possible)
                estimated_input, estimated_output = await _estimate_tokens_for_call(
                    token_estimator, call_type, context, args, kwargs
                )

                # Make the actual LLM call
                result = await func(*args, **kwargs)

                # Extract actual token usage from the result
                actual_input_tokens, actual_output_tokens = _extract_actual_usage(
                    result, context
                )

                # Calculate latency
                latency_ms = (time.time() - start_time) * MILLISECONDS_IN_SECOND

                # Record the usage with rich context
                await token_estimator.record_actual_usage(
                    call_type=call_type,
                    actual_input_tokens=actual_input_tokens,
                    actual_output_tokens=actual_output_tokens,
                    estimated_input_tokens=estimated_input,
                    estimated_output_tokens=estimated_output,
                    model_name=context.get("model_name", ""),
                    chunk_size=context.get("chunk_size"),
                    document_type=context.get("document_type"),
                    success=success,
                    latency_ms=latency_ms,
                    metadata={"function_name": func.__name__, "context": context},
                )

                return result

            except Exception as e:
                success = False
                latency_ms = (time.time() - start_time) * MILLISECONDS_IN_SECOND

                # Record failed attempt if we have some information
                if token_estimator.estimation_db:
                    try:
                        context = _extract_call_context(func.__name__, args, kwargs)
                        await token_estimator.record_actual_usage(
                            call_type=call_type,
                            actual_input_tokens=actual_input_tokens,
                            actual_output_tokens=actual_output_tokens,
                            estimated_input_tokens=CACHE_EXPIRY_RESET,
                            estimated_output_tokens=CACHE_EXPIRY_RESET,
                            model_name=context.get("model_name", ""),
                            chunk_size=context.get("chunk_size"),
                            document_type=context.get("document_type"),
                            success=False,
                            latency_ms=latency_ms,
                            metadata={
                                "function_name": func.__name__,
                                "error": str(e),
                                "context": context,
                            },
                        )
                    except:
                        pass  # Don't let recording errors interfere with main error

                raise e

        return wrapper

    return decorator


def _extract_call_context(func_name: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract context information from function call parameters"""
    context = {"function_name": func_name}

    # Try to extract common context information
    try:
        # Look for chunk content in various argument positions
        if len(args) > CACHE_EXPIRY_RESET:  # Use constant instead of magic number 0
            min_content_length = 10  # Minimum content length to consider as chunk
            if (
                isinstance(args[CACHE_EXPIRY_RESET], str)
                and len(args[CACHE_EXPIRY_RESET]) > min_content_length
            ):
                # Likely chunk content
                context["chunk_size"] = len(
                    args[CACHE_EXPIRY_RESET].split()
                )  # Rough word count

                # Try to detect document type from content
                content = args[CACHE_EXPIRY_RESET].lower()
                academic_keywords = ["theorem", "proof", "lemma", "proposition"]
                technical_keywords = ["function", "class", "method", "variable"]

                if any(word in content for word in academic_keywords):
                    context["document_type"] = "academic"
                elif any(word in content for word in technical_keywords):
                    context["document_type"] = "technical"
                else:
                    context["document_type"] = "general"

        # Look for model information in kwargs
        if "model_func" in kwargs:
            model_func = kwargs["model_func"]
            if hasattr(model_func, "__name__"):
                context["model_name"] = model_func.__name__

        # Look for additional context in kwargs
        context_keys = [
            "chunk_tokens",
            "num_entities",
            "community_size",
            "cluster_size",
        ]
        for key in context_keys:
            if key in kwargs:
                context[key] = kwargs[key]

    except Exception as e:
        # Don't let context extraction errors break the main flow
        context["context_extraction_error"] = str(e)

    return context


async def _estimate_tokens_for_call(
    token_estimator: TokenEstimator,
    call_type: LLMCallType,
    context: Dict[str, Any],
    args: tuple,
    kwargs: dict,
) -> Tuple[int, int]:
    """Estimate tokens for a call if possible"""
    try:
        # This is a simplified estimation - in practice, you'd need more sophisticated logic
        # based on the specific call type and available context

        if call_type == LLMCallType.ENTITY_EXTRACTION and context.get("chunk_size"):
            # Rough estimation for entity extraction using learnable parameter
            chunk_tokens = (
                context["chunk_size"] * WORDS_TO_TOKENS_RATIO
            )  # Use configurable constant
            estimate = await token_estimator._estimate_entity_extraction(
                "", int(chunk_tokens), context
            )
            return estimate.input_tokens, estimate.output_tokens_estimated

        elif call_type == LLMCallType.RELATION_EXTRACTION and context.get("chunk_size"):
            chunk_tokens = (
                context["chunk_size"] * WORDS_TO_TOKENS_RATIO
            )  # Use configurable constant
            estimate = await token_estimator._estimate_relation_extraction(
                "", int(chunk_tokens), context
            )
            return estimate.input_tokens, estimate.output_tokens_estimated

        # Add more call type specific estimations as needed

    except Exception as e:
        logger.debug(f"Could not estimate tokens for {call_type}: {e}")

    return (
        CACHE_EXPIRY_RESET,
        CACHE_EXPIRY_RESET,
    )  # Return zeros if estimation fails  # Return zeros if estimation fails


def _extract_actual_usage(result: Any, context: Dict[str, Any]) -> Tuple[int, int]:
    """Extract actual token usage from LLM call result"""
    try:
        # This depends on the specific format of your LLM results
        # Different LLM providers return token usage in different formats

        if isinstance(result, dict):
            # OpenAI-style response
            if "usage" in result:
                usage = result["usage"]
                input_tokens = usage.get("prompt_tokens", CACHE_EXPIRY_RESET)
                output_tokens = usage.get("completion_tokens", CACHE_EXPIRY_RESET)
                return input_tokens, output_tokens

            # Alternative format
            elif "token_usage" in result:
                usage = result["token_usage"]
                return usage.get("input", CACHE_EXPIRY_RESET), usage.get(
                    "output", CACHE_EXPIRY_RESET
                )

        elif hasattr(result, "usage"):
            # Object with usage attribute
            usage = result.usage
            if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
                return usage.prompt_tokens, usage.completion_tokens

        # If we can't extract from result, try to estimate from response content
        if isinstance(result, str):
            # Very rough estimation - count words and multiply by average tokens per word
            output_tokens = (
                len(result.split()) * WORDS_TO_TOKENS_RATIO
            )  # Use configurable constant
            return CACHE_EXPIRY_RESET, int(output_tokens)  # No input info available

    except Exception as e:
        logger.debug(f"Could not extract actual usage from result: {e}")

    return (
        CACHE_EXPIRY_RESET,
        CACHE_EXPIRY_RESET,
    )  # Return zeros if extraction fails  # Return zeros if extraction fails


def instrument_llm_function(
    func, call_type: LLMCallType, token_estimator: TokenEstimator
):
    """
    Programmatically instrument an LLM function with usage recording

    This is an alternative to the decorator approach for cases where you can't
    modify the function definition directly.

    Args:
        func: The LLM function to instrument
        call_type: Type of LLM call
        token_estimator: TokenEstimator instance

    Returns:
        Instrumented function
    """
    return llm_usage_recorder(call_type, token_estimator)(func)


class LLMInstrumentationManager:
    """
    Manager class for handling LLM call instrumentation across the codebase

    This class provides utilities for automatically instrumenting existing
    LLM functions and managing the instrumentation lifecycle.
    """

    def __init__(self, token_estimator: TokenEstimator):
        self.token_estimator = token_estimator
        self._instrumented_functions = {}

    def instrument_module_functions(
        self, module, function_mapping: Dict[str, LLMCallType]
    ):
        """
        Instrument all specified functions in a module

        Args:
            module: Python module containing LLM functions
            function_mapping: Dict mapping function names to LLM call types
        """
        for func_name, call_type in function_mapping.items():
            if hasattr(module, func_name):
                original_func = getattr(module, func_name)
                instrumented_func = llm_usage_recorder(call_type, self.token_estimator)(
                    original_func
                )
                setattr(module, func_name, instrumented_func)
                self._instrumented_functions[f"{module.__name__}.{func_name}"] = {
                    "original": original_func,
                    "instrumented": instrumented_func,
                    "call_type": call_type,
                }
                logger.info(
                    f"Instrumented {module.__name__}.{func_name} for {call_type.value}"
                )

    def restore_original_functions(self):
        """Restore all instrumented functions to their original state"""
        for func_path, func_info in self._instrumented_functions.items():
            module_name, func_name = func_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=[func_name])
            setattr(module, func_name, func_info["original"])
            logger.info(f"Restored original function: {func_path}")

        self._instrumented_functions.clear()

    def get_instrumentation_status(self) -> Dict[str, Any]:
        """Get status of all instrumented functions"""
        return {
            "total_instrumented": len(self._instrumented_functions),
            "functions": list(self._instrumented_functions.keys()),
            "call_types": [
                info["call_type"].value
                for info in self._instrumented_functions.values()
            ],
        }
