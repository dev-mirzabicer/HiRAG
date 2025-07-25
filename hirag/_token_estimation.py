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
        "gpt-4o": ModelPricing(5.0, 15.0, 128000, 4096),
        "gpt-4o-mini": ModelPricing(0.15, 0.60, 128000, 16384),
        "gpt-3.5-turbo": ModelPricing(1.0, 2.0, 16385, 4096),
        "claude-3-5-sonnet": ModelPricing(3.0, 15.0, 200000, 8192),
        "gemini-1.5-pro": ModelPricing(1.25, 5.0, 2000000, 8192),
        "gemini-1.5-flash": ModelPricing(0.075, 0.30, 1000000, 8192),
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

        # Initialize prompt token counts
        self._initialize_prompt_tokens()

        # Default estimates for variable content (all learnable parameters)
        self._default_estimates = self._initialize_default_estimates()
        
        # Dynamic estimation parameters (learnable loop counts, ratios, etc.)
        self._dynamic_params = self._initialize_dynamic_parameters()

        logger.info(f"TokenEstimator initialized with model: {tiktoken_model}")

    def _initialize_prompt_tokens(self):
        """Pre-calculate token counts for all prompt templates"""
        logger.info("Initializing prompt token counts...")

        for prompt_key, prompt_template in PROMPTS.items():
            if isinstance(prompt_template, str):
                # Count tokens in the base template (without variable substitution)
                base_tokens = len(self.tokenizer.encode(prompt_template))
                self._prompt_token_cache[prompt_key] = base_tokens
                logger.debug(f"Prompt '{prompt_key}': {base_tokens} base tokens")

    def _initialize_default_estimates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default estimates for variable content"""
        return {
            LLMCallType.ENTITY_EXTRACTION.value: {
                "base_output_min": 200,
                "base_output_max": 800,
                "entities_per_chunk_avg": 5,
                "tokens_per_entity": 40,
                "thinking_ratio": 0.3,  # 30% additional tokens for reasoning
            },
            LLMCallType.RELATION_EXTRACTION.value: {
                "base_output_min": 150,
                "base_output_max": 600,
                "relations_per_chunk_avg": 3,
                "tokens_per_relation": 35,
                "thinking_ratio": 0.25,
            },
            LLMCallType.CONTINUE_EXTRACTION.value: {
                "base_output_min": 50,
                "base_output_max": 300,
                "additional_entities_ratio": 0.2,  # 20% more entities
                "thinking_ratio": 0.2,
            },
            LLMCallType.LOOP_DETECTION.value: {
                "base_output_min": 5,
                "base_output_max": 20,
                "thinking_ratio": 0.1,
            },
            LLMCallType.HIERARCHICAL_CLUSTERING.value: {
                "base_output_min": 100,
                "base_output_max": 400,
                "tokens_per_cluster": 80,
                "thinking_ratio": 0.4,
            },
            LLMCallType.ENTITY_DISAMBIGUATION.value: {
                "base_output_min": 150,
                "base_output_max": 500,
                "tokens_per_decision": 50,
                "thinking_ratio": 0.5,
            },
            LLMCallType.ENTITY_MERGING.value: {
                "base_output_min": 100,
                "base_output_max": 300,
                "thinking_ratio": 0.3,
            },
            LLMCallType.COMMUNITY_REPORT.value: {
                "base_output_min": 500,
                "base_output_max": 2000,
                "tokens_per_entity": 20,
                "tokens_per_relation": 15,
                "thinking_ratio": 0.6,
            },
        }

    def _initialize_dynamic_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dynamic estimation parameters (all learnable)"""
        return {
            # Entity extraction dynamics
            "entity_extraction": {
                "entities_per_chunk_base": 5.0,  # Base entities per chunk
                "entities_per_100_tokens": 2.0,  # Additional entities per 100 tokens
                "chunk_complexity_factor": 1.2,  # Multiplier for complex content
                "min_entities_per_chunk": 1,
                "max_entities_per_chunk": 15,
            },
            
            # Gleaning loop dynamics
            "gleaning_loops": {
                "avg_actual_loops": 0.7,  # Most chunks don't need full gleaning
                "loop_probability_decay": 0.6,  # Each subsequent loop less likely
                "min_loop_probability": 0.1,
                "chunk_size_loop_factor": 1.0,  # Larger chunks more likely to need loops
            },
            
            # Hierarchical clustering dynamics
            "hierarchical_clustering": {
                "clustering_iterations_base": 3.0,  # Base number of iterations
                "entities_per_cluster": 5.0,  # Average entities per cluster
                "cluster_reduction_ratio": 0.4,  # How much clusters reduce each iteration
                "min_entities_for_clustering": 10,
                "max_clustering_iterations": 5,
            },
            
            # Entity disambiguation dynamics
            "entity_disambiguation": {
                "disambiguation_probability": 0.12,  # 12% of entities need disambiguation
                "avg_cluster_size": 2.8,  # Average entities per disambiguation cluster
                "complexity_disambiguation_factor": 1.3,  # Complex domains need more
                "min_disambiguation_clusters": 0,
                "max_disambiguation_cluster_size": 6,
            },
            
            # Community detection dynamics
            "community_detection": {
                "entities_per_community": 8.0,  # Average entities per community
                "community_size_variance": 0.4,  # Variance in community sizes
                "min_community_size": 3,
                "max_community_size": 25,
                "community_overlap_ratio": 0.15,  # Some entities appear in multiple communities
            },
            
            # General pipeline dynamics
            "pipeline": {
                "relation_to_entity_ratio": 0.6,  # Relations per entity
                "temporary_entity_ratio": 0.15,  # Percentage of temporary entities
                "entity_merging_ratio": 0.08,  # Percentage of entities that get merged
                "content_complexity_multiplier": 1.0,  # Global complexity adjustment
            }
        }

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

    def _estimate_entity_extraction(
        self, chunk_content: str, chunk_tokens: int
    ) -> TokenEstimate:
        """Estimate tokens for entity extraction from a single chunk"""
        # Input tokens: prompt + chunk content
        prompt_tokens = self._prompt_token_cache.get("hi_entity_extraction", 0)
        context_tokens = len(self.tokenizer.encode(str(self._get_entity_context())))
        input_tokens = prompt_tokens + context_tokens + chunk_tokens

        # Output estimation based on chunk characteristics
        defaults = self._default_estimates[LLMCallType.ENTITY_EXTRACTION.value]

        # Estimate based on chunk size and content complexity
        estimated_entities = (
            max(1, chunk_tokens // 200) * defaults["entities_per_chunk_avg"]
        )

        output_min = defaults["base_output_min"]
        output_max = defaults["base_output_max"] + (
            estimated_entities * defaults["tokens_per_entity"]
        )
        output_estimated = int((output_min + output_max) / 2)

        thinking_tokens = int(output_estimated * defaults["thinking_ratio"])

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
            },
        )

    def _estimate_relation_extraction(
        self, chunk_content: str, chunk_tokens: int
    ) -> TokenEstimate:
        """Estimate tokens for relation extraction from a single chunk"""
        prompt_tokens = self._prompt_token_cache.get("hi_relation_extraction", 0)
        context_tokens = len(self.tokenizer.encode(str(self._get_relation_context())))
        input_tokens = prompt_tokens + context_tokens + chunk_tokens

        defaults = self._default_estimates[LLMCallType.RELATION_EXTRACTION.value]

        estimated_relations = (
            max(1, chunk_tokens // 300) * defaults["relations_per_chunk_avg"]
        )

        output_min = defaults["base_output_min"]
        output_max = defaults["base_output_max"] + (
            estimated_relations * defaults["tokens_per_relation"]
        )
        output_estimated = int((output_min + output_max) / 2)

        thinking_tokens = int(output_estimated * defaults["thinking_ratio"])

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
            },
        )

    def _estimate_continue_extraction(
        self, chunk_content: str, chunk_tokens: int
    ) -> TokenEstimate:
        """Estimate tokens for continue extraction (gleaning)"""
        prompt_tokens = self._prompt_token_cache.get("entiti_continue_extraction", 0)
        # Continue extraction includes conversation history
        history_tokens = chunk_tokens // 2  # Rough estimate of history size
        input_tokens = prompt_tokens + history_tokens

        defaults = self._default_estimates[LLMCallType.CONTINUE_EXTRACTION.value]

        output_min = defaults["base_output_min"]
        output_max = defaults["base_output_max"]
        output_estimated = int((output_min + output_max) / 2)

        thinking_tokens = int(output_estimated * defaults["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.CONTINUE_EXTRACTION,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={"chunk_tokens": chunk_tokens},
        )

    def _estimate_loop_detection(
        self, chunk_content: str, chunk_tokens: int
    ) -> TokenEstimate:
        """Estimate tokens for loop detection in gleaning"""
        prompt_tokens = self._prompt_token_cache.get("entiti_if_loop_extraction", 0)
        history_tokens = chunk_tokens // 2
        input_tokens = prompt_tokens + history_tokens

        defaults = self._default_estimates[LLMCallType.LOOP_DETECTION.value]

        output_min = defaults["base_output_min"]
        output_max = defaults["base_output_max"]
        output_estimated = int((output_min + output_max) / 2)

        thinking_tokens = int(output_estimated * defaults["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.LOOP_DETECTION,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={"chunk_tokens": chunk_tokens},
        )

    def estimate_hierarchical_clustering_tokens(
        self, num_entities: int, clustering_iterations: int = 3
    ) -> List[TokenEstimate]:
        """Estimate tokens for hierarchical clustering operations"""
        estimates = []

        for iteration in range(clustering_iterations):
            # Each iteration processes fewer entities
            entities_this_iter = max(1, num_entities // (2**iteration))

            estimate = self._estimate_single_clustering_iteration(entities_this_iter)
            estimates.append(estimate)

        return estimates

    def _estimate_single_clustering_iteration(self, num_entities: int) -> TokenEstimate:
        """Estimate tokens for a single hierarchical clustering iteration"""
        prompt_tokens = self._prompt_token_cache.get("summary_clusters", 0)

        # Input includes entity descriptions
        entity_descriptions_tokens = (
            num_entities * 50
        )  # Average tokens per entity description
        input_tokens = prompt_tokens + entity_descriptions_tokens

        defaults = self._default_estimates[LLMCallType.HIERARCHICAL_CLUSTERING.value]

        num_clusters = max(1, num_entities // 5)  # Rough estimate of clusters formed

        output_min = defaults["base_output_min"]
        output_max = defaults["base_output_max"] + (
            num_clusters * defaults["tokens_per_cluster"]
        )
        output_estimated = int((output_min + output_max) / 2)

        thinking_tokens = int(output_estimated * defaults["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.HIERARCHICAL_CLUSTERING,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={"num_entities": num_entities, "estimated_clusters": num_clusters},
        )

    def estimate_disambiguation_tokens(
        self, num_entity_clusters: int, avg_cluster_size: int = 3
    ) -> List[TokenEstimate]:
        """Estimate tokens for entity disambiguation operations"""
        estimates = []

        for _ in range(num_entity_clusters):
            estimate = self._estimate_single_disambiguation(avg_cluster_size)
            estimates.append(estimate)

        return estimates

    def _estimate_single_disambiguation(self, cluster_size: int) -> TokenEstimate:
        """Estimate tokens for disambiguating a single entity cluster"""
        prompt_tokens = self._prompt_token_cache.get("entity_disambiguation", 0)

        # Input includes entity details and context
        entity_context_tokens = cluster_size * 200  # Average context per entity
        input_tokens = prompt_tokens + entity_context_tokens

        defaults = self._default_estimates[LLMCallType.ENTITY_DISAMBIGUATION.value]

        output_min = defaults["base_output_min"]
        output_max = defaults["base_output_max"] + (
            cluster_size * defaults["tokens_per_decision"]
        )
        output_estimated = int((output_min + output_max) / 2)

        thinking_tokens = int(output_estimated * defaults["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.ENTITY_DISAMBIGUATION,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={"cluster_size": cluster_size},
        )

    def estimate_community_report_tokens(
        self, num_communities: int, avg_community_size: int = 10
    ) -> List[TokenEstimate]:
        """Estimate tokens for community report generation"""
        estimates = []

        for _ in range(num_communities):
            estimate = self._estimate_single_community_report(avg_community_size)
            estimates.append(estimate)

        return estimates

    def _estimate_single_community_report(self, community_size: int) -> TokenEstimate:
        """Estimate tokens for generating a single community report"""
        prompt_tokens = self._prompt_token_cache.get("community_report", 0)

        # Input includes community entities and relationships
        community_context_tokens = (
            community_size * 100
        )  # Average tokens per community member
        input_tokens = prompt_tokens + community_context_tokens

        defaults = self._default_estimates[LLMCallType.COMMUNITY_REPORT.value]

        output_min = defaults["base_output_min"]
        output_max = defaults["base_output_max"] + (
            community_size * defaults["tokens_per_entity"]
        )
        output_estimated = int((output_min + output_max) / 2)

        thinking_tokens = int(output_estimated * defaults["thinking_ratio"])

        return TokenEstimate(
            call_type=LLMCallType.COMMUNITY_REPORT,
            input_tokens=input_tokens,
            output_tokens_min=output_min,
            output_tokens_max=output_max,
            output_tokens_estimated=output_estimated,
            thinking_tokens=thinking_tokens,
            metadata={"community_size": community_size},
        )

    def estimate_full_pipeline(
        self,
        document_content: Union[str, List[str]],
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> PipelineEstimate:
        """
        Estimate tokens for the complete HiRAG ingestion pipeline

        Args:
            document_content: Input document(s) to be processed
            config_overrides: Optional configuration overrides

        Returns:
            PipelineEstimate object with complete token breakdown
        """
        # Merge configuration
        config = {**self.global_config, **(config_overrides or {})}

        # Process input
        if isinstance(document_content, str):
            documents = [document_content]
        else:
            documents = document_content

        # Calculate document statistics
        total_document_tokens = sum(self.count_tokens(doc) for doc in documents)

        # Estimate chunk creation
        chunk_token_size = config.get("chunk_token_size", 1200)
        chunk_overlap = config.get("chunk_overlap_token_size", 100)
        effective_chunk_size = chunk_token_size - chunk_overlap

        num_chunks = max(1, total_document_tokens // effective_chunk_size)
        token_per_chunk = total_document_tokens / num_chunks

        logger.info(
            f"Pipeline estimation: {total_document_tokens} tokens → {num_chunks} chunks"
        )

        all_estimates = []

        # 1. Chunk processing (entity & relation extraction with dynamic gleaning)
        chunk_estimates = self._estimate_chunk_processing_dynamic(
            num_chunks=num_chunks,
            token_per_chunk=token_per_chunk,
            enable_hierarchical=config.get("enable_hierachical_mode", True),
            max_gleaning_iterations=config.get("entity_extract_max_gleaning", 1)
        )
        all_estimates.extend(chunk_estimates)

        # 2. Hierarchical clustering
        if config.get("enable_hierachical_mode", True):
            estimated_entities = self._estimate_entity_count(num_chunks, token_per_chunk)
            clustering_estimates = self._estimate_hierarchical_clustering_dynamic(estimated_entities)
            all_estimates.extend(clustering_estimates)

        # 3. Entity disambiguation
        if config.get("enable_entity_disambiguation", True):
            disambiguation_estimates = self._estimate_disambiguation_dynamic(estimated_entities)
            all_estimates.extend(disambiguation_estimates)

        # 4. Community reports
        estimated_communities = self._estimate_community_count(estimated_entities)
        community_estimates = self._estimate_community_reports_dynamic(estimated_communities, estimated_entities)
        all_estimates.extend(community_estimates)

        # Calculate cost estimates
        model_name = self._get_primary_model_name(config)
        estimated_cost = self._calculate_cost(all_estimates, model_name)

        return PipelineEstimate(
            document_tokens=total_document_tokens,
            num_chunks=num_chunks,
            token_per_chunk=token_per_chunk,
            call_estimates=all_estimates,
            estimated_cost_usd=estimated_cost,
            metadata={
                "model_name": model_name,
                "estimated_entities": estimated_entities,
                "estimated_communities": estimated_communities,
                "estimated_gleaning_loops": self._estimate_total_gleaning_loops(num_chunks),
                "estimated_clustering_iterations": self._estimate_clustering_iterations(estimated_entities),
                "config_snapshot": config,
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
        """Calculate estimated cost in USD for the given estimates and model"""
        if model_name not in self.MODEL_PRICING:
            logger.warning(
                f"No pricing data for model {model_name}, using gpt-4o-mini as fallback"
            )
            model_name = "gpt-4o-mini"

        pricing = self.MODEL_PRICING[model_name]

        total_input_tokens = sum(est.total_input_tokens for est in estimates)
        total_output_tokens = sum(est.total_output_tokens for est in estimates)

        input_cost = (total_input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (total_output_tokens / 1000) * pricing.output_cost_per_1k

        return input_cost + output_cost

    async def record_actual_usage(
        self,
        call_type: LLMCallType,
        actual_input_tokens: int,
        actual_output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record actual token usage to improve future estimates"""
        if not self.estimation_db:
            return

        usage_record = {
            "call_type": call_type.value,
            "actual_input_tokens": actual_input_tokens,
            "actual_output_tokens": actual_output_tokens,
            "timestamp": asyncio.get_event_loop().time(),
            "metadata": metadata or {},
        }

        record_id = compute_mdhash_id(
            f"{call_type.value}_{actual_input_tokens}_{actual_output_tokens}",
            prefix="usage-",
        )

        await self.estimation_db.upsert({record_id: usage_record})
        logger.debug(
            f"Recorded actual usage for {call_type.value}: {actual_input_tokens}→{actual_output_tokens}"
        )

    async def improve_estimates(self):
        """Analyze recorded usage data to improve estimation accuracy"""
        if not self.estimation_db:
            logger.warning("No estimation database available for learning")
            return

        logger.info("Analyzing usage data to improve estimates...")

        # This would implement machine learning or statistical analysis
        # to update the default estimates based on actual usage patterns
        # For now, we'll implement a simple averaging approach

        all_records = await self.estimation_db.get_all()
        if not all_records:
            logger.info("No usage data available for learning")
            return

        # Group records by call type
        usage_by_type = defaultdict(list)
        for record in all_records.values():
            call_type = record.get("call_type")
            if call_type:
                usage_by_type[call_type].append(record)

        # Update estimates based on actual usage
        for call_type, records in usage_by_type.items():
            if len(records) < 5:  # Need minimum data points
                continue

            avg_output = sum(r["actual_output_tokens"] for r in records) / len(records)
            current_estimate = self._default_estimates.get(call_type, {}).get(
                "base_output_max", 0
            )

            # Simple learning: adjust estimate by 10% towards actual average
            if current_estimate > 0:
                adjustment_factor = 0.1
                new_estimate = (
                    current_estimate * (1 - adjustment_factor)
                    + avg_output * adjustment_factor
                )

                # Update the estimate
                if call_type in self._default_estimates:
                    old_estimate = self._default_estimates[call_type]["base_output_max"]
                    self._default_estimates[call_type]["base_output_max"] = int(
                        new_estimate
                    )

                    logger.info(
                        f"Updated {call_type} estimate: {old_estimate} → {int(new_estimate)} "
                        f"(based on {len(records)} samples)"
                    )

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
    
    def _estimate_entity_count(self, num_chunks: int, token_per_chunk: float) -> int:
        """Estimate total entities using dynamic parameters"""
        entity_params = self._dynamic_params["entity_extraction"]
        
        base_entities = num_chunks * entity_params["entities_per_chunk_base"]
        token_based_entities = (token_per_chunk / 100) * entity_params["entities_per_100_tokens"] * num_chunks
        
        total_entities = base_entities + token_based_entities
        total_entities *= entity_params["chunk_complexity_factor"]
        
        return max(
            entity_params["min_entities_per_chunk"] * num_chunks,
            min(int(total_entities), entity_params["max_entities_per_chunk"] * num_chunks)
        )
    
    def _estimate_total_gleaning_loops(self, num_chunks: int) -> float:
        """Estimate total gleaning loops across all chunks"""
        gleaning_params = self._dynamic_params["gleaning_loops"]
        return num_chunks * gleaning_params["avg_actual_loops"]
    
    def _estimate_clustering_iterations(self, num_entities: int) -> int:
        """Estimate hierarchical clustering iterations"""
        clustering_params = self._dynamic_params["hierarchical_clustering"]
        
        if num_entities < clustering_params["min_entities_for_clustering"]:
            return 0
        
        iterations = int(clustering_params["clustering_iterations_base"])
        return min(iterations, clustering_params["max_clustering_iterations"])
    
    def _estimate_community_count(self, num_entities: int) -> int:
        """Estimate number of communities"""
        community_params = self._dynamic_params["community_detection"]
        
        estimated_communities = num_entities / community_params["entities_per_community"]
        # Account for overlap
        estimated_communities *= (1 - community_params["community_overlap_ratio"])
        
        return max(1, int(estimated_communities))
    
    def _estimate_chunk_processing_dynamic(
        self,
        num_chunks: int,
        token_per_chunk: float,
        enable_hierarchical: bool = True,
        max_gleaning_iterations: int = 1
    ) -> List[TokenEstimate]:
        """Optimized chunk processing estimation without mock chunks"""
        estimates = []
        
        # Entity extraction for all chunks
        entity_estimate = self._estimate_entity_extraction("", int(token_per_chunk))
        for _ in range(num_chunks):
            estimates.append(entity_estimate)
        
        # Relation extraction (if hierarchical mode)
        if enable_hierarchical:
            relation_estimate = self._estimate_relation_extraction("", int(token_per_chunk))
            for _ in range(num_chunks):
                estimates.append(relation_estimate)
        
        # Dynamic gleaning estimation
        gleaning_params = self._dynamic_params["gleaning_loops"]
        actual_gleaning_loops = int(num_chunks * gleaning_params["avg_actual_loops"])
        
        continue_estimate = self._estimate_continue_extraction("", int(token_per_chunk))
        loop_estimate = self._estimate_loop_detection("", int(token_per_chunk))
        
        for _ in range(actual_gleaning_loops):
            estimates.append(continue_estimate)
            estimates.append(loop_estimate)
        
        return estimates
    
    def _estimate_hierarchical_clustering_dynamic(self, num_entities: int) -> List[TokenEstimate]:
        """Dynamic hierarchical clustering estimation"""
        clustering_params = self._dynamic_params["hierarchical_clustering"]
        
        if num_entities < clustering_params["min_entities_for_clustering"]:
            return []
        
        estimates = []
        current_entities = num_entities
        iterations = 0
        
        while (current_entities > clustering_params["entities_per_cluster"] and 
               iterations < clustering_params["max_clustering_iterations"]):
            
            estimate = self._estimate_single_clustering_iteration(current_entities)
            estimates.append(estimate)
            
            # Reduce entities for next iteration
            current_entities = int(current_entities * clustering_params["cluster_reduction_ratio"])
            iterations += 1
        
        return estimates
    
    def _estimate_disambiguation_dynamic(self, num_entities: int) -> List[TokenEstimate]:
        """Dynamic entity disambiguation estimation"""
        disambig_params = self._dynamic_params["entity_disambiguation"]
        
        entities_needing_disambiguation = int(
            num_entities * disambig_params["disambiguation_probability"]
        )
        
        if entities_needing_disambiguation == 0:
            return []
        
        # Group entities into clusters
        avg_cluster_size = disambig_params["avg_cluster_size"]
        num_clusters = max(1, int(entities_needing_disambiguation / avg_cluster_size))
        
        estimates = []
        for _ in range(num_clusters):
            estimate = self._estimate_single_disambiguation(int(avg_cluster_size))
            estimates.append(estimate)
        
        return estimates
    
    def _estimate_community_reports_dynamic(
        self, 
        num_communities: int, 
        num_entities: int
    ) -> List[TokenEstimate]:
        """Dynamic community report estimation"""
        community_params = self._dynamic_params["community_detection"]
        
        estimates = []
        avg_community_size = int(num_entities / max(1, num_communities))
        
        # Ensure community size is within bounds
        avg_community_size = max(
            community_params["min_community_size"],
            min(avg_community_size, community_params["max_community_size"])
        )
        
        for _ in range(num_communities):
            estimate = self._estimate_single_community_report(avg_community_size)
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
        tiktoken_model=global_config.get("tiktoken_model_name", "gpt-4o"),
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
