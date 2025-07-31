"""
Centralized Configuration System for HiRAG

This module provides a unified, type-safe configuration system that consolidates
all configuration parameters from across the HiRAG codebase. It supports:
- YAML-based configuration files
- Environment variable overrides
- Type validation and default values
- Hierarchical configuration structure
- Backward compatibility with existing code

Usage:
    from hirag.config import get_config, ConfigManager

    # Get the global configuration instance
    config = get_config()

    # Access configuration values
    chunk_size = config.text_processing.chunk_token_size
    api_key = config.api.openai.api_key

    # Load from custom file
    config_manager = ConfigManager("custom_config.yaml")
    config = config_manager.config
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Type aliases for better clarity
StrOrNone = Optional[str]
IntOrNone = Optional[int]
FloatOrNone = Optional[float]
BoolOrNone = Optional[bool]


@dataclass
class APIConfig:
    """API configuration for different LLM providers"""

    @dataclass
    class OpenAI:
        api_key: StrOrNone = None
        base_url: StrOrNone = None
        model: str = "gpt-4o"
        embedding_model: str = "text-embedding-ada-002"
        embedding_dim: int = 1536

    @dataclass
    class GLM:
        api_key: StrOrNone = None
        base_url: str = "https://open.bigmodel.cn/api/paas/v4"
        model: str = "glm-4-plus"
        embedding_model: str = "embedding-3"
        embedding_dim: int = 2048

    @dataclass
    class Deepseek:
        api_key: StrOrNone = None
        base_url: str = "https://api.deepseek.com"
        model: str = "deepseek-chat"

    @dataclass
    class Google:
        api_key: StrOrNone = None
        model: str = "gemini-2.5-pro"
        embedding_model: str = "gemini-embedding-exp-03-07"
        embedding_dim: int = 3072

    openai: OpenAI = field(default_factory=OpenAI)
    glm: GLM = field(default_factory=GLM)
    deepseek: Deepseek = field(default_factory=Deepseek)
    google: Google = field(default_factory=Google)


@dataclass
class TextProcessingConfig:
    """Configuration for text processing and chunking"""

    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"
    max_token_size: int = 8192
    overlap_token_size: int = 128  # From _op.py
    encoding_num_threads: int = 16  # From _op.py


@dataclass
class EntityProcessingConfig:
    """Configuration for entity extraction and processing"""

    extract_max_gleaning: int = 1
    summary_to_max_tokens: int = 500
    max_length_in_cluster: int = 60000  # From _cluster_utils.py

    # Entity Disambiguation & Merging (EDM) settings
    enable_disambiguation: bool = True
    lexical_similarity_threshold: float = 0.85
    semantic_similarity_threshold: float = 0.88
    max_cluster_size: int = 6
    max_context_tokens: int = 4000
    min_merge_confidence: float = 0.8
    vector_search_top_k: int = 50
    memory_batch_size: int = 1000

    # EDM performance settings
    embedding_batch_size: int = 32
    max_concurrent_llm_calls: int = 3
    max_llm_retries: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 30.0


@dataclass
class GraphOperationsConfig:
    """Configuration for graph operations and clustering"""

    cluster_algorithm: str = "leiden"
    max_cluster_size: int = 10
    cluster_seed: int = 0xDEADBEEF
    node_embedding_algorithm: str = "node2vec"

    # Node2Vec parameters
    node2vec_dimensions: int = 1536
    node2vec_num_walks: int = 10
    node2vec_walk_length: int = 40
    node2vec_window_size: int = 2
    node2vec_iterations: int = 3
    node2vec_random_seed: int = 3

    # Clustering parameters from _cluster_utils.py
    random_seed: int = 224
    umap_n_neighbors: int = 15
    umap_metric: str = "cosine"
    max_clusters: int = 50
    gmm_n_init: int = 5
    hierarchical_layers: int = 50
    reduction_dimension: int = 2
    cluster_threshold: float = 0.1
    similarity_threshold: float = 0.98

    # Neo4j specific settings
    neo4j_max_hops: int = 10
    neo4j_weight_property: str = "weight"
    neo4j_default_algorithm: str = "cypher_shortest"

    # Vector database settings
    cosine_better_than_threshold: float = 0.2


@dataclass
class PerformanceConfig:
    """Configuration for performance and concurrency settings"""

    # Embedding settings
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 8
    embeddings_batch_size: int = 64  # From _op.py and _cluster_utils.py

    # Model concurrency
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 8
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 8

    # Community report settings
    community_report_max_token_size: int = 12000  # From _op.py

    # Rate limiting defaults
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    tokens_per_minute: int = 150000
    concurrent_requests: int = 10
    backpressure_threshold: float = 0.8


@dataclass
class QualityControlConfig:
    """Configuration for quality control and validation"""

    query_better_than_threshold: float = 0.2

    # Prompt field separators and delimiters
    graph_field_sep: str = "<SEP>"
    record_delimiter: str = "<|>"
    record_sep: str = "<|RECORD|>"
    complete_delimiter: str = "<|COMPLETE|>"


@dataclass
class SystemBehaviorConfig:
    """Configuration for system behavior and infrastructure"""

    working_dir: str = "./hirag_cache"
    enable_local: bool = True
    enable_naive_rag: bool = False
    enable_hierarchical_mode: bool = True
    enable_llm_cache: bool = True
    always_create_working_dir: bool = True
    enable_entity_names_vdb: bool = True

    # Advanced infrastructure systems
    enable_token_estimation: bool = True
    enable_checkpointing: bool = True
    enable_retry_management: bool = True
    enable_rate_limiting: bool = True
    enable_progress_tracking: bool = True
    enable_estimation_learning: bool = True

    # Checkpointing system
    checkpoint_auto_interval: float = 30.0
    checkpoint_max_history: int = 10

    # Retry management
    retry_default_max_attempts: int = 3
    retry_default_initial_delay: float = 1.0
    retry_default_max_delay: float = 60.0
    retry_enable_circuit_breaker: bool = True

    # Rate limiting
    rate_limiting_adaptive_adjustment: bool = True
    rate_limiting_conservative_mode: bool = False

    # Progress tracking
    progress_dashboard_type: str = "terminal"
    progress_update_interval: float = 1.0
    progress_enable_web_dashboard: bool = False
    progress_web_port: int = 8080

    # Estimation database
    estimation_db_max_records: int = 10000
    estimation_db_cleanup_days: int = 90


@dataclass
class HiRAGConfig:
    """Main configuration class containing all HiRAG settings"""

    api: APIConfig = field(default_factory=APIConfig)
    text_processing: TextProcessingConfig = field(default_factory=TextProcessingConfig)
    entity_processing: EntityProcessingConfig = field(
        default_factory=EntityProcessingConfig
    )
    graph_operations: GraphOperationsConfig = field(
        default_factory=GraphOperationsConfig
    )
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    system_behavior: SystemBehaviorConfig = field(default_factory=SystemBehaviorConfig)

    # Additional parameters for extensibility
    addon_params: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """
    Configuration manager that handles loading, validation, and environment variable overrides
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager

        Args:
            config_file: Path to the configuration file. If None, will look for
                        'config.yaml' in the current directory or use defaults.
        """
        self.config_file = config_file or self._find_config_file()
        self.config = self._load_config()
        self._apply_env_overrides()
        self._validate_config()

    def _find_config_file(self) -> Optional[str]:
        """Find the configuration file in common locations"""
        possible_locations = [
            "config.yaml",
            "hirag_config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
        ]

        for location in possible_locations:
            if os.path.exists(location):
                logger.info(f"Found configuration file: {location}")
                return location

        logger.info("No configuration file found, using defaults")
        return None

    def _load_config(self) -> HiRAGConfig:
        """Load configuration from file or use defaults"""
        if not self.config_file or not os.path.exists(self.config_file):
            logger.info("Using default configuration")
            return HiRAGConfig()

        try:
            with open(self.config_file, "r") as f:
                yaml_data = yaml.safe_load(f) or {}

            return self._create_config_from_dict(yaml_data)

        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_file}: {e}")
            logger.info("Falling back to default configuration")
            return HiRAGConfig()

    def _create_config_from_dict(self, data: Dict[str, Any]) -> HiRAGConfig:
        """Create HiRAGConfig from dictionary data"""
        config = HiRAGConfig()

        # Map legacy YAML structure to new structure
        if "openai" in data:
            self._update_nested_config(config.api.openai, data["openai"])
        if "glm" in data:
            self._update_nested_config(config.api.glm, data["glm"])
        if "deepseek" in data:
            self._update_nested_config(config.api.deepseek, data["deepseek"])
        if "google" in data:
            self._update_nested_config(config.api.google, data["google"])

        # Handle model_params section
        if "model_params" in data:
            model_params = data["model_params"]
            if "openai_embedding_dim" in model_params:
                config.api.openai.embedding_dim = model_params["openai_embedding_dim"]
            if "glm_embedding_dim" in model_params:
                config.api.glm.embedding_dim = model_params["glm_embedding_dim"]
            if "google_embedding_dim" in model_params:
                config.api.google.embedding_dim = model_params["google_embedding_dim"]
            if "max_token_size" in model_params:
                config.text_processing.max_token_size = model_params["max_token_size"]

        # Handle hirag section
        if "hirag" in data:
            hirag_data = data["hirag"]
            if "working_dir" in hirag_data:
                config.system_behavior.working_dir = hirag_data["working_dir"]
            if "enable_llm_cache" in hirag_data:
                config.system_behavior.enable_llm_cache = hirag_data["enable_llm_cache"]
            if "enable_hierachical_mode" in hirag_data:
                config.system_behavior.enable_hierarchical_mode = hirag_data[
                    "enable_hierachical_mode"
                ]
            if "embedding_batch_num" in hirag_data:
                config.performance.embedding_batch_num = hirag_data[
                    "embedding_batch_num"
                ]
            if "embedding_func_max_async" in hirag_data:
                config.performance.embedding_func_max_async = hirag_data[
                    "embedding_func_max_async"
                ]
            if "enable_naive_rag" in hirag_data:
                config.system_behavior.enable_naive_rag = hirag_data["enable_naive_rag"]

        return config

    def _update_nested_config(self, config_obj: Any, data: Dict[str, Any]):
        """Update nested configuration object with data from dictionary"""
        for key, value in data.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration"""
        env_mappings = {
            # API Keys
            "OPENAI_API_KEY": "api.openai.api_key",
            "OPENAI_BASE_URL": "api.openai.base_url",
            "GLM_API_KEY": "api.glm.api_key",
            "DEEPSEEK_API_KEY": "api.deepseek.api_key",
            "GOOGLE_API_KEY": "api.google.api_key",
            # System behavior
            "HIRAG_WORKING_DIR": "system_behavior.working_dir",
            "HIRAG_ENABLE_CACHE": "system_behavior.enable_llm_cache",
            "HIRAG_ENABLE_HIERARCHICAL": "system_behavior.enable_hierarchical_mode",
            # Performance settings
            "HIRAG_EMBEDDING_BATCH_SIZE": "performance.embedding_batch_num",
            "HIRAG_MAX_ASYNC": "performance.embedding_func_max_async",
            # Text processing
            "HIRAG_CHUNK_SIZE": "text_processing.chunk_token_size",
            "HIRAG_CHUNK_OVERLAP": "text_processing.chunk_overlap_token_size",
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                self._set_nested_config_value(self.config, config_path, env_value)
                logger.info(f"Applied environment override: {env_var} -> {config_path}")

    def _set_nested_config_value(self, config: Any, path: str, value: str):
        """Set a nested configuration value using dot notation"""
        parts = path.split(".")
        current = config

        for part in parts[:-1]:
            current = getattr(current, part)

        final_attr = parts[-1]

        # Type conversion based on the current type
        if hasattr(current, final_attr):
            current_value = getattr(current, final_attr)
            if isinstance(current_value, bool):
                value = value.lower() in ("true", "1", "yes", "on")  # type: ignore[arg-type]
            elif isinstance(current_value, int):
                value = int(value)  # type: ignore[arg-type]
            elif isinstance(current_value, float):
                value = float(value)  # type: ignore[arg-type]

        setattr(current, final_attr, value)

    def _validate_config(self):
        """Validate configuration values"""
        # Basic validation - can be extended
        if self.config.text_processing.chunk_token_size <= 0:
            raise ValueError("chunk_token_size must be positive")

        if self.config.text_processing.chunk_overlap_token_size < 0:
            raise ValueError("chunk_overlap_token_size must be non-negative")

        if (
            self.config.entity_processing.lexical_similarity_threshold < 0
            or self.config.entity_processing.lexical_similarity_threshold > 1
        ):
            raise ValueError("lexical_similarity_threshold must be between 0 and 1")

        if self.config.performance.embedding_batch_num <= 0:
            raise ValueError("embedding_batch_num must be positive")

        logger.info("Configuration validation passed")

    def get_legacy_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration in the legacy format for backward compatibility
        This method helps migrate existing code gradually
        """
        return {
            # Legacy HiRAG class fields
            "working_dir": self.config.system_behavior.working_dir,
            "enable_local": self.config.system_behavior.enable_local,
            "enable_naive_rag": self.config.system_behavior.enable_naive_rag,
            "enable_hierachical_mode": self.config.system_behavior.enable_hierarchical_mode,
            "chunk_token_size": self.config.text_processing.chunk_token_size,
            "chunk_overlap_token_size": self.config.text_processing.chunk_overlap_token_size,
            "tiktoken_model_name": self.config.text_processing.tiktoken_model_name,
            "entity_extract_max_gleaning": self.config.entity_processing.extract_max_gleaning,
            "entity_summary_to_max_tokens": self.config.entity_processing.summary_to_max_tokens,
            "graph_cluster_algorithm": self.config.graph_operations.cluster_algorithm,
            "max_graph_cluster_size": self.config.graph_operations.max_cluster_size,
            "graph_cluster_seed": self.config.graph_operations.cluster_seed,
            "node_embedding_algorithm": self.config.graph_operations.node_embedding_algorithm,
            "embedding_batch_num": self.config.performance.embedding_batch_num,
            "embedding_func_max_async": self.config.performance.embedding_func_max_async,
            "query_better_than_threshold": self.config.quality_control.query_better_than_threshold,
            "best_model_max_token_size": self.config.performance.best_model_max_token_size,
            "best_model_max_async": self.config.performance.best_model_max_async,
            "cheap_model_max_token_size": self.config.performance.cheap_model_max_token_size,
            "cheap_model_max_async": self.config.performance.cheap_model_max_async,
            "enable_llm_cache": self.config.system_behavior.enable_llm_cache,
            "always_create_working_dir": self.config.system_behavior.always_create_working_dir,
            "addon_params": self.config.addon_params,
            # EDM configuration
            "enable_entity_disambiguation": self.config.entity_processing.enable_disambiguation,
            "edm_lexical_similarity_threshold": self.config.entity_processing.lexical_similarity_threshold,
            "edm_semantic_similarity_threshold": self.config.entity_processing.semantic_similarity_threshold,
            "edm_max_cluster_size": self.config.entity_processing.max_cluster_size,
            "edm_max_context_tokens": self.config.entity_processing.max_context_tokens,
            "edm_min_merge_confidence": self.config.entity_processing.min_merge_confidence,
            "enable_entity_names_vdb": self.config.system_behavior.enable_entity_names_vdb,
            "edm_vector_search_top_k": self.config.entity_processing.vector_search_top_k,
            "edm_memory_batch_size": self.config.entity_processing.memory_batch_size,
            "edm_embedding_batch_size": self.config.entity_processing.embedding_batch_size,
            "edm_max_concurrent_llm_calls": self.config.entity_processing.max_concurrent_llm_calls,
            # Advanced infrastructure
            "enable_token_estimation": self.config.system_behavior.enable_token_estimation,
            "enable_checkpointing": self.config.system_behavior.enable_checkpointing,
            "enable_retry_management": self.config.system_behavior.enable_retry_management,
            "enable_rate_limiting": self.config.system_behavior.enable_rate_limiting,
            "enable_progress_tracking": self.config.system_behavior.enable_progress_tracking,
            "enable_estimation_learning": self.config.system_behavior.enable_estimation_learning,
            "checkpoint_auto_interval": self.config.system_behavior.checkpoint_auto_interval,
            "checkpoint_max_history": self.config.system_behavior.checkpoint_max_history,
            "retry_default_max_attempts": self.config.system_behavior.retry_default_max_attempts,
            "retry_default_initial_delay": self.config.system_behavior.retry_default_initial_delay,
            "retry_default_max_delay": self.config.system_behavior.retry_default_max_delay,
            "retry_enable_circuit_breaker": self.config.system_behavior.retry_enable_circuit_breaker,
            "rate_limiting_adaptive_adjustment": self.config.system_behavior.rate_limiting_adaptive_adjustment,
            "rate_limiting_conservative_mode": self.config.system_behavior.rate_limiting_conservative_mode,
            "progress_dashboard_type": self.config.system_behavior.progress_dashboard_type,
            "progress_update_interval": self.config.system_behavior.progress_update_interval,
            "progress_enable_web_dashboard": self.config.system_behavior.progress_enable_web_dashboard,
            "progress_web_port": self.config.system_behavior.progress_web_port,
            "estimation_db_max_records": self.config.system_behavior.estimation_db_max_records,
            "estimation_db_cleanup_days": self.config.system_behavior.estimation_db_cleanup_days,
        }


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config(config_file: Optional[str] = None) -> HiRAGConfig:
    """
    Get the global configuration instance

    Args:
        config_file: Optional path to configuration file. Only used on first call.

    Returns:
        HiRAGConfig: The global configuration instance
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_file)

    return _config_manager.config


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance

    Args:
        config_file: Optional path to configuration file. Only used on first call.

    Returns:
        ConfigManager: The global configuration manager instance
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_file)

    return _config_manager


def reload_config(config_file: Optional[str] = None):
    """
    Reload the configuration from file

    Args:
        config_file: Optional path to configuration file
    """
    global _config_manager
    _config_manager = ConfigManager(config_file)
    logger.info("Configuration reloaded")


# Convenience functions for common configuration access patterns
def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_config().api


def get_text_processing_config() -> TextProcessingConfig:
    """Get text processing configuration"""
    return get_config().text_processing


def get_entity_processing_config() -> EntityProcessingConfig:
    """Get entity processing configuration"""
    return get_config().entity_processing


def get_graph_operations_config() -> GraphOperationsConfig:
    """Get graph operations configuration"""
    return get_config().graph_operations


def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return get_config().performance


def get_quality_control_config() -> QualityControlConfig:
    """Get quality control configuration"""
    return get_config().quality_control


def get_system_behavior_config() -> SystemBehaviorConfig:
    """Get system behavior configuration"""
    return get_config().system_behavior

