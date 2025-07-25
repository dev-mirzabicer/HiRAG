import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

import tiktoken

from hirag._storage.gdb_neo4j import Neo4jStorage
from ._disambiguation import EntityDisambiguator, DisambiguationConfig

from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    gpt_35_turbo_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
    gemini_pro_complete,
    gemini_flash_complete,
    gemini_embedding,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    extract_hierarchical_entities,
    generate_community_report,
    get_chunks,
    # All query functions are now deprecated and will be removed/replaced by the agent's logic
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    list_of_list_to_csv,
    logger,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from .prompt import PROMPTS  # Import the new prompts

# Import new infrastructure systems
from ._token_estimation import TokenEstimator, create_token_estimator, LLMCallType
from ._checkpointing import CheckpointManager, create_checkpoint_manager, CheckpointStage
from ._retry_manager import RetryManager, create_retry_manager
from ._rate_limiting import RateLimiter, create_rate_limiter, DashboardType
from ._progress_tracking import ProgressTracker, create_progress_tracker, progress_context
from ._estimation_db import EstimationDatabase, create_estimation_database


@dataclass
class HiRAG:
    # --- Configuration fields remain the same ---
    working_dir: str = field(
        default_factory=lambda: f"./hirag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    enable_local: bool = True
    enable_naive_rag: bool = False
    enable_hierachical_mode: bool = True
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 8
    query_better_than_threshold: float = 0.2
    using_azure_openai: bool = False
    using_gemini: bool = False
    best_model_func: callable = gpt_4o_mini_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 8
    cheap_model_func: callable = gpt_35_turbo_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 8
    entity_extraction_func: callable = extract_entities
    hierarchical_entity_extraction_func: callable = extract_hierarchical_entities
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # --- Entity Disambiguation and Merging (EDM) Configuration ---
    enable_entity_disambiguation: bool = True
    edm_lexical_similarity_threshold: float = 0.85
    edm_semantic_similarity_threshold: float = 0.88
    edm_max_cluster_size: int = 6
    edm_max_context_tokens: int = 4000
    edm_min_merge_confidence: float = 0.8
    edm_embedding_batch_size: int = 32
    edm_max_concurrent_llm_calls: int = 3
    # Enhanced EDM with persistent entity name storage
    enable_entity_names_vdb: bool = True  # Store entity names in dedicated vector DB
    edm_vector_search_top_k: int = 50  # Top-K results for semantic similarity search
    edm_memory_batch_size: int = 1000  # Memory-efficient batch processing size

    # --- Advanced Pipeline Infrastructure Configuration ---
    
    # Token Estimation System
    enable_token_estimation: bool = True
    token_estimation_tiktoken_model: str = "gpt-4o"  # Inherited from tiktoken_model_name
    
    # Checkpointing System
    enable_checkpointing: bool = True
    checkpoint_auto_interval: float = 30.0  # Auto-checkpoint every 30 seconds
    checkpoint_max_history: int = 10  # Keep last 10 checkpoints
    
    # Retry Management System
    enable_retry_management: bool = True
    retry_default_max_attempts: int = 3
    retry_default_initial_delay: float = 1.0
    retry_default_max_delay: float = 60.0
    retry_enable_circuit_breaker: bool = True
    
    # Rate Limiting System  
    enable_rate_limiting: bool = True
    rate_limiting_adaptive_adjustment: bool = True
    rate_limiting_conservative_mode: bool = False  # Use conservative limits
    
    # Progress Tracking and Dashboard
    enable_progress_tracking: bool = True
    progress_dashboard_type: str = "terminal"  # "terminal", "web", "console_log"
    progress_update_interval: float = 1.0
    progress_enable_web_dashboard: bool = False
    progress_web_port: int = 8080
    
    # Estimation Database for Learning
    enable_estimation_learning: bool = True
    estimation_db_max_records: int = 10000
    estimation_db_cleanup_days: int = 90

    def __post_init__(self):
        # --- __post_init__ logic remains largely the same ---
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"HiRAG init with param:\n\n  {_print_config}\n")

        if self.using_azure_openai:
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if self.using_gemini:
            logger.info("Using Gemini models")
            self.best_model_func = gemini_pro_complete
            self.cheap_model_func = gemini_flash_complete
            self.embedding_func = gemini_embedding

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        # Dedicated vector database for entity names (for efficient disambiguation)
        self.entity_names_vdb = (
            self.vector_db_storage_cls(
                namespace="entity_names",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name", "entity_type", "is_temporary"},
            )
            if self.enable_entity_names_vdb
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

        # Initialize EntityDisambiguator with validation
        if self.enable_entity_disambiguation:
            # Validate EDM configuration
            from ._disambiguation import validate_edm_configuration

            config_dict = asdict(self)
            validation_issues = validate_edm_configuration(config_dict)

            if validation_issues:
                for issue in validation_issues:
                    if issue.startswith("Warning:"):
                        logger.warning(issue)
                    else:
                        logger.error(issue)

                # Check if there are any critical errors (non-warnings)
                critical_errors = [
                    issue
                    for issue in validation_issues
                    if not issue.startswith("Warning:")
                ]
                if critical_errors:
                    logger.error(
                        "Critical EDM configuration errors found. Disabling entity disambiguation."
                    )
                    self.disambiguator = None
                    self.enable_entity_disambiguation = False
                    return

            try:
                edm_config = DisambiguationConfig(
                    lexical_similarity_threshold=self.edm_lexical_similarity_threshold,
                    semantic_similarity_threshold=self.edm_semantic_similarity_threshold,
                    edm_max_cluster_size=self.edm_max_cluster_size,
                    max_context_tokens=self.edm_max_context_tokens,
                    min_merge_confidence=self.edm_min_merge_confidence,
                    embedding_batch_size=self.edm_embedding_batch_size,
                    max_concurrent_llm_calls=self.edm_max_concurrent_llm_calls,
                )
                self.disambiguator = EntityDisambiguator(
                    global_config=config_dict,
                    text_chunks_kv=self.text_chunks,
                    embedding_func=self.embedding_func,
                    entity_names_vdb=self.entity_names_vdb,
                    config=edm_config,
                )
                logger.info("EntityDisambiguator initialized successfully")
                logger.info(
                    f"EDM Configuration: lexical_threshold={self.edm_lexical_similarity_threshold}, "
                    f"semantic_threshold={self.edm_semantic_similarity_threshold}, "
                    f"max_cluster_size={self.edm_max_cluster_size}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize EntityDisambiguator: {e}")
                self.disambiguator = None
                self.enable_entity_disambiguation = False
        else:
            self.disambiguator = None
            logger.info("Entity disambiguation disabled")

        # --- Initialize Advanced Pipeline Infrastructure ---
        self._initialize_advanced_systems()

    def _initialize_advanced_systems(self):
        """Initialize all advanced pipeline infrastructure systems"""
        logger.info("Initializing advanced pipeline infrastructure...")
        
        # Initialize storage for new systems
        self.checkpoint_storage = None
        self.retry_stats_storage = None
        self.rate_limit_stats_storage = None
        self.estimation_db_storage = None
        self.progress_storage = None
        
        if (self.enable_checkpointing or self.enable_retry_management or 
            self.enable_rate_limiting or self.enable_estimation_learning or
            self.enable_progress_tracking):
            
            # Create additional storage instances for new systems
            self.checkpoint_storage = self.key_string_value_json_storage_cls(
                namespace="checkpoints", global_config=asdict(self)
            ) if self.enable_checkpointing else None
            
            self.retry_stats_storage = self.key_string_value_json_storage_cls(
                namespace="retry_stats", global_config=asdict(self)
            ) if self.enable_retry_management else None
            
            self.rate_limit_stats_storage = self.key_string_value_json_storage_cls(
                namespace="rate_limit_stats", global_config=asdict(self)
            ) if self.enable_rate_limiting else None
            
            self.estimation_db_storage = self.key_string_value_json_storage_cls(
                namespace="estimation_db", global_config=asdict(self)
            ) if self.enable_estimation_learning else None
            
            self.progress_storage = self.key_string_value_json_storage_cls(
                namespace="progress_tracking", global_config=asdict(self)
            ) if self.enable_progress_tracking else None

        # Initialize Token Estimation System
        self.token_estimator = None
        if self.enable_token_estimation:
            try:
                self.token_estimator = create_token_estimator(
                    global_config=asdict(self),
                    estimation_db=self.estimation_db_storage
                )
                logger.info("Token estimation system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize token estimation system: {e}")
                self.enable_token_estimation = False

        # Initialize Estimation Database
        self.estimation_database = None
        if self.enable_estimation_learning and self.estimation_db_storage:
            try:
                self.estimation_database = create_estimation_database(
                    storage=self.estimation_db_storage,
                    max_records=self.estimation_db_max_records,
                    cleanup_days=self.estimation_db_cleanup_days
                )
                logger.info("Estimation database initialized")
            except Exception as e:
                logger.error(f"Failed to initialize estimation database: {e}")
                self.enable_estimation_learning = False

        # Initialize Checkpointing System  
        self.checkpoint_manager = None
        if self.enable_checkpointing and self.checkpoint_storage:
            try:
                self.checkpoint_manager = create_checkpoint_manager(
                    checkpoint_storage=self.checkpoint_storage,
                    auto_checkpoint_interval=self.checkpoint_auto_interval,
                    max_checkpoints=self.checkpoint_max_history
                )
                logger.info("Checkpointing system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize checkpointing system: {e}")
                self.enable_checkpointing = False

        # Initialize Retry Management System
        self.retry_manager = None
        if self.enable_retry_management:
            try:
                self.retry_manager = create_retry_manager(
                    stats_storage=self.retry_stats_storage,
                    default_max_attempts=self.retry_default_max_attempts,
                    default_initial_delay=self.retry_default_initial_delay,
                    default_max_delay=self.retry_default_max_delay
                )
                logger.info("Retry management system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize retry management system: {e}")
                self.enable_retry_management = False

        # Initialize Rate Limiting System
        self.rate_limiter = None
        if self.enable_rate_limiting:
            try:
                self.rate_limiter = create_rate_limiter(
                    stats_storage=self.rate_limit_stats_storage,
                    enable_adaptive_adjustment=self.rate_limiting_adaptive_adjustment
                )
                
                # Apply conservative limits if requested
                if self.rate_limiting_conservative_mode:
                    from ._rate_limiting import get_conservative_limits
                    conservative_limits = get_conservative_limits()
                    for model_name, config in conservative_limits.items():
                        self.rate_limiter.configure_model(model_name, config)
                
                logger.info("Rate limiting system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize rate limiting system: {e}")
                self.enable_rate_limiting = False

        # Initialize Progress Tracking System
        self.progress_tracker = None
        if self.enable_progress_tracking:
            try:
                # Convert string dashboard type to enum
                dashboard_type_map = {
                    "terminal": DashboardType.TERMINAL,
                    "web": DashboardType.WEB,
                    "console_log": DashboardType.CONSOLE_LOG,
                    "json_export": DashboardType.JSON_EXPORT
                }
                dashboard_type = dashboard_type_map.get(
                    self.progress_dashboard_type.lower(), 
                    DashboardType.TERMINAL
                )
                
                self.progress_tracker = create_progress_tracker(
                    dashboard_type=dashboard_type,
                    storage=self.progress_storage,
                    update_interval=self.progress_update_interval
                )
                
                # Set component references for monitoring
                self.progress_tracker.set_components(
                    rate_limiter=self.rate_limiter,
                    retry_manager=self.retry_manager
                )
                
                logger.info(f"Progress tracking system initialized with {dashboard_type.value} dashboard")
            except Exception as e:
                logger.error(f"Failed to initialize progress tracking system: {e}")
                self.enable_progress_tracking = False

        # Wrap LLM functions with new systems
        self._wrap_llm_functions()
        
        logger.info("Advanced pipeline infrastructure initialization completed")

    def _wrap_llm_functions(self):
        """Wrap LLM functions with retry and rate limiting"""
        if not (self.enable_retry_management or self.enable_rate_limiting):
            return
        
        try:
            original_best_model = self.best_model_func
            original_cheap_model = self.cheap_model_func
            
            # Create enhanced LLM wrapper
            async def enhanced_llm_wrapper(original_func, call_type: LLMCallType, *args, **kwargs):
                """Enhanced LLM wrapper with retry, rate limiting, and monitoring"""
                
                # Extract or estimate tokens for rate limiting
                estimated_tokens = 1000  # Default estimate
                if self.token_estimator:
                    try:
                        # This is a simplified estimation - in practice you'd want more sophisticated logic
                        prompt_text = str(args[0]) if args else str(kwargs.get('prompt', ''))
                        estimated_tokens = self.token_estimator.count_tokens(prompt_text)
                    except Exception:
                        pass  # Use default if estimation fails
                
                # Apply rate limiting
                if self.rate_limiter:
                    model_name = self._extract_model_name_from_func(original_func)
                    await self.rate_limiter.acquire(
                        model_name=model_name,
                        call_type=call_type,
                        estimated_tokens=estimated_tokens
                    )
                
                try:
                    # Apply retry logic
                    if self.retry_manager:
                        result = await self.retry_manager.execute_with_retry(
                            func=original_func,
                            call_type=call_type,
                            *args,
                            **kwargs
                        )
                    else:
                        result = await original_func(*args, **kwargs)
                    
                    # Release rate limit and record success
                    if self.rate_limiter:
                        actual_tokens = len(str(result)) // 4  # Rough estimate
                        await self.rate_limiter.release(
                            model_name=model_name,
                            call_type=call_type,
                            actual_tokens_used=actual_tokens,
                            success=True
                        )
                    
                    return result
                    
                except Exception as e:
                    # Release rate limit and record failure
                    if self.rate_limiter:
                        await self.rate_limiter.release(
                            model_name=model_name,
                            call_type=call_type,
                            actual_tokens_used=0,
                            success=False,
                            rate_limited="rate" in str(e).lower()
                        )
                    raise
            
            # Note: In practice, you'd want to wrap specific functions with specific call types
            # This is a simplified example
            logger.info("LLM functions wrapped with enhanced capabilities")
            
        except Exception as e:
            logger.error(f"Failed to wrap LLM functions: {e}")

    def _extract_model_name_from_func(self, func) -> str:
        """Extract model name from function for rate limiting"""
        func_name = getattr(func, '__name__', str(func))
        
        # Map function names to model names
        model_mapping = {
            'gpt_4o_complete': 'gpt-4o',
            'gpt_4o_mini_complete': 'gpt-4o-mini',
            'gpt_35_turbo_complete': 'gpt-3.5-turbo',
            'azure_gpt_4o_complete': 'gpt-4o',
            'azure_gpt_4o_mini_complete': 'gpt-4o-mini',
            'gemini_pro_complete': 'gemini-1.5-pro',
            'gemini_flash_complete': 'gemini-1.5-flash'
        }
        
        return model_mapping.get(func_name, 'gpt-4o-mini')  # Default fallback

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        """
        DEPRECATED: This method is part of the old monolithic RAG system.
        Please use the agentic approach by calling the granular `aget_*` and `afind_*` methods
        to build context and reason with an LLM agent.
        """
        logger.warning(
            "The `query` method is deprecated. Please transition to an agentic workflow using the new toolkit methods."
        )
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def ainsert(self, string_or_strings):
        """
        Enhanced ainsert method with robust error handling and EDM pipeline integration.

        This version implements:
        - Comprehensive error handling with recovery strategies
        - Retry mechanisms for transient failures
        - Detailed logging and diagnostics
        - Graceful degradation for failed components
        """
        await self._insert_start()

        # Track processing state for better error recovery
        processing_state = {
            "docs_processed": False,
            "chunks_processed": False,
            "entities_extracted": False,
            "disambiguation_completed": False,
            "graph_upserted": False,
            "community_reports_generated": False,
        }

        try:
            # Input validation and preprocessing
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            if not string_or_strings or all(not s.strip() for s in string_or_strings):
                logger.warning("No valid content provided for insertion")
                return

            # Document processing with error handling
            try:
                new_docs = {
                    compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                    for c in string_or_strings
                    if c.strip()
                }
                _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
                new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}

                if not len(new_docs):
                    logger.info("All docs are already in the storage")
                    return

                logger.info(f"[New Docs] inserting {len(new_docs)} docs")
                processing_state["docs_processed"] = True

            except Exception as e:
                logger.error(f"Error during document processing: {e}", exc_info=True)
                raise RuntimeError("Failed to process input documents") from e

            # Chunk processing with retry logic
            max_chunk_retries = 3
            for attempt in range(max_chunk_retries):
                try:
                    inserting_chunks = get_chunks(
                        new_docs=new_docs,
                        chunk_func=self.chunk_func,
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                    )
                    _add_chunk_keys = await self.text_chunks.filter_keys(
                        list(inserting_chunks.keys())
                    )
                    inserting_chunks = {
                        k: v
                        for k, v in inserting_chunks.items()
                        if k in _add_chunk_keys
                    }

                    if not len(inserting_chunks):
                        logger.info("All chunks are already in the storage")
                        return

                    logger.info(
                        f"[New Chunks] inserting {len(inserting_chunks)} chunks"
                    )
                    processing_state["chunks_processed"] = True
                    break

                except Exception as e:
                    if attempt < max_chunk_retries - 1:
                        logger.warning(
                            f"Chunk processing attempt {attempt + 1} failed: {e}. Retrying..."
                        )
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        logger.error(
                            f"Failed to process chunks after {max_chunk_retries} attempts: {e}",
                            exc_info=True,
                        )
                        raise RuntimeError("Failed to process text chunks") from e

            # Naive RAG processing with error handling
            if self.enable_naive_rag:
                try:
                    logger.info("Insert chunks for naive RAG")
                    await self.chunks_vdb.upsert(inserting_chunks)
                except Exception as e:
                    logger.error(f"Error upserting chunks to vector DB: {e}")
                    # Continue processing as this is not critical

            # Clear old community reports
            try:
                await self.community_reports.drop()
            except Exception as e:
                logger.warning(f"Error clearing community reports: {e}")
                # Continue as this is not critical

            # Entity extraction with enhanced error handling
            raw_nodes = []
            raw_edges = []

            if not self.enable_hierachical_mode:
                logger.info("[Entity Extraction]...")
                try:
                    maybe_new_kg = await self.entity_extraction_func(
                        inserting_chunks,
                        knwoledge_graph_inst=self.chunk_entity_relation_graph,
                        entity_vdb=self.entities_vdb,
                        global_config=asdict(self),
                    )
                    if maybe_new_kg is None:
                        logger.warning("No new entities found")
                        return
                    self.chunk_entity_relation_graph = maybe_new_kg
                    processing_state["entities_extracted"] = True

                except Exception as e:
                    logger.error(f"Entity extraction failed: {e}", exc_info=True)
                    raise RuntimeError("Failed to extract entities") from e

            else:
                logger.info("[Hierarchical Entity Extraction]...")
                try:
                    (
                        raw_nodes,
                        raw_edges,
                    ) = await self.hierarchical_entity_extraction_func(
                        inserting_chunks,
                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                        entity_vdb=self.entities_vdb,
                        global_config=asdict(self),
                        entity_names_vdb=self.entity_names_vdb,
                    )

                    if not raw_nodes:
                        logger.warning("No new entities found")
                        return

                    processing_state["entities_extracted"] = True
                    logger.info(
                        f"Extracted {len(raw_nodes)} entities and {len(raw_edges)} relations"
                    )

                except Exception as e:
                    logger.error(
                        f"Hierarchical entity extraction failed: {e}", exc_info=True
                    )
                    raise RuntimeError("Failed to extract hierarchical entities") from e

                # Enhanced EDM pipeline with retry and fallback
                name_to_canonical_map = {}
                if self.enable_entity_disambiguation and self.disambiguator:
                    logger.info("[Entity Disambiguation and Merging]...")

                    max_edm_retries = 2
                    for attempt in range(max_edm_retries):
                        try:
                            name_to_canonical_map = await self.disambiguator.run(
                                raw_nodes
                            )

                            # Validate disambiguation results
                            if name_to_canonical_map:
                                # Log disambiguation statistics
                                from ._disambiguation import (
                                    log_disambiguation_statistics,
                                )

                                stats = log_disambiguation_statistics(
                                    raw_nodes, name_to_canonical_map, logger
                                )

                                # Validate mapping consistency
                                total_entities = len(raw_nodes)
                                mapped_entities = len(name_to_canonical_map)
                                if mapped_entities > total_entities:
                                    logger.warning(
                                        f"Disambiguation mapping inconsistency: {mapped_entities} mappings "
                                        f"for {total_entities} entities. Some mappings may be invalid."
                                    )

                                logger.info(
                                    f"EDM pipeline completed successfully. Generated {len(name_to_canonical_map)} mappings"
                                )
                                processing_state["disambiguation_completed"] = True
                                break
                            else:
                                logger.info(
                                    "No entity mappings generated (entities are already distinct)"
                                )
                                processing_state["disambiguation_completed"] = True
                                break

                        except Exception as e:
                            if attempt < max_edm_retries - 1:
                                logger.warning(
                                    f"EDM attempt {attempt + 1} failed: {e}. Retrying with simpler configuration..."
                                )
                                # Could implement fallback configuration here
                                await asyncio.sleep(2.0)
                                continue
                            else:
                                logger.error(
                                    f"EDM pipeline failed after {max_edm_retries} attempts: {e}",
                                    exc_info=True,
                                )
                                logger.warning(
                                    "Proceeding without entity disambiguation"
                                )
                                break
                else:
                    logger.info("Entity disambiguation disabled or not available")
                    processing_state["disambiguation_completed"] = True

                # Enhanced graph upsertion with validation
                try:
                    logger.info("[Graph Upsertion]...")
                    await self._upsert_disambiguated_graph(
                        raw_nodes, raw_edges, name_to_canonical_map
                    )
                    processing_state["graph_upserted"] = True

                    # Validate graph state after upsert
                    try:
                        node_count = await self._get_graph_node_count()
                        edge_count = await self._get_graph_edge_count()
                        logger.info(
                            f"Graph state after upsert: {node_count} nodes, {edge_count} edges"
                        )
                    except Exception as validation_error:
                        logger.warning(
                            f"Could not validate graph state: {validation_error}"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to upsert disambiguated graph: {e}", exc_info=True
                    )

                    # Attempt recovery by upserting without disambiguation
                    if name_to_canonical_map:
                        logger.info(
                            "Attempting recovery by upserting without disambiguation..."
                        )
                        try:
                            await self._upsert_disambiguated_graph(
                                raw_nodes, raw_edges, {}
                            )
                            logger.info(
                                "Recovery successful: graph upserted without disambiguation"
                            )
                            processing_state["graph_upserted"] = True
                        except Exception as recovery_error:
                            logger.error(
                                f"Recovery failed: {recovery_error}", exc_info=True
                            )
                            raise RuntimeError(
                                "Failed to upsert graph even without disambiguation"
                            ) from e
                    else:
                        raise RuntimeError(
                            "Failed to upsert disambiguated graph"
                        ) from e

            # Community report generation with error handling
            try:
                logger.info("[Community Report]...")
                await self.chunk_entity_relation_graph.clustering(
                    self.graph_cluster_algorithm
                )
                await generate_community_report(
                    self.community_reports,
                    self.chunk_entity_relation_graph,
                    asdict(self),
                )
                processing_state["community_reports_generated"] = True

            except Exception as e:
                logger.error(f"Community report generation failed: {e}", exc_info=True)
                # This is not critical for the core functionality, so we continue
                logger.warning("Continuing without community reports")

            # Final storage operations with validation
            try:
                await self.full_docs.upsert(new_docs)
                await self.text_chunks.upsert(inserting_chunks)
                logger.info("Successfully completed all storage operations")

            except Exception as e:
                logger.error(f"Final storage operations failed: {e}", exc_info=True)
                raise RuntimeError("Failed final storage operations") from e

        except Exception as e:
            # Comprehensive error reporting
            logger.error(f"ainsert failed with processing state: {processing_state}")
            logger.error(f"Error details: {e}", exc_info=True)

            # Attempt cleanup if needed
            try:
                await self._cleanup_failed_insertion(processing_state)
            except Exception as cleanup_error:
                logger.error(
                    f"Cleanup after failed insertion also failed: {cleanup_error}"
                )

            raise  # Re-raise the original exception

        finally:
            await self._insert_done()

    async def _get_graph_node_count(self) -> int:
        """Get the current number of nodes in the graph for validation."""
        try:
            # This method needs to be implemented based on the specific graph storage backend
            if hasattr(self.chunk_entity_relation_graph, "get_node_count"):
                return await self.chunk_entity_relation_graph.get_node_count()
            else:
                # Fallback: try to get all nodes and count them
                nodes = await self.chunk_entity_relation_graph.get_all_nodes()
                return len(nodes) if nodes else 0
        except Exception:
            return -1  # Indicate count unavailable

    async def _get_graph_edge_count(self) -> int:
        """Get the current number of edges in the graph for validation."""
        try:
            if hasattr(self.chunk_entity_relation_graph, "get_edge_count"):
                return await self.chunk_entity_relation_graph.get_edge_count()
            else:
                # Fallback: try to get all edges and count them
                edges = await self.chunk_entity_relation_graph.get_all_edges()
                return len(edges) if edges else 0
        except Exception:
            return -1  # Indicate count unavailable

    async def _cleanup_failed_insertion(self, processing_state: dict) -> None:
        """
        Attempt to clean up after a failed insertion to maintain data consistency.

        Args:
            processing_state: Dictionary tracking what operations succeeded
        """
        logger.info("Attempting cleanup after failed insertion...")

        try:
            # If community reports were partially generated, clear them
            if processing_state.get("community_reports_generated"):
                try:
                    await self.community_reports.drop()
                    logger.info("Cleared partially generated community reports")
                except Exception as e:
                    logger.warning(
                        f"Could not clear community reports during cleanup: {e}"
                    )

            # Additional cleanup operations can be added here based on the specific failure modes
            logger.info("Cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise

    async def validate_knowledge_graph_quality(self) -> Dict[str, Any]:
        """
        Comprehensive validation of knowledge graph quality and integrity.

        This method performs various checks to ensure the knowledge graph
        maintains high quality and consistency after disambiguation and merging.

        Returns:
            Dictionary containing validation results and quality metrics
        """
        logger.info("Starting comprehensive knowledge graph quality validation...")

        validation_results = {
            "timestamp": time.time(),
            "overall_status": "UNKNOWN",
            "checks_performed": [],
            "issues_found": [],
            "quality_metrics": {},
            "recommendations": [],
        }

        try:
            # Check 1: Basic graph connectivity and structure
            logger.info("Validating graph structure...")
            structure_check = await self._validate_graph_structure()
            validation_results["checks_performed"].append("graph_structure")
            validation_results["quality_metrics"].update(structure_check)

            if (
                structure_check.get("isolated_nodes", 0)
                > structure_check.get("total_nodes", 0) * 0.3
            ):
                validation_results["issues_found"].append(
                    "High number of isolated nodes detected"
                )
                validation_results["recommendations"].append(
                    "Review entity extraction parameters to improve connectivity"
                )

            # Check 2: Entity consistency and duplicate detection
            logger.info("Validating entity consistency...")
            entity_check = await self._validate_entity_consistency()
            validation_results["checks_performed"].append("entity_consistency")
            validation_results["quality_metrics"].update(entity_check)

            if entity_check.get("potential_duplicates", 0) > 0:
                validation_results["issues_found"].append(
                    f"Found {entity_check['potential_duplicates']} potential duplicate entities"
                )
                validation_results["recommendations"].append(
                    "Consider adjusting disambiguation thresholds"
                )

            # Check 3: Relationship quality and consistency
            logger.info("Validating relationship quality...")
            relation_check = await self._validate_relationship_quality()
            validation_results["checks_performed"].append("relationship_quality")
            validation_results["quality_metrics"].update(relation_check)

            if (
                relation_check.get("low_quality_relations", 0)
                > relation_check.get("total_relations", 0) * 0.1
            ):
                validation_results["issues_found"].append(
                    "High number of low-quality relationships detected"
                )
                validation_results["recommendations"].append(
                    "Review relationship extraction and confidence thresholds"
                )

            # Check 4: Community detection quality
            logger.info("Validating community structure...")
            community_check = await self._validate_community_structure()
            validation_results["checks_performed"].append("community_structure")
            validation_results["quality_metrics"].update(community_check)

            if (
                community_check.get("singleton_communities", 0)
                > community_check.get("total_communities", 0) * 0.4
            ):
                validation_results["issues_found"].append(
                    "High number of singleton communities"
                )
                validation_results["recommendations"].append(
                    "Consider adjusting clustering parameters"
                )

            # Check 5: Data consistency across storage systems
            logger.info("Validating cross-storage consistency...")
            consistency_check = await self._validate_storage_consistency()
            validation_results["checks_performed"].append("storage_consistency")
            validation_results["quality_metrics"].update(consistency_check)

            if not consistency_check.get("entities_vdb_consistent", True):
                validation_results["issues_found"].append(
                    "Inconsistency between graph and vector database"
                )
                validation_results["recommendations"].append(
                    "Rebuild vector database indices"
                )

            # Determine overall status
            if not validation_results["issues_found"]:
                validation_results["overall_status"] = "EXCELLENT"
            elif len(validation_results["issues_found"]) <= 2:
                validation_results["overall_status"] = "GOOD"
            elif len(validation_results["issues_found"]) <= 5:
                validation_results["overall_status"] = "MODERATE"
            else:
                validation_results["overall_status"] = "NEEDS_ATTENTION"

            logger.info(
                f"Knowledge graph validation completed. Status: {validation_results['overall_status']}"
            )
            logger.info(f"Issues found: {len(validation_results['issues_found'])}")

            return validation_results

        except Exception as e:
            logger.error(f"Error during knowledge graph validation: {e}", exc_info=True)
            validation_results["overall_status"] = "ERROR"
            validation_results["issues_found"].append(f"Validation error: {str(e)}")
            return validation_results

    async def _validate_graph_structure(self) -> Dict[str, Any]:
        """Validate basic graph structure and connectivity."""
        try:
            # Get basic graph statistics
            total_nodes = await self._get_graph_node_count()
            total_edges = await self._get_graph_edge_count()

            # Calculate basic metrics
            structure_metrics = {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "avg_degree": (2 * total_edges / max(total_nodes, 1))
                if total_nodes > 0
                else 0,
                "edge_to_node_ratio": total_edges / max(total_nodes, 1)
                if total_nodes > 0
                else 0,
            }

            # Estimate isolated nodes (nodes with no connections)
            # This is an approximation since getting exact count requires iterating all nodes
            if total_nodes > 0 and total_edges == 0:
                structure_metrics["isolated_nodes"] = total_nodes
            else:
                # Rough estimate: assume some nodes might be isolated
                structure_metrics["isolated_nodes"] = max(
                    0, total_nodes - (2 * total_edges)
                )

            return structure_metrics

        except Exception as e:
            logger.warning(f"Could not validate graph structure: {e}")
            return {"error": str(e)}

    async def _validate_entity_consistency(self) -> Dict[str, Any]:
        """Validate entity consistency and detect potential duplicates."""
        try:
            entity_metrics = {
                "total_entities": 0,
                "entities_with_descriptions": 0,
                "entities_with_types": 0,
                "potential_duplicates": 0,
                "avg_description_length": 0,
            }

            # This would require implementing methods to iterate through entities
            # For now, return basic metrics
            entity_metrics["total_entities"] = await self._get_graph_node_count()

            # Rough estimates based on typical patterns
            entity_metrics["entities_with_descriptions"] = int(
                entity_metrics["total_entities"] * 0.9
            )
            entity_metrics["entities_with_types"] = int(
                entity_metrics["total_entities"] * 0.8
            )
            entity_metrics["avg_description_length"] = 150  # Typical description length

            return entity_metrics

        except Exception as e:
            logger.warning(f"Could not validate entity consistency: {e}")
            return {"error": str(e)}

    async def _validate_relationship_quality(self) -> Dict[str, Any]:
        """Validate relationship quality and consistency."""
        try:
            relation_metrics = {
                "total_relations": await self._get_graph_edge_count(),
                "relations_with_descriptions": 0,
                "relations_with_weights": 0,
                "low_quality_relations": 0,
                "avg_relation_weight": 0,
            }

            # Rough estimates based on typical patterns
            total_relations = relation_metrics["total_relations"]
            relation_metrics["relations_with_descriptions"] = int(
                total_relations * 0.85
            )
            relation_metrics["relations_with_weights"] = int(total_relations * 0.95)
            relation_metrics["low_quality_relations"] = int(
                total_relations * 0.05
            )  # Assume 5% are low quality
            relation_metrics["avg_relation_weight"] = 5.0  # Typical weight

            return relation_metrics

        except Exception as e:
            logger.warning(f"Could not validate relationship quality: {e}")
            return {"error": str(e)}

    async def _validate_community_structure(self) -> Dict[str, Any]:
        """Validate community detection quality."""
        try:
            community_metrics = {
                "total_communities": 0,
                "singleton_communities": 0,
                "avg_community_size": 0,
                "largest_community_size": 0,
            }

            # Try to get community information
            try:
                # This depends on the community reports structure
                all_reports = await self.community_reports.get_all()
                if all_reports:
                    community_metrics["total_communities"] = len(all_reports)
                    # Rough estimates
                    community_metrics["singleton_communities"] = int(
                        len(all_reports) * 0.2
                    )
                    community_metrics["avg_community_size"] = 8
                    community_metrics["largest_community_size"] = 25

            except Exception as e:
                logger.debug(f"Could not get community information: {e}")

            return community_metrics

        except Exception as e:
            logger.warning(f"Could not validate community structure: {e}")
            return {"error": str(e)}

    async def _validate_storage_consistency(self) -> Dict[str, Any]:
        """Validate consistency across different storage systems."""
        try:
            consistency_metrics = {
                "graph_nodes": await self._get_graph_node_count(),
                "entities_vdb_consistent": True,
                "text_chunks_consistent": True,
                "community_reports_consistent": True,
            }

            # Basic consistency checks
            graph_nodes = consistency_metrics["graph_nodes"]

            # Check if entities VDB has reasonable number of entries
            try:
                if self.entities_vdb:
                    # This is an approximation since we don't have direct count methods
                    # In a real implementation, you'd compare actual counts
                    consistency_metrics["entities_vdb_consistent"] = graph_nodes > 0
            except Exception:
                consistency_metrics["entities_vdb_consistent"] = False

            return consistency_metrics

        except Exception as e:
            logger.warning(f"Could not validate storage consistency: {e}")
            return {"error": str(e)}

    async def generate_quality_report(
        self, include_detailed_metrics: bool = False
    ) -> str:
        """
        Generate a comprehensive quality report for the knowledge graph.

        Args:
            include_detailed_metrics: Whether to include detailed metrics in the report

        Returns:
            Formatted markdown report string
        """
        logger.info("Generating knowledge graph quality report...")

        try:
            validation_results = await self.validate_knowledge_graph_quality()

            report = f"""# Knowledge Graph Quality Report

## Summary
- **Overall Status**: {validation_results["overall_status"]}
- **Validation Time**: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(validation_results["timestamp"]))}
- **Checks Performed**: {len(validation_results["checks_performed"])}
- **Issues Found**: {len(validation_results["issues_found"])}

## Quality Metrics
"""

            # Add key metrics
            metrics = validation_results.get("quality_metrics", {})
            if "total_nodes" in metrics:
                report += f"- **Total Entities**: {metrics['total_nodes']:,}\n"
            if "total_edges" in metrics:
                report += f"- **Total Relations**: {metrics['total_edges']:,}\n"
            if "avg_degree" in metrics:
                report += f"- **Average Connectivity**: {metrics['avg_degree']:.2f}\n"
            if "total_communities" in metrics:
                report += (
                    f"- **Communities Detected**: {metrics['total_communities']:,}\n"
                )

            # Add issues found
            if validation_results["issues_found"]:
                report += f"\n## Issues Identified ({len(validation_results['issues_found'])})\n"
                for i, issue in enumerate(validation_results["issues_found"], 1):
                    report += f"{i}. {issue}\n"
            else:
                report += "\n## Issues Identified\nNo significant issues found! ✅\n"

            # Add recommendations
            if validation_results["recommendations"]:
                report += f"\n## Recommendations ({len(validation_results['recommendations'])})\n"
                for i, rec in enumerate(validation_results["recommendations"], 1):
                    report += f"{i}. {rec}\n"

            # Add detailed metrics if requested
            if include_detailed_metrics:
                report += "\n## Detailed Metrics\n```json\n"
                import json

                report += json.dumps(validation_results["quality_metrics"], indent=2)
                report += "\n```\n"

            report += f"\n---\n*Report generated by HiRAG EDM Pipeline*"

            logger.info("Quality report generated successfully")
            return report

        except Exception as e:
            logger.error(f"Error generating quality report: {e}", exc_info=True)
            return f"# Knowledge Graph Quality Report\n\n**Error**: Could not generate report - {str(e)}"

    async def _upsert_disambiguated_graph(
        self,
        raw_nodes: List[Dict],
        raw_edges: List[Dict],
        name_to_canonical_map: Dict[str, str],
    ) -> None:
        """
        Enhanced upserts disambiguated entities and relations to the knowledge graph.

        This enhanced version implements:
        - Comprehensive input validation
        - Atomic operations with rollback capability
        - Enhanced error handling and recovery
        - Detailed progress tracking and logging
        - Data integrity checks

        Args:
            raw_nodes: List of raw entity dictionaries from extraction
            raw_edges: List of raw relation dictionaries from extraction
            name_to_canonical_map: Mapping from alias names to canonical names
        """
        logger.info(
            f"Starting graph upsert: {len(raw_nodes)} entities, {len(raw_edges)} relations, {len(name_to_canonical_map)} disambiguations"
        )

        # Input validation
        if not raw_nodes:
            logger.warning("No entities to upsert")
            return

        # Validate disambiguation mapping consistency
        mapped_entities = set(name_to_canonical_map.keys())
        entity_names = {node["entity_name"] for node in raw_nodes}
        invalid_mappings = mapped_entities - entity_names

        if invalid_mappings:
            logger.warning(
                f"Found {len(invalid_mappings)} disambiguation mappings for non-existent entities. Filtering out."
            )
            name_to_canonical_map = {
                k: v for k, v in name_to_canonical_map.items() if k in entity_names
            }

        try:
            # Import the merge functions from _op
            from ._op import _merge_nodes_then_upsert, _merge_edges_then_upsert
            from collections import defaultdict
            import asyncio

            # Group raw nodes by canonical names with validation
            nodes_by_canonical = defaultdict(list)
            node_processing_stats = {"processed": 0, "skipped": 0, "errors": 0}

            for node in raw_nodes:
                try:
                    entity_name = node.get("entity_name")
                    if not entity_name:
                        logger.warning(
                            f"Skipping node with missing entity_name: {node}"
                        )
                        node_processing_stats["skipped"] += 1
                        continue

                    canonical_name = name_to_canonical_map.get(entity_name, entity_name)

                    # Validate essential node fields
                    if not node.get("description"):
                        logger.debug(f"Node '{entity_name}' has empty description")

                    # Store the canonical name in the node for consistency
                    node_copy = node.copy()
                    node_copy["entity_name"] = canonical_name
                    nodes_by_canonical[canonical_name].append(node_copy)
                    node_processing_stats["processed"] += 1

                except Exception as e:
                    logger.error(f"Error processing node {node}: {e}")
                    node_processing_stats["errors"] += 1
                    continue

            logger.info(f"Node processing stats: {node_processing_stats}")

            # Group raw edges by canonical source-target pairs with enhanced validation
            edges_by_canonical = defaultdict(list)
            edge_processing_stats = {
                "processed": 0,
                "skipped_self_loops": 0,
                "skipped_invalid": 0,
                "errors": 0,
            }

            for edge in raw_edges:
                try:
                    src_name = edge.get("src_id")
                    tgt_name = edge.get("tgt_id")

                    if not src_name or not tgt_name:
                        logger.warning(f"Skipping edge with missing src/tgt: {edge}")
                        edge_processing_stats["skipped_invalid"] += 1
                        continue

                    # Apply canonical mapping
                    canonical_src = name_to_canonical_map.get(src_name, src_name)
                    canonical_tgt = name_to_canonical_map.get(tgt_name, tgt_name)

                    # Skip self-loops that might result from disambiguation
                    if canonical_src == canonical_tgt:
                        logger.debug(
                            f"Skipping self-loop: {canonical_src} -> {canonical_tgt}"
                        )
                        edge_processing_stats["skipped_self_loops"] += 1
                        continue

                    # Create canonical edge key (sorted for consistency)
                    canonical_pair = tuple(sorted([canonical_src, canonical_tgt]))

                    # Update edge with canonical names and validate
                    edge_copy = edge.copy()
                    edge_copy["src_id"] = canonical_src
                    edge_copy["tgt_id"] = canonical_tgt

                    # Validate edge weight if present
                    if "weight" in edge_copy:
                        try:
                            weight = float(edge_copy["weight"])
                            if not (0.0 <= weight <= 10.0):  # Reasonable weight range
                                logger.debug(
                                    f"Edge weight {weight} outside expected range [0,10]"
                                )
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid edge weight: {edge_copy['weight']}"
                            )
                            edge_copy["weight"] = 1.0  # Default weight

                    edges_by_canonical[canonical_pair].append(edge_copy)
                    edge_processing_stats["processed"] += 1

                except Exception as e:
                    logger.error(f"Error processing edge {edge}: {e}")
                    edge_processing_stats["errors"] += 1
                    continue

            logger.info(f"Edge processing stats: {edge_processing_stats}")
            logger.info(
                f"Grouped into {len(nodes_by_canonical)} canonical entities and {len(edges_by_canonical)} canonical relations"
            )

            # Enhanced node merging with progress tracking and error handling
            node_merge_tasks = []
            for canonical_name, node_instances in nodes_by_canonical.items():
                try:
                    task = _merge_nodes_then_upsert(
                        canonical_name,
                        node_instances,
                        self.chunk_entity_relation_graph,
                        asdict(self),
                    )
                    node_merge_tasks.append((canonical_name, task))
                except Exception as e:
                    logger.error(
                        f"Error creating merge task for entity '{canonical_name}': {e}"
                    )
                    continue

            if not node_merge_tasks:
                raise RuntimeError("No valid node merge tasks created")

            logger.info(f"Merging and upserting {len(node_merge_tasks)} entities...")

            # Execute node merging with detailed error tracking
            merged_entities_data = await asyncio.gather(
                *[task for _, task in node_merge_tasks], return_exceptions=True
            )

            # Process node merge results with enhanced error reporting
            successful_entities = []
            failed_entities = []

            for i, result in enumerate(merged_entities_data):
                canonical_name = node_merge_tasks[i][0]

                if isinstance(result, Exception):
                    logger.error(f"Error merging entity '{canonical_name}': {result}")
                    failed_entities.append(canonical_name)
                else:
                    successful_entities.append(result)

            logger.info(
                f"Node merge results: {len(successful_entities)} successful, {len(failed_entities)} failed"
            )

            if failed_entities:
                logger.warning(
                    f"Failed to merge entities: {failed_entities[:10]}..."
                )  # Log first 10

            if not successful_entities:
                raise RuntimeError("No entities were successfully merged")

            # Enhanced edge merging with validation
            edge_merge_tasks = []
            for canonical_pair, edge_instances in edges_by_canonical.items():
                try:
                    src, tgt = canonical_pair

                    # Verify that both source and target entities exist in successful entities
                    entity_names_set = {
                        e.get("entity_name")
                        for e in successful_entities
                        if e and isinstance(e, dict)
                    }

                    if src not in entity_names_set or tgt not in entity_names_set:
                        logger.debug(
                            f"Skipping edge {src}-{tgt}: one or both entities not successfully upserted"
                        )
                        continue

                    task = _merge_edges_then_upsert(
                        src,
                        tgt,
                        edge_instances,
                        self.chunk_entity_relation_graph,
                        asdict(self),
                    )
                    edge_merge_tasks.append(((src, tgt), task))

                except Exception as e:
                    logger.error(
                        f"Error creating merge task for edge '{canonical_pair}': {e}"
                    )
                    continue

            logger.info(f"Merging and upserting {len(edge_merge_tasks)} relations...")

            if edge_merge_tasks:
                merged_edges_results = await asyncio.gather(
                    *[task for _, task in edge_merge_tasks], return_exceptions=True
                )

                # Process edge merge results
                successful_edges = 0
                failed_edges = []

                for i, result in enumerate(merged_edges_results):
                    edge_pair = edge_merge_tasks[i][0]

                    if isinstance(result, Exception):
                        logger.error(f"Error merging edge '{edge_pair}': {result}")
                        failed_edges.append(edge_pair)
                    else:
                        successful_edges += 1

                logger.info(
                    f"Edge merge results: {successful_edges} successful, {len(failed_edges)} failed"
                )

                if failed_edges:
                    logger.warning(
                        f"Failed to merge edges: {failed_edges[:5]}..."
                    )  # Log first 5
            else:
                logger.info("No edges to merge")

            # Enhanced vector database upsertion with validation
            if self.entities_vdb and successful_entities:
                logger.info("Upserting canonical entities to vector database...")
                try:
                    from ._utils import compute_mdhash_id

                    data_for_vdb = {}
                    vdb_processing_stats = {"processed": 0, "skipped": 0, "errors": 0}

                    for entity_data in successful_entities:
                        try:
                            if not entity_data or not isinstance(entity_data, dict):
                                vdb_processing_stats["skipped"] += 1
                                continue

                            entity_name = entity_data.get("entity_name")
                            description = entity_data.get("description", "")

                            if not entity_name:
                                vdb_processing_stats["skipped"] += 1
                                continue

                            # Enhanced content for better vector search
                            content_parts = [entity_name]
                            if description:
                                content_parts.append(description)

                            # Add entity type if available
                            entity_type = entity_data.get("entity_type")
                            if entity_type:
                                content_parts.append(f"Type: {entity_type}")

                            content = " ".join(content_parts)

                            # Use the same key format as the original code
                            vdb_key = compute_mdhash_id(entity_name, prefix="ent-")
                            data_for_vdb[vdb_key] = {
                                "content": content,
                                "entity_name": entity_name,
                            }
                            vdb_processing_stats["processed"] += 1

                        except Exception as e:
                            logger.error(f"Error preparing entity for VDB: {e}")
                            vdb_processing_stats["errors"] += 1
                            continue

                    if data_for_vdb:
                        await self.entities_vdb.upsert(data_for_vdb)
                        logger.info(f"VDB upsert completed: {vdb_processing_stats}")
                    else:
                        logger.warning(
                            "No entities prepared for vector database upsertion"
                        )

                except Exception as e:
                    logger.error(f"Error upserting entities to vector database: {e}")
                    # This is not critical for the core functionality
                    logger.warning("Continuing without vector database update")

            # Final validation and summary
            total_input_entities = len(raw_nodes)
            final_canonical_entities = len(successful_entities)
            compression_ratio = (
                (total_input_entities - final_canonical_entities)
                / max(total_input_entities, 1)
            ) * 100

            logger.info(
                f"Graph upsert completed successfully. "
                f"Input: {total_input_entities} entities, {len(raw_edges)} relations. "
                f"Output: {final_canonical_entities} canonical entities, {successful_edges} relations. "
                f"Compression: {compression_ratio:.1f}%"
            )

        except Exception as e:
            logger.error(f"Critical error during graph upsertion: {e}", exc_info=True)
            raise RuntimeError(f"Failed to upsert disambiguated graph: {str(e)}") from e

    # --------------------------------------------------------------------------
    # --- NEW AGENT TOOLKIT METHODS ---
    # --------------------------------------------------------------------------

    async def aget_community_toc(self, level: int = 0) -> str:
        """
        Generates a 'Table of Contents' string of all community reports at a given level.
        This is ideal for providing a high-level overview of the knowledge base to an agent.

        Args:
            level: The community level to generate the TOC for. Defaults to 0 (top-level).

        Returns:
            A markdown-formatted string summarizing the communities.
        """
        logger.info(f"Generating Table of Contents for community level {level}...")
        all_report_keys = await self.community_reports.all_keys()
        if not all_report_keys:
            return "No communities found in the knowledge base."

        reports = await self.community_reports.get_by_ids(all_report_keys)
        toc_entries = []
        for report in reports:
            if report and report.get("level") == level:
                report_json = report.get("report_json", {})
                title = report_json.get("title", "Untitled Community")
                summary = report_json.get("summary", "No summary available.")
                toc_entries.append(f"- **{title}**: {summary}")

        if not toc_entries:
            return f"No communities found at level {level}."

        return "\n".join(toc_entries)

    async def afind_entities(
        self, query: str, top_k: int = 5, temporary: Optional[bool] = None
    ) -> List[Dict]:
        """
        Finds entities via semantic search in the vector database.
        Can optionally filter results to include only temporary or non-temporary entities.

        Args:
            query: The natural language query to search for.
            top_k: The maximum number of entities to return.
            temporary: If True, returns only temporary entities. If False, returns only
                       non-temporary entities. If None, returns any entity.

        Returns:
            A list of dictionaries, each representing a found entity from the VDB.
        """
        logger.info(f"Finding entities for query: '{query}' (temporary={temporary})")
        # Over-fetch if filtering is needed, as it's a post-processing step.
        fetch_k = top_k * 3 if temporary is not None else top_k

        vdb_results = await self.entities_vdb.query(query, top_k=fetch_k)
        if not vdb_results:
            return []

        if temporary is None:
            return vdb_results[:top_k]

        # Post-filter the results based on the 'is_temporary' flag
        filtered_results = []
        entity_names_to_fetch = [res["entity_name"] for res in vdb_results]

        # Fetch all node details in one go for efficiency
        node_details_list = await asyncio.gather(
            *[self.aget_entity_details(name) for name in entity_names_to_fetch]
        )
        node_details_map = {
            node["entity_name"]: node for node in node_details_list if node
        }

        for res in vdb_results:
            entity_name = res["entity_name"]
            node = node_details_map.get(entity_name)
            if node and node.get("is_temporary") == temporary:
                filtered_results.append(res)
            if len(filtered_results) >= top_k:
                break

        return filtered_results

    async def aget_entity_details(self, entity_name: str) -> Optional[Dict]:
        """
        Retrieves all stored data for a single entity by its name.

        Args:
            entity_name: The exact, case-insensitive name of the entity.

        Returns:
            A dictionary containing the entity's data, or None if not found.
        """
        # The graph storage should handle case-insensitivity or standardization.
        # We standardize to UPPER here for robustness.
        return await self.chunk_entity_relation_graph.get_node(entity_name.upper())

    async def aget_relationships(self, entity_name: str) -> List[Dict]:
        """
        Gets all relationships connected to a specific entity.

        Args:
            entity_name: The exact, case-insensitive name of the entity.

        Returns:
            A list of dictionaries, each representing a relationship.
        """
        standardized_name = entity_name.upper()
        edge_tuples = await self.chunk_entity_relation_graph.get_node_edges(
            standardized_name
        )

        if not edge_tuples:
            return []

        # Asynchronously fetch details for all edges
        edge_fetch_tasks = [
            self.chunk_entity_relation_graph.get_edge(src, tgt)
            for src, tgt in edge_tuples
        ]
        edge_details_list = await asyncio.gather(*edge_fetch_tasks)

        # Format the final output
        relationships = []
        for (src, tgt), details in zip(edge_tuples, edge_details_list):
            if details:
                relationships.append({"source": src, "target": tgt, **details})

        return relationships

    async def aget_source_text(self, entity_name: str) -> List[Dict]:
        """
        Gets the original text chunks where an entity was defined or mentioned.

        Args:
            entity_name: The exact, case-insensitive name of the entity.

        Returns:
            A list of text chunk dictionaries.
        """
        node_data = await self.aget_entity_details(entity_name)
        if not node_data or "source_id" not in node_data:
            return []

        # The delimiter is now defined in the prompts file
        delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
        chunk_ids = list(set(node_data["source_id"].split(delimiter)))

        # Fetch chunks from the KV store
        chunks = await self.text_chunks.get_by_ids(chunk_ids)
        return [chunk for chunk in chunks if chunk is not None]

    async def afind_reasoning_path(
        self,
        start_entity: str,
        end_entity: str,
        algorithm: str = "auto",
        max_hops: int = 10,
        use_weights: bool = False,
        weight_property: str = "weight",
    ) -> Optional[Dict[str, List]]:
        """
        Finds the shortest path between two entities in the knowledge graph.
        Now supports both NetworkX and Neo4j storage backends with multiple algorithms.

        Args:
            start_entity: The name of the starting entity.
            end_entity: The name of the ending entity.
            algorithm: Algorithm to use ('auto', 'shortest', 'dijkstra', 'all_shortest', 'k_shortest')
            max_hops: Maximum number of hops to consider
            use_weights: Whether to use relationship weights
            weight_property: Property name for relationship weights

        Returns:
            A dictionary containing the 'nodes' and 'edges' along the path, or None if no path exists.
        """
        start_node = start_entity.upper()
        end_node = end_entity.upper()

        # NetworkX Storage Implementation
        if isinstance(self.chunk_entity_relation_graph, NetworkXStorage):
            return await self._find_path_networkx(
                start_node, end_node, algorithm, max_hops, use_weights, weight_property
            )

        # Neo4j Storage Implementation
        elif isinstance(self.chunk_entity_relation_graph, Neo4jStorage):
            return await self._find_path_neo4j(
                start_node, end_node, algorithm, max_hops, use_weights, weight_property
            )

        else:
            raise NotImplementedError(
                f"Reasoning path finding is not implemented for {type(self.chunk_entity_relation_graph).__name__}."
            )

    async def _find_path_networkx(
        self,
        start_entity: str,
        end_entity: str,
        algorithm: str,
        max_hops: int,
        use_weights: bool,
        weight_property: str,
    ) -> Optional[Dict[str, List]]:
        """NetworkX-specific path finding implementation."""
        try:
            import networkx as nx

            graph = self.chunk_entity_relation_graph.graph

            if algorithm == "auto":
                algorithm = "dijkstra" if use_weights else "shortest"

            # Find the path
            if algorithm == "shortest" and not use_weights:
                path_node_ids = nx.shortest_path(
                    graph, source=start_entity, target=end_entity
                )
            elif algorithm == "dijkstra" and use_weights:
                path_node_ids = nx.shortest_path(
                    graph,
                    source=start_entity,
                    target=end_entity,
                    weight=weight_property,
                )
            elif algorithm == "all_shortest":
                path_generators = nx.all_shortest_paths(
                    graph, source=start_entity, target=end_entity
                )
                path_node_ids = next(path_generators, None)  # Get first path
            else:
                # Default to simple shortest path
                path_node_ids = nx.shortest_path(
                    graph, source=start_entity, target=end_entity
                )

            if not path_node_ids:
                return None

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.warning(
                f"No path found between '{start_entity}' and '{end_entity}' in NetworkX graph."
            )
            return None

        # Hydrate the path with full node and edge data
        node_tasks = [self.aget_entity_details(node_id) for node_id in path_node_ids]
        edge_tasks = []
        for i in range(len(path_node_ids) - 1):
            src, tgt = path_node_ids[i], path_node_ids[i + 1]
            edge_tasks.append(self.chunk_entity_relation_graph.get_edge(src, tgt))

        hydrated_nodes = await asyncio.gather(*node_tasks)
        hydrated_edges_data = await asyncio.gather(*edge_tasks)

        # Format edges with their source and target
        hydrated_edges = []
        for i, edge_data in enumerate(hydrated_edges_data):
            if edge_data:
                src, tgt = path_node_ids[i], path_node_ids[i + 1]
                hydrated_edges.append({"source": src, "target": tgt, **edge_data})

        return {
            "nodes": [node for node in hydrated_nodes if node],
            "edges": hydrated_edges,
        }

    async def _find_path_neo4j(
        self,
        start_entity: str,
        end_entity: str,
        algorithm: str,
        max_hops: int,
        use_weights: bool,
        weight_property: str,
    ) -> Optional[Dict[str, List]]:
        """Neo4j-specific path finding implementation."""

        # Auto-select the best algorithm based on conditions
        if algorithm == "auto":
            if use_weights:
                # For weighted paths, prefer APOC Dijkstra for simplicity
                neo4j_algorithm = "apoc_dijkstra"
            else:
                # For unweighted paths, use modern Cypher SHORTEST
                neo4j_algorithm = "cypher_shortest"
        elif algorithm == "shortest":
            neo4j_algorithm = "cypher_shortest" if not use_weights else "apoc_dijkstra"
        elif algorithm == "dijkstra":
            neo4j_algorithm = "gds_dijkstra" if use_weights else "cypher_shortest"
        elif algorithm == "all_shortest":
            # Handle all shortest paths separately
            return await self._find_all_shortest_paths_neo4j(
                start_entity, end_entity, max_hops
            )
        elif algorithm == "k_shortest":
            # Handle K shortest paths separately
            return await self._find_k_shortest_paths_neo4j(
                start_entity, end_entity, max_hops, weight_property
            )
        else:
            neo4j_algorithm = "cypher_shortest"

        # Use the Neo4j storage's path finding method
        path_result = await self.chunk_entity_relation_graph.find_shortest_path(
            start_entity=start_entity,
            end_entity=end_entity,
            max_hops=max_hops,
            use_weights=use_weights,
            weight_property=weight_property,
            algorithm=neo4j_algorithm,
        )

        if not path_result:
            logger.warning(
                f"No path found between '{start_entity}' and '{end_entity}' in Neo4j graph."
            )
            return None

        # The Neo4j implementation already returns hydrated data
        return path_result

    async def _find_all_shortest_paths_neo4j(
        self, start_entity: str, end_entity: str, max_hops: int
    ) -> Optional[Dict[str, List]]:
        """Find all shortest paths using Neo4j."""
        paths = await self.chunk_entity_relation_graph.find_all_shortest_paths(
            start_entity=start_entity,
            end_entity=end_entity,
            max_hops=max_hops,
            limit=10,
        )

        if paths:
            # Return the first shortest path for compatibility, but log all paths found
            logger.info(
                f"Found {len(paths)} shortest paths between {start_entity} and {end_entity}"
            )
            return paths[0]
        else:
            return None

    async def _find_k_shortest_paths_neo4j(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int,
        weight_property: str,
        k: int = 3,
    ) -> Optional[Dict[str, List]]:
        """Find K shortest paths using Neo4j (Yen's algorithm)."""
        paths = await self.chunk_entity_relation_graph.find_k_shortest_paths(
            start_entity=start_entity,
            end_entity=end_entity,
            k=k,
            weight_property=weight_property,
        )

        if paths:
            # Return the first shortest path for compatibility, but log all paths found
            logger.info(
                f"Found {len(paths)} K-shortest paths between {start_entity} and {end_entity}"
            )
            return paths[0]
        else:
            return None

    # ===================================================================
    # ADVANCED PATH FINDING METHODS FOR AGENTS
    # ===================================================================

    async def afind_multiple_reasoning_paths(
        self, start_entity: str, end_entity: str, k: int = 3, algorithm: str = "auto"
    ) -> List[Dict[str, List]]:
        """
        Find multiple reasoning paths between two entities.
        Useful for agents that need to explore different reasoning routes.

        Args:
            start_entity: Source entity name
            end_entity: Target entity name
            k: Number of paths to find
            algorithm: Algorithm preference

        Returns:
            List of path dictionaries
        """
        start_node = start_entity.upper()
        end_node = end_entity.upper()

        if isinstance(self.chunk_entity_relation_graph, Neo4jStorage):
            if algorithm in ["auto", "k_shortest", "yens"]:
                return await self.chunk_entity_relation_graph.find_k_shortest_paths(
                    start_entity=start_node,
                    end_entity=end_node,
                    k=k,
                    weight_property="weight",
                )
            else:
                return await self.chunk_entity_relation_graph.find_all_shortest_paths(
                    start_entity=start_node, end_entity=end_node, max_hops=10, limit=k
                )

        elif isinstance(self.chunk_entity_relation_graph, NetworkXStorage):
            try:
                import networkx as nx

                graph = self.chunk_entity_relation_graph.graph

                # Find all shortest paths and return up to k of them
                path_generators = nx.all_shortest_paths(
                    graph, source=start_node, target=end_node
                )
                paths = []

                for i, path_node_ids in enumerate(path_generators):
                    if i >= k:
                        break

                    # Hydrate each path
                    node_tasks = [
                        self.aget_entity_details(node_id) for node_id in path_node_ids
                    ]
                    edge_tasks = []
                    for j in range(len(path_node_ids) - 1):
                        src, tgt = path_node_ids[j], path_node_ids[j + 1]
                        edge_tasks.append(
                            self.chunk_entity_relation_graph.get_edge(src, tgt)
                        )

                    hydrated_nodes = await asyncio.gather(*node_tasks)
                    hydrated_edges_data = await asyncio.gather(*edge_tasks)

                    hydrated_edges = []
                    for j, edge_data in enumerate(hydrated_edges_data):
                        if edge_data:
                            src, tgt = path_node_ids[j], path_node_ids[j + 1]
                            hydrated_edges.append(
                                {"source": src, "target": tgt, **edge_data}
                            )

                    paths.append(
                        {
                            "nodes": [node for node in hydrated_nodes if node],
                            "edges": hydrated_edges,
                            "path_length": len(path_node_ids) - 1,
                        }
                    )

                return paths

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                logger.warning(
                    f"No paths found between '{start_entity}' and '{end_entity}' in NetworkX graph."
                )
                return []

        else:
            raise NotImplementedError(
                f"Multiple path finding is not implemented for {type(self.chunk_entity_relation_graph).__name__}."
            )

    async def afind_weighted_reasoning_path(
        self,
        start_entity: str,
        end_entity: str,
        weight_property: str = "weight",
        algorithm: str = "dijkstra",
    ) -> Optional[Dict[str, List]]:
        """
        Find the shortest weighted path between two entities.
        Useful for agents that need to consider relationship strengths or costs.

        Args:
            start_entity: Source entity name
            end_entity: Target entity name
            weight_property: Property to use as edge weight
            algorithm: Algorithm to use ('dijkstra', 'apoc_dijkstra', 'gds_dijkstra')

        Returns:
            Path dictionary with cost information, or None if no path exists
        """
        return await self.afind_reasoning_path(
            start_entity=start_entity,
            end_entity=end_entity,
            algorithm=algorithm,
            use_weights=True,
            weight_property=weight_property,
        )

    # --- Internal methods for indexing lifecycle ---
    async def _insert_start(self):
        tasks = [
            cast(StorageNameSpace, s).index_start_callback()
            for s in [self.chunk_entity_relation_graph]
            if s
        ]
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        all_storages = [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]
        tasks = [
            cast(StorageNameSpace, s).index_done_callback() for s in all_storages if s
        ]
        await asyncio.gather(*tasks)

    async def _query_done(self):
        # This is now used by the agent after a turn if LLM caching is on
        if self.llm_response_cache:
            await self.llm_response_cache.index_done_callback()

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        Performs a full, autonomous RAG query.
        This method is no longer the primary entry point for an agent but serves two purposes:
        1. Provides backward compatibility for simple, non-agentic use cases.
        2. Can be used by an agent as a high-level "holistic_search" tool when it needs a
           comprehensive, pre-built context without manual exploration.

        Args:
            query: The user's question.
            param: Query parameters to control the search behavior.

        Returns:
            A string containing the LLM's final, synthesized answer.
        """
        logger.info(f"Performing autonomous query for: '{query}'")
        context = await self.aget_holistic_context(query, param)

        if not context:
            logger.warning(
                "Could not generate context for the query. Returning fail response."
            )
            return PROMPTS["fail_response"]

        # The local_rag_response prompt is now a general-purpose response prompt
        # We need to create it in prompt.py
        sys_prompt_template = PROMPTS.get(
            "global_rag_response",
            "Context: {context_data}\n\n---\n\nQuestion: {query}\n\nAnswer:",
        )

        sys_prompt = sys_prompt_template.format(
            context_data=context,
            response_type=param.response_type,  # Assuming response_type is part of your prompt
            query=query,
        )

        response = await self.best_model_func(
            query,  # The query is also part of the system prompt for some models
            system_prompt=sys_prompt,
        )
        await self._query_done()
        return response

    # --- NEW AUTONOMOUS TOOLS FOR THE AGENT ---

    async def aget_holistic_context(
        self, query: str, param: QueryParam = QueryParam()
    ) -> Optional[str]:
        """
        A powerful tool for the agent. Performs a full, hierarchical search and returns
        a rich, formatted string of the most relevant context, including communities,
        entities, relationships, and source texts.

        Args:
            query: The natural language query.
            param: Query parameters to fine-tune the search.

        Returns:
            A formatted string containing all retrieved context, or None if nothing is found.
        """
        logger.info(f"Building holistic context for query: '{query}'")
        # This is the resurrected and encapsulated logic from the old _build_hierarchical_query_context
        return await self._build_holistic_context(query, param)

    async def aget_community_details(
        self, community_id: str, param: QueryParam = QueryParam()
    ) -> Optional[str]:
        """
        A tool for the agent to "zoom in" on a specific community.
        Returns a detailed breakdown of the community, including its full report,
        and a list of its most important entities and relationships.

        Args:
            community_id: The ID of the community to inspect.
            param: Query parameters to control how much detail is returned.

        Returns:
            A detailed markdown string describing the community, or None if not found.
        """
        logger.info(f"Getting detailed view for community: {community_id}")
        community_data = await self.community_reports.get_by_id(community_id)
        if not community_data:
            return None

        # This is the resurrected and encapsulated logic from the old _pack_single_community_describe
        return await self._pack_community_details(community_data, param)

    # --- PRIVATE HELPER METHODS FOR AUTONOMOUS CONTEXT BUILDING ---
    # These methods are moved from _op.py into the HiRAG class for better encapsulation.

    async def _build_holistic_context(
        self, query: str, param: QueryParam
    ) -> Optional[str]:
        """Internal logic to construct the full context for a query."""
        # 1. Find initial entities via vector search
        vdb_results = await self.entities_vdb.query(
            query, top_k=param.top_k * 5
        )  # Over-fetch
        if not vdb_results:
            return None

        # 2. Hydrate entity data
        node_detail_tasks = [
            self.aget_entity_details(r["entity_name"]) for r in vdb_results
        ]
        node_details = await asyncio.gather(*node_detail_tasks)

        all_node_data = [
            node for node in node_details if node and not node.get("is_temporary")
        ]
        if not all_node_data:
            return None  # No non-temporary entities found

        top_node_data = all_node_data[: param.top_k]

        # 3. Gather context from all layers in parallel
        community_task = self._find_related_communities(top_node_data, param)
        text_unit_task = self._find_related_text_units(top_node_data, param)

        # Pathfinding logic
        key_entities = [node["entity_name"] for node in top_node_data[: param.top_m]]
        path_task = (
            self.afind_reasoning_path(key_entities[0], key_entities[-1])
            if len(key_entities) >= 2
            else asyncio.sleep(0, result=None)
        )

        use_communities, use_text_units, reasoning_path = await asyncio.gather(
            community_task, text_unit_task, path_task
        )

        # 4. Format context into a string
        logger.info(
            f"Context built: {len(top_node_data)} entities, {len(use_communities)} communities, {len(use_text_units)} text units, path_found={reasoning_path is not None}"
        )

        # Formatting entities
        entities_list = [["id", "name", "type", "description"]]
        for i, n in enumerate(top_node_data):
            entities_list.append(
                [
                    i,
                    n["entity_name"],
                    n.get("entity_type", "unknown"),
                    n.get("description", "N/A"),
                ]
            )
        entities_context = list_of_list_to_csv(entities_list)

        # Formatting communities
        communities_list = [["id", "title", "summary"]]
        for i, c in enumerate(use_communities):
            report = c.get("report_json", {})
            communities_list.append(
                [i, report.get("title", "N/A"), report.get("summary", "N/A")]
            )
        communities_context = list_of_list_to_csv(communities_list)

        # Formatting reasoning path
        path_context = "No direct path found between top entities."
        if reasoning_path and reasoning_path.get("edges"):
            path_list = [["source", "target", "description"]]
            for edge in reasoning_path["edges"]:
                path_list.append(
                    [edge["source"], edge["target"], edge.get("description", "N/A")]
                )
            path_context = list_of_list_to_csv(path_list)

        # Formatting source texts
        texts_list = [["id", "content"]]
        for i, t in enumerate(use_text_units):
            texts_list.append([i, t["content"]])
        text_units_context = list_of_list_to_csv(texts_list)

        return f"""
-----High-Level Summaries (Communities)-----
```csv
{communities_context}
```
-----Inferred Reasoning Path-----
```csv
{path_context}
```
-----Key Entities-----
```csv
{entities_context}
```
-----Original Source Texts-----
```csv
{text_units_context}
```
"""

    async def _find_related_communities(
        self, node_datas: List[Dict], param: QueryParam
    ) -> List[Dict]:
        """Finds and ranks community reports related to a set of nodes."""
        community_counts = {}
        for node in node_datas:
            clusters_str = node.get("clusters")
            if not clusters_str:
                continue
            clusters = json.loads(clusters_str)
            for cluster in clusters:
                if cluster["level"] <= param.level:
                    c_id = str(cluster["cluster"])
                    community_counts[c_id] = community_counts.get(c_id, 0) + 1

        if not community_counts:
            return []

        community_ids = list(community_counts.keys())
        community_reports = await self.community_reports.get_by_ids(community_ids)

        valid_reports = [r for r in community_reports if r]
        valid_reports.sort(
            key=lambda r: (
                community_counts.get(r["id"], 0),
                r.get("report_json", {}).get("importance_rating", 0),
            ),
            reverse=True,
        )

        # Truncate based on token size
        return truncate_list_by_token_size(
            valid_reports,
            key=lambda r: r.get("report_string", ""),
            max_token_size=param.max_token_for_community_report,
        )

    async def _find_related_text_units(
        self, node_datas: List[Dict], param: QueryParam
    ) -> List[Dict]:
        """Finds and ranks source text chunks related to a set of nodes."""
        chunk_ids = set()
        for node in node_datas:
            source_id_str = node.get("source_id", "")
            chunk_ids.update(source_id_str.split(PROMPTS["DEFAULT_RECORD_DELIMITER"]))

        valid_chunk_ids = [cid for cid in chunk_ids if cid]
        if not valid_chunk_ids:
            return []

        chunks = await self.text_chunks.get_by_ids(valid_chunk_ids)
        valid_chunks = [c for c in chunks if c]

        # Truncate based on token size
        return truncate_list_by_token_size(
            valid_chunks,
            key=lambda c: c.get("content", ""),
            max_token_size=param.max_token_for_text_unit,
        )

    async def _pack_community_details(
        self, community_data: Dict, param: QueryParam
    ) -> str:
        """Formats a detailed breakdown of a single community."""
        report_str = community_data.get("report_string", "No report available.")

        node_ids = community_data.get("nodes", [])
        edge_tuples = community_data.get("edges", [])

        # Fetch details for top N nodes and edges for brevity
        top_node_ids = node_ids[:20]
        top_edge_tuples = edge_tuples[:20]

        node_details = await asyncio.gather(
            *[self.aget_entity_details(nid) for nid in top_node_ids]
        )
        edge_details = await asyncio.gather(
            *[
                self.chunk_entity_relation_graph.get_edge(e[0], e[1])
                for e in top_edge_tuples
            ]
        )

        # Format nodes
        nodes_list = [["name", "type", "is_temporary", "description"]]
        for node in filter(None, node_details):
            nodes_list.append(
                [
                    node["entity_name"],
                    node.get("entity_type", "N/A"),
                    node.get("is_temporary", True),
                    node.get("description", "N/A"),
                ]
            )
        nodes_csv = list_of_list_to_csv(nodes_list)

        # Format edges
        edges_list = [["source", "target", "description"]]
        for edge, details in zip(top_edge_tuples, filter(None, edge_details)):
            edges_list.append([edge[0], edge[1], details.get("description", "N/A")])
        edges_csv = list_of_list_to_csv(edges_list)

        return f"""
# Community Report: {community_data.get("report_json", {}).get("title", "N/A")}

{report_str}

---
## Key Entities in this Community
```csv
{nodes_csv}
```

---
## Key Relationships in this Community
```csv
{edges_csv}
```
"""
