import asyncio
import json
import logging
import time
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

# Third-party imports for similarity analysis
try:
    from thefuzz import fuzz  # type: ignore
except ImportError:
    fuzz = None
    logging.warning("thefuzz not available. Install with: pip install thefuzz")

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    np = None
    cosine_similarity = None
    logging.warning(
        "numpy/sklearn not available. Install with: pip install numpy scikit-learn"
    )

# HiRAG imports
from .base import BaseKVStorage, TextChunkSchema
from ._utils import EmbeddingFunc, logger
from .prompt import PROMPTS


class UnionFind:
    """
    Efficient Union-Find data structure for clustering potential aliases.
    Used to group entities that might refer to the same concept.
    """

    def __init__(self, elements: List[str]):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, x: str) -> str:
        """Find the root of the set containing x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        """Union two sets containing x and y. Returns True if they were different sets."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def get_clusters(self) -> List[List[str]]:
        """Get all connected components as clusters."""
        clusters = defaultdict(list)
        for elem in self.parent:
            root = self.find(elem)
            clusters[root].append(elem)

        # Return only clusters with more than one element
        return [cluster for cluster in clusters.values() if len(cluster) > 1]


@dataclass
class DisambiguationConfig:
    """Enhanced configuration for the disambiguation process with memory and performance optimizations."""

    # Similarity thresholds
    lexical_similarity_threshold: float = 0.85  # thefuzz token_sort_ratio threshold
    semantic_similarity_threshold: float = (
        0.88  # cosine similarity threshold for embeddings
    )

    # Safety limits
    edm_max_cluster_size: int = 6  # Maximum entities in a cluster for LLM processing
    max_context_tokens: int = 4000  # Maximum tokens for context in LLM prompt

    # Memory management
    embedding_batch_size: int = 32  # Batch size for embedding generation
    similarity_chunk_size: int = (
        100  # Chunk size for memory-efficient similarity computation
    )
    max_memory_entities: int = 10000  # Maximum entities to process in memory at once

    # Dynamic concurrency control
    max_concurrent_llm_calls: int = 3  # Base maximum concurrent LLM calls
    adaptive_concurrency: bool = True  # Enable dynamic concurrency adjustment
    min_concurrent_calls: int = 1  # Minimum concurrent calls
    max_concurrent_calls_limit: int = 10  # Hard upper limit for concurrent calls

    # Retry and error handling
    max_llm_retries: int = 3  # Maximum retries for LLM calls
    retry_delay_base: float = 1.0  # Base delay for exponential backoff (seconds)
    retry_delay_max: float = 30.0  # Maximum retry delay (seconds)

    # Context quality improvements
    enable_intelligent_truncation: bool = True  # Use intelligent context truncation
    context_overlap_ratio: float = 0.1  # Overlap ratio for context chunks
    preserve_key_sentences: bool = True  # Preserve sentences with key terms

    # Confidence thresholds
    min_merge_confidence: float = 0.8  # Minimum confidence score for merging
    high_confidence_threshold: float = 0.95  # Threshold for high-confidence decisions

    # Performance monitoring
    enable_performance_metrics: bool = True  # Enable detailed performance logging
    log_memory_usage: bool = False  # Log memory usage (can be expensive)

    # Advanced features
    enable_embedding_cache: bool = True  # Cache embeddings for repeated use
    use_approximate_similarity: bool = False  # Use approximate similarity for very large datasets  # Minimum confidence score for merging


class EntityDisambiguator:
    """
    Identifies potential entity aliases and produces a mapping to canonical names.

    This class implements a two-stage disambiguation pipeline:
    1. Candidate Generation: Uses lexical and semantic similarity to identify potential aliases
    2. LLM Verification: Uses LLM with original text context to make final decisions

    The class is ONLY responsible for decision-making, not for merging data.
    """

    def __init__(
        self,
        global_config: Dict[str, Any],
        text_chunks_kv: BaseKVStorage[TextChunkSchema],
        embedding_func: EmbeddingFunc,
        entity_names_vdb: Optional[Any] = None,  # BaseVectorStorage
        config: Optional[DisambiguationConfig] = None,
    ):
        """
        Initialize the EntityDisambiguator.

        Args:
            global_config: Global HiRAG configuration dictionary
            text_chunks_kv: Key-value storage for text chunks
            embedding_func: Function to generate embeddings
            entity_names_vdb: Optional vector database for entity names (for efficient search)
            config: Configuration for disambiguation parameters
        """
        self.global_config = global_config
        self.text_chunks_kv = text_chunks_kv
        self.embedding_func = embedding_func
        self.entity_names_vdb = entity_names_vdb
        self.config = config or DisambiguationConfig()

        # Extract necessary functions from global config
        self.best_model_func = global_config.get("best_model_func")
        if not self.best_model_func:
            raise ValueError("best_model_func must be provided in global_config")

        logger.info("EntityDisambiguator initialized with config: %s", self.config)

    async def run(self, raw_nodes: List[Dict]) -> Dict[str, str]:
        """
        Main entry point for entity disambiguation.

        Args:
            raw_nodes: List of raw entity dictionaries from extraction

        Returns:
            Dictionary mapping alias names to their canonical names
            Example: {"The theory CL_η": "CL_η", "CL_η system": "CL_η"}
        """
        if not raw_nodes:
            logger.info("No entities provided for disambiguation")
            return {}

        logger.info(f"Starting disambiguation for {len(raw_nodes)} entities")

        try:
            # Stage 1: Generate candidate clusters
            candidate_clusters = await self._generate_candidates(raw_nodes)

            if not candidate_clusters:
                logger.info("No candidate clusters found")
                return {}

            logger.info(f"Generated {len(candidate_clusters)} candidate clusters")

            # Stage 2: Verify candidates with LLM
            name_map = await self._verify_candidates_with_llm(
                candidate_clusters, raw_nodes
            )

            logger.info(f"Disambiguation complete. Generated {len(name_map)} mappings")
            return name_map

        except Exception as e:
            logger.error(f"Error during disambiguation: {e}", exc_info=True)
            # Return empty map on error to fail safely
            return {}

    async def _generate_candidates(self, raw_nodes: List[Dict]) -> List[List[str]]:
        """
        Stage 1: Generate clusters of potential aliases using similarity analysis.

        Args:
            raw_nodes: List of raw entity dictionaries

        Returns:
            List of clusters, where each cluster is a list of entity names
        """
        # Filter for non-temporary nodes only
        non_temporary_nodes = [
            node for node in raw_nodes if not node.get("is_temporary", True)
        ]

        if len(non_temporary_nodes) < 2:
            logger.info("Insufficient non-temporary entities for disambiguation")
            return []

        entity_names = [node["entity_name"] for node in non_temporary_nodes]

        # Initialize Union-Find structure
        uf = UnionFind(entity_names)

        # Phase 1: Lexical similarity clustering
        if fuzz is not None:
            logger.info("Performing lexical similarity analysis")
            await self._lexical_similarity_pass(entity_names, uf)
        else:
            logger.warning(
                "Skipping lexical similarity analysis (thefuzz not available)"
            )

        # Phase 2: Semantic similarity clustering
        try:
            logger.info("Performing semantic similarity analysis")
            await self._semantic_similarity_pass(non_temporary_nodes, uf)
        except Exception as e:
            logger.warning(
                f"Semantic similarity pass failed: {e}. Continuing with lexical results only."
            )

        # Extract final clusters
        candidate_clusters = uf.get_clusters()

        # Filter out clusters that are too large
        filtered_clusters = [
            cluster
            for cluster in candidate_clusters
            if len(cluster) <= self.config.edm_max_cluster_size
        ]

        if len(candidate_clusters) != len(filtered_clusters):
            logger.warning(
                f"Filtered out {len(candidate_clusters) - len(filtered_clusters)} "
                f"clusters exceeding size limit of {self.config.edm_max_cluster_size}"
            )

        return filtered_clusters

    async def _lexical_similarity_pass(
        self, entity_names: List[str], uf: UnionFind
    ) -> None:
        """
        Perform lexical similarity analysis using fuzzy string matching.

        Args:
            entity_names: List of entity names to compare
            uf: Union-Find structure to update with similar entities
        """
        assert fuzz is not None
        threshold = self.config.lexical_similarity_threshold
        unions_made = 0

        for i, name1 in enumerate(entity_names):
            for j, name2 in enumerate(entity_names[i + 1 :], i + 1):
                try:
                    # Use token_sort_ratio for better handling of reordered words
                    similarity = fuzz.token_sort_ratio(name1, name2) / 100.0

                    if similarity >= threshold:
                        if uf.union(name1, name2):
                            unions_made += 1
                            logger.debug(
                                f"Lexical similarity: '{name1}' ↔ '{name2}' (score: {similarity:.3f})"
                            )

                except Exception as e:
                    logger.warning(f"Error comparing '{name1}' and '{name2}': {e}")
                    continue

        logger.info(f"Lexical similarity pass complete. Made {unions_made} unions")

    async def _semantic_similarity_pass(
        self, non_temporary_nodes: List[Dict], uf: UnionFind
    ) -> None:
        """
        Perform semantic similarity analysis using vector database search for efficiency.

        This enhanced version uses vector database capabilities to:
        - Avoid O(n²) memory usage by using optimized vector search
        - Leverage pre-computed embeddings stored during entity extraction
        - Use efficient similarity search algorithms from the vector database
        - Process entities in memory-efficient batches

        Args:
            non_temporary_nodes: List of entity dictionaries
            uf: Union-Find structure to update with similar entities
        """
        if len(non_temporary_nodes) < 2:
            logger.warning("Insufficient entities for semantic similarity analysis")
            return

        # Check if we have entity names vector database available
        if self.entity_names_vdb is None:
            logger.info(
                "Entity names vector database not available, falling back to traditional similarity computation"
            )
            await self._fallback_semantic_similarity_pass(non_temporary_nodes, uf)
            return

        logger.info(
            f"Starting vector database-based semantic similarity analysis for {len(non_temporary_nodes)} entities"
        )

        threshold = self.config.semantic_similarity_threshold
        top_k = self.global_config.get("edm_vector_search_top_k", 50)
        unions_made = 0
        total_queries = 0

        # Process entities in batches to manage memory and API limits
        batch_size = self.config.embedding_batch_size

        for i in range(0, len(non_temporary_nodes), batch_size):
            batch = non_temporary_nodes[i : i + batch_size]

            for node in batch:
                entity_name = node["entity_name"]

                try:
                    total_queries += 1

                    # Use vector database to find similar entity names
                    # Query with the entity name itself to find similar names
                    search_results = await self.entity_names_vdb.query(
                        query=entity_name, top_k=top_k
                    )

                    # Process search results to find candidates above threshold
                    for result in search_results:
                        if "entity_name" not in result:
                            continue

                        candidate_name = result["entity_name"]
                        similarity = result.get("similarity", 0.0)

                        # Skip self-matches and check threshold
                        if candidate_name == entity_name or similarity < threshold:
                            continue

                        # Skip temporary entities if specified
                        if result.get("is_temporary", False):
                            continue

                        # Union the entities if they meet criteria
                        if uf.union(entity_name, candidate_name):
                            unions_made += 1
                            logger.debug(
                                f"Vector search similarity: '{entity_name}' ↔ '{candidate_name}' "
                                f"(score: {similarity:.3f})"
                            )

                except Exception as e:
                    logger.warning(
                        f"Error in vector search for entity '{entity_name}': {e}"
                    )
                    continue

            # Log progress for large batches
            if total_queries % 100 == 0:
                logger.debug(
                    f"Processed {total_queries} vector queries, made {unions_made} unions"
                )

        logger.info(
            f"Vector database semantic similarity pass complete. Made {unions_made} unions from {total_queries} queries "
            f"(threshold: {threshold:.3f})"
        )

    async def _fallback_semantic_similarity_pass(
        self, non_temporary_nodes: List[Dict], uf: UnionFind
    ) -> None:
        """
        Fallback semantic similarity analysis using traditional methods when vector DB is unavailable.

        This is a streamlined version of the original method for backup purposes only.
        """
        if np is None or cosine_similarity is None:
            logger.warning(
                "Skipping semantic similarity analysis (numpy/sklearn not available)"
            )
            return

        logger.info(
            f"Using fallback semantic similarity analysis for {len(non_temporary_nodes)} entities"
        )

        # Collect entities with embeddings
        entity_data = []
        for node in non_temporary_nodes:
            name = node["entity_name"]
            embedding = node.get("embedding")
            if embedding is not None:
                entity_data.append({"name": name, "embedding": embedding})

        if len(entity_data) < 2:
            logger.warning(
                "Insufficient valid embeddings for fallback similarity analysis"
            )
            return

        # Simple pairwise comparison (not memory-optimized, for fallback only)
        threshold = self.config.semantic_similarity_threshold
        unions_made = 0

        embeddings = np.array([item["embedding"] for item in entity_data])
        similarity_matrix = cosine_similarity(embeddings)

        for i, item_a in enumerate(entity_data):
            for j, item_b in enumerate(entity_data[i + 1 :], i + 1):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    if uf.union(item_a["name"], item_b["name"]):
                        unions_made += 1
                        logger.debug(
                            f"Fallback similarity: '{item_a['name']}' ↔ '{item_b['name']}' "
                            f"(score: {similarity:.3f})"
                        )

        logger.info(
            f"Fallback semantic similarity pass complete. Made {unions_made} unions"
        )

    async def _verify_candidates_with_llm(
        self, candidate_clusters: List[List[str]], raw_nodes: List[Dict]
    ) -> Dict[str, str]:
        """
        Stage 2: Verify candidate clusters using LLM with dynamic concurrency control.

        Enhanced with:
        - Dynamic concurrency adjustment based on cluster sizes and system load
        - Intelligent workload balancing
        - Performance monitoring and adaptive throttling

        Args:
            candidate_clusters: List of clusters to verify
            raw_nodes: Original entity data for context lookup

        Returns:
            Dictionary mapping alias names to canonical names
        """
        if not candidate_clusters:
            return {}

        # Create lookup map for entity data
        entity_lookup = {node["entity_name"]: node for node in raw_nodes}

        # Dynamic concurrency calculation
        base_concurrency = self.config.max_concurrent_llm_calls
        if self.config.adaptive_concurrency:
            # Adjust concurrency based on cluster characteristics
            avg_cluster_size = sum(
                len(cluster) for cluster in candidate_clusters
            ) / len(candidate_clusters)
            complexity_factor = min(
                2.0, avg_cluster_size / 3.0
            )  # Clusters of 3+ are more complex

            # More concurrent calls for simpler clusters, fewer for complex ones
            adjusted_concurrency = max(
                self.config.min_concurrent_calls,
                min(
                    self.config.max_concurrent_calls_limit,
                    int(base_concurrency / complexity_factor),
                ),
            )

            logger.info(
                f"Dynamic concurrency: {adjusted_concurrency} (base: {base_concurrency}, "
                f"avg cluster size: {avg_cluster_size:.1f}, complexity factor: {complexity_factor:.2f})"
            )
        else:
            adjusted_concurrency = base_concurrency

        # Create semaphore with dynamic limit
        semaphore = asyncio.Semaphore(adjusted_concurrency)

        # Enhanced cluster processing with retry logic
        async def process_cluster_with_retry(cluster: List[str]) -> Dict[str, str]:
            """Process a single cluster with retry logic and error handling."""
            for attempt in range(self.config.max_llm_retries + 1):
                try:
                    async with semaphore:
                        # Add small delay between retries with exponential backoff
                        if attempt > 0:
                            delay = min(
                                self.config.retry_delay_base * (2 ** (attempt - 1)),
                                self.config.retry_delay_max,
                            )
                            logger.debug(
                                f"Retrying cluster {cluster} after {delay:.1f}s delay (attempt {attempt + 1})"
                            )
                            await asyncio.sleep(delay)

                        return await self._verify_single_cluster(cluster, entity_lookup)

                except Exception as e:
                    if attempt < self.config.max_llm_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for cluster {cluster}: {e}"
                        )
                        continue
                    else:
                        logger.error(
                            f"All {self.config.max_llm_retries + 1} attempts failed for cluster {cluster}: {e}"
                        )
                        return {}

            return {}

        # Sort clusters by size for better load balancing (smaller clusters first)
        sorted_clusters = sorted(candidate_clusters, key=len)

        logger.info(
            f"Processing {len(sorted_clusters)} clusters with concurrency limit of {adjusted_concurrency}"
        )

        # Process all clusters concurrently with performance monitoring
        start_time = time.time() if self.config.enable_performance_metrics else None

        cluster_results = await asyncio.gather(
            *[process_cluster_with_retry(cluster) for cluster in sorted_clusters],
            return_exceptions=True,
        )

        # Performance logging
        if self.config.enable_performance_metrics and start_time:
            processing_time = time.time() - start_time
            successful_clusters = sum(
                1 for result in cluster_results if isinstance(result, dict) and result
            )
            logger.info(
                f"LLM verification completed in {processing_time:.2f}s. "
                f"Processed {len(sorted_clusters)} clusters with {successful_clusters} successful verifications. "
                f"Average time per cluster: {processing_time / len(sorted_clusters):.2f}s"
            )

        # Merge results from all clusters with enhanced error reporting
        final_name_map = {}
        successful_verifications = 0
        failed_verifications = 0
        exception_count = 0

        for i, result in enumerate(cluster_results):
            cluster = sorted_clusters[i]

            if isinstance(result, Exception):
                logger.error(
                    f"Unhandled exception processing cluster {cluster}: {result}"
                )
                exception_count += 1
                continue

            if isinstance(result, dict):
                if result:  # Non-empty result means successful merge
                    final_name_map.update(result)
                    successful_verifications += 1
                    logger.debug(f"Successful merge for cluster {cluster}: {result}")
                else:  # Empty result means no merge (could be valid decision or failure)
                    failed_verifications += 1
            else:
                logger.warning(
                    f"Unexpected result type for cluster {cluster}: {type(result)}"
                )
                failed_verifications += 1

        logger.info(
            f"LLM verification summary: {successful_verifications} successful merges, "
            f"{failed_verifications} no-merge decisions, {exception_count} exceptions. "
            f"Generated {len(final_name_map)} total entity mappings."
        )

        return final_name_map

    async def _verify_single_cluster(
        self, cluster: List[str], entity_lookup: Dict[str, Dict]
    ) -> Dict[str, str]:
        """
        Verify a single cluster using LLM analysis with enhanced context processing.

        This enhanced version implements:
        - Full context preservation (no truncation)
        - Better error handling and validation
        - Enhanced decision processing with quality checks
        - Comprehensive logging for debugging

        Args:
            cluster: List of entity names to verify
            entity_lookup: Lookup dictionary for entity data

        Returns:
            Dictionary mapping aliases to canonical name for this cluster
        """
        if len(cluster) < 2:
            return {}

        try:
            # Gather entity data and original text context
            cluster_data = []
            source_chunks_needed = set()

            for entity_name in cluster:
                entity = entity_lookup.get(entity_name)
                if not entity:
                    logger.warning(f"Entity '{entity_name}' not found in lookup")
                    continue

                # Collect source chunk IDs
                source_id_str = entity.get("source_id", "")
                chunk_ids = []
                if source_id_str:
                    delimiter = PROMPTS.get("DEFAULT_RECORD_DELIMITER", "<SEP>")
                    chunk_ids = [
                        cid.strip()
                        for cid in source_id_str.split(delimiter)
                        if cid.strip()
                    ]
                    source_chunks_needed.update(chunk_ids)

                cluster_data.append(
                    {
                        "entity_name": entity_name,
                        "entity_description": entity.get("description", ""),
                        "entity_type": entity.get("entity_type", ""),
                        "source_ids": chunk_ids,
                    }
                )

            if not cluster_data:
                logger.warning(f"No valid entities found in cluster: {cluster}")
                return {}

            # Fetch original text chunks
            chunk_lookup = await self._fetch_source_chunks(list(source_chunks_needed))

            # Add full original text context to each entity (no truncation)
            for entity_data in cluster_data:
                contexts = []
                for chunk_id in entity_data["source_ids"]:
                    chunk = chunk_lookup.get(chunk_id)
                    if chunk and isinstance(chunk, dict):
                        content = chunk.get("content", "")
                        if content:
                            contexts.append(content)

                # Combine all contexts with clear separators
                if contexts:
                    combined_context = "\n\n---\n\n".join(contexts)
                    entity_data["original_text_context"] = combined_context
                else:
                    entity_data["original_text_context"] = "No context available."

            # Prepare LLM prompt
            prompt_key = "entity_disambiguation"
            if prompt_key not in PROMPTS:
                logger.error(f"Required prompt '{prompt_key}' not found in PROMPTS")
                return {}

            prompt_template = PROMPTS[prompt_key]
            cluster_json = json.dumps(cluster_data, indent=2)
            prompt = prompt_template.format(input_json_for_cluster=cluster_json)

            # Call LLM
            assert self.best_model_func is not None
            llm_response = await self.best_model_func(prompt)

            # Parse and validate LLM response
            decision = await self._parse_and_validate_llm_decision(
                llm_response, cluster
            )

            if decision is None:
                logger.warning(f"Invalid LLM decision for cluster {cluster}")
                return {}

            # Process decision with enhanced validation
            if decision["decision"] == "MERGE":
                canonical_name = decision["canonical_name"]
                aliases = decision["aliases"]
                confidence = decision.get("confidence_score", 0.0)

                # Validate that canonical + aliases matches cluster
                expected_entities = set([canonical_name] + aliases)
                actual_entities = set(cluster)

                if expected_entities != actual_entities:
                    logger.warning(
                        f"LLM decision entities mismatch. Expected: {expected_entities}, "
                        f"Actual: {actual_entities}"
                    )
                    return {}

                # Check confidence threshold
                if confidence < self.config.min_merge_confidence:
                    logger.info(
                        f"LLM decision confidence {confidence:.3f} below threshold "
                        f"{self.config.min_merge_confidence}. Skipping merge."
                    )
                    return {}

                # Create name mapping
                name_map = {alias: canonical_name for alias in aliases}

                logger.info(
                    f"MERGE decision for cluster {cluster}: canonical='{canonical_name}', "
                    f"aliases={aliases}, confidence={confidence:.3f}"
                )

                return name_map

            else:  # DO_NOT_MERGE
                confidence = decision.get("confidence_score", 0.0)
                logger.info(
                    f"DO_NOT_MERGE decision for cluster {cluster} "
                    f"(confidence: {confidence:.3f})"
                )
                return {}

        except Exception as e:
            logger.error(f"Error verifying cluster {cluster}: {e}", exc_info=True)
            return {}

    async def _fetch_source_chunks(self, chunk_ids: List[str]) -> Dict[str, TextChunkSchema]:
        """
        Fetch source text chunks from the key-value store.

        Args:
            chunk_ids: List of chunk IDs to fetch

        Returns:
            Dict mapping chunk ID to chunk dictionary
        """
        if not chunk_ids:
            return {}

        try:
            # Filter out empty or invalid chunk IDs
            valid_chunk_ids = sorted(list(set(cid for cid in chunk_ids if cid and cid.strip())))

            if not valid_chunk_ids:
                return {}

            chunks = await self.text_chunks_kv.get_by_ids(valid_chunk_ids)
            return {cid: chunk for cid, chunk in zip(valid_chunk_ids, chunks) if chunk}

        except Exception as e:
            logger.error(f"Error fetching source chunks: {e}")
            return {}

    async def _parse_and_validate_llm_decision(
        self, llm_response: str, cluster: List[str]
    ) -> Optional[Dict]:
        """
        Parse and validate the LLM's disambiguation decision.

        Args:
            llm_response: Raw response from the LLM
            cluster: Original cluster for validation

        Returns:
            Validated decision dictionary or None if invalid
        """
        try:
            # Extract JSON from response
            response_cleaned = llm_response.strip()

            # Handle cases where LLM might wrap JSON in markdown
            if response_cleaned.startswith("```json"):
                response_cleaned = response_cleaned[7:]
            if response_cleaned.endswith("```"):
                response_cleaned = response_cleaned[:-3]
            response_cleaned = response_cleaned.strip()

            decision = json.loads(response_cleaned)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw LLM response: {llm_response}")
            return None

        # Validate decision structure
        if not isinstance(decision, dict):
            logger.error("LLM decision is not a dictionary")
            return None

        # Check required fields
        required_fields = ["decision", "confidence_score", "justification"]
        for field in required_fields:
            if field not in decision:
                logger.error(f"Missing required field '{field}' in LLM decision")
                return None

        # Validate decision value
        decision_value = decision["decision"]
        if decision_value not in ["MERGE", "DO_NOT_MERGE"]:
            logger.error(f"Invalid decision value: {decision_value}")
            return None

        # Validate confidence score
        confidence = decision["confidence_score"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            logger.error(f"Invalid confidence score: {confidence}")
            return None

        # Additional validation for MERGE decisions
        if decision_value == "MERGE":
            if "canonical_name" not in decision:
                logger.error("MERGE decision missing 'canonical_name'")
                return None

            if "aliases" not in decision:
                logger.error("MERGE decision missing 'aliases'")
                return None

            canonical_name = decision["canonical_name"]
            aliases = decision["aliases"]

            if not isinstance(aliases, list):
                logger.error("'aliases' must be a list")
                return None

            # Validate that canonical + aliases matches cluster
            all_entities = set([canonical_name] + aliases)
            cluster_entities = set(cluster)

            if all_entities != cluster_entities:
                logger.error(
                    f"Entity mismatch. Decision entities: {all_entities}, "
                    f"Cluster entities: {cluster_entities}"
                )
                return None

        return decision


# Configuration helper functions
def create_disambiguation_config(**kwargs) -> DisambiguationConfig:
    """
    Create a DisambiguationConfig with custom parameters.

    Args:
        **kwargs: Configuration parameters to override

    Returns:
        DisambiguationConfig instance
    """
    return DisambiguationConfig(**kwargs)


def get_default_edm_config() -> Dict[str, Any]:
    """
    Get default configuration parameters for EDM integration into HiRAG.

    Returns:
        Dictionary of configuration parameters
    """
    return {
        "enable_entity_disambiguation": True,
        "edm_lexical_similarity_threshold": 0.85,
        "edm_semantic_similarity_threshold": 0.88,
        "edm_max_cluster_size": 6,
        "edm_max_context_tokens": 4000,
        "edm_min_merge_confidence": 0.8,
        "edm_embedding_batch_size": 32,
        "edm_max_concurrent_llm_calls": 3,
    }


def validate_edm_configuration(config: Dict[str, Any]) -> List[str]:
    """
    Enhanced validation for EDM configuration parameters with comprehensive checks.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation messages (empty if all valid)
    """
    issues = []

    # Check thresholds
    lexical_threshold = config.get("edm_lexical_similarity_threshold", 0.85)
    if not 0.0 <= lexical_threshold <= 1.0:
        issues.append(
            f"edm_lexical_similarity_threshold must be between 0.0 and 1.0, got {lexical_threshold}"
        )

    semantic_threshold = config.get("edm_semantic_similarity_threshold", 0.88)
    if not 0.0 <= semantic_threshold <= 1.0:
        issues.append(
            f"edm_semantic_similarity_threshold must be between 0.0 and 1.0, got {semantic_threshold}"
        )

    # Check cluster size limits
    max_cluster_size = config.get("edm_max_cluster_size", 6)
    if not isinstance(max_cluster_size, int) or max_cluster_size < 2:
        issues.append(
            f"edm_max_cluster_size must be an integer >= 2, got {max_cluster_size}"
        )

    # Check confidence threshold
    min_confidence = config.get("edm_min_merge_confidence", 0.8)
    if not 0.0 <= min_confidence <= 1.0:
        issues.append(
            f"edm_min_merge_confidence must be between 0.0 and 1.0, got {min_confidence}"
        )

    # Check concurrency limits
    max_concurrent = config.get("edm_max_concurrent_llm_calls", 3)
    if not isinstance(max_concurrent, int) or max_concurrent < 1:
        issues.append(
            f"edm_max_concurrent_llm_calls must be an integer >= 1, got {max_concurrent}"
        )

    # Validate new vector database configuration
    enable_names_vdb = config.get("enable_entity_names_vdb", True)
    if not isinstance(enable_names_vdb, bool):
        issues.append(
            f"enable_entity_names_vdb must be a boolean, got {enable_names_vdb}"
        )

    vector_search_top_k = config.get("edm_vector_search_top_k", 50)
    if not isinstance(vector_search_top_k, int) or vector_search_top_k < 1:
        issues.append(
            f"edm_vector_search_top_k must be an integer >= 1, got {vector_search_top_k}"
        )
    elif vector_search_top_k > 1000:
        issues.append(
            f"Warning: edm_vector_search_top_k ({vector_search_top_k}) is very large. This may impact performance."
        )

    memory_batch_size = config.get("edm_memory_batch_size", 1000)
    if not isinstance(memory_batch_size, int) or memory_batch_size < 1:
        issues.append(
            f"edm_memory_batch_size must be an integer >= 1, got {memory_batch_size}"
        )

    # Validate embedding batch size
    embedding_batch_size = config.get("edm_embedding_batch_size", 32)
    if not isinstance(embedding_batch_size, int) or embedding_batch_size < 1:
        issues.append(
            f"edm_embedding_batch_size must be an integer >= 1, got {embedding_batch_size}"
        )

    # Check for potential performance issues
    if not enable_names_vdb and max_cluster_size > 5:
        issues.append(
            f"Warning: With entity_names_vdb disabled and max_cluster_size ({max_cluster_size}) > 5, "
            f"performance may degrade significantly due to O(n²) similarity computation."
        )

    # Warn about potentially problematic combinations
    if lexical_threshold > semantic_threshold:
        issues.append(
            f"Warning: lexical threshold ({lexical_threshold}) is higher than semantic threshold ({semantic_threshold}). This may reduce disambiguation effectiveness."
        )

    if max_cluster_size > 10:
        issues.append(
            f"Warning: edm_max_cluster_size ({max_cluster_size}) is quite large. This may impact LLM performance and accuracy."
        )

    # Check context token limits
    max_context_tokens = config.get("edm_max_context_tokens", 4000)
    if not isinstance(max_context_tokens, int) or max_context_tokens < 100:
        issues.append(
            f"edm_max_context_tokens must be an integer >= 100, got {max_context_tokens}"
        )
    elif max_context_tokens > 100000:
        issues.append(
            f"Warning: edm_max_context_tokens ({max_context_tokens}) is very large. This may impact LLM performance and cost."
        )

    return issues


def log_disambiguation_statistics(
    raw_nodes: List[Dict], name_to_canonical_map: Dict[str, str], logger_instance=None
) -> Dict[str, Any]:
    """
    Log and return statistics about the disambiguation process.

    Args:
        raw_nodes: Original raw nodes before disambiguation
        name_to_canonical_map: Mapping from aliases to canonical names
        logger_instance: Logger to use (defaults to module logger)

    Returns:
        Dictionary containing disambiguation statistics
    """
    if logger_instance is None:
        logger_instance = logger

    total_entities = len(raw_nodes)
    total_mappings = len(name_to_canonical_map)

    # Count entities by temporary status
    temporary_count = sum(1 for node in raw_nodes if node.get("is_temporary", False))
    non_temporary_count = total_entities - temporary_count

    # Count canonical entities (unique values in the mapping + unmapped entities)
    canonical_names = set(name_to_canonical_map.values())
    mapped_aliases = set(name_to_canonical_map.keys())
    unmapped_entities = set(node["entity_name"] for node in raw_nodes) - mapped_aliases
    total_canonical = len(canonical_names) + len(unmapped_entities)

    # Calculate compression ratio
    compression_ratio = (total_entities - total_canonical) / max(total_entities, 1)

    stats = {
        "total_entities": total_entities,
        "temporary_entities": temporary_count,
        "non_temporary_entities": non_temporary_count,
        "total_mappings": total_mappings,
        "canonical_entities": total_canonical,
        "compression_ratio": compression_ratio,
        "entities_merged": total_entities - total_canonical,
    }

    logger_instance.info(f"Disambiguation Statistics:")
    logger_instance.info(f"  Total entities: {total_entities}")
    logger_instance.info(f"  Non-temporary entities: {non_temporary_count}")
    logger_instance.info(f"  Entities merged: {stats['entities_merged']}")
    logger_instance.info(f"  Final canonical entities: {total_canonical}")
    logger_instance.info(f"  Compression ratio: {compression_ratio:.2%}")

    return stats


def create_disambiguation_report(
    clusters_processed: List[List[str]],
    successful_merges: Dict[str, str],
    failed_clusters: List[List[str]],
) -> str:
    """
    Create a detailed report of the disambiguation process.

    Args:
        clusters_processed: List of clusters that were processed
        successful_merges: Dictionary of successful merges (alias -> canonical)
        failed_clusters: List of clusters that failed processing

    Returns:
        Formatted markdown report
    """
    total_clusters = len(clusters_processed)
    successful_clusters = len(
        [
            c
            for c in clusters_processed
            if any(entity in successful_merges for entity in c)
        ]
    )

    report = f"""# Entity Disambiguation Report

## Summary
- **Total clusters processed**: {total_clusters}
- **Successful merges**: {len(successful_merges)}
- **Failed clusters**: {len(failed_clusters)}
- **Success rate**: {(successful_clusters / max(total_clusters, 1)):.1%}

## Successful Merges
"""

    if successful_merges:
        # Group by canonical name
        canonical_groups = {}
        for alias, canonical in successful_merges.items():
            if canonical not in canonical_groups:
                canonical_groups[canonical] = []
            canonical_groups[canonical].append(alias)

        for canonical, aliases in canonical_groups.items():
            report += (
                f"- **{canonical}** ← {', '.join(f'`{alias}`' for alias in aliases)}\n"
            )
    else:
        report += "No successful merges.\n"

    if failed_clusters:
        report += f"\n## Failed Clusters ({len(failed_clusters)})\n"
        for i, cluster in enumerate(failed_clusters):
            report += f"{i + 1}. {', '.join(f'`{entity}`' for entity in cluster)}\n"

    return report


def optimize_disambiguation_config(
    dataset_characteristics: Dict[str, Any],
    current_config: Optional[DisambiguationConfig] = None,
) -> DisambiguationConfig:
    """
    Intelligently optimize disambiguation configuration based on dataset characteristics.

    This function analyzes dataset properties and automatically adjusts disambiguation
    parameters for optimal performance and accuracy.

    Args:
        dataset_characteristics: Dictionary containing dataset metrics like:
            - entity_count: Number of entities
            - avg_entity_name_length: Average length of entity names
            - domain_complexity: Domain complexity score (0.0-1.0)
            - text_chunk_count: Number of text chunks
            - estimated_processing_time: Expected processing time

        current_config: Existing configuration to optimize (optional)

    Returns:
        Optimized DisambiguationConfig instance
    """
    # Start with default config or provided config
    if current_config:
        base_config = {
            "lexical_similarity_threshold": current_config.lexical_similarity_threshold,
            "semantic_similarity_threshold": current_config.semantic_similarity_threshold,
            "edm_max_cluster_size": current_config.edm_max_cluster_size,
            "max_context_tokens": current_config.max_context_tokens,
            "embedding_batch_size": current_config.embedding_batch_size,
            "max_concurrent_llm_calls": current_config.max_concurrent_llm_calls,
            "min_merge_confidence": current_config.min_merge_confidence,
            "adaptive_concurrency": current_config.adaptive_concurrency,
            "max_llm_retries": current_config.max_llm_retries,
        }
    else:
        base_config = {
            "lexical_similarity_threshold": 0.85,
            "semantic_similarity_threshold": 0.88,
            "edm_max_cluster_size": 6,
            "max_context_tokens": 4000,
            "embedding_batch_size": 32,
            "max_concurrent_llm_calls": 3,
            "min_merge_confidence": 0.8,
            "adaptive_concurrency": True,
            "max_llm_retries": 3,
        }

    # Extract dataset characteristics
    entity_count = dataset_characteristics.get("entity_count", 1000)
    avg_name_length = dataset_characteristics.get("avg_entity_name_length", 20)
    domain_complexity = dataset_characteristics.get("domain_complexity", 0.5)
    chunk_count = dataset_characteristics.get("text_chunk_count", 100)

    # Optimization logic based on dataset size
    if entity_count < 100:
        # Small dataset: more conservative settings
        base_config["lexical_similarity_threshold"] = 0.90
        base_config["semantic_similarity_threshold"] = 0.90
        base_config["min_merge_confidence"] = 0.85
        base_config["max_concurrent_llm_calls"] = 2

    elif entity_count < 1000:
        # Medium dataset: balanced settings
        base_config["lexical_similarity_threshold"] = 0.85
        base_config["semantic_similarity_threshold"] = 0.88
        base_config["min_merge_confidence"] = 0.8
        base_config["max_concurrent_llm_calls"] = 3

    else:
        # Large dataset: more aggressive settings for efficiency
        base_config["lexical_similarity_threshold"] = 0.80
        base_config["semantic_similarity_threshold"] = 0.85
        base_config["min_merge_confidence"] = 0.75
        base_config["max_concurrent_llm_calls"] = 5
        base_config["embedding_batch_size"] = 64

    # Adjust based on domain complexity
    if domain_complexity > 0.8:
        # High complexity: be more conservative
        base_config["min_merge_confidence"] += 0.05
        base_config["lexical_similarity_threshold"] += 0.03
        base_config["semantic_similarity_threshold"] += 0.03
        base_config["edm_max_cluster_size"] = min(
            base_config["edm_max_cluster_size"], 4
        )

    elif domain_complexity < 0.3:
        # Low complexity: can be more aggressive
        base_config["min_merge_confidence"] -= 0.05
        base_config["lexical_similarity_threshold"] -= 0.02
        base_config["semantic_similarity_threshold"] -= 0.02
        base_config["edm_max_cluster_size"] = max(
            base_config["edm_max_cluster_size"], 8
        )

    # Adjust based on entity name characteristics
    if avg_name_length > 50:
        # Long entity names: rely more on semantic similarity
        base_config["lexical_similarity_threshold"] -= 0.05
        base_config["semantic_similarity_threshold"] += 0.02

    elif avg_name_length < 10:
        # Short entity names: rely more on lexical similarity
        base_config["lexical_similarity_threshold"] += 0.05
        base_config["semantic_similarity_threshold"] -= 0.02

    # Adjust concurrency based on dataset size
    if chunk_count > 1000:
        base_config["max_concurrent_llm_calls"] = min(
            base_config["max_concurrent_llm_calls"] + 2, 8
        )
        base_config["embedding_batch_size"] = min(
            base_config["embedding_batch_size"] * 2, 128
        )

    # Ensure values stay within valid ranges
    base_config["lexical_similarity_threshold"] = max(
        0.5, min(1.0, base_config["lexical_similarity_threshold"])
    )
    base_config["semantic_similarity_threshold"] = max(
        0.5, min(1.0, base_config["semantic_similarity_threshold"])
    )
    base_config["min_merge_confidence"] = max(
        0.5, min(1.0, base_config["min_merge_confidence"])
    )
    base_config["max_concurrent_llm_calls"] = max(
        1, min(10, base_config["max_concurrent_llm_calls"])
    )
    base_config["edm_max_cluster_size"] = max(
        2, min(15, base_config["edm_max_cluster_size"])
    )
    base_config["embedding_batch_size"] = max(
        8, min(256, base_config["embedding_batch_size"])
    )

    return DisambiguationConfig(**base_config)


def estimate_dataset_characteristics(raw_nodes: List[Dict]) -> Dict[str, Any]:
    """
    Estimate dataset characteristics for configuration optimization.

    Args:
        raw_nodes: List of raw entity dictionaries

    Returns:
        Dictionary containing dataset characteristics
    """
    if not raw_nodes:
        return {
            "entity_count": 0,
            "avg_entity_name_length": 0,
            "domain_complexity": 0.5,
            "text_chunk_count": 0,
        }

    # Basic metrics
    entity_count = len(raw_nodes)
    entity_names = [node.get("entity_name", "") for node in raw_nodes]
    valid_names = [name for name in entity_names if name]

    avg_name_length = sum(len(name) for name in valid_names) / max(len(valid_names), 1)

    # Estimate domain complexity based on entity types and description diversity
    entity_types = set(node.get("entity_type", "unknown") for node in raw_nodes)
    type_diversity = len(entity_types) / max(entity_count, 1)

    # Estimate complexity based on various factors
    complexity_factors = [
        type_diversity,  # More types = more complex
        min(1.0, avg_name_length / 30),  # Longer names = more complex
        min(1.0, entity_count / 1000),  # More entities = more complex
    ]

    domain_complexity = sum(complexity_factors) / len(complexity_factors)

    # Estimate text chunk count based on source_id information
    all_source_ids = set()
    for node in raw_nodes:
        source_id_str = node.get("source_id", "")
        if source_id_str:
            chunk_ids = source_id_str.split("<SEP>")
            all_source_ids.update(chunk_ids)

    return {
        "entity_count": entity_count,
        "avg_entity_name_length": avg_name_length,
        "domain_complexity": domain_complexity,
        "text_chunk_count": len(all_source_ids),
        "entity_type_diversity": type_diversity,
        "estimated_processing_time": entity_count * 0.1,  # Rough estimate in seconds
    }