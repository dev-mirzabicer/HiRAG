"""
Entity Disambiguation and Merging (EDM) Pipeline

This module implements a robust entity disambiguation system that intelligently identifies
and merges entity aliases while preserving distinct entities with subtle differences.
The system uses a two-stage approach: candidate generation through similarity analysis,
followed by LLM-based verification with comprehensive context.

Architecture:
- EntityDisambiguator: Main class responsible for decision-making only
- Union-Find data structure for efficient clustering of candidate aliases
- LLM-based verification with original text context for accuracy
- Conservative approach to prevent false-positive merges

Author: Claude (Anthropic)
"""

import asyncio
import json
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

# Third-party imports for similarity analysis
try:
    from thefuzz import fuzz
except ImportError:
    fuzz = None
    logging.warning("thefuzz not available. Install with: pip install thefuzz")

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    np = None
    cosine_similarity = None
    logging.warning("numpy/sklearn not available. Install with: pip install numpy scikit-learn")

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
    """Configuration for the disambiguation process."""
    # Similarity thresholds
    lexical_similarity_threshold: float = 0.85  # thefuzz token_sort_ratio threshold
    semantic_similarity_threshold: float = 0.88  # cosine similarity threshold for embeddings
    
    # Safety limits
    edm_max_cluster_size: int = 6  # Maximum entities in a cluster for LLM processing
    max_context_tokens: int = 4000  # Maximum tokens for context in LLM prompt
    
    # Processing limits
    embedding_batch_size: int = 32  # Batch size for embedding generation
    max_concurrent_llm_calls: int = 3  # Maximum concurrent LLM calls for cluster validation
    
    # Confidence thresholds
    min_merge_confidence: float = 0.8  # Minimum confidence score for merging


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
        config: Optional[DisambiguationConfig] = None
    ):
        """
        Initialize the EntityDisambiguator.
        
        Args:
            global_config: Global HiRAG configuration dictionary
            text_chunks_kv: Key-value storage for text chunks
            embedding_func: Function to generate embeddings
            config: Configuration for disambiguation parameters
        """
        self.global_config = global_config
        self.text_chunks_kv = text_chunks_kv
        self.embedding_func = embedding_func
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
            name_map = await self._verify_candidates_with_llm(candidate_clusters, raw_nodes)
            
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
            node for node in raw_nodes 
            if not node.get("is_temporary", True)
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
            logger.warning("Skipping lexical similarity analysis (thefuzz not available)")
        
        # Phase 2: Semantic similarity clustering
        try:
            logger.info("Performing semantic similarity analysis")
            await self._semantic_similarity_pass(non_temporary_nodes, uf)
        except Exception as e:
            logger.warning(f"Semantic similarity pass failed: {e}. Continuing with lexical results only.")
        
        # Extract final clusters
        candidate_clusters = uf.get_clusters()
        
        # Filter out clusters that are too large
        filtered_clusters = [
            cluster for cluster in candidate_clusters 
            if len(cluster) <= self.config.edm_max_cluster_size
        ]
        
        if len(candidate_clusters) != len(filtered_clusters):
            logger.warning(
                f"Filtered out {len(candidate_clusters) - len(filtered_clusters)} "
                f"clusters exceeding size limit of {self.config.edm_max_cluster_size}"
            )
        
        return filtered_clusters
    
    async def _lexical_similarity_pass(self, entity_names: List[str], uf: UnionFind) -> None:
        """
        Perform lexical similarity analysis using fuzzy string matching.
        
        Args:
            entity_names: List of entity names to compare
            uf: Union-Find structure to update with similar entities
        """
        threshold = self.config.lexical_similarity_threshold
        unions_made = 0
        
        for i, name1 in enumerate(entity_names):
            for j, name2 in enumerate(entity_names[i+1:], i+1):
                try:
                    # Use token_sort_ratio for better handling of reordered words
                    similarity = fuzz.token_sort_ratio(name1, name2) / 100.0
                    
                    if similarity >= threshold:
                        if uf.union(name1, name2):
                            unions_made += 1
                            logger.debug(f"Lexical similarity: '{name1}' ↔ '{name2}' (score: {similarity:.3f})")
                
                except Exception as e:
                    logger.warning(f"Error comparing '{name1}' and '{name2}': {e}")
                    continue
        
        logger.info(f"Lexical similarity pass complete. Made {unions_made} unions")
    
    async def _semantic_similarity_pass(self, non_temporary_nodes: List[Dict], uf: UnionFind) -> None:
        """
        Perform semantic similarity analysis using embeddings.
        
        Args:
            non_temporary_nodes: List of entity dictionaries with embeddings
            uf: Union-Find structure to update with similar entities
        """
        if np is None or cosine_similarity is None:
            logger.warning("Skipping semantic similarity analysis (numpy/sklearn not available)")
            return
        
        # Collect entity names and embeddings
        entity_names = []
        embeddings = []
        
        for node in non_temporary_nodes:
            name = node["entity_name"]
            embedding = node.get("embedding")
            
            if embedding is None:
                # Generate embedding for the entity description
                try:
                    description = node.get("description", name)
                    embedding_result = await self.embedding_func([description])
                    embedding = embedding_result[0] if embedding_result else None
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for '{name}': {e}")
                    continue
            
            if embedding is not None:
                entity_names.append(name)
                embeddings.append(embedding)
        
        if len(embeddings) < 2:
            logger.warning("Insufficient embeddings for semantic similarity analysis")
            return
        
        # Convert to numpy array and compute cosine similarity
        try:
            embeddings_array = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)
            
            threshold = self.config.semantic_similarity_threshold
            unions_made = 0
            
            for i, name1 in enumerate(entity_names):
                for j, name2 in enumerate(entity_names[i+1:], i+1):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= threshold:
                        if uf.union(name1, name2):
                            unions_made += 1
                            logger.debug(f"Semantic similarity: '{name1}' ↔ '{name2}' (score: {similarity:.3f})")
            
            logger.info(f"Semantic similarity pass complete. Made {unions_made} unions")
            
        except Exception as e:
            logger.error(f"Error in semantic similarity computation: {e}")
            raise
    
    async def _verify_candidates_with_llm(
        self, 
        candidate_clusters: List[List[str]], 
        raw_nodes: List[Dict]
    ) -> Dict[str, str]:
        """
        Stage 2: Verify candidate clusters using LLM with original text context.
        
        Args:
            candidate_clusters: List of clusters to verify
            raw_nodes: Original entity data for context lookup
            
        Returns:
            Dictionary mapping alias names to canonical names
        """
        # Create lookup map for entity data
        entity_lookup = {node["entity_name"]: node for node in raw_nodes}
        
        # Process clusters with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_llm_calls)
        
        async def process_cluster(cluster: List[str]) -> Dict[str, str]:
            async with semaphore:
                return await self._verify_single_cluster(cluster, entity_lookup)
        
        # Process all clusters concurrently
        cluster_results = await asyncio.gather(
            *[process_cluster(cluster) for cluster in candidate_clusters],
            return_exceptions=True
        )
        
        # Merge results from all clusters
        final_name_map = {}
        successful_verifications = 0
        
        for i, result in enumerate(cluster_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing cluster {i}: {result}")
                continue
            
            if isinstance(result, dict) and result:
                final_name_map.update(result)
                successful_verifications += 1
        
        logger.info(
            f"LLM verification complete. {successful_verifications}/{len(candidate_clusters)} "
            f"clusters successfully processed"
        )
        
        return final_name_map
    
    async def _verify_single_cluster(
        self, 
        cluster: List[str], 
        entity_lookup: Dict[str, Dict]
    ) -> Dict[str, str]:
        """
        Verify a single cluster using LLM analysis.
        
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
                if source_id_str:
                    delimiter = PROMPTS.get("DEFAULT_RECORD_DELIMITER", "<SEP>")
                    chunk_ids = source_id_str.split(delimiter)
                    source_chunks_needed.update(chunk_ids)
                
                cluster_data.append({
                    "entity_name": entity_name,
                    "entity_description": entity.get("description", ""),
                    "entity_type": entity.get("entity_type", ""),
                    "source_ids": chunk_ids if source_id_str else []
                })
            
            if not cluster_data:
                logger.warning(f"No valid entities found in cluster: {cluster}")
                return {}
            
            # Fetch original text chunks
            source_chunks = await self._fetch_source_chunks(list(source_chunks_needed))
            chunk_lookup = {chunk["id"] if isinstance(chunk, dict) and "id" in chunk else str(i): chunk 
                          for i, chunk in enumerate(source_chunks) if chunk}
            
            # Add original text context to each entity
            for entity_data in cluster_data:
                contexts = []
                for chunk_id in entity_data["source_ids"]:
                    chunk = chunk_lookup.get(chunk_id)
                    if chunk and isinstance(chunk, dict):
                        content = chunk.get("content", "")
                        if content:
                            contexts.append(content)
                
                # Combine and truncate context if necessary
                combined_context = " ... ".join(contexts)
                if len(combined_context) > self.config.max_context_tokens * 4:  # Rough token estimation
                    combined_context = combined_context[:self.config.max_context_tokens * 4] + "..."
                
                entity_data["original_text_context"] = combined_context or "No context available."
            
            # Prepare LLM prompt
            prompt_key = "entity_disambiguation"
            if prompt_key not in PROMPTS:
                logger.error(f"Required prompt '{prompt_key}' not found in PROMPTS")
                return {}
            
            prompt_template = PROMPTS[prompt_key]
            cluster_json = json.dumps(cluster_data, indent=2)
            
            prompt = prompt_template.format(input_json_for_cluster=cluster_json)
            
            # Call LLM
            llm_response = await self.best_model_func(prompt)
            
            # Parse and validate LLM response
            decision = await self._parse_and_validate_llm_decision(llm_response, cluster)
            
            if decision is None:
                logger.warning(f"Invalid LLM decision for cluster {cluster}")
                return {}
            
            # Process decision
            if decision["decision"] == "MERGE":
                canonical_name = decision["canonical_name"]
                aliases = decision["aliases"]
                
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
                confidence = decision.get("confidence_score", 0.0)
                if confidence < self.config.min_merge_confidence:
                    logger.info(
                        f"LLM decision confidence {confidence} below threshold "
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
                logger.info(f"DO_NOT_MERGE decision for cluster {cluster}")
                return {}
        
        except Exception as e:
            logger.error(f"Error verifying cluster {cluster}: {e}", exc_info=True)
            return {}
    
    async def _fetch_source_chunks(self, chunk_ids: List[str]) -> List[Optional[Dict]]:
        """
        Fetch source text chunks from the key-value store.
        
        Args:
            chunk_ids: List of chunk IDs to fetch
            
        Returns:
            List of chunk dictionaries or None for missing chunks
        """
        if not chunk_ids:
            return []
        
        try:
            # Filter out empty or invalid chunk IDs
            valid_chunk_ids = [cid for cid in chunk_ids if cid and cid.strip()]
            
            if not valid_chunk_ids:
                return []
            
            chunks = await self.text_chunks_kv.get_by_ids(valid_chunk_ids)
            return chunks
        
        except Exception as e:
            logger.error(f"Error fetching source chunks: {e}")
            return []
    
    async def _parse_and_validate_llm_decision(
        self, 
        llm_response: str, 
        cluster: List[str]
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
    Validate EDM configuration parameters and return a list of warnings/errors.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation messages (empty if all valid)
    """
    issues = []
    
    # Check thresholds
    lexical_threshold = config.get("edm_lexical_similarity_threshold", 0.85)
    if not 0.0 <= lexical_threshold <= 1.0:
        issues.append(f"edm_lexical_similarity_threshold must be between 0.0 and 1.0, got {lexical_threshold}")
    
    semantic_threshold = config.get("edm_semantic_similarity_threshold", 0.88)
    if not 0.0 <= semantic_threshold <= 1.0:
        issues.append(f"edm_semantic_similarity_threshold must be between 0.0 and 1.0, got {semantic_threshold}")
    
    # Check cluster size limits
    max_cluster_size = config.get("edm_max_cluster_size", 6)
    if not isinstance(max_cluster_size, int) or max_cluster_size < 2:
        issues.append(f"edm_max_cluster_size must be an integer >= 2, got {max_cluster_size}")
    
    # Check confidence threshold
    min_confidence = config.get("edm_min_merge_confidence", 0.8)
    if not 0.0 <= min_confidence <= 1.0:
        issues.append(f"edm_min_merge_confidence must be between 0.0 and 1.0, got {min_confidence}")
    
    # Check concurrency limits
    max_concurrent = config.get("edm_max_concurrent_llm_calls", 3)
    if not isinstance(max_concurrent, int) or max_concurrent < 1:
        issues.append(f"edm_max_concurrent_llm_calls must be an integer >= 1, got {max_concurrent}")
    
    # Warn about potentially problematic combinations
    if lexical_threshold > semantic_threshold:
        issues.append(f"Warning: lexical threshold ({lexical_threshold}) is higher than semantic threshold ({semantic_threshold}). This may reduce disambiguation effectiveness.")
    
    if max_cluster_size > 10:
        issues.append(f"Warning: edm_max_cluster_size ({max_cluster_size}) is quite large. This may impact LLM performance and accuracy.")
    
    return issues


def log_disambiguation_statistics(
    raw_nodes: List[Dict], 
    name_to_canonical_map: Dict[str, str],
    logger_instance = None
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
    failed_clusters: List[List[str]]
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
    successful_clusters = len([c for c in clusters_processed if any(entity in successful_merges for entity in c)])
    
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
            report += f"- **{canonical}** ← {', '.join(f'`{alias}`' for alias in aliases)}\n"
    else:
        report += "No successful merges.\n"
    
    if failed_clusters:
        report += f"\n## Failed Clusters ({len(failed_clusters)})\n"
        for i, cluster in enumerate(failed_clusters):
            report += f"{i+1}. {', '.join(f'`{entity}`' for entity in cluster)}\n"
    
    return report
