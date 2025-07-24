import asyncio
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from hirag.prompt import PROMPTS
from thefuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseKVStorage, TextChunkSchema
from ._utils import logger, EmbeddingFunc


class EntityDisambiguator:
    """
    Identifies potential entity aliases and produces a mapping to canonical names.
    This class is ONLY responsible for the decision-making, not for merging data.
    """

    def __init__(
        self,
        global_config: dict,
        text_chunks_kv: BaseKVStorage[TextChunkSchema],
        embedding_func: EmbeddingFunc,
    ):
        self.global_config = global_config
        self.text_chunks_kv = text_chunks_kv
        self.llm_func = global_config["best_model_func"]
        self.embedding_func = embedding_func
        self.convert_response_to_json_func = global_config[
            "convert_response_to_json_func"
        ]
        self.name_sim_threshold = global_config.get("edm_name_sim_threshold", 0.90)
        self.name_emb_sim_threshold = global_config.get(
            "edm_name_emb_sim_threshold", 0.95
        )
        self.max_cluster_size_for_llm = global_config.get("edm_max_cluster_size", 10)

    async def run(self, raw_nodes: List[Dict]) -> Dict[str, str]:
        """
        Main entry point for the EDM pipeline.

        Args:
            raw_nodes: List of raw entity dictionaries from extraction

        Returns:
            A dictionary mapping alias names to their canonical name.
            e.g., {"The theory CL_η": "CL_η", "CL_η system": "CL_η"}
        """
        logger.info(f"Starting Entity Disambiguation for {len(raw_nodes)} raw nodes.")
        if not raw_nodes:
            return {}

        candidate_clusters = await self._generate_candidates(raw_nodes)
        if not candidate_clusters:
            logger.info("No candidate clusters for disambiguation.")
            return {}

        llm_decisions = await self._verify_candidates_with_llm(candidate_clusters)

        # Build the final mapping from the LLM's decisions
        name_to_canonical = {}
        for decision in llm_decisions:
            if decision.get("decision") == "MERGE":
                canonical_name = decision["canonical_name"]
                for alias in decision.get("aliases", []):
                    name_to_canonical[alias] = canonical_name

        logger.info(
            f"Disambiguation complete. Found {len(name_to_canonical)} aliases to be merged."
        )
        return name_to_canonical

    async def _generate_candidates(self, raw_nodes: List[Dict]) -> List[List[Dict]]:
        """
        Stage 1: Generate candidate clusters based on lexical and semantic similarity of entity NAMES.

        This stage uses Union-Find data structure for efficient clustering based on:
        1. Lexical similarity (string-based comparison)
        2. Semantic similarity (embedding-based comparison of entity names)

        Args:
            raw_nodes: List of raw entity dictionaries

        Returns:
            List of candidate clusters, where each cluster contains entities suspected to be aliases
        """
        # Focus only on non-temporary entities for disambiguation
        non_temp_nodes = [n for n in raw_nodes if not n.get("is_temporary", False)]
        if len(non_temp_nodes) < 2:
            logger.info("Less than 2 non-temporary entities, no disambiguation needed.")
            return []

        node_map = {n["entity_name"]: n for n in non_temp_nodes}
        unique_names = list(node_map.keys())

        logger.info(
            f"Analyzing {len(unique_names)} unique non-temporary entity names for similarity."
        )

        # Step 1: Batch embed all unique entity names for semantic comparison
        logger.info(
            f"Embedding {len(unique_names)} unique entity names for disambiguation..."
        )
        try:
            name_embeddings_arr = await self.embedding_func(unique_names)
            name_to_embedding = {
                name: emb for name, emb in zip(unique_names, name_embeddings_arr)
            }
        except Exception as e:
            logger.error(f"Failed to generate embeddings for entity names: {e}")
            # Fall back to lexical similarity only
            name_embeddings_arr = None
            name_to_embedding = {}

        # Step 2: Initialize Union-Find data structure for clustering
        parent = {name: name for name in unique_names}

        def find(name):
            if parent[name] == name:
                return name
            parent[name] = find(parent[name])  # Path compression
            return parent[name]

        def union(name1, name2):
            root1, root2 = find(name1), find(name2)
            if root1 != root2:
                parent[root2] = root1

        # Step 3a: Lexical Similarity Pass
        logger.info("Performing lexical similarity clustering...")
        lexical_pairs = 0
        for i in range(len(unique_names)):
            for j in range(i + 1, len(unique_names)):
                name1, name2 = unique_names[i], unique_names[j]
                # Use token_sort_ratio to handle word order differences
                score = fuzz.token_sort_ratio(name1, name2) / 100.0
                if score > self.name_sim_threshold:
                    union(name1, name2)
                    lexical_pairs += 1
        logger.info(f"Found {lexical_pairs} lexically similar pairs.")

        # Step 3b: Semantic Similarity Pass (if embeddings are available)
        semantic_pairs = 0
        if name_embeddings_arr is not None:
            logger.info("Performing semantic similarity clustering...")
            sim_matrix = cosine_similarity(name_embeddings_arr)
            for i in range(len(unique_names)):
                for j in range(i + 1, len(unique_names)):
                    if sim_matrix[i, j] > self.name_emb_sim_threshold:
                        union(unique_names[i], unique_names[j])
                        semantic_pairs += 1
            logger.info(f"Found {semantic_pairs} semantically similar pairs.")

        # Step 4: Consolidate clusters from Union-Find structure
        clusters = defaultdict(list)
        for name in unique_names:
            root = find(name)
            clusters[root].append(node_map[name])

        # Return only clusters with more than one member
        candidate_clusters = [
            cluster for cluster in clusters.values() if len(cluster) > 1
        ]

        # Log cluster statistics
        if candidate_clusters:
            cluster_sizes = [len(cluster) for cluster in candidate_clusters]
            logger.info(
                f"Candidate clusters: {len(candidate_clusters)} clusters, "
                f"sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes) / len(cluster_sizes):.1f}"
            )

        return candidate_clusters

    async def _verify_candidates_with_llm(
        self, candidate_clusters: List[List[Dict]]
    ) -> List[Dict]:
        """
        Stage 2: LLM Verification of candidate clusters.

        For each candidate cluster, use the LLM to determine if the entities
        are truly the same concept based on their original text context.

        Args:
            candidate_clusters: List of candidate clusters from Stage 1

        Returns:
            List of LLM decision dictionaries
        """
        # Filter clusters that are too large for LLM processing
        processable_clusters = []
        for cluster in candidate_clusters:
            if len(cluster) <= self.max_cluster_size_for_llm:
                processable_clusters.append(cluster)
            else:
                logger.warning(
                    f"Cluster with {len(cluster)} entities exceeds max size {self.max_cluster_size_for_llm}, skipping LLM verification."
                )

        if not processable_clusters:
            logger.info("No clusters suitable for LLM verification.")
            return []

        logger.info(
            f"Processing {len(processable_clusters)} clusters through LLM verification..."
        )

        # Process clusters in parallel for efficiency
        tasks = [
            self._process_single_cluster(cluster) for cluster in processable_clusters
        ]
        llm_decisions = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during processing
        valid_decisions = []
        for i, decision in enumerate(llm_decisions):
            if isinstance(decision, Exception):
                logger.error(f"Error processing cluster {i}: {decision}")
                # Default to DO_NOT_MERGE for safety
                valid_decisions.append(
                    {
                        "decision": "DO_NOT_MERGE",
                        "confidence_score": 0.0,
                        "justification": f"Error during processing: {str(decision)}",
                        "cluster_entities": [
                            entity["entity_name"] for entity in processable_clusters[i]
                        ],
                    }
                )
            else:
                valid_decisions.append(decision)

        return valid_decisions

    async def _process_single_cluster(self, cluster: List[Dict]) -> Dict:
        """
        Process a single candidate cluster through LLM verification.

        Args:
            cluster: List of entity dictionaries suspected to be aliases

        Returns:
            LLM decision dictionary with merge/no-merge decision
        """
        logger.debug(
            f"Processing cluster with entities: {[e['entity_name'] for e in cluster]}"
        )

        # Fetch original text context for each entity
        cluster_with_context = []
        for entity in cluster:
            source_ids = entity.get("source_id", "").split("<SEP>")
            original_contexts = []

            for source_id in source_ids:
                if source_id:
                    try:
                        chunk_data = await self.text_chunks_kv.get(source_id)
                        if chunk_data:
                            original_contexts.append(chunk_data["content"])
                    except Exception as e:
                        logger.warning(
                            f"Could not retrieve context for source_id {source_id}: {e}"
                        )

            # Combine contexts if multiple source chunks
            combined_context = (
                " ... ".join(original_contexts)
                if original_contexts
                else "No context available"
            )

            cluster_with_context.append(
                {
                    "entity_name": entity["entity_name"],
                    "entity_description": entity["description"],
                    "original_text_context": combined_context,
                }
            )

        # Build the LLM prompt with the cluster data
        prompt = self._build_disambiguation_prompt(cluster_with_context)

        try:
            # Call LLM for disambiguation decision
            llm_response = await self.llm_func(prompt)

            # Parse the JSON response
            decision_data = await self.convert_response_to_json_func(llm_response)

            # Validate the response structure
            if not self._validate_llm_decision(decision_data, cluster):
                logger.warning(
                    f"Invalid LLM response for cluster, defaulting to DO_NOT_MERGE: {decision_data}"
                )
                decision_data = {
                    "decision": "DO_NOT_MERGE",
                    "confidence_score": 0.0,
                    "justification": "Invalid LLM response format",
                }

            # Add cluster information for consolidation stage
            decision_data["cluster_entities"] = [
                entity["entity_name"] for entity in cluster
            ]

            logger.debug(
                f"LLM decision for cluster: {decision_data['decision']} "
                f"(confidence: {decision_data.get('confidence_score', 'N/A')})"
            )

            return decision_data

        except Exception as e:
            logger.error(f"Error during LLM processing for cluster: {e}")
            return {
                "decision": "DO_NOT_MERGE",
                "confidence_score": 0.0,
                "justification": f"Error during LLM processing: {str(e)}",
                "cluster_entities": [entity["entity_name"] for entity in cluster],
            }

    def _build_disambiguation_prompt(self, cluster_with_context: List[Dict]) -> str:
        """
        Build the LLM prompt for entity disambiguation.

        Args:
            cluster_with_context: List of entities with their original text context

        Returns:
            Formatted prompt string for the LLM
        """
        # Get the prompt template
        prompt_template = PROMPTS["entity_disambiguation"]

        # Convert cluster data to JSON string
        import json

        cluster_json = json.dumps(cluster_with_context, indent=2, ensure_ascii=False)

        return prompt_template.format(input_json_for_cluster=cluster_json)

    def _validate_llm_decision(self, decision_data: Dict, cluster: List[Dict]) -> bool:
        """
        Validate the structure and content of an LLM decision.

        Args:
            decision_data: The parsed LLM response
            cluster: The original cluster of entities

        Returns:
            True if the decision is valid, False otherwise
        """
        if not isinstance(decision_data, dict):
            return False

        # Check required fields
        required_fields = ["decision", "confidence_score", "justification"]
        for field in required_fields:
            if field not in decision_data:
                return False

        # Validate decision value
        if decision_data["decision"] not in ["MERGE", "DO_NOT_MERGE"]:
            return False

        # Validate confidence score
        confidence = decision_data.get("confidence_score")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False

        # For MERGE decisions, validate merge-specific fields
        if decision_data["decision"] == "MERGE":
            if "canonical_name" not in decision_data or "aliases" not in decision_data:
                return False

            canonical_name = decision_data["canonical_name"]
            aliases = decision_data["aliases"]

            if not isinstance(canonical_name, str) or not canonical_name.strip():
                return False

            if not isinstance(aliases, list):
                return False

            # Check that canonical_name + aliases covers all entities in cluster
            all_names = set([canonical_name] + aliases)
            cluster_names = set([entity["entity_name"] for entity in cluster])
            if all_names != cluster_names:
                logger.warning(
                    f"LLM decision names {all_names} don't match cluster names {cluster_names}"
                )
                return False

        return True
