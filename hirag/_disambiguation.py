"""
Entity Disambiguation and Merging (EDM) Module

This module implements a robust three-stage pipeline for identifying and merging
entities that refer to the same concept but are named differently across text chunks.

The pipeline consists of:
1. Candidate Generation: Use lexical and semantic similarity of entity names
2. LLM Verification: Use context and original text to make final decisions
3. Graph Consolidation: Apply merge decisions to create the final graph

Author: HiRAG Development Team
"""

import asyncio
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
from thefuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseKVStorage, TextChunkSchema
from ._utils import logger, compute_mdhash_id, clean_str, convert_response_to_json
from ._op import _handle_entity_relation_summary, _merge_nodes_then_upsert, _merge_edges_then_upsert


class EntityDisambiguator:
    """
    A comprehensive entity disambiguation system that identifies and merges
    entities referring to the same concept but with different names.
    
    The system uses a conservative approach: it's better to leave duplicates
    than to incorrectly merge different entities.
    """
    
    def __init__(self, global_config: dict, text_chunks_kv: BaseKVStorage[TextChunkSchema], embedding_func):
        """
        Initialize the EntityDisambiguator.
        
        Args:
            global_config: Global HiRAG configuration dictionary
            text_chunks_kv: Key-value storage for text chunks (to retrieve source context)
            embedding_func: Function to generate embeddings for entity names
        """
        self.global_config = global_config
        self.text_chunks_kv = text_chunks_kv
        self.llm_func = global_config["best_model_func"]
        self.embedding_func = embedding_func
        self.convert_response_to_json_func = global_config["convert_response_to_json_func"]
        
        # Thresholds for candidate generation (configurable via global_config)
        self.name_sim_threshold = global_config.get("edm_name_sim_threshold", 0.90)
        self.name_emb_sim_threshold = global_config.get("edm_name_emb_sim_threshold", 0.95)
        
        # Maximum cluster size for LLM verification (to avoid overwhelming the LLM)
        self.max_cluster_size_for_llm = global_config.get("edm_max_cluster_size", 10)

    async def run(self, raw_nodes: List[Dict], raw_edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Main entry point for the EDM pipeline.
        
        Args:
            raw_nodes: List of raw entity dictionaries from extraction
            raw_edges: List of raw relationship dictionaries from extraction
            
        Returns:
            Tuple of (final_nodes, final_edges) after disambiguation and merging
        """
        logger.info(f"Starting Entity Disambiguation for {len(raw_nodes)} raw nodes.")
        
        if not raw_nodes:
            logger.info("No entities to disambiguate, returning empty results.")
            return [], raw_edges

        # Stage 1: Generate candidate clusters based on name similarity
        candidate_clusters = await self._generate_candidates(raw_nodes)
        logger.info(f"Generated {len(candidate_clusters)} candidate clusters for verification.")
        
        if not candidate_clusters:
            logger.info("No candidate clusters found, entities are sufficiently distinct.")
            return raw_nodes, raw_edges

        # Stage 2: LLM verification of candidate clusters
        llm_decisions = await self._verify_candidates_with_llm(candidate_clusters)
        logger.info(f"LLM verification complete. {sum(1 for d in llm_decisions if d['decision'] == 'MERGE')} merges approved.")

        # Stage 3: Consolidate the graph based on LLM decisions
        final_nodes, final_edges = await self._consolidate_graph(raw_nodes, raw_edges, llm_decisions)
        logger.info(f"Graph consolidated. Final node count: {len(final_nodes)}")
        
        return final_nodes, final_edges

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

        node_map = {n['entity_name']: n for n in non_temp_nodes}
        unique_names = list(node_map.keys())
        
        logger.info(f"Analyzing {len(unique_names)} unique non-temporary entity names for similarity.")

        # Step 1: Batch embed all unique entity names for semantic comparison
        logger.info(f"Embedding {len(unique_names)} unique entity names for disambiguation...")
        try:
            name_embeddings_arr = await self.embedding_func(unique_names)
            name_to_embedding = {name: emb for name, emb in zip(unique_names, name_embeddings_arr)}
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
        candidate_clusters = [cluster for cluster in clusters.values() if len(cluster) > 1]
        
        # Log cluster statistics
        if candidate_clusters:
            cluster_sizes = [len(cluster) for cluster in candidate_clusters]
            logger.info(f"Candidate clusters: {len(candidate_clusters)} clusters, "
                       f"sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        
        return candidate_clusters

    async def _verify_candidates_with_llm(self, candidate_clusters: List[List[Dict]]) -> List[Dict]:
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
                logger.warning(f"Cluster with {len(cluster)} entities exceeds max size {self.max_cluster_size_for_llm}, skipping LLM verification.")
        
        if not processable_clusters:
            logger.info("No clusters suitable for LLM verification.")
            return []
        
        logger.info(f"Processing {len(processable_clusters)} clusters through LLM verification...")
        
        # Process clusters in parallel for efficiency
        tasks = [self._process_single_cluster(cluster) for cluster in processable_clusters]
        llm_decisions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred during processing
        valid_decisions = []
        for i, decision in enumerate(llm_decisions):
            if isinstance(decision, Exception):
                logger.error(f"Error processing cluster {i}: {decision}")
                # Default to DO_NOT_MERGE for safety
                valid_decisions.append({
                    "decision": "DO_NOT_MERGE",
                    "confidence_score": 0.0,
                    "justification": f"Error during processing: {str(decision)}",
                    "cluster_entities": [entity["entity_name"] for entity in processable_clusters[i]]
                })
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
        logger.debug(f"Processing cluster with entities: {[e['entity_name'] for e in cluster]}")
        
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
                        logger.warning(f"Could not retrieve context for source_id {source_id}: {e}")
            
            # Combine contexts if multiple source chunks
            combined_context = " ... ".join(original_contexts) if original_contexts else "No context available"
            
            cluster_with_context.append({
                "entity_name": entity["entity_name"],
                "entity_description": entity["description"],
                "original_text_context": combined_context
            })
        
        # Build the LLM prompt with the cluster data
        prompt = self._build_disambiguation_prompt(cluster_with_context)
        
        try:
            # Call LLM for disambiguation decision
            llm_response = await self.llm_func(prompt)
            
            # Parse the JSON response
            decision_data = await self.convert_response_to_json_func(llm_response)
            
            # Validate the response structure
            if not self._validate_llm_decision(decision_data, cluster):
                logger.warning(f"Invalid LLM response for cluster, defaulting to DO_NOT_MERGE: {decision_data}")
                decision_data = {
                    "decision": "DO_NOT_MERGE",
                    "confidence_score": 0.0,
                    "justification": "Invalid LLM response format"
                }
            
            # Add cluster information for consolidation stage
            decision_data["cluster_entities"] = [entity["entity_name"] for entity in cluster]
            
            logger.debug(f"LLM decision for cluster: {decision_data['decision']} "
                        f"(confidence: {decision_data.get('confidence_score', 'N/A')})")
            
            return decision_data
            
        except Exception as e:
            logger.error(f"Error during LLM processing for cluster: {e}")
            return {
                "decision": "DO_NOT_MERGE",
                "confidence_score": 0.0,
                "justification": f"Error during LLM processing: {str(e)}",
                "cluster_entities": [entity["entity_name"] for entity in cluster]
            }

    def _build_disambiguation_prompt(self, cluster_with_context: List[Dict]) -> str:
        """
        Build the LLM prompt for entity disambiguation.
        
        Args:
            cluster_with_context: List of entities with their original text context
            
        Returns:
            Formatted prompt string for the LLM
        """
        # Get the prompt template (will be added to prompts.py)
        prompt_template = """You are a meticulous knowledge graph curator and an expert in Combinatory Logic. Your task is to perform entity disambiguation. You will be given a cluster of entities that are suspected to be aliases for the same underlying concept. Your job is to analyze the provided evidence and make a definitive judgment.

# Goal
Determine if the entities in the provided list are aliases for one another. If they are, you must select the best canonical name. If they are not, you must state that they should not be merged.

# Rules of Judgment
1. **Identity vs. Similarity**: Do not merge entities that are merely similar. Merge only if you are highly confident they refer to the exact same concept, postulate, or object. Subtle differences in definitions matter.
2. **Context is Ground Truth**: The `original_text_context` is the most important piece of evidence. An entity's meaning is defined by its use in the source document. Your justification MUST reference specific phrases from this context.
3. **Be Conservative**: If there is any ambiguity or insufficient evidence to prove the entities are identical, you MUST decide "DO_NOT_MERGE". It is better to leave two aliases separate than to incorrectly merge two distinct concepts.
4. **Canonical Name Selection**: If you decide to merge, the canonical name should be the most precise and commonly used term. Prefer formal, shorter names (e.g., "CL_η") over descriptive, longer ones (e.g., "The theory CL_η").

# Input Data
You will be provided with a JSON list of candidate entities. Each entity object has the following structure:
{{
  "entity_name": "The name extracted for this entity.",
  "entity_description": "The description generated for this entity.",
  "original_text_context": "The full text chunk from which this entity was extracted."
}}

# Output Format
You MUST respond with a single, well-formed JSON object with the following structure. Do not add any text outside of this JSON object.

- For a MERGE decision:
{{
  "decision": "MERGE",
  "canonical_name": "<The chosen canonical name>",
  "aliases": ["<list of other names to be merged>"],
  "confidence_score": <A float from 0.0 to 1.0 indicating your confidence in this decision>,
  "justification": "A detailed explanation for why these entities are identical, referencing specific evidence from the provided context."
}}

- For a DO_NOT_MERGE decision:
{{
  "decision": "DO_NOT_MERGE",
  "confidence_score": <A float from 0.0 to 1.0 indicating your confidence in this decision>,
  "justification": "A detailed explanation of the subtle differences that prevent these entities from being merged, referencing specific evidence from the provided context."
}}

# Real Data

Candidate Cluster:
{input_json_for_cluster}

# Your Decision:"""
        
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
                logger.warning(f"LLM decision names {all_names} don't match cluster names {cluster_names}")
                return False
        
        return True

    async def _consolidate_graph(self, raw_nodes: List[Dict], raw_edges: List[Dict], 
                               llm_decisions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Stage 3: Consolidate the graph by applying LLM merge decisions.
        
        Args:
            raw_nodes: Original list of raw entity dictionaries
            raw_edges: Original list of raw relationship dictionaries  
            llm_decisions: List of LLM decisions from Stage 2
            
        Returns:
            Tuple of (final_nodes, final_edges) after applying merge decisions
        """
        logger.info("Consolidating graph based on LLM decisions...")
        
        # Build a mapping from entity names to their canonical names
        name_to_canonical = {}
        canonical_to_merged_data = {}
        
        # Process merge decisions
        merge_count = 0
        for decision in llm_decisions:
            if decision["decision"] == "MERGE":
                canonical_name = decision["canonical_name"]
                aliases = decision["aliases"]
                
                # Map all names (canonical + aliases) to the canonical name
                for name in [canonical_name] + aliases:
                    name_to_canonical[name] = canonical_name
                canonical_to_merged_data[canonical_name] = decision
                merge_count += 1
        
        logger.info(f"Applying {merge_count} merge decisions affecting {len(name_to_canonical)} entity names.")
        
        # Consolidate nodes
        final_nodes = []
        processed_canonical_names = set()
        
        for node in raw_nodes:
            entity_name = node["entity_name"]
            canonical_name = name_to_canonical.get(entity_name, entity_name)
            
            if canonical_name in processed_canonical_names:
                # This entity is part of a merge that we've already processed
                continue
            
            if canonical_name != entity_name:
                # This entity is being merged - create merged entity
                merged_node = await self._create_merged_entity(
                    canonical_name, raw_nodes, name_to_canonical, canonical_to_merged_data[canonical_name]
                )
                final_nodes.append(merged_node)
                processed_canonical_names.add(canonical_name)
            else:
                # This entity is not being merged - keep as is
                final_nodes.append(node)
        
        # Consolidate edges by updating entity references
        final_edges = []
        for edge in raw_edges:
            src_canonical = name_to_canonical.get(edge["src_id"], edge["src_id"])
            tgt_canonical = name_to_canonical.get(edge["tgt_id"], edge["tgt_id"])
            
            # Update edge with canonical names
            updated_edge = edge.copy()
            updated_edge["src_id"] = src_canonical
            updated_edge["tgt_id"] = tgt_canonical
            
            final_edges.append(updated_edge)
        
        logger.info(f"Graph consolidation complete. {len(raw_nodes)} -> {len(final_nodes)} nodes, "
                   f"{len(raw_edges)} -> {len(final_edges)} edges.")
        
        return final_nodes, final_edges

    async def _create_merged_entity(self, canonical_name: str, raw_nodes: List[Dict], 
                                  name_to_canonical: Dict[str, str], merge_decision: Dict) -> Dict:
        """
        Create a merged entity from multiple entities that have been determined to be aliases.
        
        Args:
            canonical_name: The canonical name for the merged entity
            raw_nodes: All raw nodes 
            name_to_canonical: Mapping from entity names to canonical names
            merge_decision: The LLM decision that determined this merge
            
        Returns:
            Merged entity dictionary
        """
        # Find all nodes that map to this canonical name
        entities_to_merge = []
        for node in raw_nodes:
            if name_to_canonical.get(node["entity_name"], node["entity_name"]) == canonical_name:
                entities_to_merge.append(node)
        
        if not entities_to_merge:
            logger.error(f"No entities found for canonical name {canonical_name}")
            return {}
        
        # Merge entity data using similar logic to _merge_nodes_then_upsert
        entity_types = [entity["entity_type"] for entity in entities_to_merge]
        descriptions = [entity["description"] for entity in entities_to_merge]
        source_ids = []
        is_temporary_values = []
        
        for entity in entities_to_merge:
            source_ids.extend(entity.get("source_id", "").split("<SEP>"))
            is_temporary_values.append(entity.get("is_temporary", False))
        
        # Choose most common entity type
        from collections import Counter
        most_common_type = Counter(entity_types).most_common(1)[0][0]
        
        # Combine descriptions
        unique_descriptions = list(set(descriptions))
        combined_description = "<SEP>".join(sorted(unique_descriptions))
        
        # Combine source IDs
        unique_source_ids = list(set([sid for sid in source_ids if sid.strip()]))
        combined_source_id = "<SEP>".join(unique_source_ids)
        
        # Determine final is_temporary status (majority vote)
        is_temporary_count = sum(1 for val in is_temporary_values if val)
        final_is_temporary = is_temporary_count / len(is_temporary_values) > 0.5
        
        # Create merged entity
        merged_entity = {
            "entity_name": canonical_name,
            "entity_type": most_common_type,
            "description": combined_description,
            "source_id": combined_source_id,
            "is_temporary": final_is_temporary
        }
        
        # Copy any additional fields from the first entity (like embeddings)
        for key, value in entities_to_merge[0].items():
            if key not in merged_entity:
                merged_entity[key] = value
        
        logger.debug(f"Created merged entity '{canonical_name}' from {len(entities_to_merge)} aliases.")
        
        return merged_entity