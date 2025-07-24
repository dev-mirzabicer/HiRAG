import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken

from hirag._storage.gdb_neo4j import Neo4jStorage

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
    _merge_nodes_then_upsert,
    _merge_edges_then_upsert,
    _handle_entity_relation_summary,
    # All query functions are now deprecated and will be removed/replaced by the agent's logic
)
from ._disambiguation import EntityDisambiguator
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
    
    # Entity Disambiguation and Merging (EDM) Configuration
    edm_name_sim_threshold: float = 0.90  # Lexical similarity threshold for candidate generation
    edm_name_emb_sim_threshold: float = 0.95  # Semantic similarity threshold for candidate generation
    edm_max_cluster_size: int = 10  # Maximum cluster size for LLM verification

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
        
        # Initialize the Entity Disambiguator
        self.disambiguator = EntityDisambiguator(
            global_config=asdict(self),
            text_chunks_kv=self.text_chunks,
            embedding_func=self.embedding_func
        )

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
        # --- ainsert logic remains the same, as it's the core indexing pipeline ---
        await self._insert_start()
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

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
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            await self.community_reports.drop()

            if not self.enable_hierachical_mode:
                logger.info("[Entity Extraction with EDM]...")
                # Get raw entities and relationships without graph insertion
                raw_nodes, raw_edges = await self.entity_extraction_func(
                    inserting_chunks,
                    global_config=asdict(self),
                )
            else:
                logger.info("[Hierarchical Entity Extraction with EDM]...")
                # Get raw entities and relationships without graph insertion  
                raw_nodes, raw_edges = await self.hierarchical_entity_extraction_func(
                    inserting_chunks,
                    global_config=asdict(self),
                )

            if not raw_nodes and not raw_edges:
                logger.warning("No new entities or relationships found")
                return
            
            logger.info(f"[Entity Disambiguation] Processing {len(raw_nodes)} entities...")
            # Apply Entity Disambiguation and Merging pipeline
            final_nodes, final_edges = await self.disambiguator.run(raw_nodes, raw_edges)
            
            logger.info(f"[Graph Consolidation] Upserting {len(final_nodes)} nodes and {len(final_edges)} edges...")
            # Consolidate the disambiguated entities into the knowledge graph
            await self._upsert_consolidated_graph(final_nodes, final_edges)

            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _upsert_consolidated_graph(self, final_nodes: List[Dict], final_edges: List[Dict]):
        """
        Upsert the consolidated (disambiguated) nodes and edges to the knowledge graph.
        
        This method takes the output from the Entity Disambiguation pipeline and properly
        integrates it into the graph storage, handling entity summaries and vector indexing.
        
        Args:
            final_nodes: List of disambiguated entity dictionaries
            final_edges: List of disambiguated relationship dictionaries
        """
        if not final_nodes and not final_edges:
            logger.warning("No nodes or edges to upsert")
            return
            
        logger.info(f"Upserting {len(final_nodes)} nodes and {len(final_edges)} edges to graph storage...")
        
        # Process nodes: merge descriptions and upsert to graph
        node_tasks = []
        for node in final_nodes:
            task = self._process_and_upsert_node(node)
            node_tasks.append(task)
        
        if node_tasks:
            await asyncio.gather(*node_tasks)
        
        # Process edges: merge descriptions and upsert to graph  
        edge_tasks = []
        for edge in final_edges:
            task = self._process_and_upsert_edge(edge)
            edge_tasks.append(task)
            
        if edge_tasks:
            await asyncio.gather(*edge_tasks)
            
        logger.info("Graph consolidation complete")

    async def _process_and_upsert_node(self, node: Dict):
        """
        Process and upsert a single node to the graph storage.
        
        Args:
            node: Entity dictionary to be upserted
        """
        entity_name = node["entity_name"]
        
        # Check if entity already exists in graph
        existing_node = await self.chunk_entity_relation_graph.get_node(entity_name)
        
        if existing_node:
            # Merge with existing node
            merged_node = await _merge_nodes_then_upsert(
                entity_name, 
                [node], 
                self.chunk_entity_relation_graph,
                asdict(self)
            )
        else:
            # Summarize description if needed
            if len(node.get("description", "")) > self.entity_summary_to_max_tokens:
                logger.debug(f"Summarizing description for entity: {entity_name}")
                node["description"] = await _handle_entity_relation_summary(
                    node["description"], 
                    asdict(self)
                )
            
            # Upsert new node
            await self.chunk_entity_relation_graph.upsert_node(entity_name, node)
        
        # Update entity vector database if enabled
        if self.enable_local and self.entities_vdb and not node.get("is_temporary", False):
            entity_data = {
                entity_name: {
                    "content": node.get("description", ""),
                    "entity_name": entity_name
                }
            }
            await self.entities_vdb.upsert(entity_data)

    async def _process_and_upsert_edge(self, edge: Dict):
        """
        Process and upsert a single edge to the graph storage.
        
        Args:
            edge: Relationship dictionary to be upserted  
        """
        src_id = edge["src_id"]
        tgt_id = edge["tgt_id"]
        
        # Check if edge already exists
        if await self.chunk_entity_relation_graph.has_edge(src_id, tgt_id):
            # Merge with existing edge
            await _merge_edges_then_upsert(
                (src_id, tgt_id),
                [edge],
                self.chunk_entity_relation_graph,
                asdict(self)
            )
        else:
            # Summarize description if needed
            if len(edge.get("description", "")) > self.entity_summary_to_max_tokens:
                edge["description"] = await _handle_entity_relation_summary(
                    edge["description"],
                    asdict(self)
                )
            
            # Upsert new edge
            await self.chunk_entity_relation_graph.upsert_edge(
                src_id, tgt_id, edge
            )

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
