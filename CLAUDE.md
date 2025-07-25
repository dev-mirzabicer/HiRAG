Welcome to HiRAG. This document is the definitive guide for any agent or developer interacting with this codebase. By understanding this guide, you will gain an expert-level comprehension of the system's architecture, capabilities, and operational workflows.

## 1. Mission and Philosophy

**HiRAG is not just another RAG system; it is a knowledge structuring engine.**

Its primary mission is to ingest unstructured text, automatically build a multi-layered, hierarchical knowledge graph from it, and equip AI agents with a powerful toolkit to reason over this structured knowledge.

The core philosophy is **"Structure First, Answers Second."** Instead of performing a flat semantic search over raw text chunks, HiRAG creates a rich, interconnected graph of concepts. This graph has multiple levels of abstraction, from fine-grained entities and relationships to high-level conceptual communities. This hierarchical structure allows an agent to perform complex reasoning, starting from a high-level overview and progressively drilling down to the most granular details and source text.

### Core Architectural Flow

The HiRAG system follows a sophisticated ingestion and querying pipeline:

1.  **Ingestion & Chunking**: Raw documents are processed, deduplicated, and broken down into manageable, token-aware text chunks.
2.  **Hierarchical Entity & Relation Extraction**: A powerful LLM-based process extracts entities, their types, descriptions, and relationships. Crucially, it also performs **hierarchical clustering**, summarizing groups of related entities into higher-level concepts, building the graph's vertical structure.
3.  **Entity Disambiguation & Merging (EDM)**: A critical pipeline step that identifies and merges entity aliases (e.g., "CL_η" and "The theory CL_η"). This ensures the knowledge graph is clean, consistent, and free of redundant nodes.
4.  **Graph Construction**: The disambiguated entities and relations are upserted into a graph database backend (like NetworkX or Neo4j), forming a rich, interconnected knowledge graph.
5.  **Community Detection & Reporting**: The system runs a clustering algorithm (e.g., Leiden) on the graph to identify dense communities of related concepts. It then uses an LLM to generate detailed, human-readable reports for each community, creating the highest level of abstraction.
6.  **Agentic Querying**: The system exposes a suite of powerful, granular tools for an AI agent. Instead of a single query function, the agent can now explore the graph, find entities, trace relationships, discover reasoning paths, and pull context from different levels of the hierarchy to construct its own answers.

## 2. Directory Structure

The codebase is organized logically to separate storage backends, core logic, and configuration.

```
└── ./
    └── hirag
        ├── _storage                # Concrete storage backend implementations
        │   ├── __init__.py
        │   ├── gdb_neo4j.py        # Neo4j graph database backend
        │   ├── gdb_networkx.py     # In-memory NetworkX graph backend
        │   ├── kv_json.py          # Simple file-based JSON key-value store
        │   └── vdb_nanovectordb.py # Simple file-based vector database
        ├── __init__.py
        ├── _cluster_utils.py       # Logic for hierarchical entity clustering
        ├── _disambiguation.py      # The Entity Disambiguation & Merging (EDM) pipeline
        ├── _llm.py                 # Abstractions for calling different LLM providers
        ├── _op.py                  # Core data processing operations (extraction, merging, reporting)
        ├── _splitter.py            # Advanced text and token splitting utilities
        ├── _utils.py               # General utility functions (hashing, logging, etc.)
        ├── _validation.py          # Centralized data validation functions
        ├── base.py                 # Core data schemas and abstract storage classes
        ├── hirag.py                # The main HiRAG class and central orchestrator
        └── prompt.py               # The "brain" of the system: all LLM prompts
```

## 3. Core Concepts & Data Structures (`hirag/base.py`)

The foundation of HiRAG is built on a set of well-defined abstract classes and data schemas.

*   **`StorageNameSpace`**: A base class for all storage modules, ensuring they have a `namespace` and access to the global configuration. It defines lifecycle callbacks: `index_start_callback`, `index_done_callback`.
*   **`BaseKVStorage[T]`**: An abstract interface for key-value stores (e.g., for text chunks, community reports).
*   **`BaseVectorStorage`**: An abstract interface for vector databases, defining `query` and `upsert` methods.
*   **`BaseGraphStorage`**: An abstract interface for graph databases, defining essential graph operations like `upsert_node`, `get_edge`, `clustering`, `community_schema`, etc.

### Key Data Schemas:

*   **`TextChunkSchema`**: Represents a piece of text derived from a full document.
    *   `tokens`: Number of tokens in the chunk.
    *   `content`: The actual text content.
    *   `full_doc_id`: The ID of the source document.
    *   `chunk_order_index`: The position of this chunk within the document.
*   **`SingleCommunitySchema` & `CommunitySchema`**: Represent a detected community in the graph.
    *   `level`: The hierarchical level of the community.
    *   `title`: A human-readable title.
    *   `nodes` & `edges`: The components of the community.
    *   `chunk_ids`: Source text chunks associated with the community's nodes.
    *   `sub_communities`: A list of child community IDs, forming the hierarchy.
    *   `report_string` & `report_json` (in `CommunitySchema`): The LLM-generated analysis of the community.
*   **`QueryParam`**: A dataclass that allows fine-grained control over the querying and context-building process, specifying things like retrieval depth (`level`), number of entities (`top_k`), and token limits for different context types.

---

## 4. File-by-File Deep Dive

### `hirag/hirag.py`: The Central Orchestrator

This is the most important file, containing the main `HiRAG` class that brings all the components together.

#### **`HiRAG` Class & Configuration**

The class is a dataclass holding the entire system's configuration. Key parameters include:

*   **Storage & Backend Selection**:
    *   `working_dir`: Where all cached files and databases are stored.
    *   `key_string_value_json_storage_cls`: The class to use for KV storage (default: `JsonKVStorage`).
    *   `vector_db_storage_cls`: The class for vector storage (default: `NanoVectorDBStorage`).
    *   `graph_storage_cls`: The class for graph storage (default: `NetworkXStorage`). Can be set to `Neo4jStorage`.
*   **LLM & Embedding Configuration**:
    *   `using_azure_openai` / `using_gemini`: Booleans to automatically switch to Azure or Gemini models.
    *   `best_model_func` / `cheap_model_func`: Allows specifying different models for complex reasoning vs. routine tasks.
    *   `embedding_func`: The function used to generate vector embeddings.
    *   `enable_llm_cache`: A boolean to cache LLM responses, saving time and cost on repeated runs.
*   **Ingestion & Processing Control**:
    *   `chunk_func`, `chunk_token_size`, `chunk_overlap_token_size`: Control how text is split into chunks.
    *   `graph_cluster_algorithm`: The algorithm for community detection (default: `"leiden"`).
    *   `enable_hierachical_mode`: A boolean to enable/disable the hierarchical entity extraction and clustering.
*   **Entity Disambiguation & Merging (EDM) Configuration**:
    *   `enable_entity_disambiguation`: Master switch for the EDM pipeline.
    *   `edm_lexical_similarity_threshold`, `edm_semantic_similarity_threshold`: Thresholds for finding potential entity aliases.
    *   `edm_max_cluster_size`: Safety limit to prevent sending excessively large alias clusters to the LLM.
    *   `edm_min_merge_confidence`: The minimum confidence score required from the LLM to perform a merge.
    *   `enable_entity_names_vdb`: A crucial performance optimization that stores entity names in a dedicated vector DB for faster semantic similarity checks.

#### **`__post_init__`**

This method initializes the entire HiRAG ecosystem based on the configuration. It:
1.  Sets up the working directory.
2.  Switches LLM and embedding functions if Azure or Gemini is specified.
3.  Instantiates all storage backends (`full_docs`, `text_chunks`, `community_reports`, `chunk_entity_relation_graph`, `entities_vdb`, etc.).
4.  Wraps the LLM and embedding functions with rate limiters (`limit_async_func_call`) and caching logic.
5.  Initializes the `EntityDisambiguator` if `enable_entity_disambiguation` is true, passing it the necessary storage backends and configuration.

#### **`ainsert(string_or_strings)`: The Ingestion Pipeline**

This is the main method for adding new knowledge to the system. It's a robust, multi-step asynchronous process:

1.  **Doc Processing**: Input strings are converted into documents and filtered against existing ones in `full_docs`.
2.  **Chunking**: `get_chunks` is called to split the new documents into `TextChunkSchema` objects.
3.  **Hierarchical Extraction**: `hierarchical_entity_extraction_func` (`extract_hierarchical_entities` by default) is called. This is a core innovation where an LLM extracts entities and then recursively summarizes clusters of them to build up a hierarchy. It returns `raw_nodes` and `raw_edges`.
4.  **Disambiguation**: If enabled, the `disambiguator.run(raw_nodes)` method is called. It takes the raw extracted nodes, finds potential aliases, and uses an LLM to produce a `name_to_canonical_map`.
5.  **Graph Upsertion**: The `_upsert_disambiguated_graph` method is called with the raw nodes, edges, and the disambiguation map. It intelligently merges the data before writing it to the graph storage backend.
6.  **Community Building**: After the graph is updated, `chunk_entity_relation_graph.clustering()` is called to detect communities. Then, `generate_community_report` uses an LLM to create analytical reports for each new community.
7.  **Finalization**: All changes are committed to the storage backends via their `index_done_callback` methods.

#### **Agent Toolkit: The New Query Paradigm**

The old monolithic `query` method is deprecated. It has been replaced by a suite of granular, asynchronous tools designed for an AI agent to use.

*   **Exploration & Discovery**:
    *   `aget_community_toc(level)`: Gets a "Table of Contents" of all communities at a given level. This is the agent's entry point for a high-level overview.
    *   `afind_entities(query, top_k)`: Performs a semantic search for entities in the vector DB.
*   **Drill-Down & Investigation**:
    *   `aget_community_details(community_id)`: "Zooms in" on a single community, returning its full report and a list of its key entities and relationships.
    *   `aget_entity_details(entity_name)`: Retrieves all stored data for a single entity.
    *   `aget_relationships(entity_name)`: Fetches all relationships connected to a given entity.
    *   `aget_source_text(entity_name)`: Traces an entity back to the original text chunk(s) it was extracted from.
*   **Reasoning & Pathfinding**:
    *   `afind_reasoning_path(start_entity, end_entity, ...)`: A powerful tool to find the shortest path between two entities. It intelligently abstracts over the graph backend (NetworkX or Neo4j) and the chosen algorithm (e.g., Dijkstra, Cypher SHORTEST).
    *   `afind_multiple_reasoning_paths(...)`: Finds *k* different paths, allowing an agent to explore multiple lines of reasoning.
    *   `afind_weighted_reasoning_path(...)`: Finds paths based on relationship weights, useful for considering relationship strength or cost.
*   **Holistic Context**:
    *   `aget_holistic_context(query, param)`: A high-level tool that runs a full hierarchical search and packages all the retrieved context (communities, entities, paths, source text) into a single, richly formatted string for the agent.

#### **Quality Assurance**

*   `validate_knowledge_graph_quality()`: A comprehensive method to run a suite of checks on the graph's integrity, consistency, and quality.
*   `generate_quality_report()`: Produces a human-readable markdown report based on the validation results.

### `hirag/_op.py`: Core Data Processing Operations

This file contains the "verbs" of the HiRAG system—the functions that perform the actual work of transforming data.

*   **Extraction Logic**:
    *   `_handle_single_entity_extraction` / `_handle_single_relationship_extraction`: These functions parse the structured string output from the LLM into Python dictionaries. They use the validation functions from `_validation.py`.
    *   `extract_entities` / `extract_hierarchical_entities`: These are the main orchestrators for the extraction process. They iterate through text chunks, format the appropriate prompts (`hi_entity_extraction`, `hi_relation_extraction`), call the LLM, and aggregate the results. `extract_hierarchical_entities` adds the crucial step of calling the `Hierarchical_Clustering` logic.
*   **Merging Logic**:
    *   `_merge_nodes_then_upsert`: This function takes all instances of an entity that should be merged. It intelligently combines their `source_id`s, determines the most common `entity_type`, and calls `_handle_entity_relation_summary` to synthesize a new, coherent description from all the old ones.
    *   `_merge_edges_then_upsert`: Similarly, this merges multiple instances of the same relationship, summing their weights and combining their descriptions.
*   **Community Reporting Logic**:
    *   `generate_community_report`: Orchestrates the creation of community reports. It iterates through the detected communities, level by level.
    *   `_pack_single_community_describe`: A complex function that prepares the context for the `community_report` prompt. It pulls all nodes and edges for a community, formats them into CSV-like strings, and intelligently truncates them to fit within the LLM's token limit. Crucially, if the community is too large, it will use the reports of its *sub-communities* as a summarized context.

### `hirag/_cluster_utils.py`: Hierarchical Clustering

This file implements the logic for building the "Hi" in HiRAG.

*   **`Hierarchical_Clustering.perform_clustering`**: This is the core method. It's an iterative process:
    1.  It takes the embeddings of the current layer of entities.
    2.  It uses UMAP (`global_cluster_embeddings`) to reduce the dimensionality of the embeddings, making clustering more effective.
    3.  It applies a Gaussian Mixture Model (`GMM_cluster`) to find conceptual clusters of entities.
    4.  For each resulting cluster, it packages the descriptions of the contained entities and sends them to an LLM using the `summary_clusters` prompt.
    5.  The LLM's job is to synthesize one or more *new, higher-level* entities that abstract the theme of the cluster.
    6.  These new entities, along with their own embeddings, become the input for the next layer of clustering.
    7.  The process stops when clustering no longer yields significant improvements (measured by cluster sparsity).

### `hirag/_disambiguation.py`: Entity Disambiguation & Merging (EDM)

This file contains the self-contained, robust pipeline for cleaning the knowledge graph.

*   **`EntityDisambiguator` Class**: The main orchestrator for the EDM process.
*   **`DisambiguationConfig`**: A dataclass to hold all configuration parameters for the pipeline.
*   **`run(raw_nodes)`**: The main entry point, which executes a two-stage process:
    1.  **Candidate Generation (`_generate_candidates`)**: This stage finds potential aliases without using an LLM. It uses a `UnionFind` data structure to efficiently group entities.
        *   `_lexical_similarity_pass`: Uses fuzzy string matching (`thefuzz`) to find lexically similar names (e.g., "SYSTEM CL" vs "CL SYSTEM").
        *   `_semantic_similarity_pass`: Uses the dedicated `entity_names_vdb` to perform fast vector searches, finding semantically similar names (e.g., "The theory of extensionality" vs "Extensionality property").
    2.  **LLM Verification (`_verify_candidates_with_llm`)**: This stage takes the candidate clusters and verifies them.
        *   For each cluster, `_verify_single_cluster` is called. It constructs a detailed prompt (`entity_disambiguation`) containing the names, types, descriptions, and—most importantly—the full original text context for each entity in the cluster.
        *   The LLM is asked to make a "MERGE" or "DO_NOT_MERGE" decision and return it in a structured JSON format, including a confidence score and justification.
        *   `_parse_and_validate_llm_decision` ensures the LLM's output is valid and adheres to the required format.
*   The final output of the `run` method is a `name_to_canonical_map` dictionary, which is then used during graph upsertion.

### `hirag/_storage/`: Storage Backends

This directory provides the concrete implementations for the storage interfaces defined in `base.py`.

*   **`gdb_networkx.py`**: An in-memory graph implementation using the popular `networkx` library. It's simple and fast for smaller graphs. It serializes the graph to a `.graphml` file. Clustering is performed using the `graspologic` library.
*   **`gdb_neo4j.py`**: A robust, persistent graph implementation that connects to a Neo4j database.
    *   **Pathfinding**: This is a standout feature. It provides multiple, highly optimized pathfinding methods (`find_shortest_path`, `find_all_shortest_paths`, `find_k_shortest_paths`) that leverage native Cypher queries and the Graph Data Science (GDS) library. It can use modern `SHORTEST` path syntax, APOC procedures for Dijkstra, or the full GDS library for Dijkstra and Yen's K-shortest paths. This makes it extremely powerful for reasoning tasks.
    *   **Clustering**: It uses `gds.leiden.write` to perform clustering directly within the database.
*   **`vdb_nanovectordb.py`**: A lightweight, file-based vector database. It's a simple dependency-free option for getting started.
*   **`kv_json.py`**: A simple key-value store that uses a single JSON file as its backend.

### `hirag/prompt.py`: The Brain of the System

This file is arguably the most critical for the quality of the system's output. It contains all the prompts used to instruct the LLMs.

*   **`hi_entity_extraction`**: A highly detailed and rule-intensive prompt for extracting entities. It strictly defines the entity types (`postulate`, `object`, `concept`, `property`, `proof`) and the crucial `is_temporary` flag, providing examples for each. It includes strict rules about what *not* to extract (e.g., editorial comments, content from within proofs).
*   **`hi_relation_extraction`**: The corresponding prompt for extracting relationships between the entities found by the previous prompt.
*   **`summary_clusters`**: The prompt used during hierarchical clustering. It instructs the LLM to act as a "knowledge architect," synthesizing new, higher-level attribute entities that abstract a cluster of concepts.
*   **`summarize_entity_descriptions`**: The prompt used when merging nodes. It instructs the LLM to synthesize a single, coherent description from a list of descriptions for the same entity.
*   **`community_report`**: A prompt that instructs the LLM to act as a research analyst. It asks for a structured JSON output containing a title, summary, importance rating, and detailed findings for a given community.
*   **`entity_disambiguation`**: The prompt for the EDM verification step. It provides the LLM with the candidate cluster and the full text context, asking for a structured JSON decision (`MERGE` or `DO_NOT_MERGE`) with justification and a confidence score.

### Utility Files

*   **`_llm.py`**: Provides standardized, asynchronous functions (`openai_complete_if_cache`, `gemini_complete_if_cache`, etc.) for interacting with different LLM APIs. It handles caching, retries on failure (`tenacity`), and API-specific formatting.
*   **`_utils.py`**: Contains essential helper functions, including `compute_mdhash_id` for creating deterministic IDs, `clean_str` for sanitizing text, JSON parsing helpers, and the `limit_async_func_call` decorator for rate limiting.
*   **`_validation.py`**: Centralizes validation logic (e.g., `_validate_entity_record_attributes`) to ensure that data parsed from LLM outputs conforms to the expected structure before being processed further.
*   **`_splitter.py`**: Contains the `SeparatorSplitter`, a sophisticated text splitter that can handle token-based splitting with overlap, respecting a list of separator sequences.

## 5. Agentic Workflow in Practice

With the new toolkit, an agent's interaction with HiRAG becomes a dynamic, multi-step reasoning process.

**Scenario**: A user asks, "How does the concept of a weak Cartesian closed category (wCCC) relate to typed λ-calculus?"

**Agent's Thought Process & Actions**:

1.  **Initial Exploration**: "I need to understand the high-level concepts in the knowledge base. I'll start with the Table of Contents."
    *   **Action**: `hirag.aget_community_toc(level=0)`
    *   **Result**: A list of top-level communities. The agent sees a title like "Weak Cartesian Closed Categories (wCCCs) as a Model for Typed Combinatory Logic".

2.  **Drill-Down**: "That community seems highly relevant. I'll get the detailed report for it."
    *   **Action**: `hirag.aget_community_details(community_id="...")`
    *   **Result**: A detailed markdown report, including the summary, key findings, and lists of important entities like "WCCC", "TYPED COMBINATORY LOGIC", "(β_nat) EQUATION", and "RETRACTION".

3.  **Entity Investigation**: "The report mentions 'RETRACTION' is a key consequence. I need to understand exactly what that is and how it's connected."
    *   **Action 1**: `hirag.aget_entity_details("RETRACTION")` -> Gets the full, synthesized description of the 'retraction' concept.
    *   **Action 2**: `hirag.aget_relationships("WEAK CARTESIAN CLOSED CATEGORY (WCCC)")` -> Gets all relationships for WCCC, confirming the one to "RETRACTION" and its description.
    *   **Action 3**: `hirag.aget_source_text("RETRACTION")` -> Pulls the original text where 'retraction' was defined, providing ground-truth context.

4.  **Reasoning Path**: "The user asked about the relationship to 'typed λ-calculus'. I see 'TYPED COMBINATORY LOGIC' in the community. Let me find the reasoning path between 'WCCC' and 'TYPED COMBINATORY LOGIC' to build a logical argument."
    *   **Action**: `hirag.afind_reasoning_path("WEAK CARTESIAN CLOSED CATEGORY (WCCC)", "TYPED COMBINATORY LOGIC")`
    *   **Result**: A path showing the nodes and edges connecting the two concepts, likely highlighting that WCCCs provide the categorical semantics for the logic.

5.  **Answer Synthesis**: The agent now has a wealth of structured context: the high-level community summary, detailed definitions of key entities, the specific relationships connecting them, the original source text for verification, and a logical path. It can now synthesize a comprehensive, accurate, and well-supported answer for the user, referencing all the evidence it has gathered.

## 6. Conclusion

The HiRAG codebase represents a significant evolution in Retrieval-Augmented Generation. By prioritizing the creation of a structured, hierarchical, and clean knowledge graph, it moves beyond simple semantic search. The true power of the system is unlocked through its agentic toolkit, which transforms the RAG process from a monolithic query-answer flow into a dynamic, exploratory reasoning dialogue. As developers and agents, you are now equipped not just with a tool to find information, but with a structured universe of knowledge to explore and reason over.