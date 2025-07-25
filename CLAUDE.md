# AGENTS.md: A Comprehensive Guide to the HiRAG Codebase

## 1. Mission and Core Philosophy

Welcome to HiRAG. This document is the definitive guide for any agent or developer interacting with this codebase. By understanding this guide, you will gain an expert-level comprehension of the system's architecture, capabilities, and operational workflows.

**HiRAG (Hierarchical Retrieval-Augmented Generation)** is an advanced AI framework designed to build, explore, and reason over complex information domains. Its core mission is to transcend the limitations of traditional, "flat" RAG systems by constructing a multi-layered, hierarchical knowledge graph from unstructured text. This allows for nuanced, context-aware reasoning that moves from high-level summaries down to the finest-grained source data.

The system is built on three foundational pillars:

1.  **Hierarchical Abstraction:** Information is not treated as a simple collection of text chunks. Instead, HiRAG builds a pyramid of knowledge: raw text is processed into atomic **entities** and **relationships**, which are then clustered into conceptual **communities**. These communities are further analyzed to generate high-level, LLM-powered **reports**, creating a browsable, multi-layered understanding of the data.
2.  **Graph-Powered Reasoning:** At its heart, HiRAG uses a knowledge graph to represent the intricate connections within the information. This graph is not merely a database; it is a dynamic reasoning engine. It supports sophisticated operations like pathfinding between concepts, which allows an agent to uncover implicit lines of reasoning and explain how disparate ideas are connected. The system is architected to support multiple graph backends, from the in-memory `NetworkX` for rapid prototyping to the scalable, persistent `Neo4j` for production deployments.
3.  **Agentic Toolkit Architecture:** The latest evolution of HiRAG moves beyond a monolithic query engine. It now exposes a rich **toolkit of granular, asynchronous methods** (`aget_*`, `afind_*`). This empowers an AI agent to become an active participant in the reasoning process. Instead of receiving a pre-packaged context, an agent can now intelligently explore the knowledge graph, ask targeted questions, follow lines of inquiry, and synthesize information from multiple sources to construct a truly comprehensive and defensible answer.

---

## 2. Directory Structure

The codebase is organized logically to separate concerns, from the main application logic to swappable storage backends and core operational utilities.

```
└── ./
    └── hirag
        ├── _storage
        │   ├── __init__.py           # Makes the storage directory a package
        │   ├── gdb_neo4j.py          # Neo4j graph database storage implementation
        │   ├── gdb_networkx.py       # NetworkX graph database storage implementation
        │   ├── kv_json.py            # JSON-based key-value storage implementation
        │   └── vdb_nanovectordb.py   # NanoVectorDB vector storage implementation
        ├── __init__.py               # Initializes the hirag package
        ├── _cluster_utils.py         # Utilities for hierarchical entity clustering
        ├── _llm.py                   # Abstraction layer for LLM API calls
        ├── _op.py                    # Core operational logic (chunking, extraction, reporting)
        ├── _splitter.py              # Text splitting and chunking logic
        ├── _utils.py                 # General utility functions
        ├── _validation.py            # Centralized validation for data structures
        ├── base.py                   # Base classes and type definitions for storage
        ├── hirag.py                  # Main HiRAG class and agent toolkit entry point
        └── prompt.py                 # The "brain" of the system: all LLM prompts
```

---

## 3. File-by-File Deep Dive

This section provides an exhaustive analysis of every file in the codebase, detailing its purpose, classes, and functions.

### 3.1. `hirag/hirag.py` - The Core Engine

This is the main entry point and the heart of the HiRAG framework. It defines the `HiRAG` class, which orchestrates the entire indexing and querying lifecycle, and exposes the powerful new agentic toolkit.

#### **`HiRAG` Dataclass**

This class holds the configuration and state for a HiRAG instance.

**Key Configuration Parameters:**

*   `working_dir`: The directory to store all cached data, graphs, and databases.
*   `enable_hierachical_mode`: If `True`, uses the advanced hierarchical entity extraction and clustering. If `False`, falls back to a flatter entity extraction model.
*   `chunk_func`, `chunk_token_size`, `chunk_overlap_token_size`: Parameters controlling how source documents are split into manageable text chunks.
*   `graph_cluster_algorithm`, `max_graph_cluster_size`: Defines the algorithm (e.g., `leiden`) used to find communities in the knowledge graph.
*   `embedding_func`, `best_model_func`, `cheap_model_func`: Allows for complete customization of the AI models used for embedding, high-quality reasoning, and cheaper, high-throughput tasks, respectively. The system pre-configures these for OpenAI, Azure OpenAI, and Google Gemini.
*   `key_string_value_json_storage_cls`, `vector_db_storage_cls`, `graph_storage_cls`: These parameters make the storage backend completely swappable. You can plug in any class that adheres to the `Base*Storage` interfaces defined in `base.py`.
*   `addon_params`: A flexible dictionary for passing backend-specific parameters, such as Neo4j connection credentials.

**`__post_init__(self)`:**

This method is the constructor's workhorse. It initializes all components based on the configuration:
1.  **Model Selection:** Automatically switches to Azure or Gemini model functions if `using_azure_openai` or `using_gemini` is set to `True`.
2.  **Storage Initialization:** Instantiates all storage backends (for documents, text chunks, community reports, knowledge graph, and vector databases).
3.  **LLM Caching & Rate Limiting:** Wraps the selected LLM and embedding functions with rate limiters (`limit_async_func_call`) and an optional caching layer that uses the configured KV store. This dramatically improves performance and reduces cost on repeated queries.

#### **Indexing Pipeline: `ainsert(self, string_or_strings)`**

This is the sole method for adding new information to the HiRAG system. It orchestrates a sophisticated, asynchronous pipeline:

1.  **Input & Deduplication:** Takes a string or list of strings, computes a hash for each, and filters out any documents that already exist in the `full_docs` storage.
2.  **Chunking:** The new documents are passed to the `get_chunks` function, which uses the configured `chunk_func` (e.g., `chunking_by_token_size`) to split them into text chunks. These are also deduplicated against the `text_chunks` store.
3.  **Hierarchical Entity Extraction:** This is the core of knowledge creation. The `hierarchical_entity_extraction_func` is called.
    *   It first performs an initial pass to extract entities and relationships from the text chunks using the `hi_entity_extraction` and `hi_relation_extraction` prompts.
    *   Crucially, it then invokes the `Hierarchical_Clustering` algorithm from `_cluster_utils.py`. This algorithm iteratively clusters the extracted entities, uses the `summary_clusters` prompt to synthesize new, higher-level "attribute" entities, and builds out the hierarchy.
4.  **Graph Construction:** All extracted entities (from all hierarchical levels) and relationships are merged and upserted into the configured graph storage backend (`chunk_entity_relation_graph`). This process intelligently merges descriptions and recalculates weights for existing nodes and edges.
5.  **Vector Indexing:** The descriptions of all non-temporary entities are embedded using the configured `embedding_func` and stored in the `entities_vdb` for semantic search.
6.  **Community Detection:** The `clustering` method is called on the graph storage instance, which runs an algorithm like Leiden to partition the graph into communities.
7.  **Community Reporting:** The `generate_community_report` function is called. It iterates through the detected communities, packs their constituent nodes and edges into a context, and uses the `community_report` prompt to generate a rich, structured JSON summary for each one. These reports are stored in the `community_reports` KV store.
8.  **Finalization:** All storage backends are instructed to commit their changes to disk (`_insert_done`).

#### **The Agentic Toolkit: A New Paradigm**

The old, monolithic `aquery` method is now deprecated. It is replaced by a suite of granular tools that an agent can use to perform sophisticated, multi-step reasoning.

*   **`async def aget_community_toc(self, level: int = 0) -> str`**
    *   **Purpose:** Provides a high-level "Table of Contents" of the knowledge base.
    *   **How it Works:** Fetches all community reports for a given hierarchy level, extracts their titles and summaries, and formats them into a markdown list.
    *   **Use Case:** An agent's first step to understand the major themes in the data.

*   **`async def afind_entities(self, query: str, top_k: int = 5, temporary: Optional[bool] = None) -> List[Dict]`**
    *   **Purpose:** The primary tool for finding relevant entry points into the graph.
    *   **How it Works:** Performs a semantic vector search on the `entities_vdb`. It includes a powerful `temporary` flag to filter for either foundational concepts (`False`) or locally-defined, temporary ones (`True`).
    *   **Use Case:** Finding the most relevant entities related to a user's query.

*   **`async def aget_entity_details(self, entity_name: str) -> Optional[Dict]`**
    *   **Purpose:** Retrieves the full, consolidated data for a single entity.
    *   **How it Works:** A direct lookup in the graph store for a given entity name.
    *   **Use Case:** Getting the complete description, type, and source information for an entity of interest.

*   **`async def aget_relationships(self, entity_name: str) -> List[Dict]`**
    *   **Purpose:** Explores the immediate connections of an entity.
    *   **How it Works:** Fetches all edges connected to the given entity from the graph store and hydrates them with their full data (description, weight, etc.).
    *   **Use Case:** Understanding how a concept is related to its neighbors.

*   **`async def aget_source_text(self, entity_name: str) -> List[Dict]`**
    *   **Purpose:** Grounds an entity in its original, raw context.
    *   **How it Works:** Retrieves the `source_id` field from an entity's data, which contains the IDs of the original text chunks, and fetches those chunks from the `text_chunks` KV store.
    *   **Use Case:** When an agent needs to read the exact text where a concept was defined or discussed.

*   **`async def afind_reasoning_path(self, start_entity: str, end_entity: str, ...)`**
    *   **Purpose:** One of the most powerful tools. It finds the shortest path between two concepts, effectively uncovering a "reasoning chain".
    *   **How it Works:** This method is a sophisticated dispatcher. It detects the type of graph backend (`NetworkX` or `Neo4j`) and calls the appropriate private implementation (`_find_path_networkx` or `_find_path_neo4j`). It can handle weighted (Dijkstra) and unweighted (BFS) searches and automatically selects the best algorithm if `algorithm` is set to `'auto'`.
    *   **Use Case:** Answering "How is X related to Y?" questions by showing the intermediate steps.

*   **`async def afind_multiple_reasoning_paths(self, start_entity: str, end_entity: str, k: int = 3, ...)`**
    *   **Purpose:** Finds several alternative paths between two entities.
    *   **How it Works:** Leverages the `all_shortest_paths` (NetworkX) or `find_k_shortest_paths` (Neo4j) capabilities of the backend.
    *   **Use Case:** Exploring different ways two concepts might be related, offering more comprehensive explanations.

*   **`async def aget_holistic_context(self, query: str, param: QueryParam = QueryParam()) -> Optional[str]`**
    *   **Purpose:** A high-level tool that can be used by an agent when it wants a comprehensive, pre-built context without manual exploration.
    *   **How it Works:** This method encapsulates the entire hierarchical retrieval process. It finds relevant entities, their related communities, their source texts, and a reasoning path between the top entities, then formats all of this information into a single, rich context string.
    *   **Use Case:** A powerful "one-shot" search tool for an agent to quickly gather broad context.

*   **`async def aget_community_details(self, community_id: str, ...)`**
    *   **Purpose:** Allows an agent to "zoom in" on a specific community from the Table of Contents.
    *   **How it Works:** Fetches the full community report and a detailed list of its most important constituent entities and relationships, formatting them into a detailed markdown string.
    *   **Use Case:** Deep-diving into a specific topic area after an initial high-level exploration.

### 3.2. `hirag/prompt.py` - The System's Brain

This file is arguably the most critical component for the quality of the generated knowledge graph. It contains the meticulously engineered prompts that instruct the LLMs on how to perform their tasks.

*   **`PROMPTS["hi_entity_extraction"]`**: This is a highly detailed and strict prompt for extracting entities.
    *   **Role:** It casts the LLM as a "precise and rigorous expert in Combinatory Logic."
    *   **Key Instructions:**
        *   Defines a strict set of `Entity Types` (`postulate`, `object`, `concept`, `property`, `proof`).
        *   Introduces the **crucial** concept of `is_temporary`. This flag is vital for distinguishing foundational concepts from local variables, which prevents the graph from being cluttered with noise.
        *   Imposes **Strict Extraction Rules**, commanding the LLM to extract *only* factual information, handle proofs as atomic units, and avoid extracting entities from within a proof's internal logic.
    *   **Impact:** This prompt is responsible for the high fidelity and low noise of the base layer of the knowledge graph.

*   **`PROMPTS["hi_relation_extraction"]`**: This prompt works in tandem with entity extraction to build the graph's connections.
    *   **Key Instructions:**
        *   Focuses on *meaningful, factual relationships*.
        *   Prioritizes connections between non-temporary entities.
        *   Asks the LLM to infer the "why" behind a relationship and include it in the description.
        *   Requires a `relationship_strength` score (1-10), which provides valuable weight data for graph algorithms.

*   **`PROMPTS["summary_clusters"]`**: This prompt drives the hierarchical abstraction process.
    *   **Role:** Casts the LLM as a "knowledge architect."
    *   **Goal:** Instead of just grouping entities, it instructs the LLM to *synthesize new, higher-level attribute entities* that abstract or encompass the concepts in a given cluster.
    *   **Key Instructions:**
        *   The new entity must be a genuine abstraction, not a synonym.
        *   The new entity must **always be non-temporary**.
        *   It must create relationships linking the original entities to the new summary entity.
    *   **Impact:** This is the engine of hierarchical knowledge building, creating the upper layers of the conceptual pyramid.

*   **`PROMPTS["community_report"]`**: This prompt generates the human- and agent-readable summaries for each community.
    *   **Role:** Casts the LLM as a "research analyst."
    *   **Output Format:** Demands a **strict JSON object** with a `title`, `summary`, `importance_rating`, `rating_explanation`, and an array of `detailed_findings`.
    *   **Key Instructions:**
        *   All claims must be evidence-based from the provided community data.
        *   The report must focus on the non-temporary entities to explain the community's core theme.
    *   **Impact:** This creates the rich, structured, and explorable summaries that the agent toolkit relies on.

### 3.3. The `_storage` Directory - Swappable Backends

This directory contains the concrete implementations of the storage interfaces defined in `base.py`. This design makes HiRAG highly flexible and adaptable to different deployment needs.

*   **`gdb_networkx.py` -> `NetworkXStorage`**:
    *   **Purpose:** An in-memory graph database using the popular `networkx` library.
    *   **Best For:** Rapid prototyping, smaller datasets, and environments where no external database is available.
    *   **Key Feature:** It loads the entire graph into memory from a `.graphml` file. All operations are fast but are limited by available RAM.

*   **`gdb_neo4j.py` -> `Neo4jStorage`**:
    *   **Purpose:** A persistent, scalable graph database using Neo4j.
    *   **Best For:** Production deployments, large datasets, and complex graph queries.
    *   **Key Features:**
        *   Connects to an external Neo4j server using official drivers.
        *   **Advanced Pathfinding:** This class contains a suite of powerful, new pathfinding methods that leverage Neo4j's capabilities:
            *   `find_shortest_path`: A comprehensive method that can use modern Cypher, GDS (Graph Data Science library), or APOC procedures.
            *   `find_all_shortest_paths`: Finds all paths of the shortest possible length.
            *   `find_k_shortest_paths`: Uses Yen's algorithm via GDS to find the *k* shortest paths, which may have different lengths. This is excellent for exploring alternative reasoning routes.
        *   Uses Cypher queries to implement all the base graph operations, ensuring scalability.

*   **`kv_json.py` -> `JsonKVStorage`**:
    *   **Purpose:** A simple, file-based key-value store.
    *   **How it Works:** Stores data as a single large JSON file. It's simple and requires no external dependencies.
    *   **Use Case:** Default storage for LLM caches, document maps, and community reports.

*   **`vdb_nanovectordb.py` -> `NanoVectorDBStorage`**:
    *   **Purpose:** A simple, file-based vector database for semantic search.
    *   **How it Works:** Provides basic vector upsert and cosine similarity search functionalities, persisting the index to a JSON file.
    *   **Use Case:** Default vector store for entity and chunk embeddings.

### 3.4. Core Logic and Utilities

*   **`_op.py` (Operations):**
    *   This file contains the procedural logic that drives the `ainsert` pipeline.
    *   **`extract_entities` & `extract_hierarchical_entities`:** These are the master functions for knowledge extraction. They manage the asynchronous calls to the LLM for each text chunk, parse the structured output, and orchestrate the merging and upserting of the resulting nodes and edges into the graph. The hierarchical version integrates the `Hierarchical_Clustering` step.
    *   **`generate_community_report`:** This function orchestrates the creation of community summaries. It retrieves the community structure from the graph, iterates through each community level-by-level (starting from the most granular), packs the context, and calls the LLM with the `community_report` prompt.
    *   **`get_chunks`:** A utility that takes raw documents and uses a specified chunking strategy to produce the initial `TextChunkSchema` objects.

*   **`_cluster_utils.py`:**
    *   This file is dedicated to the new hierarchical clustering functionality.
    *   **`Hierarchical_Clustering` class:** This is the main component. Its `perform_clustering` method implements the iterative clustering-and-summarization loop.
        *   It uses UMAP and Gaussian Mixture Models (`GMM_cluster`) to find conceptual clusters in the entity embedding space.
        *   It checks for convergence using `cluster_sparsity` to avoid excessive layering.
        *   For each cluster, it calls the LLM with the `summary_clusters` prompt to generate new, higher-level entities.
        *   It then prepares the newly synthesized entities for the next layer of clustering, effectively building the knowledge hierarchy.

*   **`_llm.py`:**
    *   This is the abstraction layer for all LLM interactions.
    *   It provides singleton `get_*_async_client_instance` functions to manage API clients efficiently.
    *   The core functions are `*_complete_if_cache` and `*_embedding`. They are wrapped with a tenacious `@retry` decorator to handle transient network errors and API rate limits gracefully. They also seamlessly integrate with the optional `hashing_kv` caching layer.

*   **`_validation.py`:**
    *   A crucial new utility for robustness. It provides a centralized `validate` function to safely parse the structured but potentially malformed output from LLMs.
    *   It defines specific validation rules (e.g., `_validate_entity_record_attributes`) that check the length and format of the extracted tuples, preventing errors from propagating through the system.

*   **`base.py`:**
    *   Defines the foundational abstract base classes (`BaseGraphStorage`, `BaseKVStorage`, `BaseVectorStorage`) that all storage implementations must inherit from. This ensures a consistent interface and makes the backends swappable.
    *   It also defines the core data schemas like `QueryParam`, `TextChunkSchema`, and `CommunitySchema`, providing type-hinting and structure throughout the codebase.

---

## 4. Workflows in Detail

### The Indexing Workflow (`ainsert`)

This is the process of teaching HiRAG new information. It is a linear pipeline that transforms raw text into a rich, multi-layered knowledge graph.

1.  **Input:** A developer provides a list of strings (documents) to `hirag.ainsert()`.
2.  **Chunking:** The documents are broken down into smaller, overlapping `TextChunkSchema` objects.
3.  **Layer 0 Extraction:** The `hierarchical_entity_extraction_func` begins. It sends each chunk to an LLM using the `hi_entity_extraction` and `hi_relation_extraction` prompts. The LLM returns lists of entities and relationships.
4.  **Hierarchical Clustering & Synthesis:** The `Hierarchical_Clustering` process kicks in.
    a. The extracted entities are clustered based on their embedding similarity.
    b. For each cluster, the LLM is called with the `summary_clusters` prompt.
    c. The LLM synthesizes *new, higher-level* entities that abstract the cluster's theme.
    d. These new entities become the input for the next layer of clustering. This loop continues until the clusters become too sparse or no new meaningful abstractions can be formed.
5.  **Graph Population:** All entities and relationships from all hierarchical layers are upserted into the graph storage backend (e.g., `Neo4jStorage`). The system intelligently merges data for entities that appear multiple times.
6.  **Vector Indexing:** All non-temporary entities are embedded and stored in the `entities_vdb` for fast semantic retrieval.
7.  **Community Detection:** The `leiden` algorithm is run on the final graph, partitioning it into communities of closely related concepts.
8.  **Community Reporting:** The system iterates through the communities, calls the LLM with the `community_report` prompt, and stores the resulting structured JSON reports in the `community_reports` KV store.
9.  **Ready:** The system has now fully indexed the information and is ready for reasoning.

### The Agentic Reasoning Workflow (An Example)

This demonstrates how an AI agent uses the new toolkit to answer a complex query, showcasing the power of the new architecture.

**Query:** "How does the theory of weak Cartesian closed categories (wCCCs) provide a model for typed Combinatory Logic, and what is the specific role of the (β_nat) equation in this?"

1.  **Agent's First Step: Get an Overview.**
    *   The agent calls `hirag.aget_community_toc(level=0)`.
    *   It receives a list of top-level topics. It sees one titled: `Weak Cartesian Closed Categories (wCCCs) as a Model for Typed Combinatory Logic`. This is a perfect match.

2.  **Agent's Second Step: Zoom In.**
    *   The agent calls `hirag.aget_community_details(community_id=...)` on the community it just found.
    *   It receives a detailed markdown report containing the full summary, key entities (`wCCC`, `(β_nat) equation`, `typed Combinatory Logic`, `retraction`), and key relationships.

3.  **Agent's Third Step: Find the Reasoning Path.**
    *   To understand the direct connection, the agent asks for the path: `hirag.afind_reasoning_path(start_entity="WEAK CARTESIAN CLOSED CATEGORY (WCCC)", end_entity="TYPED COMBINATORY LOGIC")`.
    *   The result shows a direct edge: `(wCCC) -[provides semantics for]-> (typed CL)`.

4.  **Agent's Fourth Step: Investigate a Specific Component.**
    *   The agent now focuses on the second part of the query. It calls `hirag.aget_relationships(entity_name="(β_nat) EQUATION")`.
    *   It discovers a critical relationship: `(wCCC) -[is defined by]-> ((β_nat) equation)`.

5.  **Agent's Fifth Step: Go to the Source.**
    *   For maximum precision, the agent wants the original definition. It calls `hirag.aget_source_text(entity_name="(β_nat) EQUATION")`.
    *   It receives the exact text chunk where the equation was formally defined.

6.  **Synthesize the Answer:**
    *   The agent now has all the pieces: the high-level summary, the key entities, the direct path, the role of the specific equation, and the original source text. It can now construct a detailed, multi-faceted, and fully-grounded answer for the user, explaining each step of its reasoning process.


### Workflow Visualization

```mermaid
graph TD
    %% =================================================================
    %% STYLING & LEGEND DEFINITIONS
    %% =================================================================
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef input fill:#cce5ff,stroke:#007bff,stroke-width:2px,font-weight:bold;
    classDef storage fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef llmCall fill:#f8d7da,stroke:#dc3545,stroke-width:3px,font-weight:bold;
    classDef process fill:#e0f2f7,stroke:#3498db,stroke-width:2px;
    classDef data fill:#fff3cd,stroke:#ffc107,stroke-width:2px,rx:5,ry:5;
    classDef decision diamond,fill:#f0f0f0,stroke:#999,stroke-width:2px,font-weight:bold;
    classDef loop style,fill:#f2f2f2,stroke:#666,stroke-width:1px,font-style:italic;
    classDef stage_title fill:#343a40,color:#ffffff,stroke:#343a40,stroke-width:2px;
    classDef utility fill:#e6e6fa,stroke:#8a2be2,stroke-width:1px,font-size:smaller;
    classDef callback fill:#f0e68c,stroke:#daa520,stroke-width:1px;
    classDef agent fill:#e8dff5,stroke:#9b59b6,stroke-width:2px,font-weight:bold;
    classDef annotation fill:#fdfde0,stroke:#bdbcbc,stroke-width:1px,font-style:italic,font-size:smaller;

    %% =================================================================
    %% STAGE 0: INITIALIZATION
    %% =================================================================
    subgraph STAGE_0 ["HiRAG Initialization (hirag.py)"]
        direction TB
        Init["HiRAG(...)"]:::input --> ConfigCheck{"using_azure_openai? / using_gemini?"}:::decision
        ConfigCheck -- "Yes" --> SwitchLLMs("Switch default LLM/Embedding funcs (e.g., gpt_4o_complete -> azure_gpt_4o_complete)"):::process
        ConfigCheck -- "No" --> InitStorages
        SwitchLLMs --> InitStorages("Initialize Storage Classes (JsonKVStorage, NetworkXStorage, etc.)"):::process
        InitStorages --> WrapFuncs("Wrap LLM/Embedding funcs with rate limiters & cache handlers ('limit_async_func_call', 'partial')"):::process
        WrapFuncs --> Ready["HiRAG Instance Ready"]:::input
    end

    %% =================================================================
    %% STAGE 1: DOCUMENT INGESTION & CHUNKING
    %% =================================================================
    subgraph STAGE_1 ["(ainsert) 1: Input Processing & Chunking"]
        direction TB
        Ready --> Start(["Start Ingestion ('ainsert(string_or_strings)')"]):::input
        Start --> P1_CallbackStart("Call '_insert_start()' callbacks"):::callback
        P1_CallbackStart --> P1_InputData{"Input: 'string_or_strings' (Union[str, List[str]])"}:::data
        P1_InputData -- "List[str]" --> P1_ComputeDocID("Compute MD5 Hash for each Doc ('_utils.compute_mdhash_id')"):::process
        P1_ComputeDocID -- "Dict[doc_id, {'content': str}]" --> P1_FilterDocs("Filter existing Docs ('full_docs.filter_keys')"):::storage
        P1_FilterDocs -- "New Docs (Dict[doc_id, {'content': str}])" --> P1_StoreDocs("Upsert New Docs to KV Store ('full_docs.upsert')"):::storage
        P1_FilterDocs -- "All Docs Exist" --> End1("Stop: No new content")
        
        P1_StoreDocs --> P1_GetChunks("Get Chunks ('_op.get_chunks')"):::process
        P1_GetChunks -- "Uses: tiktoken, chunk_func (e.g., _op.chunking_by_token_size)" --> P1_ChunkFuncDetails("_splitter.SeparatorSplitter: Encode, Split by separators, Merge, Handle Overlap"):::utility
        P1_ChunkFuncDetails -- "List[Dict]" --> P1_ComputeChunkID("Compute MD5 Hash for each Chunk ('_utils.compute_mdhash_id')"):::process
        P1_ComputeChunkID -- "Dict[chunk_id, TextChunkSchema]" --> P1_FilterChunks("Filter existing Chunks ('text_chunks.filter_keys')"):::storage
        P1_FilterChunks -- "New Chunks (Dict[chunk_id, TextChunkSchema])" --> P1_StoreChunksKV("Upsert New Chunks to Text KV Store ('text_chunks.upsert')"):::storage
        P1_FilterChunks -- "All Chunks Exist" --> End2("Stop: No new content")
        
        P1_StoreChunksKV --> P1_NaiveRagCheck{"enable_naive_rag?"}:::decision
        P1_NaiveRagCheck -- "Yes" --> P1_StoreChunksVDB("Upsert Chunks to Vector DB ('chunks_vdb.upsert')"):::storage
        P1_StoreChunksVDB --> P1_ClearReports("Clear old reports for re-generation ('community_reports.drop')"):::process
        P1_ClearReports --> P1_ToStage2
        P1_NaiveRagCheck -- "No" --> P1_ToStage2("Proceed to Extraction")
    end

    %% =================================================================
    %% STAGE 2: ENTITY & RELATION EXTRACTION
    %% =================================================================
    subgraph STAGE_2 ["(ainsert) 2: Entity & Relation Extraction"]
        direction TB
        P1_ToStage2 -- "Data: New Chunks" --> P2_ModeCheck{"enable_hierarchical_mode?"}:::decision
        
        %% --- HIERARCHICAL PATH ---
        P2_ModeCheck -- "Yes" --> S2_H_Title
        subgraph S2_H_Title ["Hierarchical Path ('_op.extract_hierarchical_entities')"]
            direction TB
            S2_H_LoopChunks1("Parallel Processing of New Chunks ('asyncio.gather')"):::loop --> S2_H_ProcessSingleEntity("Call '_op._process_single_content_entity' for each chunk"):::process
            S2_H_ProcessSingleEntity -- "Chunk Content (str)" --> S2_H_FormatEntityPrompt("Format 'hi_entity_extraction' prompt ('prompt.py')"):::process
            S2_H_FormatEntityPrompt --> S2_H_LLM_ExtractEntities("LLM Call: Extract Entities ('best_model_func')"):::llmCall
            
            subgraph S2_H_Gleaning1 ["Iterative Gleaning Loop ('entity_extract_max_gleaning')"]
                direction TB
                S2_H_LLM_ExtractEntities -- "Initial LLM Response (str)" --> S2_H_GleanLoopStart("Start Gleaning"):::loop
                S2_H_GleanLoopStart -- "History (OpenAI messages), 'entiti_continue_extraction' prompt" --> S2_H_LLM_Glean("LLM Call: Refine/Continue ('best_model_func')"):::llmCall
                S2_H_LLM_Glean -- "Glean Result (str)" --> S2_H_AppendResult("Append to final_result, update history ('_utils.pack_user_ass_to_openai_messages')"):::process
                S2_H_AppendResult --> S2_H_CheckLoop{"More to glean? (LLM call with 'entiti_if_loop_extraction' prompt)"}:::decision
                S2_H_CheckLoop -- "Yes" --> S2_H_GleanLoopStart
            end
            S2_H_CheckLoop -- "No" --> S2_H_ParseEntities("Parse Records ('_utils.split_string_by_multi_markers', '_op._handle_single_entity_extraction')"):::process
            S2_H_ParseEntities -- "Uses: '_validation.validate', '_utils.clean_str'" --> S2_H_CollectInitialNodes("Collect chunk-local entities (Dict[entity_name, List[Dict]])"):::data
            S2_H_CollectInitialNodes --> S2_H_AggregatedEntities("All Entities from all Chunks Aggregated (Dict[entity_name, Dict])"):::data
            
            S2_H_AggregatedEntities --> S2_H_GetEmbeddings("Get Embeddings for all unique entities ('entities_vdb.embedding_func')"):::process
            S2_H_GetEmbeddings -- "Entities with Embeddings (List[Dict])" --> S2_H_HierarchicalLoop("Start Hierarchical Clustering Loop ('_cluster_utils.Hierarchical_Clustering.perform_clustering')"):::loop
            
            subgraph S2_H_Clustering ["Hierarchical Abstraction (_cluster_utils.py)"]
                direction TB
                S2_H_HierarchicalLoop --> S2_H_Cluster("Perform UMAP Reduction & GMM Clustering ('_cluster_utils.perform_clustering')"):::process
                S2_H_Cluster -- "Semantic Clusters (List[List[int]])" --> S2_H_LoopClusters("For each new cluster..."):::loop
                S2_H_LoopClusters --> S2_H_CalcLength("Calculate total token length ('tiktoken')"):::utility
                S2_H_CalcLength -- "Length > max_length_in_cluster?" --> S2_H_ReduceNodes{"Reduce cluster size? ('random.sample')"}:::decision
                S2_H_ReduceNodes -- "Yes" --> S2_H_CalcLength
                S2_H_ReduceNodes -- "No / Done" --> S2_H_FormatSummaryPrompt("Format 'summary_clusters' prompt ('prompt.py')"):::process
                S2_H_FormatSummaryPrompt --> S2_H_LLM_SummarizeCluster("LLM Call: Summarize cluster to create new 'attribute' entities/relations ('best_model_func')"):::llmCall
                S2_H_LLM_SummarizeCluster -- "New abstract entities/relations (str)" --> S2_H_ParseNew("Parse & Add to 'maybe_nodes' & 'maybe_edges' ('_op._handle_single_entity_extraction', '_op._handle_single_relationship_extraction')"):::data
                S2_H_ParseNew -- "Data: New Abstract Entities" --> S2_H_CheckHierLoop{"More layers needed? (based on sparsity/entity num)"}:::decision
                S2_H_CheckHierLoop -- "Yes, feed new entities back" --> S2_H_HierarchicalLoop
            end
            
            S2_H_CheckHierLoop -- "No, Loop Done" --> S2_H_LoopChunks2("Parallel Processing of New Chunks ('asyncio.gather')"):::loop
            S2_H_LoopChunks2 --> S2_H_ProcessSingleRelation("Call '_op._process_single_content_relation' for each chunk"):::process
            S2_H_ProcessSingleRelation -- "Chunk Content (str), Context Entities (List[str])" --> S2_H_FormatRelationPrompt("Format 'hi_relation_extraction' prompt ('prompt.py')"):::process
            S2_H_FormatRelationPrompt --> S2_H_LLM_ExtractRelations("LLM Call: Extract Relations ('best_model_func')"):::llmCall
            S2_H_LLM_ExtractRelations -- "Initial Response" --> S2_H_GleanLoopStart2("Start Gleaning Loop"):::loop
            S2_H_GleanLoopStart2 --> S2_H_LLM_Glean2("LLM Call: Refine/Continue"):::llmCall
            S2_H_LLM_Glean2 --> S2_H_AppendResult2("Append to final_result"):::process
            S2_H_AppendResult2 --> S2_H_CheckLoop2{"More to glean?"}:::decision
            S2_H_CheckLoop2 -- "Yes" --> S2_H_GleanLoopStart2
            S2_H_CheckLoop2 -- "No" --> S2_H_ParseRelations("Parse Records ('_utils.split_string_by_multi_markers', '_op._handle_single_relationship_extraction')"):::process
            S2_H_ParseRelations -- "Uses: '_validation.validate', '_utils.clean_str', '_utils.is_float_regex'" --> S2_H_CollectInitialEdges("Collect chunk-local relations (Dict[tuple, List[Dict]])"):::data
        end
        
        %% --- STANDARD PATH ---
        P2_ModeCheck -- "No" --> S2_S_Title
        subgraph S2_S_Title ["Standard Path ('_op.extract_entities')"]
            direction TB
            S2_S_LoopChunks("Parallel Processing of New Chunks ('asyncio.gather')"):::loop --> S2_S_ProcessSingle("Call '_op._process_single_content' for each chunk"):::process
            S2_S_ProcessSingle -- "Chunk Content (str)" --> S2_S_FormatPrompt("Format 'entity_extraction' prompt ('prompt.py')"):::process
            S2_S_FormatPrompt --> S2_S_LLM_Extract("LLM Call: Extract Entities & Relations ('best_model_func')"):::llmCall
            S2_S_LLM_Extract -- "Initial LLM Response (str)" --> S2_S_Gleaning("Gleaning Loop (same as hierarchical path)"):::loop
            S2_S_Gleaning -- "Final LLM Response (str)" --> S2_S_Parse("Parse Records ('_op._handle_single_entity/relationship_extraction')"):::process
            S2_S_Parse -- "Entities (Dict[name, List[Dict]]), Relations (Dict[tuple, List[Dict]])" --> S2_S_Collect("Collect into 'maybe_nodes' & 'maybe_edges' dicts"):::data
        end
    end

    %% =================================================================
    %% STAGE 3: GRAPH CONSOLIDATION & MERGING
    %% =================================================================
    subgraph STAGE_3 ["(ainsert) 3: Unified Graph Construction"]
        direction TB
        S2_H_CollectInitialEdges --> P3_AggregateAll
        S2_S_Collect --> P3_AggregateAll("Consolidate all 'maybe_nodes' & 'maybe_edges' from all extraction paths"):::data
        S2_H_ParseNew -- "Feedback Loop from Hierarchical Clustering" --> P3_AggregateAll

        P3_AggregateAll -- "Dict[entity_name, List[Dict]]" --> P3_ProcessNodes("Process Unique Entities in Parallel ('asyncio.gather')"):::process
        P3_ProcessNodes -- "For each unique Entity Name..." --> P3_MergeNode("Call '_op._merge_nodes_then_upsert'"):::process
        P3_MergeNode -- "Uses: 'graph.get_node', 'utils.split_string_by_multi_markers'" --> P3_GatherDesc("Combine descriptions, source_ids, types, is_temporary status"):::data
        P3_GatherDesc -- "Combined Description (str)" --> P3_LLM_SummarizeNode("LLM Call: Summarize description ('_op._handle_entity_relation_summary' using 'cheap_model_func')"):::llmCall
        P3_LLM_SummarizeNode -- "Clean, summarized description (str)" --> P3_UpsertNode("Upsert unified Node to Graph ('graph.upsert_node')"):::storage
        P3_UpsertNode -- "New/Updated Entity Data (Dict)" --> P3_VDBCheck{"enable_local?"}:::decision
        P3_VDBCheck -- "Yes" --> P3_UpsertEntityVDB("Upsert to Entities VDB ('entities_vdb.upsert')"):::storage
        P3_VDBCheck -- "No" --> P3_ToEdges
        P3_UpsertEntityVDB --> P3_ToEdges
        
        P3_ToEdges --> P3_ProcessEdges("Process Unique Edges in Parallel ('asyncio.gather')"):::process
        P3_ProcessEdges -- "For each unique Edge (src, tgt)..." --> P3_MergeEdge("Call '_op._merge_edges_then_upsert'"):::process
        P3_MergeEdge -- "Uses: 'graph.has_edge', 'get_edge'" --> P3_GatherEdgeData("Combine descriptions, source_ids, weights, order"):::data
        P3_GatherEdgeData -- "Combined Description (str)" --> P3_LLM_SummarizeEdge("LLM Call: Summarize description ('_op._handle_entity_relation_summary' using 'cheap_model_func')"):::llmCall
        P3_LLM_SummarizeEdge -- "Clean, summarized description (str)" --> P3_UpsertEdge("Upsert unified Edge to Graph ('graph.upsert_edge')"):::storage
        P3_UpsertEdge --> P3_FinalGraph("Unified Knowledge Graph (BaseGraphStorage: NetworkXStorage or Neo4jStorage)"):::storage
    end

    %% =================================================================
    %% STAGE 4: GLOBAL GRAPH ANALYSIS
    %% =================================================================
    subgraph STAGE_4 ["(ainsert) 4: Community Detection"]
        direction TB
        P3_FinalGraph -- "Data: Populated Graph" --> P4_ClusterCall("Call 'graph.clustering(graph_cluster_algorithm)'"):::process
        subgraph P4_ClusteringImpl ["Storage-Specific Clustering"]
            P4_ClusterCall -- "NetworkX" --> P4_NX_Cluster("_storage/gdb_networkx.py: _leiden_clustering (uses graspologic)"):::process
            P4_ClusterCall -- "Neo4j" --> P4_Neo4j_Cluster("_storage/gdb_neo4j.py: clustering (uses GDS Leiden)"):::process
        end
        P4_NX_Cluster --> P4_WriteCommunityIDs
        P4_Neo4j_Cluster --> P4_WriteCommunityIDs("Write 'communityIds' property to each node in the graph"):::storage
    end

    %% =================================================================
    %% STAGE 5: COMMUNITY REPORT GENERATION
    %% =================================================================
    subgraph STAGE_5 ["(ainsert) 5: Community Report Generation"]
        direction TB
        P4_WriteCommunityIDs -- "Data: Graph with Community IDs" --> P5_ReportCall("Call '_op.generate_community_report()'"):::process
        P5_ReportCall --> P5_GetSchema("Get Community Schema from Graph ('graph.community_schema')"):::process
        P5_GetSchema -- "Dict[community_id, SingleCommunitySchema]" --> P5_SortLevels("Sort communities by level (deepest first)"):::process
        P5_SortLevels --> P5_LoopLevels("For each community level..."):::loop
        P5_LoopLevels --> P5_LoopCommunities("Parallel Processing of Communities in Level ('asyncio.gather')"):::loop
        P5_LoopCommunities --> P5_PackContext("Call '_op._pack_single_community_describe'"):::process
        P5_PackContext -- "Uses: 'graph.get_node/edge/degree', 'utils.truncate_list_by_token_size', 'utils.list_of_list_to_csv', '_op._pack_single_community_by_sub_communities'" --> P5_PackedContextData{"Packed Context (CSV-like string)"}:::data
        P5_PackedContextData --> P5_FormatReportPrompt("Format 'community_report' prompt ('prompt.py')"):::process
        P5_FormatReportPrompt --> P5_LLM_GenerateReport("LLM Call: Generate JSON report ('best_model_func')"):::llmCall
        P5_LLM_GenerateReport -- "LLM Response (str)" --> P5_ParseReport("Parse & Validate JSON ('convert_response_to_json_func')"):::process
        P5_ParseReport -- "Parsed Report (Dict)" --> P5_FormatStr("Format report to string ('_op._community_report_json_to_str')"):::process
        P5_FormatStr -- "CommunitySchema (Dict)" --> P5_StoreReport("Upsert final report to KV Store ('community_reports.upsert')"):::storage
        P5_StoreReport --> P5_LoopCommunities
        P5_StoreReport --> P5_LoopLevels
    end

    P5_StoreReport --> P_FinalCallback("Call '_insert_done()' callbacks for all storages"):::callback
    P_FinalCallback --> End_Ingest(["Ingestion Complete"]):::input

    %% =================================================================
    %% STAGE 6: AGENTIC WORKFLOW & QUERY
    %% =================================================================
    subgraph STAGE_6 ["Agentic Reasoning Cycle"]
        direction TB
        Agent["Agent (Internal State: Goal, History, Current Knowledge)"]:::agent
        Agent -- "1. Formulates a plan" --> AgentThought{"Agent Thought Process: 'What do I need to know next?'"}:::annotation
        AgentThought --> AgentToolChoice{"2. Chooses a tool from the HiRAG toolkit"}:::decision
        
        subgraph Q_AGENT_TOOLS ["HiRAG Toolkit (hirag.py methods)"]
            direction TB
            AgentToolChoice -- "aget_community_toc(level)" --> Q_GetToc("Get all community reports, filter by level, format to markdown"):::process
            AgentToolChoice -- "afind_entities(query, top_k, temporary)" --> Q_FindEntities("Query Entities VDB ('entities_vdb.query')"):::storage
            AgentToolChoice -- "aget_entity_details(name)" --> Q_GetNodeDetails("Get Node from Graph ('graph.get_node')"):::storage
            AgentToolChoice -- "aget_relationships(name)" --> Q_GetNodeEdges("Get Node Edges ('graph.get_node_edges')"):::storage
            AgentToolChoice -- "afind_reasoning_path(start, end, ...)" --> Q_PathNeo4j("Call backend-specific pathfinding (e.g., Neo4j's 'find_k_shortest_paths')"):::process
            AgentToolChoice -- "aget_holistic_context(query, param)" --> Q_HolisticContext("Call '_build_holistic_context'"):::process
        end

        Q_GetToc --> ToolResult
        Q_FindEntities --> ToolResult
        Q_GetNodeDetails --> ToolResult
        Q_GetNodeEdges --> ToolResult
        Q_PathNeo4j --> ToolResult
        Q_HolisticContext --> ToolResult

        ToolResult{"3. HiRAG executes tool & returns data (JSON, str, List, etc.)"}:::data
        ToolResult -- "4. Updates Agent's knowledge" --> Agent
        Agent -- "5. Goal achieved?" --> EndAgent{End Cycle}:::decision
        EndAgent -- "No, continue reasoning" --> AgentThought
        EndAgent -- "Yes" --> FinalAnswer("Synthesize Final Answer"):::input
    end

    %% =================================================================
    %% CLASS DIAGRAM
    %% =================================================================
    subgraph CLASS_DIAGRAM ["Expanded Code Structure (Class & Data Relationships)"]
        direction TB
        classDef class_main fill:#e8dff5,stroke:#9b59b6,stroke-width:2px;
        classDef class_abstract fill:#f0f0f0,stroke:#999,stroke-width:2px,font-style:italic;
        classDef class_concrete fill:#d4edda,stroke:#28a745,stroke-width:2px;
        classDef class_util fill:#e0f2f7,stroke:#3498db,stroke-width:1px;
        classDef class_data fill:#fff3cd,stroke:#ffc107,stroke-width:1px;

        HiRAG_Class[HiRAG]:::class_main
        
        BaseGraphStorage_Class["<i>BaseGraphStorage</i>"]:::class_abstract
        BaseKVStorage_Class["<i>BaseKVStorage</i>"]:::class_abstract
        BaseVectorStorage_Class["<i>BaseVectorStorage</i>"]:::class_abstract
        
        NetworkXStorage_Class[NetworkXStorage]:::class_concrete
        Neo4jStorage_Class[Neo4jStorage]:::class_concrete
        JsonKVStorage_Class[JsonKVStorage]:::class_concrete
        NanoVectorDBStorage_Class[NanoVectorDBStorage]:::class_concrete
        
        HierarchicalClustering_Class[Hierarchical_Clustering]:::class_util
        SeparatorSplitter_Class[SeparatorSplitter]:::class_util
        
        QueryParam_Schema[QueryParam]:::class_data
        TextChunkSchema_Schema[TextChunkSchema]:::class_data
        CommunitySchema_Schema[CommunitySchema]:::class_data

        HiRAG_Class "1" o-- "1" BaseGraphStorage_Class : has a
        HiRAG_Class "1" o-- "3" BaseKVStorage_Class : has a
        HiRAG_Class "1" o-- "2" BaseVectorStorage_Class : has a
        HiRAG_Class --|> HierarchicalClustering_Class : uses
        HiRAG_Class --|> SeparatorSplitter_Class : uses
        
        BaseGraphStorage_Class <|-- NetworkXStorage_Class : implements
        BaseGraphStorage_Class <|-- Neo4jStorage_Class : implements
        BaseKVStorage_Class <|-- JsonKVStorage_Class : implements
        BaseVectorStorage_Class <|-- NanoVectorDBStorage_Class : implements
        
        HiRAG_Class ..> QueryParam_Schema : uses
        JsonKVStorage_Class ..> TextChunkSchema_Schema : stores
        JsonKVStorage_Class ..> CommunitySchema_Schema : stores
    end
```