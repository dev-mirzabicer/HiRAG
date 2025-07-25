## Introduction

Welcome to the definitive guide for agents and developers interacting with the **HiRAG (Hierarchical Retrieval-Augmented Generation)** codebase. This document provides an exhaustive, in-depth exploration of the entire system, designed to make you an expert on its architecture, components, workflows, and underlying philosophy.

HiRAG is a sophisticated, end-to-end system for transforming unstructured text into a structured, hierarchical knowledge graph. It goes beyond traditional RAG by not only storing text chunks but also extracting entities and relationships, organizing them into a graph, and then discovering high-level conceptual communities within that graph. This multi-layered structure enables a powerful, context-aware retrieval process that can answer complex queries by navigating from high-level concepts down to the specific source text.

The system is engineered for robustness, observability, and extensibility, featuring advanced infrastructure for checkpointing, rate limiting, automatic retries, and progress tracking. It is designed to be used as a powerful toolkit for intelligent agents, providing them with a suite of tools to explore, reason about, and synthesize information from a rich knowledge base.

This guide will walk you through every file, every class, and every critical function, providing the detailed understanding necessary to leverage, maintain, and extend the HiRAG system effectively.

---

## System Architecture

The HiRAG pipeline can be understood as a series of data transformation stages, managed by a robust infrastructure. The core data flow is as follows:

1.  **Ingestion & Chunking**: Raw text documents are ingested, validated, and broken down into smaller, manageable `TextChunkSchema` objects. This process is handled by functions in `_op.py` and configurable via `hirag.py`.

2.  **Hierarchical Knowledge Extraction**: This is the core of HiRAG's innovation. An LLM processes each text chunk to perform:
    *   **Entity Extraction**: Identifying key entities (postulates, objects, concepts, etc.) and classifying them as 'temporary' or 'non-temporary'.
    *   **Relationship Extraction**: Identifying the connections between these entities.
    *   This process, defined in `_op.py` (`extract_hierarchical_entities`), produces a stream of raw nodes and edges.

3.  **Entity Disambiguation & Merging (EDM)**: Before adding to the graph, the system identifies and merges entities that are aliases for the same concept. This crucial step, managed by the `EntityDisambiguator` class in `_disambiguation.py`, uses a combination of lexical similarity, semantic similarity (vector search), and LLM-based verification to create a clean, canonical set of entities.

4.  **Graph Construction**: The disambiguated entities and their relationships are upserted into a graph database. This creates a structured network of knowledge. HiRAG supports multiple backends (`NetworkXStorage`, `Neo4jStorage`) defined in the `_storage` directory.

5.  **Graph Clustering & Community Detection**: Once the graph is built, a clustering algorithm (e.g., Leiden) is run to identify densely connected subgraphs, or "communities." This process, initiated from `hirag.py` and implemented in the storage backends (`gdb_networkx.py`, `gdb_neo4j.py`), groups related concepts together.

6.  **Community Report Generation**: For each discovered community, an LLM is used to generate a high-level, human-readable report. This report summarizes the community's core theme, key entities, and overall importance. This is orchestrated by `generate_community_report` in `_op.py`.

7.  **Storage**: Throughout the process, data is persisted across multiple storage layers:
    *   **KV Storage (`kv_json.py`)**: Stores documents, text chunks, community reports, and caches.
    *   **Vector Storage (`vdb_nanovectordb.py`)**: Stores embeddings for entities and (optionally) chunks to enable semantic search. A dedicated VDB is used for entity names to accelerate the disambiguation process.
    *   **Graph Storage (`gdb_networkx.py`, `gdb_neo4j.py`)**: Stores the final knowledge graph of entities and relationships.

8.  **Agent Toolkit**: The final output is not just a single RAG system, but a toolkit of queryable methods (`afind_entities`, `aget_relationships`, `afind_reasoning_path`, etc.) in `hirag.py` that an intelligent agent can use to explore the knowledge base dynamically.

This entire pipeline is supported by a suite of advanced infrastructure modules that manage its execution: `_checkpointing.py`, `_retry_manager.py`, `_rate_limiting.py`, `_token_estimation.py`, and `_progress_tracking.py`.

---

## Directory and File Structure

The codebase is organized into a single main package, `hirag`, with a logical separation of concerns.

```
└── ./
    └── hirag
        ├── _storage                # Pluggable storage backend implementations.
        │   ├── __init__.py         # Makes the storage directory a package.
        │   ├── gdb_neo4j.py        # Neo4j graph database storage implementation.
        │   ├── gdb_networkx.py     # NetworkX in-memory graph storage implementation.
        │   ├── kv_json.py          # Simple JSON-based key-value storage.
        │   └── vdb_nanovectordb.py # NanoVectorDB for vector storage.
        ├── __init__.py             # Package initializer, exposes HiRAG class.
        ├── _checkpointing.py       # System for saving and resuming pipeline progress.
        ├── _cluster_utils.py       # Algorithms for hierarchical entity clustering.
        ├── _disambiguation.py      # Entity Disambiguation and Merging (EDM) pipeline.
        ├── _estimation_db.py       # Database for learning and improving token estimates.
        ├── _llm.py                 # Wrappers for interacting with LLM APIs (OpenAI, Gemini, Azure).
        ├── _op.py                  # Core data processing operations (chunking, extraction, reporting).
        ├── _progress_tracking.py   # Real-time progress monitoring and dashboard.
        ├── _rate_limiting.py       # Manages API call rates to avoid hitting limits.
        ├── _retry_manager.py       # Handles transient errors with backoff and circuit breakers.
        ├── _splitter.py            # Text splitting utilities.
        ├── _token_estimation.py    # System for estimating API token usage and cost.
        ├── _utils.py               # General utility functions (hashing, JSON handling, etc.).
        ├── _validation.py          # Centralized data validation functions.
        ├── base.py                 # Core data schemas and abstract base classes for storage.
        ├── hirag.py                # Main HiRAG class, configuration, and public interface (Agent Toolkit).
        └── prompt.py               # All LLM prompts used in the system.
```

---

## File-by-File Deep Dive

This section provides a rigorous analysis of each file in the codebase.

### `/hirag/hirag.py`

*   **Purpose**: This is the main entry point and central orchestrator of the HiRAG system. It defines the `HiRAG` dataclass, which holds all configuration parameters, initializes all storage backends and infrastructure systems, and exposes the public API for both indexing (`ainsert`) and querying (the Agent Toolkit).

*   **Key Classes & Functions**:
    *   **`HiRAG` (dataclass)**: This class is the heart of the system.
        *   **Configuration**: It holds over 50 configuration fields, controlling everything from chunking strategy and LLM models to the behavior of the advanced infrastructure systems (e.g., `enable_checkpointing`, `edm_lexical_similarity_threshold`).
        *   **`__post_init__`**: This method is critical. It reads the configuration, creates instances of all necessary storage backends (KV, Vector, Graph), initializes the infrastructure managers (`TokenEstimator`, `CheckpointManager`, `RetryManager`, `RateLimiter`, `ProgressTracker`), and wraps the LLM functions with rate limiting and retry logic. It also initializes the `EntityDisambiguator`.
        *   **`ainsert(string_or_strings)`**: The primary method for indexing data. It orchestrates the entire pipeline: chunking, hierarchical entity/relation extraction, entity disambiguation, graph upsertion, clustering, and community report generation. The implementation is robust, with detailed error handling and state tracking.
        *   **Agent Toolkit Methods**: This is the modern query interface.
            *   `aget_community_toc(level)`: Provides a high-level summary of communities.
            *   `afind_entities(query, top_k, temporary)`: Performs semantic search for entities.
            *   `aget_entity_details(entity_name)`: Retrieves all data for a specific entity.
            *   `aget_relationships(entity_name)`: Fetches all connections for an entity.
            *   `aget_source_text(entity_name)`: Retrieves the original text chunks for an entity.
            *   `afind_reasoning_path(...)`: Finds the shortest path between two entities, supporting multiple algorithms and backends (NetworkX, Neo4j). It contains backend-specific logic (`_find_path_networkx`, `_find_path_neo4j`).
            *   `afind_multiple_reasoning_paths(...)`: Finds the k-shortest paths.
        *   **`aquery(query, param)` (Deprecated)**: The legacy, monolithic query method. It builds a single, large context string and passes it to an LLM. It is preserved for backward compatibility but superseded by the agent toolkit.
        *   **`_upsert_disambiguated_graph(...)`**: A crucial internal method that takes the raw extracted nodes and edges, applies the canonical mapping from the disambiguation step, and carefully merges and upserts the data into the graph storage, handling potential conflicts and self-loops.

### `/hirag/base.py`

*   **Purpose**: Defines the abstract foundations and data contracts for the entire system. It ensures that different storage implementations adhere to a consistent interface and establishes the core data schemas.

*   **Key Classes & Schemas**:
    *   **`QueryParam` (dataclass)**: Defines all parameters that control the behavior of a query, such as retrieval mode, hierarchy level, and token limits for different context types.
    *   **`TextChunkSchema` (TypedDict)**: The schema for a single chunk of text, containing its content, token count, and source document ID.
    *   **`SingleCommunitySchema` & `CommunitySchema` (TypedDict)**: Defines the structure of a community, including its nodes, edges, level, and the generated report.
    *   **`StorageNameSpace` (dataclass)**: A base class for all storage backends, providing a common namespace and configuration.
    *   **`BaseVectorStorage`, `BaseKVStorage`, `BaseGraphStorage` (Abstract Classes)**: These define the required methods for any new storage implementation. For example, `BaseGraphStorage` mandates methods like `upsert_node`, `get_edge`, `clustering`, etc., ensuring that the core logic in `hirag.py` can work with any compliant backend.

### `/hirag/_op.py`

*   **Purpose**: Contains the core, stateless data processing operations that form the steps of the ingestion pipeline. These functions are called by `hirag.py` to transform data from one stage to the next.

*   **Key Functions**:
    *   **`get_chunks(...)`**: Takes raw documents and uses a specified chunking function (`chunking_by_token_size` or `chunking_by_seperators`) to produce a dictionary of `TextChunkSchema`.
    *   **`extract_hierarchical_entities(...)`**: This is a major function. It takes text chunks and orchestrates the LLM-based extraction of entities and relationships. It manages a multi-step process for each chunk, including an initial extraction, subsequent "gleaning" loops to find missed entities, and a check to prevent infinite loops. It now returns raw nodes and edges, including embeddings, ready for the disambiguation pipeline. It also handles storing entity names in a dedicated VDB if enabled.
    *   **`_merge_nodes_then_upsert(...)` & `_merge_edges_then_upsert(...)`**: These functions contain the logic for merging data. When multiple extractions produce the same entity or relationship, these functions intelligently combine their properties (e.g., summing weights, concatenating descriptions, choosing the most common entity type) before upserting the final, merged record into the graph. They also handle summarizing long descriptions using an LLM.
    *   **`generate_community_report(...)`**: Manages the process of creating summary reports for each community. It iterates through communities by level (from most specific to most general), packs the community's data into a context string for the LLM using `_pack_single_community_describe`, calls the LLM with the appropriate prompt, and stores the resulting JSON report.
    *   **Query Helper Functions (Deprecated)**: Functions like `find_most_related_community_from_entities` are now largely superseded by the more encapsulated logic within the `HiRAG` class's agent toolkit methods.

### `/hirag/_disambiguation.py`

*   **Purpose**: Implements the complete Entity Disambiguation and Merging (EDM) pipeline. Its goal is to identify and resolve aliases in the extracted entities before they are added to the knowledge graph, ensuring data quality and consistency.

*   **Key Classes & Functions**:
    *   **`UnionFind`**: An efficient data structure used to group potential aliases into clusters during the candidate generation phase.
    *   **`DisambiguationConfig`**: A dataclass holding all configuration parameters for the EDM process, such as similarity thresholds, concurrency limits, and batch sizes.
    *   **`EntityDisambiguator`**: The main class that orchestrates the disambiguation process.
        *   **`run(raw_nodes)`**: The main entry point. It executes the two-stage pipeline:
            1.  **`_generate_candidates(raw_nodes)`**: This first stage finds potential aliases. It uses two passes: `_lexical_similarity_pass` (using `thefuzz` for string matching) and `_semantic_similarity_pass` (using cosine similarity on entity embeddings). The semantic pass is highly optimized to use a vector database (`entity_names_vdb`) for efficient nearest-neighbor searches, avoiding a slow N^2 comparison.
            2.  **`_verify_candidates_with_llm(...)`**: This second stage takes the candidate clusters and sends each one to an LLM for a final judgment. It constructs a detailed prompt (`entity_disambiguation` from `prompt.py`) containing the entities' names, descriptions, and original source text. The LLM is asked to decide whether to "MERGE" or "DO_NOT_MERGE" and provide a justification. This function uses an `asyncio.Semaphore` for dynamic concurrency control.
        *   **`_parse_and_validate_llm_decision(...)`**: A robust parser for the LLM's JSON output, ensuring the decision is well-formed and consistent with the input cluster before it is accepted.
    *   **Configuration Helpers**: `create_disambiguation_config`, `validate_edm_configuration`, etc., provide utilities for managing and validating the EDM settings.

### `/hirag/_storage/` (Directory)

This directory contains the concrete implementations of the storage base classes defined in `base.py`.

*   **`gdb_networkx.py`**: An in-memory graph storage implementation using the `networkx` library. It's simple and fast for smaller datasets. It loads/saves the graph to a `.graphml` file. It implements the `clustering` method using the `graspologic` library's Leiden algorithm.
*   **`gdb_neo4j.py`**: A robust, persistent graph storage implementation for the Neo4j graph database. It translates HiRAG operations into Cypher queries. It features a comprehensive and powerful pathfinding implementation (`find_shortest_path`, `find_all_shortest_paths`, `find_k_shortest_paths`) that can leverage modern Cypher syntax, APOC procedures, and the Graph Data Science (GDS) library for advanced algorithms like Dijkstra and Yen's.
*   **`kv_json.py`**: A basic key-value store that uses a single JSON file as its backend. Simple and portable, but not suitable for very large datasets.
*   **`vdb_nanovectordb.py`**: A lightweight, file-based vector database implementation. It handles the creation of embeddings (by calling the configured `embedding_func`) and performs cosine similarity searches.

### `/hirag/_llm.py`

*   **Purpose**: Provides standardized, asynchronous, and fault-tolerant wrappers for various LLM and embedding model APIs. It abstracts away the specific client libraries for OpenAI, Azure OpenAI, and Google Gemini.

*   **Key Functions**:
    *   **`get_*_async_client_instance()`**: Singleton patterns to get or create API client instances.
    *   **`openai_complete_if_cache`, `gemini_complete_if_cache`, `azure_openai_complete_if_cache`**: These are the core completion functions. They are wrapped with a `@retry` decorator from the `tenacity` library, which automatically retries API calls on specific transient errors (like `RateLimitError` or `APIConnectionError`) with exponential backoff. They also integrate with the `hashing_kv` (LLM cache) to avoid redundant API calls.
    *   **Model-specific wrappers**: Functions like `gpt_4o_complete`, `gemini_pro_complete`, etc., are convenient shortcuts that call the underlying completion functions with the correct model name.
    *   **Embedding functions**: `openai_embedding`, `gemini_embedding`, `azure_openai_embedding` are the corresponding functions for generating embeddings. They are decorated with `@wrap_embedding_func_with_attrs` to attach metadata like embedding dimension and max token size, which is used elsewhere in the system.

### `/hirag/prompt.py`

*   **Purpose**: A centralized repository for all LLM prompts used in the HiRAG pipeline. This separation of logic and prompts makes the system easier to maintain and customize.

*   **Key Prompts**:
    *   **`hi_entity_extraction`**: A highly detailed and complex prompt that instructs the LLM on how to extract entities. It includes strict rules about factual information, handling proofs, and the crucial distinction between 'temporary' and 'non-temporary' entities.
    *   **`hi_relation_extraction`**: Instructs the LLM on how to find relationships between the entities extracted in the previous step.
    *   **`summary_clusters`**: Used during hierarchical clustering to have the LLM synthesize a new, higher-level entity that summarizes a cluster of existing entities.
    *   **`entity_disambiguation`**: The prompt used in the EDM pipeline. It presents a cluster of potential aliases to the LLM and asks for a structured JSON output with a "MERGE" or "DO_NOT_MERGE" decision, confidence score, and justification.
    *   **`community_report`**: A detailed prompt that asks the LLM to act as a research analyst and write a structured JSON report about a community of entities, including a title, summary, importance rating, and detailed findings.

### Advanced Infrastructure Modules

These modules provide the robust, cross-cutting concerns that make the HiRAG pipeline reliable and observable.

*   **`/_checkpointing.py`**:
    *   **Purpose**: To make the long-running ingestion process resumable. If the pipeline fails, it can be restarted and will continue from the last successfully completed stage.
    *   **Key Classes**: `CheckpointStage` and `CheckpointStatus` (Enums), `StageCheckpoint` and `PipelineCheckpoint` (Dataclasses to hold state), and `CheckpointManager`.
    *   **Workflow**: The `CheckpointManager` creates a unique session ID and saves the `PipelineCheckpoint` state to the KV store at the end of each stage (e.g., after `CHUNK_CREATION`, `ENTITY_EXTRACTION`). When HiRAG starts, it checks for a resumable session and, if found, determines which stage to resume from. It also tracks processed items (like chunk IDs) to avoid re-processing them on resume.

*   **`/_retry_manager.py`**:
    *   **Purpose**: To handle transient errors (e.g., network issues, temporary API server errors) gracefully.
    *   **Key Classes**: `FailureType` and `RetryStrategy` (Enums), `RetryConfig` (Dataclass), `CircuitBreaker`, and `RetryManager`.
    *   **Workflow**: The `RetryManager` wraps LLM calls. If an error occurs, it uses `classify_failure` to categorize it (e.g., `RATE_LIMIT`, `SERVER_ERROR`). Based on the category, it determines if a retry is appropriate and calculates a delay using exponential backoff with jitter. It also implements a **Circuit Breaker** pattern: if a specific type of call fails repeatedly, the circuit "opens," and subsequent calls fail immediately for a "recovery timeout" period, preventing the system from overwhelming a failing service.

*   **`/_rate_limiting.py`**:
    *   **Purpose**: To prevent the system from exceeding API rate limits (e.g., requests per minute, tokens per minute).
    *   **Key Classes**: `RateLimitType` (Enum), `RateLimitConfig`, `ModelRateConfig`, `TokenBucket`, and `RateLimiter`.
    *   **Workflow**: The `RateLimiter` uses the **Token Bucket algorithm**. For each model and limit type (e.g., `gpt-4o-mini_requests_per_minute`), it maintains a bucket of "tokens." Before an API call, the system must `acquire` permission, which checks if all relevant buckets have enough tokens. If not, it waits asynchronously until the buckets are refilled over time. This smooths out bursts of requests and ensures compliance with API limits. It also supports adaptive backpressure.

*   **`/_token_estimation.py`**:
    *   **Purpose**: To predict the token usage and monetary cost of a pipeline run *before* it starts.
    *   **Key Classes**: `LLMCallType` (Enum), `TokenEstimate` and `PipelineEstimate` (Dataclasses), `ModelPricing`, and `TokenEstimator`.
    *   **Workflow**: The `TokenEstimator` analyzes the input documents and the pipeline configuration. It calculates the number of chunks and then iterates through each stage of the pipeline, creating a `TokenEstimate` for every anticipated LLM call. It does this by combining fixed token counts from prompt templates with dynamic estimates for variable content (e.g., estimating that a 1200-token chunk will yield ~5 entities). The final `PipelineEstimate` provides a detailed breakdown of expected token usage and cost.

*   **`/_estimation_db.py`**:
    *   **Purpose**: To improve the accuracy of the `TokenEstimator` over time.
    *   **Key Classes**: `UsageRecord`, `UsageStatistics`, `EstimationDatabase`.
    *   **Workflow**: After an actual LLM call, the system can record the *actual* token usage to the `EstimationDatabase`. Over time, the `TokenEstimator` can analyze this historical data to learn and refine its internal parameters (e.g., it might learn that for a specific type of document, a chunk yields 7 entities on average, not 5), making future estimates more accurate.

*   **`/_progress_tracking.py`**:
    *   **Purpose**: To provide real-time monitoring of the ingestion pipeline.
    *   **Key Classes**: `DashboardType` (Enum), `ProgressMetrics`, `LiveStatistics`, and `ProgressTracker`.
    *   **Workflow**: The `ProgressTracker` runs in a background asyncio task. It collects metrics from all other infrastructure components (checkpoints, retries, rate limits) and aggregates them. If `rich` is installed, it displays a live, updating terminal dashboard with progress bars, ETA, token usage, and error statistics. This gives the user a clear view into the pipeline's status.

---

## Core Workflows

### 1. Indexing Workflow (`ainsert`)

This is the step-by-step journey of a document through the HiRAG pipeline.

1.  **Start & Checkpoint**: `ainsert` is called. The `CheckpointManager` is checked for a resumable session. If none, a new session is started.
2.  **Document Ingestion**: Input strings are hashed to create unique document IDs and stored in the `full_docs` KV store.
3.  **Chunking**: `get_chunks` is called. The documents are split into `TextChunkSchema` objects based on the configured token size and overlap. These are stored in the `text_chunks` KV store.
4.  **Token Estimation (Optional)**: Before starting, the `TokenEstimator` can be used to predict the cost and duration of the run.
5.  **Progress Tracking Start**: The `ProgressTracker` starts, displaying the dashboard.
6.  **Hierarchical Extraction**: `extract_hierarchical_entities` is called. It iterates through each new chunk.
    *   For each chunk, an LLM call is made using the `hi_entity_extraction` prompt.
    *   The call is managed by the `RateLimiter` (waits if necessary) and `RetryManager` (retries on failure).
    *   The LLM response is parsed to get a list of raw entity and relationship dictionaries.
    *   This process may loop with "gleaning" prompts to ensure all information is extracted.
    *   The function gathers all raw nodes and edges from all chunks.
7.  **Entity Disambiguation**: The collected `raw_nodes` are passed to `EntityDisambiguator.run()`.
    *   **Candidate Generation**: `UnionFind` is used to cluster entities based on high lexical (`thefuzz`) and semantic (vector search on `entity_names_vdb`) similarity.
    *   **LLM Verification**: Each candidate cluster is sent to the LLM with the `entity_disambiguation` prompt for a final MERGE/DO_NOT_MERGE decision.
    *   The result is a `name_to_canonical_map` dictionary.
8.  **Graph Upsertion**: `_upsert_disambiguated_graph` is called.
    *   It iterates through the raw nodes and edges, replacing alias names with their canonical names from the map.
    *   It groups all instances of the same canonical entity/relationship together.
    *   `_merge_nodes_then_upsert` and `_merge_edges_then_upsert` are called to combine the data and write it to the configured graph storage (`NetworkX` or `Neo4j`).
    *   Canonical entities are also upserted into the `entities_vdb` for semantic search.
9.  **Clustering**: The `clustering` method on the graph storage object is called. This runs an algorithm (e.g., Leiden) to partition the graph into communities. The community ID for each node is stored as a node property.
10. **Community Reporting**: `generate_community_report` is called.
    *   It reads the community structure from the graph.
    *   For each community, it packs its entities and relationships into a context string.
    *   It calls the LLM with the `community_report` prompt to generate a JSON summary.
    *   The final reports are saved to the `community_reports` KV store.
11. **Finalization**: All storage backends commit their changes (`index_done_callback`). The checkpoint is marked as `COMPLETED`. The progress tracker stops.

### 2. Querying Workflow (Agent Toolkit)

This workflow describes how an intelligent agent uses the HiRAG toolkit to answer a complex question, such as "How does the concept of a Weak Cartesian Closed Category (wCCC) relate to the λβp' calculus?"

1.  **High-Level Overview**: The agent first wants to understand the main topics in the knowledge base. It calls `aget_community_toc(level=0)`. This returns a list of top-level community summaries. The agent sees a community titled "Weak Cartesian Closed Categories (wCCCs) as a Model for Typed Combinatory Logic" and identifies it as relevant.

2.  **Zooming In**: The agent decides to investigate this community further. It calls `aget_community_details(community_id=...)`. This returns the full report for that community, along with a list of its key entities and relationships. The agent sees "WCCL (FORMAL THEORY OF WCCC)" and "λβp' (TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING)" listed as key entities.

3.  **Entity Exploration**: The agent now wants to understand these two entities in detail.
    *   It calls `aget_entity_details(entity_name="WCCL (FORMAL THEORY OF WCCC)")`.
    *   It calls `aget_entity_details(entity_name="λβp' (TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING)")`.
    *   These calls return the full, merged descriptions of each entity.

4.  **Finding the Connection**: The agent's primary goal is to find the relationship between them. It calls `afind_reasoning_path(start_entity="WCCL (FORMAL THEORY OF WCCC)", end_entity="λβp' (TYPED λβ-CALCULUS WITH SURJECTIVE PAIRING)")`.
    *   The system queries the graph database (`Neo4j` or `NetworkX`) to find the shortest path.
    *   It returns a path object containing the nodes and edges connecting the two entities. The agent discovers a direct edge with the description: "Are proven to be equivalent formal systems, connected via equality-preserving translations."

5.  **Gathering Evidence**: The agent now has a direct claim but wants the original proof or definition. For both the "WCCL" and "λβp'" entities, it calls `aget_source_text(entity_name=...)`. This retrieves the original text chunks where these entities were defined.

6.  **Synthesis**: The agent now has all the necessary information: the high-level summary, the detailed entity definitions, the direct relationship connecting them, and the original source text. It can now synthesize a comprehensive, accurate, and fully cited answer to the original question and present it to the user.

---

## Conclusion and Developer Guide

The HiRAG codebase represents a powerful and flexible framework for building advanced knowledge graphs. Its modular architecture, pluggable storage, and robust infrastructure make it a solid foundation for complex information retrieval and agentic systems.