import re
import json
import asyncio
import tiktoken
import networkx as nx
import time
import logging
from contextlib import contextmanager
from typing import Union, Tuple, List, Dict, Optional, Callable, Any
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from ._validation import validate
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .config import get_text_processing_config, get_performance_config
from ._cluster_utils import Hierarchical_Clustering

# Optional instrumentation imports - set to None if not available
try:
    from ._llm import record_llm_usage_with_context
    from ._token_estimation import LLMCallType
except ImportError:
    record_llm_usage_with_context = None
    LLMCallType = None


@contextmanager
def timer():
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.info(f"[Retrieval Time: {elapsed_time:.6f} seconds]")


def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size: Optional[int] = None,
    max_token_size: Optional[int] = None,
):
    # Use configuration defaults if not provided
    text_config = get_text_processing_config()
    if overlap_token_size is None:
        overlap_token_size = text_config.overlap_token_size
    if max_token_size is None:
        max_token_size = text_config.chunk_token_size  # Use chunk_token_size as max
    # tokenizer
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size: Optional[int] = None,
    max_token_size: Optional[int] = None,
):
    # Use configuration defaults if not provided
    text_config = get_text_processing_config()
    if overlap_token_size is None:
        overlap_token_size = text_config.overlap_token_size
    if max_token_size is None:
        max_token_size = text_config.chunk_token_size
    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    # Use configuration for encoding settings
    text_config = get_text_processing_config()
    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=text_config.encoding_num_threads)
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Summarize the entity or relation description,is used during entity extraction and when merging nodes or edges in the knowledge graph

    Args:
        entity_or_relation_name: entity or relation name
        description: description
        global_config: global configuration
    """
    use_llm_func: Callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    # The tuple now has 5 elements: ("entity", name, type, desc, is_temporary)
    if not validate("record_attributes_for_entity", record_attributes):
        return None

    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(
        record_attributes[2].lower()
    )  # Standardize to lowercase for consistency
    entity_description = clean_str(record_attributes[3])

    # Parse the new 'is_temporary' field
    is_temporary_str = clean_str(record_attributes[4].lower())
    is_temporary = is_temporary_str == "true"

    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
        is_temporary=is_temporary,  # Add the new property to the dictionary
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if not validate("record_attributes_for_relationship", record_attributes):
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    existing_is_temporary = False

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:  # already exist
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
        existing_is_temporary = already_node.get("is_temporary", False)

    new_temporary_guesses = [dp.get("is_temporary", False) for dp in nodes_data]

    is_temporary_true_count = sum(1 for guess in new_temporary_guesses if guess is True)
    is_temporary_true_count += 1 if existing_is_temporary else 0
    total_count = len(new_temporary_guesses) + 1
    final_is_temporary = is_temporary_true_count / total_count > 0.9

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        is_temporary=final_is_temporary,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        if already_edge is not None:
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
            already_description.append(already_edge["description"])
            already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"{src_id}-{tgt_id}", description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )


# TODO:
# extract entities with normal and attribute entities
async def extract_hierarchical_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
    entity_names_vdb: Optional[BaseVectorStorage] = None,
    token_estimator = None,  # Add token estimator parameter
) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relations from text chunks
    
    This modified version ensures all entities have embeddings and returns raw data
    for the Entity Disambiguation and Merging (EDM) pipeline.

    Args:
        chunks: text chunks
        knowledge_graph_inst: knowledge graph instance
        entity_vdb: entity vector database
        global_config: global configuration
        entity_names_vdb: optional entity names vector database for disambiguation
        token_estimator: optional token estimator for learning

    Returns:
        Tuple[List[Dict], List[Dict]]: (raw_nodes, raw_edges) with embeddings
    """
    use_llm_func: Callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    # Check if instrumentation is available
    if token_estimator and (record_llm_usage_with_context is None or LLMCallType is None):
        logger.warning("LLM instrumentation not available, proceeding without recording")
        token_estimator = None

    ordered_chunks = list(chunks.items())
    entity_extract_prompt = PROMPTS[
        "hi_entity_extraction"
    ]  # give 3 examples in the prompt context
    relation_extract_prompt = PROMPTS["hi_relation_extraction"]

    context_base_entity = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS[
        "entiti_continue_extraction"
    ]  # means low quality in the last extraction
    if_loop_prompt = PROMPTS[
        "entiti_if_loop_extraction"
    ]  # judge if there are still entities still need to be extracted

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content_entity(
        chunk_key_dp: tuple[str, TextChunkSchema],
    ):  # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(
            **context_base_entity, input_text=content
        )  # fill in the parameter
        
        # Record LLM usage for entity extraction
        final_result = await use_llm_func(hint_prompt)  # feed into LLM with the prompt
        if token_estimator and record_llm_usage_with_context and LLMCallType:
            await record_llm_usage_with_context(
                token_estimator=token_estimator,
                call_type=LLMCallType.ENTITY_EXTRACTION,
                actual_response=final_result,
                chunk_content=content,
                chunk_size=len(content.split()),
                document_type="general",  # Could be enhanced with detection
                model_name=getattr(use_llm_func, '__name__', 'unknown_model'),
                success=True,
                metadata={"chunk_key": chunk_key, "stage": "initial_extraction"}
            )

        history = pack_user_ass_to_openai_messages(
            hint_prompt, final_result
        )  # set as history
        
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            
            # Record gleaning LLM usage
            if token_estimator and record_llm_usage_with_context and LLMCallType:
                await record_llm_usage_with_context(
                    token_estimator=token_estimator,
                    call_type=LLMCallType.CONTINUE_EXTRACTION,
                    actual_response=glean_result,
                    chunk_content=content,
                    chunk_size=len(content.split()),
                    document_type="general",
                    model_name=getattr(use_llm_func, '__name__', 'unknown_model'),
                    success=True,
                    metadata={"chunk_key": chunk_key, "gleaning_iteration": now_glean_index}
                )

            history += pack_user_ass_to_openai_messages(
                continue_prompt, glean_result
            )  # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = (
                await use_llm_func(  # judge if we still need the next iteration
                    if_loop_prompt, history_messages=history
                )
            )
            
            # Record loop detection LLM usage
            if token_estimator and record_llm_usage_with_context and LLMCallType:
                await record_llm_usage_with_context(
                    token_estimator=token_estimator,
                    call_type=LLMCallType.LOOP_DETECTION,
                    actual_response=if_loop_result,
                    chunk_content=content,
                    chunk_size=len(content.split()),
                    document_type="general",
                    model_name=getattr(use_llm_func, '__name__', 'unknown_model'),
                    success=True,
                    metadata={"chunk_key": chunk_key, "gleaning_iteration": now_glean_index}
                )
            
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(  # split entities from result --> list of entities
            final_result,
            [
                context_base_entity["record_delimiter"],
                context_base_entity["completion_delimiter"],
            ],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(  # split entity
                record, [context_base_entity["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(  # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1  # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][  # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # extract entities
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    entity_results = await asyncio.gather(
        *[_process_single_content_entity(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar

    # fetch all entities from results
    all_entities = {}
    for item in entity_results:
        for k, v in item[0].items():
            value = v[0]
            all_entities[k] = v[0]
    context_entities = {
        key[0]: list(x[0].keys()) for key, x in zip(ordered_chunks, entity_results)
    }

    # fetch embeddings for base entities
    entity_discriptions = [v["description"] for k, v in all_entities.items()]
    entity_sequence_embeddings = []
    # Use configuration for batch size
    performance_config = get_performance_config()
    embeddings_batch_size = performance_config.embeddings_batch_size
    num_embeddings_batches = (
        len(entity_discriptions) + embeddings_batch_size - 1
    ) // embeddings_batch_size
    for i in range(num_embeddings_batches):
        start_index = i * embeddings_batch_size
        end_index = min((i + 1) * embeddings_batch_size, len(entity_discriptions))
        batch = entity_discriptions[start_index:end_index]
        result = await entity_vdb.embedding_func(batch)
        entity_sequence_embeddings.extend(result)
    entity_embeddings = entity_sequence_embeddings
    for (k, v), x in zip(all_entities.items(), entity_embeddings):
        value = v
        value["embedding"] = x
        all_entities[k] = value

    already_processed = 0

    async def _process_single_content_relation(
        chunk_key_dp: tuple[str, TextChunkSchema],
    ):  # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        entities = context_entities[chunk_key]
        context_base_relation = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entities=",".join(entities),
        )
        hint_prompt = relation_extract_prompt.format(
            **context_base_relation, input_text=content
        )  # fill in the parameter
        
        # Record LLM usage for relation extraction
        final_result = await use_llm_func(hint_prompt)  # feed into LLM with the prompt
        if token_estimator and record_llm_usage_with_context and LLMCallType:
            await record_llm_usage_with_context(
                token_estimator=token_estimator,
                call_type=LLMCallType.RELATION_EXTRACTION,
                actual_response=final_result,
                chunk_content=content,
                chunk_size=len(content.split()),
                document_type="general",
                model_name=getattr(use_llm_func, '__name__', 'unknown_model'),
                success=True,
                metadata={"chunk_key": chunk_key, "entities_count": len(entities)}
            )

        history = pack_user_ass_to_openai_messages(
            hint_prompt, final_result
        )  # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            
            # Record relation gleaning LLM usage
            if token_estimator and record_llm_usage_with_context and LLMCallType:
                await record_llm_usage_with_context(
                    token_estimator=token_estimator,
                    call_type=LLMCallType.CONTINUE_EXTRACTION,
                    actual_response=glean_result,
                    chunk_content=content,
                    chunk_size=len(content.split()),
                    document_type="general",
                    model_name=getattr(use_llm_func, '__name__', 'unknown_model'),
                    success=True,
                    metadata={"chunk_key": chunk_key, "stage": "relation_gleaning", "gleaning_iteration": now_glean_index}
                )

            history += pack_user_ass_to_openai_messages(
                continue_prompt, glean_result
            )  # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = (
                await use_llm_func(  # judge if we still need the next iteration
                    if_loop_prompt, history_messages=history
                )
            )
            
            # Record relation loop detection LLM usage
            if token_estimator and record_llm_usage_with_context and LLMCallType:
                await record_llm_usage_with_context(
                    token_estimator=token_estimator,
                    call_type=LLMCallType.LOOP_DETECTION,
                    actual_response=if_loop_result,
                    chunk_content=content,
                    chunk_size=len(content.split()),
                    document_type="general",
                    model_name=getattr(use_llm_func, '__name__', 'unknown_model'),
                    success=True,
                    metadata={"chunk_key": chunk_key, "stage": "relation_loop_detection", "gleaning_iteration": now_glean_index}
                )
            
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(  # split entities from result --> list of entities
            final_result,
            [
                context_base_relation["record_delimiter"],
                context_base_relation["completion_delimiter"],
            ],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(  # split entity
                record, [context_base_relation["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(  # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1  # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][  # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # extract relations
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    relation_results = await asyncio.gather(
        *[_process_single_content_relation(c) for c in ordered_chunks]
    )
    print()

    # fetch all relations from results
    all_relations = {}
    for item in relation_results:
        for k, v in item[1].items():
            all_relations[k] = v

    # TODO: hierarchical clustering
    logger.info(f"[Hierarchical Clustering]")
    hierarchical_cluster = Hierarchical_Clustering()
    hierarchical_clustered_entities_relations = (
        await hierarchical_cluster.perform_clustering(
            entity_vdb=entity_vdb, 
            global_config=global_config, 
            entities=all_entities,
            token_estimator=token_estimator  # Pass token estimator to clustering
        )
    )
    hierarchical_clustered_entities = [
        [x for x in y if "entity_name" in x.keys()]
        for y in hierarchical_clustered_entities_relations
    ]
    hierarchical_clustered_relations = [
        [x for x in y if "src_id" in x.keys()]
        for y in hierarchical_clustered_entities_relations
    ]

    # Collect all raw nodes and edges
    maybe_nodes = defaultdict(list)  # for all chunks
    maybe_edges = defaultdict(list)
    
    # Base extracted entities and relations
    for m_nodes, m_edges in zip(entity_results, relation_results):
        for k, v in m_nodes[0].items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges[1].items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    
    # Hierarchical clustered entities
    for cluster_layer in hierarchical_clustered_entities:
        for item in cluster_layer:
            maybe_nodes[item["entity_name"]].extend([item])
    
    # Hierarchical clustered relations
    for cluster_layer in hierarchical_clustered_relations:
        for item in cluster_layer:
            maybe_edges[tuple(sorted((item["src_id"], item["tgt_id"])))].extend([item])

    # Convert to lists for disambiguation
    raw_nodes = []
    for entity_name, entity_instances in maybe_nodes.items():
        # Take the first instance as representative and ensure it has embeddings
        representative = entity_instances[0].copy()
        
        # Ensure embedding exists
        if "embedding" not in representative:
            try:
                description = representative.get("description", entity_name)
                embedding_result = await entity_vdb.embedding_func([description])
                representative["embedding"] = embedding_result[0] if embedding_result else None
            except Exception as e:
                logger.warning(f"Failed to generate embedding for entity '{entity_name}': {e}")
                representative["embedding"] = None
        
        raw_nodes.append(representative)

    raw_edges = []
    for edge_key, edge_instances in maybe_edges.items():
        # Take the first instance as representative
        representative = edge_instances[0].copy()
        raw_edges.append(representative)

    logger.info(f"Extracted {len(raw_nodes)} unique entities and {len(raw_edges)} unique relations for disambiguation")
    
    # Store entity names in dedicated vector database for efficient disambiguation
    if entity_names_vdb is not None and raw_nodes:
        try:
            logger.info(f"Storing {len(raw_nodes)} entity names in vector database for disambiguation")
            
            # Prepare data for entity names vector database
            entity_name_data = {}
            for node in raw_nodes:
                entity_name = node.get("entity_name", "")
                if entity_name and node.get("embedding") is not None:
                    # Store just the entity name (not the full description) for efficient name-based search
                    entity_name_data[entity_name] = {
                        "content": entity_name,  # Store the name itself as content
                        "entity_name": entity_name,
                        "entity_type": node.get("entity_type", ""),
                        "is_temporary": node.get("is_temporary", False),
                    }
            
            if entity_name_data:
                await entity_names_vdb.upsert(entity_name_data)
                logger.info(f"Successfully stored {len(entity_name_data)} entity names in vector database")
            else:
                logger.warning("No valid entity names to store in vector database")
                
        except Exception as e:
            logger.error(f"Failed to store entity names in vector database: {e}")
            # Don't fail the entire extraction process if name storage fails
    
    return raw_nodes, raw_edges


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: Callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())  # chunks

    entity_extract_prompt = PROMPTS[
        "entity_extraction"
    ]  # give 3 examples in the prompt context
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS[
        "entiti_continue_extraction"
    ]  # means low quality in the last extraction
    if_loop_prompt = PROMPTS[
        "entiti_if_loop_extraction"
    ]  # judge if there are still entities still need to be extracted

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(
        chunk_key_dp: tuple[str, TextChunkSchema],
    ):  # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text=content
        )  # fill in the parameter
        final_result = await use_llm_func(hint_prompt)  # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(
            hint_prompt, final_result
        )  # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(
                continue_prompt, glean_result
            )  # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = (
                await use_llm_func(  # judge if we still need the next iteration
                    if_loop_prompt, history_messages=history
                )
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(  # split entities from result --> list of entities
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(  # split entity
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(  # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1  # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][  # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)  # for all chunks
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(  # store the nodes
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(  # store the edges
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {  # key is the md5 hash of the entity name string
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": str(dp["entity_name"])
                + str(dp[
                    "description"
                ]),  # entity name and description construct the content
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knwoledge_graph_inst


def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,
    max_token_size: int,
    already_reports: dict[str, CommunitySchema],
) -> tuple[str, int, set[str], set[tuple[str, str]]]:
    # TODO
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(encode_string_by_tiktoken(sub_communities_describe)),
        set(already_nodes),
        set(already_edges),
    )


async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: Optional[int] = None,
    already_reports: dict[str, CommunitySchema] = {},
    global_config: dict = {},
) -> str:
    # Use configuration default if not provided
    if max_token_size is None:
        performance_config = get_performance_config()
        max_token_size = performance_config.community_report_max_token_size
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get("entity_type", "UNKNOWN") if node_data else "UNKNOWN",
            node_data.get("description", "UNKNOWN") if node_data else "UNKNOWN",
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN") if edge_data else "UNKNOWN",
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    token_estimator = None,  # Add token estimator parameter
):
    """Generate community reports with optional LLM usage recording"""
    
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func: Callable = global_config["best_model_func"]
    use_string_json_convert_func: Callable = global_config[
        "convert_response_to_json_func"
    ]

    # Check if instrumentation is available
    if token_estimator and (record_llm_usage_with_context is None or LLMCallType is None):
        logger.warning("LLM instrumentation not available, proceeding without recording")
        token_estimator = None

    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = (
        list(communities_schema.keys()),
        list(communities_schema.values()),
    )
    already_processed = 0

    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        prompt = community_report_prompt.format(input_text=describe)
        
        # Record LLM usage for community report generation
        response = await use_llm_func(prompt, **llm_extra_kwargs)
        if token_estimator and record_llm_usage_with_context and LLMCallType:
            # Calculate community size for context
            community_size = len(community.get("nodes", [])) + len(community.get("edges", []))
            
            await record_llm_usage_with_context(
                token_estimator=token_estimator,
                call_type=LLMCallType.COMMUNITY_REPORT,
                actual_response=response,
                chunk_content=describe,  # Use the community description as content
                chunk_size=len(describe.split()),
                document_type="general",
                model_name=getattr(use_llm_func, '__name__', 'unknown_model'),
                success=True,
                metadata={
                    "community_id": community.get("id", "unknown"),
                    "community_level": community.get("level", 0),
                    "community_size": community_size,
                    "nodes_count": len(community.get("nodes", [])),
                    "edges_count": len(community.get("edges", [])),
                }
            )
        
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # clear the progress bar
    await community_report_kv.upsert(community_datas)


async def find_most_related_community_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    community_reports: BaseKVStorage[CommunitySchema],
):
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    _related_community_datas = await asyncio.gather(  # get community reports
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    related_community_keys = sorted(  # sort by ratings
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [  # community reports sorted by ratings
        related_community_datas[k] for k in related_community_keys
    ]

    use_community_reports = truncate_list_by_token_size(  # in case community reprot is longer than token limitation
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.max_token_for_community_report,
    )
    if query_param.community_single_one:
        use_community_reports = use_community_reports[:1]
    return use_community_reports


async def find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [  # the entities related to the retrieved entities
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(  # get relations related to the retrieved entities
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )  # where the source entities are the retrieved entities
    all_one_hop_nodes = set()  # find the one hop neighbors
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(  # get node information from storage
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {  # find the text chunks of the 1-hop neighbors entities
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:  # Check if this_edges is not None or empty
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,  # count of relations related to the chunk
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units_extended = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units_sorted = sorted(  # sort by relation counts
        all_text_units_extended, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units_truncated = truncate_list_by_token_size(
        all_text_units_sorted,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units_truncated]
    return all_text_units


async def find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        if this_edges:  # Check if this_edges is not None or empty
            all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    return all_edges_data


async def find_most_related_edges_from_paths(
    path_datas: list[dict],
    path: list[str],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    # all_related_edges = await asyncio.gather(
    #     *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    # )
    # all_reasoning_path = await asyncio.gather(
    #                         *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in knowledge_graph_inst._graph.subgraph(path).edges()]
    #                     )
    # Handle different graph storage implementations
    try:
        # This works for NetworkX-based implementations
        if hasattr(knowledge_graph_inst, '_graph') and hasattr(knowledge_graph_inst._graph, 'subgraph'):  # type: ignore
            all_reasoning_path = knowledge_graph_inst._graph.subgraph(path).edges()  # type: ignore
        else:
            # Fallback: get all edges between path nodes manually
            all_reasoning_path = []
            for i in range(len(path) - 1):
                src, tgt = path[i], path[i + 1]
                if await knowledge_graph_inst.has_edge(src, tgt):
                    all_reasoning_path.append((src, tgt))
    except Exception as e:
        logger.warning(f"Failed to get subgraph edges: {e}")
        all_reasoning_path = []
    
    all_edges = set()
    all_edges.update([tuple(sorted(e)) for e in all_reasoning_path])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_bridge_knowledge,
    )
    return all_edges_data
