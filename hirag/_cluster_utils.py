import logging
import random
import re
import numpy as np
import tiktoken
import umap
import copy
import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Callable, Any, Union
import numpy.typing as npt
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from collections import Counter, defaultdict
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage
)
from ._utils import split_string_by_multi_markers, clean_str, is_float_regex
from ._validation import validate
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .config import get_graph_operations_config, get_entity_processing_config

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

# Get configuration for default values, but allow runtime override
def get_clustering_defaults():
    """Get clustering defaults from centralized configuration"""
    graph_config = get_graph_operations_config()
    entity_config = get_entity_processing_config()
    return {
        'random_seed': graph_config.random_seed,
        'umap_n_neighbors': graph_config.umap_n_neighbors,
        'umap_metric': graph_config.umap_metric,
        'max_clusters': graph_config.max_clusters,
        'gmm_n_init': graph_config.gmm_n_init,
        'hierarchical_layers': graph_config.hierarchical_layers,
        'max_length_in_cluster': entity_config.max_length_in_cluster,
        'reduction_dimension': graph_config.reduction_dimension,
        'cluster_threshold': graph_config.cluster_threshold,
        'similarity_threshold': graph_config.similarity_threshold,
        'embeddings_batch_size': 64  # This was in the original code
    }

# Set a random seed for reproducibility using configuration
_defaults = get_clustering_defaults()
RANDOM_SEED = _defaults['random_seed']
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: Optional[str] = None
) -> np.ndarray:
    # Use configuration defaults if not provided
    defaults = get_clustering_defaults()
    if n_neighbors is None:
        n_neighbors = defaults['umap_n_neighbors']
    if metric is None:
        metric = defaults['umap_metric']
        
    # Auto-calculate n_neighbors if still None
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    
    # Ensure metric is not None for type safety
    if metric is None:
        metric = 'euclidean'  # Default fallback
        
    reduced_embeddings_result = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    # Handle the potential tuple return from UMAP and ensure we get ndarray
    if isinstance(reduced_embeddings_result, tuple):
        reduced_embeddings: np.ndarray = reduced_embeddings_result[0]
    else:
        reduced_embeddings: np.ndarray = np.asarray(reduced_embeddings_result)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, 
    dim: int, 
    num_neighbors: Optional[int] = None, 
    metric: Optional[str] = None
) -> np.ndarray:
    # Use configuration defaults if not provided
    defaults = get_clustering_defaults()
    if num_neighbors is None:
        num_neighbors = 10  # Keep original default for local clustering
    if metric is None:
        metric = defaults['umap_metric']
    
    # Ensure metric is not None for type safety
    if metric is None:
        metric = 'euclidean'  # Default fallback
        
    reduced_embeddings_result = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    # Handle the potential tuple return from UMAP and ensure we get ndarray
    if isinstance(reduced_embeddings_result, tuple):
        reduced_embeddings: np.ndarray = reduced_embeddings_result[0]
    else:
        reduced_embeddings: np.ndarray = np.asarray(reduced_embeddings_result)
    return reduced_embeddings


def fit_gaussian_mixture(n_components, embeddings, random_state):
    gm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        n_init=5,
        init_params='k-means++'
        )
    gm.fit(embeddings)
    return gm.bic(embeddings)


def get_optimal_clusters(embeddings, max_clusters: Optional[int] = None, random_state: int = 0, rel_tol: float = 1e-3):
    # Use configuration defaults if not provided
    defaults = get_clustering_defaults()
    _max_clusters = defaults['max_clusters'] if max_clusters is None else max_clusters
    _max_clusters = min(len(embeddings), _max_clusters)
    n_clusters = np.arange(1, _max_clusters)
    bics = []
    prev_bic = float('inf')
    for n in tqdm(n_clusters):
        bic = fit_gaussian_mixture(n, embeddings, random_state)
        # print(bic)
        bics.append(bic)
        # early stop
        if (abs(prev_bic - bic) / abs(prev_bic)) < rel_tol:
            break
        prev_bic = bic
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0, n_init: Optional[int] = None):
    # Use configuration defaults if not provided
    defaults = get_clustering_defaults()
    _n_init = defaults['gmm_n_init'] if n_init is None else n_init
        
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(
        n_components=n_clusters, 
        random_state=random_state, 
        n_init=_n_init,
        init_params='k-means++')
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)        # [num, cluster_num]
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(     # (num, 2)
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_clusters = [[] for _ in range(len(embeddings))]
    embedding_to_index = {tuple(embedding): idx for idx, embedding in enumerate(embeddings)}
    for i in tqdm(range(n_global_clusters)):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue

        # embedding indices
        indices = [
            embedding_to_index[tuple(embedding)]
            for embedding in global_cluster_embeddings_
        ]

        # update
        for idx in indices:
            all_clusters[idx].append(i)

    all_clusters = [np.array(cluster) for cluster in all_clusters]

    if verbose:
        logging.info(f"Total Clusters: {n_global_clusters}")
    return all_clusters


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if not validate("record_attributes_for_entity_clustering", record_attributes):
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if not validate("record_attributes_for_relationship_clustering", record_attributes):
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


class ClusteringAlgorithm(ABC):
    @abstractmethod
    async def perform_clustering(self, entities: Dict, **kwargs) -> list[list[dict]]:
        pass


class Hierarchical_Clustering(ClusteringAlgorithm):
    async def perform_clustering(
        self,
        entities: Dict,
        **kwargs,
    ) -> list[list[dict]]:
        # Use configuration defaults if not provided
        defaults = get_clustering_defaults()
        layers = kwargs.get("layers", defaults['hierarchical_layers'])
        max_length_in_cluster = kwargs.get("max_length_in_cluster", defaults['max_length_in_cluster'])
        reduction_dimension = kwargs.get("reduction_dimension", defaults['reduction_dimension'])
        cluster_threshold = kwargs.get("cluster_threshold", defaults['cluster_threshold'])
        threshold = kwargs.get("threshold", defaults['similarity_threshold'])
        use_llm_func: Callable[..., Any] = kwargs["global_config"]["best_model_func"]
        verbose = kwargs.get("verbose", False)
        tokenizer = kwargs.get("tokenizer", tiktoken.get_encoding("cl100k_base"))
        thredshold_change_rate = kwargs.get("thredshold_change_rate", 0.05)

        # Get the embeddings from the nodes
        nodes = list(entities.values())
        embeddings = np.array([x["embedding"] for x in nodes])
        
        hierarchical_clusters = [nodes]
        pre_cluster_sparsity = 0.01
        for layer in range(layers):
            logging.info(f"############ Layer[{layer}] Clustering ############")
            # Perform the clustering
            clusters = perform_clustering(
                embeddings, dim=reduction_dimension, threshold=cluster_threshold
            )
            # Initialize an empty list to store the clusters of nodes
            node_clusters = []
            # Iterate over each unique label in the clusters
            unique_clusters = np.unique(np.concatenate(clusters))
            logging.info(f"[Clustered Label Num: {len(unique_clusters)} / Last Layer Total Entity Num: {len(nodes)}]")
            # calculate the number of nodes belong to each cluster
            cluster_sizes = Counter(np.concatenate(clusters))
            # calculate cluster sparsity
            cluster_sparsity = 1 - sum([x * (x - 1) for x in cluster_sizes.values()])/(len(nodes) * (len(nodes) - 1))
            cluster_sparsity_change_rate = (abs(cluster_sparsity - pre_cluster_sparsity) / (pre_cluster_sparsity + 1e-8))
            pre_cluster_sparsity = cluster_sparsity
            logging.info(f"[Cluster Sparsity: {round(cluster_sparsity, 4) * 100}%]")
            # stop if there will be no improvements on clustering
            if cluster_sparsity >= threshold:
                logging.info(f"[Stop Clustering at Layer{layer} with Cluster Sparsity {cluster_sparsity}]")
                break
            if cluster_sparsity_change_rate < thredshold_change_rate:
                logging.info(f"[Stop Clustering at Layer{layer} with Cluster Sparsity Change Rate {round(cluster_sparsity_change_rate, 4) * 100}%]")
                break
            # summarize
            for label in unique_clusters:
                # Get the indices of the nodes that belong to this cluster
                indices = [i for i, cluster in enumerate(clusters) if label in cluster]
                # Add the corresponding nodes to the node_clusters list
                cluster_nodes = [nodes[i] for i in indices]
                # Base case: if the cluster only has one node, do not attempt to recluster it
                logging.info(f"[Label{str(int(label))} Size: {len(cluster_nodes)}]")
                if len(cluster_nodes) == 1:
                    node_clusters += cluster_nodes
                    continue
                # Calculate the total length of the text in the nodes
                total_length = sum(
                    [len(tokenizer.encode(node["description"])) + len(tokenizer.encode(node["entity_name"])) for node in cluster_nodes]
                )
                base_discount = 0.8
                discount_times = 0
                # If the total length exceeds the maximum allowed length, reduce the node size
                while total_length > max_length_in_cluster:
                    logging.info(
                        f"Reducing cluster size with {base_discount * 100 * (base_discount**discount_times):.2f}% of entities"
                    )

                    # for node in cluster_nodes:
                    #     description = node["description"]
                    #     node['description'] = description[:int(len(description) * base_discount)]
                    
                    # Randomly select 80% of the nodes
                    num_to_select = max(1, int(len(cluster_nodes) * base_discount))  # Ensure at least one node is selected
                    cluster_nodes = random.sample(cluster_nodes, num_to_select)

                    # Recalculate the total length
                    total_length = sum(
                        [len(tokenizer.encode(node["description"])) + len(tokenizer.encode(node["entity_name"])) for node in cluster_nodes]
                    )
                    discount_times += 1
                # summarize and generate new entities
                entity_description_list = [f"({x['entity_name']}, {x['description']})" for x in cluster_nodes]
                context_base_summarize = dict(
                    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
                    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
                    meta_attribute_list=PROMPTS["META_ENTITY_TYPES"],
                    entity_description_list=",".join(entity_description_list)
                    )
                summarize_prompt = PROMPTS["summary_clusters"]
                hint_prompt = summarize_prompt.format(**context_base_summarize)
                summarize_result = await use_llm_func(hint_prompt)
                
                # Record LLM usage for hierarchical clustering
                token_estimator = kwargs.get("token_estimator")
                try:
                    from ._llm import record_llm_usage_with_context
                    from ._token_estimation import LLMCallType
                    
                    cluster_description = f"Clustering {len(cluster_nodes)} entities"
                    await record_llm_usage_with_context(
                        token_estimator=token_estimator,
                        call_type=LLMCallType.HIERARCHICAL_CLUSTERING,
                        actual_response=summarize_result,
                        chunk_content=cluster_description,
                        chunk_size=len(cluster_description.split()),
                        document_type="general",
                        model_name=getattr(use_llm_func, '__name__', 'unknown_model'),
                        success=True,
                        metadata={
                            "layer": layer,
                            "cluster_label": int(label),
                            "cluster_size": len(cluster_nodes),
                            "total_length": total_length,
                        }
                    )
                except ImportError:
                    pass  # Skip recording if imports fail
                chunk_key = ""
                # resolve results
                records = split_string_by_multi_markers(                                            # split entities from result --> list of entities
                    summarize_result,
                    [context_base_summarize["record_delimiter"], context_base_summarize["completion_delimiter"]],
                )
                maybe_nodes = defaultdict(list)
                maybe_edges = defaultdict(list)
                for record in records:
                    record = re.search(r"\((.*)\)", record)
                    if record is None:
                        continue
                    record = record.group(1)
                    record_attributes = split_string_by_multi_markers(          # split entity
                        record, [context_base_summarize["tuple_delimiter"]]
                    )
                    if_entities = await _handle_single_entity_extraction(       # get the name, type, desc, source_id of entity--> dict
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
                # fetch all entities from results
                entity_results = (dict(maybe_nodes), dict(maybe_edges))
                all_entities_relations = {}
                for item in entity_results:
                    for k, v in item.items():
                        value = v[0]
                        all_entities_relations[k] = v[0]
                # fetch embeddings
                entity_discriptions = [v["description"] for k, v in all_entities_relations.items()]
                entity_sequence_embeddings = []
                embeddings_batch_size = get_clustering_defaults()['embeddings_batch_size']
                num_embeddings_batches = (len(entity_discriptions) + embeddings_batch_size - 1) // embeddings_batch_size
                for i in range(num_embeddings_batches):
                    start_index = i * embeddings_batch_size
                    end_index = min((i + 1) * embeddings_batch_size, len(entity_discriptions))
                    batch = entity_discriptions[start_index:end_index]
                    entity_vdb: BaseVectorStorage = kwargs["entity_vdb"]
                    result = await entity_vdb.embedding_func(batch)
                    entity_sequence_embeddings.extend(result)
                entity_embeddings = entity_sequence_embeddings
                for (k, v), x in zip(all_entities_relations.items(), entity_embeddings):
                    value = v
                    value["embedding"] = x
                    all_entities_relations[k] = value
                # append the attribute entities of current clustered set to results
                all_entities_relations = [v for k, v in all_entities_relations.items()]
                node_clusters += all_entities_relations
            hierarchical_clusters.append(node_clusters)
            # update nodes to be clustered in the next layer
            nodes = copy.deepcopy([x for x in node_clusters if "entity_name" in x.keys()])
            # filter the duplicate entities
            seen = set()        
            unique_nodes = []
            for item in nodes:
                entity_name = item['entity_name']
                if entity_name not in seen:
                    seen.add(entity_name)
                    unique_nodes.append(item)
            nodes = unique_nodes
            embeddings = np.array([x["embedding"] for x in unique_nodes])
            # stop if the number of deduplicated cluster is too small
            if len(embeddings) <= 2:
                logging.info(f"[Stop Clustering at Layer{layer} with entity num {len(embeddings)}]")
                break
        return hierarchical_clusters