import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast, Optional
import networkx as nx
import numpy as np

from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> Optional[nx.Graph]:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    @property
    def graph(self) -> nx.Graph:
        """Access to the underlying NetworkX graph object."""
        return self._graph

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        if self._graph.has_node(node_id):
            degree = self._graph.degree(node_id)
            return int(degree) if isinstance(degree, int) else 0
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = 0
        if self._graph.has_node(src_id):
            degree = self._graph.degree(src_id)
            src_degree = int(degree) if isinstance(degree, int) else 0
        
        tgt_degree = 0
        if self._graph.has_node(tgt_id):
            degree = self._graph.degree(tgt_id)
            tgt_degree = int(degree) if isinstance(degree, int) else 0
        
        return src_degree + tgt_degree

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        # Initialize with proper types
        results: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "level": 0,
                "title": "",
                "edges": set(),
                "nodes": set(),
                "chunk_ids": set(),
                "occurrence": 0.0,
                "sub_communities": [],
            }
        )
        max_num_ids = 0
        levels: dict[int, set[str]] = defaultdict(set)
        
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                
                # Ensure nodes is a set and add node
                nodes_set = results[cluster_key]["nodes"]
                if not isinstance(nodes_set, set):
                    nodes_set = set()
                    results[cluster_key]["nodes"] = nodes_set
                nodes_set.add(node_id)
                
                # Ensure edges is a set and update edges
                edges_set = results[cluster_key]["edges"]
                if not isinstance(edges_set, set):
                    edges_set = set()
                    results[cluster_key]["edges"] = edges_set
                edges_set.update([tuple(sorted(e)) for e in this_node_edges])
                
                # Ensure chunk_ids is a set and update chunk_ids
                chunk_ids_set = results[cluster_key]["chunk_ids"]
                if not isinstance(chunk_ids_set, set):
                    chunk_ids_set = set()
                    results[cluster_key]["chunk_ids"] = chunk_ids_set
                
                # Handle source_id safely
                source_id = node_data.get("source_id", "")
                if isinstance(source_id, str) and source_id:
                    chunk_ids_set.update(source_id.split(GRAPH_FIELD_SEP))
                
                max_num_ids = max(max_num_ids, len(chunk_ids_set))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                sub_communities = []
                for c in next_level_comms:
                    c_nodes = results[c]["nodes"]
                    comm_nodes = results[comm]["nodes"]
                    if isinstance(c_nodes, set) and isinstance(comm_nodes, set):
                        if c_nodes.issubset(comm_nodes):
                            sub_communities.append(c)
                results[comm]["sub_communities"] = sub_communities

        # Convert to final format
        final_results: dict[str, SingleCommunitySchema] = {}
        for k, v in results.items():
            # Convert edges to list of tuples (str, str)
            edges_set = v["edges"]
            if isinstance(edges_set, set):
                edges_list: list[tuple[str, str]] = []
                for edge in edges_set:
                    if isinstance(edge, tuple) and len(edge) == 2:
                        edges_list.append((str(edge[0]), str(edge[1])))
            else:
                edges_list = []
            
            # Convert chunk_ids to proper format
            chunk_ids = v["chunk_ids"]
            chunk_ids_list = list(chunk_ids) if isinstance(chunk_ids, set) else []
            
            final_results[k] = {
                "level": int(v["level"]),
                "title": str(v["title"]),
                "edges": edges_list,
                "nodes": list(v["nodes"]) if isinstance(v["nodes"], set) else [],
                "chunk_ids": chunk_ids_list,
                "occurrence": float(len(chunk_ids_list) / max_num_ids if max_num_ids > 0 else 0.0),
                "sub_communities": list(v["sub_communities"]) if isinstance(v["sub_communities"], list) else [],
            }
        return final_results

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, Any]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden
        """
        It uses the hierarchical_leiden function from the graspologic library
        The Leiden algorithm is used in the HiRAG.ainsert method
        """
        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, Any]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
