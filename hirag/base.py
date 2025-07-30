from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, Any
from ._utils import EmbeddingFunc
import numpy as np


@dataclass
class QueryParam:
    mode: Literal["hi_global", "hi_local", "hi_bridge", "hi_nobridge", "naive", "hi"] = "hi"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20         # retrieve top-k entities
    top_m: int = 10         # retrieve top-m entities in each retrieved community
    # naive search
    naive_max_token_for_text_unit = 10000
    # hi search
    max_token_for_text_unit: int = 20000
    max_token_for_local_context: int = 20000
    max_token_for_bridge_knowledge: int = 12500
    max_token_for_community_report: int = 12500
    community_single_one: bool = False


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,
        "title": str,
        "edges": list[tuple[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)


class CommunitySchema(SingleCommunitySchema):
    report_string: str
    report_json: dict


T = TypeVar("T")


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError

    async def get_all(self) -> dict[str, T]:
        raise NotImplementedError

    async def delete(self, id: str):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ):
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Return the community representation with report and nodes"""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in HiRAG.")

    async def get_node_count(self) -> int:
        """Return the total number of nodes in the graph."""
        raise NotImplementedError

    async def get_all_nodes(self) -> list[str]:
        """Return all node IDs in the graph."""
        raise NotImplementedError

    async def get_edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        raise NotImplementedError

    async def get_all_edges(self) -> list[tuple[str, str]]:
        """Return all edges in the graph as (source, target) tuples."""
        raise NotImplementedError

    async def find_shortest_path(self, start_entity: str, end_entity: str, **kwargs) -> Union[dict, None]:
        """Find shortest path between two entities."""
        raise NotImplementedError

    async def find_all_shortest_paths(self, start_entity: str, end_entity: str, **kwargs) -> list[dict]:
        """Find all shortest paths between two entities."""
        raise NotImplementedError

    async def find_k_shortest_paths(self, start_entity: str, end_entity: str, k: int, **kwargs) -> list[dict]:
        """Find k shortest paths between two entities."""
        raise NotImplementedError

    @property
    def graph(self) -> Any:
        """Access to the underlying graph object."""
        raise NotImplementedError
