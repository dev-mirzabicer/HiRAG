import asyncio
import json
from typing import Dict, List, Optional, Union
from collections import defaultdict
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass

from ..base import BaseGraphStorage, SingleCommunitySchema
from .._utils import logger
from ..prompt import GRAPH_FIELD_SEP

neo4j_lock = asyncio.Lock()


def make_path_idable(path):
    return path.replace(".", "_").replace("/", "__").replace("-", "_")


@dataclass
class Neo4jStorage(BaseGraphStorage):
    def __post_init__(self):
        self.neo4j_url = self.global_config["addon_params"].get("neo4j_url", None)
        self.neo4j_auth = self.global_config["addon_params"].get("neo4j_auth", None)
        self.namespace = (
            f"{make_path_idable(self.global_config['working_dir'])}__{self.namespace}"
        )
        logger.info(f"Using the label {self.namespace} for Neo4j as identifier")
        if self.neo4j_url is None or self.neo4j_auth is None:
            raise ValueError("Missing neo4j_url or neo4j_auth in addon_params")
        self.async_driver = AsyncGraphDatabase.driver(
            self.neo4j_url, auth=self.neo4j_auth
        )

    # ===================================================================
    # PATH FINDING METHODS - New comprehensive implementation
    # ===================================================================

    async def find_shortest_path(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 10,
        use_weights: bool = False,
        weight_property: str = "weight",
        algorithm: str = "cypher_shortest",
    ) -> Optional[Dict[str, List]]:
        """
        Find the shortest path between two entities using various Neo4j algorithms.

        Args:
            start_entity: Source entity name
            end_entity: Target entity name
            max_hops: Maximum number of hops to consider
            use_weights: Whether to use relationship weights
            weight_property: Property name for relationship weights
            algorithm: Algorithm to use ('cypher_shortest', 'gds_dijkstra', 'apoc_dijkstra', 'legacy_shortest')

        Returns:
            Dictionary with 'nodes' and 'edges' lists, or None if no path exists
        """
        if algorithm == "cypher_shortest":
            return await self._find_path_cypher_shortest(
                start_entity, end_entity, max_hops, use_weights, weight_property
            )
        elif algorithm == "gds_dijkstra":
            return await self._find_path_gds_dijkstra(
                start_entity, end_entity, weight_property
            )
        elif algorithm == "apoc_dijkstra":
            return await self._find_path_apoc_dijkstra(
                start_entity, end_entity, weight_property
            )
        elif algorithm == "legacy_shortest":
            return await self._find_path_legacy_shortest(
                start_entity, end_entity, max_hops
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    async def _find_path_cypher_shortest(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int,
        use_weights: bool,
        weight_property: str,
    ) -> Optional[Dict[str, List]]:
        """
        Use modern Cypher SHORTEST syntax (Neo4j 5.x+) for path finding.
        This uses bidirectional BFS and is the most efficient for unweighted paths.
        """
        async with self.async_driver.session() as session:
            try:
                if use_weights:
                    # For weighted paths, we need to use a different approach
                    # as SHORTEST doesn't directly support weights
                    logger.warning(
                        "Cypher SHORTEST doesn't support weights directly. Falling back to legacy approach."
                    )
                    return await self._find_path_legacy_shortest(
                        start_entity, end_entity, max_hops
                    )

                # Modern SHORTEST syntax for unweighted paths
                query = f"""
                MATCH (start:{self.namespace} {{id: $start_id}})
                MATCH (end:{self.namespace} {{id: $end_id}})
                MATCH path = SHORTEST 1 (start)-[:{self.namespace}_RELATED*1..{max_hops}]-(end)
                RETURN 
                    [n IN nodes(path) | properties(n)] AS node_data,
                    [r IN relationships(path) | {{
                        source: startNode(r).id,
                        target: endNode(r).id,
                        properties: properties(r)
                    }}] AS edge_data,
                    length(path) AS path_length
                """

                result = await session.run(
                    query, start_id=start_entity, end_id=end_entity
                )
                record = await result.single()

                if not record:
                    logger.info(
                        f"No path found between {start_entity} and {end_entity}"
                    )
                    return None

                return {
                    "nodes": record["node_data"],
                    "edges": record["edge_data"],
                    "path_length": record["path_length"],
                }

            except Exception as e:
                logger.error(f"Error in Cypher SHORTEST path finding: {e}")
                # Fallback to legacy method
                return await self._find_path_legacy_shortest(
                    start_entity, end_entity, max_hops
                )

    async def _find_path_gds_dijkstra(
        self, start_entity: str, end_entity: str, weight_property: str
    ) -> Optional[Dict[str, List]]:
        """
        Use Neo4j Graph Data Science library for weighted shortest path (Dijkstra's algorithm).
        Requires GDS library to be installed and graph projection to be created.
        """
        async with self.async_driver.session() as session:
            try:
                graph_name = f"hirag_temp_graph_{self.namespace}"

                # Step 1: Create graph projection
                projection_query = f"""
                CALL gds.graph.project(
                    '{graph_name}',
                    '{self.namespace}',
                    '{self.namespace}_RELATED',
                    {{
                        relationshipProperties: ['{weight_property}']
                    }}
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """

                await session.run(projection_query)
                logger.debug(f"Created GDS graph projection: {graph_name}")

                # Step 2: Run Dijkstra algorithm
                dijkstra_query = f"""
                MATCH (source:{self.namespace} {{id: $start_id}})
                MATCH (target:{self.namespace} {{id: $end_id}})
                CALL gds.shortestPath.dijkstra.stream('{graph_name}', {{
                    sourceNode: source,
                    targetNode: target,
                    relationshipWeightProperty: '{weight_property}'
                }})
                YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
                RETURN 
                    [nodeId IN nodeIds | gds.util.asNode(nodeId)] AS path_nodes,
                    nodeIds,
                    costs,
                    totalCost,
                    nodes(path) AS detailed_nodes,
                    relationships(path) AS detailed_edges
                """

                result = await session.run(
                    dijkstra_query, start_id=start_entity, end_id=end_entity
                )
                record = await result.single()

                if record:
                    # Process the path data
                    nodes = []
                    for node in record["detailed_nodes"]:
                        node_dict = dict(node)
                        nodes.append(node_dict)

                    edges = []
                    for edge in record["detailed_edges"]:
                        edge_dict = {
                            "source": edge.start_node["id"],
                            "target": edge.end_node["id"],
                            **dict(edge),
                        }
                        edges.append(edge_dict)

                    path_result = {
                        "nodes": nodes,
                        "edges": edges,
                        "total_cost": record["totalCost"],
                        "costs": record["costs"],
                    }
                else:
                    path_result = None

                # Step 3: Clean up graph projection
                cleanup_query = f"CALL gds.graph.drop('{graph_name}')"
                await session.run(cleanup_query)
                logger.debug(f"Cleaned up GDS graph projection: {graph_name}")

                return path_result

            except Exception as e:
                logger.error(f"Error in GDS Dijkstra path finding: {e}")
                # Try to clean up projection if it exists
                try:
                    cleanup_query = f"CALL gds.graph.drop('{graph_name}')"
                    await session.run(cleanup_query)
                except:
                    pass
                return None

    async def _find_path_apoc_dijkstra(
        self, start_entity: str, end_entity: str, weight_property: str
    ) -> Optional[Dict[str, List]]:
        """
        Use APOC procedures for weighted shortest path.
        This is simpler than GDS as it doesn't require graph projections.
        """
        async with self.async_driver.session() as session:
            try:
                query = f"""
                MATCH (start:{self.namespace} {{id: $start_id}})
                MATCH (end:{self.namespace} {{id: $end_id}})
                CALL apoc.algo.dijkstra(start, end, '{self.namespace}_RELATED', '{weight_property}') 
                YIELD path, weight
                RETURN 
                    [n IN nodes(path) | properties(n)] AS node_data,
                    [r IN relationships(path) | {{
                        source: startNode(r).id,
                        target: endNode(r).id,
                        properties: properties(r)
                    }}] AS edge_data,
                    weight AS total_weight,
                    length(path) AS path_length
                """

                result = await session.run(
                    query, start_id=start_entity, end_id=end_entity
                )
                record = await result.single()

                if record:
                    return {
                        "nodes": record["node_data"],
                        "edges": record["edge_data"],
                        "total_weight": record["total_weight"],
                        "path_length": record["path_length"],
                    }
                else:
                    logger.info(
                        f"No weighted path found between {start_entity} and {end_entity}"
                    )
                    return None

            except Exception as e:
                logger.error(f"Error in APOC Dijkstra path finding: {e}")
                return None

    async def _find_path_legacy_shortest(
        self, start_entity: str, end_entity: str, max_hops: int
    ) -> Optional[Dict[str, List]]:
        """
        Use legacy shortestPath() function for compatibility with older Neo4j versions.
        """
        async with self.async_driver.session() as session:
            try:
                query = f"""
                MATCH (start:{self.namespace} {{id: $start_id}})
                MATCH (end:{self.namespace} {{id: $end_id}})
                MATCH path = shortestPath((start)-[:{self.namespace}_RELATED*1..{max_hops}]-(end))
                RETURN 
                    [n IN nodes(path) | properties(n)] AS node_data,
                    [r IN relationships(path) | {{
                        source: startNode(r).id,
                        target: endNode(r).id,
                        properties: properties(r)
                    }}] AS edge_data,
                    length(path) AS path_length
                """

                result = await session.run(
                    query, start_id=start_entity, end_id=end_entity
                )
                record = await result.single()

                if record:
                    return {
                        "nodes": record["node_data"],
                        "edges": record["edge_data"],
                        "path_length": record["path_length"],
                    }
                else:
                    logger.info(
                        f"No path found between {start_entity} and {end_entity}"
                    )
                    return None

            except Exception as e:
                logger.error(f"Error in legacy shortest path finding: {e}")
                return None

    async def find_all_shortest_paths(
        self, start_entity: str, end_entity: str, max_hops: int = 10, limit: int = 10
    ) -> List[Dict[str, List]]:
        """
        Find all shortest paths between two entities.

        Args:
            start_entity: Source entity name
            end_entity: Target entity name
            max_hops: Maximum number of hops
            limit: Maximum number of paths to return

        Returns:
            List of path dictionaries
        """
        async with self.async_driver.session() as session:
            try:
                # Try modern SHORTEST syntax first
                query = f"""
                MATCH (start:{self.namespace} {{id: $start_id}})
                MATCH (end:{self.namespace} {{id: $end_id}})
                MATCH path = SHORTEST {limit} (start)-[:{self.namespace}_RELATED*1..{max_hops}]-(end)
                RETURN 
                    [n IN nodes(path) | properties(n)] AS node_data,
                    [r IN relationships(path) | {{
                        source: startNode(r).id,
                        target: endNode(r).id,
                        properties: properties(r)
                    }}] AS edge_data,
                    length(path) AS path_length
                ORDER BY path_length
                """

                result = await session.run(
                    query, start_id=start_entity, end_id=end_entity
                )
                paths = []

                async for record in result:
                    paths.append(
                        {
                            "nodes": record["node_data"],
                            "edges": record["edge_data"],
                            "path_length": record["path_length"],
                        }
                    )

                if paths:
                    return paths

                # Fallback to legacy allShortestPaths
                legacy_query = f"""
                MATCH (start:{self.namespace} {{id: $start_id}})
                MATCH (end:{self.namespace} {{id: $end_id}})
                MATCH path = allShortestPaths((start)-[:{self.namespace}_RELATED*1..{max_hops}]-(end))
                RETURN 
                    [n IN nodes(path) | properties(n)] AS node_data,
                    [r IN relationships(path) | {{
                        source: startNode(r).id,
                        target: endNode(r).id,
                        properties: properties(r)
                    }}] AS edge_data,
                    length(path) AS path_length
                ORDER BY path_length
                LIMIT {limit}
                """

                result = await session.run(
                    legacy_query, start_id=start_entity, end_id=end_entity
                )
                async for record in result:
                    paths.append(
                        {
                            "nodes": record["node_data"],
                            "edges": record["edge_data"],
                            "path_length": record["path_length"],
                        }
                    )

                return paths

            except Exception as e:
                logger.error(f"Error finding all shortest paths: {e}")
                return []

    async def find_k_shortest_paths(
        self,
        start_entity: str,
        end_entity: str,
        k: int = 3,
        weight_property: str = "weight",
    ) -> List[Dict[str, List]]:
        """
        Find K shortest paths using Yen's algorithm (requires GDS library).

        Args:
            start_entity: Source entity name
            end_entity: Target entity name
            k: Number of shortest paths to find
            weight_property: Property name for relationship weights

        Returns:
            List of k shortest paths
        """
        async with self.async_driver.session() as session:
            try:
                graph_name = f"hirag_yens_graph_{self.namespace}"

                # Create graph projection for Yen's algorithm
                projection_query = f"""
                CALL gds.graph.project(
                    '{graph_name}',
                    '{self.namespace}',
                    '{self.namespace}_RELATED',
                    {{
                        relationshipProperties: ['{weight_property}']
                    }}
                )
                """

                await session.run(projection_query)

                # Run Yen's K shortest paths algorithm
                yens_query = f"""
                MATCH (source:{self.namespace} {{id: $start_id}})
                MATCH (target:{self.namespace} {{id: $end_id}})
                CALL gds.shortestPath.yens.stream('{graph_name}', {{
                    sourceNode: source,
                    targetNode: target,
                    k: {k},
                    relationshipWeightProperty: '{weight_property}'
                }})
                YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
                RETURN 
                    index,
                    [nodeId IN nodeIds | gds.util.asNode(nodeId)] AS path_nodes,
                    totalCost,
                    costs,
                    nodes(path) AS detailed_nodes,
                    relationships(path) AS detailed_edges
                ORDER BY index
                """

                result = await session.run(
                    yens_query, start_id=start_entity, end_id=end_entity
                )
                paths = []

                async for record in result:
                    nodes = [dict(node) for node in record["detailed_nodes"]]
                    edges = []
                    for edge in record["detailed_edges"]:
                        edge_dict = {
                            "source": edge.start_node["id"],
                            "target": edge.end_node["id"],
                            **dict(edge),
                        }
                        edges.append(edge_dict)

                    paths.append(
                        {
                            "index": record["index"],
                            "nodes": nodes,
                            "edges": edges,
                            "total_cost": record["totalCost"],
                            "costs": record["costs"],
                        }
                    )

                # Clean up projection
                cleanup_query = f"CALL gds.graph.drop('{graph_name}')"
                await session.run(cleanup_query)

                return paths

            except Exception as e:
                logger.error(f"Error in Yen's K shortest paths: {e}")
                # Clean up on error
                try:
                    cleanup_query = f"CALL gds.graph.drop('{graph_name}')"
                    await session.run(cleanup_query)
                except:
                    pass
                return []

    # ===================================================================
    # EXISTING METHODS (keeping the same interface)
    # ===================================================================

    async def _init_workspace(self):
        await self.async_driver.verify_authentication()
        await self.async_driver.verify_connectivity()

    async def index_start_callback(self):
        logger.info("Init Neo4j workspace")
        await self._init_workspace()

    async def has_node(self, node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",
                node_id=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN COUNT(r) > 0 AS exists",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def node_degree(self, node_id: str) -> int:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id "
                f"RETURN size((n)-[]-(:{self.namespace})) AS degree",
                node_id=node_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $src_id AND t.id = $tgt_id "
                f"RETURN size((s)-[]-(:{self.namespace})) + size((t)-[]-(:{self.namespace})) AS degree",
                src_id=src_id,
                tgt_id=tgt_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def get_node(self, node_id: str) -> Union[dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN properties(n) AS node_data",
                node_id=node_id,
            )
            record = await result.single()
            raw_node_data = record["node_data"] if record else None

        if raw_node_data is None:
            return None

        raw_node_data["clusters"] = json.dumps(
            [
                {
                    "level": index,
                    "cluster": cluster_id,
                }
                for index, cluster_id in enumerate(
                    raw_node_data.get("communityIds", [])
                )
            ]
        )
        return raw_node_data

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN properties(r) AS edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["edge_data"] if record else None

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) WHERE s.id = $source_id "
                "RETURN s.id AS source, t.id AS target",
                source_id=source_node_id,
            )
            edges = []
            async for record in result:
                edges.append((record["source"], record["target"]))
            return edges

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        entity_type = node_data.get("entity_type", "UNKNOWN").strip('"')
        async with self.async_driver.session() as session:
            await session.run(
                f"MERGE (n:{self.namespace}:{entity_type} {{id: $node_id}}) "
                "SET n += $node_data",
                node_id=node_id,
                node_data=node_data,
            )

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        edge_data.setdefault("weight", 0.0)
        async with self.async_driver.session() as session:
            await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                f"MERGE (s)-[r:{self.namespace}_RELATED]->(t) "
                "SET r += $edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
                edge_data=edge_data,
            )

    async def clustering(self, algorithm: str):
        if algorithm != "leiden":
            raise ValueError(
                f"Clustering algorithm {algorithm} not supported in Neo4j implementation"
            )

        random_seed = self.global_config["graph_cluster_seed"]
        max_level = self.global_config["max_graph_cluster_size"]
        async with self.async_driver.session() as session:
            try:
                # Project the graph with undirected relationships
                await session.run(
                    f"""
                    CALL gds.graph.project(
                        'graph_{self.namespace}',
                        ['{self.namespace}'],
                        {{
                            {self.namespace}_RELATED: {{
                                orientation: 'UNDIRECTED',
                                properties: ['weight']
                            }}
                        }}
                    )
                    """
                )

                # Run Leiden algorithm
                result = await session.run(
                    f"""
                    CALL gds.leiden.write(
                        'graph_{self.namespace}',
                        {{
                            writeProperty: 'communityIds',
                            includeIntermediateCommunities: true,
                            relationshipWeightProperty: "weight",
                            maxLevels: {max_level},
                            tolerance: 0.0001,
                            gamma: 1.0,
                            theta: 0.01,
                            randomSeed: {random_seed}
                        }}
                    )
                    YIELD communityCount, modularities;
                    """
                )
                result = await result.single()
                community_count: int = result["communityCount"]
                modularities = result["modularities"]
                logger.info(
                    f"Performed graph clustering with {community_count} communities and modularities {modularities}"
                )
            finally:
                # Drop the projected graph
                await session.run(f"CALL gds.graph.drop('graph_{self.namespace}')")

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )

        async with self.async_driver.session() as session:
            result = await session.run(
                f"""
                MATCH (n:{self.namespace})
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:{self.namespace}) | m.id] AS connected_nodes
                RETURN n.id AS node_id, n.source_id AS source_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            max_num_ids = 0
            async for record in result:
                for index, c_id in enumerate(record["cluster_key"]):
                    node_id = str(record["node_id"])
                    source_id = record["source_id"]
                    level = index
                    cluster_key = str(c_id)
                    connected_nodes = record["connected_nodes"]

                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [
                            tuple(sorted([node_id, str(connected)]))
                            for connected in connected_nodes
                            if connected != node_id
                        ]
                    )
                    chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                    results[cluster_key]["chunk_ids"].update(chunk_ids)
                    max_num_ids = max(
                        max_num_ids, len(results[cluster_key]["chunk_ids"])
                    )

            # Process results
            for k, v in results.items():
                v["edges"] = [list(e) for e in v["edges"]]
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
                v["occurrence"] = len(v["chunk_ids"]) / max_num_ids

            # Compute sub-communities
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    async def index_done_callback(self):
        await self.async_driver.close()

    async def _debug_delete_all_node_edges(self):
        async with self.async_driver.session() as session:
            try:
                # Delete all relationships in the namespace
                await session.run(f"MATCH (n:{self.namespace})-[r]-() DELETE r")

                # Delete all nodes in the namespace
                await session.run(f"MATCH (n:{self.namespace}) DELETE n")

                logger.info(
                    f"All nodes and edges in namespace '{self.namespace}' have been deleted."
                )
            except Exception as e:
                logger.error(f"Error deleting nodes and edges: {str(e)}")
                raise
