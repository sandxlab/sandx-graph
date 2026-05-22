"""GraphBuilder — construct knowledge graphs from resolved entity clusters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KnowledgeGraph:
    nodes: dict[str, dict]  # node_id → attributes
    edges: list[tuple[str, str, dict]]  # (source, target, attributes)
    n_nodes: int = field(init=False)
    n_edges: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)


class GraphBuilder:
    """Construct a KnowledgeGraph from sandx-er resolution output.

    Usage:
        builder = GraphBuilder()
        graph = builder.from_clusters(result.clusters)
    """

    def from_clusters(self, clusters: list) -> KnowledgeGraph:
        """Build a graph where each cluster becomes a canonical node.

        Args:
            clusters: List of EntityCluster objects from sandx-er.

        Returns:
            KnowledgeGraph with one node per resolved entity.
        """
        raise NotImplementedError("Phase 2")

    def from_dataframe(self, nodes_df, edges_df) -> KnowledgeGraph:
        """Build a graph directly from node and edge DataFrames."""
        raise NotImplementedError("Phase 2")
