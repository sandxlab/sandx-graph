"""GraphBuilder — construct knowledge graphs from resolved entity data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


@dataclass
class KnowledgeGraph:
    """An undirected weighted knowledge graph over resolved entities.

    Nodes are entity identifiers (strings). Edges carry a float weight in [0, 1]
    representing confidence or strength of relationship.

    Args:
        nodes: Dict mapping node_id → attribute dict. The attribute dict may be
               empty or contain any key-value metadata (record_ids, confidence, etc.).
        edges: List of (source_id, target_id, weight) triples.
    """

    nodes: dict[str, dict]
    edges: list[tuple[str, str, float]]
    n_nodes: int = field(init=False)
    n_edges: int = field(init=False)
    _adj: dict[str, list[tuple[str, float]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        self._build_adj()

    def _build_adj(self) -> None:
        self._adj = {nid: [] for nid in self.nodes}
        for src, tgt, w in self.edges:
            self._adj.setdefault(src, []).append((tgt, w))
            self._adj.setdefault(tgt, []).append((src, w))

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str) -> list[str]:
        """Return IDs of all nodes adjacent to node_id."""
        return [nid for nid, _ in self._adj.get(node_id, [])]

    def neighbors_weighted(self, node_id: str) -> list[tuple[str, float]]:
        """Return (neighbor_id, weight) pairs for all adjacent nodes."""
        return list(self._adj.get(node_id, []))

    def degree(self, node_id: str) -> int:
        """Number of edges incident to node_id."""
        return len(self._adj.get(node_id, []))

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def has_edge(self, a: str, b: str) -> bool:
        return any(nid == b for nid, _ in self._adj.get(a, []))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return the edge list as a DataFrame with columns: source, target, weight."""
        if not self.edges:
            return pd.DataFrame(columns=["source", "target", "weight"])
        return pd.DataFrame(self.edges, columns=["source", "target", "weight"])

    def to_networkx(self):
        """Export to a NetworkX Graph. Requires: pip install networkx."""
        try:
            import networkx as nx  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "to_networkx() requires networkx. Install with: pip install networkx"
            ) from exc

        G = nx.Graph()
        for nid, attrs in self.nodes.items():
            G.add_node(nid, **attrs)
        for src, tgt, w in self.edges:
            G.add_edge(src, tgt, weight=w)
        return G

    def __repr__(self) -> str:
        return f"KnowledgeGraph(n_nodes={self.n_nodes}, n_edges={self.n_edges})"


class GraphBuilder:
    """Construct KnowledgeGraph objects from various input formats.

    Usage:
        # From sandx-er resolution output
        graph = GraphBuilder().from_clusters(result.clusters)

        # From explicit node/edge DataFrames
        graph = GraphBuilder().from_dataframe(nodes_df, edges_df)
    """

    def from_resolution(
        self,
        result,
        edges: list[tuple[str, str, float]] | pd.DataFrame | None = None,
        *,
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str = "weight",
        include_singletons: bool = False,
    ) -> KnowledgeGraph:
        """Build a graph from a sandx-er ResolutionResult.

        Duck-typed: ``result`` only needs a ``.clusters`` attribute where each
        cluster has ``.canonical_id``, ``.record_ids``, ``.confidence``, ``.size``.
        No hard dependency on sandx-er is introduced.

        Args:
            result:             ResolutionResult from EntityResolver.resolve().
            edges:              Relationship data between resolved entities.
                                - list of (src_canonical_id, tgt_canonical_id, weight), or
                                - DataFrame with source/target columns and optional weight.
                                If None, graph has nodes only.
            source_col:         Source column name when edges is a DataFrame.
            target_col:         Target column name when edges is a DataFrame.
            weight_col:         Weight column name when edges is a DataFrame.
                                If column absent, defaults to 1.0.
            include_singletons: Include single-record clusters (default: False).

        Returns:
            KnowledgeGraph with one node per qualifying resolved cluster.
        """
        nodes: dict[str, dict] = {}
        for cluster in result.clusters:
            if not include_singletons and cluster.size == 1:
                continue
            nodes[cluster.canonical_id] = {
                "record_ids": list(cluster.record_ids),
                "confidence": float(cluster.confidence),
                "size": int(cluster.size),
            }

        edge_list: list[tuple[str, str, float]] = []
        if edges is not None:
            if isinstance(edges, pd.DataFrame):
                for required in (source_col, target_col):
                    if required not in edges.columns:
                        raise ValueError(
                            f"Column '{required}' not found in edges DataFrame. "
                            f"Available: {list(edges.columns)}"
                        )
                for _, row in edges.iterrows():
                    src = str(row[source_col])
                    tgt = str(row[target_col])
                    w = float(row[weight_col]) if weight_col in edges.columns else 1.0
                    nodes.setdefault(src, {})
                    nodes.setdefault(tgt, {})
                    edge_list.append((src, tgt, w))
            else:
                for src, tgt, w in edges:
                    nodes.setdefault(str(src), {})
                    nodes.setdefault(str(tgt), {})
                    edge_list.append((str(src), str(tgt), float(w)))

        return KnowledgeGraph(nodes=nodes, edges=edge_list)

    def from_clusters(self, clusters: list) -> KnowledgeGraph:
        """Build a graph where each resolved entity cluster becomes a node.

        Args:
            clusters: List of EntityCluster objects from sandx-er.
                      Each cluster becomes a node; edges are not added
                      (use from_dataframe or add relationship data separately).

        Returns:
            KnowledgeGraph with one node per resolved entity, no edges.
        """
        nodes: dict[str, dict] = {}
        for cluster in clusters:
            nodes[cluster.canonical_id] = {
                "record_ids": list(cluster.record_ids),
                "confidence": float(cluster.confidence),
                "size": int(cluster.size),
            }
        return KnowledgeGraph(nodes=nodes, edges=[])

    def from_dataframe(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame | None = None,
        *,
        node_id_col: str = "node_id",
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str = "weight",
    ) -> KnowledgeGraph:
        """Build a graph from node and (optionally) edge DataFrames.

        Args:
            nodes_df:     DataFrame with a node ID column and optional attribute columns.
            edges_df:     DataFrame with source/target columns and optional weight column.
                          If None, graph has nodes only.
            node_id_col:  Name of the node ID column in nodes_df (default: "node_id").
            source_col:   Name of the source column in edges_df (default: "source").
            target_col:   Name of the target column in edges_df (default: "target").
            weight_col:   Name of the weight column in edges_df (default: "weight").
                          If column is absent, all edge weights default to 1.0.

        Returns:
            KnowledgeGraph with nodes and edges as specified.
        """
        if node_id_col not in nodes_df.columns:
            raise ValueError(
                f"Column '{node_id_col}' not found in nodes_df. "
                f"Available: {list(nodes_df.columns)}"
            )

        nodes: dict[str, dict] = {}
        for _, row in nodes_df.iterrows():
            nid = str(row[node_id_col])
            attrs = {k: v for k, v in row.items() if k != node_id_col}
            nodes[nid] = attrs

        edges: list[tuple[str, str, float]] = []
        if edges_df is not None:
            for required in (source_col, target_col):
                if required not in edges_df.columns:
                    raise ValueError(
                        f"Column '{required}' not found in edges_df. "
                        f"Available: {list(edges_df.columns)}"
                    )
            for _, row in edges_df.iterrows():
                src = str(row[source_col])
                tgt = str(row[target_col])
                w = float(row[weight_col]) if weight_col in edges_df.columns else 1.0
                # Add missing nodes referenced in edges
                nodes.setdefault(src, {})
                nodes.setdefault(tgt, {})
                edges.append((src, tgt, w))

        return KnowledgeGraph(nodes=nodes, edges=edges)

    def from_similarity_matrix(
        self,
        ids: list[str],
        similarity: np.ndarray,
        threshold: float = 0.5,
    ) -> KnowledgeGraph:
        """Build a graph from a pairwise similarity matrix.

        Args:
            ids:        Node IDs corresponding to rows/cols of similarity.
            similarity: Square float matrix of shape (N, N) with values in [0, 1].
            threshold:  Minimum similarity to create an edge (default: 0.5).

        Returns:
            KnowledgeGraph with an edge for every pair above threshold.
        """
        n = len(ids)
        if similarity.shape != (n, n):
            raise ValueError(
                f"similarity shape {similarity.shape} must be ({n}, {n})"
            )

        nodes = {nid: {} for nid in ids}
        edges: list[tuple[str, str, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                w = float(similarity[i, j])
                if w >= threshold:
                    edges.append((ids[i], ids[j], w))

        return KnowledgeGraph(nodes=nodes, edges=edges)
