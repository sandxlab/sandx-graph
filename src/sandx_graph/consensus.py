"""ConsensusEngine — neighborhood consensus computation over knowledge graphs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .builder import KnowledgeGraph


@dataclass
class ConsensusScore:
    """Consensus score for a single graph node.

    Attributes:
        node_id:               Target node.
        score:                 Consensus score in [0, 1]. Higher = stronger neighborhood agreement.
        supporting_neighbors:  Direct neighbors with edge weight ≥ 0.5.
        conflicting_neighbors: Direct neighbors with edge weight < 0.5.
        n_neighbors_visited:   Total neighbors visited at the specified depth.
    """

    node_id: str
    score: float
    supporting_neighbors: list[str]
    conflicting_neighbors: list[str]
    n_neighbors_visited: int

    def __repr__(self) -> str:
        return (
            f"ConsensusScore(node={self.node_id!r}, score={self.score:.3f}, "
            f"support={len(self.supporting_neighbors)}, "
            f"conflict={len(self.conflicting_neighbors)})"
        )


class ConsensusEngine:
    """Compute neighborhood consensus for nodes in a KnowledgeGraph.

    Consensus quantifies how strongly a node's neighborhood agrees — measured
    as the weighted mean of edge weights up to a given traversal depth.

    High consensus (→ 1.0): the node is connected to high-confidence neighbors.
    Low consensus (→ 0.0): the node has weak or conflicting edges.

    Isolated nodes (degree 0) return score 1.0 by convention.

    Args:
        graph:             The KnowledgeGraph to reason over.
        support_threshold: Edge weight above which a neighbor is "supporting"
                           (default: 0.5).

    Usage:
        engine = ConsensusEngine(graph)
        score = engine.compute("entity_42", depth=2)
        all_scores = engine.compute_all(depth=1)
    """

    def __init__(self, graph: KnowledgeGraph, support_threshold: float = 0.5) -> None:
        self.graph = graph
        self.support_threshold = support_threshold

    # ------------------------------------------------------------------
    # Single-node computation
    # ------------------------------------------------------------------

    def compute(self, node_id: str, depth: int = 2) -> ConsensusScore:
        """Compute the consensus score for node_id up to the given depth.

        Args:
            node_id: Target node identifier. Must exist in the graph.
            depth:   Neighborhood traversal depth (default: 2).
                     depth=1: direct neighbors only.
                     depth=2: neighbors and their neighbors.

        Returns:
            ConsensusScore with score, supporting/conflicting neighbor lists.

        Raises:
            ValueError: If node_id is not in the graph.
            ValueError: If depth < 1.
        """
        if node_id not in self.graph.nodes:
            raise ValueError(
                f"Node '{node_id}' not found in graph "
                f"(graph has {self.graph.n_nodes} nodes)."
            )
        if depth < 1:
            raise ValueError(f"depth must be ≥ 1, got {depth}.")

        # BFS up to depth; collect all edge weights encountered
        visited: dict[str, int] = {node_id: 0}
        queue: deque[str] = deque([node_id])
        all_weights: list[float] = []

        while queue:
            current = queue.popleft()
            current_depth = visited[current]
            if current_depth >= depth:
                continue
            for neighbor, weight in self.graph.neighbors_weighted(current):
                all_weights.append(weight)
                if neighbor not in visited:
                    visited[neighbor] = current_depth + 1
                    queue.append(neighbor)

        score = float(np.mean(all_weights)) if all_weights else 1.0
        n_visited = len(visited) - 1  # exclude self

        # Direct neighbors only for support/conflict breakdown
        direct = self.graph.neighbors_weighted(node_id)
        supporting = [n for n, w in direct if w >= self.support_threshold]
        conflicting = [n for n, w in direct if w < self.support_threshold]

        return ConsensusScore(
            node_id=node_id,
            score=score,
            supporting_neighbors=supporting,
            conflicting_neighbors=conflicting,
            n_neighbors_visited=n_visited,
        )

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_all(self, depth: int = 2) -> dict[str, ConsensusScore]:
        """Compute consensus scores for every node in the graph.

        Args:
            depth: Neighborhood traversal depth (default: 2).

        Returns:
            Dict mapping node_id → ConsensusScore for all nodes.
        """
        return {nid: self.compute(nid, depth=depth) for nid in self.graph.nodes}

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self, depth: int = 1) -> dict[str, float]:
        """Return summary statistics over all node consensus scores.

        Returns:
            Dict with keys: mean, median, std, min, max.
        """
        if self.graph.n_nodes == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        scores = np.array([
            self.compute(nid, depth=depth).score for nid in self.graph.nodes
        ])
        return {
            "mean":   float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std":    float(np.std(scores)),
            "min":    float(np.min(scores)),
            "max":    float(np.max(scores)),
        }
