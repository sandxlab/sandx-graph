"""Tests for GraphBuilder and KnowledgeGraph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sandx_graph.builder import GraphBuilder, KnowledgeGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph() -> KnowledgeGraph:
    nodes = {"a": {}, "b": {}, "c": {}, "d": {}}
    edges = [("a", "b", 0.9), ("b", "c", 0.6), ("c", "d", 0.3)]
    return KnowledgeGraph(nodes=nodes, edges=edges)


@pytest.fixture
def builder() -> GraphBuilder:
    return GraphBuilder()


# ---------------------------------------------------------------------------
# KnowledgeGraph — basic attributes
# ---------------------------------------------------------------------------

def test_n_nodes_n_edges(simple_graph):
    assert simple_graph.n_nodes == 4
    assert simple_graph.n_edges == 3


def test_has_node(simple_graph):
    assert simple_graph.has_node("a")
    assert not simple_graph.has_node("z")


def test_has_edge(simple_graph):
    assert simple_graph.has_edge("a", "b")
    assert simple_graph.has_edge("b", "a")  # undirected
    assert not simple_graph.has_edge("a", "c")


def test_degree(simple_graph):
    assert simple_graph.degree("b") == 2
    assert simple_graph.degree("a") == 1
    assert simple_graph.degree("d") == 1


def test_neighbors(simple_graph):
    assert set(simple_graph.neighbors("b")) == {"a", "c"}


def test_neighbors_weighted(simple_graph):
    result = dict(simple_graph.neighbors_weighted("a"))
    assert result == {"b": 0.9}


def test_isolated_node():
    g = KnowledgeGraph(nodes={"x": {}}, edges=[])
    assert g.neighbors("x") == []
    assert g.degree("x") == 0


def test_repr(simple_graph):
    r = repr(simple_graph)
    assert "n_nodes=4" in r
    assert "n_edges=3" in r


# ---------------------------------------------------------------------------
# KnowledgeGraph — export
# ---------------------------------------------------------------------------

def test_to_dataframe(simple_graph):
    df = simple_graph.to_dataframe()
    assert list(df.columns) == ["source", "target", "weight"]
    assert len(df) == 3


def test_to_dataframe_empty():
    g = KnowledgeGraph(nodes={"a": {}}, edges=[])
    df = g.to_dataframe()
    assert list(df.columns) == ["source", "target", "weight"]
    assert len(df) == 0


@pytest.mark.networkx
def test_to_networkx(simple_graph):
    pytest.importorskip("networkx")
    G = simple_graph.to_networkx()
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3
    assert G["a"]["b"]["weight"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# GraphBuilder.from_dataframe
# ---------------------------------------------------------------------------

def test_from_dataframe_nodes_only(builder):
    nodes_df = pd.DataFrame({"node_id": ["x", "y", "z"]})
    g = builder.from_dataframe(nodes_df)
    assert g.n_nodes == 3
    assert g.n_edges == 0


def test_from_dataframe_with_edges(builder):
    nodes_df = pd.DataFrame({"node_id": ["a", "b", "c"]})
    edges_df = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"], "weight": [0.8, 0.5]})
    g = builder.from_dataframe(nodes_df, edges_df)
    assert g.n_nodes == 3
    assert g.n_edges == 2
    assert dict(g.neighbors_weighted("a"))["b"] == pytest.approx(0.8)


def test_from_dataframe_default_weight(builder):
    nodes_df = pd.DataFrame({"node_id": ["a", "b"]})
    edges_df = pd.DataFrame({"source": ["a"], "target": ["b"]})  # no weight col
    g = builder.from_dataframe(nodes_df, edges_df)
    assert g.n_edges == 1
    assert dict(g.neighbors_weighted("a"))["b"] == pytest.approx(1.0)


def test_from_dataframe_missing_node_col(builder):
    nodes_df = pd.DataFrame({"id": ["a"]})
    with pytest.raises(ValueError, match="node_id"):
        builder.from_dataframe(nodes_df)


def test_from_dataframe_missing_edge_col(builder):
    nodes_df = pd.DataFrame({"node_id": ["a", "b"]})
    edges_df = pd.DataFrame({"src": ["a"], "target": ["b"]})
    with pytest.raises(ValueError, match="source"):
        builder.from_dataframe(nodes_df, edges_df)


def test_from_dataframe_implicit_nodes(builder):
    """Nodes referenced in edges but absent from nodes_df are added automatically."""
    nodes_df = pd.DataFrame({"node_id": ["a"]})
    edges_df = pd.DataFrame({"source": ["a"], "target": ["b"], "weight": [0.7]})
    g = builder.from_dataframe(nodes_df, edges_df)
    assert g.has_node("b")


def test_from_dataframe_custom_columns(builder):
    nodes_df = pd.DataFrame({"nid": ["a", "b"]})
    edges_df = pd.DataFrame({"src": ["a"], "tgt": ["b"], "w": [0.6]})
    g = builder.from_dataframe(
        nodes_df, edges_df,
        node_id_col="nid", source_col="src", target_col="tgt", weight_col="w",
    )
    assert g.n_edges == 1


# ---------------------------------------------------------------------------
# GraphBuilder.from_similarity_matrix
# ---------------------------------------------------------------------------

def test_from_similarity_matrix(builder):
    ids = ["a", "b", "c"]
    sim = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.2], [0.3, 0.2, 1.0]])
    g = builder.from_similarity_matrix(ids, sim, threshold=0.5)
    assert g.n_nodes == 3
    assert g.n_edges == 1  # only a-b passes threshold
    assert g.has_edge("a", "b")
    assert not g.has_edge("a", "c")


def test_from_similarity_matrix_shape_mismatch(builder):
    ids = ["a", "b"]
    sim = np.ones((3, 3))
    with pytest.raises(ValueError, match="shape"):
        builder.from_similarity_matrix(ids, sim, threshold=0.5)


def test_from_similarity_matrix_all_above(builder):
    ids = ["a", "b", "c"]
    sim = np.ones((3, 3))
    g = builder.from_similarity_matrix(ids, sim, threshold=0.0)
    assert g.n_edges == 3  # upper triangle: (0,1), (0,2), (1,2)


def test_from_similarity_matrix_none_above(builder):
    ids = ["a", "b"]
    sim = np.array([[1.0, 0.3], [0.3, 1.0]])
    g = builder.from_similarity_matrix(ids, sim, threshold=0.9)
    assert g.n_edges == 0


# ---------------------------------------------------------------------------
# GraphBuilder.from_clusters (duck-typing; no sandx-er import required)
# ---------------------------------------------------------------------------

class _FakeCluster:
    def __init__(self, cid, record_ids, confidence, size):
        self.canonical_id = cid
        self.record_ids = set(record_ids)
        self.confidence = confidence
        self.size = size


def test_from_clusters(builder):
    clusters = [
        _FakeCluster("e1", ["r0", "r1"], 0.9, 2),
        _FakeCluster("e2", ["r2"], 0.7, 1),
    ]
    g = builder.from_clusters(clusters)
    assert g.n_nodes == 2
    assert g.n_edges == 0
    assert g.nodes["e1"]["confidence"] == pytest.approx(0.9)
    assert g.nodes["e1"]["size"] == 2


def test_from_clusters_empty(builder):
    g = builder.from_clusters([])
    assert g.n_nodes == 0
    assert g.n_edges == 0
