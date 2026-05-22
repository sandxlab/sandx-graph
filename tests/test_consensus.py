"""Tests for ConsensusEngine."""

from __future__ import annotations

import pytest

from sandx_graph.builder import KnowledgeGraph
from sandx_graph.consensus import ConsensusEngine, ConsensusScore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_graph() -> KnowledgeGraph:
    """a -0.9- b -0.6- c -0.3- d"""
    nodes = {"a": {}, "b": {}, "c": {}, "d": {}}
    edges = [("a", "b", 0.9), ("b", "c", 0.6), ("c", "d", 0.3)]
    return KnowledgeGraph(nodes=nodes, edges=edges)


@pytest.fixture
def high_consensus_graph() -> KnowledgeGraph:
    """All edges have weight 1.0."""
    nodes = {"a": {}, "b": {}, "c": {}}
    edges = [("a", "b", 1.0), ("a", "c", 1.0), ("b", "c", 1.0)]
    return KnowledgeGraph(nodes=nodes, edges=edges)


@pytest.fixture
def low_consensus_graph() -> KnowledgeGraph:
    """All edges have weight near 0."""
    nodes = {"a": {}, "b": {}, "c": {}}
    edges = [("a", "b", 0.1), ("a", "c", 0.2)]
    return KnowledgeGraph(nodes=nodes, edges=edges)


@pytest.fixture
def engine(linear_graph) -> ConsensusEngine:
    return ConsensusEngine(linear_graph)


# ---------------------------------------------------------------------------
# ConsensusScore repr
# ---------------------------------------------------------------------------

def test_repr():
    s = ConsensusScore(
        node_id="x", score=0.75,
        supporting_neighbors=["a", "b"],
        conflicting_neighbors=["c"],
        n_neighbors_visited=3,
    )
    r = repr(s)
    assert "x" in r
    assert "0.750" in r


# ---------------------------------------------------------------------------
# Isolated node → score 1.0
# ---------------------------------------------------------------------------

def test_isolated_node():
    g = KnowledgeGraph(nodes={"solo": {}}, edges=[])
    engine = ConsensusEngine(g)
    cs = engine.compute("solo", depth=1)
    assert cs.score == pytest.approx(1.0)
    assert cs.n_neighbors_visited == 0
    assert cs.supporting_neighbors == []
    assert cs.conflicting_neighbors == []


# ---------------------------------------------------------------------------
# Basic compute — depth 1
# ---------------------------------------------------------------------------

def test_compute_depth1_endpoint(engine):
    """Node 'a' has one direct neighbor 'b' with weight 0.9."""
    cs = engine.compute("a", depth=1)
    assert cs.node_id == "a"
    assert cs.score == pytest.approx(0.9)
    assert cs.n_neighbors_visited == 1
    assert "b" in cs.supporting_neighbors
    assert cs.conflicting_neighbors == []


def test_compute_depth1_middle(engine):
    """Node 'b' has neighbors a (0.9) and c (0.6); both ≥ 0.5."""
    cs = engine.compute("b", depth=1)
    assert cs.score == pytest.approx(0.75)  # mean(0.9, 0.6)
    assert set(cs.supporting_neighbors) == {"a", "c"}
    assert cs.conflicting_neighbors == []


def test_compute_depth1_low_weight(engine):
    """Node 'd' has one neighbor c with weight 0.3 < 0.5."""
    cs = engine.compute("d", depth=1)
    assert cs.score == pytest.approx(0.3)
    assert cs.supporting_neighbors == []
    assert "c" in cs.conflicting_neighbors


# ---------------------------------------------------------------------------
# Depth 2 BFS covers wider neighborhood
# ---------------------------------------------------------------------------

def test_compute_depth2_a(engine):
    """'a' at depth=2: BFS collects a→b(0.9), b→a(0.9), b→c(0.6); mean = 0.8."""
    cs = engine.compute("a", depth=2)
    assert cs.score == pytest.approx(0.8)
    assert cs.n_neighbors_visited >= 2


def test_compute_depth2_b(engine):
    """'b' at depth=2: BFS collects b→a(0.9), b→c(0.6), a→b(0.9), c→b(0.6), c→d(0.3); mean = 0.66."""
    cs = engine.compute("b", depth=2)
    assert cs.score == pytest.approx(0.66)
    assert cs.n_neighbors_visited >= 3


# ---------------------------------------------------------------------------
# High / low consensus
# ---------------------------------------------------------------------------

def test_high_consensus(high_consensus_graph):
    engine = ConsensusEngine(high_consensus_graph)
    cs = engine.compute("a", depth=2)
    assert cs.score == pytest.approx(1.0)


def test_low_consensus(low_consensus_graph):
    engine = ConsensusEngine(low_consensus_graph)
    cs = engine.compute("a", depth=1)
    assert cs.score < 0.5


# ---------------------------------------------------------------------------
# Custom support_threshold
# ---------------------------------------------------------------------------

def test_custom_threshold():
    g = KnowledgeGraph(
        nodes={"a": {}, "b": {}, "c": {}},
        edges=[("a", "b", 0.7), ("a", "c", 0.4)],
    )
    engine = ConsensusEngine(g, support_threshold=0.6)
    cs = engine.compute("a", depth=1)
    assert "b" in cs.supporting_neighbors
    assert "c" in cs.conflicting_neighbors


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_compute_unknown_node(engine):
    with pytest.raises(ValueError, match="not found"):
        engine.compute("zzz", depth=1)


def test_compute_bad_depth(engine):
    with pytest.raises(ValueError, match="depth must be"):
        engine.compute("a", depth=0)


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def test_compute_all(linear_graph):
    engine = ConsensusEngine(linear_graph)
    all_scores = engine.compute_all(depth=1)
    assert set(all_scores.keys()) == {"a", "b", "c", "d"}
    for cs in all_scores.values():
        assert isinstance(cs, ConsensusScore)
        assert 0.0 <= cs.score <= 1.0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_keys(linear_graph):
    engine = ConsensusEngine(linear_graph)
    s = engine.summary(depth=1)
    assert set(s.keys()) == {"mean", "median", "std", "min", "max"}


def test_summary_empty_graph():
    engine = ConsensusEngine(KnowledgeGraph(nodes={}, edges=[]))
    s = engine.summary()
    assert s == {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


def test_summary_single_node():
    g = KnowledgeGraph(nodes={"a": {}}, edges=[])
    engine = ConsensusEngine(g)
    s = engine.summary()
    assert s["mean"] == pytest.approx(1.0)
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(1.0)


def test_summary_ordering(linear_graph):
    engine = ConsensusEngine(linear_graph)
    s = engine.summary(depth=1)
    assert s["min"] <= s["mean"] <= s["max"]
    assert s["std"] >= 0.0
