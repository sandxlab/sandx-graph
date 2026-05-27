# sandx-graph

**Graph intelligence engine — knowledge graph construction, neighborhood consensus, semantic linkage.**

[![CI](https://github.com/sandxlab/sandx-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/sandxlab/sandx-graph/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [SandX Lab](https://github.com/sandxlab) computational infrastructure ecosystem.

---

## What It Does

`sandx-graph` is the graph reasoning layer that operates downstream of `sandx-er`. It constructs knowledge graphs from resolved entity clusters and computes neighborhood consensus — a measure of how strongly each node's local neighborhood agrees.

```
sandx-er clusters  →  GraphBuilder  →  KnowledgeGraph  →  ConsensusEngine  →  consensus scores
```

## Status

> **v0.1 — Working**

| Component | Status |
|-----------|--------|
| `GraphBuilder` — construct graphs from clusters, DataFrames, similarity matrices | **Working** |
| `KnowledgeGraph` — undirected weighted graph with adjacency traversal | **Working** |
| `ConsensusEngine` — BFS neighborhood consensus computation | **Working** |
| NetworkX export | **Working** (optional dep) |
| PyPI package | **Working** |

## Installation

```bash
pip install sandx-graph
```

Or from source:

```bash
git clone https://github.com/sandxlab/sandx-graph
cd sandx-graph
pip install -e ".[dev]"
```

For NetworkX export:

```bash
pip install "sandx-graph[networkx]"
```

## Quick Start

### From sandx-er resolution output

```python
import pandas as pd
from sandx_er import EntityResolver
from sandx_graph import GraphBuilder, ConsensusEngine

# Resolve records into entity clusters
records = pd.DataFrame({
    "name": ["Acme Corp", "Acme Corp.", "GlobalTech Inc", "Global Tech"],
    "city": ["Boston", "Boston", "New York", "New York"],
})
er = EntityResolver(blocking="lsh", similarity="jaccard", threshold=0.4)
result = er.resolve(records)

# Build knowledge graph from resolved clusters
builder = GraphBuilder()
graph = builder.from_clusters(result.clusters)
print(graph)  # KnowledgeGraph(n_nodes=2, n_edges=0)

# Add relationship edges (here via similarity matrix)
import numpy as np
ids = [c.canonical_id for c in result.clusters]
sim = np.array([[1.0, 0.3], [0.3, 1.0]])
graph = builder.from_similarity_matrix(ids, sim, threshold=0.5)
```

### From DataFrames

```python
import pandas as pd
from sandx_graph import GraphBuilder, ConsensusEngine

nodes_df = pd.DataFrame({"node_id": ["e1", "e2", "e3"], "label": ["Acme", "GlobalTech", "Initech"]})
edges_df = pd.DataFrame({"source": ["e1", "e2"], "target": ["e2", "e3"], "weight": [0.85, 0.62]})

builder = GraphBuilder()
graph = builder.from_dataframe(nodes_df, edges_df)

# Compute neighborhood consensus
engine = ConsensusEngine(graph)
score = engine.compute("e1", depth=2)
print(score)
# ConsensusScore(node='e1', score=0.735, support=2, conflict=0)

# Batch over all nodes
all_scores = engine.compute_all(depth=1)
stats = engine.summary(depth=1)
print(stats)
# {'mean': 0.735, 'median': 0.735, 'std': 0.115, 'min': 0.620, 'max': 0.850}
```

## Consensus Score

`ConsensusEngine` runs BFS from a node up to a given depth, collecting all edge weights encountered. The consensus score is the weighted mean of those edges.

| Score | Interpretation |
|-------|---------------|
| → 1.0 | Node connected to high-confidence, strongly agreeing neighbors |
| → 0.5 | Mixed neighborhood — some support, some conflict |
| → 0.0 | Weak or conflicting edges throughout the neighborhood |

Isolated nodes (degree 0) return score **1.0** by convention.

## API Reference

### `GraphBuilder`

| Method | Description |
|--------|-------------|
| `from_clusters(clusters)` | One node per `sandx-er` EntityCluster; no edges |
| `from_dataframe(nodes_df, edges_df, ...)` | Build from node/edge DataFrames |
| `from_similarity_matrix(ids, similarity, threshold)` | Build from pairwise similarity matrix |

### `KnowledgeGraph`

| Attribute / Method | Description |
|--------------------|-------------|
| `n_nodes`, `n_edges` | Graph size |
| `nodes` | Dict of node_id → attribute dict |
| `edges` | List of (source, target, weight) triples |
| `neighbors(node_id)` | Adjacent node IDs |
| `neighbors_weighted(node_id)` | (neighbor_id, weight) pairs |
| `degree(node_id)` | Number of incident edges |
| `has_node(node_id)`, `has_edge(a, b)` | Membership checks |
| `to_dataframe()` | Edge list as pandas DataFrame |
| `to_networkx()` | Export to NetworkX Graph |

### `ConsensusEngine`

| Method | Description |
|--------|-------------|
| `compute(node_id, depth=2)` | Consensus score for one node |
| `compute_all(depth=2)` | Scores for all nodes |
| `summary(depth=1)` | Mean/median/std/min/max over all nodes |

## Related

- [`sandx-er`](https://github.com/sandxlab/sandx-er) — upstream entity resolution (primary input)
- [`sandx-embed`](https://github.com/sandxlab/sandx-embed) — shared embedding infrastructure
- [sandx.io](https://sandx.io) — project home

## License

Apache 2.0 — see [LICENSE](LICENSE)
