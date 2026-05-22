# sandx-graph

**Graph intelligence, consensus computation, and semantic linkage for the SandX platform.**

Part of the [SandX Lab](https://github.com/sandxlab) computational infrastructure ecosystem.

---

## What It Does

`sandx-graph` is the graph reasoning layer that operates downstream of `sandx-er`. It constructs and queries knowledge graphs from resolved entity clusters, computes neighborhood consensus, and provides semantic linkage across graph nodes.

```
sandx-er output (resolved clusters) → sandx-graph → knowledge graph + consensus
```

## Status

> **Phase 1 — Architecture & Foundations**

| Component | Status |
|-----------|--------|
| `sandx_graph.builder` — graph construction from entity clusters | Skeleton |
| `sandx_graph.consensus` — neighborhood consensus computation | Skeleton |
| `sandx_graph.query` — graph traversal and semantic search | Skeleton |
| Python SDK on PyPI | Planned (Phase 2) |

## Use Cases

| Use Case | Description |
|----------|------------|
| Knowledge graph construction | Build enterprise or scientific KGs from resolved records |
| AI memory infrastructure | Persistent structured memory for LLM agents |
| Fraud detection | Graph-based anomaly detection in identity networks |
| Recommendation | Graph collaborative filtering with neighborhood consensus |

## Quick Start (planned API)

```python
from sandx_graph import GraphBuilder, ConsensusEngine

# build a knowledge graph from resolved entity clusters
builder = GraphBuilder()
graph = builder.from_clusters(resolution_result.clusters)

# compute neighborhood consensus
consensus = ConsensusEngine(graph)
scores = consensus.compute(node_id="entity_42")
```

## Related

- [`sandx-er`](https://github.com/sandxlab/sandx-er) — upstream entity resolution (primary input)
- [`sandx-embed`](https://github.com/sandxlab/sandx-embed) — node embedding representations (shared dependency)
- [`sandx-compute`](https://github.com/sandxlab/sandx-compute) — distributed compute orchestration

## License

Apache 2.0 — see [LICENSE](LICENSE)
