# Graph Intelligence — Domain Overview

**Domain:** Graph-based reasoning, knowledge graphs, graph consensus, semantic linkage
**SandX engine:** `sandx-graph`
**Phase 2 priority:** #3 (after sandx-er and sandx-embed)

---

## What Is Graph Intelligence?

Graph Intelligence refers to computational methods that extract meaning, structure, and consensus from graph-structured data — systems where entities are nodes and relationships are edges. Modern AI increasingly depends on graph structures for knowledge representation, reasoning, recommendation, and memory.

SandX-Graph is the reasoning layer that operates on the output of `sandx-er` (resolved entity clusters) and `sandx-embed` (latent representations), constructing and querying knowledge graphs and computing consensus over graph neighborhoods.

---

## Core Problems

### Knowledge Graph Construction
Given a set of resolved entities and their attributes, build a structured graph representation that captures relationships, hierarchy, and semantics. This is the downstream task of `sandx-er` — resolved identity clusters become nodes; extracted relationships become edges.

### Graph Consensus
In distributed or noisy environments, multiple observers may submit conflicting claims about graph edges (e.g., "A is related to B" with varying confidence). Graph consensus reconciles these claims into a consistent, high-confidence graph state. Related to: belief propagation, collective classification, message passing.

### Neighborhood Similarity
Two nodes in a graph are semantically similar not only if their attributes match, but if their neighborhoods are similar — the entities they are connected to are also similar. Neighborhood consensus computation extends pairwise similarity to structural similarity.

### Semantic Linkage
Connecting entities across heterogeneous graphs or ontologies by semantic meaning rather than syntactic identifier. Required for cross-source knowledge integration.

---

## Key Methods

| Method | Application |
|--------|------------|
| Graph Neural Networks (GNNs) | Node classification, link prediction, graph matching |
| Message Passing Neural Networks (MPNNs) | Neighborhood aggregation, consensus |
| Knowledge Graph Embeddings (TransE, RotatE) | Entity and relation representation |
| Graph Attention Networks (GATs) | Attention-weighted neighbor aggregation |
| Loopy Belief Propagation | Probabilistic inference over graph structures |
| Graph Contrastive Learning | Self-supervised node representation |

---

## Relationship to Other SandX Engines

```
sandx-er output (resolved entity clusters)
         │
         ▼
sandx-graph (graph construction + reasoning)
         │
         uses
         │
         ▼
sandx-embed (node representations, neighborhood similarity)
```

`sandx-graph` is not independent — it is most powerful as the downstream layer of `sandx-er`. A SandX data pipeline typically runs: embed → resolve (ER) → reason (graph).

---

## Use Cases

| Use Case | Description |
|----------|------------|
| Knowledge graph construction | Build enterprise or scientific knowledge graphs from resolved records |
| Recommendation systems | Graph-based collaborative filtering with neighborhood consensus |
| AI memory infrastructure | Persistent structured memory for LLM agents via knowledge graphs |
| Fraud detection | Graph-based anomaly detection in financial identity networks |
| Drug discovery | Biological entity linkage and relationship reasoning |
| Scientific database integration | Cross-source entity and relationship reconciliation |

---

## Key References

- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs (GraphSAGE). *NeurIPS.*
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR.*
- Bordes, A. et al. (2013). Translating Embeddings for Modeling Multi-relational Data (TransE). *NeurIPS.*
- Velickovic, P. et al. (2018). Graph Attention Networks. *ICLR.*
- Nickel, M., Murphy, K., Tresp, V., & Gabrilovich, E. (2016). A Review of Relational Machine Learning for Knowledge Graphs. *Proceedings of the IEEE.*
- Schlichtkrull, M. et al. (2018). Modeling Relational Data with Graph Convolutional Networks (R-GCN). *ESWC.* — Extends GCNs to multi-relational graphs; foundational for knowledge graph reasoning over heterogeneous entity types.
- Wang, Q. et al. (2017). Knowledge Graph Embedding: A Survey of Approaches and Applications. *IEEE TKDE.* — Comprehensive survey of KG embedding methods; useful reference for `sandx-graph` representation choices.
- Shi, B., & Weninger, T. (2018). Open-World Knowledge Graph Completion. *AAAI.* — Addresses the problem of reasoning over incomplete knowledge graphs; relevant to consensus computation under missing edges.
