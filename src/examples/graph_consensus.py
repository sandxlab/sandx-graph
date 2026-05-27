"""sandx-graph — knowledge graph construction and consensus scoring.

Demonstrates the full sandx-graph workflow using a small synthetic dataset
of tech companies. No external dependencies required beyond sandx-graph itself.

Install:  pip install sandx-graph
Run:      python -m examples.graph_consensus
"""

from __future__ import annotations

import pandas as pd

from sandx_graph import ConsensusEngine, GraphBuilder

W = 60
SEP  = "=" * W
RULE = "-" * W


def run() -> None:
    print()
    print("  " + SEP)
    print("   sandx-graph  --  Knowledge Graph & Consensus")
    print("  " + SEP)

    # ── 1. Build graph from DataFrames ────────────────────────────────────
    nodes_df = pd.DataFrame({
        "node_id":    ["apple", "microsoft", "google", "amazon", "meta"],
        "label":      ["Apple Inc", "Microsoft Corp", "Google LLC", "Amazon Inc", "Meta Platforms"],
        "sector":     ["tech"] * 5,
    })

    # Relationship edges: pairwise similarity (e.g. from embedding model or analyst)
    edges_df = pd.DataFrame({
        "source": ["apple",     "apple",     "google",    "microsoft", "amazon"],
        "target": ["microsoft", "google",    "amazon",    "google",    "meta"],
        "weight": [0.82,        0.78,        0.75,        0.71,        0.66],
    })

    builder = GraphBuilder()
    graph = builder.from_dataframe(nodes_df, edges_df, node_id_col="node_id")
    print(f"\n  {graph}")
    print()

    # ── 2. Inspect the graph ─────────────────────────────────────────────
    print("  EDGES")
    print("  " + RULE)
    print(f"  {'SOURCE':<14}  {'TARGET':<14}  WEIGHT")
    print("  " + RULE)
    for src, tgt, w in sorted(graph.edges, key=lambda e: -e[2]):
        label_src = nodes_df.set_index("node_id").loc[src, "label"]
        label_tgt = nodes_df.set_index("node_id").loc[tgt, "label"]
        bar = "#" * int(w * 24)
        print(f"  {label_src:<14}  {label_tgt:<14}  {w:.2f}  {bar}")
    print()

    # ── 3. Consensus scoring ─────────────────────────────────────────────
    engine = ConsensusEngine(graph, support_threshold=0.7)
    all_scores = engine.compute_all(depth=2)

    print("  CONSENSUS SCORES  (depth=2)")
    print("  " + RULE)
    print(f"  {'ENTITY':<16}  {'SCORE':>5}  {'SUPPORT':>7}  {'CONFLICT':>8}")
    print("  " + RULE)

    node_labels = nodes_df.set_index("node_id")["label"].to_dict()
    for nid, cs in sorted(all_scores.items(), key=lambda x: -x[1].score):
        label = node_labels[nid]
        print(
            f"  {label:<16}  {cs.score:>5.3f}  "
            f"{len(cs.supporting_neighbors):>7}  {len(cs.conflicting_neighbors):>8}"
        )

    stats = engine.summary(depth=1)
    print()
    print(f"  Mean consensus: {stats['mean']:.3f}  |  "
          f"Min: {stats['min']:.3f}  |  Max: {stats['max']:.3f}")

    # ── 4. Neighbor traversal ────────────────────────────────────────────
    print()
    print("  NEIGHBORHOOD  (apple, depth=1)")
    print("  " + RULE)
    for neighbor, weight in graph.neighbors_weighted("apple"):
        label = node_labels[neighbor]
        print(f"  {label:<16}  weight={weight:.2f}")

    print()
    print("  " + SEP)
    print(f"   {graph.n_nodes} nodes  |  {graph.n_edges} edges  |  consensus computed")
    print("  " + SEP)
    print()

    # ── 5. DataFrame export ──────────────────────────────────────────────
    print("  Edge list as DataFrame:")
    print()
    edge_df = graph.to_dataframe()
    print(edge_df.to_string(index=False))
    print()


if __name__ == "__main__":
    run()
