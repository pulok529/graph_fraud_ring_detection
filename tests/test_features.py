"""Tests for graph feature engineering and cycle detection heuristics."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from fraud_rings.features import node_features
from fraud_rings.graph import build_device_graph, build_graph
from fraud_rings.simulate import simulate


def test_feature_frame_integrity():
    """Verify feature dataframe has one row per account and contains strictly no NaNs."""
    accounts, transfers = simulate(n_accounts=500, n_rings=3, seed=1)
    G = build_graph(accounts, transfers)
    device_G = build_device_graph(accounts)

    feat_df = node_features(G, device_G, accounts)

    # One row per account
    assert len(feat_df) == len(accounts)
    assert feat_df["account_id"].is_unique

    # No NaNs in any feature column
    assert not feat_df.isna().any().any(), "Feature frame contains NaN values"

    # Verify critical columns exist
    expected_cols = [
        "in_degree",
        "out_degree",
        "weighted_in_amount",
        "weighted_out_amount",
        "pagerank",
        "clustering",
        "k_core",
        "cycle_count",
        "reciprocity",
        "account_age",
        "kyc_level",
        "shared_device_count",
        "amount_similarity",
        "community_id",
        "community_size",
        "community_density",
    ]
    for col in expected_cols:
        assert col in feat_df.columns, f"Missing feature column '{col}'"


def test_hand_built_triangle_cycle_count():
    """Verify that cycle count is strictly positive for a hand-built directed triangle."""
    # Hand-built 3-cycle: A -> B -> C -> A
    G = nx.DiGraph()
    G.add_edge("node_A", "node_B", amount=500.0)
    G.add_edge("node_B", "node_C", amount=500.0)
    G.add_edge("node_C", "node_A", amount=500.0)

    accounts = pd.DataFrame(
        [
            {
                "account_id": "node_A",
                "created_day": 20,
                "kyc_level": 2,
                "device_id": "d1",
                "is_ring_member": 1,
                "ring_id": 0,
            },
            {
                "account_id": "node_B",
                "created_day": 20,
                "kyc_level": 2,
                "device_id": "d2",
                "is_ring_member": 1,
                "ring_id": 0,
            },
            {
                "account_id": "node_C",
                "created_day": 20,
                "kyc_level": 2,
                "device_id": "d3",
                "is_ring_member": 1,
                "ring_id": 0,
            },
        ]
    )

    device_G = nx.Graph()
    device_G.add_nodes_from(["node_A", "node_B", "node_C"])

    feat_df = node_features(G, device_G, accounts)

    for node_id in ["node_A", "node_B", "node_C"]:
        node_cycles = feat_df.loc[node_id, "cycle_count"]
        msg = f"Expected node '{node_id}' to have cycle count > 0, got {node_cycles}"
        assert node_cycles > 0, msg
