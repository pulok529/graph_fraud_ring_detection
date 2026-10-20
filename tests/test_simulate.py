"""Tests for synthetic transaction graph simulation and ring generation."""

from __future__ import annotations

import networkx as nx

from fraud_rings.simulate import simulate


def test_simulate_ring_rate():
    """Verify that ring membership rate remains realistically between 2% and 7%."""
    accounts_df, _ = simulate(n_accounts=3000, n_rings=12, seed=5)
    rate = accounts_df["is_ring_member"].mean()
    assert 0.02 <= rate <= 0.07, f"Ring rate {rate:.4f} outside expected [0.02, 0.07]"

    # Test smaller scale
    acc_small, _ = simulate(n_accounts=1000, n_rings=5, seed=42)
    rate_small = acc_small["is_ring_member"].mean()
    assert 0.02 <= rate_small <= 0.07, f"Small ring rate {rate_small:.4f} outside bounds"


def test_simulate_ids_unique():
    """Ensure all generated account IDs are strictly unique."""
    accounts_df, _ = simulate(n_accounts=1500, n_rings=6, seed=123)
    assert accounts_df["account_id"].is_unique
    assert len(accounts_df) == 1500


def test_cycles_exist_inside_rings():
    """Verify that directed transfer cycles exist among fraud ring members."""
    accounts_df, transfers_df = simulate(n_accounts=2000, n_rings=8, seed=42)
    ring_members = set(accounts_df[accounts_df["is_ring_member"] == 1]["account_id"])

    # Subgraph of transfers strictly between ring members
    ring_tx = transfers_df[
        transfers_df["src"].isin(ring_members) & transfers_df["dst"].isin(ring_members)
    ]
    G_ring = nx.DiGraph()
    for _, row in ring_tx.iterrows():
        G_ring.add_edge(row["src"], row["dst"])

    # Count simple cycles
    cycles = list(nx.simple_cycles(G_ring))
    assert len(cycles) > 0, "Expected directed cycles to exist inside fraud rings"
