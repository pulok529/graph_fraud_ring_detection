"""Graph construction utilities for financial transactions and entity sharing footprints."""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(accounts: pd.DataFrame, transfers: pd.DataFrame) -> nx.DiGraph:
    """Build a directed transaction graph with account node attributes.

    Parameters
    ----------
    accounts : pd.DataFrame
        DataFrame containing account metadata with columns:
        [account_id, created_day, kyc_level, device_id, is_ring_member, ring_id].
    transfers : pd.DataFrame
        DataFrame containing transfer records with columns:
        [src, dst, amount, day].

    Returns
    -------
    nx.DiGraph
        Directed graph where nodes represent accounts with attributes and
        edges represent financial transfers with aggregated amount and count.
    """
    G = nx.DiGraph()

    # Add all account nodes with attributes
    for _, row in accounts.iterrows():
        node_id = str(row["account_id"])
        G.add_node(
            node_id,
            account_id=node_id,
            created_day=int(row["created_day"]),
            kyc_level=int(row["kyc_level"]),
            device_id=str(row["device_id"]),
            is_ring_member=int(row["is_ring_member"]),
            ring_id=int(row["ring_id"]),
        )

    # Aggregate transfers between src and dst to preserve total volume
    for _, row in transfers.iterrows():
        u = str(row["src"])
        v = str(row["dst"])
        amt = float(row["amount"])
        day = int(row["day"])

        if G.has_edge(u, v):
            G[u][v]["amount"] += amt
            G[u][v]["count"] += 1
            G[u][v]["last_day"] = max(G[u][v]["last_day"], day)
        else:
            G.add_edge(u, v, amount=amt, count=1, last_day=day)

    return G


def build_device_graph(accounts: pd.DataFrame) -> nx.Graph:
    """Build an undirected graph linking accounts that share device identifiers.

    Parameters
    ----------
    accounts : pd.DataFrame
        DataFrame containing account metadata with columns:
        [account_id, device_id].

    Returns
    -------
    nx.Graph
        Undirected graph where edges connect accounts sharing the same device.
    """
    device_G = nx.Graph()

    # Ensure all accounts exist as nodes
    for acc_id in accounts["account_id"]:
        device_G.add_node(str(acc_id))

    # Group accounts by device_id and create clique edges
    device_groups = accounts.groupby("device_id")["account_id"].apply(list)
    for dev_id, acc_list in device_groups.items():
        if len(acc_list) > 1:
            for i in range(len(acc_list)):
                for j in range(i + 1, len(acc_list)):
                    device_G.add_edge(str(acc_list[i]), str(acc_list[j]), device_id=str(dev_id))

    return device_G
