"""Feature extraction pipeline for transaction graphs and community structures."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def _count_bounded_cycles(G: nx.DiGraph, max_per_node: int = 100) -> dict[str, int]:
    """Compute bounded simple cycles (lengths 2, 3, and 4) through each node.

    Uses efficient successor-predecessor set intersections to count 2-cycles
    (mutual edges), 3-cycles (triangles), and sample bounded 4-cycles with a cap
    to preserve sub-minute execution on large transaction graphs.

    Parameters
    ----------
    G : nx.DiGraph
        Directed transaction graph.
    max_per_node : int, default=100
        Maximum cycle count to record per node to prevent explosion on hubs.

    Returns
    -------
    dict[str, int]
        Mapping from node ID to total cycle count through that node.
    """
    succ_map = {n: set(G.successors(n)) for n in G}
    pred_map = {n: set(G.predecessors(n)) for n in G}
    cycle_counts: dict[str, int] = {}

    for u in G:
        cycles = 0
        s_u = succ_map[u]
        p_u = pred_map[u]

        # Length 2 cycles: u -> v -> u (mutual edges)
        mutual = s_u & p_u
        cycles += len(mutual)

        # Length 3 cycles: u -> v -> w -> u
        for v in s_u:
            # w must be in succ(v) and in pred(u)
            w_candidates = (succ_map[v] & p_u) - {u, v}
            cycles += len(w_candidates)

        # Length 4 cycles (bounded search): u -> v -> w -> x -> u
        if cycles < max_per_node:
            sampled_v = list(s_u)[:12]
            for v in sampled_v:
                if cycles >= max_per_node:
                    break
                w_nodes = list(succ_map[v] - {u, v})[:12]
                for w in w_nodes:
                    x_candidates = (succ_map[w] & p_u) - {u, v, w}
                    cycles += len(x_candidates)
                    if cycles >= max_per_node:
                        break

        cycle_counts[u] = min(cycles, max_per_node)

    return cycle_counts


def node_features(
    G: nx.DiGraph,
    device_graph: nx.Graph,
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """Extract graph topology, device footprints, and community features per account.

    Parameters
    ----------
    G : nx.DiGraph
        Directed transaction graph.
    device_graph : nx.Graph
        Undirected device-sharing graph.
    accounts : pd.DataFrame
        Account metadata with columns:
        [account_id, created_day, kyc_level, is_ring_member, ring_id].

    Returns
    -------
    pd.DataFrame
        One row per account containing engineered node and community features,
        with no NaN values.
    """
    node_list = [str(a) for a in accounts["account_id"]]

    # 1. Undirected projection for structural metrics
    G_undirected = G.to_undirected()

    # 2. Centrality and clustering metrics
    if len(G) > 0:
        try:
            pagerank_map = nx.pagerank(G, alpha=0.85, max_iter=200)
        except Exception:
            pagerank_map = {n: 1.0 / max(1, len(G)) for n in G}
    else:
        pagerank_map = {}

    clustering_map = nx.clustering(G_undirected) if len(G_undirected) > 0 else {}
    core_map = nx.core_number(G_undirected) if len(G_undirected) > 0 else {}
    cycle_map = _count_bounded_cycles(G)

    # 3. Community detection on undirected projection (Louvain with fixed seed)
    if len(G_undirected) > 0:
        comms = nx.algorithms.community.louvain_communities(G_undirected, seed=42)
    else:
        comms = []

    node_to_comm: dict[str, int] = {}
    comm_size_map: dict[str, int] = {}
    comm_density_map: dict[str, float] = {}

    for c_idx, comm in enumerate(comms):
        c_size = len(comm)
        # Compute directed subgraph density inside community
        if c_size > 1:
            sub = G.subgraph(comm)
            c_density = float(nx.density(sub))
        else:
            c_density = 0.0

        for node in comm:
            node_to_comm[node] = c_idx
            comm_size_map[node] = c_size
            comm_density_map[node] = c_density

    # 4. Assemble feature rows
    rows: list[dict[str, object]] = []
    accounts_by_id = accounts.set_index("account_id")

    for node in node_list:
        acc_meta = (
            accounts_by_id.loc[node]
            if node in accounts_by_id.index
            else None
        )
        created_day = int(acc_meta["created_day"]) if acc_meta is not None else 90
        kyc_level = int(acc_meta["kyc_level"]) if acc_meta is not None else 2
        is_ring = (
            int(acc_meta["is_ring_member"])
            if (acc_meta is not None and "is_ring_member" in acc_meta)
            else 0
        )
        ring_id = (
            int(acc_meta["ring_id"])
            if (acc_meta is not None and "ring_id" in acc_meta)
            else -1
        )

        # Graph degrees and amounts
        if node in G:
            in_deg = int(G.in_degree(node))
            out_deg = int(G.out_degree(node))
            in_amt = float(sum(d.get("amount", 0.0) for _, _, d in G.in_edges(node, data=True)))
            out_edges_data = list(G.out_edges(node, data=True))
            out_amt = float(sum(d.get("amount", 0.0) for _, _, d in out_edges_data))

            # Reciprocity
            succ_set = set(G.successors(node))
            pred_set = set(G.predecessors(node))
            mutual = len(succ_set & pred_set)
            tot_deg = in_deg + out_deg
            reciprocity = float((2.0 * mutual) / tot_deg) if tot_deg > 0 else 0.0

            # Amount similarity: Coefficient of variation of out-amounts
            out_amounts = [float(d.get("amount", 0.0)) for _, _, d in out_edges_data]
            if len(out_amounts) >= 2:
                m_amt = float(np.mean(out_amounts))
                amt_cv = float(np.std(out_amounts, ddof=0) / m_amt) if m_amt > 1e-6 else 0.0
            else:
                amt_cv = 0.0
        else:
            in_deg = 0
            out_deg = 0
            in_amt = 0.0
            out_amt = 0.0
            reciprocity = 0.0
            amt_cv = 0.0

        # Device graph footprint
        shared_dev = int(device_graph.degree(node)) if node in device_graph else 0

        # Account age (days since creation, where recent = smaller age)
        account_age = max(0, 180 - created_day)

        rows.append(
            {
                "account_id": node,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "weighted_in_amount": in_amt,
                "weighted_out_amount": out_amt,
                "pagerank": float(pagerank_map.get(node, 0.0)),
                "clustering": float(clustering_map.get(node, 0.0)),
                "k_core": int(core_map.get(node, 0)),
                "cycle_count": int(cycle_map.get(node, 0)),
                "reciprocity": reciprocity,
                "account_age": account_age,
                "kyc_level": kyc_level,
                "shared_device_count": shared_dev,
                "amount_similarity": amt_cv,
                "community_id": int(node_to_comm.get(node, -1)),
                "community_size": int(comm_size_map.get(node, 1)),
                "community_density": float(comm_density_map.get(node, 0.0)),
                "is_ring_member": is_ring,
                "ring_id": ring_id,
            }
        )

    feat_df = pd.DataFrame(rows)
    feat_df.fillna(0.0, inplace=True)
    feat_df.set_index("account_id", drop=False, inplace=True)
    return feat_df
