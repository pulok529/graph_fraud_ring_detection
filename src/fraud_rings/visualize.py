"""Visualization utilities for fraud ring networks and model diagnostics."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_top_ring(
    G: nx.DiGraph,
    accounts: pd.DataFrame,
    community_nodes: Iterable[str],
    path: str | Path,
) -> None:
    """Draw a spring-layout plot of the highest-scoring flagged community plus 1-hop neighbors.

    Saves a publication-quality diagram with ring members colored in cyan (#22D3EE),
    peripheral/external neighbors in grey, and an obsidian background (#0B1020).

    Parameters
    ----------
    G : nx.DiGraph
        Directed transaction graph.
    accounts : pd.DataFrame
        Account metadata with ['account_id', 'is_ring_member'].
    community_nodes : Iterable[str]
        Account IDs belonging to the top flagged community.
    path : str or Path
        Output file path for the saved PNG.
    """
    comm_set = set(str(n) for n in community_nodes)

    # Collect 1-hop neighborhood
    neighbors: set[str] = set()
    for u in comm_set:
        if u in G:
            neighbors.update(G.predecessors(u))
            neighbors.update(G.successors(u))

    # Bound external neighbors for clarity
    external_neighbors = list(neighbors - comm_set)
    if len(external_neighbors) > 40:
        external_neighbors = external_neighbors[:40]

    plot_nodes = list(comm_set | set(external_neighbors))
    subG = G.subgraph(plot_nodes)

    # Build node color and size mappings
    ring_map = dict(
        zip(
            accounts["account_id"].astype(str),
            accounts["is_ring_member"].astype(int),
            strict=False,
        )
    )

    node_colors = []
    node_sizes = []
    for n in subG.nodes():
        is_ring = ring_map.get(str(n), 0) == 1
        if is_ring:
            node_colors.append("#22D3EE")  # cyan
        else:
            node_colors.append("#64748B")  # slate grey

        if n in comm_set:
            node_sizes.append(380)
        else:
            node_sizes.append(140)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor("#0B1020")
    ax.set_facecolor("#0B1020")

    pos = nx.spring_layout(subG, seed=42, k=0.45, iterations=60)

    # Draw edges
    nx.draw_networkx_edges(
        subG,
        pos,
        ax=ax,
        edge_color="#334155",
        alpha=0.6,
        arrows=True,
        arrowsize=10,
        width=1.2,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        subG,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#0F172A",
        linewidths=1.2,
    )

    # Labels for community nodes
    comm_sub_pos = {n: pos[n] for n in comm_set if n in pos}
    labels = {n: str(n) for n in comm_sub_pos}
    nx.draw_networkx_labels(
        subG,
        comm_sub_pos,
        labels=labels,
        ax=ax,
        font_size=7,
        font_color="#F8FAFC",
        font_weight="bold",
    )

    ax.set_title(
        "Detected Fraud Ring & 1-Hop Neighborhood (Cyan = Ring Member)",
        color="#F8FAFC",
        fontsize=12,
        pad=16,
    )
    ax.axis("off")

    os.makedirs(Path(path).parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    path: str | Path,
) -> None:
    """Save Precision-Recall curve diagram for the supervised fraud detector.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_prob : array-like
        Predicted risk probabilities.
    path : str or Path
        Destination path for the output PNG.
    """
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("#0B1020")
    ax.set_facecolor("#0B1020")

    ax.plot(rec, prec, color="#22D3EE", linewidth=2.5, label=f"PR Curve (AP = {ap:.4f})")
    ax.set_xlabel("Recall", color="#F8FAFC", fontsize=11)
    ax.set_ylabel("Precision", color="#F8FAFC", fontsize=11)
    ax.set_title("Node-Level Precision-Recall Performance", color="#F8FAFC", fontsize=12, pad=12)
    ax.tick_params(colors="#94A3B8")
    ax.grid(color="#1E293B", linestyle="--", linewidth=0.7)

    for spine in ax.spines.values():
        spine.set_color("#334155")

    leg = ax.legend(facecolor="#1E293B", edgecolor="#334155")
    for text in leg.get_texts():
        text.set_color("#F8FAFC")

    os.makedirs(Path(path).parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
