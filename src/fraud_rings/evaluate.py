"""Evaluation metrics for node-level classification and community-level ring retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_node_level(
    y_true: np.ndarray | pd.Series | list[int],
    y_prob: np.ndarray | pd.Series | list[float],
) -> dict[str, float]:
    """Compute node-level ranking and classification performance metrics.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels (1 for ring member, 0 otherwise).
    y_prob : array-like
        Predicted fraud probabilities.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - node_roc_auc : Receiver Operating Characteristic AUC.
        - node_pr_auc : Precision-Recall AUC (Average Precision).
        - node_precision_at_k : Precision in top-k where k is the number of true ring members.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    # 1. ROC-AUC
    if len(np.unique(y_t)) < 2:
        roc_auc = 0.5
    else:
        roc_auc = float(roc_auc_score(y_t, y_p))

    # 2. PR-AUC
    k_true = int(np.sum(y_t == 1))
    if k_true == 0:
        pr_auc = 0.0
        prec_at_k = 0.0
    else:
        pr_auc = float(average_precision_score(y_t, y_p))

        # 3. Precision@k (where k = number of positive cases)
        sorted_indices = np.argsort(y_p)[::-1]
        top_k_indices = sorted_indices[:k_true]
        prec_at_k = float(np.sum(y_t[top_k_indices] == 1) / k_true)

    return {
        "node_roc_auc": float(np.round(roc_auc, 4)),
        "node_pr_auc": float(np.round(pr_auc, 4)),
        "node_precision_at_k": float(np.round(prec_at_k, 4)),
    }


def evaluate_community_level(
    ranked_communities: pd.DataFrame,
    accounts_df: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, float]:
    """Compute community-level retrieval precision and true ring recall.

    Parameters
    ----------
    ranked_communities : pd.DataFrame
        DataFrame of detected communities sorted by suspicion score,
        with column 'member_ids'.
    accounts_df : pd.DataFrame
        Account metadata with columns ['account_id', 'is_ring_member', 'ring_id'].
    top_n : int, default=10
        Number of top communities to evaluate.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - community_precision_at_10 : Share of top-10 communities where >=50%
          of members are true ring members.
        - ring_recall : Share of true rings with at least half their members
          recovered inside the flagged communities.
    """
    if ranked_communities.empty or accounts_df.empty:
        return {
            "community_precision_at_10": 0.0,
            "ring_recall": 0.0,
        }

    ring_members_set = set(
        accounts_df[accounts_df["is_ring_member"] == 1]["account_id"].astype(str)
    )
    top_comms = ranked_communities.head(top_n)

    # 1. Precision@top_n communities (hit if >=50% members are ring members)
    hits = 0
    flagged_accounts: set[str] = set()

    for _, row in top_comms.iterrows():
        members = set(str(m) for m in row["member_ids"])
        flagged_accounts.update(members)
        if len(members) == 0:
            continue
        ring_member_count = len(members & ring_members_set)
        if ring_member_count / len(members) >= 0.5:
            hits += 1

    precision_at_10 = float(hits / max(1, len(top_comms)))

    # 2. Ring recall: Share of true rings with >=50% members in flagged accounts
    true_rings = [r for r in accounts_df["ring_id"].unique() if int(r) >= 0]
    if len(true_rings) == 0:
        ring_recall = 0.0
    else:
        recalled_rings = 0
        for r_id in true_rings:
            ring_nodes = set(accounts_df[accounts_df["ring_id"] == r_id]["account_id"].astype(str))
            if len(ring_nodes) == 0:
                continue
            recovered_count = len(ring_nodes & flagged_accounts)
            if recovered_count / len(ring_nodes) >= 0.5:
                recalled_rings += 1
        ring_recall = float(recalled_rings / len(true_rings))

    return {
        "community_precision_at_10": float(np.round(precision_at_10, 4)),
        "ring_recall": float(np.round(ring_recall, 4)),
    }


def evaluate_all(
    node_preds_df: pd.DataFrame,
    community_preds_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
) -> dict[str, float]:
    """Calculate combined node-level and community-level evaluation metrics.

    Parameters
    ----------
    node_preds_df : pd.DataFrame
        DataFrame with 'is_ring_member' and 'pred_prob'.
    community_preds_df : pd.DataFrame
        Ranked communities DataFrame.
    accounts_df : pd.DataFrame
        Full account metadata.

    Returns
    -------
    dict[str, float]
        Combined dictionary containing node and community metrics.
    """
    node_metrics = evaluate_node_level(
        y_true=node_preds_df["is_ring_member"],
        y_prob=node_preds_df["pred_prob"],
    )
    comm_metrics = evaluate_community_level(
        ranked_communities=community_preds_df,
        accounts_df=accounts_df,
        top_n=10,
    )
    return {**node_metrics, **comm_metrics}
