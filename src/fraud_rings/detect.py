"""Detection modules for fraud rings via unsupervised heuristics and supervised modeling."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, KFold


def detect_rings_unsupervised(
    features_df: pd.DataFrame,
    G: nx.DiGraph | None = None,
) -> pd.DataFrame:
    """Detect and rank suspicious communities using an unsupervised heuristic ring score.

    Combines internal edge density, cycle ratio, mean account age (younger = riskier),
    device sharing concentration, and amount similarity into a composite score.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame produced by node_features().
    G : nx.DiGraph, optional
        Directed transaction graph. If provided, used to verify community subgraph density.

    Returns
    -------
    pd.DataFrame
        Ranked communities with composite score, size, member account IDs,
        and constituent signal metrics.
    """
    # Group nodes by Louvain community_id (skip unassigned -1 if present)
    valid_nodes = features_df[features_df["community_id"] >= 0]
    comm_groups = valid_nodes.groupby("community_id")

    records: list[dict[str, object]] = []

    for c_id, group in comm_groups:
        size = len(group)
        member_ids = list(group["account_id"])

        # Density signal
        if G is not None and size > 1:
            density = float(nx.density(G.subgraph(member_ids)))
        else:
            density = float(group["community_density"].mean())
        density_norm = float(np.clip(density * 2.0, 0.0, 1.0))

        # Cycle ratio signal (average cycles per member)
        cycle_sum = float(group["cycle_count"].sum())
        cycle_ratio = cycle_sum / max(1, size)
        cycle_norm = float(np.clip(cycle_ratio / 3.0, 0.0, 1.0))

        # Account age signal: younger = riskier
        mean_age = float(group["account_age"].mean())
        # Account age ranges from 0 to 180; age < 40 indicates recent creation
        age_risk = float(np.clip((180.0 - mean_age) / 180.0, 0.0, 1.0))

        # Device sharing footprint
        mean_dev_sharing = float(group["shared_device_count"].mean())
        device_risk = float(np.clip(mean_dev_sharing / 3.0, 0.0, 1.0))

        # Amount similarity: lower CV = higher similarity = riskier
        mean_cv = float(group["amount_similarity"].mean())
        # Exponential drop-off on CV (cv=0 -> sim=1.0, cv=1.0 -> sim=0.13)
        amount_sim_score = float(np.exp(-2.0 * mean_cv))

        # Penalty for tiny communities (rings are typically 5-15 members)
        size_multiplier = 1.0 if size >= 4 else (0.4 if size >= 3 else 0.1)

        # Weighted combination of heuristic signals
        composite_score = float(
            (
                0.25 * density_norm
                + 0.25 * cycle_norm
                + 0.20 * age_risk
                + 0.15 * device_risk
                + 0.15 * amount_sim_score
            )
            * size_multiplier
        )

        records.append(
            {
                "community_id": int(c_id),
                "score": float(np.round(composite_score, 4)),
                "size": size,
                "member_ids": member_ids,
                "density": float(np.round(density, 4)),
                "cycle_ratio": float(np.round(cycle_ratio, 4)),
                "mean_age": float(np.round(mean_age, 2)),
                "mean_device_sharing": float(np.round(mean_dev_sharing, 2)),
                "mean_amount_cv": float(np.round(mean_cv, 4)),
            }
        )

    comm_df = pd.DataFrame(records)
    if not comm_df.empty:
        comm_df.sort_values(by="score", ascending=False, inplace=True)
        comm_df.reset_index(drop=True, inplace=True)
        comm_df["rank"] = np.arange(1, len(comm_df) + 1)
    else:
        comm_df["rank"] = []

    return comm_df


def detect_rings_supervised(
    features_df: pd.DataFrame,
    accounts_df: pd.DataFrame | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train HistGradientBoostingClassifier with GroupKFold cross-validation by community.

    Enforces community-based partitioning to prevent data leakage across interconnected
    graph clusters, providing unbiased out-of-fold risk probabilities.

    Parameters
    ----------
    features_df : pd.DataFrame
        Account-level engineered features.
    accounts_df : pd.DataFrame, optional
        Account metadata containing true labels 'is_ring_member'. If None, reads from features_df.
    n_splits : int, default=5
        Number of cross-validation folds.
    seed : int, default=42
        Random seed for model training.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        node_preds_df : Account-level predictions with columns
            [account_id, pred_prob, pred_label, is_ring_member, community_id].
        community_preds_df : Community-level aggregated predictions with columns
            [community_id, mean_prob, max_prob, size, member_ids, rank].
    """
    feature_cols = [
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
        "community_size",
        "community_density",
    ]

    df = features_df.copy()
    if accounts_df is not None and "is_ring_member" in accounts_df.columns:
        lbl_map = dict(zip(accounts_df["account_id"], accounts_df["is_ring_member"], strict=False))
        df["is_ring_member"] = df["account_id"].map(lbl_map).fillna(0).astype(int)
    elif "is_ring_member" not in df.columns:
        df["is_ring_member"] = 0

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["is_ring_member"].to_numpy(dtype=int)
    groups = df["community_id"].to_numpy(dtype=int)

    # Setup GroupKFold cross-validation
    unique_groups = np.unique(groups)
    effective_splits = max(2, min(n_splits, len(unique_groups)))

    oof_probs = np.zeros(len(df), dtype=float)

    if len(unique_groups) >= 2 and np.sum(y) > 0:
        cv = GroupKFold(n_splits=effective_splits)
        for train_idx, val_idx in cv.split(X, y, groups=groups):
            # HistGradientBoostingClassifier with balanced weights and regularized leaf size
            clf = HistGradientBoostingClassifier(
                random_state=seed,
                max_iter=100,
                min_samples_leaf=10,
                class_weight="balanced",
            )
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])
            oof_probs[val_idx] = probs[:, 1]
    else:
        # Fallback if insufficient groups or labels
        kf = KFold(n_splits=min(3, len(df)), shuffle=True, random_state=seed)
        for train_idx, val_idx in kf.split(X):
            clf = HistGradientBoostingClassifier(random_state=seed, max_iter=50)
            clf.fit(X[train_idx], y[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    # Node-level predictions DataFrame
    node_preds_df = pd.DataFrame(
        {
            "account_id": df["account_id"].values,
            "pred_prob": np.round(oof_probs, 4),
            "pred_label": (oof_probs >= 0.5).astype(int),
            "is_ring_member": y,
            "community_id": groups,
        }
    )
    node_preds_df.sort_values(by="pred_prob", ascending=False, inplace=True)
    node_preds_df.reset_index(drop=True, inplace=True)

    # Aggregate to community-level predictions
    valid_preds = node_preds_df[node_preds_df["community_id"] >= 0]
    comm_recs: list[dict[str, object]] = []

    for c_id, grp in valid_preds.groupby("community_id"):
        mean_p = float(grp["pred_prob"].mean())
        max_p = float(grp["pred_prob"].max())
        comm_recs.append(
            {
                "community_id": int(c_id),
                "mean_prob": float(np.round(mean_p, 4)),
                "max_prob": float(np.round(max_p, 4)),
                "size": len(grp),
                "member_ids": list(grp["account_id"]),
            }
        )

    community_preds_df = pd.DataFrame(comm_recs)
    if not community_preds_df.empty:
        community_preds_df.sort_values(by="mean_prob", ascending=False, inplace=True)
        community_preds_df.reset_index(drop=True, inplace=True)
        community_preds_df["rank"] = np.arange(1, len(community_preds_df) + 1)
    else:
        community_preds_df["rank"] = []

    return node_preds_df, community_preds_df


def extract_flagged_clusters(
    G: nx.DiGraph,
    node_preds_df: pd.DataFrame,
    threshold: float = 0.5,
    min_size: int = 3,
) -> pd.DataFrame:
    """Group accounts the classifier flags into candidate rings.

    Louvain communities on the full graph are dominated by shared merchants, so a
    ring ends up inside a much larger community. Instead, take the subgraph induced
    by flagged accounts (probability >= threshold) and return its connected
    components, ranked by mean probability.
    """
    probs = dict(
        zip(node_preds_df["account_id"].astype(str), node_preds_df["pred_prob"], strict=False)
    )
    flagged = [n for n in G.nodes if probs.get(str(n), 0.0) >= threshold]
    sub = G.subgraph(flagged).to_undirected()

    rows = []
    for comp in nx.connected_components(sub):
        if len(comp) < min_size:
            continue
        p = [probs[str(n)] for n in comp]
        rows.append(
            {
                "mean_prob": round(float(np.mean(p)), 4),
                "max_prob": round(float(np.max(p)), 4),
                "size": len(comp),
                "member_ids": sorted(str(n) for n in comp),
            }
        )

    cols = ["community_id", "mean_prob", "max_prob", "size", "member_ids", "rank"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values(["mean_prob", "size"], ascending=False)
    df = df.reset_index(drop=True)
    df["community_id"] = np.arange(len(df))
    df["rank"] = np.arange(1, len(df) + 1)
    return df[cols]
