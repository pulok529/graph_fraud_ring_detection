"""End-to-end execution pipeline and CLI for graph fraud ring detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_rings.detect import (
    detect_rings_supervised,
    detect_rings_unsupervised,
    extract_flagged_clusters,
)
from fraud_rings.evaluate import evaluate_all
from fraud_rings.features import node_features
from fraud_rings.graph import build_device_graph, build_graph
from fraud_rings.simulate import simulate
from fraud_rings.visualize import plot_pr_curve, plot_top_ring


def run_pipeline(
    accounts: int = 3000,
    rings: int = 12,
    seed: int = 5,
    out_dir: str = "artifacts",
) -> dict[str, object]:
    """Execute the full simulation, feature engineering, detection, and evaluation workflow.

    Parameters
    ----------
    accounts : int, default=3000
        Number of accounts to simulate.
    rings : int, default=12
        Number of fraud rings to embed.
    seed : int, default=5
        Random seed for simulation and model stability.
    out_dir : str, default="artifacts"
        Output directory to save execution artifacts.

    Returns
    -------
    dict[str, object]
        Dictionary containing computed metrics and result DataFrames.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Simulate data
    accounts_df, transfers_df = simulate(n_accounts=accounts, n_rings=rings, seed=seed)

    # 2. Build graph representations
    G = build_graph(accounts_df, transfers_df)
    device_G = build_device_graph(accounts_df)

    # 3. Extract node and community features
    feat_df = node_features(G, device_G, accounts_df)

    # 4. Run detectors
    # Heuristic Unsupervised detector
    unsup_comm_df = detect_rings_unsupervised(feat_df, G)

    # Supervised HistGradientBoosting detector with GroupKFold
    node_preds_df, sup_comm_df = detect_rings_supervised(
        features_df=feat_df,
        accounts_df=accounts_df,
        seed=seed,
    )

    # Candidate rings = connected groups of flagged accounts
    sup_comm_df = extract_flagged_clusters(G, node_preds_df)

    # 5. Evaluate performance
    metrics = evaluate_all(
        node_preds_df=node_preds_df,
        community_preds_df=sup_comm_df,
        accounts_df=accounts_df,
    )

    # 6. Save artifacts
    # Save metrics JSON
    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save top communities CSV
    # Blend top communities or store supervised community rankings
    top_comms_file = out_path / "top_communities.csv"
    # Convert member_ids list to semicolon-delimited string for clean CSV storage
    save_comm_df = sup_comm_df.copy()
    save_comm_df["member_ids"] = save_comm_df["member_ids"].apply(
        lambda ids: ";".join(str(i) for i in ids)
    )
    save_comm_df.to_csv(top_comms_file, index=False)

    # Save additional data tables for API fast startup
    accounts_df.to_csv(out_path / "accounts.csv", index=False)
    feat_df.to_csv(out_path / "features.csv", index=False)
    node_preds_df.to_csv(out_path / "node_preds.csv", index=False)

    # Save unsupervised community heuristics for API signal queries
    unsup_save = unsup_comm_df.copy()
    unsup_save["member_ids"] = unsup_save["member_ids"].apply(
        lambda ids: ";".join(str(i) for i in ids)
    )
    unsup_save.to_csv(out_path / "heuristic_communities.csv", index=False)

    # 7. Generate diagnostic plots
    # Plot top detected ring
    if not sup_comm_df.empty:
        top_comm_members = sup_comm_df.iloc[0]["member_ids"]
        if isinstance(top_comm_members, str):
            top_comm_members = top_comm_members.split(";")
    elif not unsup_comm_df.empty:
        top_comm_members = unsup_comm_df.iloc[0]["member_ids"]
    else:
        top_comm_members = []

    top_ring_path = out_path / "top_ring.png"
    plot_top_ring(
        G=G,
        accounts=accounts_df,
        community_nodes=top_comm_members,
        path=top_ring_path,
    )

    # Plot Precision-Recall curve
    plot_pr_curve(
        y_true=node_preds_df["is_ring_member"],
        y_prob=node_preds_df["pred_prob"],
        path=out_path / "pr_curve.png",
    )

    # 8. Print formatted markdown table (lines <= 100 characters)
    table_lines = [
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Node ROC-AUC | {metrics['node_roc_auc']:.4f} |",
        f"| Node PR-AUC | {metrics['node_pr_auc']:.4f} |",
        f"| Node Precision@k | {metrics['node_precision_at_k']:.4f} |",
        f"| Flagged-cluster precision (top 10) | {metrics['community_precision_at_10']:.4f} |",
        f"| Ring recall (flagged clusters) | {metrics['ring_recall']:.4f} |",
    ]
    print("\n".join(table_lines))

    return {
        "metrics": metrics,
        "accounts_df": accounts_df,
        "transfers_df": transfers_df,
        "features_df": feat_df,
        "node_preds_df": node_preds_df,
        "supervised_communities": sup_comm_df,
        "unsupervised_communities": unsup_comm_df,
    }


def main() -> None:
    """CLI entrypoint for running the fraud detection pipeline."""
    parser = argparse.ArgumentParser(description="Run graph-based fraud ring detection pipeline.")
    parser.add_argument(
        "--accounts",
        type=int,
        default=3000,
        help="Number of accounts to simulate (default: 3000)",
    )
    parser.add_argument(
        "--rings",
        type=int,
        default=12,
        help="Number of fraud rings to embed (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="Random seed for reproducibility (default: 5)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts",
        help="Output directory for artifacts (default: artifacts/)",
    )

    args = parser.parse_args()
    run_pipeline(
        accounts=args.accounts,
        rings=args.rings,
        seed=args.seed,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
