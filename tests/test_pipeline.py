"""Integration tests for the full fraud detection pipeline execution."""

from __future__ import annotations

from pathlib import Path

from fraud_rings.pipeline import run_pipeline


def test_pipeline_execution(tmp_path: Path):
    """Run pipeline on 800 accounts and 5 rings, verifying ROC-AUC > 0.7 and artifact creation."""
    result = run_pipeline(
        accounts=800,
        rings=5,
        seed=5,
        out_dir=str(tmp_path),
    )

    metrics = result["metrics"]
    roc_auc = metrics.get("node_roc_auc", 0.0)

    # Supervised node ROC-AUC > 0.7
    assert roc_auc > 0.7, f"Expected supervised node ROC-AUC > 0.7, got {roc_auc:.4f}"

    # Verify expected artifact files exist
    expected_files = [
        tmp_path / "metrics.json",
        tmp_path / "top_communities.csv",
        tmp_path / "top_ring.png",
        tmp_path / "pr_curve.png",
    ]

    for file_path in expected_files:
        assert file_path.exists(), f"Expected artifact '{file_path.name}' to exist"
        assert file_path.stat().st_size > 0, f"Artifact '{file_path.name}' is empty"
