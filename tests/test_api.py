"""Tests for FastAPI REST endpoints and risk score queries."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from fraud_rings.pipeline import run_pipeline


def test_api_endpoints_with_tmp_artifacts(tmp_path: Path):
    """Verify health, top communities retrieval, and 404 for unknown accounts."""
    # Pre-generate artifacts in tmp directory
    run_pipeline(accounts=800, rings=5, seed=1, out_dir=str(tmp_path))

    app = create_app(artifacts_dir=str(tmp_path))

    with TestClient(app) as client:
        # 1. Health check returns 200
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json() == {"status": "ok"}

        # 2. Top communities non-empty
        comm_resp = client.get("/communities/top?limit=5")
        assert comm_resp.status_code == 200
        comm_data = comm_resp.json()
        assert isinstance(comm_data, list)
        assert len(comm_data) > 0
        first_comm = comm_data[0]
        assert "community_id" in first_comm
        assert "score" in first_comm
        assert "heuristic_signals" in first_comm
        assert len(first_comm["member_ids"]) > 0

        # 3. Known account returns risk assessment
        known_acc = first_comm["member_ids"][0]
        known_resp = client.get(f"/accounts/{known_acc}/risk")
        assert known_resp.status_code == 200
        acc_risk = known_resp.json()
        assert acc_risk["account_id"] == known_acc
        assert "node_probability" in acc_risk
        assert "community_id" in acc_risk
        assert "top_signals" in acc_risk

        # 4. Unknown account returns 404
        unknown_resp = client.get("/accounts/non_existent_account_99999/risk")
        assert unknown_resp.status_code == 404
        assert "not found" in unknown_resp.json()["detail"].lower()
