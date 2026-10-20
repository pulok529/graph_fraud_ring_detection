"""FastAPI REST service for querying graph fraud ring predictions and diagnostics."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from fraud_rings.pipeline import run_pipeline


class HeuristicSignals(BaseModel):
    """Average graph signals across the members of a flagged cluster."""

    mean_cycle_count: float = Field(..., description="Average short cycles through members")
    mean_account_age: float = Field(..., description="Average member account age in days")
    mean_device_sharing: float = Field(..., description="Average device sharing degree")
    mean_amount_cv: float = Field(..., description="Average coefficient of variation of amounts")
    mean_reciprocity: float = Field(..., description="Average share of reciprocated transfers")


class CommunityResponse(BaseModel):
    """Ranked suspicious community entry."""

    community_id: int
    rank: int
    score: float
    size: int
    member_ids: list[str]
    heuristic_signals: HeuristicSignals


class NodeRiskResponse(BaseModel):
    """Detailed risk assessment for a single financial account."""

    account_id: str
    node_probability: float
    community_id: int
    community_score: float
    top_signals: dict[str, Any]


def _ensure_artifacts_loaded(app: FastAPI, artifacts_path: Path) -> None:
    """Load model artifacts from disk or run pipeline with 1500 accounts if missing."""
    req_files = [
        artifacts_path / "top_communities.csv",
        artifacts_path / "node_preds.csv",
        artifacts_path / "features.csv",
    ]

    if not all(f.exists() for f in req_files):
        # Auto-initialize artifacts using 1500 accounts
        run_pipeline(accounts=1500, rings=8, seed=42, out_dir=str(artifacts_path))

    # Read community rankings
    top_comm_df = pd.read_csv(artifacts_path / "top_communities.csv")
    node_preds_df = pd.read_csv(artifacts_path / "node_preds.csv")
    features_df = pd.read_csv(artifacts_path / "features.csv")

    # Format community objects
    formatted_comms: list[dict[str, Any]] = []
    comm_score_map: dict[int, float] = {}
    member_cluster: dict[str, int] = {}

    for _, row in top_comm_df.iterrows():
        c_id = int(row["community_id"])
        raw_m = str(row["member_ids"])
        member_ids = raw_m.split(";") if ";" in raw_m else [raw_m]
        score = float(row.get("mean_prob", row.get("score", 0.0)))
        comm_score_map[c_id] = score

        member_feats = features_df[features_df["account_id"].astype(str).isin(member_ids)]
        h_sig = {
            "mean_cycle_count": round(float(member_feats["cycle_count"].mean()), 4),
            "mean_account_age": round(float(member_feats["account_age"].mean()), 2),
            "mean_device_sharing": round(float(member_feats["shared_device_count"].mean()), 4),
            "mean_amount_cv": round(float(member_feats["amount_similarity"].mean()), 4),
            "mean_reciprocity": round(float(member_feats["reciprocity"].mean()), 4),
        }
        for m in member_ids:
            member_cluster[m] = c_id

        formatted_comms.append(
            {
                "community_id": c_id,
                "rank": int(row.get("rank", len(formatted_comms) + 1)),
                "score": float(round(score, 4)),
                "size": int(row["size"]),
                "member_ids": member_ids,
                "heuristic_signals": h_sig,
            }
        )

    # Fast node lookup dictionaries
    node_preds_map = node_preds_df.set_index("account_id").to_dict(orient="index")
    features_map = features_df.set_index("account_id").to_dict(orient="index")

    # Store on app.state
    app.state.communities = formatted_comms
    app.state.comm_scores = comm_score_map
    app.state.member_cluster = member_cluster
    app.state.node_preds = node_preds_map
    app.state.features = features_map


def create_app(artifacts_dir: str | Path | None = None) -> FastAPI:
    """Application factory for the fraud detection service.

    Parameters
    ----------
    artifacts_dir : str or Path, optional
        Path to stored pipeline artifacts. Defaults to environment variable
        `ARTIFACTS_DIR` or local directory 'artifacts'.

    Returns
    -------
    FastAPI
        Configured FastAPI application instance.
    """
    if artifacts_dir is None:
        raw_dir = os.environ.get("ARTIFACTS_DIR", "artifacts")
    else:
        raw_dir = str(artifacts_dir)

    target_artifacts_path = Path(raw_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _ensure_artifacts_loaded(app, target_artifacts_path)
        yield

    app = FastAPI(
        title="Graph Fraud Ring Detection API",
        version="1.0.0",
        description="Detects synthetic fraud rings with graph heuristics and gradient boosting.",
        lifespan=lifespan,
    )

    # Pre-populate state for testing environments that bypass lifespan
    try:
        _ensure_artifacts_loaded(app, target_artifacts_path)
    except Exception:
        pass

    @app.get("/health", tags=["System"])
    def health() -> dict[str, str]:
        """Health check endpoint confirming service status."""
        return {"status": "ok"}

    @app.get("/communities/top", response_model=list[CommunityResponse], tags=["Detection"])
    def get_top_communities(limit: int = Query(default=10, ge=1, le=100)) -> list[dict[str, Any]]:
        """Retrieve top suspicious communities ranked by risk score with heuristic signals."""
        comms: list[dict[str, Any]] = getattr(app.state, "communities", [])
        return comms[:limit]

    @app.get("/accounts/{account_id}/risk", response_model=NodeRiskResponse, tags=["Detection"])
    def get_account_risk(account_id: str) -> dict[str, Any]:
        """Retrieve risk prediction, community context, and top graph signals for an account."""
        node_preds: dict[str, dict[str, Any]] = getattr(app.state, "node_preds", {})
        features: dict[str, dict[str, Any]] = getattr(app.state, "features", {})
        comm_scores: dict[int, float] = getattr(app.state, "comm_scores", {})
        member_cluster: dict[str, int] = getattr(app.state, "member_cluster", {})

        if account_id not in node_preds and account_id not in features:
            raise HTTPException(
                status_code=404,
                detail=f"Account '{account_id}' not found in registered records.",
            )

        pred_info = node_preds.get(account_id, {})
        feat_info = features.get(account_id, {})

        node_prob = float(pred_info.get("pred_prob", 0.0))
        # -1 means the account is not part of any flagged cluster
        comm_id = int(member_cluster.get(account_id, -1))
        comm_score = float(comm_scores.get(comm_id, 0.0))

        top_signals = {
            "cycle_count": int(feat_info.get("cycle_count", 0)),
            "shared_device_count": int(feat_info.get("shared_device_count", 0)),
            "amount_similarity": round(float(feat_info.get("amount_similarity", 0.0)), 4),
            "reciprocity": round(float(feat_info.get("reciprocity", 0.0)), 4),
            "account_age": int(feat_info.get("account_age", 90)),
        }

        return {
            "account_id": account_id,
            "node_probability": node_prob,
            "community_id": comm_id,
            "community_score": comm_score,
            "top_signals": top_signals,
        }

    return app


app = create_app()
