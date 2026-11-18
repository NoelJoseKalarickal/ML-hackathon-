"""Pydantic request and response schemas for PathSnap FastAPI backend (Laptop 1)."""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field


class CandidateRecord(BaseModel):
    """One (GPS observation, candidate road segment) row sent to the model API."""

    model_config = ConfigDict(extra="allow")

    trip_id: Union[str, int]
    obs_idx: int
    arc_id: int
    perp_dist_m: float
    heading_diff_deg: float
    cos_heading_diff: float
    speed_mps: float
    proj_ratio: float
    arc_length_m: float
    dist_rank: float
    dist_diff_from_min: float
    node_degree: float
    gps_bearing: float
    road_bearing: float


class PredictRequest(BaseModel):
    """Batch candidate scoring request for one or more trips."""

    candidates: list[CandidateRecord] = Field(..., min_length=1)


class ScoredCandidate(BaseModel):
    """Model probability output preserving exact (trip_id, obs_idx, arc_id) correspondence."""

    trip_id: str
    obs_idx: int
    arc_id: int
    pred_prob: float
    is_top1: int


class PredictResponse(BaseModel):
    """API response containing candidate probabilities and classifier-only Top-1 selections."""

    status: str = "ok"
    model_name: str
    trip_ids: list[str]
    num_candidates: int
    num_observations: int
    inference_time_ms: float
    predictions: list[ScoredCandidate]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    algorithm_family: str
    feature_columns: list[str]
    metadata: dict[str, Any]
