"""PathSnap FastAPI Model Inference Server (Laptop 1).

Run from the repository root:
    uvicorn deployment.main:app --host 0.0.0.0 --port 8000

Endpoints:
  - GET  /health       : Server status, loaded model info, and expected feature schema.
  - GET  /model-info   : Full model metadata & training/tuning metrics.
  - POST /predict      : Score candidate road segments and return calibrated `pred_prob`
                         (also aliased at POST /score and POST /predict_candidates).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from deployment.schemas import (
    CandidateRecord,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ScoredCandidate,
)
from models.config import DEFAULT_ARTIFACTS_DIR
from models.inference import CandidateScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pathsnap.api")

_scorer: CandidateScorer | None = None


def get_scorer() -> CandidateScorer:
    global _scorer
    if _scorer is None:
        logger.info("Loading PathSnap model artifacts from %s ...", DEFAULT_ARTIFACTS_DIR)
        _scorer = CandidateScorer(artifacts_dir=DEFAULT_ARTIFACTS_DIR)
        logger.info(
            "Loaded model '%s' (%s) expecting %d features: %s",
            _scorer.metadata.get("model_display_name", "HistGradientBoosting"),
            _scorer.metadata.get("algorithm_family", "Boosting"),
            len(_scorer.feature_columns),
            _scorer.feature_columns,
        )
    return _scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_scorer()
    yield


app = FastAPI(
    title="PathSnap Map-Matching Model API",
    description="Distributed FastAPI inference server (Laptop 1) serving calibrated candidate probabilities for HMM/Viterbi route reconstruction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    scorer = get_scorer()
    client_host = request.client.host if request.client else "unknown"
    logger.info("[GET /health] client=%s | status=200 OK", client_host)
    return HealthResponse(
        status="ok",
        model_name=scorer.metadata.get("model_display_name", "HistGradientBoosting"),
        algorithm_family=scorer.metadata.get("algorithm_family", "Boosting"),
        feature_columns=scorer.feature_columns,
        metadata=scorer.metadata,
    )


@app.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    scorer = get_scorer()
    client_host = request.client.host if request.client else "unknown"
    logger.info("[GET /model-info] client=%s | status=200 OK", client_host)
    return {
        "feature_columns": scorer.feature_columns,
        "metadata": scorer.metadata,
    }


def _score_payload(
    payload: Union[PredictRequest, list[CandidateRecord]],
    request: Request,
    endpoint_name: str,
) -> PredictResponse:
    scorer = get_scorer()
    records = payload.candidates if isinstance(payload, PredictRequest) else payload
    if not records:
        raise HTTPException(status_code=400, detail="Empty candidate list provided.")

    df = pd.DataFrame([r.model_dump() for r in records])
    try:
        scored_df, latency_ms = scorer.score_candidates(df, strict_no_labels=True)
    except ValueError as exc:
        logger.warning("[%s] Rejected invalid request: %s", endpoint_name, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    trip_ids = sorted(scored_df["trip_id"].unique().tolist())
    n_candidates = int(len(scored_df))
    n_obs = int(scored_df.groupby(["trip_id", "obs_idx"], sort=False).ngroups)
    model_name = scorer.metadata.get("model_display_name", "HistGradientBoosting")
    client_host = request.client.host if request.client else "unknown"

    # Explicit evaluator-visible terminal log for every inference request
    logger.info(
        "[POST %s] ts=%s | client=%s | trips=%s | obs=%d | candidates=%d | latency=%.2f ms | status=200 OK",
        endpoint_name,
        datetime.now().strftime("%H:%M:%S"),
        client_host,
        trip_ids,
        n_obs,
        n_candidates,
        latency_ms,
    )

    preds = [
        ScoredCandidate(
            trip_id=str(row.trip_id),
            obs_idx=int(row.obs_idx),
            arc_id=int(row.arc_id),
            pred_prob=float(row.pred_prob),
            is_top1=int(row.is_top1),
        )
        for row in scored_df.itertuples(index=False)
    ]

    return PredictResponse(
        status="ok",
        model_name=model_name,
        trip_ids=trip_ids,
        num_candidates=n_candidates,
        num_observations=n_obs,
        inference_time_ms=round(latency_ms, 3),
        predictions=preds,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: Union[PredictRequest, list[CandidateRecord]],
    request: Request,
) -> PredictResponse:
    return _score_payload(payload, request, "/predict")


@app.post("/score", response_model=PredictResponse)
def score(
    payload: Union[PredictRequest, list[CandidateRecord]],
    request: Request,
) -> PredictResponse:
    return _score_payload(payload, request, "/score")


@app.post("/predict_candidates", response_model=PredictResponse)
def predict_candidates(
    payload: Union[PredictRequest, list[CandidateRecord]],
    request: Request,
) -> PredictResponse:
    return _score_payload(payload, request, "/predict_candidates")
