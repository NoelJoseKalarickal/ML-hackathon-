"""Comprehensive evaluation metrics for PathSnap candidate classification and route quality."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from itertools import groupby
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _dedup_consecutive(seq: list[int]) -> list[int]:
    """Collapse consecutive duplicate arc IDs while preserving traversal order."""
    return [k for k, _ in groupby(seq)]


def _fast_lcs_length(a: list[int], b: list[int]) -> int:
    """Fast O(K log N) Longest Common Subsequence length via patience sorting."""
    if not a or not b:
        return 0
    positions_in_b: dict[int, list[int]] = defaultdict(list)
    for idx, val in enumerate(b):
        positions_in_b[val].append(idx)

    tails: list[int] = []
    for val in a:
        pos_list = positions_in_b.get(val)
        if not pos_list:
            continue
        for pos in reversed(pos_list):
            i = bisect_left(tails, pos)
            if i == len(tails):
                tails.append(pos)
            else:
                tails[i] = pos
    return len(tails)


def compute_candidate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute candidate-level binary classification metrics and confusion matrix."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    has_both_classes = len(np.unique(y_true)) == 2
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "pos_precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "pos_recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "threshold": float(threshold),
    }


def extract_top1_predictions(
    df: pd.DataFrame,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """Select the highest-probability candidate per (trip_id, obs_idx) deterministically."""
    work = pd.DataFrame(
        {
            "trip_id": df["trip_id"].values,
            "obs_idx": df["obs_idx"].values,
            "arc_id": df["arc_id"].values,
            "pred_prob": np.asarray(y_prob, dtype=np.float64),
        }
    )
    if "label" in df.columns:
        work["label"] = df["label"].values

    work.sort_values(
        ["trip_id", "obs_idx", "pred_prob"],
        ascending=[True, True, False],
        kind="mergesort",
        inplace=True,
    )
    top1 = work.drop_duplicates(subset=["trip_id", "obs_idx"], keep="first").copy()
    top1.sort_values(["trip_id", "obs_idx"], kind="mergesort", inplace=True)
    top1.reset_index(drop=True, inplace=True)
    return top1


def build_scored_candidates_df(
    df: pd.DataFrame,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """Return a clean DataFrame preserving (trip_id, obs_idx, arc_id) with pred_prob and is_top1."""
    cols = ["trip_id", "obs_idx", "arc_id"]
    if "label" in df.columns:
        cols.append("label")
    out = df[cols].copy().reset_index(drop=True)
    out["pred_prob"] = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    # Mark highest-prob candidate per (trip_id, obs_idx) with deterministic first-max tie-break
    best_idx = out.groupby(["trip_id", "obs_idx"], sort=False)["pred_prob"].idxmax()
    out["is_top1"] = 0
    out.loc[best_idx.values, "is_top1"] = 1
    return out


def compute_route_continuity(
    top1_df: pd.DataFrame,
    arcs_by_trip: Optional[dict[str, dict[int, tuple[int, int]]]] = None,
) -> dict[str, Any]:
    """
    Compute road-network Route Continuity across consecutive predicted GPS observations.

    Primary (`route_continuity`):
      Fraction of consecutive transitions (t -> t+1) where:
        - pred_arc[t] == pred_arc[t+1], OR
        - to_node(pred_arc[t]) == from_node(pred_arc[t+1]) (strict directed adjacency).
    """
    if not arcs_by_trip:
        return {
            "route_continuity": None,
            "route_continuity_undirected": None,
            "route_continuity_within_2hops": None,
            "total_transitions": 0,
            "continuous_transitions_directed": 0,
            "trips_evaluated_for_continuity": 0,
            "note": "N/A (road network .arcs not provided)",
        }

    total_transitions = 0
    directed_ok = 0
    undirected_ok = 0
    twohop_ok = 0
    trips_evaluated = 0

    for tid, grp in top1_df.groupby("trip_id", sort=False):
        tid_str = str(tid)
        arc_map = arcs_by_trip.get(tid_str)
        if not arc_map:
            continue
        trips_evaluated += 1
        pred_arcs = grp.sort_values("obs_idx", kind="mergesort")["arc_id"].to_numpy(dtype=np.int64)
        if len(pred_arcs) < 2:
            continue

        out_adj: dict[int, set[int]] = defaultdict(set)
        for u, v in arc_map.values():
            out_adj[u].add(v)

        for i in range(len(pred_arcs) - 1):
            a1 = int(pred_arcs[i])
            a2 = int(pred_arcs[i + 1])
            total_transitions += 1
            if a1 == a2:
                directed_ok += 1
                undirected_ok += 1
                twohop_ok += 1
                continue

            ep1 = arc_map.get(a1)
            ep2 = arc_map.get(a2)
            if ep1 is None or ep2 is None:
                continue
            u1, v1 = ep1
            u2, v2 = ep2

            if v1 == u2:
                directed_ok += 1
                undirected_ok += 1
                twohop_ok += 1
            else:
                if (u1 == u2) or (u1 == v2) or (v1 == v2):
                    undirected_ok += 1
                if u2 in out_adj.get(v1, ()):
                    twohop_ok += 1

    if total_transitions == 0:
        return {
            "route_continuity": None,
            "route_continuity_undirected": None,
            "route_continuity_within_2hops": None,
            "total_transitions": 0,
            "continuous_transitions_directed": 0,
            "trips_evaluated_for_continuity": trips_evaluated,
            "note": "N/A (no matching trip .arcs transitions found)",
        }

    return {
        "route_continuity": float(directed_ok / total_transitions),
        "route_continuity_undirected": float(undirected_ok / total_transitions),
        "route_continuity_within_2hops": float(twohop_ok / total_transitions),
        "total_transitions": int(total_transitions),
        "continuous_transitions_directed": int(directed_ok),
        "trips_evaluated_for_continuity": int(trips_evaluated),
        "note": "Strict directed continuity (same arc or to_node == from_node)",
    }


def compute_sequence_metrics(
    df: pd.DataFrame,
    top1_df: pd.DataFrame,
    routes_by_trip: Optional[dict[str, list[int]]] = None,
    obs_meta: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Compute per-trip macro-averaged Arc-Set F1 and Order-Preserving LCS Sequence F1."""
    ref_by_trip: dict[str, list[int]] = {}
    if routes_by_trip:
        for tid, seq in routes_by_trip.items():
            ref_by_trip[str(tid)] = _dedup_consecutive(seq)
    elif obs_meta is not None and "true_arc_id" in obs_meta.columns:
        for tid, grp in obs_meta.groupby("trip_id", sort=False):
            valid = grp.sort_values("obs_idx", kind="mergesort")["true_arc_id"]
            valid = valid[valid >= 0].astype(int).tolist()
            if valid:
                ref_by_trip[str(tid)] = _dedup_consecutive(valid)
    elif "label" in df.columns:
        pos_df = df[df["label"] == 1].sort_values(["trip_id", "obs_idx"], kind="mergesort")
        for tid, grp in pos_df.groupby("trip_id", sort=False):
            ref_by_trip[str(tid)] = _dedup_consecutive(grp["arc_id"].astype(int).tolist())

    set_precs, set_recs, set_f1s = [], [], []
    lcs_precs, lcs_recs, lcs_f1s = [], [], []

    for tid, grp in top1_df.groupby("trip_id", sort=False):
        tid_str = str(tid)
        ref_seq = ref_by_trip.get(tid_str)
        if not ref_seq:
            continue
        pred_seq = _dedup_consecutive(
            grp.sort_values("obs_idx", kind="mergesort")["arc_id"].astype(int).tolist()
        )
        if not pred_seq:
            for lst in (set_precs, set_recs, set_f1s, lcs_precs, lcs_recs, lcs_f1s):
                lst.append(0.0)
            continue

        pred_set, ref_set = set(pred_seq), set(ref_seq)
        inter = len(pred_set & ref_set)
        sp, sr = inter / len(pred_set), inter / len(ref_set)
        set_precs.append(sp)
        set_recs.append(sr)
        set_f1s.append((2 * sp * sr / (sp + sr)) if (sp + sr) > 0 else 0.0)

        lcs_len = _fast_lcs_length(pred_seq, ref_seq)
        lp, lr = lcs_len / len(pred_seq), lcs_len / len(ref_seq)
        lcs_precs.append(lp)
        lcs_recs.append(lr)
        lcs_f1s.append((2 * lp * lr / (lp + lr)) if (lp + lr) > 0 else 0.0)

    if not set_f1s:
        return {
            "sequence_arc_set_precision": None,
            "sequence_arc_set_recall": None,
            "sequence_arc_set_f1": None,
            "sequence_lcs_precision": None,
            "sequence_lcs_recall": None,
            "sequence_lcs_f1": None,
            "trips_evaluated_for_sequence": 0,
        }

    return {
        "sequence_arc_set_precision": float(np.mean(set_precs)),
        "sequence_arc_set_recall": float(np.mean(set_recs)),
        "sequence_arc_set_f1": float(np.mean(set_f1s)),
        "sequence_lcs_precision": float(np.mean(lcs_precs)),
        "sequence_lcs_recall": float(np.mean(lcs_recs)),
        "sequence_lcs_f1": float(np.mean(lcs_f1s)),
        "trips_evaluated_for_sequence": int(len(set_f1s)),
    }


def evaluate_predictions(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    obs_meta: Optional[pd.DataFrame] = None,
    arcs_by_trip: Optional[dict[str, dict[int, tuple[int, int]]]] = None,
    routes_by_trip: Optional[dict[str, list[int]]] = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Full PathSnap evaluation suite returning structured metric dicts."""
    y_true = df["label"].to_numpy(dtype=np.int64)
    cand_metrics = compute_candidate_metrics(y_true, y_prob, threshold=threshold)
    top1_df = extract_top1_predictions(df, y_prob)

    obs_has_pos = df.groupby(["trip_id", "obs_idx"], sort=False)["label"].max()
    obs_in_candidates = int(len(obs_has_pos))
    cand_present_from_df = int((obs_has_pos == 1).sum())
    top1_hits = int((top1_df["label"] == 1).sum())

    if obs_meta is not None:
        subset_trips = set(df["trip_id"].unique())
        meta_sub = obs_meta[obs_meta["trip_id"].isin(subset_trips)]
        total_obs = int(len(meta_sub))
        if "candidate_found" in meta_sub.columns:
            cand_present = int((meta_sub["candidate_found"] == 1).sum())
        else:
            cand_present = cand_present_from_df
        obs_meta_used = True
    else:
        total_obs = obs_in_candidates
        cand_present = cand_present_from_df
        obs_meta_used = False

    miss_count = max(total_obs - cand_present, 0)
    obs_metrics = {
        "top1_accuracy": float(top1_hits / max(total_obs, 1)),
        "top1_accuracy_given_candidate": float(top1_hits / max(cand_present, 1)),
        "candidate_miss_rate": float(miss_count / max(total_obs, 1)),
        "total_observations": total_obs,
        "candidate_present_observations": cand_present,
        "candidate_miss_count": miss_count,
        "top1_hits": top1_hits,
        "obs_meta_supplied": obs_meta_used,
    }

    cont_metrics = compute_route_continuity(top1_df, arcs_by_trip=arcs_by_trip)
    seq_metrics = compute_sequence_metrics(
        df, top1_df, routes_by_trip=routes_by_trip, obs_meta=obs_meta
    )

    per_trip: dict[str, dict[str, Any]] = {}
    for tid, grp in top1_df.groupby("trip_id", sort=False):
        t_hits = int((grp["label"] == 1).sum())
        t_obs = int(len(grp))
        per_trip[str(tid)] = {
            "observations_scored": t_obs,
            "top1_hits": t_hits,
            "top1_accuracy_scored": float(t_hits / max(t_obs, 1)),
        }

    return {
        "candidate": cand_metrics,
        "observation": obs_metrics,
        "continuity": cont_metrics,
        "sequence": seq_metrics,
        "per_trip": per_trip,
    }
