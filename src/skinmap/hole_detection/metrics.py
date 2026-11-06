"""Metrics for evaluating hole detection performance."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def compute_metrics(
    detector,
    ground_truth_holes: Optional[np.ndarray] = None,
    match_threshold: float = 1.5,
) -> Dict[str, float]:
    """Compute quantitative metrics for a fitted detector."""
    metrics: Dict[str, float] = {}

    detected = None
    if hasattr(detector, "hole_centers") and detector.hole_centers is not None:
        detected = np.asarray(detector.hole_centers)
    elif hasattr(detector, "filtered_holes") and detector.filtered_holes is not None:
        detected = np.asarray(detector.filtered_holes)
    else:
        detected = np.empty((0, 0))

    raw = None
    if hasattr(detector, "hole_samples") and detector.hole_samples is not None:
        raw = np.asarray(detector.hole_samples)
    elif hasattr(detector, "all_endpoints") and detector.all_endpoints is not None:
        raw = np.asarray(detector.all_endpoints)
    else:
        raw = np.empty((0, 0))

    metrics["n_detected_holes"] = len(detected)
    metrics["n_raw_endpoints"] = len(raw)
    if hasattr(detector, "acceptance_rate"):
        metrics["rejection_rate"] = 1.0 - float(detector.acceptance_rate)
    else:
        metrics["rejection_rate"] = (
            (len(raw) - len(detected)) / max(1, len(raw)) if len(raw) else 0.0
        )
    metrics["hellinger_distance"] = getattr(detector, "hellinger_distance", 0.0) or 0.0

    if len(detected) > 0:
        metrics["hole_field_mean"] = 0.0
        metrics["hole_field_max"] = 0.0
        metrics["hole_field_std"] = 0.0

        comp_vol = getattr(detector, "component_volumes", None)
        if comp_vol is not None and len(comp_vol):
            metrics["mean_volume_estimate"] = float(np.mean(comp_vol))
            metrics["max_volume_estimate"] = float(np.max(comp_vol))
            metrics["min_volume_estimate"] = float(np.min(comp_vol))

        if hasattr(detector, "X") and detector.X is not None and len(detector.X):
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(detector.X)
            distances, _ = nn.kneighbors(detected)
            distances = distances.ravel()
            metrics["mean_distance_to_data"] = float(np.mean(distances))
            metrics["min_distance_to_data"] = float(np.min(distances))
            metrics["max_distance_to_data"] = float(np.max(distances))

        if hasattr(detector, "_gaussian_pdf") and hasattr(detector, "_kde_pdf"):
            p_at = detector._gaussian_pdf(detected)
            q_at = detector._kde_pdf(detected)
            density_ratio = q_at / (p_at + 1e-10)
            metrics["mean_density_ratio"] = float(np.mean(density_ratio))
            metrics["median_density_ratio"] = float(np.median(density_ratio))
        else:
            metrics["mean_density_ratio"] = 0.0
            metrics["median_density_ratio"] = 0.0

        traj_lengths = None
        if (
            hasattr(detector, "trajectory_lengths")
            and detector.trajectory_lengths is not None
        ):
            traj_lengths = np.asarray(detector.trajectory_lengths)
        elif hasattr(detector, "gradient_trajectories"):
            traj_lengths = np.asarray(
                [
                    np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
                    for traj in detector.gradient_trajectories
                    if len(traj) > 1
                ]
            )
        if traj_lengths is not None and len(traj_lengths):
            metrics["mean_trajectory_length"] = float(np.mean(traj_lengths))
            metrics["median_trajectory_length"] = float(np.median(traj_lengths))
            metrics["max_trajectory_length"] = float(np.max(traj_lengths))
        else:
            metrics["mean_trajectory_length"] = 0.0
            metrics["median_trajectory_length"] = 0.0
            metrics["max_trajectory_length"] = 0.0
    else:
        metrics["hole_field_mean"] = 0.0
        metrics["hole_field_max"] = 0.0
        metrics["hole_field_std"] = 0.0
        metrics["mean_distance_to_data"] = 0.0
        metrics["min_distance_to_data"] = 0.0
        metrics["max_distance_to_data"] = 0.0
        metrics["mean_density_ratio"] = 0.0
        metrics["median_density_ratio"] = 0.0
        traj_lengths = None
        if (
            hasattr(detector, "trajectory_lengths")
            and detector.trajectory_lengths is not None
        ):
            traj_lengths = np.asarray(detector.trajectory_lengths)
        elif hasattr(detector, "gradient_trajectories"):
            traj_lengths = np.asarray(
                [
                    np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
                    for traj in detector.gradient_trajectories
                    if len(traj) > 1
                ]
            )
        if traj_lengths is not None and len(traj_lengths):
            metrics["mean_trajectory_length"] = float(np.mean(traj_lengths))
            metrics["median_trajectory_length"] = float(np.median(traj_lengths))
            metrics["max_trajectory_length"] = float(np.max(traj_lengths))
        else:
            metrics["mean_trajectory_length"] = 0.0
            metrics["median_trajectory_length"] = 0.0
            metrics["max_trajectory_length"] = 0.0

    if ground_truth_holes is not None and len(ground_truth_holes) > 0:
        gt_metrics = _spatial_metrics(detected, ground_truth_holes, match_threshold)
        metrics.update(gt_metrics)

    return metrics


def _spatial_metrics(
    detected: np.ndarray, ground_truth: np.ndarray, threshold: float
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    n_gt = len(ground_truth)
    n_detected = len(detected)

    if n_detected == 0:
        metrics.update(
            {
                "n_matched": 0,
                "n_missed": n_gt,
                "n_false_positives": 0,
                "recall": 0.0,
                "precision": 0.0,
                "f1_score": 0.0,
                "mean_localization_error": float("inf"),
                "median_localization_error": float("inf"),
                "max_localization_error": float("inf"),
                "std_localization_error": float("inf"),
            }
        )
        return metrics

    distances = cdist(detected, ground_truth)
    size = max(n_detected, n_gt)
    cost = np.full((size, size), threshold * 10.0)
    cost[:n_detected, :n_gt] = distances

    rows, cols = linear_sum_assignment(cost)

    matches = []
    errors = []
    for r, c in zip(rows, cols):
        if r < n_detected and c < n_gt and distances[r, c] < threshold:
            matches.append((r, c))
            errors.append(distances[r, c])

    n_matched = len(matches)
    n_missed = n_gt - n_matched
    n_false = n_detected - n_matched

    precision = n_matched / n_detected if n_detected else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    metrics["n_matched"] = n_matched
    metrics["n_missed"] = n_missed
    metrics["n_false_positives"] = n_false
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = f1

    if errors:
        metrics["mean_localization_error"] = float(np.mean(errors))
        metrics["median_localization_error"] = float(np.median(errors))
        metrics["max_localization_error"] = float(np.max(errors))
        metrics["std_localization_error"] = float(np.std(errors))
    else:
        metrics["mean_localization_error"] = float("inf")
        metrics["median_localization_error"] = float("inf")
        metrics["max_localization_error"] = float("inf")
        metrics["std_localization_error"] = float("inf")

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Create a compact human-readable summary string."""
    lines = ["Hole Detection Metrics", "-" * 24]
    lines.append(f"Detected holes: {metrics.get('n_detected_holes', 0)}")
    lines.append(f"Raw endpoints:  {metrics.get('n_raw_endpoints', 0)}")
    lines.append(f"Rejection rate: {metrics.get('rejection_rate', 0.0):.3f}")

    if "hellinger_distance" in metrics:
        lines.append(f"Hellinger distance: {metrics['hellinger_distance']:.4f}")

    if "precision" in metrics and "recall" in metrics:
        lines.append(
            f"Precision / Recall / F1: "
            f"{metrics['precision']:.3f} / {metrics['recall']:.3f} / {metrics.get('f1_score', 0.0):.3f}"
        )
        if np.isfinite(metrics.get("mean_localization_error", np.inf)):
            lines.append(
                f"Mean localization error: {metrics['mean_localization_error']:.4f}"
            )
    return "\n".join(lines)
