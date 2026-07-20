"""Bootstrap analysis utilities for uncertainty estimation."""

from typing import Callable, Dict, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    **metric_kwargs,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        metric_fn: Metric function that takes (y_true, y_pred, **kwargs)
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level for interval (default: 0.95)
        random_state: Random seed
        **metric_kwargs: Additional arguments for metric function

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_iterations):
        # Sample with replacement
        indices = rng.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute metric
        try:
            score = metric_fn(y_true_boot, y_pred_boot, **metric_kwargs)
            bootstrap_scores.append(score)
        except Exception as e:
            # Skip invalid bootstrap samples (e.g., only one class present)
            logger.debug(f"Skipping bootstrap iteration due to: {e}")
            continue

    if len(bootstrap_scores) == 0:
        logger.warning("No valid bootstrap samples, returning NaN confidence intervals")
        return np.nan, np.nan, np.nan

    bootstrap_scores = np.array(bootstrap_scores)
    mean_score = np.mean(bootstrap_scores)

    # Compute percentile-based confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    return mean_score, lower_bound, upper_bound


def compute_classification_metrics_with_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    random_state: int = 42,
    is_multilabel: bool = False,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute classification metrics with bootstrap confidence intervals.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_iterations: Number of bootstrap iterations
        random_state: Random seed
        is_multilabel: Whether this is multi-label classification

    Returns:
        Dictionary mapping metric names to (mean, lower_CI, upper_CI) tuples
    """
    metrics = {}

    if is_multilabel:
        # Multi-label classification metrics
        metric_configs = [
            ("hamming_loss", hamming_loss, {}),
            (
                "jaccard_micro",
                jaccard_score,
                {"average": "micro", "zero_division": 0},
            ),
            (
                "jaccard_macro",
                jaccard_score,
                {"average": "macro", "zero_division": 0},
            ),
            (
                "jaccard_samples",
                jaccard_score,
                {"average": "samples", "zero_division": 0},
            ),
            (
                "precision_macro",
                precision_score,
                {"average": "macro", "zero_division": 0},
            ),
            ("recall_macro", recall_score, {"average": "macro", "zero_division": 0}),
            ("f1_macro", f1_score, {"average": "macro", "zero_division": 0}),
        ]
    else:
        # Single-label classification metrics
        metric_configs = [
            ("accuracy", accuracy_score, {}),
            ("balanced_accuracy", balanced_accuracy_score, {}),
            (
                "precision_macro",
                precision_score,
                {"average": "macro", "zero_division": 0},
            ),
            ("recall_macro", recall_score, {"average": "macro", "zero_division": 0}),
            ("f1_macro", f1_score, {"average": "macro", "zero_division": 0}),
        ]

    for metric_name, metric_fn, metric_kwargs in metric_configs:
        mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(
            y_true,
            y_pred,
            metric_fn,
            n_iterations=n_iterations,
            random_state=random_state,
            **metric_kwargs,
        )
        metrics[metric_name] = (mean_val, lower_ci, upper_ci)

    return metrics


def compute_regression_metrics_with_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute regression metrics with bootstrap confidence intervals.

    Args:
        y_true: True values
        y_pred: Predicted values
        n_iterations: Number of bootstrap iterations
        random_state: Random seed

    Returns:
        Dictionary mapping metric names to (mean, lower_CI, upper_CI) tuples
    """
    metrics = {}

    metric_configs = [
        ("mae", mean_absolute_error, {}),
        ("rmse", lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)), {}),
        ("r2", r2_score, {}),
    ]

    for metric_name, metric_fn, metric_kwargs in metric_configs:
        mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(
            y_true,
            y_pred,
            metric_fn,
            n_iterations=n_iterations,
            random_state=random_state,
            **metric_kwargs,
        )
        metrics[metric_name] = (mean_val, lower_ci, upper_ci)

    return metrics


def format_metric_with_ci(
    metric_value: float, lower_ci: float, upper_ci: float, precision: int = 4
) -> str:
    """Format a metric value with confidence interval as a string.

    Args:
        metric_value: Point estimate of the metric
        lower_ci: Lower bound of confidence interval
        upper_ci: Upper bound of confidence interval
        precision: Number of decimal places

    Returns:
        Formatted string like "0.8523 (0.8301-0.8745)"
    """
    return f"{metric_value:.{precision}f} ({lower_ci:.{precision}f}-{upper_ci:.{precision}f})"
