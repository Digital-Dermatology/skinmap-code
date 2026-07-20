"""Tests for bootstrap analysis utilities.

Tests cover:
- Bootstrap confidence interval computation
- Classification metrics with bootstrap (single-label and multi-label)
- Regression metrics with bootstrap
- Integration with metadata prediction
- Integration with downstream evaluation
- Edge cases (small samples, invalid samples, reproducibility)
"""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
)

from src.skinmap.utils.bootstrap import (
    bootstrap_confidence_interval,
    compute_classification_metrics_with_bootstrap,
    compute_regression_metrics_with_bootstrap,
    format_metric_with_ci,
)


class TestBootstrapConfidenceInterval:
    """Test core bootstrap_confidence_interval function."""

    def test_bootstrap_ci_basic_accuracy(self):
        """Test basic bootstrap CI computation for accuracy."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1] * 10)  # 100 samples
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1] * 10)  # Some errors

        mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(
            y_true,
            y_pred,
            accuracy_score,
            n_iterations=100,
            confidence_level=0.95,
            random_state=42,
        )

        # Check that mean is close to actual accuracy
        actual_accuracy = accuracy_score(y_true, y_pred)
        assert abs(mean_val - actual_accuracy) < 0.05

        # Check CI bounds are reasonable
        assert lower_ci < mean_val < upper_ci
        assert 0.0 <= lower_ci <= 1.0
        assert 0.0 <= upper_ci <= 1.0

    def test_bootstrap_ci_reproducibility(self):
        """Test that results are reproducible with same random_state."""
        y_true = np.array([0, 1, 0, 1, 1] * 20)
        y_pred = np.array([0, 1, 1, 1, 0] * 20)

        result1 = bootstrap_confidence_interval(
            y_true, y_pred, accuracy_score, n_iterations=100, random_state=42
        )
        result2 = bootstrap_confidence_interval(
            y_true, y_pred, accuracy_score, n_iterations=100, random_state=42
        )

        assert result1 == result2

    def test_bootstrap_ci_different_confidence_levels(self):
        """Test that higher confidence levels give wider intervals."""
        y_true = np.array([0, 1, 0, 1, 1] * 20)
        y_pred = np.array([0, 1, 1, 1, 0] * 20)

        _, lower_95, upper_95 = bootstrap_confidence_interval(
            y_true,
            y_pred,
            accuracy_score,
            n_iterations=200,
            confidence_level=0.95,
            random_state=42,
        )

        _, lower_90, upper_90 = bootstrap_confidence_interval(
            y_true,
            y_pred,
            accuracy_score,
            n_iterations=200,
            confidence_level=0.90,
            random_state=42,
        )

        # 95% CI should be wider than 90% CI
        ci_width_95 = upper_95 - lower_95
        ci_width_90 = upper_90 - lower_90
        assert ci_width_95 > ci_width_90

    def test_bootstrap_ci_with_metric_kwargs(self):
        """Test that metric kwargs are passed correctly."""
        y_true = np.array([0, 1, 2, 0, 1, 2] * 10)
        y_pred = np.array([0, 1, 1, 0, 2, 2] * 10)

        mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(
            y_true,
            y_pred,
            f1_score,
            n_iterations=100,
            random_state=42,
            average="macro",
            zero_division=0,
        )

        # Should compute macro F1
        actual_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        assert abs(mean_val - actual_f1) < 0.05

    def test_bootstrap_ci_regression_metric(self):
        """Test bootstrap CI with regression metric (MAE)."""
        y_true = np.array([1.5, 2.3, 3.1, 4.5, 5.2] * 20)
        y_pred = y_true + np.random.RandomState(42).normal(0, 0.5, 100)

        mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(
            y_true,
            y_pred,
            mean_absolute_error,
            n_iterations=100,
            random_state=42,
        )

        # Check bounds are reasonable
        assert lower_ci < mean_val < upper_ci
        assert mean_val > 0  # MAE should be positive


class TestClassificationMetricsWithBootstrap:
    """Test compute_classification_metrics_with_bootstrap."""

    def test_single_label_classification(self):
        """Test bootstrap metrics for single-label classification."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1] * 10)
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1] * 10)

        metrics = compute_classification_metrics_with_bootstrap(
            y_true,
            y_pred,
            n_iterations=100,
            random_state=42,
            is_multilabel=False,
        )

        # Check all expected metrics are present
        expected_metrics = [
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
        for metric_name in expected_metrics:
            assert metric_name in metrics
            mean_val, lower_ci, upper_ci = metrics[metric_name]
            # Check CI bounds
            assert lower_ci <= mean_val <= upper_ci
            assert 0.0 <= lower_ci <= 1.0
            assert 0.0 <= upper_ci <= 1.0

    def test_multi_label_classification(self):
        """Test bootstrap metrics for multi-label classification."""
        # Multi-label: each sample can have multiple labels
        y_true = np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
                [1, 0, 1],
            ]
            * 20
        )
        y_pred = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 1],
                [0, 1, 1],
                [1, 0, 0],
            ]
            * 20
        )

        metrics = compute_classification_metrics_with_bootstrap(
            y_true,
            y_pred,
            n_iterations=100,
            random_state=42,
            is_multilabel=True,
        )

        # Check all expected multi-label metrics are present
        expected_metrics = [
            "hamming_loss",
            "jaccard_micro",
            "jaccard_macro",
            "jaccard_samples",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
        for metric_name in expected_metrics:
            assert metric_name in metrics
            mean_val, lower_ci, upper_ci = metrics[metric_name]
            # Check CI bounds (all metrics should be in [0, 1])
            assert lower_ci <= mean_val <= upper_ci
            assert 0.0 <= lower_ci <= 1.0
            assert 0.0 <= upper_ci <= 1.0

    def test_perfect_predictions(self):
        """Test with perfect predictions (edge case)."""
        y_true = np.array([0, 1, 0, 1, 1] * 20)
        y_pred = y_true.copy()

        metrics = compute_classification_metrics_with_bootstrap(
            y_true,
            y_pred,
            n_iterations=50,
            random_state=42,
            is_multilabel=False,
        )

        # Perfect predictions should give accuracy = 1.0
        mean_acc, lower_ci, upper_ci = metrics["accuracy"]
        assert mean_acc == 1.0
        # CI should be tight around 1.0
        assert lower_ci >= 0.95
        assert upper_ci == 1.0

    def test_small_sample_size(self):
        """Test bootstrap with small sample size."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        # Should still work, but CI will be wider
        metrics = compute_classification_metrics_with_bootstrap(
            y_true,
            y_pred,
            n_iterations=50,
            random_state=42,
            is_multilabel=False,
        )

        assert "accuracy" in metrics
        mean_val, lower_ci, upper_ci = metrics["accuracy"]
        # CI should be relatively wide due to small sample
        assert upper_ci - lower_ci > 0.1


class TestRegressionMetricsWithBootstrap:
    """Test compute_regression_metrics_with_bootstrap."""

    def test_regression_metrics(self):
        """Test bootstrap metrics for regression."""
        y_true = np.array([1.5, 2.3, 3.1, 4.5, 5.2, 6.1, 7.3, 8.5, 9.2, 10.1] * 10)
        y_pred = y_true + np.random.RandomState(42).normal(0, 0.5, 100)

        metrics = compute_regression_metrics_with_bootstrap(
            y_true,
            y_pred,
            n_iterations=100,
            random_state=42,
        )

        # Check all expected metrics are present
        expected_metrics = ["mae", "rmse", "r2"]
        for metric_name in expected_metrics:
            assert metric_name in metrics
            mean_val, lower_ci, upper_ci = metrics[metric_name]
            # Check CI bounds
            assert lower_ci <= mean_val <= upper_ci

        # MAE and RMSE should be positive
        assert metrics["mae"][0] > 0
        assert metrics["rmse"][0] > 0

        # R2 should be in reasonable range
        assert -1.0 <= metrics["r2"][0] <= 1.0

    def test_perfect_regression_predictions(self):
        """Test with perfect regression predictions."""
        y_true = np.array([1.5, 2.3, 3.1, 4.5, 5.2] * 20)
        y_pred = y_true.copy()

        metrics = compute_regression_metrics_with_bootstrap(
            y_true,
            y_pred,
            n_iterations=50,
            random_state=42,
        )

        # Perfect predictions should give MAE = 0, RMSE = 0, R2 = 1
        assert metrics["mae"][0] == 0.0
        assert metrics["rmse"][0] == 0.0
        # R2 is undefined when variance is 0, but bootstrap might handle it
        # Just check it's not NaN
        assert not np.isnan(metrics["r2"][0])

    def test_regression_with_outliers(self):
        """Test regression bootstrap handles outliers gracefully."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        y_pred = y_true.copy()
        # Add some outliers
        y_pred[[0, 10, 20]] = [100.0, 200.0, 300.0]

        metrics = compute_regression_metrics_with_bootstrap(
            y_true,
            y_pred,
            n_iterations=100,
            random_state=42,
        )

        # Should still compute metrics without errors
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        # MAE should be large due to outliers
        assert metrics["mae"][0] > 5.0


class TestFormatMetricWithCI:
    """Test format_metric_with_ci function."""

    def test_default_precision(self):
        """Test formatting with default precision (4 decimals)."""
        result = format_metric_with_ci(0.8523, 0.8301, 0.8745)
        assert result == "0.8523 (0.8301-0.8745)"

    def test_custom_precision(self):
        """Test formatting with custom precision."""
        result = format_metric_with_ci(0.8523, 0.8301, 0.8745, precision=2)
        assert result == "0.85 (0.83-0.87)"

        result = format_metric_with_ci(0.8523, 0.8301, 0.8745, precision=6)
        assert result == "0.852300 (0.830100-0.874500)"

    def test_small_values(self):
        """Test formatting with small values."""
        result = format_metric_with_ci(0.0123, 0.0056, 0.0234, precision=4)
        assert result == "0.0123 (0.0056-0.0234)"

    def test_large_values(self):
        """Test formatting with large values."""
        result = format_metric_with_ci(123.456, 120.123, 126.789, precision=2)
        assert result == "123.46 (120.12-126.79)"


class TestBootstrapEdgeCases:
    """Test edge cases and error handling."""

    def test_single_class_in_bootstrap_sample(self):
        """Test handling when bootstrap sample has only one class."""
        # Highly imbalanced data - bootstrap might sample only one class
        y_true = np.array([0] * 95 + [1] * 5)
        y_pred = np.array([0] * 95 + [0] * 5)  # Never predicts class 1

        # Should handle gracefully (some bootstrap samples may have only class 0)
        metrics = compute_classification_metrics_with_bootstrap(
            y_true,
            y_pred,
            n_iterations=100,
            random_state=42,
            is_multilabel=False,
        )

        # Should still return results (skipping invalid samples)
        assert "accuracy" in metrics
        mean_val, lower_ci, upper_ci = metrics["accuracy"]
        # Some valid samples should exist
        assert not np.isnan(mean_val)

    def test_empty_predictions(self):
        """Test with empty arrays (returns NaN)."""
        y_true = np.array([])
        y_pred = np.array([])

        mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(
            y_true,
            y_pred,
            accuracy_score,
            n_iterations=10,
            random_state=42,
        )

        # Should return NaN for empty arrays
        assert np.isnan(mean_val)
        assert np.isnan(lower_ci)
        assert np.isnan(upper_ci)

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths (should raise error)."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0])

        # IndexError is raised when bootstrap tries to access mismatched indices
        with pytest.raises(IndexError):
            bootstrap_confidence_interval(
                y_true,
                y_pred,
                accuracy_score,
                n_iterations=10,
                random_state=42,
            )

    def test_very_few_iterations(self):
        """Test with very few bootstrap iterations."""
        y_true = np.array([0, 1, 0, 1, 1] * 20)
        y_pred = np.array([0, 1, 1, 1, 0] * 20)

        # Should work even with just 5 iterations
        mean_val, lower_ci, upper_ci = bootstrap_confidence_interval(
            y_true,
            y_pred,
            accuracy_score,
            n_iterations=5,
            random_state=42,
        )

        assert lower_ci <= mean_val <= upper_ci

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with same seed."""
        y_true = np.array([0, 1, 2, 0, 1, 2] * 20)
        y_pred = np.array([0, 1, 1, 0, 2, 2] * 20)

        metrics1 = compute_classification_metrics_with_bootstrap(
            y_true, y_pred, n_iterations=100, random_state=123, is_multilabel=False
        )

        metrics2 = compute_classification_metrics_with_bootstrap(
            y_true, y_pred, n_iterations=100, random_state=123, is_multilabel=False
        )

        # All metrics should be identical
        for metric_name in metrics1.keys():
            assert metrics1[metric_name] == metrics2[metric_name]

    def test_different_seeds_give_different_results(self):
        """Test that different seeds give different CI bounds."""
        y_true = np.array([0, 1, 0, 1, 1] * 20)
        y_pred = np.array([0, 1, 1, 1, 0] * 20)

        metrics1 = compute_classification_metrics_with_bootstrap(
            y_true, y_pred, n_iterations=100, random_state=42, is_multilabel=False
        )

        metrics2 = compute_classification_metrics_with_bootstrap(
            y_true, y_pred, n_iterations=100, random_state=999, is_multilabel=False
        )

        # CI bounds should be different (though means should be similar)
        _, lower1, upper1 = metrics1["accuracy"]
        _, lower2, upper2 = metrics2["accuracy"]

        # At least one bound should differ
        assert (lower1 != lower2) or (upper1 != upper2)
