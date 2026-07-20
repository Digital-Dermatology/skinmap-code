"""Integration tests for bootstrap analysis with metadata prediction and downstream evaluation.

Tests cover:
- Metadata prediction with bootstrap enabled
- Downstream evaluation with bootstrap enabled
- CSV output includes confidence interval columns
- Bootstrap parameters are passed correctly through the pipeline
- Results are reproducible
"""

import numpy as np
import pandas as pd
import pytest

from src.skinmap.utils.metadata import predict_missing_metadata


class TestMetadataPredictionWithBootstrap:
    """Test predict_missing_metadata with bootstrap enabled."""

    @pytest.fixture
    def sample_data(self):
        """Create sample embeddings and metadata for testing."""
        np.random.seed(42)
        n_samples = 100
        embeddings = np.random.randn(n_samples, 128)

        df = pd.DataFrame(
            {
                "img_path": [f"img_{i}.jpg" for i in range(n_samples)],
                # Single-label classification: 60 known, 40 missing
                "laterality": (["left"] * 30 + ["right"] * 30 + [None] * 40),
                # Regression: 70 known, 30 missing
                "age": (
                    list(range(20, 60, 1))[:40]
                    + list(range(30, 60, 1))[:30]
                    + [None] * 30
                ),
                # Multi-class: 50 known, 50 missing
                "region": (
                    ["head"] * 15 + ["trunk"] * 15 + ["limbs"] * 20 + [None] * 50
                ),
            }
        )

        return embeddings, df

    def test_single_label_classification_with_bootstrap(self, sample_data):
        """Test single-label classification produces CI columns."""
        embeddings, df = sample_data

        df_out, metrics_df, _ = predict_missing_metadata(
            embeddings,
            df,
            attributes=["laterality"],
            test_size=0.2,
            random_state=42,
            bootstrap_n_iterations=100,
        )

        # Check metrics dataframe has CI columns
        assert "laterality" in metrics_df["attribute"].values
        laterality_metrics = metrics_df[metrics_df["attribute"] == "laterality"].iloc[0]

        # Check all expected metrics and their CI bounds exist
        assert "accuracy" in laterality_metrics.index
        assert "accuracy_ci_lower" in laterality_metrics.index
        assert "accuracy_ci_upper" in laterality_metrics.index
        assert "balanced_accuracy" in laterality_metrics.index
        assert "balanced_accuracy_ci_lower" in laterality_metrics.index
        assert "balanced_accuracy_ci_upper" in laterality_metrics.index
        assert "f1_macro" in laterality_metrics.index
        assert "f1_macro_ci_lower" in laterality_metrics.index
        assert "f1_macro_ci_upper" in laterality_metrics.index

        # Check CI bounds are reasonable
        assert (
            laterality_metrics["accuracy_ci_lower"]
            <= laterality_metrics["accuracy"]
            <= laterality_metrics["accuracy_ci_upper"]
        )
        assert 0.0 <= laterality_metrics["accuracy_ci_lower"] <= 1.0
        assert 0.0 <= laterality_metrics["accuracy_ci_upper"] <= 1.0

    def test_regression_with_bootstrap(self, sample_data):
        """Test regression produces CI columns."""
        embeddings, df = sample_data

        df_out, metrics_df, _ = predict_missing_metadata(
            embeddings,
            df,
            attributes=["age"],
            test_size=0.2,
            random_state=42,
            bootstrap_n_iterations=100,
        )

        # Check metrics dataframe has CI columns
        assert "age" in metrics_df["attribute"].values
        age_metrics = metrics_df[metrics_df["attribute"] == "age"].iloc[0]

        # Check all expected metrics and their CI bounds exist
        assert "mae" in age_metrics.index
        assert "mae_ci_lower" in age_metrics.index
        assert "mae_ci_upper" in age_metrics.index
        assert "rmse" in age_metrics.index
        assert "rmse_ci_lower" in age_metrics.index
        assert "rmse_ci_upper" in age_metrics.index
        assert "r2" in age_metrics.index
        assert "r2_ci_lower" in age_metrics.index
        assert "r2_ci_upper" in age_metrics.index

        # Check CI bounds are reasonable
        assert (
            age_metrics["mae_ci_lower"]
            <= age_metrics["mae"]
            <= age_metrics["mae_ci_upper"]
        )
        assert age_metrics["mae"] >= 0  # MAE should be non-negative

    def test_multiple_attributes_with_bootstrap(self, sample_data):
        """Test multiple attributes all get bootstrap CIs."""
        embeddings, df = sample_data

        df_out, metrics_df, _ = predict_missing_metadata(
            embeddings,
            df,
            attributes=["laterality", "age", "region"],
            test_size=0.2,
            random_state=42,
            bootstrap_n_iterations=50,  # Fewer iterations for speed
        )

        # Check all attributes are in metrics
        assert len(metrics_df) == 3
        assert set(metrics_df["attribute"].values) == {"laterality", "age", "region"}

        # Check each has CI columns
        for attr in ["laterality", "region"]:  # Classification
            attr_metrics = metrics_df[metrics_df["attribute"] == attr].iloc[0]
            assert "accuracy_ci_lower" in attr_metrics.index
            assert "accuracy_ci_upper" in attr_metrics.index

        # Check age (regression) has different CI columns
        age_metrics = metrics_df[metrics_df["attribute"] == "age"].iloc[0]
        assert "mae_ci_lower" in age_metrics.index
        assert "mae_ci_upper" in age_metrics.index

    def test_bootstrap_disabled_no_ci_columns(self, sample_data):
        """Test that without bootstrap, no CI columns are added."""
        embeddings, df = sample_data

        df_out, metrics_df, _ = predict_missing_metadata(
            embeddings,
            df,
            attributes=["laterality"],
            test_size=0.2,
            random_state=42,
            bootstrap_n_iterations=0,  # Disabled
        )

        # Check metrics dataframe does NOT have CI columns
        laterality_metrics = metrics_df[metrics_df["attribute"] == "laterality"].iloc[0]

        assert "accuracy" in laterality_metrics.index
        assert "accuracy_ci_lower" not in laterality_metrics.index
        assert "accuracy_ci_upper" not in laterality_metrics.index

    def test_bootstrap_reproducibility(self, sample_data):
        """Test that bootstrap results are reproducible with same seed."""
        embeddings, df = sample_data

        _, metrics_df1, _ = predict_missing_metadata(
            embeddings,
            df.copy(),
            attributes=["laterality"],
            test_size=0.2,
            random_state=42,
            bootstrap_n_iterations=100,
        )

        _, metrics_df2, _ = predict_missing_metadata(
            embeddings,
            df.copy(),
            attributes=["laterality"],
            test_size=0.2,
            random_state=42,
            bootstrap_n_iterations=100,
        )

        # Results should be identical
        pd.testing.assert_frame_equal(metrics_df1, metrics_df2)

    def test_predictions_unaffected_by_bootstrap(self, sample_data):
        """Test that predictions are the same with/without bootstrap."""
        embeddings, df = sample_data

        df_out1, _, _ = predict_missing_metadata(
            embeddings,
            df.copy(),
            attributes=["laterality"],
            test_size=0.2,
            random_state=42,
            bootstrap_n_iterations=0,  # No bootstrap
        )

        df_out2, _, _ = predict_missing_metadata(
            embeddings,
            df.copy(),
            attributes=["laterality"],
            test_size=0.2,
            random_state=42,
            bootstrap_n_iterations=100,  # With bootstrap
        )

        # Predictions should be identical (bootstrap only affects metrics)
        pd.testing.assert_frame_equal(
            df_out1[["img_path", "laterality_pred"]],
            df_out2[["img_path", "laterality_pred"]],
        )

    def test_ci_width_increases_with_fewer_samples(self, sample_data):
        """Test that CI width increases when test set is smaller (higher uncertainty)."""
        embeddings, df = sample_data

        # Large test set (more samples = tighter CI)
        _, metrics_large, _ = predict_missing_metadata(
            embeddings,
            df.copy(),
            attributes=["laterality"],
            test_size=0.5,  # 50% test
            random_state=42,
            bootstrap_n_iterations=100,
        )

        # Small test set (fewer samples = wider CI)
        _, metrics_small, _ = predict_missing_metadata(
            embeddings,
            df.copy(),
            attributes=["laterality"],
            test_size=0.1,  # 10% test
            random_state=42,
            bootstrap_n_iterations=100,
        )

        large_metrics = metrics_large.iloc[0]
        small_metrics = metrics_small.iloc[0]

        ci_width_large = (
            large_metrics["accuracy_ci_upper"] - large_metrics["accuracy_ci_lower"]
        )
        ci_width_small = (
            small_metrics["accuracy_ci_upper"] - small_metrics["accuracy_ci_lower"]
        )

        # Smaller test set should have wider CI (more uncertainty)
        assert ci_width_small > ci_width_large


class TestDownstreamEvaluationWithBootstrap:
    """Test downstream evaluation functions with bootstrap."""

    @pytest.fixture
    def mock_dataset_with_metadata(self):
        """Create a mock dataset with metadata for testing."""
        from unittest.mock import Mock

        dataset = Mock()

        # Create metadata
        n_samples = 100
        meta_data = pd.DataFrame(
            {
                "gender": ["male"] * 50 + ["female"] * 50,
                "region": ["head"] * 30 + ["trunk"] * 40 + ["limbs"] * 30,
                "age": np.random.randint(18, 80, n_samples).astype(float),
            }
        )
        dataset.meta_data = meta_data

        return dataset

    def test_classification_task_with_bootstrap(self, mock_dataset_with_metadata):
        """Test classification evaluation with bootstrap produces CI columns."""
        from src.skinmap.evaluation.downstream import _evaluate_classification_task

        # Create sample embeddings
        X = np.random.randn(100, 128)

        results, confusion_matrices, label_encoder = _evaluate_classification_task(
            X,
            mock_dataset_with_metadata,
            col="gender",
            test_size=0.2,
            random_state=42,
            classifier_names=["linear"],
            learned_metric_L=None,
            return_results=True,
            use_random_baseline=False,
            bootstrap_n_iterations=50,
        )

        # Check results have CI columns
        assert "linear" in results
        linear_results = results["linear"]

        # Check all expected metrics and their CI bounds exist
        assert "accuracy" in linear_results
        assert "accuracy_ci_lower" in linear_results
        assert "accuracy_ci_upper" in linear_results
        assert "balanced_accuracy" in linear_results
        assert "balanced_accuracy_ci_lower" in linear_results
        assert "balanced_accuracy_ci_upper" in linear_results

        # Check CI bounds are reasonable
        assert (
            linear_results["accuracy_ci_lower"]
            <= linear_results["accuracy"]
            <= linear_results["accuracy_ci_upper"]
        )

    def test_regression_task_with_bootstrap(self, mock_dataset_with_metadata):
        """Test regression evaluation with bootstrap produces CI columns."""
        from src.skinmap.evaluation.downstream import _evaluate_regression_task

        # Create sample embeddings
        X = np.random.randn(100, 128)

        results = _evaluate_regression_task(
            X,
            mock_dataset_with_metadata,
            col="age",
            test_size=0.2,
            random_state=42,
            regressor_names=["linear"],
            return_results=True,
            use_random_baseline=False,
            bootstrap_n_iterations=50,
        )

        # Check results have CI columns
        assert "linear" in results
        linear_results = results["linear"]

        # Check all expected metrics and their CI bounds exist
        assert "mae" in linear_results
        assert "mae_ci_lower" in linear_results
        assert "mae_ci_upper" in linear_results
        assert "rmse" in linear_results
        assert "rmse_ci_lower" in linear_results
        assert "rmse_ci_upper" in linear_results

        # Check CI bounds are reasonable
        assert (
            linear_results["mae_ci_lower"]
            <= linear_results["mae"]
            <= linear_results["mae_ci_upper"]
        )

    def test_multiple_classifiers_with_bootstrap(self, mock_dataset_with_metadata):
        """Test that all classifiers get bootstrap CIs."""
        from src.skinmap.evaluation.downstream import _evaluate_classification_task

        X = np.random.randn(100, 128)

        results, _, _ = _evaluate_classification_task(
            X,
            mock_dataset_with_metadata,
            col="gender",
            test_size=0.2,
            random_state=42,
            classifier_names=["linear", "knn10"],
            learned_metric_L=None,
            return_results=True,
            use_random_baseline=False,
            bootstrap_n_iterations=30,  # Fewer iterations for speed
        )

        # Check both classifiers have CI columns
        for clf_name in ["linear", "knn10"]:
            assert clf_name in results
            assert "accuracy_ci_lower" in results[clf_name]
            assert "accuracy_ci_upper" in results[clf_name]

    def test_bootstrap_disabled_in_downstream(self, mock_dataset_with_metadata):
        """Test that without bootstrap, no CI columns in downstream results."""
        from src.skinmap.evaluation.downstream import _evaluate_classification_task

        X = np.random.randn(100, 128)

        results, _, _ = _evaluate_classification_task(
            X,
            mock_dataset_with_metadata,
            col="gender",
            test_size=0.2,
            random_state=42,
            classifier_names=["linear"],
            learned_metric_L=None,
            return_results=True,
            use_random_baseline=False,
            bootstrap_n_iterations=0,  # Disabled
        )

        # Check NO CI columns
        linear_results = results["linear"]
        assert "accuracy" in linear_results
        assert "accuracy_ci_lower" not in linear_results
        assert "accuracy_ci_upper" not in linear_results


class TestBootstrapEndToEnd:
    """End-to-end integration tests for bootstrap in full pipeline."""

    def test_csv_output_includes_ci_columns(self, tmp_path):
        """Test that saved CSV files include CI columns when bootstrap is enabled."""
        from src.skinmap.utils.metadata import predict_missing_metadata

        # Create sample data
        np.random.seed(42)
        n_samples = 100
        embeddings = np.random.randn(n_samples, 64)
        df = pd.DataFrame(
            {
                "img_path": [f"img_{i}.jpg" for i in range(n_samples)],
                "laterality": ["left"] * 30 + ["right"] * 30 + [None] * 40,
            }
        )

        output_dir = str(tmp_path / "test_output")

        _, metrics_df, _ = predict_missing_metadata(
            embeddings,
            df,
            attributes=["laterality"],
            test_size=0.2,
            random_state=42,
            output_dir=output_dir,
            bootstrap_n_iterations=50,
        )

        # Save to CSV
        csv_path = tmp_path / "metrics.csv"
        metrics_df.to_csv(csv_path, index=False)

        # Read back and verify columns exist
        loaded_df = pd.read_csv(csv_path)
        assert "accuracy_ci_lower" in loaded_df.columns
        assert "accuracy_ci_upper" in loaded_df.columns
        assert "f1_macro_ci_lower" in loaded_df.columns
        assert "f1_macro_ci_upper" in loaded_df.columns

    def test_bootstrap_with_minimal_data(self):
        """Test bootstrap works with minimal amount of data."""
        from src.skinmap.utils.metadata import predict_missing_metadata

        # Very small dataset (edge case)
        embeddings = np.random.randn(20, 32)
        df = pd.DataFrame(
            {
                "img_path": [f"img_{i}.jpg" for i in range(20)],
                "category": ["A"] * 7 + ["B"] * 6 + [None] * 7,
            }
        )

        # Should still work, though CI may be wide
        df_out, metrics_df, _ = predict_missing_metadata(
            embeddings,
            df,
            attributes=["category"],
            test_size=0.3,
            random_state=42,
            bootstrap_n_iterations=20,  # Few iterations for small data
        )

        # Check it completed without errors
        assert len(metrics_df) == 1
        assert "accuracy_ci_lower" in metrics_df.columns
        assert "accuracy_ci_upper" in metrics_df.columns
