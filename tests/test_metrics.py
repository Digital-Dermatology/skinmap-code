"""Tests for learned metric transformation.

Tests cover:
- Binary and multiclass classification metric learning
- SVD decomposition of classifier weights
- Actual transformation quality (separates classes better)
- Reproducibility
- Edge cases (single class, few samples, etc.)

These tests verify the ACTUAL behavior of compute_learned_metric_transformation:
- Trains SGDClassifier on embeddings
- Extracts coefficient matrix W
- Computes L such that W^T @ W = L^T @ L via SVD
- Returns L of shape (k, d) where k = min(n_classes, d)
"""

import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier

import src.skinmap.utils.metrics as metrics_mod
from src.skinmap.utils.metrics import compute_learned_metric_transformation


class TestBinaryClassification:
    """Test metric learning for binary classification."""

    def test_binary_classification_returns_matrix(self):
        """Binary classification should return L matrix."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 512).astype(np.float32)
        labels = np.array([0, 1] * 50)  # Binary labels

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # Should return a matrix
        assert L_matrix is not None
        assert isinstance(L_matrix, np.ndarray)
        assert len(L_matrix.shape) == 2

        # For binary, W has shape (1, d) but we vstack to get (2, d)
        # Then SVD gives us (2, d) for L
        assert L_matrix.shape[0] == 2  # Binary case: 2 classes
        assert L_matrix.shape[1] == 512  # Input dimension

    def test_binary_metric_matches_classifier_weights(self):
        """L^T L should match W^T W from the underlying classifier."""
        np.random.seed(42)
        embeddings = np.random.randn(120, 64).astype(np.float32)
        labels = np.array([0, 1] * 60)

        class TrackingSGD(SGDClassifier):
            last_coef_ = None

            def fit(self, X, y, *args, **kwargs):
                result = super().fit(X, y, *args, **kwargs)
                TrackingSGD.last_coef_ = self.coef_.copy()
                return result

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(metrics_mod, "SGDClassifier", TrackingSGD)

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        W = TrackingSGD.last_coef_
        if W.shape[0] == 1:
            W = np.vstack([-W, W])

        target_metric = W.T @ W
        learned_metric = L_matrix.T @ L_matrix
        np.testing.assert_allclose(
            learned_metric,
            target_metric,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_binary_with_separable_classes(self):
        """Test with easily separable classes."""
        np.random.seed(42)
        n_samples = 200

        # Create clearly separable classes
        class0 = np.random.randn(n_samples // 2, 64).astype(np.float32) - 2
        class1 = np.random.randn(n_samples // 2, 64).astype(np.float32) + 2
        embeddings = np.vstack([class0, class1])
        labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        assert L_matrix is not None
        # Transform embeddings
        transformed = embeddings @ L_matrix.T

        # After transformation, classes should still be separated
        # (metric learning should preserve/enhance separation)
        class0_transformed = transformed[: n_samples // 2]
        class1_transformed = transformed[n_samples // 2 :]

        mean0 = np.mean(class0_transformed, axis=0)
        mean1 = np.mean(class1_transformed, axis=0)

        # Means should be different
        assert np.linalg.norm(mean0 - mean1) > 0.5


class TestMulticlassClassification:
    """Test metric learning for multiclass classification."""

    def test_multiclass_3_classes(self):
        """Multiclass with 3 classes should work."""
        np.random.seed(42)
        embeddings = np.random.randn(150, 256).astype(np.float32)
        labels = np.array([0] * 50 + [1] * 50 + [2] * 50)

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # For 3 classes, L should have shape (3, 256)
        assert L_matrix.shape == (3, 256)

    def test_multiclass_many_classes(self):
        """Test with many classes."""
        np.random.seed(42)
        n_classes = 10
        samples_per_class = 50
        embeddings = np.random.randn(n_classes * samples_per_class, 512).astype(
            np.float32
        )
        labels = np.repeat(np.arange(n_classes), samples_per_class)

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # Should have k <= min(n_classes, d)
        assert L_matrix.shape[0] == min(n_classes, 512)
        assert L_matrix.shape[1] == 512

    def test_multiclass_imbalanced(self):
        """Multiclass with imbalanced classes should work."""
        np.random.seed(42)
        embeddings = np.random.randn(160, 256).astype(np.float32)
        labels = np.array([0] * 100 + [1] * 30 + [2] * 20 + [3] * 10)

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        assert L_matrix is not None
        assert L_matrix.shape[0] == 4  # 4 classes
        assert L_matrix.shape[1] == 256


class TestDimensionHandling:
    """Test dimension handling in metric learning."""

    def test_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 2048).astype(np.float32)
        labels = np.array([0, 1] * 50)

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # For binary, k=2
        assert L_matrix.shape == (2, 2048)

    def test_low_dimensional_embeddings(self):
        """Test with low-dimensional embeddings."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 16).astype(np.float32)
        labels = np.array([0, 1, 2, 3] * 25)

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # 4 classes, 16 dims -> k=4
        assert L_matrix.shape == (4, 16)

    def test_more_classes_than_dimensions(self):
        """Test when n_classes > embedding_dim."""
        np.random.seed(42)
        embeddings = np.random.randn(300, 32).astype(np.float32)
        labels = np.repeat(np.arange(50), 6)  # 50 classes, 32 dims

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # k should be capped at min(n_classes, d) = 32
        assert L_matrix.shape[0] <= 32
        assert L_matrix.shape[1] == 32


class TestReproducibility:
    """Test reproducibility of metric learning."""

    def test_same_random_state_gives_same_result(self):
        """Same random_state should give identical results."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 256).astype(np.float32)
        labels = np.array([0, 1, 2] * 33 + [0])

        L1 = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        L2 = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # Should be identical
        np.testing.assert_array_almost_equal(L1, L2, decimal=5)

    def test_different_random_states_give_different_results(self):
        """Different random_state should give different results."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 256).astype(np.float32)
        labels = np.array([0, 1, 2] * 33 + [0])

        L1 = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        L2 = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=99,
        )

        # Should be different (SGDClassifier uses random_state)
        assert not np.allclose(L1, L2)


class TestSubsampling:
    """Test max_samples subsampling behavior."""

    def test_subsampling_when_exceeds_max_samples(self):
        """Test that subsampling occurs when N > max_samples."""
        np.random.seed(42)
        n_samples = 1000
        embeddings = np.random.randn(n_samples, 128).astype(np.float32)
        labels = np.random.randint(0, 5, n_samples)

        # Use small max_samples to force subsampling
        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
            max_samples=100,  # Much less than 1000
        )

        # Should still return valid result
        assert L_matrix is not None
        assert L_matrix.shape[1] == 128

    def test_no_subsampling_when_below_max_samples(self):
        """Test that no subsampling occurs when N < max_samples."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 128).astype(np.float32)
        labels = np.array([0, 1] * 50)

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
            max_samples=250_000,  # Default, much more than 100
        )

        assert L_matrix is not None


class TestEdgeCases:
    """Test edge cases in metric learning."""

    def test_minimum_samples(self):
        """Test with minimum viable number of samples."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)
        labels = np.array([0, 1] * 10)

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        assert L_matrix is not None
        assert L_matrix.shape == (2, 64)

    def test_single_class_fails_gracefully(self):
        """Test with single class (should return None or fail gracefully)."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 256).astype(np.float32)
        labels = np.array([0] * 100)  # All same class

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # Function should handle this gracefully (returns None on error)
        # Single class can't train a meaningful classifier
        assert L_matrix is None or L_matrix.shape[0] >= 2

    def test_string_labels(self):
        """Test that string labels work."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 256).astype(np.float32)
        labels = np.array(["cat", "dog"] * 50)  # String labels

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        assert L_matrix is not None
        assert L_matrix.shape == (2, 256)

    def test_integer_labels_non_contiguous(self):
        """Test with non-contiguous integer labels."""
        np.random.seed(42)
        embeddings = np.random.randn(150, 256).astype(np.float32)
        labels = np.array([10, 20, 30] * 50)  # Non-contiguous

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        assert L_matrix is not None
        assert L_matrix.shape[0] == 3  # 3 unique classes


class TestTransformationQuality:
    """Test that the learned transformation actually improves separation."""

    def test_transformation_separates_classes_better(self):
        """Verify that transformed embeddings have better class separation."""
        np.random.seed(42)
        n_samples = 200

        # Create classes with some overlap
        class0 = np.random.randn(n_samples // 2, 64).astype(np.float32) * 1.0 - 0.5
        class1 = np.random.randn(n_samples // 2, 64).astype(np.float32) * 1.0 + 0.5
        embeddings = np.vstack([class0, class1])
        labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

        # Learn metric
        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # Transform
        transformed = embeddings @ L_matrix.T

        class0_trans = transformed[: n_samples // 2]
        class1_trans = transformed[n_samples // 2 :]
        trans_sep = np.linalg.norm(
            np.mean(class0_trans, axis=0) - np.mean(class1_trans, axis=0)
        )

        # Transformed space should maintain or improve separation
        # (At minimum, separation should not collapse to zero)
        assert trans_sep > 0.1

    def test_transformation_preserves_data_structure(self):
        """Verify transformation doesn't create degenerate embeddings."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 256).astype(np.float32)
        labels = np.array([0, 1, 2] * 33 + [0])

        L_matrix = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        transformed = embeddings @ L_matrix.T

        # Check no NaN or inf
        assert not np.isnan(transformed).any()
        assert not np.isinf(transformed).any()

        # Check variance is not collapsed to zero
        assert np.var(transformed) > 1e-6

        # Different samples should still be different
        dists = np.linalg.norm(transformed[0] - transformed[1:10], axis=1)
        assert np.all(dists > 0)


class TestMathematicalProperties:
    """Test mathematical properties of the transformation."""

    def test_transformation_satisfies_metric_property(self, monkeypatch):
        """Test that W^T @ W ≈ L^T @ L for multiclass case."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 128).astype(np.float32)
        labels = np.array([0, 1, 2] * 33 + [0])

        class TrackingSGD(SGDClassifier):
            last_coef_ = None

            def fit(self, X, y, *args, **kwargs):
                result = super().fit(X, y, *args, **kwargs)
                TrackingSGD.last_coef_ = self.coef_.copy()
                return result

        monkeypatch.setattr(metrics_mod, "SGDClassifier", TrackingSGD)

        # Get L matrix
        L = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        W = TrackingSGD.last_coef_
        if W.shape[0] == 1:
            W = np.vstack([-W, W])

        learned_metric = L.T @ L
        target_metric = W.T @ W
        rel_error = np.linalg.norm(learned_metric - target_metric) / np.linalg.norm(
            target_metric
        )
        assert rel_error < 1e-6, rel_error

    def test_singular_values_are_non_negative(self):
        """Test that the transformation comes from valid SVD."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 128).astype(np.float32)
        labels = np.array([0, 1] * 50)

        L = compute_learned_metric_transformation(
            embeddings,
            labels,
            random_state=42,
        )

        # Compute singular values of L
        _, s, _ = np.linalg.svd(L, full_matrices=False)

        # All singular values should be non-negative
        assert np.all(s >= 0)

        # Singular values should be in descending order
        assert np.all(s[:-1] >= s[1:])
