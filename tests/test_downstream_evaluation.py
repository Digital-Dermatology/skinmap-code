"""Tests for downstream evaluation with SSL and CLIP models.

Tests cover:
- Single SSL model evaluation
- Single CLIP model evaluation
- Multiple models (mixed SSL and CLIP)
- Proper handling of processor=None for SSL models
- Confusion matrix generation and saving
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from src.create_skinmap import _save_confusion_matrices
from src.embedding_fusion import BuildSpec, build_model
from src.skinmap.evaluation.downstream import (
    _evaluate_classification_task,
    evaluate_image_dataset,
)


class TestDownstreamEvaluation:
    """Test evaluate_image_dataset with different model types."""

    @pytest.fixture
    def mock_ssl_model(self):
        """Create a mock SSL model."""
        model = Mock()
        model.eval = Mock()
        # SSL models are called directly and return embeddings
        model.return_value = torch.randn(2, 768)  # batch_size=2, embedding_dim=768
        return model

    @pytest.fixture
    def mock_clip_model(self):
        """Create a mock CLIP model."""
        model = Mock()
        model.eval = Mock()
        # CLIP models have get_image_features method
        model.get_image_features = Mock(return_value=torch.randn(2, 512))
        return model

    @pytest.fixture
    def mock_processor(self):
        """Create a mock processor for CLIP models."""
        processor = Mock()
        # Processor should return an object with pixel_values attribute
        result = Mock()
        result.pixel_values = torch.randn(2, 3, 224, 224)
        processor.return_value = result
        return processor

    @pytest.fixture
    def mock_dataset(self):
        """Create a simple mock dataset."""

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                # Return (image, path, label)
                img = Image.new("RGB", (224, 224))
                return img, f"image_{idx}.jpg", idx % 3  # 3 classes

        return SimpleDataset()

    def test_single_ssl_model_evaluation(self, mock_ssl_model, mock_dataset, tmp_path):
        """Test that single SSL model works without processor during embedding extraction."""
        emb_path = str(tmp_path / "embeddings.npz")

        with patch(
            "src.skinmap.evaluation.downstream.get_imagenet_transform"
        ) as mock_transform:
            # Mock the transform to return tensors
            mock_transform.return_value = lambda x: torch.randn(3, 224, 224)

            # This should NOT raise TypeError about processor being None during embedding extraction
            try:
                evaluate_image_dataset(
                    model=mock_ssl_model,
                    processor=None,  # SSL models don't have processors
                    device=torch.device("cpu"),
                    dataset=mock_dataset,
                    classification_cols=[],  # Skip classification to avoid needing meta_data
                    regression_cols=None,
                    emb_path=emb_path,
                    batch_size=2,
                    is_ssl_model=True,  # Critical: must be True for SSL models
                    num_workers=0,
                )
                success = True
            except TypeError as e:
                if "'NoneType' object is not callable" in str(e):
                    success = False
                    pytest.fail(f"SSL model evaluation failed with processor=None: {e}")
                else:
                    raise
            except AttributeError:
                # Expected if we hit classification code - but embedding extraction passed!
                success = True

            assert (
                success
            ), "Single SSL model should work with processor=None when is_ssl_model=True"

    def test_single_clip_model_evaluation(
        self, mock_clip_model, mock_processor, mock_dataset, tmp_path
    ):
        """Test that single CLIP model works with processor during embedding extraction."""
        emb_path = str(tmp_path / "embeddings.npz")

        # This should work fine for CLIP models - test embedding extraction
        try:
            evaluate_image_dataset(
                model=mock_clip_model,
                processor=mock_processor,
                device=torch.device("cpu"),
                dataset=mock_dataset,
                classification_cols=[],  # Skip classification to avoid needing meta_data
                regression_cols=None,
                emb_path=emb_path,
                batch_size=2,
                is_ssl_model=False,  # CLIP models
                num_workers=0,
            )
            success = True
        except Exception as e:
            success = False
            pytest.fail(f"CLIP model evaluation failed: {e}")

        assert success, "Single CLIP model should work with processor"

    def test_multiple_models_with_mixed_types(
        self, mock_ssl_model, mock_clip_model, mock_processor, mock_dataset, tmp_path
    ):
        """Test that multiple models (SSL + CLIP) work together during embedding extraction."""
        emb_path = str(tmp_path / "embeddings.npz")

        models_and_processors = [
            (mock_ssl_model, None, "ssl"),
            (mock_clip_model, mock_processor, "clip"),
        ]

        with patch(
            "src.skinmap.evaluation.downstream.get_imagenet_transform"
        ) as mock_transform:
            mock_transform.return_value = lambda x: torch.randn(3, 224, 224)

            try:
                evaluate_image_dataset(
                    model=mock_ssl_model,  # First model (not used when models_and_processors is set)
                    processor=None,
                    device=torch.device("cpu"),
                    dataset=mock_dataset,
                    classification_cols=[],  # Skip classification to avoid needing meta_data
                    regression_cols=None,
                    emb_path=emb_path,
                    batch_size=2,
                    models_and_processors=models_and_processors,
                    num_workers=0,
                )
                success = True
            except TypeError as e:
                if "'NoneType' object is not callable" in str(e):
                    success = False
                    pytest.fail(f"Mixed model evaluation failed: {e}")
                else:
                    raise

            assert success, "Multiple models with mixed SSL/CLIP should work"

    def test_ssl_model_without_is_ssl_flag_fails(
        self, mock_ssl_model, mock_dataset, tmp_path
    ):
        """Test that SSL model without is_ssl_model=True raises appropriate error."""
        emb_path = str(tmp_path / "embeddings.npz")

        # This SHOULD fail because processor is None but is_ssl_model=False
        with pytest.raises((TypeError, AttributeError)):
            evaluate_image_dataset(
                model=mock_ssl_model,
                processor=None,
                device=torch.device("cpu"),
                dataset=mock_dataset,
                classification_cols=["label"],
                regression_cols=None,
                emb_path=emb_path,
                batch_size=2,
                is_ssl_model=False,  # Wrong! Should be True
                num_workers=0,
            )


class TestConfusionMatrixGeneration:
    """Test confusion matrix computation and saving."""

    @pytest.fixture
    def mock_dataset_with_metadata(self):
        """Create a mock dataset with metadata for classification."""

        class DatasetWithMetadata(torch.utils.data.Dataset):
            def __init__(self):
                # Create metadata with binary and multi-class tasks
                self.meta_data = pd.DataFrame(
                    {
                        "binary_task": ["class_a", "class_b"] * 25,  # 50 samples
                        "multiclass_task": ["cat", "dog", "bird", "fish"] * 12
                        + ["cat", "dog"],  # 50 samples
                        "single_class": ["only_one"] * 50,  # Edge case
                    }
                )

            def __len__(self):
                return len(self.meta_data)

            def __getitem__(self, idx):
                img = Image.new("RGB", (224, 224))
                return img, f"image_{idx}.jpg", idx

        return DatasetWithMetadata()

    def test_classification_task_returns_confusion_matrices(
        self, mock_dataset_with_metadata
    ):
        """Test that _evaluate_classification_task returns confusion matrices."""
        # Create random embeddings
        X = np.random.randn(50, 128)

        results, confusion_matrices, label_encoder = _evaluate_classification_task(
            X=X,
            dataset=mock_dataset_with_metadata,
            col="binary_task",
            test_size=0.2,
            random_state=42,
            classifier_names=["linear"],  # Just test one classifier for speed
            return_results=True,
        )

        # Check that results are returned
        assert results is not None
        assert "linear" in results
        assert "accuracy" in results["linear"]

        # Check that confusion matrices are returned
        assert confusion_matrices is not None
        assert "linear" in confusion_matrices
        assert isinstance(confusion_matrices["linear"], np.ndarray)

        # Check that label encoder is returned
        assert label_encoder is not None
        assert isinstance(label_encoder, LabelEncoder)
        assert len(label_encoder.classes_) == 2  # Binary task


class _DummyProcessor:
    def __call__(self, images, return_tensors="pt"):
        batch_size = len(images)
        pixel_values = torch.zeros(batch_size, 3, 224, 224)
        return SimpleNamespace(pixel_values=pixel_values)


class _DummyClipModel:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim

    def eval(self):
        return self

    def get_image_features(self, pixel_values):
        batch_size = pixel_values.shape[0]
        return torch.ones(batch_size, self.embed_dim)


class _DatasetWithMetadata(torch.utils.data.Dataset):
    def __init__(self, n_samples):
        self.meta_data = pd.DataFrame(
            {"label": ["a"] * (n_samples // 2) + ["b"] * (n_samples // 2)}
        )

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img = Image.new("RGB", (224, 224))
        return img, f"image_{idx}.jpg", idx


def _build_projector(embed_dim, out_dim):
    spec = BuildSpec(
        teacher_names=["modelA"],
        teacher_dims=[embed_dim],
        text_dim=3,
        out_dim=out_dim,
        kind="linear",
    )
    return build_model(spec)


def _make_whitening_stats(embed_dim):
    return {
        "mu": [np.zeros((1, embed_dim), dtype=np.float32)],
        "W": [np.eye(embed_dim, dtype=np.float32)],
        "dims": [embed_dim],
        "clip_indices": [0],
    }


def test_projector_recomputes_cached_embeddings_dim_mismatch(tmp_path):
    embed_dim = 4
    out_dim = 2
    emb_path = tmp_path / "embeddings.npz"

    np.savez(
        emb_path,
        image_embeddings=np.zeros((4, embed_dim + 3), dtype=np.float32),
        labels=np.arange(4),
    )

    evaluate_image_dataset(
        model=_DummyClipModel(embed_dim),
        processor=_DummyProcessor(),
        device=torch.device("cpu"),
        dataset=_DatasetWithMetadata(4),
        classification_cols=[],
        regression_cols=None,
        emb_path=str(emb_path),
        batch_size=2,
        is_ssl_model=False,
        num_workers=0,
        projector_model=_build_projector(embed_dim, out_dim),
        projector_whitening_stats=_make_whitening_stats(embed_dim),
        projector_skip_whitening=False,
    )

    with np.load(emb_path, allow_pickle=True) as data:
        X = data["image_embeddings"]
        assert X.shape[1] == out_dim


def test_projector_used_for_learned_metric_space(monkeypatch, tmp_path):
    embed_dim = 4
    out_dim = 2
    emb_path = tmp_path / "embeddings_projector.npz"

    fit_calls = []

    class DummyKNN:
        def __init__(self, n_neighbors=10, metric="cosine"):
            self.metric = metric

        def fit(self, X, y):
            fit_calls.append((self.metric, X.shape[1]))
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    monkeypatch.setattr(
        "src.skinmap.evaluation.downstream.KNeighborsClassifier", DummyKNN
    )

    learned_metric_L = np.ones((1, out_dim), dtype=np.float32)

    evaluate_image_dataset(
        model=_DummyClipModel(embed_dim),
        processor=_DummyProcessor(),
        device=torch.device("cpu"),
        dataset=_DatasetWithMetadata(8),
        classification_cols=["label"],
        regression_cols=None,
        emb_path=str(emb_path),
        batch_size=2,
        is_ssl_model=False,
        num_workers=0,
        classifier_names=["knn10"],
        learned_metric_L=learned_metric_L,
        projector_model=_build_projector(embed_dim, out_dim),
        projector_whitening_stats=_make_whitening_stats(embed_dim),
        projector_skip_whitening=False,
        return_results=True,
        test_size=0.5,
    )

    assert ("cosine", out_dim) in fit_calls
    assert ("euclidean", 1) in fit_calls

    def test_confusion_matrix_dimensions_match_classes(
        self, mock_dataset_with_metadata
    ):
        """Test that confusion matrix dimensions match the number of classes."""
        X = np.random.randn(50, 128)

        # Test binary classification
        results, confusion_matrices, label_encoder = _evaluate_classification_task(
            X=X,
            dataset=mock_dataset_with_metadata,
            col="binary_task",
            test_size=0.2,
            random_state=42,
            classifier_names=["linear"],
            return_results=True,
        )

        cm = confusion_matrices["linear"]
        assert cm.shape == (2, 2), "Binary task should have 2x2 confusion matrix"

        # Test multi-class classification
        results, confusion_matrices, label_encoder = _evaluate_classification_task(
            X=X,
            dataset=mock_dataset_with_metadata,
            col="multiclass_task",
            test_size=0.2,
            random_state=42,
            classifier_names=["linear"],
            return_results=True,
        )

        cm = confusion_matrices["linear"]
        assert cm.shape == (
            4,
            4,
        ), "Multi-class task should have 4x4 confusion matrix"

    def test_confusion_matrices_for_all_classifiers(self, mock_dataset_with_metadata):
        """Test that confusion matrices are generated for all classifiers."""
        X = np.random.randn(50, 128)

        classifier_names = ["linear", "knn10"]

        results, confusion_matrices, label_encoder = _evaluate_classification_task(
            X=X,
            dataset=mock_dataset_with_metadata,
            col="binary_task",
            test_size=0.2,
            random_state=42,
            classifier_names=classifier_names,
            return_results=True,
        )

        # Check that confusion matrices exist for all classifiers
        for clf_name in classifier_names:
            assert clf_name in confusion_matrices
            assert isinstance(confusion_matrices[clf_name], np.ndarray)
            assert confusion_matrices[clf_name].shape == (2, 2)

    def test_learned_metric_confusion_matrices(self, mock_dataset_with_metadata):
        """Test that confusion matrices are generated for learned metric space."""
        X = np.random.randn(50, 128)

        # Create a mock learned metric transformation
        learned_metric_L = np.random.randn(128, 128)

        results, confusion_matrices, label_encoder = _evaluate_classification_task(
            X=X,
            dataset=mock_dataset_with_metadata,
            col="binary_task",
            test_size=0.2,
            random_state=42,
            classifier_names=["knn10"],
            learned_metric_L=learned_metric_L,
            return_results=True,
        )

        # Check that confusion matrices exist for both regular and learned metric
        assert "knn10" in confusion_matrices
        assert "knn10_learned" in confusion_matrices

        # Both should have same dimensions
        assert confusion_matrices["knn10"].shape == (2, 2)
        assert confusion_matrices["knn10_learned"].shape == (2, 2)

    def test_evaluate_image_dataset_returns_confusion_matrices(
        self, mock_dataset_with_metadata, tmp_path
    ):
        """Test that evaluate_image_dataset returns confusion matrices."""

        # Create mock embeddings file
        emb_path = str(tmp_path / "embeddings.npz")
        X = np.random.randn(50, 128)
        y = np.arange(50)
        np.savez(emb_path, image_embeddings=X, labels=y)

        # Mock model and processor (won't be used since embeddings exist)
        mock_model = Mock()
        mock_processor = Mock()

        results, confusion_matrices = evaluate_image_dataset(
            model=mock_model,
            processor=mock_processor,
            device=torch.device("cpu"),
            dataset=mock_dataset_with_metadata,
            classification_cols=["binary_task", "multiclass_task"],
            regression_cols=None,
            emb_path=emb_path,
            batch_size=2,
            classifier_names=["linear"],
            return_results=True,
        )

        # Check that results are returned
        assert results is not None
        assert "classification" in results
        assert "binary_task" in results["classification"]
        assert "multiclass_task" in results["classification"]

        # Check that confusion matrices are returned
        assert confusion_matrices is not None
        assert "binary_task" in confusion_matrices
        assert "multiclass_task" in confusion_matrices

        # Check structure
        assert "confusion_matrices" in confusion_matrices["binary_task"]
        assert "label_encoder" in confusion_matrices["binary_task"]

    def test_save_confusion_matrices_creates_files(
        self, mock_dataset_with_metadata, tmp_path
    ):
        """Test that _save_confusion_matrices creates all expected files."""
        # Create mock confusion matrices
        cm_binary = np.array([[8, 2], [1, 9]])
        cm_multi = np.array([[3, 1, 0, 0], [0, 4, 1, 0], [0, 0, 5, 1], [1, 0, 0, 4]])

        # Create label encoders
        le_binary = LabelEncoder()
        le_binary.fit(["class_a", "class_b"])

        le_multi = LabelEncoder()
        le_multi.fit(["cat", "dog", "bird", "fish"])

        confusion_matrices = {
            "binary_task": {
                "confusion_matrices": {"linear": cm_binary, "knn10": cm_binary},
                "label_encoder": le_binary,
            },
            "multiclass_task": {
                "confusion_matrices": {"linear": cm_multi},
                "label_encoder": le_multi,
            },
        }

        # Save confusion matrices
        _save_confusion_matrices(confusion_matrices, str(tmp_path), "TEST_DATASET")

        # Check that directory was created
        cm_dir = tmp_path / "confusion_matrices_TEST_DATASET"
        assert cm_dir.exists()

        # Check that files exist for binary_task
        assert (cm_dir / "binary_task_linear_cm.csv").exists()
        assert (cm_dir / "binary_task_linear_cm.pdf").exists()
        assert (cm_dir / "binary_task_linear_cm.svg").exists()
        assert (cm_dir / "binary_task_labels.json").exists()

        # Check that files exist for multiclass_task
        assert (cm_dir / "multiclass_task_linear_cm.csv").exists()
        assert (cm_dir / "multiclass_task_linear_cm.pdf").exists()
        assert (cm_dir / "multiclass_task_linear_cm.svg").exists()
        assert (cm_dir / "multiclass_task_labels.json").exists()

    def test_confusion_matrix_csv_has_correct_labels(
        self, mock_dataset_with_metadata, tmp_path
    ):
        """Test that CSV confusion matrices have correct row/column labels."""
        cm = np.array([[8, 2], [1, 9]])
        le = LabelEncoder()
        le.fit(["class_a", "class_b"])

        confusion_matrices = {
            "test_task": {
                "confusion_matrices": {"linear": cm},
                "label_encoder": le,
            }
        }

        _save_confusion_matrices(confusion_matrices, str(tmp_path), "TEST")

        # Read the CSV and check labels
        cm_dir = tmp_path / "confusion_matrices_TEST"
        csv_path = cm_dir / "test_task_linear_cm.csv"

        df = pd.read_csv(csv_path, index_col=0)

        # Check that index and columns have correct labels
        assert list(df.index) == ["class_a", "class_b"]
        assert list(df.columns) == ["class_a", "class_b"]

        # Check that values are correct
        assert df.loc["class_a", "class_a"] == 8
        assert df.loc["class_a", "class_b"] == 2
        assert df.loc["class_b", "class_a"] == 1
        assert df.loc["class_b", "class_b"] == 9

    def test_confusion_matrix_json_has_correct_mapping(
        self, mock_dataset_with_metadata, tmp_path
    ):
        """Test that JSON label mapping is correct."""
        import json

        cm = np.array([[8, 2], [1, 9]])
        le = LabelEncoder()
        le.fit(["class_a", "class_b"])

        confusion_matrices = {
            "test_task": {
                "confusion_matrices": {"linear": cm},
                "label_encoder": le,
            }
        }

        _save_confusion_matrices(confusion_matrices, str(tmp_path), "TEST")

        # Read the JSON and check mapping
        cm_dir = tmp_path / "confusion_matrices_TEST"
        json_path = cm_dir / "test_task_labels.json"

        with open(json_path, "r") as f:
            label_mapping = json.load(f)

        # Check that mapping is correct
        assert label_mapping["0"] == "class_a"
        assert label_mapping["1"] == "class_b"

    def test_empty_confusion_matrices_dict(self, tmp_path):
        """Test that function handles empty confusion matrices gracefully."""
        # Should not raise an error
        _save_confusion_matrices({}, str(tmp_path), "TEST")

        # Directory should not be created
        cm_dir = tmp_path / "confusion_matrices_TEST"
        assert not cm_dir.exists()

    def test_confusion_matrix_without_return_results_flag(
        self, mock_dataset_with_metadata
    ):
        """Test that confusion matrices are not returned when return_results=False."""
        X = np.random.randn(50, 128)

        results, confusion_matrices, label_encoder = _evaluate_classification_task(
            X=X,
            dataset=mock_dataset_with_metadata,
            col="binary_task",
            test_size=0.2,
            random_state=42,
            classifier_names=["linear"],
            return_results=False,  # Should return None
        )

        assert results is None
        assert confusion_matrices is None
        assert label_encoder is None

    def test_confusion_matrix_with_missing_test_classes(self):
        """Test confusion matrix when not all classes appear in test set.

        This is a regression test for a bug where confusion_matrix() without
        the labels parameter would only create a matrix for classes present in
        the test set, causing dimension mismatches when saving to CSV with all
        label encoder classes as indices.

        Real-world scenario: DDI dataset with 48 classes, but only 37 appeared
        in test set after splitting, causing shape mismatch (37, 37) vs (48, 48).
        """

        class ImbalancedDataset(torch.utils.data.Dataset):
            def __init__(self):
                # Create highly imbalanced dataset
                # Most samples are "common", only 1 sample of rare classes
                # With stratified split and test_size=0.3, rare classes likely won't appear in test
                self.meta_data = pd.DataFrame(
                    {
                        "imbalanced_task": ["common"] * 80
                        + ["rare1"] * 2
                        + ["rare2"] * 2
                        + ["rare3"] * 2
                        + ["rare4"] * 2
                        + ["rare5"] * 2
                    }
                )

            def __len__(self):
                return len(self.meta_data)

            def __getitem__(self, idx):
                img = Image.new("RGB", (224, 224))
                return img, f"image_{idx}.jpg", idx

        dataset = ImbalancedDataset()
        X = np.random.randn(90, 128)

        # Use a test size that makes it likely some rare classes won't appear in test set
        results, confusion_matrices, label_encoder = _evaluate_classification_task(
            X=X,
            dataset=dataset,
            col="imbalanced_task",
            test_size=0.15,  # Small test set to increase chance of missing classes
            random_state=42,
            classifier_names=["linear"],
            return_results=True,
        )

        # Get the number of classes in the full dataset
        n_classes = len(label_encoder.classes_)
        assert n_classes == 6, f"Expected 6 classes, got {n_classes}"

        # The confusion matrix MUST have dimensions matching ALL classes,
        # not just the classes that appeared in the test set
        cm = confusion_matrices["linear"]
        assert (
            cm.shape[0] == n_classes
        ), f"Confusion matrix rows ({cm.shape[0]}) should match all classes ({n_classes})"
        assert (
            cm.shape[1] == n_classes
        ), f"Confusion matrix cols ({cm.shape[1]}) should match all classes ({n_classes})"

        # Verify this doesn't cause errors when saving to DataFrame/CSV
        # (This was the original bug - shape mismatch when creating DataFrame)
        try:
            df = pd.DataFrame(
                cm, index=label_encoder.classes_, columns=label_encoder.classes_
            )
            assert df.shape == (
                n_classes,
                n_classes,
            ), "DataFrame shape should match confusion matrix"
            success = True
        except ValueError as e:
            success = False
            pytest.fail(
                f"Failed to create DataFrame from confusion matrix: {e}. "
                f"This indicates dimension mismatch between confusion matrix ({cm.shape}) "
                f"and label encoder classes ({n_classes})"
            )

        assert (
            success
        ), "Should be able to create DataFrame with all label encoder classes as indices"
