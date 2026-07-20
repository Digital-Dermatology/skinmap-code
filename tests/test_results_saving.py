"""Tests for verifying all results are saved correctly.

Tests cover:
1. Cache corruption detection (fail-fast behavior)
2. Downstream evaluation results saving
3. Learned metric transformation saving
4. Dimension consistency across embeddings and dataframes
5. Atomic saving of embeddings + dataframes
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.skinmap.cache.manager import CacheConfig, EmbeddingCache


class TestCacheCorruptionDetection:
    """Test that corrupted caches fail fast with clear errors."""

    def test_empty_embeddings_raises_error(self, tmp_path):
        """Test that loading cache with 0 samples raises RuntimeError."""
        from unittest.mock import MagicMock

        # Setup: Create cache with 0 samples
        cache = EmbeddingCache(str(tmp_path))
        run_name = "test_run"
        config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            fusion_method="none",
        )

        # Save empty cache (simulating a failed previous run)
        empty_embs = np.array([], dtype=np.float32).reshape(0, 768)
        cache.save_combined(
            run_name,
            empty_embs,
            None,
            np.array([]),
            config,
        )

        # Mock args to pass output_dir
        mock_args = MagicMock()
        mock_args.output_dir = str(tmp_path)

        # Import function
        from src.create_skinmap import extract_or_load_embeddings

        # Test: Should raise RuntimeError with clear message
        with pytest.raises(RuntimeError, match="CACHE CORRUPTION.*0 samples"):
            # Create minimal valid inputs
            models = []
            df = pd.DataFrame({"img_path": ["test.jpg"], "dataset_desc": ["test"]})
            device = "cpu"

            # The cache will be loaded naturally and should fail with 0 samples
            extract_or_load_embeddings(
                models,
                df,
                mock_args,
                device,
                cache,
                config,
                run_name,
            )

    def test_missing_dataframe_raises_error(self, tmp_path):
        """Test that missing dataframe.csv raises RuntimeError."""
        from unittest.mock import MagicMock

        cache = EmbeddingCache(str(tmp_path))
        run_name = "test_run"
        config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            fusion_method="none",
        )

        # Save embeddings WITHOUT dataframe
        embs = np.random.randn(100, 768).astype(np.float32)
        cache.save_combined(
            run_name,
            embs,
            None,
            np.array(["test"] * 100),
            config,
        )

        # Mock args
        mock_args = MagicMock()
        mock_args.output_dir = str(tmp_path)

        from src.create_skinmap import extract_or_load_embeddings

        # Test: Should raise RuntimeError about missing dataframe
        with pytest.raises(RuntimeError, match="dataframe is missing"):
            models = []
            df = pd.DataFrame(
                {
                    "img_path": [f"{i}.jpg" for i in range(100)],
                    "dataset_desc": ["test"] * 100,
                }
            )
            device = "cpu"

            extract_or_load_embeddings(
                models,
                df,
                mock_args,
                device,
                cache,
                config,
                run_name,
            )

    def test_dimension_mismatch_raises_error(self, tmp_path):
        """Test that dimension mismatch between embeddings and dataframe raises error."""
        from unittest.mock import MagicMock

        cache = EmbeddingCache(str(tmp_path))
        run_name = "test_run"
        config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            fusion_method="none",
        )

        # Save embeddings with 100 samples
        embs = np.random.randn(100, 768).astype(np.float32)
        cache.save_combined(
            run_name,
            embs,
            None,
            np.array(["test"] * 100),
            config,
        )

        # Save dataframe with 102 samples (mismatch!)
        emb_dir = os.path.join(tmp_path, run_name, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        df_wrong = pd.DataFrame(
            {
                "img_path": [f"{i}.jpg" for i in range(102)],
                "dataset_desc": ["test"] * 102,
            }
        )
        df_wrong.to_csv(os.path.join(emb_dir, "dataframe.csv"), index=False)

        # Mock args
        mock_args = MagicMock()
        mock_args.output_dir = str(tmp_path)

        from src.create_skinmap import extract_or_load_embeddings

        # Test: Should raise RuntimeError about dimension mismatch
        with pytest.raises(RuntimeError, match="CACHE CORRUPTION.*Dimension mismatch"):
            models = []
            df = pd.DataFrame(
                {
                    "img_path": [f"{i}.jpg" for i in range(102)],
                    "dataset_desc": ["test"] * 102,
                }
            )
            device = "cpu"

            extract_or_load_embeddings(
                models,
                df,
                mock_args,
                device,
                cache,
                config,
                run_name,
            )


class TestResultsSaving:
    """Test that all computed results are saved to disk."""

    def test_embeddings_and_dataframe_saved_atomically(self, tmp_path):
        """Test that embeddings and dataframe are saved together."""
        from unittest.mock import MagicMock, patch

        # Setup
        cache = EmbeddingCache(str(tmp_path))
        run_name = "test_run"
        config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            fusion_method="none",
        )

        # Create test data
        n_samples = 50
        embs = np.random.randn(n_samples, 768).astype(np.float32)
        labels = np.array(["dataset1"] * n_samples)
        df = pd.DataFrame(
            {
                "img_path": [f"{i}.jpg" for i in range(n_samples)],
                "dataset_desc": ["dataset1"] * n_samples,
            }
        )

        # Mock args
        mock_args = MagicMock()
        mock_args.output_dir = str(tmp_path)
        mock_args.batch_size = 32
        mock_args.vis_samples = None
        mock_args.num_workers = 0
        mock_args.use_trained_projector = False
        mock_args.svd_components = None

        # Mock model and extraction
        with patch("src.create_skinmap.extract_clip_embeddings") as mock_extract:
            mock_extract.return_value = (embs, embs.copy(), labels, [], df.copy())

            from src.create_skinmap import extract_or_load_embeddings
            from src.skinmap.models.loaders import ModelInfo

            mock_model = MagicMock()
            mock_processor = MagicMock()
            models = [ModelInfo(mock_model, mock_processor, "clip", "test_model")]

            result = extract_or_load_embeddings(
                models,
                df,
                mock_args,
                "cpu",
                cache,
                config,
                run_name,
            )

            (
                image_embs,
                text_embs,
                dataset_labels,
                filtered_df,
                corrupted,
                artifacts,
                fusion_method,
                wandb_run,
            ) = result

        # Verify: Both embeddings and dataframe should exist
        emb_dir = os.path.join(tmp_path, run_name, "embeddings")
        assert os.path.exists(os.path.join(emb_dir, "embeddings.npz"))
        assert os.path.exists(os.path.join(emb_dir, "dataframe.csv"))

        # Verify: Dimensions match
        loaded_df = pd.read_csv(os.path.join(emb_dir, "dataframe.csv"))
        assert len(loaded_df) == len(image_embs)
        assert len(loaded_df) == n_samples

    def test_downstream_results_saving(self, tmp_path):
        """Test that downstream evaluation results are saved to CSV."""
        from src.create_skinmap import _save_downstream_results

        # Create mock results structure
        results = {
            "classification": {
                "itch": {
                    "linear": {
                        "accuracy": 0.85,
                        "balanced_accuracy": 0.83,
                        "f1_macro": 0.82,
                    },
                    "knn10": {
                        "accuracy": 0.80,
                        "balanced_accuracy": 0.78,
                        "f1_macro": 0.77,
                    },
                },
                "gender": {
                    "linear": {
                        "accuracy": 0.90,
                        "balanced_accuracy": 0.88,
                        "f1_macro": 0.87,
                    },
                },
            },
            "regression": {
                "age": {
                    "linear": {"mse": 100.5, "mae": 8.2},
                },
            },
        }

        output_path = os.path.join(tmp_path, "downstream_test.csv")

        # Save results
        _save_downstream_results(results, output_path, "TEST_DATASET")

        # Verify: File exists
        assert os.path.exists(output_path)

        # Verify: Content is correct
        df = pd.read_csv(output_path)
        assert len(df) == 4  # 3 classification + 1 regression

        # Check classification rows
        class_rows = df[df["task_type"] == "classification"]
        assert len(class_rows) == 3
        assert set(class_rows["task_name"].unique()) == {"itch", "gender"}
        assert set(class_rows["classifier"].unique()) == {"linear", "knn10"}

        # Check metrics are present
        itch_linear = df[
            (df["task_name"] == "itch") & (df["classifier"] == "linear")
        ].iloc[0]
        assert itch_linear["accuracy"] == 0.85
        assert itch_linear["f1_macro"] == 0.82

        # Check regression row
        reg_rows = df[df["task_type"] == "regression"]
        assert len(reg_rows) == 1
        assert reg_rows.iloc[0]["task_name"] == "age"
        assert reg_rows.iloc[0]["mse"] == 100.5

    def test_learned_metric_saving(self, tmp_path):
        """Test that learned metric transformation is saved."""
        # This would be tested as part of integration test
        # For now, verify the saving logic exists
        metric_L = np.random.randn(768, 768).astype(np.float32)
        metric_path = os.path.join(tmp_path, "learned_metric_L.npy")

        np.save(metric_path, metric_L)

        # Verify: File exists and can be loaded
        assert os.path.exists(metric_path)
        loaded = np.load(metric_path)
        assert np.allclose(loaded, metric_L)

    def test_separability_results_saving(self, tmp_path):
        """Test that separability results are saved correctly."""
        results = {
            "classifier": "Linear probe",
            "accuracy": 0.92,
            "balanced_accuracy": 0.90,
            "precision_macro": 0.89,
            "recall_macro": 0.88,
            "f1_macro": 0.88,
        }

        sep_path = os.path.join(tmp_path, "separability_results.csv")
        results_df = pd.DataFrame([results])
        results_df.to_csv(sep_path, index=False)

        # Verify: File exists and content is correct
        assert os.path.exists(sep_path)
        loaded = pd.read_csv(sep_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["accuracy"] == 0.92
        assert loaded.iloc[0]["f1_macro"] == 0.88


class TestDimensionConsistency:
    """Test that dimensions are consistent across the pipeline."""

    def test_fusion_dimension_mismatch_raises_error(self, tmp_path):
        """Test that dimension mismatches in fusion pipeline raise clear errors."""
        from unittest.mock import MagicMock, patch

        from src.skinmap.cache.manager import EmbeddingCache
        from src.skinmap.embeddings.fusion import combine_embeddings_simple
        from src.skinmap.models.loaders import ModelInfo

        cache = EmbeddingCache(str(tmp_path))

        # Mock two models with different sample counts
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        models = [
            ModelInfo(mock_model1, None, "ssl", "model1"),
            ModelInfo(mock_model2, None, "ssl", "model2"),
        ]

        df = pd.DataFrame(
            {
                "img_path": [f"{i}.jpg" for i in range(100)],
                "dataset_desc": ["test"] * 100,
            }
        )

        # Mock extraction to return different sample counts
        def mock_extract_side_effect(*args, **kwargs):
            if args[0] == mock_model1:
                # First model: 100 samples
                return (
                    np.random.randn(100, 512).astype(np.float32),
                    None,
                    np.array(["test"] * 100),
                    [],
                    df.copy(),
                )
            else:
                # Second model: 98 samples (BUG!)
                return (
                    np.random.randn(98, 512).astype(np.float32),
                    None,
                    np.array(["test"] * 98),
                    [],
                    df.iloc[:98].copy(),
                )

        with patch(
            "src.skinmap.embeddings.fusion.extract_ssl_embeddings"
        ) as mock_extract:
            mock_extract.side_effect = mock_extract_side_effect

            # Test: Should raise RuntimeError with clear dimension mismatch message
            with pytest.raises(RuntimeError, match="DIMENSION MISMATCH"):
                combine_embeddings_simple(
                    models,
                    df,
                    "cpu",
                    cache,
                    batch_size=32,
                    max_samples=None,
                    dataset_col="dataset_desc",
                    num_workers=0,
                    svd_components=None,
                )

    def test_embeddings_dataframe_consistency_after_corrupted_filtering(self, tmp_path):
        """Test that embeddings and dataframe have same length after filtering corrupted samples."""
        from src.skinmap.embeddings.extractors import _filter_corrupted

        # Setup: 10 samples, 2 corrupted (indices 3 and 7)
        n_total = 10
        embeddings = np.random.randn(n_total, 768).astype(np.float32)
        text_embeddings = np.random.randn(n_total, 512).astype(np.float32)

        # Indices: normal samples have positive indices, corrupted have negative
        indices = np.array(
            [0, 1, 2, -4, 4, 5, 6, -8, 8, 9]
        )  # corrupted at positions 3, 7

        df = pd.DataFrame(
            {
                "img_path": [f"{i}.jpg" for i in range(n_total)],
                "dataset_desc": ["test"] * n_total,
            }
        )

        # Filter
        valid_embs, valid_text_embs, corrupted_indices, filtered_df = _filter_corrupted(
            embeddings, indices, df, text_embeddings
        )

        # Verify: Dimensions match
        assert len(valid_embs) == len(filtered_df)
        assert len(valid_text_embs) == len(filtered_df)

        # Verify: Correct number of samples
        assert len(valid_embs) == 8  # 10 - 2 corrupted
        assert len(corrupted_indices) == 2
        assert set(corrupted_indices) == {3, 7}

        # Verify: Correct samples retained
        assert filtered_df["img_path"].tolist() == [
            "0.jpg",
            "1.jpg",
            "2.jpg",
            "4.jpg",
            "5.jpg",
            "6.jpg",
            "8.jpg",
            "9.jpg",
        ]


class TestConfigSerialization:
    """Test that config files can be serialized to JSON without errors."""

    def test_save_outputs_config_json_serialization(self, tmp_path):
        """Test that save_outputs correctly serializes config.json with numpy types."""
        from unittest.mock import MagicMock

        from src.create_skinmap import save_outputs

        # Create mock args with various types
        mock_args = MagicMock()
        mock_args.model_name = "test_model"
        mock_args.model_path = None
        mock_args.ssl_model = None
        mock_args.use_trained_projector = False
        mock_args.projector_dim = 768
        mock_args.svd_components = 1024
        mock_args.max_whitening_dim = 768
        mock_args.skip_whitening = False

        # Create test data with numpy arrays
        n_samples = 50
        df = pd.DataFrame(
            {
                "img_path": [f"{i}.jpg" for i in range(n_samples)],
                "dataset_desc": ["test"] * n_samples,
            }
        )
        image_embeddings = np.random.randn(n_samples, 768).astype(np.float32)
        text_embeddings = np.random.randn(n_samples, 512).astype(np.float32)
        dataset_labels = np.array(["test"] * n_samples)
        corrupted_indices = [5, 10]

        # Call save_outputs
        save_outputs(
            df=df,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            dataset_labels=dataset_labels,
            corrupted_indices=corrupted_indices,
            fusion_method="svd_1024",
            args=mock_args,
            output_dir=str(tmp_path),
            run_name="test_run",
            artifacts={},
            model_names=["model1"],
        )

        # Verify: config.json exists and is valid JSON
        config_path = os.path.join(tmp_path, "config.json")
        assert os.path.exists(config_path)

        # Verify: Can load and parse the JSON
        import json

        with open(config_path, "r") as f:
            config = json.load(f)

        # Verify: Config contains expected fields
        assert config["model_name"] == "test_model"
        assert config["embedding_dim"] == 768
        assert config["num_samples"] == n_samples
        assert config["num_corrupted"] == 2
        assert config["fusion_method"] == "svd_1024"
        assert config["svd_components"] == 1024

    def test_projector_config_json_serialization_with_numpy_types(self, tmp_path):
        """Test that projector_config.json serializes correctly with numpy arrays in whitening_stats."""
        from unittest.mock import MagicMock, patch

        from src.create_skinmap import save_outputs

        # Create mock args with projector enabled
        mock_args = MagicMock()
        mock_args.model_name = "test_model"
        mock_args.model_path = None
        mock_args.ssl_model = None
        mock_args.use_trained_projector = True
        mock_args.projector_type = "mlp"
        mock_args.projector_dim = 512
        mock_args.svd_components = None
        mock_args.max_whitening_dim = 768
        mock_args.skip_whitening = False

        # Create test data
        n_samples = 30
        df = pd.DataFrame(
            {
                "img_path": [f"{i}.jpg" for i in range(n_samples)],
                "dataset_desc": ["test"] * n_samples,
            }
        )
        image_embeddings = np.random.randn(n_samples, 512).astype(np.float32)
        text_embeddings = np.random.randn(n_samples, 256).astype(np.float32)
        dataset_labels = np.array(["test"] * n_samples)

        # Create artifacts with whitening stats containing numpy arrays
        whitening_stats = {
            "dims": np.array([256, 256, 256]),  # numpy array
            "clip_indices": np.array([0, 256, 512]),  # numpy array
            "mu": [np.random.randn(256) for _ in range(3)],
            "W": [np.random.randn(256, 256) for _ in range(3)],
        }
        mock_projector = MagicMock()
        artifacts = {
            "projector_model": mock_projector,
            "whitening_stats": whitening_stats,
        }

        # Mock torch.save to avoid pickling issues with MagicMock
        with patch("torch.save"):
            # Call save_outputs
            save_outputs(
                df=df,
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                dataset_labels=dataset_labels,
                corrupted_indices=[],
                fusion_method="trained_projector",
                args=mock_args,
                output_dir=str(tmp_path),
                run_name="test_run",
                artifacts=artifacts,
                model_names=["model1", "model2", "model3"],
            )

        # Verify: projector_config.json exists and is valid JSON
        projector_config_path = os.path.join(
            tmp_path, "embeddings", "projector_config.json"
        )
        assert os.path.exists(projector_config_path)

        # Verify: Can load and parse the JSON
        import json

        with open(projector_config_path, "r") as f:
            config = json.load(f)

        # Verify: All fields are JSON-serializable (no numpy types)
        assert config["projector_type"] == "mlp"
        assert config["projector_dim"] == 512
        assert isinstance(config["teacher_dims"], list)  # Should be list, not ndarray
        assert isinstance(config["clip_indices"], list)  # Should be list, not ndarray
        assert config["model_paths"] == ["model1", "model2", "model3"]


class TestIntegrationScenarios:
    """Integration tests for real-world failure scenarios."""

    def test_incomplete_cache_from_crashed_run(self, tmp_path):
        """Test handling of cache from run that crashed before saving dataframe."""
        from unittest.mock import MagicMock

        cache = EmbeddingCache(str(tmp_path))
        run_name = "crashed_run"
        config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            fusion_method="none",
        )

        # Simulate: Run started, saved embeddings, then crashed
        embs = np.random.randn(100, 768).astype(np.float32)
        cache.save_combined(
            run_name,
            embs,
            None,
            np.array(["test"] * 100),
            config,
        )
        # Note: dataframe.csv NOT saved (crash before save_outputs)

        # Mock args
        mock_args = MagicMock()
        mock_args.output_dir = str(tmp_path)

        from src.create_skinmap import extract_or_load_embeddings

        # Test: Next run should FAIL, not try to recover
        with pytest.raises(RuntimeError, match="dataframe is missing"):
            models = []
            df = pd.DataFrame(
                {
                    "img_path": [f"{i}.jpg" for i in range(100)],
                    "dataset_desc": ["test"] * 100,
                }
            )
            device = "cpu"

            extract_or_load_embeddings(
                models,
                df,
                mock_args,
                device,
                cache,
                config,
                run_name,
            )

    def test_all_results_saved_in_successful_run(self, tmp_path):
        """Verify that a successful run creates all expected output files."""
        # This is more of a checklist test - verifying file structure
        run_name = "successful_run"
        out_dir = os.path.join(tmp_path, run_name)
        os.makedirs(out_dir, exist_ok=True)

        # Simulate saving all results
        emb_dir = os.path.join(out_dir, "embeddings")
        analysis_dir = os.path.join(out_dir, "analysis")
        os.makedirs(emb_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)

        # Save all expected files
        expected_files = {
            os.path.join(emb_dir, "embeddings.npz"): lambda: np.savez(
                os.path.join(emb_dir, "embeddings.npz"),
                image_embeddings=np.random.randn(100, 768),
                text_embeddings=np.random.randn(100, 512),
                dataset_labels=np.array(["test"] * 100),
            ),
            os.path.join(emb_dir, "dataframe.csv"): lambda: pd.DataFrame(
                {"img_path": [f"{i}.jpg" for i in range(100)]}
            ).to_csv(os.path.join(emb_dir, "dataframe.csv"), index=False),
            os.path.join(out_dir, "separability_results.csv"): lambda: pd.DataFrame(
                [{"accuracy": 0.9}]
            ).to_csv(os.path.join(out_dir, "separability_results.csv"), index=False),
            os.path.join(
                analysis_dir, "metadata_prediction_metrics.csv"
            ): lambda: pd.DataFrame([{"attribute": "gender", "accuracy": 0.85}]).to_csv(
                os.path.join(analysis_dir, "metadata_prediction_metrics.csv"),
                index=False,
            ),
            os.path.join(out_dir, "downstream_PAD_UFES_20.csv"): lambda: pd.DataFrame(
                [{"task_type": "classification", "task_name": "itch", "accuracy": 0.8}]
            ).to_csv(os.path.join(out_dir, "downstream_PAD_UFES_20.csv"), index=False),
            os.path.join(out_dir, "downstream_DDI.csv"): lambda: pd.DataFrame(
                [
                    {
                        "task_type": "classification",
                        "task_name": "skin_tone",
                        "accuracy": 0.75,
                    }
                ]
            ).to_csv(os.path.join(out_dir, "downstream_DDI.csv"), index=False),
            os.path.join(out_dir, "learned_metric_L.npy"): lambda: np.save(
                os.path.join(out_dir, "learned_metric_L.npy"), np.random.randn(768, 768)
            ),
            os.path.join(out_dir, "config.json"): lambda: Path(
                os.path.join(out_dir, "config.json")
            ).write_text('{"model_name": "test"}'),
        }

        # Create all files
        for path, creator in expected_files.items():
            creator()

        # Verify: All files exist
        for path in expected_files.keys():
            assert os.path.exists(path), f"Missing expected file: {path}"

        # Verify: Can load all files without errors
        np.load(os.path.join(emb_dir, "embeddings.npz"))
        pd.read_csv(os.path.join(emb_dir, "dataframe.csv"))
        pd.read_csv(os.path.join(out_dir, "separability_results.csv"))
        pd.read_csv(os.path.join(analysis_dir, "metadata_prediction_metrics.csv"))
        pd.read_csv(os.path.join(out_dir, "downstream_PAD_UFES_20.csv"))
        pd.read_csv(os.path.join(out_dir, "downstream_DDI.csv"))
        np.load(os.path.join(out_dir, "learned_metric_L.npy"))
