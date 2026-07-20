from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.create_skinmap import (
    generate_run_name,
    load_cached_separability_results,
    reuse_cached_metadata_predictions,
)
from src.skinmap.cache.manager import get_single_model_cache_path


def test_reuse_cached_metadata_predictions_success(tmp_path):
    filtered_df = pd.DataFrame(
        {
            "img_path": ["a.jpg", "b.jpg"],
            "laterality": ["left", None],
        }
    )
    cached_df = pd.DataFrame(
        {
            "img_path": ["a.jpg", "b.jpg"],
            "laterality_pred": ["left", "right"],
        }
    )
    cached_path = tmp_path / "dataframe.csv"
    cached_df.to_csv(cached_path, index=False)

    updated_df, reused = reuse_cached_metadata_predictions(
        filtered_df,
        ["laterality"],
        str(cached_path),
    )

    assert reused is True
    assert "laterality_pred" in updated_df.columns
    missing_mask = updated_df["laterality"].isna()
    assert updated_df.loc[missing_mask, "laterality_pred"].tolist() == ["right"]


def test_reuse_cached_metadata_predictions_incomplete(tmp_path):
    filtered_df = pd.DataFrame(
        {
            "img_path": ["a.jpg", "b.jpg"],
            "laterality": ["left", None],
        }
    )
    # Missing prediction for b.jpg
    cached_df = pd.DataFrame(
        {
            "img_path": ["a.jpg"],
            "laterality_pred": ["left"],
        }
    )
    cached_path = tmp_path / "dataframe.csv"
    cached_df.to_csv(cached_path, index=False)

    updated_df, reused = reuse_cached_metadata_predictions(
        filtered_df,
        ["laterality"],
        str(cached_path),
    )

    assert reused is False
    # When cache reuse fails, should return original DataFrame unchanged
    assert updated_df.equals(filtered_df)


def test_load_cached_separability_results(tmp_path):
    sep_path = tmp_path / "sep.csv"
    df = pd.DataFrame([{"classifier": "Linear", "accuracy": 0.9}])
    df.to_csv(sep_path, index=False)

    cached = load_cached_separability_results(str(sep_path))
    assert cached is not None
    assert cached.equals(df)


def test_load_cached_separability_results_empty(tmp_path):
    sep_path = tmp_path / "sep.csv"
    pd.DataFrame().to_csv(sep_path, index=False)

    cached = load_cached_separability_results(str(sep_path))
    assert cached is None


class TestRunNameCacheConsistency:
    """Test that run names are consistent with cache paths.

    This ensures that when running a single model that was previously part of a
    combined run, the artifacts are saved to the same folder as the embeddings cache.
    """

    def test_single_model_run_name_matches_cache_path(self, tmp_path):
        """Test that single model run_name matches the cache folder name."""
        output_dir = str(tmp_path)
        mock_args = MagicMock()
        mock_args.ssl_model = None
        mock_args.model_path = None

        # Test case 1: Model path with "assets/" prefix
        model_path = "assets/clip-something"
        model_names = [model_path]

        run_name = generate_run_name(mock_args, model_names)
        cache_path = get_single_model_cache_path(model_path, output_dir)

        # Extract the folder name from cache path
        # Cache path format: output_dir/folder_name/embeddings/embeddings.npz
        cache_folder = Path(cache_path).parent.parent.name

        assert (
            run_name == cache_folder
        ), f"Run name '{run_name}' != cache folder '{cache_folder}'"

        # Both should be "clip-something" (with "assets/" stripped)
        assert run_name == "clip-something"
        assert cache_folder == "clip-something"

    def test_single_model_with_slashes_consistency(self, tmp_path):
        """Test model path with slashes is cleaned consistently."""
        output_dir = str(tmp_path)
        mock_args = MagicMock()
        mock_args.ssl_model = None
        mock_args.model_path = None

        model_path = "suinleelab/monet"
        model_names = [model_path]

        run_name = generate_run_name(mock_args, model_names)
        cache_path = get_single_model_cache_path(model_path, output_dir)

        cache_folder = Path(cache_path).parent.parent.name

        assert run_name == cache_folder
        assert run_name == "suinleelab_monet"

    def test_single_model_with_assets_prefix_and_slashes(self, tmp_path):
        """Test model path with both 'assets/' prefix and slashes."""
        output_dir = str(tmp_path)
        mock_args = MagicMock()
        mock_args.ssl_model = None
        mock_args.model_path = None

        model_path = "assets/models/checkpoint-1000"
        model_names = [model_path]

        run_name = generate_run_name(mock_args, model_names)
        cache_path = get_single_model_cache_path(model_path, output_dir)

        cache_folder = Path(cache_path).parent.parent.name

        assert run_name == cache_folder
        # Should be "models_checkpoint-1000" (with "assets/" stripped, slashes -> underscores)
        assert run_name == "models_checkpoint-1000"

    def test_combined_models_with_assets_prefix(self, tmp_path):
        """Test combined run names also strip 'assets/' consistently."""
        mock_args = MagicMock()
        mock_args.ssl_model = None
        mock_args.model_path = None
        mock_args.svd_components = None
        mock_args.use_trained_projector = False

        model_names = ["assets/model1", "assets/model2"]

        run_name = generate_run_name(mock_args, model_names)

        # Combined run name should use cleaned model names
        # Should be "combined_model1_model2" (not "combined_assets_model1_assets_model2")
        assert run_name == "combined_model1_model2"
        assert "assets" not in run_name

    def test_combined_models_mixed_paths(self, tmp_path):
        """Test combined models with mixed path formats."""
        mock_args = MagicMock()
        mock_args.ssl_model = None
        mock_args.model_path = None
        mock_args.svd_components = None
        mock_args.use_trained_projector = False

        model_names = [
            "suinleelab/monet",
            "assets/clip-custom",
            "google/derm-foundation",
        ]

        run_name = generate_run_name(mock_args, model_names)

        # Should clean all paths consistently
        assert (
            "combined_suinleelab_monet_clip-custom_google_derm-foundation" == run_name
        )
        assert "assets" not in run_name

    def test_ssl_model_run_name(self, tmp_path):
        """Test SSL model run name generation."""
        mock_args = MagicMock()
        mock_args.ssl_model = "dino_qderma"
        mock_args.model_path = None

        model_names = ["some_model"]

        run_name = generate_run_name(mock_args, model_names)

        # SSL models use special naming
        assert run_name == "ssl_dino_qderma"
