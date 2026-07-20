"""Tests for metadata prediction caching functionality.

Tests cover:
- Cache reuse when predictions are complete
- Cache rejection when predictions are incomplete
- Cache rejection when columns are missing
- Dimension preservation (critical for embeddings alignment)
- Handling of duplicate img_paths in cache
"""


import pandas as pd
import pytest

from src.create_skinmap import reuse_cached_metadata_predictions


class TestReuseMetadataPredictions:
    """Test reuse_cached_metadata_predictions function."""

    @pytest.fixture
    def sample_filtered_df(self):
        """Create a sample filtered_df with missing metadata."""
        return pd.DataFrame(
            {
                "img_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "laterality": ["left", None, "right"],
                "age": [25, 30, None],
            }
        )

    @pytest.fixture
    def complete_cached_df(self):
        """Create a cached df with complete predictions."""
        return pd.DataFrame(
            {
                "img_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "laterality": ["left", None, "right"],
                "laterality_pred": ["left", "right", "right"],
                "age": [25, 30, None],
                "age_pred": [25.0, 30.0, 35.0],
            }
        )

    @pytest.fixture
    def incomplete_cached_df(self):
        """Create a cached df with incomplete predictions."""
        return pd.DataFrame(
            {
                "img_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "laterality": ["left", None, "right"],
                "laterality_pred": [
                    "left",
                    None,
                    "right",
                ],  # Missing prediction for img2
                "age": [25, 30, None],
                "age_pred": [25.0, 30.0, 35.0],
            }
        )

    def test_cache_reuse_with_complete_predictions(
        self, sample_filtered_df, complete_cached_df, tmp_path
    ):
        """Should reuse cache when all predictions are complete."""
        cached_path = tmp_path / "cached.csv"
        complete_cached_df.to_csv(cached_path, index=False)

        result_df, reused = reuse_cached_metadata_predictions(
            sample_filtered_df, ["laterality", "age"], str(cached_path)
        )

        assert reused is True
        assert len(result_df) == len(sample_filtered_df)  # Dimension preserved
        assert "laterality_pred" in result_df.columns
        assert "age_pred" in result_df.columns
        assert result_df["laterality_pred"].tolist() == ["left", "right", "right"]
        assert result_df["age_pred"].tolist() == [25.0, 30.0, 35.0]

    def test_cache_rejection_with_incomplete_predictions(
        self, sample_filtered_df, incomplete_cached_df, tmp_path
    ):
        """Should reject cache when predictions are incomplete."""
        cached_path = tmp_path / "cached.csv"
        incomplete_cached_df.to_csv(cached_path, index=False)

        result_df, reused = reuse_cached_metadata_predictions(
            sample_filtered_df, ["laterality", "age"], str(cached_path)
        )

        assert reused is False
        assert len(result_df) == len(sample_filtered_df)  # Dimension preserved
        # Should return original df unchanged
        pd.testing.assert_frame_equal(result_df, sample_filtered_df)

    def test_cache_rejection_with_missing_columns(
        self, sample_filtered_df, complete_cached_df, tmp_path
    ):
        """Should reject cache when required prediction columns are missing."""
        cached_path = tmp_path / "cached.csv"
        # Remove age_pred column
        incomplete_cache = complete_cached_df.drop(columns=["age_pred"])
        incomplete_cache.to_csv(cached_path, index=False)

        result_df, reused = reuse_cached_metadata_predictions(
            sample_filtered_df, ["laterality", "age"], str(cached_path)
        )

        assert reused is False
        assert len(result_df) == len(sample_filtered_df)  # Dimension preserved
        # Should return original df unchanged
        pd.testing.assert_frame_equal(result_df, sample_filtered_df)

    def test_cache_rejection_when_no_cache_file(self, sample_filtered_df):
        """Should reject cache when cache file doesn't exist."""
        result_df, reused = reuse_cached_metadata_predictions(
            sample_filtered_df, ["laterality", "age"], "/nonexistent/path.csv"
        )

        assert reused is False
        assert len(result_df) == len(sample_filtered_df)  # Dimension preserved
        pd.testing.assert_frame_equal(result_df, sample_filtered_df)

    def test_cache_rejection_when_no_attributes(
        self, sample_filtered_df, complete_cached_df, tmp_path
    ):
        """Should reject cache when no attributes requested."""
        cached_path = tmp_path / "cached.csv"
        complete_cached_df.to_csv(cached_path, index=False)

        result_df, reused = reuse_cached_metadata_predictions(
            sample_filtered_df, [], str(cached_path)
        )

        assert reused is False
        assert len(result_df) == len(sample_filtered_df)  # Dimension preserved
        pd.testing.assert_frame_equal(result_df, sample_filtered_df)

    def test_dimension_preservation_with_subset_match(
        self, sample_filtered_df, complete_cached_df, tmp_path
    ):
        """Should preserve dimensions when filtered_df is subset of cache."""
        cached_path = tmp_path / "cached.csv"
        # Add extra row to cache that's not in filtered_df
        extra_cache = pd.concat(
            [
                complete_cached_df,
                pd.DataFrame(
                    {
                        "img_path": ["img4.jpg"],
                        "laterality": [None],
                        "laterality_pred": ["left"],
                        "age": [40],
                        "age_pred": [40.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        extra_cache.to_csv(cached_path, index=False)

        result_df, reused = reuse_cached_metadata_predictions(
            sample_filtered_df, ["laterality", "age"], str(cached_path)
        )

        assert reused is True
        # CRITICAL: Must preserve original dimensions
        assert len(result_df) == len(sample_filtered_df)
        assert len(result_df) == 3  # Not 4

    def test_dimension_preservation_when_cache_has_duplicates(
        self, sample_filtered_df, tmp_path
    ):
        """Should raise error when cache has duplicate img_paths that would break alignment."""
        cached_path = tmp_path / "cached.csv"
        # Create cache with duplicate img_path
        duplicate_cache = pd.DataFrame(
            {
                "img_path": ["img1.jpg", "img1.jpg", "img2.jpg", "img3.jpg"],
                "laterality": ["left", "left", None, "right"],
                "laterality_pred": ["left", "left", "right", "right"],
                "age": [25, 25, 30, None],
                "age_pred": [25.0, 25.0, 30.0, 35.0],
            }
        )
        duplicate_cache.to_csv(cached_path, index=False)

        # CRITICAL: Must raise error to prevent dimension mismatch with embeddings
        with pytest.raises(RuntimeError, match="CACHE CORRUPTION"):
            result_df, reused = reuse_cached_metadata_predictions(
                sample_filtered_df, ["laterality", "age"], str(cached_path)
            )

    def test_cache_rejection_when_missing_img_path_column(
        self, sample_filtered_df, complete_cached_df, tmp_path
    ):
        """Should reject cache when img_path column is missing."""
        cached_path = tmp_path / "cached.csv"
        # Remove img_path column
        bad_cache = complete_cached_df.drop(columns=["img_path"])
        bad_cache.to_csv(cached_path, index=False)

        result_df, reused = reuse_cached_metadata_predictions(
            sample_filtered_df, ["laterality", "age"], str(cached_path)
        )

        assert reused is False
        assert len(result_df) == len(sample_filtered_df)  # Dimension preserved
        pd.testing.assert_frame_equal(result_df, sample_filtered_df)

    def test_partial_attribute_coverage(
        self, sample_filtered_df, complete_cached_df, tmp_path
    ):
        """Should reject cache when only some attributes have predictions."""
        cached_path = tmp_path / "cached.csv"
        # Only have laterality_pred, missing age_pred
        partial_cache = complete_cached_df.drop(columns=["age_pred"])
        partial_cache.to_csv(cached_path, index=False)

        result_df, reused = reuse_cached_metadata_predictions(
            sample_filtered_df, ["laterality", "age"], str(cached_path)
        )

        assert reused is False
        assert len(result_df) == len(sample_filtered_df)  # Dimension preserved
        pd.testing.assert_frame_equal(result_df, sample_filtered_df)

    def test_all_values_present_no_predictions_needed(
        self, complete_cached_df, tmp_path
    ):
        """Should handle case where no values are missing (edge case)."""
        # Create df where all values are present
        df_no_missing = pd.DataFrame(
            {
                "img_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "laterality": ["left", "right", "right"],
                "age": [25, 30, 35],
            }
        )

        cached_path = tmp_path / "cached.csv"
        complete_cached_df.to_csv(cached_path, index=False)

        result_df, reused = reuse_cached_metadata_predictions(
            df_no_missing, ["laterality", "age"], str(cached_path)
        )

        # Should still reuse cache and add pred columns
        assert reused is True
        assert len(result_df) == len(df_no_missing)  # Dimension preserved
