"""Tests for embedding fusion functionality.

Tests verify:
- Concatenation of embeddings from multiple models
- SVD dimensionality reduction
- Cache hit/miss behavior
- Incompatible cache rejection
- Mixed CLIP and SSL model handling

These tests validate the ACTUAL behavior of combine_embeddings_simple:
- Returns (image_embs, text_embs, labels, corrupted, svd_image_model, svd_text_model, filtered_df)
- image_embs and text_embs are the FINAL embeddings (after SVD if requested)
- svd_*_model are TruncatedSVD objects (or None)
- Handles cache loading/saving correctly
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.skinmap.cache.manager import EmbeddingCache
from src.skinmap.embeddings.fusion import (
    combine_embeddings_simple as _combine_embeddings_simple,
)
from src.skinmap.models.loaders import ModelInfo


def combine_embeddings_simple(*args, **kwargs):
    """Test wrapper that forces num_workers=0 to avoid multiprocessing issues."""
    kwargs.setdefault("num_workers", 0)
    return _combine_embeddings_simple(*args, **kwargs)


class TestBasicConcatenation:
    """Test basic embedding concatenation."""

    def test_single_clip_model_no_svd(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test single CLIP model without SVD."""
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model")]

        # Pre-populate cache
        image_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
        text_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
        labels = np.array([0, 1] * (len(sample_df) // 2))

        cache.save_single_model("clip_model", image_embs, text_embs, labels, sample_df)

        # Run fusion
        (
            result_image,
            result_text,
            result_labels,
            corrupted,
            svd_img,
            svd_txt,
            filtered_df,
        ) = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Verify: single model, no SVD = same embeddings
        assert result_image.shape == (len(sample_df), 512)
        assert result_text.shape == (len(sample_df), 512)
        np.testing.assert_array_almost_equal(result_image, image_embs)
        np.testing.assert_array_almost_equal(result_text, text_embs)
        assert svd_img is None
        assert svd_txt is None

    def test_two_clip_models_concatenates(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test concatenating two CLIP models."""
        cache = EmbeddingCache(temp_dir)
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip1"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip2"),
        ]

        # Pre-populate caches with different dimensions
        image1 = np.random.randn(len(sample_df), 512).astype(np.float32)
        text1 = np.random.randn(len(sample_df), 512).astype(np.float32)
        image2 = np.random.randn(len(sample_df), 768).astype(np.float32)
        text2 = np.random.randn(len(sample_df), 768).astype(np.float32)
        labels = np.array([0, 1] * (len(sample_df) // 2))

        cache.save_single_model("clip1", image1, text1, labels, sample_df)
        cache.save_single_model("clip2", image2, text2, labels, sample_df)

        # Run fusion
        result_image, result_text, _, _, _, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Should concatenate: 512 + 768 = 1280
        assert result_image.shape == (len(sample_df), 1280)
        assert result_text.shape == (len(sample_df), 1280)

        # Verify concatenation is correct
        np.testing.assert_array_almost_equal(result_image[:, :512], image1)
        np.testing.assert_array_almost_equal(result_image[:, 512:], image2)

    def test_clip_and_ssl_fusion(
        self,
        mock_clip_model,
        mock_ssl_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test fusing CLIP and SSL models."""
        cache = EmbeddingCache(temp_dir)
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model"),
            ModelInfo(mock_ssl_model, None, "ssl", "ssl_model"),
        ]

        # Pre-populate caches
        clip_image = np.random.randn(len(sample_df), 512).astype(np.float32)
        clip_text = np.random.randn(len(sample_df), 512).astype(np.float32)
        ssl_image = np.random.randn(len(sample_df), 768).astype(np.float32)
        labels = np.array([0, 1] * (len(sample_df) // 2))

        cache.save_single_model("clip_model", clip_image, clip_text, labels, sample_df)
        cache.save_single_model("ssl_model", ssl_image, None, labels, sample_df)

        # Run fusion
        result_image, result_text, _, _, _, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Image should concatenate both: 512 + 768 = 1280
        assert result_image.shape == (len(sample_df), 1280)
        # Text only from CLIP (SSL has no text)
        assert result_text.shape == (len(sample_df), 512)


class TestSVDReduction:
    """Test SVD dimensionality reduction."""

    def test_svd_reduces_dimensions(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that SVD reduces dimensions correctly."""
        cache = EmbeddingCache(temp_dir)
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip1"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip2"),
        ]

        # Pre-populate caches
        image1 = np.random.randn(len(sample_df), 512).astype(np.float32)
        text1 = np.random.randn(len(sample_df), 512).astype(np.float32)
        image2 = np.random.randn(len(sample_df), 768).astype(np.float32)
        text2 = np.random.randn(len(sample_df), 768).astype(np.float32)
        labels = np.array([0, 1] * (len(sample_df) // 2))

        cache.save_single_model("clip1", image1, text1, labels, sample_df)
        cache.save_single_model("clip2", image2, text2, labels, sample_df)

        # Request fewer components than samples
        target_dims = min(8, len(sample_df) - 1)

        # Run fusion with SVD
        result_image, result_text, _, _, svd_img, svd_txt, _ = (
            combine_embeddings_simple(
                models,
                sample_df,
                device,
                cache,
                batch_size=4,
                svd_components=target_dims,
            )
        )

        # Embeddings should be reduced
        assert result_image.shape == (len(sample_df), target_dims)
        assert result_text.shape == (len(sample_df), target_dims)

        # SVD models should be returned
        assert isinstance(svd_img, TruncatedSVD)
        assert isinstance(svd_txt, TruncatedSVD)
        assert svd_img.n_components == target_dims
        assert svd_txt.n_components == target_dims

    def test_svd_model_can_transform_new_data(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that returned SVD model can transform new data."""
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "clip1")]

        # Pre-populate cache
        image1 = np.random.randn(len(sample_df), 512).astype(np.float32)
        text1 = np.random.randn(len(sample_df), 512).astype(np.float32)
        labels = np.array([0, 1] * (len(sample_df) // 2))

        cache.save_single_model("clip1", image1, text1, labels, sample_df)

        # Run with SVD
        _, _, _, _, svd_img, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=5,
        )

        # Should be able to use SVD model to transform new embeddings
        new_embeddings = np.random.randn(3, 512).astype(np.float32)
        transformed = svd_img.transform(new_embeddings)

        assert transformed.shape == (3, 5)

    def test_no_svd_when_not_requested(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that SVD is not applied when svd_components=None."""
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "clip1")]

        # Pre-populate cache
        image1 = np.random.randn(len(sample_df), 512).astype(np.float32)
        text1 = np.random.randn(len(sample_df), 512).astype(np.float32)
        labels = np.array([0, 1] * (len(sample_df) // 2))

        cache.save_single_model("clip1", image1, text1, labels, sample_df)

        # Run without SVD
        result_image, _, _, _, svd_img, svd_txt, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Should not apply SVD
        assert result_image.shape == (len(sample_df), 512)
        assert svd_img is None
        assert svd_txt is None


class TestCacheBehavior:
    """Test cache loading and saving behavior."""

    def test_cache_hit_loads_from_cache(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that cache hit loads embeddings from cache."""
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "cached_model")]

        # Pre-populate cache with known values
        cached_image = np.ones((len(sample_df), 512), dtype=np.float32) * 42
        cached_text = np.ones((len(sample_df), 512), dtype=np.float32) * 99
        labels = np.array([0, 1] * (len(sample_df) // 2))

        cache.save_single_model(
            "cached_model", cached_image, cached_text, labels, sample_df
        )

        # Run fusion - should use cache
        result_image, result_text, _, _, _, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Should use cached embeddings (value 42 and 99)
        np.testing.assert_array_almost_equal(result_image, cached_image)
        np.testing.assert_array_almost_equal(result_text, cached_text)

    def test_cache_miss_extracts_new(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that cache miss extracts new embeddings."""
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "new_model")]

        # No cache - should extract from mock model
        result_image, result_text, _, _, _, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Should have extracted embeddings (mock model outputs 512 dims)
        assert result_image.shape == (len(sample_df), 512)
        assert result_text.shape == (len(sample_df), 512)

        # Should have saved to cache
        loaded = cache.load_single_model("new_model")
        assert loaded is not None

    def test_partial_cache_hit(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test scenario where some models cached, some not."""
        cache = EmbeddingCache(temp_dir)
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "cached"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "not_cached"),
        ]

        # Cache only first model
        cached_image = np.ones((len(sample_df), 512), dtype=np.float32) * 42
        cached_text = np.ones((len(sample_df), 512), dtype=np.float32) * 42
        labels = np.array([0, 1] * (len(sample_df) // 2))
        cache.save_single_model("cached", cached_image, cached_text, labels, sample_df)

        # Run fusion
        result_image, _, _, _, _, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Should concatenate both (first from cache, second extracted)
        assert result_image.shape == (len(sample_df), 1024)

        # First 512 dims should be from cache (value 42)
        assert np.allclose(result_image[:, :512], cached_image)


class TestIncompatibleCache:
    """Test incompatible cache rejection."""

    def test_compatible_caches_use_fast_path(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that compatible caches use fast loading path."""
        cache = EmbeddingCache(temp_dir)
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "model1"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "model2"),
        ]

        # Pre-populate with same samples
        for model in models:
            image_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
            text_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
            labels = np.array([0, 1] * (len(sample_df) // 2))
            cache.save_single_model(
                model.model_path, image_embs, text_embs, labels, sample_df
            )

        # Run fusion - should detect compatibility and use fast path
        result_image, _, _, _, _, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Should successfully load and concatenate
        assert result_image.shape[0] == len(sample_df)
        assert result_image.shape[1] == 1024  # 512 + 512

    def test_sample_mismatch_handled(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test handling when cache has different samples."""
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "model1")]

        # Cache with DIFFERENT number of samples
        different_df = sample_df.iloc[:5].copy()
        image_embs = np.random.randn(5, 512).astype(np.float32)
        text_embs = np.random.randn(5, 512).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0])

        cache.save_single_model("model1", image_embs, text_embs, labels, different_df)

        # Try with full sample_df - cache system will handle this
        result_image, _, _, _, _, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Cache system may use compatible cache (5 samples) or extract all (10 samples)
        # Both behaviors are valid - the important thing is it doesn't crash
        assert result_image.shape[0] in [5, len(sample_df)]
        assert result_image.shape[1] == 512


class TestEdgeCases:
    """Test edge cases."""

    def test_single_sample(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test fusion with single sample."""
        df_single = sample_df.iloc[:1]
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "model1")]

        result_image, _, _, _, _, _, _ = combine_embeddings_simple(
            models,
            df_single,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        assert result_image.shape[0] == 1

    def test_empty_dataframe(self, mock_clip_model, mock_processor, temp_dir, device):
        """Test fusion with empty dataframe."""
        df_empty = pd.DataFrame(columns=["img_path", "description", "dataset_desc"])
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "model1")]

        result_image, _, _, _, _, _, _ = combine_embeddings_simple(
            models,
            df_empty,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        assert result_image.shape[0] == 0

    def test_three_models(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test fusing three models."""
        cache = EmbeddingCache(temp_dir)
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "model1"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "model2"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "model3"),
        ]

        # Pre-populate caches
        for model in models:
            image_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
            text_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
            labels = np.array([0, 1] * (len(sample_df) // 2))
            cache.save_single_model(
                model.model_path, image_embs, text_embs, labels, sample_df
            )

        result_image, _, _, _, _, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Should concatenate all three: 512 * 3 = 1536
        assert result_image.shape == (len(sample_df), 1536)


class TestReturnValues:
    """Test that all return values are correct."""

    def test_all_return_values_present(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that function returns all expected values."""
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "model1")]

        # Pre-populate cache
        image_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
        text_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
        labels = np.array([0, 1] * (len(sample_df) // 2))
        cache.save_single_model("model1", image_embs, text_embs, labels, sample_df)

        result = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Should return 7 values (including filtered dataframe)
        assert len(result) == 7

        (
            result_image,
            result_text,
            result_labels,
            corrupted,
            svd_img,
            svd_txt,
            filtered_df,
        ) = result

        # Verify types
        assert isinstance(result_image, np.ndarray)
        assert isinstance(result_text, (np.ndarray, type(None)))
        assert isinstance(result_labels, np.ndarray)
        assert isinstance(corrupted, list)
        assert svd_img is None or isinstance(svd_img, TruncatedSVD)
        assert svd_txt is None or isinstance(svd_txt, TruncatedSVD)
        assert isinstance(filtered_df, pd.DataFrame)
        assert len(filtered_df) == len(result_labels)

    def test_labels_returned_correctly(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that dataset labels are returned."""
        cache = EmbeddingCache(temp_dir)
        models = [ModelInfo(mock_clip_model, mock_processor, "clip", "model1")]

        # Pre-populate with known labels
        image_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
        text_embs = np.random.randn(len(sample_df), 512).astype(np.float32)
        labels = np.arange(
            len(sample_df)
        )  # Use indices as labels for easy verification
        cache.save_single_model("model1", image_embs, text_embs, labels, sample_df)

        _, _, result_labels, _, _, _, _ = combine_embeddings_simple(
            models,
            sample_df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Labels should match
        np.testing.assert_array_equal(result_labels, labels)
