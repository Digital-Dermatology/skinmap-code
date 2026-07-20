"""Tests for cache management functionality.

Tests cover:
- Cache configuration validation
- Compatible/incompatible cache loading
- Individual cache management
- Edge cases with dimension mismatches
"""

import numpy as np
import pandas as pd

from src.skinmap.cache.manager import CacheConfig, EmbeddingCache


class TestCacheConfig:
    """Test CacheConfig compatibility checking."""

    def test_identical_configs_are_compatible(self):
        """Identical configs should be compatible."""
        config1 = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )
        config2 = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )
        assert config1.is_compatible_with(config2)

    def test_different_fusion_methods_incompatible(self):
        """Different fusion methods should be incompatible."""
        config1 = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )
        config2 = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            fusion_method="svd_512",
        )
        assert not config1.is_compatible_with(config2)

    def test_different_whitening_dim_incompatible_for_projector(self):
        """Different whitening dims incompatible for trained projector."""
        config1 = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )
        config2 = CacheConfig(
            max_whitening_dim=512,
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )
        assert not config1.is_compatible_with(config2)

    def test_different_skip_whitening_incompatible(self):
        """Different skip_whitening values incompatible for projector."""
        config1 = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )
        config2 = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=True,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )
        assert not config1.is_compatible_with(config2)

    def test_whitening_params_dont_matter_for_simple_fusion(self):
        """Whitening params don't matter for simple fusion methods."""
        config1 = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            fusion_method="none",
        )
        config2 = CacheConfig(
            max_whitening_dim=512,
            skip_whitening=True,
            fusion_method="none",
        )
        # Both are "none" fusion, so compatible
        assert config1.is_compatible_with(config2)

    def test_cache_config_roundtrip_and_svd_mismatch(self):
        """Round-trip via dict should preserve values; SVD components must match."""
        config = CacheConfig(
            max_whitening_dim=256,
            skip_whitening=True,
            svd_components=1024,
            fusion_method="svd_1024",
        )
        restored = CacheConfig.from_dict(config.to_dict())
        assert restored.svd_components == 1024
        incompatible = CacheConfig(
            max_whitening_dim=256,
            skip_whitening=True,
            svd_components=512,
            fusion_method="svd_1024",
        )
        assert not restored.is_compatible_with(incompatible)


class TestEmbeddingCache:
    """Test EmbeddingCache save/load functionality."""

    def test_save_and_load_single_model(self, temp_dir, sample_embeddings):
        """Test saving and loading single model embeddings."""
        cache = EmbeddingCache(temp_dir)
        image_embs, text_embs, labels = sample_embeddings

        df = pd.DataFrame(
            {
                "img_path": [f"img_{i}.jpg" for i in range(len(labels))],
                "description": [f"desc_{i}" for i in range(len(labels))],
            }
        )

        # Save
        cache.save_single_model("test_model", image_embs, text_embs, labels, df)

        # Load
        result = cache.load_single_model("test_model")
        assert result is not None

        loaded_image, loaded_text, loaded_labels, loaded_df = result
        np.testing.assert_array_equal(loaded_image, image_embs)
        np.testing.assert_array_equal(loaded_text, text_embs)
        np.testing.assert_array_equal(loaded_labels, labels)
        assert len(loaded_df) == len(df)

    def test_load_nonexistent_cache_returns_none(self, temp_dir):
        """Loading non-existent cache should return None."""
        cache = EmbeddingCache(temp_dir)
        result = cache.load_single_model("nonexistent_model")
        assert result is None

    def test_save_and_load_combined_with_config(self, temp_dir, sample_embeddings):
        """Test saving and loading combined embeddings with config."""
        cache = EmbeddingCache(temp_dir)
        image_embs, text_embs, labels = sample_embeddings

        config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )

        # Save
        cache.save_combined(
            "test_run",
            image_embs,
            text_embs,
            labels,
            config=config,
            extra_data={"test": "data"},
        )

        # Load with matching config
        result = cache.load_combined("test_run", expected_config=config)
        assert result is not None
        loaded_img, loaded_text, loaded_labels, metadata = result
        np.testing.assert_array_equal(loaded_img, image_embs)
        np.testing.assert_array_equal(loaded_text, text_embs)
        np.testing.assert_array_equal(loaded_labels, labels)
        assert metadata["config"].fusion_method == "trained_projector"
        assert metadata["corrupted_indices"] == []

    def test_load_incompatible_config_returns_none(self, temp_dir, sample_embeddings):
        """Loading with incompatible config should return None."""
        cache = EmbeddingCache(temp_dir)
        image_embs, text_embs, labels = sample_embeddings

        save_config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )

        # Save with one config
        cache.save_combined(
            "test_run",
            image_embs,
            text_embs,
            labels,
            config=save_config,
        )

        # Try to load with incompatible config
        load_config = CacheConfig(
            max_whitening_dim=512,  # Different!
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )

        result = cache.load_combined("test_run", expected_config=load_config)
        assert result is None  # Should reject incompatible cache

    def test_save_combined_with_numpy_none_text(self, temp_dir, sample_embeddings):
        """Numpy object None text embeddings should round-trip cleanly."""
        cache = EmbeddingCache(temp_dir)
        image_embs, _, labels = sample_embeddings
        text_embs = np.array(None, dtype=object)
        config = CacheConfig(
            max_whitening_dim=512,
            skip_whitening=True,
            fusion_method="none",
        )

        cache.save_combined(
            "np_none",
            image_embs,
            text_embs,
            labels,
            config=config,
        )

        result = cache.load_combined("np_none")
        assert result is not None
        _, loaded_text, _, metadata = result
        assert loaded_text is None
        assert metadata["text_dim"] == 0

    def test_load_combined_without_expected_config_returns_metadata(
        self, temp_dir, sample_embeddings
    ):
        """Loading without passing expected config should still return metadata."""
        cache = EmbeddingCache(temp_dir)
        image_embs, text_embs, labels = sample_embeddings
        config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            fusion_method="concat",
        )

        cache.save_combined(
            "meta_only",
            image_embs,
            text_embs,
            labels,
            config=config,
            extra_data={"corrupted_indices": np.array([1, 2], dtype=np.int64)},
        )

        result = cache.load_combined("meta_only")
        assert result is not None
        _, _, _, metadata = result
        assert metadata["corrupted_indices"] == [1, 2]

    def test_check_individual_compatibility_all_present(
        self, temp_dir, sample_embeddings
    ):
        """Test checking compatibility when all caches present."""
        cache = EmbeddingCache(temp_dir)
        image_embs, text_embs, labels = sample_embeddings

        df = pd.DataFrame(
            {
                "img_path": [f"img_{i}.jpg" for i in range(len(labels))],
            }
        )

        # Save two model caches
        cache.save_single_model("model1", image_embs, text_embs, labels, df)
        cache.save_single_model("model2", image_embs, text_embs, labels, df)

        # Check compatibility
        compatible, dfs = cache.check_individual_compatibility(["model1", "model2"], df)
        assert compatible
        assert len(dfs) == 2

    def test_check_individual_compatibility_missing_cache(
        self, temp_dir, sample_embeddings
    ):
        """Test compatibility check when cache is missing."""
        cache = EmbeddingCache(temp_dir)
        image_embs, text_embs, labels = sample_embeddings

        df = pd.DataFrame(
            {
                "img_path": [f"img_{i}.jpg" for i in range(len(labels))],
            }
        )

        # Save only one cache
        cache.save_single_model("model1", image_embs, text_embs, labels, df)

        # Check with missing model2
        compatible, dfs = cache.check_individual_compatibility(["model1", "model2"], df)
        assert not compatible
        assert len(dfs) == 0

    def test_check_individual_compatibility_length_mismatch(self, temp_dir):
        """Test compatibility check with length mismatch."""
        cache = EmbeddingCache(temp_dir)

        # Create different sized embeddings
        image_embs1 = np.random.randn(10, 512).astype(np.float32)
        text_embs1 = np.random.randn(10, 512).astype(np.float32)
        labels1 = np.array([0, 1] * 5)
        df1 = pd.DataFrame({"img_path": [f"img_{i}.jpg" for i in range(10)]})

        image_embs2 = np.random.randn(8, 512).astype(np.float32)  # Different length!
        text_embs2 = np.random.randn(8, 512).astype(np.float32)
        labels2 = np.array([0, 1] * 4)
        df2 = pd.DataFrame({"img_path": [f"img_{i}.jpg" for i in range(8)]})

        # Save both
        cache.save_single_model("model1", image_embs1, text_embs1, labels1, df1)
        cache.save_single_model("model2", image_embs2, text_embs2, labels2, df2)

        # Check compatibility - should fail due to length mismatch
        compatible, dfs = cache.check_individual_compatibility(
            ["model1", "model2"], df1
        )
        assert not compatible

    def test_check_individual_compatibility_sample_mismatch(self, temp_dir):
        """Test compatibility check with sample order mismatch."""
        cache = EmbeddingCache(temp_dir)

        image_embs = np.random.randn(5, 512).astype(np.float32)
        text_embs = np.random.randn(5, 512).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0])

        # Different sample orders
        df1 = pd.DataFrame({"img_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]})
        df2 = pd.DataFrame(
            {"img_path": ["a.jpg", "c.jpg", "b.jpg", "d.jpg", "e.jpg"]}
        )  # Swapped!

        cache.save_single_model("model1", image_embs, text_embs, labels, df1)
        cache.save_single_model("model2", image_embs, text_embs, labels, df2)

        # Check compatibility - should pass (same set, order can be aligned)
        compatible, dfs = cache.check_individual_compatibility(
            ["model1", "model2"], df1
        )
        assert compatible


class TestCacheEdgeCases:
    """Test edge cases in caching."""

    def test_empty_embeddings(self, temp_dir):
        """Test caching empty embeddings."""
        cache = EmbeddingCache(temp_dir)

        image_embs = np.array([]).reshape(0, 512).astype(np.float32)
        text_embs = np.array([]).reshape(0, 512).astype(np.float32)
        labels = np.array([])
        df = pd.DataFrame()

        # Should handle gracefully
        cache.save_single_model("empty", image_embs, text_embs, labels, df)
        result = cache.load_single_model("empty")
        assert result is not None
        assert len(result[0]) == 0

    def test_cache_with_special_characters_in_name(self, temp_dir, sample_embeddings):
        """Test caching with special characters in model name."""
        cache = EmbeddingCache(temp_dir)
        image_embs, text_embs, labels = sample_embeddings

        df = pd.DataFrame({"img_path": [f"img_{i}.jpg" for i in range(len(labels))]})

        # Model name with special characters
        model_name = "model/with/slashes"
        cache.save_single_model(model_name, image_embs, text_embs, labels, df)

        result = cache.load_single_model(model_name)
        assert result is not None

    def test_overwrite_existing_cache(self, temp_dir, sample_embeddings):
        """Test overwriting existing cache."""
        cache = EmbeddingCache(temp_dir)
        image_embs, text_embs, labels = sample_embeddings

        df = pd.DataFrame({"img_path": [f"img_{i}.jpg" for i in range(len(labels))]})

        # Save once
        cache.save_single_model("model", image_embs, text_embs, labels, df)

        # Save again with different data
        new_image_embs = image_embs * 2
        cache.save_single_model("model", new_image_embs, text_embs, labels, df)

        # Should load the new data
        result = cache.load_single_model("model")
        np.testing.assert_array_equal(result[0], new_image_embs)
