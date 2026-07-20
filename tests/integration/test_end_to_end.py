"""End-to-end integration tests.

Tests the full pipeline from data loading to embedding extraction to caching.
"""


import numpy as np
import pandas as pd

from src.skinmap.cache.manager import CacheConfig, EmbeddingCache
from src.skinmap.data.preprocessing import normalize_multilabel_columns
from src.skinmap.embeddings.extractors import (
    extract_clip_embeddings,
    extract_ssl_embeddings,
)
from src.skinmap.embeddings.fusion import combine_embeddings_simple
from src.skinmap.embeddings.projector import extract_combined_embeddings_with_projector
from src.skinmap.models.loaders import ModelInfo


class TestSingleModelPipeline:
    """Test complete single model pipeline."""

    def test_clip_model_full_pipeline(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test full pipeline with single CLIP model."""
        # Step 1: Normalize data
        df = normalize_multilabel_columns(sample_df)

        # Step 2: Extract embeddings
        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        # Verify extraction
        assert len(image_embs) == len(filtered_df)
        assert len(corrupted) == 0  # No corrupted samples in fixture

        # Step 3: Cache embeddings
        cache = EmbeddingCache(temp_dir)
        config = CacheConfig(
            max_whitening_dim=512,
            skip_whitening=False,
            fusion_method="none",
        )

        cache.save_combined(
            "test_run",
            image_embs,
            text_embs,
            labels,
            config=config,
        )

        # Step 4: Load from cache
        cached_data = cache.load_combined("test_run", expected_config=config)
        assert cached_data is not None

        cached_image, cached_text, cached_labels, _ = cached_data
        np.testing.assert_array_equal(cached_image, image_embs)

    def test_ssl_model_full_pipeline(self, mock_ssl_model, sample_df, temp_dir, device):
        """Test full pipeline with single SSL model."""
        df = normalize_multilabel_columns(sample_df)

        image_embs, text_embs, labels, corrupted, filtered_df = extract_ssl_embeddings(
            mock_ssl_model,
            df,
            device,
            batch_size=4,
        )

        assert len(image_embs) == len(filtered_df)
        assert text_embs is None  # SSL doesn't produce text embeddings

        # Cache and reload
        cache = EmbeddingCache(temp_dir)
        config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            fusion_method="none",
        )

        cache.save_combined("ssl_run", image_embs, None, labels, config=config)
        cached_data = cache.load_combined("ssl_run", expected_config=config)

        assert cached_data is not None


class TestMultiModelPipeline:
    """Test multi-model fusion pipeline."""

    def test_simple_fusion_pipeline(
        self,
        mock_clip_model,
        mock_ssl_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test simple concatenation + SVD fusion."""
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        # Create model infos
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model"),
            ModelInfo(mock_ssl_model, None, "ssl", "ssl_model"),
        ]

        # Run fusion (request 8 components since we only have 10 samples)
        # SVD can produce at most min(n_samples, n_features) components
        (
            image_embs,
            text_embs,
            labels,
            corrupted,
            svd_image,
            svd_text,
            _,
        ) = combine_embeddings_simple(
            models,
            df,
            device,
            cache,
            batch_size=4,
            svd_components=8,  # Must be < n_samples (10)
        )

        # Verify fusion
        assert len(image_embs) == len(df)
        assert image_embs.shape[1] == 8  # SVD reduced

    def test_fusion_uses_individual_caches(
        self,
        mock_clip_model,
        mock_ssl_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test that fusion reuses individual model caches."""
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        # Pre-populate individual caches
        image_embs1 = np.random.randn(len(df), 512).astype(np.float32)
        text_embs1 = np.random.randn(len(df), 512).astype(np.float32)
        labels = np.array([0, 1] * (len(df) // 2))

        cache.save_single_model("clip_model", image_embs1, text_embs1, labels, df)

        image_embs2 = np.random.randn(len(df), 768).astype(np.float32)
        cache.save_single_model("ssl_model", image_embs2, None, labels, df)

        # Now run fusion - should use cached embeddings
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model"),
            ModelInfo(mock_ssl_model, None, "ssl", "ssl_model"),
        ]

        (
            combined_image,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = combine_embeddings_simple(
            models,
            df,
            device,
            cache,
            batch_size=4,
            svd_components=None,
        )

        # Should have concatenated dimensions
        assert combined_image.shape[1] == 512 + 768


class TestPipelineWithCorruptedSamples:
    """Test pipeline with corrupted samples."""

    def test_corrupted_samples_filtered_correctly(
        self,
        mock_clip_model,
        mock_processor,
        sample_df_with_corrupted,
        temp_dir,
        device,
    ):
        """Test that corrupted samples are filtered throughout pipeline."""
        df, num_corrupted = sample_df_with_corrupted

        # Extract embeddings
        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        # Verify filtering
        assert len(filtered_df) == len(df) - num_corrupted
        assert len(image_embs) == len(filtered_df)
        assert len(corrupted) == num_corrupted

        # Embeddings and dataframe should be in sync
        for i in range(len(filtered_df)):
            # Index should match
            assert i < len(image_embs)

    def test_cache_preserves_filtering(
        self,
        mock_clip_model,
        mock_processor,
        sample_df_with_corrupted,
        temp_dir,
        device,
    ):
        """Test that cached data preserves filtering."""
        df, num_corrupted = sample_df_with_corrupted

        # Extract with filtering
        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        # Cache the filtered results
        cache = EmbeddingCache(temp_dir)
        config = CacheConfig(
            max_whitening_dim=512,
            skip_whitening=False,
            fusion_method="none",
        )

        cache.save_combined(
            "filtered_run", image_embs, text_embs, labels, config=config
        )

        # Load from cache
        cached_data = cache.load_combined("filtered_run", expected_config=config)
        assert cached_data is not None

        cached_image, _, _, _ = cached_data

        # Cached data should have same filtered length
        assert len(cached_image) == len(df) - num_corrupted


class TestPipelineEdgeCases:
    """Test edge cases in full pipeline."""

    def test_empty_dataframe_pipeline(
        self, mock_clip_model, mock_processor, temp_dir, device
    ):
        """Test pipeline with empty dataframe."""
        df = pd.DataFrame(columns=["img_path", "description", "dataset_desc"])

        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        assert len(image_embs) == 0
        assert len(filtered_df) == 0

    def test_single_sample_pipeline(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test pipeline with single sample."""
        df_single = sample_df.iloc[:1]

        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df_single,
            device,
            batch_size=4,
        )

        assert len(image_embs) == 1
        assert len(filtered_df) == 1

        # Should be cacheable
        cache = EmbeddingCache(temp_dir)
        config = CacheConfig(
            max_whitening_dim=512,
            skip_whitening=False,
            fusion_method="none",
        )

        cache.save_combined("single_run", image_embs, text_embs, labels, config=config)

        cached_data = cache.load_combined("single_run", expected_config=config)
        assert cached_data is not None

    def test_all_corrupted_samples_pipeline(
        self, mock_clip_model, mock_processor, temp_dir, device
    ):
        """Test pipeline when all samples are corrupted."""
        df = pd.DataFrame(
            {
                "img_path": ["/corrupt1.jpg", "/corrupt2.jpg", "/corrupt3.jpg"],
                "description": ["Test"] * 3,
                "dataset_desc": ["A"] * 3,
            }
        )

        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        # All should be filtered out
        assert len(filtered_df) == 0
        assert len(image_embs) == 0
        assert len(corrupted) == 3

    def test_config_mismatch_prevents_cache_load(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that config mismatch prevents loading incompatible cache."""
        df = normalize_multilabel_columns(sample_df)

        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        # Save with one config
        cache = EmbeddingCache(temp_dir)
        save_config = CacheConfig(
            max_whitening_dim=768,
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )

        cache.save_combined(
            "test_run", image_embs, text_embs, labels, config=save_config
        )

        # Try to load with different config
        load_config = CacheConfig(
            max_whitening_dim=512,  # Different!
            skip_whitening=False,
            projector_dim=768,
            projector_type="linear",
            fusion_method="trained_projector",
        )

        cached_data = cache.load_combined("test_run", expected_config=load_config)

        # Should not load due to mismatch
        assert cached_data is None


class TestCriticalBugFixes:
    """Integration tests for the critical bug fixes."""

    def test_dimension_mismatch_bug_fixed(
        self, mock_clip_model, mock_processor, sample_df, temp_dir, device
    ):
        """Test that dimension mismatch bug is fixed in full pipeline.

        Original bug: stored original embeddings but expected whitened dims.
        This test verifies the fix works end-to-end.
        """
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        # Extract embeddings
        image_embs, text_embs, labels, _, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        # Original dims
        original_dim = image_embs.shape[1]

        # Simulate dimension truncation scenario
        truncated_dim = min(256, original_dim)

        # The fix ensures this works without errors
        config = CacheConfig(
            max_whitening_dim=truncated_dim,
            skip_whitening=False,
            projector_dim=truncated_dim,
            projector_type="linear",
            fusion_method="trained_projector",
        )

        # Should save and load without dimension errors
        cache.save_combined("dim_test", image_embs, text_embs, labels, config=config)

        cached_data = cache.load_combined("dim_test", expected_config=config)
        assert cached_data is not None

    def test_corrupted_sample_sync_maintained(
        self,
        mock_clip_model,
        mock_processor,
        sample_df_with_corrupted,
        temp_dir,
        device,
    ):
        """Test that DataFrame-embedding sync is maintained with corrupted samples.

        Original bug: embeddings and dataframe could get out of sync.
        This test verifies they stay synchronized.
        """
        df, num_corrupted = sample_df_with_corrupted

        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        # Critical: lengths must match
        assert len(image_embs) == len(filtered_df)

        # Each embedding should correspond to its row in dataframe
        for i in range(len(filtered_df)):
            # Can access both without index errors
            assert i < len(image_embs)
            assert i < len(filtered_df)

    def test_idx_zero_corrupted_handled(
        self, mock_clip_model, mock_processor, temp_dir, device
    ):
        """Test that idx=0 corrupted is handled correctly.

        Original bug: -0 == 0, so idx=0 corrupted wasn't detected.
        Fix: use -idx-1 instead of -idx.
        """
        # First sample corrupted
        df = pd.DataFrame(
            {
                "img_path": ["/corrupt.jpg", temp_dir + "/valid.jpg"],
                "description": ["Test1", "Test2"],
                "dataset_desc": ["A", "A"],
            }
        )

        # Create the valid image
        from PIL import Image

        img = Image.new("RGB", (224, 224))
        img.save(temp_dir + "/valid.jpg")

        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        # idx=0 should be detected as corrupted
        assert 0 in corrupted
        assert len(filtered_df) == 1
        # Only the valid sample should remain
        assert "/valid.jpg" in filtered_df["img_path"].values[0]


class TestTrainedProjectorPipeline:
    """Integration tests for trained projector pipeline.

    This is the critical test that would have caught the whitened_image_embeddings
    deletion bug. These tests exercise the full extract_combined_embeddings_with_projector
    function end-to-end.
    """

    def test_trained_projector_basic_pipeline(
        self,
        mock_clip_model,
        mock_ssl_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test basic trained projector pipeline with multiple models."""
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        # Create model infos for CLIP + SSL
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model_1"),
            ModelInfo(mock_ssl_model, None, "ssl", "ssl_model_1"),
        ]

        # Run trained projector pipeline
        (
            image_embs,
            text_embs,
            labels,
            corrupted,
            projector_model,
            whitening_stats,
            filtered_df,
            wandb_run,
        ) = extract_combined_embeddings_with_projector(
            models=models,
            df=df,
            device=device,
            cache=cache,
            batch_size=4,
            projector_dim=8,  # Must be < n_samples (10)
            projector_type="linear",
            n_epochs=2,  # Just a few epochs for testing
            skip_whitening=False,
        )

        # Verify outputs
        assert len(image_embs) == len(df)
        assert len(text_embs) == len(df)
        assert image_embs.shape[1] == 8  # projector_dim
        assert text_embs.shape[1] == 8
        assert projector_model is not None
        assert whitening_stats is not None
        assert len(filtered_df) == len(df)

    def test_trained_projector_with_skip_whitening(
        self,
        mock_clip_model,
        mock_ssl_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test trained projector pipeline with skip_whitening=True."""
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model_2"),
            ModelInfo(mock_ssl_model, None, "ssl", "ssl_model_2"),
        ]

        # Run with skip_whitening=True (different code path!)
        (
            image_embs,
            text_embs,
            labels,
            corrupted,
            projector_model,
            whitening_stats,
            filtered_df,
            wandb_run,
        ) = extract_combined_embeddings_with_projector(
            models=models,
            df=df,
            device=device,
            cache=cache,
            batch_size=4,
            projector_dim=8,  # Must be < n_samples (10)
            projector_type="linear",
            n_epochs=2,
            skip_whitening=True,  # This is the key difference
        )

        # Verify outputs
        assert len(image_embs) == len(df)
        assert image_embs.shape[1] == 8
        assert projector_model is not None

    def test_trained_projector_mlp_projector(
        self,
        mock_clip_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test trained projector with MLP projector type."""
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model_3"),
        ]

        (
            image_embs,
            text_embs,
            labels,
            corrupted,
            projector_model,
            whitening_stats,
            filtered_df,
            wandb_run,
        ) = extract_combined_embeddings_with_projector(
            models=models,
            df=df,
            device=device,
            cache=cache,
            batch_size=4,
            projector_dim=8,  # Must be < n_samples (10)
            projector_type="mlp",  # Test MLP projector
            n_epochs=2,
            skip_whitening=False,
        )

        # Verify MLP projector works
        assert len(image_embs) == len(df)
        assert image_embs.shape[1] == 8
        assert projector_model is not None

    def test_trained_projector_with_dimension_truncation(
        self,
        mock_clip_model,
        mock_ssl_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test trained projector with dimension truncation via max_whitening_dim.

        This specifically tests the scenario that caused the original dimension bug.
        """
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model_4"),
            ModelInfo(mock_ssl_model, None, "ssl", "ssl_model_4"),
        ]

        # Use dimension truncation
        (
            image_embs,
            text_embs,
            labels,
            corrupted,
            projector_model,
            whitening_stats,
            filtered_df,
            wandb_run,
        ) = extract_combined_embeddings_with_projector(
            models=models,
            df=df,
            device=device,
            cache=cache,
            batch_size=4,
            projector_dim=8,  # Must be < n_samples (10)
            projector_type="linear",
            n_epochs=2,
            max_whitening_dim=256,  # Truncate dimensions
            skip_whitening=False,
        )

        # Should complete without dimension errors
        assert len(image_embs) == len(df)
        assert image_embs.shape[1] == 8

    def test_trained_projector_multiple_clip_models(
        self,
        mock_clip_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test trained projector with multiple CLIP models (tests text embedding path)."""
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        # Multiple CLIP models (all provide text embeddings)
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model_5a"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model_5b"),
        ]

        (
            image_embs,
            text_embs,
            labels,
            corrupted,
            projector_model,
            whitening_stats,
            filtered_df,
            wandb_run,
        ) = extract_combined_embeddings_with_projector(
            models=models,
            df=df,
            device=device,
            cache=cache,
            batch_size=4,
            projector_dim=8,  # Must be < n_samples (10)
            projector_type="linear",
            n_epochs=2,
            skip_whitening=False,
        )

        # Should handle multiple text embeddings correctly
        assert len(image_embs) == len(df)
        assert len(text_embs) == len(df)
        assert image_embs.shape[1] == 8
        assert text_embs.shape[1] == 8

    def test_trained_projector_with_domain_balanced_sampler(
        self,
        mock_clip_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test trained projector with domain-balanced sampling."""
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "clip_model_6"),
        ]

        (
            image_embs,
            text_embs,
            labels,
            corrupted,
            projector_model,
            whitening_stats,
            filtered_df,
            wandb_run,
        ) = extract_combined_embeddings_with_projector(
            models=models,
            df=df,
            device=device,
            cache=cache,
            batch_size=4,
            projector_dim=8,  # Must be < n_samples (10)
            projector_type="linear",
            n_epochs=2,
            use_domain_balanced=True,  # Test domain balancing
            skip_whitening=False,
        )

        assert len(image_embs) == len(df)
        assert image_embs.shape[1] == 8

    def test_trained_projector_caches_individual_models(
        self,
        mock_clip_model,
        mock_ssl_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Test that trained projector uses cached individual model embeddings."""
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "cached_clip"),
            ModelInfo(mock_ssl_model, None, "ssl", "cached_ssl"),
        ]

        # First run - will cache individual models
        (
            image_embs_1,
            text_embs_1,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = extract_combined_embeddings_with_projector(
            models=models,
            df=df,
            device=device,
            cache=cache,
            batch_size=4,
            projector_dim=8,  # Must be < n_samples (10)
            projector_type="linear",
            n_epochs=1,
            skip_whitening=False,
        )

        # Second run - should load from cache
        (
            image_embs_2,
            text_embs_2,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = extract_combined_embeddings_with_projector(
            models=models,
            df=df,
            device=device,
            cache=cache,
            batch_size=4,
            projector_dim=8,  # Must be < n_samples (10)
            projector_type="linear",
            n_epochs=1,
            skip_whitening=False,
        )

        # Results should be identical (cache was used)
        # Note: Projector training has randomness, so we just check shapes
        assert image_embs_1.shape == image_embs_2.shape
        assert text_embs_1.shape == text_embs_2.shape

    def test_trained_projector_bug_regression(
        self,
        mock_clip_model,
        mock_ssl_model,
        mock_processor,
        sample_df,
        temp_dir,
        device,
    ):
        """Regression test for the whitened_image_embeddings deletion bug.

        This test specifically catches the bug where whitened_image_embeddings
        was deleted before being used to create training samples.

        If the bug exists, this will fail with:
        NameError: cannot access free variable 'whitened_image_embeddings'
        where it is not associated with a value in enclosing scope
        """
        df = normalize_multilabel_columns(sample_df)
        cache = EmbeddingCache(temp_dir)

        # Use 7 models like in the real command that failed
        models = [
            ModelInfo(mock_clip_model, mock_processor, "clip", "model_1"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "model_2"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "model_3"),
            ModelInfo(mock_clip_model, mock_processor, "clip", "model_4"),
            ModelInfo(mock_ssl_model, None, "ssl", "model_5"),
            ModelInfo(mock_ssl_model, None, "ssl", "model_6"),
            ModelInfo(mock_ssl_model, None, "ssl", "model_7"),
        ]

        # This should NOT raise NameError about whitened_image_embeddings
        (
            image_embs,
            text_embs,
            labels,
            corrupted,
            projector_model,
            whitening_stats,
            filtered_df,
            wandb_run,
        ) = extract_combined_embeddings_with_projector(
            models=models,
            df=df,
            device=device,
            cache=cache,
            batch_size=4,
            projector_dim=8,  # Must be < n_samples (10)
            projector_type="mlp",
            n_epochs=2,
            skip_whitening=False,  # The bug occurred with whitening enabled
        )

        # If we get here, the bug is fixed!
        assert len(image_embs) == len(df)
        assert image_embs.shape[1] == 8
        assert projector_model is not None
