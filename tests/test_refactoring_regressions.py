"""
Regression tests for refactoring bugs.

This test suite ensures that critical bugs introduced during the refactoring
from monolithic create_skinmap.py to modular architecture don't reoccur.
"""

import os
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import pytest
from PIL import Image

from src.skinmap.data.preprocessing import (
    create_thumbnail,
    create_thumbnail_column_parallel,
    get_stable_hash,
)


class TestBackwardCompatibilityExports:
    """
    Test that create_skinmap.py exports necessary symbols for backward compatibility.

    Bug: After refactoring, SSL_MODEL_NAMES and get_imagenet_transform were moved
    to modular files but not re-exported from create_skinmap.py, breaking
    combined_embedder.py which imports them.
    """

    def test_ssl_model_names_export(self):
        """Test that SSL_MODEL_NAMES can be imported from create_skinmap."""
        from src.create_skinmap import SSL_MODEL_NAMES

        assert SSL_MODEL_NAMES is not None
        assert isinstance(SSL_MODEL_NAMES, list)
        assert len(SSL_MODEL_NAMES) > 0
        # Check it contains expected SSL model names
        assert any("dino" in name or "ibot" in name for name in SSL_MODEL_NAMES)

    def test_get_imagenet_transform_export(self):
        """Test that get_imagenet_transform can be imported from create_skinmap."""
        from src.create_skinmap import get_imagenet_transform

        assert get_imagenet_transform is not None
        assert callable(get_imagenet_transform)

        # Test that it returns a valid transform
        transform = get_imagenet_transform()
        assert transform is not None

    def test_combined_embedder_can_import(self):
        """
        Test that combined_embedder.py can successfully import its dependencies.

        This is the actual use case that broke after refactoring.
        """
        try:
            from src.create_skinmap import SSL_MODEL_NAMES, get_imagenet_transform

            # If we get here, the imports worked
            assert SSL_MODEL_NAMES is not None
            assert get_imagenet_transform is not None
        except ImportError as e:
            pytest.fail(
                f"combined_embedder.py imports failed: {e}. "
                "create_skinmap.py must re-export SSL_MODEL_NAMES and get_imagenet_transform"
            )

    def test_exports_match_modular_sources(self):
        """Test that re-exported symbols match their modular sources."""
        from src.create_skinmap import SSL_MODEL_NAMES as exported_ssl_names
        from src.create_skinmap import get_imagenet_transform as exported_transform
        from src.skinmap.data.transforms import (
            get_imagenet_transform as source_transform,
        )
        from src.skinmap.models.loaders import SSL_MODEL_NAMES as source_ssl_names

        # Verify they're the same objects (not copies)
        assert exported_ssl_names is source_ssl_names
        assert exported_transform is source_transform


class TestThumbnailCachingWithMissingSource:
    """
    Test thumbnail generation prioritizes cached thumbnails over source validation.

    Bug: create_thumbnail() checked if source image exists BEFORE checking if
    thumbnail already exists, causing all thumbnails to fail when source paths
    were temporarily unavailable, resulting in empty atlas parquet files.
    """

    @pytest.fixture
    def temp_thumbnail_dir(self):
        """Create a temporary directory for thumbnails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def temp_image(self, temp_thumbnail_dir):
        """Create a temporary test image."""
        img_path = os.path.join(temp_thumbnail_dir, "test_image.jpg")
        img = Image.new("RGB", (512, 512), color="red")
        img.save(img_path, format="JPEG")
        return img_path

    def test_cached_thumbnail_used_when_source_missing(
        self, temp_thumbnail_dir, temp_image
    ):
        """
        Test that cached thumbnails are reused even when source images don't exist.

        This is the critical regression test for the bug.
        """
        # Step 1: Create a thumbnail with valid source image
        result = create_thumbnail(
            temp_image, temp_thumbnail_dir, max_size=256, quality=85
        )

        assert result is not None
        assert result.startswith("thumbnail/")
        thumbnail_filename = result.split("/")[1]
        thumbnail_path = os.path.join(temp_thumbnail_dir, thumbnail_filename)
        assert os.path.exists(thumbnail_path)

        # Step 2: Delete the source image to simulate missing/unavailable source
        os.remove(temp_image)
        assert not os.path.exists(temp_image)

        # Step 3: Try to get thumbnail again - should return cached version
        result_cached = create_thumbnail(
            temp_image,  # Source doesn't exist!
            temp_thumbnail_dir,
            max_size=256,
            quality=85,
        )

        # CRITICAL: Should return cached thumbnail, not None
        assert result_cached is not None, (
            "create_thumbnail() returned None when source image is missing "
            "but cached thumbnail exists. This causes empty atlas parquet files!"
        )
        assert result_cached == result
        assert os.path.exists(thumbnail_path)

    def test_stable_hash_doesnt_require_existing_file(self, temp_thumbnail_dir):
        """Test that get_stable_hash works even for non-existent paths."""
        fake_path = "/nonexistent/path/to/image.jpg"

        # Should not raise an exception
        hash1 = get_stable_hash(fake_path)
        hash2 = get_stable_hash(fake_path)

        assert hash1 == hash2  # Stable
        assert len(hash1) == 16  # Correct length

    def test_thumbnail_column_parallel_with_missing_sources(self, temp_thumbnail_dir):
        """
        Test parallel thumbnail generation handles missing sources gracefully.

        Simulates the exact scenario that caused 0 rows in atlas parquet.
        """
        # Create some test images and their thumbnails
        test_paths = []
        for i in range(5):
            img_path = os.path.join(temp_thumbnail_dir, f"test_{i}.jpg")
            img = Image.new("RGB", (256, 256), color=(i * 50, i * 50, i * 50))
            img.save(img_path)
            test_paths.append(img_path)

        # Generate thumbnails first time
        series = pd.Series(test_paths)
        result1 = create_thumbnail_column_parallel(
            series,
            thumbnail_dir=temp_thumbnail_dir,
            max_size=128,
            quality=85,
            max_workers=2,
        )

        # All should succeed
        assert result1.notna().all()
        assert len(result1) == 5

        # Delete all source images (simulating Docker mount issue, path changes, etc.)
        for path in test_paths:
            os.remove(path)

        # Try to regenerate thumbnails with missing sources
        result2 = create_thumbnail_column_parallel(
            series,
            thumbnail_dir=temp_thumbnail_dir,
            max_size=128,
            quality=85,
            max_workers=2,
        )

        # CRITICAL: Should still return all thumbnails from cache
        assert result2.notna().all(), (
            "Thumbnail generation failed when source images missing but cached "
            "thumbnails exist. This causes atlas parquet to have 0 rows!"
        )
        assert len(result2) == 5
        # Should be same paths as before
        assert (result1 == result2).all()

    def test_thumbnail_creation_fails_gracefully_without_cache(
        self, temp_thumbnail_dir
    ):
        """Test that missing source without cache returns None (expected behavior)."""
        nonexistent_path = "/tmp/nonexistent_image_12345.jpg"

        result = create_thumbnail(
            nonexistent_path, temp_thumbnail_dir, max_size=256, quality=85
        )

        # Should return None when no cache and no source
        assert result is None

    def test_thumbnail_hash_consistency_across_runs(
        self, temp_thumbnail_dir, temp_image
    ):
        """Test that the same image path always generates the same thumbnail hash."""
        # Generate thumbnail twice
        result1 = create_thumbnail(temp_image, temp_thumbnail_dir)

        # Create another identical image at same path (after deleting first)
        os.remove(temp_image)
        img = Image.new("RGB", (512, 512), color="blue")  # Different content
        img.save(temp_image)

        result2 = create_thumbnail(temp_image, temp_thumbnail_dir)

        # Should generate same hash (path-based, not content-based)
        assert result1 == result2


class TestAtlasParquetGeneration:
    """
    Integration tests for atlas parquet generation with thumbnail caching.
    """

    @pytest.fixture
    def temp_thumbnail_dir(self):
        """Create a temporary directory for thumbnails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for atlas generation."""
        args = MagicMock()
        args.thumbnail_size = 256
        args.thumbnail_quality = 85
        args.umap_n_neighbors = 15
        args.umap_min_dist = 0.1
        args.umap_metric = "cosine"
        args.umap_fast = False
        args.seed = 42
        return args

    def test_atlas_generation_doesnt_filter_all_rows_on_cache_reuse(
        self, temp_thumbnail_dir, mock_args
    ):
        """
        Test that atlas generation doesn't create empty parquet when reusing thumbnails.

        This is an integration test for the complete bug scenario.
        """
        from src.skinmap.data.preprocessing import create_thumbnail_column_parallel

        # Create test data
        n_samples = 10
        img_paths = [f"/fake/data/image_{i}.jpg" for i in range(n_samples)]
        df = pd.DataFrame(
            {
                "img_path": img_paths,
                "description": [f"Test image {i}" for i in range(n_samples)],
                "dataset_desc": ["test_dataset"] * n_samples,
            }
        )

        # Pre-create thumbnails with stable hashes (simulate previous run)
        for img_path in img_paths:
            img_hash = get_stable_hash(img_path)
            thumbnail_path = os.path.join(temp_thumbnail_dir, f"{img_hash}.jpg")

            # Create a dummy thumbnail
            img = Image.new("RGB", (128, 128), color="red")
            img.save(thumbnail_path, format="JPEG")

        # Now try to generate thumbnails (source images don't exist!)
        df_copy = df.copy()
        df_copy["image"] = create_thumbnail_column_parallel(
            df_copy["img_path"],
            thumbnail_dir=temp_thumbnail_dir,
            max_size=256,
            quality=85,
        )

        # Filter out failed thumbnails (this is what atlas.py does)
        filtered_df = df_copy[df_copy["image"].notnull()].reset_index(drop=True)

        # CRITICAL: Should NOT result in empty dataframe
        assert len(filtered_df) == n_samples, (
            f"Atlas filtering removed all rows! Expected {n_samples}, got {len(filtered_df)}. "
            "This is the bug that caused 'Loaded dataframe with 0 rows' error."
        )
        assert filtered_df["image"].notna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
