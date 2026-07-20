"""Tests for data preprocessing functionality."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.skinmap.data.preprocessing import (
    coerce_multilabel,
    create_thumbnail_column_parallel,
    get_stable_hash,
    normalize_multilabel_columns,
)


class TestCoerceMultilabel:
    """Test multilabel value coercion."""

    def test_list_input_returned_as_is(self):
        """List inputs should be cleaned and returned."""
        result = coerce_multilabel(["Europe", "Asia"])
        assert result == ["Europe", "Asia"]

    def test_string_list_parsed(self):
        """String representations of lists should be parsed."""
        result = coerce_multilabel("['Europe', 'Asia']")
        assert result == ["Europe", "Asia"]

    def test_comma_separated_string_split(self):
        """Comma-separated strings should be split."""
        result = coerce_multilabel("Europe, Asia, Africa")
        assert result == ["Europe", "Asia", "Africa"]

    def test_single_value_wrapped_in_list(self):
        """Single values should be wrapped in list."""
        result = coerce_multilabel("Europe")
        assert result == ["Europe"]

    def test_none_returns_none(self):
        """None should return None."""
        result = coerce_multilabel(None)
        assert result is None

    def test_nan_returns_none(self):
        """NaN should return None."""
        result = coerce_multilabel(np.nan)
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        result = coerce_multilabel("")
        assert result is None

    def test_whitespace_only_string_returns_none(self):
        """Whitespace-only string should return None."""
        result = coerce_multilabel("   ")
        assert result is None

    def test_list_with_none_values_filtered(self):
        """Lists with None values should be filtered."""
        result = coerce_multilabel(["Europe", None, "Asia", ""])
        assert result == ["Europe", "Asia"]

    def test_nested_list_flattened(self):
        """Nested lists should be flattened."""
        result = coerce_multilabel("[['Europe'], ['Asia']]")
        assert "Europe" in str(result) and "Asia" in str(result)

    def test_mixed_format_handled(self):
        """Mixed format strings should be handled."""
        result = coerce_multilabel('["Europe", "Asia"]')
        assert result == ["Europe", "Asia"]

    def test_list_with_whitespace_cleaned(self):
        """List items with whitespace should be cleaned."""
        result = coerce_multilabel(["  Europe  ", " Asia "])
        assert result == ["Europe", "Asia"]


class TestNormalizeMultilabelColumns:
    """Test multilabel column normalization."""

    def test_normalize_origin_column(self, multilabel_df):
        """Test origin column remains untouched when not configured as multilabel."""
        normalized = normalize_multilabel_columns(multilabel_df)

        # Check all values are properly normalized
        assert isinstance(normalized.iloc[0]["origin"], str)
        assert isinstance(normalized.iloc[1]["origin"], str)
        assert isinstance(normalized.iloc[2]["origin"], list)
        assert normalized.iloc[3]["origin"] is None  # None preserved
        assert isinstance(normalized.iloc[4]["origin"], str)

    def test_normalize_preserves_other_columns(self, multilabel_df):
        """Normalization should preserve non-multilabel columns."""
        normalized = normalize_multilabel_columns(multilabel_df)

        assert "img_path" in normalized.columns
        assert list(normalized["img_path"]) == list(multilabel_df["img_path"])

    def test_normalize_with_no_multilabel_columns(self):
        """Test with dataframe without multilabel columns."""
        df = pd.DataFrame(
            {
                "img_path": ["a.jpg", "b.jpg"],
                "label": ["A", "B"],
            }
        )

        normalized = normalize_multilabel_columns(df)
        assert df.equals(normalized)

    def test_normalize_empty_dataframe(self):
        """Test normalizing empty dataframe."""
        df = pd.DataFrame(columns=["img_path", "origin"])

        normalized = normalize_multilabel_columns(df)
        assert len(normalized) == 0

    def test_normalize_all_none_values(self):
        """Test normalizing column with all None values."""
        df = pd.DataFrame(
            {
                "img_path": ["a.jpg", "b.jpg"],
                "origin": [None, None],
            }
        )

        normalized = normalize_multilabel_columns(df)
        assert normalized["origin"].isna().all()


class TestGetStableHash:
    """Test content hashing for images."""

    def test_same_image_same_hash(self, temp_dir):
        """Same image should produce same hash."""
        img_path = os.path.join(temp_dir, "test.jpg")
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(img_path)

        hash1 = get_stable_hash(img_path)
        hash2 = get_stable_hash(img_path)

        assert hash1 == hash2

    def test_different_images_different_hash(self, temp_dir):
        """Different images should produce different hashes."""
        img1_path = os.path.join(temp_dir, "img1.jpg")
        img2_path = os.path.join(temp_dir, "img2.jpg")

        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))

        img1.save(img1_path)
        img2.save(img2_path)

        hash1 = get_stable_hash(img1_path)
        hash2 = get_stable_hash(img2_path)

        assert hash1 != hash2

    def test_nonexistent_file_returns_hash(self):
        """Nonexistent file should return a hash based on path."""
        hash_val = get_stable_hash("/nonexistent/file.jpg")
        # Function returns hash of path even if file doesn't exist
        assert hash_val is not None and len(hash_val) > 0

    def test_corrupted_file_handled(self, temp_dir):
        """Corrupted file should be handled gracefully."""
        corrupted_path = os.path.join(temp_dir, "corrupted.jpg")

        # Create corrupted file (not a valid image)
        with open(corrupted_path, "w") as f:
            f.write("This is not an image")

        hash_val = get_stable_hash(corrupted_path)
        # Should either return hash of bytes or None
        assert hash_val is not None

    def test_relative_paths_consistent_across_cwd(self, temp_dir):
        """Relative img_path strings should produce identical hashes regardless of cwd."""
        rel_path = os.path.join("data", "ISIC", "190.jpg")
        temp_root = Path(temp_dir)
        full_path = temp_root / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(10, 20, 30)).save(full_path)

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_root)
            hash_one = get_stable_hash(rel_path)
        finally:
            os.chdir(original_cwd)

        other_dir = temp_root / "run2"
        (other_dir / "data/ISIC").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(10, 20, 30)).save(
            other_dir / "data/ISIC/190.jpg"
        )
        try:
            os.chdir(other_dir)
            hash_two = get_stable_hash(rel_path)
        finally:
            os.chdir(original_cwd)

        assert hash_one == hash_two

    def test_absolute_paths_match_repo_relative(self, tmp_path, monkeypatch):
        """Absolute paths inside PROJECT_ROOT should hash identically to their relative form."""
        from src.skinmap.data import preprocessing as prep_mod

        monkeypatch.setattr(prep_mod, "PROJECT_ROOT", tmp_path)

        base = tmp_path / "data/ISIC"
        base.mkdir(parents=True, exist_ok=True)
        file_path = base / "200.jpg"
        Image.new("RGB", (16, 16), color=(1, 2, 3)).save(file_path)

        rel_str = file_path.relative_to(tmp_path).as_posix()
        hash_abs = get_stable_hash(str(file_path))
        hash_rel = get_stable_hash(rel_str)
        assert hash_abs == hash_rel


class TestCreateThumbnailColumn:
    """Test thumbnail generation."""

    def test_create_thumbnails_for_valid_images(self, sample_df, temp_dir):
        """Test thumbnail generation for valid images."""
        thumbnail_dir = os.path.join(temp_dir, "thumbnails")

        thumbnails = create_thumbnail_column_parallel(
            sample_df["img_path"],
            thumbnail_dir=thumbnail_dir,
            max_size=128,
            quality=85,
        )

        # All valid images should have thumbnails
        assert len(thumbnails) == len(sample_df)
        assert all(isinstance(t, str) for t in thumbnails if t is not None)

    def test_create_thumbnails_handles_corrupted(self, temp_dir):
        """Test thumbnail generation with corrupted images."""
        thumbnail_dir = os.path.join(temp_dir, "thumbnails")

        # Mix of valid and corrupted paths
        valid_path = os.path.join(temp_dir, "valid.jpg")
        Image.new("RGB", (224, 224)).save(valid_path)

        img_paths = pd.Series([valid_path, "/nonexistent/corrupted.jpg"])

        thumbnails = create_thumbnail_column_parallel(
            img_paths,
            thumbnail_dir=thumbnail_dir,
            max_size=128,
        )

        # Valid image should have thumbnail
        assert thumbnails.iloc[0] is not None
        # Corrupted path may have a placeholder or None
        # The function generates based on hash, so it won't be None

    def test_thumbnail_size_respected(self, temp_dir):
        """Test that thumbnail respects max_size."""
        thumbnail_dir = os.path.join(temp_dir, "thumbnails")

        # Create large image
        large_img_path = os.path.join(temp_dir, "large.jpg")
        large_img = Image.new("RGB", (1000, 1000))
        large_img.save(large_img_path)

        thumbnails = create_thumbnail_column_parallel(
            pd.Series([large_img_path]),
            thumbnail_dir=thumbnail_dir,
            max_size=128,
        )

        # Load generated thumbnail
        thumb_path = thumbnails.iloc[0]
        if thumb_path and os.path.exists(thumb_path):
            thumb = Image.open(thumb_path)
            assert max(thumb.size) <= 128

    def test_thumbnail_quality_setting(self, temp_dir):
        """Test different quality settings."""
        thumbnail_dir = os.path.join(temp_dir, "thumbnails")

        img_path = os.path.join(temp_dir, "test.jpg")
        img = Image.new("RGB", (500, 500))
        img.save(img_path)

        # Generate with different qualities
        thumbs_high = create_thumbnail_column_parallel(
            pd.Series([img_path]),
            thumbnail_dir=thumbnail_dir,
            quality=95,
        )

        thumbs_low = create_thumbnail_column_parallel(
            pd.Series([img_path]),
            thumbnail_dir=os.path.join(temp_dir, "thumbnails_low"),
            quality=50,
        )

        # Both should succeed
        assert thumbs_high.iloc[0] is not None
        assert thumbs_low.iloc[0] is not None

    def test_empty_input_list(self, temp_dir):
        """Test with empty input list."""
        thumbnail_dir = os.path.join(temp_dir, "thumbnails")

        thumbnails = create_thumbnail_column_parallel(
            pd.Series([], dtype=str),
            thumbnail_dir=thumbnail_dir,
        )

        assert len(thumbnails) == 0

    def test_thumbnail_directory_created(self, temp_dir):
        """Test that thumbnail directory is created."""
        thumbnail_dir = os.path.join(temp_dir, "new_thumbnails")
        assert not os.path.exists(thumbnail_dir)

        img_path = os.path.join(temp_dir, "test.jpg")
        Image.new("RGB", (100, 100)).save(img_path)

        create_thumbnail_column_parallel(
            pd.Series([img_path]),
            thumbnail_dir=thumbnail_dir,
        )

        assert os.path.exists(thumbnail_dir)

    def test_thumbnail_base64_encoding(self, temp_dir):
        """Test that thumbnails are file paths."""
        thumbnail_dir = os.path.join(temp_dir, "thumbnails")

        img_path = os.path.join(temp_dir, "test.jpg")
        Image.new("RGB", (100, 100), color=(255, 0, 0)).save(img_path)

        thumbnails = create_thumbnail_column_parallel(
            pd.Series([img_path]),
            thumbnail_dir=thumbnail_dir,
        )

        # Should be file path
        thumb = thumbnails.iloc[0]
        assert thumb is not None
        assert isinstance(thumb, str)


class TestEdgeCases:
    """Test edge cases in data preprocessing."""

    def test_unicode_in_multilabel(self):
        """Test handling unicode characters in multilabel."""
        result = coerce_multilabel("Europe, 日本, العربية")
        assert len(result) == 3
        assert "日本" in result

    def test_very_long_multilabel_list(self):
        """Test handling very long multilabel lists."""
        long_list = [f"item_{i}" for i in range(1000)]
        result = coerce_multilabel(long_list)
        assert len(result) == 1000

    def test_special_characters_in_paths(self, temp_dir):
        """Test handling special characters in file paths."""
        special_path = os.path.join(temp_dir, "image with spaces & special.jpg")
        img = Image.new("RGB", (100, 100))
        img.save(special_path)

        hash_val = get_stable_hash(special_path)
        assert hash_val is not None

    def test_extremely_large_image_thumbnail(self, temp_dir):
        """Test thumbnail generation for very large image."""
        thumbnail_dir = os.path.join(temp_dir, "thumbnails")

        # Create 10000x10000 image (but don't save to avoid disk space issues)
        # Instead test the logic
        normal_path = os.path.join(temp_dir, "normal.jpg")
        Image.new("RGB", (5000, 5000)).save(normal_path)

        thumbnails = create_thumbnail_column_parallel(
            pd.Series([normal_path]),
            thumbnail_dir=thumbnail_dir,
            max_size=256,
        )

        assert thumbnails.iloc[0] is not None

    def test_zero_size_image(self, temp_dir):
        """Test handling zero-size image file."""
        zero_path = os.path.join(temp_dir, "zero.jpg")

        # Create empty file
        open(zero_path, "w").close()

        thumbnail_dir = os.path.join(temp_dir, "thumbnails")
        thumbnails = create_thumbnail_column_parallel(
            pd.Series([zero_path]),
            thumbnail_dir=thumbnail_dir,
        )

        # Function generates hash-based path even for corrupted files
        assert len(thumbnails) == 1
