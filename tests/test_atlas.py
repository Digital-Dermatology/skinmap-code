"""Tests for Embedding Atlas integration."""

import os
from pathlib import Path
from types import SimpleNamespace


# Disable numba caching before importing UMAP-dependent modules to avoid filesystem issues
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
NUMBA_CACHE_DIR = Path(__file__).resolve().parent / ".numba_cache"
NUMBA_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

from numba.core import config as numba_config

numba_config.CACHING = False
numba_config.CACHE_DIR = str(NUMBA_CACHE_DIR)

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.skinmap.visualization.atlas import ATLAS_VECTOR_COLUMN, generate_atlas


def _make_args(**overrides):
    """Create a minimal args namespace for atlas generation."""
    defaults = {
        "model_name": "clip/test-model",
        "ssl_model": None,
        "svd_components": 128,
        "umap_n_neighbors": 5,
        "umap_min_dist": 0.1,
        "umap_metric": "cosine",
        "umap_fast": False,
        "seed": 123,
        "thumbnail_size": 32,
        "thumbnail_quality": 80,
        "atlas_store_vectors": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_sample_inputs(tmp_path, num_samples: int = 4):
    """Create sample dataframe + embeddings with real image paths."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    paths = []
    for idx in range(num_samples):
        img_path = img_dir / f"img_{idx}.png"
        Image.new("RGB", (8, 8), color=(idx * 20 % 255, 0, 0)).save(img_path)
        paths.append(str(img_path))

    df = pd.DataFrame(
        {
            "img_path": paths,
            "description": [f"sample {i}" for i in range(num_samples)],
            "dataset_desc": [
                "setA" if i % 2 == 0 else "setB" for i in range(num_samples)
            ],
            "condition": ["cond"] * num_samples,
        }
    )

    image_embeddings = np.random.randn(num_samples, 16).astype(np.float32)
    return df, image_embeddings


def test_generate_atlas_creates_required_artifacts(tmp_path):
    """Atlas generation should create thumbnails, parquet, UMAP model, and config metadata."""
    df, image_embeddings = _make_sample_inputs(tmp_path)
    args = _make_args()

    atlas_df = generate_atlas(
        df=df,
        image_embeddings=image_embeddings,
        output_dir=str(tmp_path),
        args=args,
        store_vectors=False,
        force_recompute=True,
    )

    atlas_path = tmp_path / "atlas_input.parquet"
    assert atlas_path.exists()
    assert len(atlas_df) == len(df)
    assert {"image", "text", "dataset"}.issubset(atlas_df.columns)
    assert {"x", "y"}.issubset(atlas_df.columns)

    # UMAP model + thumbnails exist
    umap_model = tmp_path / "analysis" / "umap_model.joblib"
    assert umap_model.exists()
    thumb_dir = tmp_path / "thumbnail"
    assert thumb_dir.is_dir()
    assert any(thumb_dir.iterdir())


@pytest.mark.parametrize("store_vectors", [True, False])
def test_generate_atlas_vector_storage(tmp_path, store_vectors):
    """Atlas should toggle between x/y projection vs embedding storage."""
    df, image_embeddings = _make_sample_inputs(tmp_path)
    args = _make_args(atlas_store_vectors=store_vectors)

    atlas_df = generate_atlas(
        df=df,
        image_embeddings=image_embeddings,
        output_dir=str(tmp_path),
        args=args,
        store_vectors=store_vectors,
        force_recompute=True,
    )

    if store_vectors:
        assert ATLAS_VECTOR_COLUMN in atlas_df.columns
        assert "x" not in atlas_df.columns and "y" not in atlas_df.columns
    else:
        assert {"x", "y"}.issubset(atlas_df.columns)
        assert ATLAS_VECTOR_COLUMN not in atlas_df.columns


def test_generate_atlas_reuses_existing_parquet(tmp_path):
    """When reuse_existing is True, atlas parquet should be reused without recompute."""
    df, image_embeddings = _make_sample_inputs(tmp_path)
    args = _make_args()

    atlas_df = generate_atlas(
        df=df,
        image_embeddings=image_embeddings,
        output_dir=str(tmp_path),
        args=args,
        store_vectors=False,
        force_recompute=True,
    )

    atlas_path = tmp_path / "atlas_input.parquet"
    mtime_before = atlas_path.stat().st_mtime

    reused_df = generate_atlas(
        df=df,
        image_embeddings=image_embeddings,
        output_dir=str(tmp_path),
        args=args,
        store_vectors=False,
        force_recompute=False,
        reuse_existing=True,
    )

    assert atlas_path.stat().st_mtime == mtime_before
    assert reused_df.equals(atlas_df)


def test_generate_atlas_force_recompute_preserves_existing_thumbnails(tmp_path):
    """Even when recomputing, existing thumbnails should be reused instead of deleted."""
    df, image_embeddings = _make_sample_inputs(tmp_path)
    args = _make_args(umap_fast=True, umap_n_jobs=1)

    generate_atlas(
        df=df,
        image_embeddings=image_embeddings,
        output_dir=str(tmp_path),
        args=args,
        store_vectors=False,
        force_recompute=True,
    )

    thumb_dir = tmp_path / "thumbnail"
    thumb_files = {p.name for p in thumb_dir.iterdir()}
    assert thumb_files, "Expected thumbnails after first run"

    generate_atlas(
        df=df,
        image_embeddings=image_embeddings,
        output_dir=str(tmp_path),
        args=args,
        store_vectors=False,
        force_recompute=True,
    )

    thumb_files_after = {p.name for p in thumb_dir.iterdir()}
    assert thumb_files.issubset(thumb_files_after)


def test_generate_atlas_handles_mixed_string_numeric_columns(tmp_path):
    """Atlas should handle object columns with string numeric values without pyarrow errors.

    This tests the fix for the bug where columns like icd_chapter containing
    string values like '2.0' caused pyarrow conversion failures when saving to parquet.
    """
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    paths = []
    num_samples = 6

    for idx in range(num_samples):
        img_path = img_dir / f"img_{idx}.png"
        Image.new("RGB", (8, 8), color=(idx * 20 % 255, 0, 0)).save(img_path)
        paths.append(str(img_path))

    # Create DataFrame with problematic string numeric values in ICD columns
    df = pd.DataFrame(
        {
            "img_path": paths,
            "description": [f"sample {i}" for i in range(num_samples)],
            "dataset_desc": ["testset"] * num_samples,
            # String numeric values that caused the original bug
            "icd_chapter": ["2.0", "13.0", "1.0", "2.0", "12.0", "13.0"],
            "icd_chapter_title": ["Neoplasms", "Diseases of the musculoskeletal system"]
            * 3,
            "icd_chapter_range": ["C00-D49", "M00-M99"] * 3,
            "icd_block": ["C43-C44", "M80-M85"] * 3,
            "icd_code": ["L30.9", "L40.0", "L30.9", "L40.0", "L30.9", "L40.0"],
            # Additional mixed type columns
            "age": ["25.0", "30", "45.5", "50", "60.0", "35"],
            "condition": ["dermatitis", "psoriasis"] * 3,
        }
    )

    image_embeddings = np.random.randn(num_samples, 16).astype(np.float32)
    args = _make_args()

    # Generate atlas - this should NOT raise pyarrow.lib.ArrowInvalid
    atlas_df = generate_atlas(
        df=df,
        image_embeddings=image_embeddings,
        output_dir=str(tmp_path),
        args=args,
        store_vectors=False,
        force_recompute=True,
    )

    # Verify parquet was created successfully
    atlas_path = tmp_path / "atlas_input.parquet"
    assert atlas_path.exists()
    assert len(atlas_df) == num_samples

    # Read back the parquet file to ensure it's valid
    loaded_df = pd.read_parquet(atlas_path)
    assert len(loaded_df) == num_samples

    # Verify ICD columns are present and all values are strings
    if "icd_chapter" in loaded_df.columns:
        assert (
            loaded_df["icd_chapter"].dtype == object
            or loaded_df["icd_chapter"].dtype.name == "string"
        )
        # Values should be preserved (converted to strings)
        assert all(isinstance(val, str) for val in loaded_df["icd_chapter"])

    # Verify required atlas columns exist
    assert "image" in loaded_df.columns
    assert "text" in loaded_df.columns
    assert "dataset" in loaded_df.columns
    assert "x" in loaded_df.columns
    assert "y" in loaded_df.columns
