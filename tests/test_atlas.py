"""Tests for Embedding Atlas integration."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import joblib

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
from sklearn.decomposition import TruncatedSVD

from skinmap.pipeline.create_skinmap import save_outputs
from skinmap.visualization.atlas import ATLAS_VECTOR_COLUMN, generate_atlas


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
        launch_cli=False,
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

    # Config contains artifact metadata compatible with Embedding Atlas CLI
    config_path = tmp_path / "embedding_pipeline_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config["artifacts"]["atlas_parquet"] == "atlas_input.parquet"
    assert config["vector"]["dimension"] == image_embeddings.shape[1]
    assert config["svd"]["components"] == args.svd_components
    assert "umap" in config
    umap_rel = config["umap"]["model"]
    assert (tmp_path / umap_rel).exists()


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
        launch_cli=False,
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
        launch_cli=False,
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
        launch_cli=False,
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
        launch_cli=False,
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
        launch_cli=False,
        force_recompute=True,
    )

    thumb_files_after = {p.name for p in thumb_dir.iterdir()}
    assert thumb_files.issubset(thumb_files_after)


def test_embedding_pipeline_config_points_to_loadable_artifacts(tmp_path):
    """Pipeline config should reference artifacts that can be loaded after saves."""
    df, image_embeddings = _make_sample_inputs(tmp_path)
    text_embeddings = np.random.randn(len(df), 10).astype(np.float32)
    dataset_labels = np.zeros(len(df), dtype=np.int32)

    # Fit tiny SVDs to act as saved artifacts
    svd_image = TruncatedSVD(n_components=4, random_state=0).fit(
        np.random.randn(32, image_embeddings.shape[1])
    )
    svd_text = TruncatedSVD(n_components=4, random_state=0).fit(
        np.random.randn(32, text_embeddings.shape[1])
    )

    args = _make_args(
        svd_components=4,
        atlas_store_vectors=True,
    )
    # Extra fields used by save_outputs/persist_artifacts
    args.model_path = None
    args.use_trained_projector = False
    args.projector_type = "linear"
    args.projector_dim = 4
    args.max_whitening_dim = 768
    args.skip_whitening = False

    artifacts = {
        "svd_image_model": svd_image,
        "svd_text_model": svd_text,
    }

    save_outputs(
        df=df,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        dataset_labels=dataset_labels,
        corrupted_indices=[],
        fusion_method="svd",
        args=args,
        output_dir=str(tmp_path),
        run_name="test_run",
        artifacts=artifacts,
        model_names=[args.model_name],
    )

    # Atlas generation runs after artifacts are written so config references them
    generate_atlas(
        df=df,
        image_embeddings=image_embeddings,
        output_dir=str(tmp_path),
        args=args,
        store_vectors=True,
        launch_cli=False,
        force_recompute=True,
    )

    config_path = tmp_path / "embedding_pipeline_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    image_svd_rel = config["svd"]["image"]
    text_svd_rel = config["svd"]["text"]
    assert image_svd_rel and text_svd_rel

    # Stored artifacts must load and be usable
    image_svd = joblib.load(tmp_path / image_svd_rel)
    text_svd = joblib.load(tmp_path / text_svd_rel)
    image_svd.transform(np.random.randn(2, image_embeddings.shape[1]))
    text_svd.transform(np.random.randn(2, text_embeddings.shape[1]))

    # UMAP model should also be loadable from config
    umap_rel = config["umap"]["model"]
    reducer = joblib.load(tmp_path / umap_rel)
    reducer.transform(np.random.randn(2, image_embeddings.shape[1]))


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
        launch_cli=False,
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
