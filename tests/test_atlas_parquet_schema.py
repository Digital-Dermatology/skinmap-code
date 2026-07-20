"""Test to verify parquet schema and data types after atlas generation.

This test ensures that the fix for embedding storage is working correctly:
- Embeddings should be stored as List[float], not String
- List columns (origin, origin_pred) should be stored as List, not String
- Regular string columns should remain as String
- File size should be efficient
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# Disable numba caching before importing UMAP
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
NUMBA_CACHE_DIR = Path(__file__).resolve().parent / ".numba_cache"
NUMBA_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

from numba.core import config as numba_config

numba_config.CACHING = False
numba_config.CACHE_DIR = str(NUMBA_CACHE_DIR)

from src.skinmap.visualization.atlas import ATLAS_VECTOR_COLUMN, generate_atlas


def test_atlas_parquet_schema_efficiency(tmp_path):
    """Verify embeddings and lists are stored efficiently, not as strings."""
    # Create sample data with lists and arrays
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    num_samples = 10
    paths = []

    for idx in range(num_samples):
        img_path = img_dir / f"img_{idx}.png"
        Image.new("RGB", (8, 8), color=(idx * 20 % 255, 0, 0)).save(img_path)
        paths.append(str(img_path))

    df = pd.DataFrame(
        {
            "img_path": paths,
            "description": [f"sample {i}" for i in range(num_samples)],
            "dataset_desc": ["testset"] * num_samples,
            "origin": [["Spain"], ["USA", "Canada"], ["Germany"], ["France"]] * 2
            + [["Italy"], ["UK"]],
            "origin_pred": [["Spain"], ["USA"], ["Germany"], ["France"]] * 2
            + [["Italy"], ["UK"]],
            "condition": ["dermatitis", "psoriasis"] * 5,
            "icd_chapter": [
                "2.0",
                "13.0",
                "1.0",
                "2.0",
                "12.0",
                "13.0",
                "1.0",
                "2.0",
                "12.0",
                "13.0",
            ],
        }
    )

    image_embeddings = np.random.randn(num_samples, 128).astype(np.float32)

    args = SimpleNamespace(
        model_name="test/model",
        ssl_model=None,
        svd_components=None,
        umap_n_neighbors=5,
        umap_min_dist=0.1,
        umap_metric="cosine",
        umap_fast=False,
        seed=123,
        thumbnail_size=32,
        thumbnail_quality=80,
        atlas_store_vectors=True,
    )

    # Generate atlas with vector storage
    generate_atlas(
        df=df,
        image_embeddings=image_embeddings,
        output_dir=str(tmp_path),
        args=args,
        store_vectors=True,
        force_recompute=True,
    )

    atlas_path = tmp_path / "atlas_input.parquet"
    assert atlas_path.exists(), "Atlas parquet file should exist"

    # Load and verify
    loaded_df = pd.read_parquet(atlas_path)
    pq_file = pq.ParquetFile(atlas_path)
    schema_str = str(pq_file.schema)
    arrow_schema = pq_file.schema_arrow

    # Test 1: Embedding should be stored as List, not String
    assert ATLAS_VECTOR_COLUMN in loaded_df.columns, "Embedding column should exist"
    embedding_value = loaded_df[ATLAS_VECTOR_COLUMN].iloc[0]

    # Check schema - embedding should be List, not String
    embedding_schema_line = None
    for line in schema_str.split("\n"):
        if "embedding" in line.lower():
            embedding_schema_line = line
            break

    assert embedding_schema_line is not None, "Embedding column not found in schema"
    assert (
        "List" in embedding_schema_line or "list" in embedding_schema_line
    ), f"Embedding should be stored as List, not String! Got: {embedding_schema_line}"
    assert (
        "binary" not in embedding_schema_line or "String)" in embedding_schema_line
    ), f"Embedding should NOT be binary String! Got: {embedding_schema_line}"

    # Value should be numpy array
    assert isinstance(
        embedding_value, np.ndarray
    ), f"Embedding should be numpy array, got {type(embedding_value)}"
    assert (
        embedding_value.dtype == np.float32
    ), f"Embedding should be float32, got {embedding_value.dtype}"
    assert (
        len(embedding_value) == 128
    ), f"Embedding should have 128 dimensions, got {len(embedding_value)}"

    # Test 2: origin and origin_pred should be stored as List, not String
    for col in ["origin", "origin_pred"]:
        if col in loaded_df.columns:
            value = loaded_df[col].iloc[1]  # Use index 1 which has multiple values

            # Check schema
            col_schema_line = None
            for line in schema_str.split("\n"):
                if col in line and "optional" in line:
                    col_schema_line = line
                    break

            if col_schema_line:
                # Should be List, not binary String
                assert (
                    "List" in col_schema_line or "list" in col_schema_line
                ), f"{col} should be stored as List, not String! Got: {col_schema_line}"
                # Allow "String)" which is the element type inside the list
                if "binary" in col_schema_line:
                    assert (
                        "String)" in col_schema_line
                    ), f"{col} should NOT be binary String! Got: {col_schema_line}"

            # Should be array/list when loaded
            assert isinstance(
                value, (list, np.ndarray)
            ), f"{col} should be list/array, got {type(value)}"

    # Test 3: Regular string columns should be String (and stored as string in parquet)
    string_cols = ["condition", "description", "dataset_desc", "icd_chapter"]
    for col in string_cols:
        if col in loaded_df.columns:
            value = loaded_df[col].iloc[0]
            assert isinstance(value, str), f"{col} should be string, got {type(value)}"
            if col in arrow_schema.names:
                assert pa.types.is_string(
                    arrow_schema.field(col).type
                ), f"{col} should be stored as string in parquet schema"

    # Test 4: File size efficiency check
    file_size_mb = atlas_path.stat().st_size / 1024 / 1024
    # With 128-dim float32 embeddings, each embedding should be ~512 bytes
    # Plus metadata, thumbnails, etc. Reasonable upper bound is ~200KB per sample
    max_size_per_sample_kb = 200  # Conservative estimate
    assert file_size_mb / num_samples * 1024 < max_size_per_sample_kb, (
        f"File size seems too large! Got {file_size_mb / num_samples * 1024:.2f} KB/sample, "
        f"expected < {max_size_per_sample_kb} KB/sample"
    )
