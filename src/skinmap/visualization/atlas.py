"""Atlas (embedding parquet) generation for the SkinMap analysis pipeline."""

import os

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import umap
from loguru import logger

from ..data.preprocessing import create_thumbnail_column_parallel

ATLAS_VECTOR_COLUMN = "embedding"


def generate_atlas(
    df: pd.DataFrame,
    image_embeddings: np.ndarray,
    output_dir: str,
    args,
    store_vectors: bool = False,
    force_recompute: bool = False,
    reuse_existing: bool = True,
) -> pd.DataFrame:
    """Generate atlas parquet file and config for Embedding Atlas.

    Args:
        df: DataFrame with metadata
        image_embeddings: Image embedding vectors
        output_dir: Output directory for atlas files
        args: Argument namespace with UMAP and thumbnail settings
        store_vectors: If True, store full embedding vectors in atlas
        force_recompute: Force atlas regeneration even if cached file exists
        reuse_existing: Reuse cached atlas when available and not forcing recompute

    Returns:
        DataFrame with atlas data
    """
    atlas_path = os.path.join(output_dir, "atlas_input.parquet")
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    if reuse_existing and not force_recompute and os.path.exists(atlas_path):
        logger.info(f"Reusing existing atlas from {atlas_path}")
        atlas_df = pd.read_parquet(atlas_path)
        return atlas_df

    # Remove existing atlas artifacts when recomputing
    if os.path.exists(atlas_path):
        logger.info(f"Removing existing atlas file {atlas_path}")
        os.remove(atlas_path)

    thumbnail_dir = os.path.join(output_dir, "thumbnail")
    if os.path.exists(thumbnail_dir):
        logger.info(
            "Thumbnail directory already exists; existing files will be reused where possible"
        )

    # Create UMAP model
    logger.info("Creating UMAP projection for atlas...")

    umap_kwargs = {
        "n_neighbors": args.umap_n_neighbors,
        "min_dist": args.umap_min_dist,
        "metric": args.umap_metric,
    }

    if args.umap_fast:
        n_jobs = args.umap_n_jobs if args.umap_n_jobs else -1
        logger.info(f"Using fast UMAP mode (multi-threaded with {n_jobs} jobs)")
        umap_kwargs["n_jobs"] = n_jobs
    else:
        logger.info("Using reproducible UMAP mode")
        umap_kwargs["random_state"] = args.seed

    reducer = umap.UMAP(**umap_kwargs)

    # Fit UMAP and optionally transform
    umap_model_path = os.path.join(analysis_dir, "umap_model.joblib")

    if store_vectors:
        logger.info("Atlas will store full embedding vectors; fitting UMAP model")
        reducer.fit(image_embeddings)
    else:
        logger.info("Atlas will store 2D projections; fitting and transforming")
        projection = reducer.fit_transform(image_embeddings)
        df["x"] = projection[:, 0]
        df["y"] = projection[:, 1]

    # Save UMAP model
    try:
        joblib.dump(reducer, umap_model_path)
        logger.info(f"Saved UMAP model to {umap_model_path}")
    except Exception as exc:
        logger.warning(f"Failed to save UMAP model: {exc}")

    # Prepare DataFrame for Embedding Atlas
    required_cols = ["img_path", "description", "dataset_desc"]
    if not store_vectors:
        required_cols.extend(["x", "y"])

    optional_cols = [
        "description_short",
        "modality",
        "release_year",
        "condition",
        "body_location",
        "laterality",
        "body_region",
        "gender",
        "age",
        "fitzpatrick",
        "origin",
        # ICD hierarchy and labels
        "icd_code",
        "icd_description",
        "icd_chapter",
        "icd_chapter_title",
        "icd_chapter_range",
        "icd_chapter_label",
        "icd_block",
        "icd_block_desc",
        "icd_block_label",
        "icd_category",
        "icd_category_desc",
        "icd_category_label",
    ]

    # Select columns that exist
    atlas_cols = [col for col in required_cols if col in df.columns]
    atlas_cols.extend([col for col in optional_cols if col in df.columns])

    # Add prediction columns
    pred_cols = [col for col in df.columns if col.endswith("_pred")]
    atlas_cols.extend(pred_cols)

    atlas_df = df[atlas_cols].copy()

    # Rename columns for atlas
    atlas_df = atlas_df.rename(
        columns={
            "description": "text",
            "description_short": "text_short",
            "dataset_desc": "dataset",
        }
    )

    # Add embedding vectors if storing
    if store_vectors:
        vectors = image_embeddings.astype(np.float32)
        # Store as list of arrays (not .tolist() which creates strings!)
        atlas_df[ATLAS_VECTOR_COLUMN] = list(vectors)
        logger.info(f"Added {len(vectors)} embedding vectors to atlas")

    # Generate thumbnails
    logger.info("Generating thumbnails for atlas...")
    thumbnail_dir = os.path.join(output_dir, "thumbnail")
    atlas_df["image"] = create_thumbnail_column_parallel(
        atlas_df["img_path"],
        thumbnail_dir=thumbnail_dir,
        max_size=getattr(
            args,
            "thumbnail_size",
            getattr(args, "thumbnail_max_size", 256),
        ),
        quality=getattr(args, "thumbnail_quality", 85),
    )

    # Filter out failed thumbnails and reorder columns
    atlas_df = atlas_df[atlas_df["image"].notnull()].reset_index(drop=True)
    cols = list(atlas_df.columns)
    cols.insert(0, cols.pop(cols.index("image")))
    atlas_df = atlas_df.loc[:, cols]

    # Save to parquet with explicit schema to preserve types correctly
    # This ensures embeddings are stored as lists (not strings) and ints stay as ints

    # Columns with object dtype that should remain as strings (not auto-inferred as numeric)
    # These can contain numeric-looking strings like "2.0" that PyArrow might misinterpret
    string_columns = {
        "icd_chapter",
        "icd_chapter_title",
        "icd_chapter_range",
        "icd_chapter_label",
        "icd_block",
        "icd_block_desc",
        "icd_block_label",
        "icd_category",
        "icd_category_desc",
        "icd_category_label",
        "icd_code",
        "icd_description",
    }

    # Force ICD/text columns to real string dtype so pyarrow doesn't coerce them to numbers
    for col in string_columns:
        if col in atlas_df.columns:
            atlas_df[col] = atlas_df[col].astype("string")

    # Build schema up front so PyArrow does not try to coerce numeric-looking strings
    base_schema = pa.Schema.from_pandas(atlas_df, preserve_index=False)
    schema_fields = []
    for field in base_schema:
        if field.name == ATLAS_VECTOR_COLUMN and store_vectors:
            # Define embedding column as list of float32
            schema_fields.append(pa.field(ATLAS_VECTOR_COLUMN, pa.list_(pa.float32())))
        elif field.name in string_columns:
            # Force these columns to be strings to avoid type inference issues
            schema_fields.append(pa.field(field.name, pa.string()))
        elif field.type == pa.int64():
            # Keep integer columns as int64
            schema_fields.append(pa.field(field.name, pa.int64()))
        elif field.type == pa.float64():
            # Keep float columns as float64
            schema_fields.append(pa.field(field.name, pa.float64()))
        else:
            # Keep other types as-is
            schema_fields.append(field)

    new_schema = pa.schema(schema_fields)
    table = pa.Table.from_pandas(
        atlas_df,
        schema=new_schema,
        safe=False,
        preserve_index=False,
    )
    pq.write_table(table, atlas_path)
    logger.info(f"Saved Atlas input to {atlas_path} ({len(atlas_df)} samples)")

    return atlas_df
