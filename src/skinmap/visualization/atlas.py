"""Atlas generation for interactive visualization with Embedding Atlas."""

import json
import os
import subprocess
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import umap
from loguru import logger

from ..data.preprocessing import PROJECT_ROOT, create_thumbnail_column_parallel
from ..models.loaders import SSL_MODEL_NAMES

ATLAS_VECTOR_COLUMN = "embedding"


def generate_atlas(
    df: pd.DataFrame,
    image_embeddings: np.ndarray,
    output_dir: str,
    args,
    store_vectors: bool = False,
    launch_cli: bool = False,
    force_recompute: bool = False,
    reuse_existing: bool = True,
    label_column: Optional[str] = None,
    filtering_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Generate atlas parquet file and config for Embedding Atlas.

    Args:
        df: DataFrame with metadata
        image_embeddings: Image embedding vectors
        output_dir: Output directory for atlas files
        args: Argument namespace with UMAP and thumbnail settings
        store_vectors: If True, store full embedding vectors in atlas
        launch_cli: If True, launch Embedding Atlas CLI after generation
        force_recompute: Force atlas regeneration even if cached file exists
        reuse_existing: Reuse cached atlas when available and not forcing recompute
        label_column: Column to use for automatic cluster labels (falls back to args.label_column)
        filtering_threshold: If set, remove samples with score_composite below this
            threshold using filtering_scores.csv from the embeddings directory.

    Returns:
        DataFrame with atlas data
    """
    if label_column is None:
        label_column = getattr(args, "label_column", None)
    atlas_path = os.path.join(output_dir, "atlas_input.parquet")
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    if reuse_existing and not force_recompute and os.path.exists(atlas_path):
        logger.info(f"Reusing existing atlas from {atlas_path}")
        atlas_df = pd.read_parquet(atlas_path)
        if launch_cli:
            _launch_atlas_cli(atlas_path, store_vectors, label_column=label_column)
        return atlas_df

    # Remove existing atlas artifacts when recomputing
    if os.path.exists(atlas_path):
        logger.info(f"Removing existing atlas file {atlas_path}")
        os.remove(atlas_path)

    # Apply filtering if threshold is set and filtering_scores.csv exists
    embeddings_dir = os.path.join(output_dir, "embeddings")
    scores_path = os.path.join(embeddings_dir, "filtering_scores.csv")
    if filtering_threshold is not None and os.path.exists(scores_path):
        scores_df = pd.read_csv(scores_path, usecols=["score_composite"])
        assert len(scores_df) == len(df), (
            f"filtering_scores.csv row count ({len(scores_df)}) != "
            f"dataframe row count ({len(df)})"
        )
        keep_mask = scores_df["score_composite"].values >= filtering_threshold
        n_before = len(df)
        df = df[keep_mask].reset_index(drop=True)
        image_embeddings = image_embeddings[keep_mask]
        logger.info(
            f"Filtering: kept {len(df):,}/{n_before:,} samples "
            f"(threshold={filtering_threshold}, removed {n_before - len(df):,})"
        )

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
    umap_model_rel = None

    if store_vectors:
        logger.info("Atlas will store full embedding vectors; fitting UMAP model")
        reducer.fit(image_embeddings)
    else:
        logger.info("Atlas will store 2D projections; fitting and transforming")
        projection = reducer.fit_transform(image_embeddings)
        df["x"] = projection[:, 0]
        df["y"] = projection[:, 1]

    # Save UMAP model. FIRST neutralise the numpy RNG objects so the pickle loads under ANY numpy:
    # the atlas BUILD may run on numpy 2.x, but the atlas SERVER is pinned numpy<2 (embedding-atlas ->
    # pynndescent<0.6 -> numpy<2). A numpy-2.x-pickled RNG state otherwise fails to unpickle on the
    # server ("MT19937 is not a known BitGenerator module") -> UMAP recomputes on every restart. The RNG
    # is not needed for transform. (Same neutralisation as scripts/convert_umap_model.py, applied here so
    # FUTURE builds are compatible by construction and never need the one-off convert.)
    def _neutralize_rng(obj, seen):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for k, v in list(d.items()):
                if isinstance(v, (np.random.RandomState, np.random.Generator, np.random.BitGenerator)):
                    d[k] = None
                elif hasattr(v, "__dict__") and not isinstance(v, np.ndarray):
                    _neutralize_rng(v, seen)

    try:
        if hasattr(reducer, "random_state"):
            reducer.random_state = None
        _neutralize_rng(reducer, set())
        joblib.dump(reducer, umap_model_path)
        umap_model_rel = os.path.relpath(umap_model_path, output_dir)
        logger.info(f"Saved UMAP model (numpy-portable; RNG neutralised) to {umap_model_path}")
    except Exception as exc:
        logger.warning(f"Failed to save UMAP model: {exc}")
        umap_model_rel = None

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

    # Create config
    _create_atlas_config(
        output_dir=output_dir,
        atlas_path=atlas_path,
        image_embeddings=image_embeddings,
        args=args,
        store_vectors=store_vectors,
        umap_model_rel=umap_model_rel,
    )

    # Launch CLI if requested
    if launch_cli:
        _launch_atlas_cli(atlas_path, store_vectors, label_column=label_column)

    return atlas_df


def _create_atlas_config(
    output_dir: str,
    atlas_path: str,
    image_embeddings: np.ndarray,
    args,
    store_vectors: bool,
    umap_model_rel: Optional[str],
):
    """Create embedding pipeline config JSON."""
    artifacts_dir = os.path.join(output_dir, "embeddings")
    analysis_dir = os.path.join(output_dir, "analysis")

    # Check for model sources
    def _parse_sources(raw_value: Optional[str]) -> list[str]:
        if not raw_value:
            return []
        return [part.strip() for part in raw_value.split(",") if part and part.strip()]

    model_sources_list: list[str] = []
    primary_sources = _parse_sources(
        getattr(args, "model_path", None)
    ) or _parse_sources(getattr(args, "model_name", None))
    model_sources_list.extend(primary_sources)

    ssl_source = getattr(args, "ssl_model", None)
    if ssl_source:
        cleaned_ssl = str(ssl_source).strip()
        if cleaned_ssl:
            model_sources_list.append(cleaned_ssl)

    # Deduplicate while preserving order
    seen_sources: set[str] = set()
    deduped_sources: list[str] = []
    for source in model_sources_list:
        if source not in seen_sources:
            deduped_sources.append(source)
            seen_sources.add(source)
    model_sources_list = deduped_sources

    def _infer_model_type(source: str) -> str:
        lowered = source.lower()
        for ssl_name in SSL_MODEL_NAMES:
            if ssl_name.lower() in lowered:
                return "ssl"
        return "clip"

    # Check for SVD models
    image_svd_rel = None
    text_svd_rel = None
    if os.path.exists(os.path.join(artifacts_dir, "svd_image.joblib")):
        image_svd_rel = os.path.join("embeddings", "svd_image.joblib")
    if os.path.exists(os.path.join(artifacts_dir, "svd_text.joblib")):
        text_svd_rel = os.path.join("embeddings", "svd_text.joblib")

    # Check for FAISS index
    faiss_rel = None
    faiss_path = os.path.join(analysis_dir, "faiss_index.bin")
    if os.path.exists(faiss_path):
        faiss_rel = os.path.join("analysis", "faiss_index.bin")

    # Check for projector artifacts
    projector_rel = None
    whitening_rel = None
    projector_config_rel = None
    if os.path.exists(os.path.join(artifacts_dir, "projector_model.pth")):
        projector_rel = os.path.join("embeddings", "projector_model.pth")
    if os.path.exists(os.path.join(artifacts_dir, "whitening_stats.npz")):
        whitening_rel = os.path.join("embeddings", "whitening_stats.npz")
    if os.path.exists(os.path.join(artifacts_dir, "projector_config.json")):
        projector_config_rel = os.path.join("embeddings", "projector_config.json")

    config = {
        "version": 1,
        "output_dir": output_dir,
        "skinmap_root": str(PROJECT_ROOT),
        "models": [
            {
                "source": model_name.strip(),
                "type": _infer_model_type(model_name),
                "is_local": os.path.exists(model_name.strip()),
            }
            for model_name in model_sources_list
        ],
        "svd": {
            "image": image_svd_rel,
            "text": text_svd_rel,
            "components": getattr(args, "svd_components", None),
        },
        "artifacts": {
            "embeddings_npz": os.path.join("embeddings", "embeddings.npz"),
            "dataframe_csv": os.path.join("embeddings", "dataframe.csv"),
            "atlas_parquet": (
                os.path.relpath(atlas_path, output_dir)
                if os.path.exists(atlas_path)
                else None
            ),
            "faiss_index": faiss_rel,
        },
        "vector": {
            "dimension": int(image_embeddings.shape[1]),
            "column": ATLAS_VECTOR_COLUMN,
            "stored_in_atlas": bool(store_vectors),
        },
    }

    # Add projector information
    if projector_rel is not None:
        config["projector"] = {
            "model": projector_rel,
            "whitening_stats": whitening_rel,
            "config": projector_config_rel,
            "used": True,
        }
    else:
        config["projector"] = {"used": False}

    # Add UMAP information
    if umap_model_rel is not None:
        config["umap"] = {
            "model": umap_model_rel,
            "n_neighbors": args.umap_n_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": args.umap_metric,
            "fast_mode": bool(getattr(args, "umap_fast", False)),
        }

    # Save config
    config_path = os.path.join(output_dir, "embedding_pipeline_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved embedding pipeline config to {config_path}")


def _launch_atlas_cli(
    atlas_path: str,
    store_vectors: bool,
    label_column: Optional[str] = None,
):
    """Launch Embedding Atlas CLI."""
    cmd = [
        "embedding-atlas",
        atlas_path,
        "--text",
        "text",
        "--image",
        "image",
        "--host",
        "0.0.0.0",
        "--port",
        "8888",
    ]

    if label_column:
        cmd.extend(["--label-column", label_column])

    if store_vectors:
        cmd.extend(["--vector", ATLAS_VECTOR_COLUMN])
    else:
        cmd.extend(["--x", "x", "--y", "y"])

    logger.info(f"Running Embedding Atlas CLI: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch Embedding Atlas CLI: {e}")
    except FileNotFoundError:
        logger.error(
            "Embedding Atlas CLI not found. Install with: pip install embedding-atlas"
        )
