#!/usr/bin/env python3
"""Create SkinMap - Refactored version using modular architecture.

This script replaces the monolithic create_skinmap.py (3400 lines) with a clean,
modular implementation using the skinmap.* package.

Key improvements:
- Bug-free dimension handling in projector pipeline
- Consistent corrupted sample tracking
- Proper cache validation with configuration tracking
- Clean separation of concerns across modules
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger

# Import functions from other modules
from src.core.src.datasets.helper import DatasetName, get_dataset
from src.core.src.pkg.embedder import Embedder
from src.embedding_fusion import BuildSpec, build_model

# Import refactored skinmap modules
from src.skinmap.cache.manager import CacheConfig, EmbeddingCache
from src.skinmap.data.preprocessing import normalize_multilabel_columns
from src.skinmap.data.transforms import get_imagenet_transform
from src.skinmap.embeddings.extractors import (
    extract_clip_embeddings,
    extract_ssl_embeddings,
)
from src.skinmap.embeddings.fusion import combine_embeddings_simple
from src.skinmap.embeddings.projector import extract_combined_embeddings_with_projector
from src.skinmap.evaluation.downstream import (
    evaluate_downstream_with_balancing_comparison,
    evaluate_image_dataset,
)
from src.skinmap.models.loaders import SSL_MODEL_NAMES, ModelInfo, load_multiple_models
from src.skinmap.utils.metadata import (
    evaluate_metadata_with_balancing_comparison,
    predict_missing_metadata,
    predict_random_baseline_metadata,
)
from src.skinmap.utils.metrics import compute_learned_metric_transformation
from src.skinmap.visualization.atlas import generate_atlas
from src.train_clip import load_model_and_processor, set_seed, setup_logger

# Re-export for backward compatibility with combined_embedder.py
__all__ = ["SSL_MODEL_NAMES", "get_imagenet_transform"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create SkinMap with modular refactored architecture"
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="suinleelab/monet",
        help="HuggingFace model name or comma-separated list for multiple models",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local model path or comma-separated list for multiple models",
    )
    parser.add_argument(
        "--ssl_model",
        type=str,
        default=None,
        help="SSL model name (dino_qderma, ibot_qderma, panderm_base, panderm_large, etc.)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint file for SSL models (required for PanDerm models)",
    )
    parser.add_argument(
        "--combine_models",
        action="store_true",
        help="Combine embeddings from multiple models",
    )

    # Fusion parameters
    parser.add_argument(
        "--svd_components",
        type=int,
        default=None,
        help="Number of SVD components for dimensionality reduction",
    )
    parser.add_argument(
        "--use_trained_projector",
        action="store_true",
        help="Use trained projector pipeline (vs simple concat+SVD)",
    )
    parser.add_argument(
        "--projector_dim",
        type=int,
        default=768,
        help="Output dimension of trained projector",
    )
    parser.add_argument(
        "--projector_type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Projector architecture",
    )
    parser.add_argument(
        "--projector_loss",
        type=str,
        default="clip",
        choices=["clip", "structure"],
        help="Loss function for projector training",
    )
    parser.add_argument(
        "--projector_epochs",
        type=int,
        default=5,
        help="Training epochs for projector",
    )
    parser.add_argument(
        "--projector_lr",
        type=float,
        default=1e-3,
        help="Learning rate for projector training",
    )
    parser.add_argument(
        "--projector_no_domain_balance",
        action="store_true",
        help="Disable domain-balanced batching for projector",
    )
    parser.add_argument(
        "--projector_pca_samples",
        type=int,
        default=100_000,
        help="Samples for PCA initialization of projector",
    )
    parser.add_argument(
        "--projector_hierarchy_lambda",
        type=float,
        default=10.0,
        help="Lambda for structure regularization when projector_loss=structure",
    )
    parser.add_argument(
        "--projector_hierarchy_levels",
        type=int,
        default=1,
        help="Number of hierarchy levels for structure loss",
    )
    parser.add_argument(
        "--projector_hierarchy_warmup",
        type=int,
        default=500,
        help="Warmup steps for hierarchy lambda ramp-up",
    )
    parser.add_argument(
        "--projector_hierarchy_weighting",
        type=str,
        default="none",
        choices=["none", "inverse"],
        help="Weighting schedule across hierarchy levels",
    )
    parser.add_argument(
        "--projector_hierarchy_margin",
        type=float,
        default=0.0,
        help="Margin applied inside hierarchical consistency regularization",
    )
    parser.add_argument(
        "--projector_consistency_lambda",
        type=float,
        default=0.0,
        help="Optional auxiliary consistency regularization weight",
    )
    parser.add_argument(
        "--max_whitening_dim",
        type=int,
        default=768,
        help="Maximum dimension for whitening (truncation threshold)",
    )
    parser.add_argument(
        "--skip_whitening",
        action="store_true",
        help="Skip whitening step (faster, let projector learn)",
    )
    parser.add_argument(
        "--projector_wandb",
        action="store_true",
        help="Log projector training metrics to Weights & Biases",
    )
    parser.add_argument(
        "--projector_wandb_project",
        type=str,
        default=None,
        help="Project name for projector wandb logging (defaults to SkinMapProjector)",
    )
    parser.add_argument(
        "--projector_wandb_run_name",
        type=str,
        default=None,
        help="Optional custom wandb run name for projector logging",
    )
    parser.add_argument(
        "--projector_validation_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation during projector training (default 0.1)",
    )
    parser.add_argument(
        "--projector_validate_every_n_epochs",
        type=int,
        default=1,
        help=(
            "Run validation every N epochs during projector training; "
            "set to 0 to skip validation"
        ),
    )

    # Data parameters
    parser.add_argument(
        "--data_csv",
        type=str,
        required=True,
        help="Path to CSV with img_path and description columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./assets/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--vis_samples",
        type=int,
        default=None,
        help="Number of samples for visualization (None = all)",
    )

    # System parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level",
    )

    # Features
    parser.add_argument(
        "--use_atlas",
        action="store_true",
        help="Generate atlas with thumbnails",
    )
    parser.add_argument(
        "--thumbnail_size",
        "--thumbnail_max_size",
        dest="thumbnail_size",
        type=int,
        default=256,
        help="Maximum thumbnail dimension",
    )
    parser.add_argument(
        "--predict_metadata",
        "--predict_missing",
        dest="predict_metadata",
        action="store_true",
        help="Predict missing metadata attributes",
    )
    parser.add_argument(
        "--metadata_attributes",
        "--prediction_attributes",
        dest="metadata_attributes",
        type=str,
        nargs="+",
        default=[
            "laterality",
            "body_region",
            "gender",
            "age",
            "fitzpatrick",
            "origin",
        ],
        help="List of metadata attributes to predict (space or comma separated)",
    )
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="Run downstream evaluation on image datasets",
    )
    parser.add_argument(
        "--use_learned_metric",
        action="store_true",
        help="Use learned metric transformation for evaluation",
    )
    parser.add_argument(
        "--metric_label_col",
        type=str,
        default="icd_category",
        help=(
            "Column to use for metric learning. "
            "Recommended ICD fields: icd_code (full), icd_category (3-char), "
            "icd_block, icd_chapter. Default: icd_category; falls back to condition if missing."
        ),
    )

    # UMAP parameters
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter",
    )
    parser.add_argument(
        "--umap_metric",
        type=str,
        default="cosine",
        help="UMAP distance metric",
    )
    parser.add_argument(
        "--umap_fast",
        action="store_true",
        help="Enable fast UMAP mode (non-reproducible but faster multi-threading)",
    )
    parser.add_argument(
        "--umap_n_jobs",
        type=int,
        default=None,
        help="Number of threads for UMAP (only used with --umap_fast)",
    )

    # Visualization types
    parser.add_argument(
        "--use_static",
        action="store_true",
        help="Generate static visualization",
    )
    parser.add_argument(
        "--atlas_store_vectors",
        action="store_true",
        help="Store full embedding vectors in atlas (not just 2D projections)",
    )
    parser.add_argument(
        "--separability_eval",
        action="store_true",
        help="Evaluate dataset separability with linear probe",
    )

    # Thumbnail parameters
    parser.add_argument(
        "--thumbnail_quality",
        type=int,
        default=85,
        help="JPEG quality for thumbnails (1-100)",
    )
    parser.add_argument(
        "--prediction_test_size",
        type=float,
        default=0.2,
        help="Test size for metadata prediction evaluation",
    )
    parser.add_argument(
        "--random_baseline",
        action="store_true",
        help="Use random baseline predictor instead of XGBoost for metadata prediction and downstream evaluation",
    )
    parser.add_argument(
        "--balancing_comparison",
        action="store_true",
        help="Run additional evaluation comparing imbalanced vs under/over-sampled training",
    )
    parser.add_argument(
        "--bootstrap_uncertainty",
        action="store_true",
        help="Enable bootstrap analysis to compute confidence intervals for predictions",
    )
    parser.add_argument(
        "--bootstrap_n_iterations",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for uncertainty estimation (default: 1000)",
    )

    return parser.parse_args()


def generate_run_name(args, model_names: List[str]) -> str:
    """Generate descriptive run name based on configuration.

    Uses the same cleaning logic as get_single_model_cache_path() to ensure
    consistency between cache paths and run names.
    """
    is_multiple = len(model_names) > 1

    if args.ssl_model and not is_multiple:
        return f"ssl_{args.ssl_model}"

    if is_multiple:
        model_parts = []
        for name in model_names:
            if "/" in name:
                if name.startswith("/") or name.startswith("."):
                    model_parts.append("_".join(Path(name).parts[-2:]))
                else:
                    # Apply same cleaning as cache path to ensure consistency
                    cleaned = name.replace("assets/", "").replace("/", "_")
                    model_parts.append(cleaned)
            else:
                model_parts.append(name)

        run_name = "combined_" + "_".join(model_parts[:3])
        if len(model_names) > 3:
            run_name += f"_and_{len(model_names)-3}_more"
        if args.svd_components:
            run_name += f"_svd{args.svd_components}"
        if args.use_trained_projector:
            run_name += "-trained_projector"
        return run_name

    # For single model, use same cleaning logic as get_single_model_cache_path()
    # to ensure run_name matches the folder where individual cache is stored
    name = model_names[0].replace("assets/", "").replace("/", "_").replace("\\", "_")
    if args.model_path:
        name += f"-{Path(args.model_path).parent.stem}-{Path(args.model_path).stem}"
    return name


def load_models(args, device: torch.device) -> List[ModelInfo]:
    """Load models based on arguments."""
    is_multiple = (
        ("," in args.model_name)
        or (args.model_path and "," in args.model_path)
        or args.combine_models
    )

    if args.ssl_model and not is_multiple:
        logger.info(f"Loading single SSL model: {args.ssl_model}")
        kwargs = {"n_head_layers": 0}
        if args.checkpoint_path:
            kwargs["checkpoint_path"] = args.checkpoint_path
        ssl_model, _, _ = Embedder.load_pretrained(
            args.ssl_model, return_info=True, **kwargs
        )
        ssl_model.to(device)
        return [ModelInfo(ssl_model, None, "ssl", args.ssl_model)]

    elif is_multiple:
        model_source = args.model_path if args.model_path else args.model_name
        logger.info(f"Loading multiple models: {model_source}")
        return load_multiple_models(model_source, device)

    else:
        model_source = args.model_path if args.model_path else args.model_name
        logger.info(f"Loading single CLIP model: {model_source}")
        model, processor = load_model_and_processor(model_source, device)
        return [ModelInfo(model, processor, "clip", model_source)]


def determine_fusion_method(args, models: List[ModelInfo]) -> str:
    """Infer fusion method string for caching/metadata."""
    if len(models) <= 1:
        return "none"
    if args.use_trained_projector:
        return "trained_projector"
    if args.svd_components:
        return f"svd_{args.svd_components}"
    return "concat"


def parse_metadata_attributes(raw_attributes: Optional[List[str]]) -> List[str]:
    """Normalise metadata attribute CLI input (supports comma or space separated)."""
    if not raw_attributes:
        return []

    parsed: List[str] = []
    for value in raw_attributes:
        if not value:
            continue
        parts = value.split(",") if "," in value else [value]
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                parsed.append(cleaned)
    return parsed


def check_metadata_update(
    new_df: pd.DataFrame,
    cache: EmbeddingCache,
    run_name: str,
    cache_config: CacheConfig,
    args,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[int], str]]:
    """Check if we can reuse cached embeddings with updated metadata.

    Args:
        new_df: New dataframe with potentially updated metadata
        cache: Cache manager
        run_name: Run name for cache lookup
        cache_config: Cache configuration
        args: Command line arguments

    Returns:
        Tuple of (image_embs, text_embs, labels, merged_df, corrupted, fusion_method) if reuse is possible and user agrees,
        None otherwise
    """
    emb_dir = os.path.join(args.output_dir, run_name, "embeddings")
    cached_df_path = os.path.join(emb_dir, "dataframe.csv")

    # Check if cached dataframe exists (check this first, before config validation)
    if not os.path.exists(cached_df_path):
        return None

    # Load cached dataframe to check for metadata changes
    try:
        cached_df = normalize_multilabel_columns(pd.read_csv(cached_df_path))
    except Exception as e:
        logger.debug(f"Failed to load cached dataframe: {e}")
        return None

    # Check if cache embeddings exist (without config validation yet)
    cache_path = cache.get_combined_cache_path(run_name)
    if not os.path.exists(cache_path):
        return None

    # Verify both dataframes have img_path column
    if "img_path" not in cached_df.columns or "img_path" not in new_df.columns:
        logger.debug("Missing img_path column - cannot check for metadata updates")
        return None

    # Get image paths
    cached_paths = set(cached_df["img_path"].values)
    new_paths = set(new_df["img_path"].values)

    # Check if all cached images are in the new CSV (old is subset of new)
    if not cached_paths.issubset(new_paths):
        missing_count = len(cached_paths - new_paths)
        logger.debug(
            f"Cached data contains {missing_count} images not in new CSV - cannot reuse"
        )
        return None

    # Check if there are new images to add
    new_images = new_paths - cached_paths

    # Check if metadata has changed for existing images
    # Compare values for common columns
    common_cols = set(cached_df.columns) & set(new_df.columns)
    common_cols.discard("img_path")  # Don't compare the join key

    changed_count = 0
    if common_cols:
        # Merge to align rows
        comparison_df = cached_df[["img_path"] + list(common_cols)].merge(
            new_df[["img_path"] + list(common_cols)],
            on="img_path",
            how="inner",
            suffixes=("_old", "_new"),
        )

        # Check for differences
        for col in common_cols:
            old_col = f"{col}_old"
            new_col = f"{col}_new"
            if old_col in comparison_df.columns and new_col in comparison_df.columns:
                # Handle NaN comparisons properly
                diff_mask = (comparison_df[old_col] != comparison_df[new_col]) & ~(
                    comparison_df[old_col].isna() & comparison_df[new_col].isna()
                )
                changed_count += diff_mask.sum()

    # Check if there are new columns in new_df
    new_cols = set(new_df.columns) - set(cached_df.columns) - {"img_path"}

    # If nothing changed and no new images, no need to ask
    if changed_count == 0 and len(new_images) == 0 and len(new_cols) == 0:
        logger.debug("Metadata has not changed - proceeding with normal cache loading")
        return None

    # Ask user if they want to reuse embeddings with updated metadata
    logger.info("=" * 60)
    logger.info("METADATA UPDATE DETECTED")
    logger.info("=" * 60)
    logger.info(f"Cached embeddings exist for {len(cached_paths)} images")
    logger.info(f"New CSV contains {len(new_paths)} images")
    if changed_count > 0:
        logger.info(f"Metadata values changed: {changed_count} cell(s)")
    if new_cols:
        logger.info(f"New metadata columns: {', '.join(sorted(new_cols))}")
    if new_images:
        logger.info(f"New images in CSV (will NOT be embedded): {len(new_images)}")
        logger.warning(
            "Note: Only images with cached embeddings will be included. "
            "New images will be ignored."
        )
    logger.info("")

    # Get user confirmation
    try:
        response = (
            input("Reuse existing embeddings with updated metadata? (y/n): ")
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        logger.info("User cancelled - proceeding with normal pipeline")
        return None

    if response != "y":
        logger.info("User declined - proceeding with normal pipeline")
        return None

    # User agreed - now load the cached embeddings
    # Note: We intentionally skip config validation here since user wants to reuse
    # existing embeddings regardless of config changes
    logger.info("Loading cached embeddings (skipping config validation)...")
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            image_embs = data["image_embeddings"]
            text_embs = data["text_embeddings"]
            labels = data["dataset_labels"]

            # Convert numpy-wrapped None to actual None
            if text_embs is not None:
                if text_embs.dtype == object and text_embs.shape == ():
                    text_embs = None

            # Get corrupted indices
            corrupted = data.get(
                "corrupted_indices", np.array([], dtype=np.int64)
            ).tolist()

            # Get fusion method from cache
            fusion_method_raw = data.get("fusion_method", "none")
            if (
                isinstance(fusion_method_raw, np.ndarray)
                and fusion_method_raw.ndim == 0
            ):
                fusion_method = fusion_method_raw.item()
            else:
                fusion_method = str(fusion_method_raw)

        logger.info(f"Loaded embeddings: {image_embs.shape}")
    except Exception as e:
        logger.error(f"Failed to load cached embeddings: {e}")
        return None

    # Check for duplicates in new_df (would cause merge to create extra rows)
    new_df_duplicates = new_df[new_df.duplicated(subset=["img_path"], keep=False)]
    if len(new_df_duplicates) > 0:
        logger.error("=" * 60)
        logger.error("DUPLICATE img_path VALUES DETECTED IN NEW CSV")
        logger.error("=" * 60)
        logger.error(
            f"Found {len(new_df_duplicates)} duplicate rows for "
            f"{new_df_duplicates['img_path'].nunique()} unique paths"
        )
        logger.error("Example duplicates:")
        for path in new_df_duplicates["img_path"].unique()[:5]:
            dup_rows = new_df[new_df["img_path"] == path]
            logger.error(f"  {path}: {len(dup_rows)} occurrences")
        logger.error("")
        logger.error("Please fix your CSV to have unique img_path values.")
        logger.error(
            "Cannot merge metadata with duplicates - proceeding with normal pipeline."
        )
        return None

    # Merge new metadata with cached dataframe
    # Keep only images that have cached embeddings (preserve order!)
    merged_df = cached_df[["img_path"]].merge(
        new_df,
        on="img_path",
        how="left",  # Keep all cached images
    )

    # Verify merge preserved order and size
    if len(merged_df) != len(cached_df):
        # This should never happen now that we check for duplicates, but keep as sanity check
        logger.error(
            f"MERGE ERROR: Dataframe size changed from {len(cached_df)} to {len(merged_df)}. "
            "This indicates duplicate img_path values in your CSV."
        )
        # Show which paths got duplicated
        merged_path_counts = merged_df["img_path"].value_counts()
        duplicated_paths = merged_path_counts[merged_path_counts > 1]
        if len(duplicated_paths) > 0:
            logger.error(
                f"Paths that appear multiple times after merge ({len(duplicated_paths)} total):"
            )
            for path, count in duplicated_paths.head(10).items():
                logger.error(f"  {path}: {count} times")
        raise RuntimeError(
            f"Merge created {len(merged_df) - len(cached_df)} extra rows. "
            "Fix duplicate img_path values in your CSV."
        )

    if not merged_df["img_path"].equals(cached_df["img_path"]):
        raise RuntimeError(
            "MERGE ERROR: Image path order changed. This breaks alignment with embeddings array."
        )

    # Save updated dataframe to cache immediately
    merged_df.to_csv(cached_df_path, index=False)
    logger.info(f"Updated cached dataframe with new metadata: {cached_df_path}")

    logger.info("=" * 60)
    logger.info(
        f"Successfully reused {len(image_embs)} embeddings with updated metadata"
    )
    logger.info("Skipping embedding extraction - proceeding to analysis")
    logger.info("=" * 60)

    return image_embs, text_embs, labels, merged_df, corrupted, fusion_method


def extract_or_load_embeddings(
    models: List[ModelInfo],
    df: pd.DataFrame,
    args,
    device: torch.device,
    cache: EmbeddingCache,
    cache_config: CacheConfig,
    run_name: str,
    wandb_run=None,
) -> tuple:
    """Extract embeddings or load from cache.

    Args:
        wandb_run: Optional wandb run instance for logging metrics

    Returns:
        Tuple of (image_embs, text_embs, labels, filtered_df, corrupted, artifacts, fusion_method, wandb_run)
    """
    emb_dir = os.path.join(args.output_dir, run_name, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    df_cache_path = os.path.join(emb_dir, "dataframe.csv")

    cached_data = cache.load_combined(run_name, expected_config=cache_config)

    if cached_data is not None:
        logger.info("Loaded embeddings from cache")
        image_embs, text_embs, dataset_labels, metadata = cached_data
        corrupted = metadata.get("corrupted_indices") or []
        fusion_method = metadata["config"].fusion_method

        # STRICT VALIDATION: Cache must have valid data
        if image_embs is None or len(image_embs) == 0:
            raise RuntimeError(
                f"CACHE CORRUPTION: Loaded cache from {run_name} has 0 samples. "
                f"This indicates a failed previous run. "
                f"Delete the cache directory and re-run: rm -rf {os.path.join(args.output_dir, run_name)}"
            )

        # STRICT VALIDATION: Dataframe must exist alongside embeddings
        if not os.path.exists(df_cache_path):
            raise RuntimeError(
                f"CACHE CORRUPTION: Embeddings cache exists but dataframe is missing.\n"
                f"Expected: {df_cache_path}\n"
                f"This indicates incomplete cache save from a previous run.\n"
                f"Delete the cache directory and re-run: rm -rf {os.path.join(args.output_dir, run_name)}"
            )

        filtered_df = normalize_multilabel_columns(pd.read_csv(df_cache_path))

        # STRICT VALIDATION: Dimensions must match
        if len(filtered_df) != len(image_embs):
            raise RuntimeError(
                f"CACHE CORRUPTION: Dimension mismatch between cached dataframe and embeddings.\n"
                f"Dataframe samples: {len(filtered_df)}\n"
                f"Embedding samples: {len(image_embs)}\n"
                f"This should never happen. Delete cache and re-run: rm -rf {os.path.join(args.output_dir, run_name)}"
            )

        logger.info(f"Cache validation passed: {len(image_embs)} samples")

        return (
            image_embs,
            text_embs,
            dataset_labels,
            filtered_df,
            corrupted,
            {},
            fusion_method,
            wandb_run,
        )

    logger.info("Computing embeddings (cache miss or incompatible)")
    artifacts: dict = {}
    corrupted: List[int] = []
    fusion_method = cache_config.fusion_method

    if len(models) == 1:
        # Single model extraction - check individual model cache first
        model_info = models[0]
        logger.info(f"Single model extraction: {model_info.model_type}")

        # Try loading from individual model cache (may exist from previous ensemble run)
        individual_cache_result = cache.load_single_model(model_info.model_path)

        if individual_cache_result is not None:
            logger.info(
                "Found individual model cache from previous run (likely from ensemble)"
            )
            image_embs, text_embs, labels, filtered_df = individual_cache_result
            corrupted = []  # Corrupted samples already filtered in cached embeddings
            logger.info(f"Loaded from individual cache: {len(image_embs)} samples")
        else:
            # Extract embeddings if not cached
            if model_info.model_type == "ssl":
                image_embs, text_embs, labels, corrupted, filtered_df = (
                    extract_ssl_embeddings(
                        model_info.model,
                        df,
                        device,
                        batch_size=args.batch_size,
                        max_samples=args.vis_samples,
                        dataset_col="dataset_desc",
                        num_workers=args.num_workers,
                    )
                )
            else:
                image_embs, text_embs, labels, corrupted, filtered_df = (
                    extract_clip_embeddings(
                        model_info.model,
                        model_info.processor,
                        df,
                        device,
                        batch_size=args.batch_size,
                        max_samples=args.vis_samples,
                        dataset_col="dataset_desc",
                        num_workers=args.num_workers,
                    )
                )

            logger.info(
                f"Extracted: {len(image_embs)} samples, {len(corrupted)} corrupted"
            )

            # Save to individual model cache for future reuse
            cache.save_single_model(
                model_info.model_path,
                image_embs,
                text_embs,
                labels,
                filtered_df,
            )
            logger.info(f"Saved to individual model cache: {model_info.model_path}")

    else:
        # Multiple models - use appropriate fusion pipeline
        if args.use_trained_projector:
            logger.info("Using trained projector pipeline")
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
                models,
                df,
                device,
                cache,
                batch_size=args.batch_size,
                max_samples=args.vis_samples,
                dataset_col="dataset_desc",
                num_workers=args.num_workers,
                projector_dim=args.projector_dim,
                projector_type=args.projector_type,
                n_epochs=args.projector_epochs,
                learning_rate=args.projector_lr,
                use_domain_balanced=not args.projector_no_domain_balance,
                pca_init_samples=args.projector_pca_samples,
                max_whitening_dim=args.max_whitening_dim,
                skip_whitening=args.skip_whitening,
                loss_type=args.projector_loss,
                hierarchy_lambda=args.projector_hierarchy_lambda,
                hierarchy_levels=args.projector_hierarchy_levels,
                hierarchy_warmup=args.projector_hierarchy_warmup,
                hierarchy_weighting=args.projector_hierarchy_weighting,
                hierarchy_margin=args.projector_hierarchy_margin,
                consistency_lambda=args.projector_consistency_lambda,
                log_to_wandb=args.projector_wandb,
                wandb_project=args.projector_wandb_project,
                wandb_run_name=args.projector_wandb_run_name,
                validation_split=args.projector_validation_split,
                validate_every_n_epochs=args.projector_validate_every_n_epochs,
            )
            artifacts = {
                "projector_model": projector_model,
                "whitening_stats": whitening_stats,
            }
        else:
            logger.info("Using simple concatenation + SVD pipeline")
            (
                image_embs,
                text_embs,
                labels,
                corrupted,
                svd_image,
                svd_text,
                filtered_df,
            ) = combine_embeddings_simple(
                models,
                df,
                device,
                cache,
                batch_size=args.batch_size,
                max_samples=args.vis_samples,
                dataset_col="dataset_desc",
                num_workers=args.num_workers,
                svd_components=args.svd_components,
            )
            artifacts = {"svd_image_model": svd_image, "svd_text_model": svd_text}

    # Save to cache (including dataframe to ensure consistency)
    logger.info(f"Saving embeddings to cache: {run_name}")
    cache.save_combined(
        run_name,
        image_embs,
        text_embs,
        labels,
        cache_config,
        extra_data={"corrupted_indices": np.array(corrupted, dtype=np.int64)},
    )

    # Save dataframe immediately with embeddings to prevent dimension mismatch on reload
    filtered_df.to_csv(df_cache_path, index=False)
    logger.info(f"Saved filtered dataframe to {df_cache_path}")

    return (
        image_embs,
        text_embs,
        labels,
        filtered_df,
        corrupted,
        artifacts,
        fusion_method,
        wandb_run,
    )


def reuse_cached_metadata_predictions(
    filtered_df: pd.DataFrame,
    metadata_attributes: List[str],
    cached_df_path: str,
) -> Tuple[pd.DataFrame, bool]:
    """Merge cached *_pred columns when they fully cover requested attributes."""
    if not metadata_attributes or not os.path.exists(cached_df_path):
        return filtered_df, False

    try:
        cached_df = normalize_multilabel_columns(pd.read_csv(cached_df_path))
    except Exception as exc:
        logger.warning(f"Failed to load cached dataframe for metadata reuse: {exc}")
        return filtered_df, False

    if "img_path" not in cached_df.columns:
        logger.info(
            "Cached dataframe missing 'img_path'; cannot reuse metadata predictions"
        )
        return filtered_df, False

    pred_cols = [col for col in cached_df.columns if col.endswith("_pred")]
    if not pred_cols:
        return filtered_df, False

    # Store original length to verify merge doesn't change row count
    original_len = len(filtered_df)

    merged_df = filtered_df.merge(
        cached_df[["img_path"] + pred_cols],
        on="img_path",
        how="left",
    )

    # CRITICAL: Verify merge didn't change the number of rows
    # This can happen if cached_df has duplicate img_paths
    if len(merged_df) != original_len:
        raise RuntimeError(
            f"CACHE CORRUPTION: Metadata prediction merge changed row count from {original_len} to {len(merged_df)}.\n"
            f"This breaks alignment with embeddings array (dimension is {original_len}).\n"
            f"The cached dataframe at {cached_df_path} has duplicate img_path entries.\n"
            f"Delete cache and re-run: rm -rf {os.path.dirname(cached_df_path)}"
        )

    for attribute in metadata_attributes:
        pred_col = f"{attribute}_pred"
        if pred_col not in merged_df.columns:
            return filtered_df, False
        missing_mask = merged_df[attribute].isna()
        if missing_mask.any() and merged_df.loc[missing_mask, pred_col].isna().any():
            return filtered_df, False

    return merged_df, True


def load_cached_separability_results(sep_path: str) -> Optional[pd.DataFrame]:
    """Load cached separability CSV if present and non-empty."""
    if not os.path.exists(sep_path):
        return None

    try:
        cached_df = pd.read_csv(sep_path)
    except Exception as exc:
        logger.warning(f"Failed to load cached separability results: {exc}")
        return None

    if cached_df.empty:
        logger.info("Cached separability file is empty; ignoring")
        return None

    return cached_df


def save_outputs(
    df: pd.DataFrame,
    image_embeddings: np.ndarray,
    text_embeddings: Optional[np.ndarray],
    dataset_labels: np.ndarray,
    corrupted_indices: List[int],
    fusion_method: str,
    args,
    output_dir: str,
    run_name: str,
    artifacts: dict,
    model_names: List[str],
):
    """Persist dataframe, embeddings, and artifact metadata (legacy compatible)."""
    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    # Save combined NPZ + dataframe for compatibility with original pipeline
    np.savez(
        os.path.join(emb_dir, "embeddings.npz"),
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        dataset_labels=dataset_labels,
        corrupted_images=np.array(corrupted_indices, dtype=np.int64),
        fusion_method=fusion_method,
    )
    df.to_csv(os.path.join(emb_dir, "dataframe.csv"), index=False)
    logger.info(f"Saved embeddings bundle to {emb_dir}")

    # Persist auxiliary artifacts (SVD/projector/etc.)
    persist_artifacts(
        emb_dir=emb_dir,
        artifacts=artifacts,
        args=args,
        model_names=model_names,
        text_embeddings=text_embeddings,
    )

    # Lightweight run configuration for reproducibility
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {
                key: convert_to_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    config = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "ssl_model": args.ssl_model,
        "run_name": run_name,
        "embedding_dim": int(image_embeddings.shape[1]),
        "num_samples": int(len(df)),
        "num_corrupted": int(len(corrupted_indices)),
        "fusion_method": fusion_method,
        "use_trained_projector": args.use_trained_projector,
        "projector_dim": args.projector_dim if args.use_trained_projector else None,
        "svd_components": args.svd_components,
        "max_whitening_dim": args.max_whitening_dim,
        "skip_whitening": args.skip_whitening,
    }

    # Convert any numpy types to JSON-serializable Python types
    config = convert_to_json_serializable(config)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")


def persist_artifacts(
    emb_dir: str,
    artifacts: dict,
    args,
    model_names: List[str],
    text_embeddings: Optional[np.ndarray],
) -> None:
    """Persist optional artifacts such as SVD and projector metadata."""
    if not artifacts:
        return

    if artifacts.get("svd_image_model") is not None:
        image_svd_path = os.path.join(emb_dir, "svd_image.joblib")
        joblib.dump(artifacts["svd_image_model"], image_svd_path)
        logger.info(f"Saved image SVD model to {image_svd_path}")

    if artifacts.get("svd_text_model") is not None:
        text_svd_path = os.path.join(emb_dir, "svd_text.joblib")
        joblib.dump(artifacts["svd_text_model"], text_svd_path)
        logger.info(f"Saved text SVD model to {text_svd_path}")

    if artifacts.get("projector_model") is not None:
        projector_model = artifacts["projector_model"]
        projector_path = os.path.join(emb_dir, "projector_model.pth")
        torch.save(projector_model.state_dict(), projector_path)
        logger.info(f"Saved projector weights to {projector_path}")

        whitening_stats = artifacts.get("whitening_stats")
        if whitening_stats:
            whitening_path = os.path.join(emb_dir, "whitening_stats.npz")
            save_dict = {
                "n_models": len(whitening_stats.get("mu", [])),
                "dims": whitening_stats.get("dims", []),
                "original_dims": whitening_stats.get(
                    "original_dims", whitening_stats.get("dims", [])
                ),
                "clip_indices": whitening_stats.get("clip_indices", []),
            }

            mu_list = whitening_stats.get("mu", [])
            W_list = whitening_stats.get("W", [])
            if mu_list:
                save_dict["mu"] = np.array(mu_list, dtype=object)
            if W_list:
                save_dict["W"] = np.array(W_list, dtype=object)

            text_whitening = whitening_stats.get("text_whitening")
            if text_whitening:
                text_mu = text_whitening.get("mu", [])
                text_W = text_whitening.get("W", [])
                save_dict["n_text_models"] = len(text_mu)
                save_dict["text_dims"] = text_whitening.get("dims", [])
                if text_mu:
                    save_dict["text_mu"] = np.array(text_mu, dtype=object)
                if text_W:
                    save_dict["text_W"] = np.array(text_W, dtype=object)

            np.savez(whitening_path, **save_dict)
            logger.info(f"Saved whitening statistics to {whitening_path}")

        def convert_to_json_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(
                obj,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {
                    key: convert_to_json_serializable(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj

        # Calculate text_dim from whitening stats (concatenated CLIP text dims)
        text_dim = None
        if whitening_stats is not None:
            text_whitening = whitening_stats.get("text_whitening")
            if text_whitening and text_whitening.get("dims"):
                text_dim = int(sum(text_whitening.get("dims", [])))

        projector_config = {
            "projector_type": args.projector_type,
            "projector_dim": args.projector_dim,
            "d_cat": (
                int(sum(whitening_stats.get("dims", [])))
                if whitening_stats is not None
                and whitening_stats.get("dims") is not None
                else None
            ),
            "text_dim": text_dim,
            "teacher_dims": (
                whitening_stats.get("dims", []) if whitening_stats is not None else None
            ),
            "model_paths": model_names,
            "clip_indices": (
                whitening_stats.get("clip_indices", [])
                if whitening_stats is not None
                else []
            ),
        }

        # Convert any numpy types to JSON-serializable Python types
        projector_config = convert_to_json_serializable(projector_config)

        projector_config_path = os.path.join(emb_dir, "projector_config.json")
        with open(projector_config_path, "w") as f:
            json.dump(projector_config, f, indent=2)
        logger.info(f"Saved projector config to {projector_config_path}")


def _load_whitening_stats(whitening_path: str) -> Optional[dict]:
    if not os.path.exists(whitening_path):
        return None

    with np.load(whitening_path, allow_pickle=True) as data:
        stats = {
            "dims": data["dims"].tolist() if "dims" in data.files else [],
            "original_dims": (
                data["original_dims"].tolist() if "original_dims" in data.files else []
            ),
            "clip_indices": (
                data["clip_indices"].tolist() if "clip_indices" in data.files else []
            ),
            "mu": data["mu"].tolist() if "mu" in data.files else [],
            "W": data["W"].tolist() if "W" in data.files else [],
        }

        text_stats = None
        if "text_dims" in data.files:
            text_stats = {
                "dims": data["text_dims"].tolist(),
                "mu": data["text_mu"].tolist() if "text_mu" in data.files else [],
                "W": data["text_W"].tolist() if "text_W" in data.files else [],
            }
        if text_stats is not None:
            stats["text_whitening"] = text_stats

    return stats


def load_projector_artifacts(
    emb_dir: str, device: torch.device
) -> Tuple[Optional[torch.nn.Module], Optional[dict], Optional[dict]]:
    projector_path = os.path.join(emb_dir, "projector_model.pth")
    projector_config_path = os.path.join(emb_dir, "projector_config.json")
    whitening_path = os.path.join(emb_dir, "whitening_stats.npz")

    if not os.path.exists(projector_path) or not os.path.exists(projector_config_path):
        return None, None, None

    with open(projector_config_path, "r") as f:
        projector_config = json.load(f)

    if (
        projector_config.get("text_dim") is None
        or projector_config.get("teacher_dims") is None
        or projector_config.get("projector_dim") is None
    ):
        logger.warning(
            "Projector config missing required fields; skipping projector loading"
        )
        return None, None, None

    spec = BuildSpec(
        teacher_names=projector_config.get("model_paths", []),
        teacher_dims=projector_config.get("teacher_dims", []),
        text_dim=projector_config.get("text_dim"),
        out_dim=projector_config.get("projector_dim"),
        kind=projector_config.get("projector_type", "linear"),
    )
    projector_model = build_model(spec)
    projector_model.to(device)
    state_dict = torch.load(projector_path, map_location=device)
    projector_model.load_state_dict(state_dict)
    projector_model.eval()

    whitening_stats = _load_whitening_stats(whitening_path)

    return projector_model, whitening_stats, projector_config


def _save_downstream_results(results: dict, output_path: str, dataset_name: str):
    """Save downstream evaluation results to CSV.

    Args:
        results: Dictionary with structure {task_type: {task_name: {classifier: metrics}}}
        output_path: Path to save CSV file
        dataset_name: Name of dataset for logging
    """
    rows = []

    for task_type in ["classification", "regression"]:
        if task_type not in results or not results[task_type]:
            continue

        for task_name, classifiers in results[task_type].items():
            for classifier_name, metrics in classifiers.items():
                row = {
                    "task_type": task_type,
                    "task_name": task_name,
                    "classifier": classifier_name,
                }
                row.update(metrics)
                rows.append(row)

    if rows:
        results_df = pd.DataFrame(rows)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved {dataset_name} downstream results to {output_path}")
    else:
        logger.warning(f"No downstream results to save for {dataset_name}")


def _log_downstream_to_wandb(results: dict, dataset_name: str):
    """Log downstream evaluation results to wandb.

    Args:
        results: Dictionary with structure {task_type: {task_name: {classifier: metrics}}}
        dataset_name: Name of dataset for logging prefix
    """
    for task_type in ["classification", "regression"]:
        if task_type not in results or not results[task_type]:
            continue

        for task_name, classifiers in results[task_type].items():
            for classifier_name, metrics in classifiers.items():
                # Create wandb logging key with clear hierarchy
                prefix = f"downstream/{dataset_name}/{task_type}/{task_name}/{classifier_name}"
                wandb_metrics = {
                    f"{prefix}/{metric_name}": value
                    for metric_name, value in metrics.items()
                }
                wandb.log(wandb_metrics)

    logger.info(f"Logged {dataset_name} downstream results to wandb")


def _save_confusion_matrices(
    confusion_matrices: dict,
    output_dir: str,
    dataset_name: str,
):
    """Save confusion matrices for classification tasks as CSVs and plots (PDF/SVG).

    Args:
        confusion_matrices: Dictionary with structure {task_name: {"confusion_matrices": {classifier: cm}, "label_encoder": encoder}}
        output_dir: Directory to save confusion matrices
        dataset_name: Name of dataset for logging
    """
    if not confusion_matrices:
        logger.warning(f"No confusion matrices to save for {dataset_name}")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create confusion matrices directory
    cm_dir = os.path.join(output_dir, f"confusion_matrices_{dataset_name}")
    os.makedirs(cm_dir, exist_ok=True)

    for task_name, task_data in confusion_matrices.items():
        cms = task_data.get("confusion_matrices", {})
        label_encoder = task_data.get("label_encoder")

        if not cms:
            continue

        # Get class labels
        class_labels = label_encoder.classes_ if label_encoder is not None else None

        for classifier_name, cm in cms.items():
            # Save as CSV with labels if available
            if class_labels is not None:
                cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
                csv_path = os.path.join(cm_dir, f"{task_name}_{classifier_name}_cm.csv")
                cm_df.to_csv(csv_path)
                logger.info(
                    f"Saved confusion matrix CSV for {task_name}/{classifier_name} to {csv_path}"
                )
            else:
                logger.warning(
                    f"No labels available for {task_name}/{classifier_name}, skipping CSV export"
                )

            # Generate and save plot with publication-quality styling
            # Set publication-quality font sizes
            plt.rcParams.update(
                {
                    "font.size": 14,
                    "axes.labelsize": 16,
                    "axes.titlesize": 18,
                    "xtick.labelsize": 13,
                    "ytick.labelsize": 13,
                    "legend.fontsize": 14,
                    "figure.titlesize": 18,
                }
            )

            plt.figure(figsize=(max(10, len(cm) * 0.8), max(8, len(cm) * 0.6)))

            # Use labels if available, otherwise use numeric indices
            display_labels = (
                class_labels if class_labels is not None else np.arange(len(cm))
            )

            # Create heatmap with annotations and larger font
            annot_fontsize = max(10, 14 - len(cm) // 5)  # Scale down for large matrices
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=display_labels,
                yticklabels=display_labels,
                cbar_kws={"label": "Count", "shrink": 0.8},
                annot_kws={"fontsize": annot_fontsize},
                linewidths=0.5,
                linecolor="gray",
            )

            plt.title(
                f"Confusion Matrix: {dataset_name} - {task_name}\n{classifier_name}",
                pad=20,
            )
            plt.ylabel("True Label", fontsize=16, labelpad=10)
            plt.xlabel("Predicted Label", fontsize=16, labelpad=10)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Save plot as PDF with publication-quality DPI
            pdf_path = os.path.join(cm_dir, f"{task_name}_{classifier_name}_cm.pdf")
            plt.savefig(
                pdf_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True
            )
            logger.info(
                f"Saved confusion matrix PDF for {task_name}/{classifier_name} to {pdf_path}"
            )

            # Save plot as SVG (vector format for publications)
            svg_path = os.path.join(cm_dir, f"{task_name}_{classifier_name}_cm.svg")
            plt.savefig(svg_path, bbox_inches="tight", format="svg", transparent=True)
            logger.info(
                f"Saved confusion matrix SVG for {task_name}/{classifier_name} to {svg_path}"
            )

            plt.close()

        # Save label encoder mapping
        if label_encoder is not None:
            labels_path = os.path.join(cm_dir, f"{task_name}_labels.json")
            label_mapping = {
                int(i): str(label) for i, label in enumerate(label_encoder.classes_)
            }
            with open(labels_path, "w") as f:
                json.dump(label_mapping, f, indent=2)
            logger.info(f"Saved label mapping for {task_name} to {labels_path}")


def _save_metadata_confusion_matrices(confusion_matrices: dict, output_dir: str):
    """Save confusion matrices for metadata prediction tasks as CSVs and plots (PDF/SVG).

    Args:
        confusion_matrices: Dictionary with structure {attribute: {"confusion_matrix": cm, "label_encoder": encoder}}
        output_dir: Directory to save confusion matrices
    """
    if not confusion_matrices:
        logger.info("No confusion matrices to save for metadata prediction")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create confusion matrices directory
    cm_dir = os.path.join(output_dir, "confusion_matrices_metadata")
    os.makedirs(cm_dir, exist_ok=True)

    for attribute, data in confusion_matrices.items():
        cm = data.get("confusion_matrix")
        label_encoder = data.get("label_encoder")

        if cm is None:
            continue

        # Get class labels
        class_labels = label_encoder.classes_ if label_encoder is not None else None

        # Save as CSV with labels if available
        if class_labels is not None:
            cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
            csv_path = os.path.join(cm_dir, f"{attribute}_cm.csv")
            cm_df.to_csv(csv_path)
            logger.info(f"Saved confusion matrix CSV for {attribute} to {csv_path}")
        else:
            logger.warning(f"No labels available for {attribute}, skipping CSV export")

        # Generate and save plot with publication-quality styling
        # Set publication-quality font sizes
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.labelsize": 16,
                "axes.titlesize": 18,
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
                "legend.fontsize": 14,
                "figure.titlesize": 18,
            }
        )

        plt.figure(figsize=(max(10, len(cm) * 0.8), max(8, len(cm) * 0.6)))

        # Use labels if available, otherwise use numeric indices
        display_labels = (
            class_labels if class_labels is not None else np.arange(len(cm))
        )

        # Create heatmap with annotations and larger font
        annot_fontsize = max(10, 14 - len(cm) // 5)  # Scale down for large matrices
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar_kws={"label": "Count", "shrink": 0.8},
            annot_kws={"fontsize": annot_fontsize},
            linewidths=0.5,
            linecolor="gray",
        )

        plt.title(f"Confusion Matrix: Metadata Prediction - {attribute}", pad=20)
        plt.ylabel("True Label", fontsize=16, labelpad=10)
        plt.xlabel("Predicted Label", fontsize=16, labelpad=10)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot as PDF with publication-quality DPI
        pdf_path = os.path.join(cm_dir, f"{attribute}_cm.pdf")
        plt.savefig(
            pdf_path, dpi=300, bbox_inches="tight", format="pdf", transparent=True
        )
        logger.info(f"Saved confusion matrix PDF for {attribute} to {pdf_path}")

        # Save plot as SVG (vector format for publications)
        svg_path = os.path.join(cm_dir, f"{attribute}_cm.svg")
        plt.savefig(svg_path, bbox_inches="tight", format="svg", transparent=True)
        logger.info(f"Saved confusion matrix SVG for {attribute} to {svg_path}")

        plt.close()

        # Save label encoder mapping
        if label_encoder is not None:
            labels_path = os.path.join(cm_dir, f"{attribute}_labels.json")
            label_mapping = {
                int(i): str(label) for i, label in enumerate(label_encoder.classes_)
            }
            with open(labels_path, "w") as f:
                json.dump(label_mapping, f, indent=2)
            logger.info(f"Saved label mapping for {attribute} to {labels_path}")


def main():
    """Main execution function."""
    args = parse_args()
    setup_logger(args.log_level)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models
    models = load_models(args, device)
    model_names = [m.model_path for m in models]
    logger.info(f"Loaded {len(models)} model(s)")

    # Generate run name
    run_name = generate_run_name(args, model_names)

    # Append "_random_baseline" suffix if using random baseline
    if args.random_baseline:
        run_name = f"{run_name}_random_baseline"
        logger.info(f"Run name (with random baseline): {run_name}")
    else:
        logger.info(f"Run name: {run_name}")

    # Setup directories
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    analysis_dir = os.path.join(out_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Initialize wandb for tracking all metrics (not just projector training)
    # This allows comparison across all runs including non-trainable projections
    wandb_run = None
    if args.projector_wandb or (
        args.downstream_eval or args.separability_eval or args.predict_metadata
    ):
        project_name = args.projector_wandb_project or "SkinMap"
        wandb_run = wandb.init(project=project_name, name=run_name, config=vars(args))
        logger.info(f"Initialized wandb run: {wandb_run.name}")

    # Initialize cache manager
    cache = EmbeddingCache(args.output_dir)

    # Load data
    logger.info(f"Loading data from {args.data_csv}")
    df = normalize_multilabel_columns(pd.read_csv(args.data_csv))
    metadata_attributes = parse_metadata_attributes(args.metadata_attributes)
    if args.vis_samples:
        df = df.sample(n=args.vis_samples, random_state=args.seed).reset_index(
            drop=True
        )
    logger.info(f"Loaded {len(df)} samples")

    # Create cache configuration
    expected_fusion = determine_fusion_method(args, models)
    cache_config = CacheConfig(
        max_whitening_dim=args.max_whitening_dim,
        skip_whitening=args.skip_whitening,
        projector_dim=args.projector_dim if args.use_trained_projector else None,
        projector_type=args.projector_type if args.use_trained_projector else None,
        svd_components=args.svd_components,
        fusion_method=expected_fusion,
    )

    # Check for metadata-only updates first (before full embedding extraction)
    metadata_reuse_result = check_metadata_update(
        df, cache, run_name, cache_config, args
    )

    if metadata_reuse_result is not None:
        # User chose to reuse embeddings with updated metadata
        (
            image_embeddings,
            text_embeddings,
            dataset_labels,
            filtered_df,
            corrupted_indices,
            fusion_method_used,
        ) = metadata_reuse_result
        artifacts = {}  # No new artifacts when reusing cache
    else:
        # Extract or load embeddings normally
        logger.info("=" * 60)
        logger.info("EXTRACTING/LOADING EMBEDDINGS")
        logger.info("=" * 60)

        (
            image_embeddings,
            text_embeddings,
            dataset_labels,
            filtered_df,
            corrupted_indices,
            artifacts,
            fusion_method_used,
            wandb_run,
        ) = extract_or_load_embeddings(
            models, df, args, device, cache, cache_config, run_name, wandb_run
        )

    logger.info(f"Final embeddings: {image_embeddings.shape}")
    logger.info(f"Final dataframe: {len(filtered_df)} samples")

    # Static visualization
    if args.use_static:
        logger.info("=" * 60)
        logger.info("GENERATING STATIC VISUALIZATION")
        logger.info("=" * 60)
        from src.train_clip import plot_embeddings

        plot_embeddings(
            args=args,
            model=models[0].model if models else None,
            processor=models[0].processor if models else None,
            device=device,
            run_name=run_name,
        )

    # Downstream evaluation on external datasets
    learned_metric_L = None
    label_col = args.metric_label_col
    if (
        label_col
        and label_col not in filtered_df.columns
        and "condition" in filtered_df.columns
    ):
        logger.info(
            "Metric label column '%s' not found, falling back to 'condition'", label_col
        )
        label_col = "condition"
    if args.downstream_eval and label_col and label_col in filtered_df.columns:
        logger.info(
            f"Computing learned metric transformation from '{label_col}' labels...",
        )
        mask = filtered_df[label_col].notna()
        if mask.sum() > 100:
            learned_metric_L = compute_learned_metric_transformation(
                embeddings=image_embeddings[mask],
                labels=filtered_df.loc[mask, label_col].values,
                random_state=args.seed,
            )

            # Save learned metric transformation
            metric_path = os.path.join(out_dir, "learned_metric_L.npy")
            np.save(metric_path, learned_metric_L)
            logger.info(f"Saved learned metric transformation to {metric_path}")
        else:
            logger.warning(
                f"Insufficient labeled samples for metric learning: {mask.sum()} (column: {label_col})"
            )

    if args.downstream_eval:
        logger.info("=" * 60)
        logger.info("RUNNING DOWNSTREAM EVALUATION ON EXTERNAL DATASETS")
        logger.info("=" * 60)

        projector_model_for_eval = None
        projector_whitening_stats = None
        if args.use_trained_projector:
            projector_model_for_eval = artifacts.get("projector_model")
            projector_whitening_stats = artifacts.get("whitening_stats")
            projector_config = None

            if projector_model_for_eval is None or projector_whitening_stats is None:
                (
                    projector_model_for_eval,
                    projector_whitening_stats,
                    projector_config,
                ) = load_projector_artifacts(
                    os.path.join(out_dir, "embeddings"), device
                )

            if (
                projector_model_for_eval is not None
                and projector_whitening_stats is None
            ):
                logger.warning(
                    "Projector whitening stats missing; skipping projector-based evaluation"
                )
                projector_model_for_eval = None

            if projector_config and projector_config.get("model_paths"):
                current_model_paths = [m.model_path for m in models]
                if projector_config.get("model_paths") != current_model_paths:
                    logger.warning(
                        "Projector model paths do not match current models; "
                        "skipping projector-based evaluation"
                    )
                    projector_model_for_eval = None
                    projector_whitening_stats = None

        eval_model = models[0].model
        eval_processor = models[0].processor
        is_ssl = models[0].model_type == "ssl"

        # Prepare models_and_processors if multiple models
        models_and_processors = None
        if len(models) > 1:
            models_and_processors = [
                (m.model, m.processor, m.model_type) for m in models
            ]

        # PAD_UFES_20 evaluation
        try:
            logger.info("Evaluating on PAD_UFES_20...")
            dataset = get_dataset(
                dataset_name=DatasetName.PAD_UFES_20,
                dataset_path=Path("/data/"),
                return_loader=False,
            )
            bootstrap_n_iter = (
                args.bootstrap_n_iterations if args.bootstrap_uncertainty else 0
            )
            pad_results, pad_confusion_matrices = evaluate_image_dataset(
                eval_model,
                eval_processor,
                device,
                dataset,
                classification_cols=[
                    "itch",
                    "grew",
                    "hurt",
                    "changed",
                    "bleed",
                    "elevation",
                    "gender",
                    "region",
                    "fitzpatrick",
                    "biopsed",
                    "diagnostic_name",
                ],
                regression_cols=["age", "diameter_1", "diameter_2"],
                emb_path=os.path.join(out_dir, "embeddings", "PAD_UFES_20.npz"),
                models_and_processors=models_and_processors,
                svd_components=args.svd_components,
                is_ssl_model=is_ssl,
                learned_metric_L=learned_metric_L,
                projector_model=projector_model_for_eval,
                projector_whitening_stats=projector_whitening_stats,
                projector_skip_whitening=args.skip_whitening,
                return_results=True,
                use_random_baseline=args.random_baseline,
                bootstrap_n_iterations=bootstrap_n_iter,
            )

            # Save PAD_UFES_20 results
            if pad_results:
                _save_downstream_results(
                    pad_results,
                    os.path.join(out_dir, "downstream_PAD_UFES_20.csv"),
                    "PAD_UFES_20",
                )

                # Log to wandb
                if wandb_run is not None:
                    _log_downstream_to_wandb(pad_results, "PAD_UFES_20")

            # Save confusion matrices
            if pad_confusion_matrices:
                _save_confusion_matrices(
                    pad_confusion_matrices,
                    out_dir,
                    "PAD_UFES_20",
                )
        except Exception as e:
            logger.warning(f"PAD_UFES_20 evaluation failed: {e}")

        # DDI evaluation
        try:
            logger.info("Evaluating on DDI...")
            dataset = get_dataset(
                dataset_name=DatasetName.DDI,
                dataset_path=Path("/data/"),
                return_loader=False,
            )
            bootstrap_n_iter = (
                args.bootstrap_n_iterations if args.bootstrap_uncertainty else 0
            )
            ddi_results, ddi_confusion_matrices = evaluate_image_dataset(
                eval_model,
                eval_processor,
                device,
                dataset,
                classification_cols=["fitzpatrick", "malignant", "disease"],
                regression_cols=None,
                emb_path=os.path.join(out_dir, "embeddings", "DDI.npz"),
                models_and_processors=models_and_processors,
                svd_components=args.svd_components,
                is_ssl_model=is_ssl,
                learned_metric_L=learned_metric_L,
                projector_model=projector_model_for_eval,
                projector_whitening_stats=projector_whitening_stats,
                projector_skip_whitening=args.skip_whitening,
                return_results=True,
                use_random_baseline=args.random_baseline,
                bootstrap_n_iterations=bootstrap_n_iter,
            )

            # Save DDI results
            if ddi_results:
                _save_downstream_results(
                    ddi_results, os.path.join(out_dir, "downstream_DDI.csv"), "DDI"
                )

                # Log to wandb
                if wandb_run is not None:
                    _log_downstream_to_wandb(ddi_results, "DDI")

            # Save confusion matrices
            if ddi_confusion_matrices:
                _save_confusion_matrices(
                    ddi_confusion_matrices,
                    out_dir,
                    "DDI",
                )
        except Exception as e:
            logger.warning(f"DDI evaluation failed: {e}")

    # Separability evaluation
    if args.separability_eval and "dataset_desc" in filtered_df.columns:
        sep_path = os.path.join(out_dir, "separability_results.csv")
        cached_sep = load_cached_separability_results(sep_path)

        if cached_sep is not None:
            logger.info(cached_sep.to_string(index=False))
        else:
            logger.info("=" * 60)
            logger.info("RUNNING SEPARABILITY EVALUATION")
            logger.info("=" * 60)

            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                balanced_accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                image_embeddings,
                filtered_df["dataset_desc"].values,
                test_size=0.2,
                random_state=args.seed,
                stratify=filtered_df["dataset_desc"].values,
            )

            # Linear probe
            logger.info("Training linear probe for dataset separability...")
            linear_clf = LogisticRegression(max_iter=5_000, random_state=args.seed)
            linear_clf.fit(X_train, y_train)
            y_pred = linear_clf.predict(X_test)

            results = {
                "classifier": "Linear probe",
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }

            logger.info(f"Separability results: {results}")
            results_df = pd.DataFrame([results])
            results_df.to_csv(sep_path, index=False)
            logger.info(f"Saved separability results to {sep_path}")

            # Log to wandb
            if wandb_run is not None:
                wandb.log(
                    {
                        f"separability/{k}": v
                        for k, v in results.items()
                        if k != "classifier"
                    }
                )
                logger.info("Logged separability results to wandb")

    # Predict missing metadata (before atlas so predictions appear in visualization)
    if args.predict_metadata and metadata_attributes:
        logger.info("=" * 60)
        if args.random_baseline:
            logger.info("PREDICTING MISSING METADATA (RANDOM BASELINE)")
        else:
            logger.info("PREDICTING MISSING METADATA")
        logger.info("=" * 60)
        metrics_path = os.path.join(analysis_dir, "metadata_prediction_metrics.csv")
        cached_df_path = os.path.join(out_dir, "embeddings", "dataframe.csv")

        # Create grouped Fitzpatrick column if Fitzpatrick is in attributes
        if "fitzpatrick" in metadata_attributes:
            logger.info("Creating grouped Fitzpatrick column (1-2, 3-4, 5-6)")

            def group_fitzpatrick(value):
                """Group Fitzpatrick skin type into 3 categories."""
                if pd.isna(value):
                    return np.nan
                try:
                    fitz_int = int(float(value))
                    if fitz_int in [1, 2]:
                        return "1-2"
                    elif fitz_int in [3, 4]:
                        return "3-4"
                    elif fitz_int in [5, 6]:
                        return "5-6"
                    else:
                        return np.nan
                except (ValueError, TypeError):
                    return np.nan

            filtered_df["fitzpatrick_grouped"] = filtered_df["fitzpatrick"].apply(
                group_fitzpatrick
            )

            # Add grouped version to attributes list (after fitzpatrick)
            if "fitzpatrick_grouped" not in metadata_attributes:
                fitz_idx = metadata_attributes.index("fitzpatrick")
                metadata_attributes.insert(fitz_idx + 1, "fitzpatrick_grouped")
                logger.info("Added 'fitzpatrick_grouped' to prediction attributes")

        # Skip cache reuse for random baseline (always recompute for reproducibility)
        if args.random_baseline:
            reused_predictions = False
        else:
            filtered_df, reused_predictions = reuse_cached_metadata_predictions(
                filtered_df,
                metadata_attributes,
                cached_df_path,
            )

        if reused_predictions:
            logger.info(f"Reused cached metadata predictions from {cached_df_path}")
            if os.path.exists(metrics_path):
                try:
                    metrics_df_cached = pd.read_csv(metrics_path)
                    if not metrics_df_cached.empty:
                        logger.info(
                            "Cached prediction metrics:\n"
                            f"{metrics_df_cached.to_string(index=False)}"
                        )
                except Exception as exc:
                    logger.warning(f"Failed to read cached metrics: {exc}")
        else:
            if args.random_baseline:
                # Use random baseline predictor
                filtered_df, metrics_df, metadata_confusion_matrices = (
                    predict_random_baseline_metadata(
                        filtered_df,
                        metadata_attributes,
                        test_size=args.prediction_test_size,
                        random_state=args.seed,
                        output_dir=out_dir,
                    )
                )
            else:
                # Use XGBoost predictor
                bootstrap_n_iter = (
                    args.bootstrap_n_iterations if args.bootstrap_uncertainty else 0
                )
                filtered_df, metrics_df, metadata_confusion_matrices = (
                    predict_missing_metadata(
                        image_embeddings,
                        filtered_df,
                        metadata_attributes,
                        test_size=args.prediction_test_size,
                        random_state=args.seed,
                        output_dir=out_dir,
                        bootstrap_n_iterations=bootstrap_n_iter,
                    )
                )

            if not metrics_df.empty:
                metrics_df.to_csv(metrics_path, index=False)
                logger.info(f"Saved prediction metrics to {metrics_path}")

                # Log to wandb
                if wandb_run is not None:
                    for _, row in metrics_df.iterrows():
                        attribute = row.get("attribute", "unknown")
                        metric_dict = row.to_dict()
                        wandb_metrics = {
                            f"metadata_prediction/{attribute}/{k}": v
                            for k, v in metric_dict.items()
                            if k != "attribute"
                        }
                        wandb.log(wandb_metrics)
                    logger.info("Logged metadata prediction metrics to wandb")

            # Save confusion matrices for metadata prediction
            if metadata_confusion_matrices:
                _save_metadata_confusion_matrices(metadata_confusion_matrices, out_dir)

    # Balancing comparison evaluation (ADDITIONAL)
    if args.balancing_comparison:
        logger.info("=" * 60)
        logger.info("BALANCING COMPARISON EVALUATION")
        logger.info("=" * 60)

        # Metadata prediction
        if args.predict_metadata and metadata_attributes:
            results_df = evaluate_metadata_with_balancing_comparison(
                image_embeddings,
                filtered_df,
                metadata_attributes,
                test_size=args.prediction_test_size,
                random_state=args.seed,
                output_dir=out_dir,
            )
            if wandb_run and not results_df.empty:
                for _, row in results_df.iterrows():
                    wandb.log(
                        {
                            f"balance_meta/{row['attribute']}/{row['strategy']}/{k}": v
                            for k, v in row.items()
                            if k not in ["attribute", "strategy"]
                        }
                    )

        # Downstream tasks
        if args.downstream_eval:
            for ds_name, ds_enum, cols in [
                (
                    "PAD_UFES_20",
                    DatasetName.PAD_UFES_20,
                    [
                        "itch",
                        "grew",
                        "hurt",
                        "changed",
                        "bleed",
                        "elevation",
                        "gender",
                        "region",
                        "fitzpatrick",
                        "biopsed",
                        "diagnostic_name",
                    ],
                ),
                ("DDI", DatasetName.DDI, ["fitzpatrick", "malignant", "disease"]),
            ]:
                try:
                    emb_path = os.path.join(out_dir, "embeddings", f"{ds_name}.npz")
                    if not os.path.exists(emb_path):
                        continue

                    dataset = get_dataset(
                        dataset_name=ds_enum,
                        dataset_path=Path("/data/"),
                        return_loader=False,
                    )
                    with np.load(emb_path, allow_pickle=True) as data:
                        embs = data["image_embeddings"]

                    results_df = evaluate_downstream_with_balancing_comparison(
                        embs,
                        dataset,
                        cols,
                        test_size=0.2,
                        random_state=args.seed,
                        output_dir=out_dir,
                        dataset_name=ds_name,
                    )
                    if wandb_run and not results_df.empty:
                        for _, row in results_df.iterrows():
                            wandb.log(
                                {
                                    f"balance_down/{ds_name}/{row['task']}/{row['strategy']}/{k}": v
                                    for k, v in row.items()
                                    if k not in ["dataset", "task", "strategy"]
                                }
                            )
                except Exception as e:
                    logger.warning(f"{ds_name} balancing comparison failed: {e}")

    # Save all outputs and artifacts BEFORE atlas generation so the atlas config
    # picks up freshly written SVD/projector files.
    logger.info("=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)

    save_outputs(
        filtered_df,
        image_embeddings,
        text_embeddings,
        dataset_labels,
        corrupted_indices,
        fusion_method_used,
        args,
        out_dir,
        run_name,
        artifacts,
        model_names,
    )

    # Generate or reuse the atlas parquet (runs after the embedding artifacts are
    # persisted).
    if args.use_atlas:
        logger.info("=" * 60)
        logger.info("EMBEDDING ATLAS")
        logger.info("=" * 60)
        atlas_path = os.path.join(out_dir, "atlas_input.parquet")
        atlas_exists = os.path.exists(atlas_path)
        atlas_df = generate_atlas(
            df=filtered_df,
            image_embeddings=image_embeddings,
            output_dir=out_dir,
            args=args,
            store_vectors=args.atlas_store_vectors,
            force_recompute=args.use_atlas or not atlas_exists,
            reuse_existing=True,
        )
        logger.info(f"Atlas ready with {len(atlas_df)} samples (path: {atlas_path})")
    else:
        logger.info("Skipping atlas generation (use --use_atlas to enable)")

    # Finish wandb run after all logging is complete
    if wandb_run is not None:
        wandb.finish()
        logger.info("Finished wandb run")

    logger.info("=" * 60)
    logger.info("SkinMap creation COMPLETE!")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Samples: {len(filtered_df)}")
    logger.info(f"Embedding dim: {image_embeddings.shape[1]}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
