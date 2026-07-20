"""Simple embedding fusion with concatenation and SVD."""

import gc
import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.decomposition import TruncatedSVD

_CONCAT_CHUNK_ROWS = 8_192

from ..cache.manager import EmbeddingCache
from ..models.loaders import ModelInfo, normalize_model_tuple
from .extractors import extract_clip_embeddings, extract_ssl_embeddings


def combine_embeddings_simple(
    models: List[ModelInfo],
    df: pd.DataFrame,
    device: torch.device,
    cache: EmbeddingCache,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    dataset_col: str = "dataset_desc",
    num_workers: int = 4,
    svd_components: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    List[int],
    Optional[TruncatedSVD],
    Optional[TruncatedSVD],
    pd.DataFrame,
]:
    """Combine embeddings from multiple models using simple concatenation + optional SVD.

    Args:
        models: List of ModelInfo objects
        df: DataFrame with image paths and descriptions
        device: torch device
        cache: EmbeddingCache instance
        batch_size: Batch size for extraction
        max_samples: Maximum samples to process
        dataset_col: Column for dataset labels
        num_workers: DataLoader workers
        svd_components: Optional number of SVD components for dimensionality reduction

    Returns:
        Tuple of (
            image_embeddings,
            text_embeddings (or None for SSL-only),
            dataset_labels,
            corrupted_indices,
            svd_image_model,
            svd_text_model,
            filtered_df with corrupted/failed samples removed
        )
    """
    logger.info("=" * 60)
    logger.info(
        f"Simple Fusion Pipeline ({'concat+SVD' if svd_components else 'concat'})"
    )
    logger.info("=" * 60)

    temp_records = []
    cached_results: List[
        Optional[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, pd.DataFrame]]
    ] = [None] * len(models)
    dataset_labels = None
    corrupted_images = []
    filtered_df = df.copy()
    num_samples: Optional[int] = None

    # Check cache compatibility
    model_paths = [m.model_path for m in models]
    are_compatible, cached_dfs = cache.check_individual_compatibility(model_paths, df)

    if are_compatible:
        logger.info("Loading from compatible caches")
        for idx, model_info in enumerate(models):
            result = cache.load_single_model(model_info.model_path)
            if result is None:
                raise RuntimeError(f"Cache load failed: {model_info.model_path}")

            cached_results[idx] = result
            image_embs, text_embs, labels, _ = result

            if dataset_labels is None:
                dataset_labels = labels

        filtered_df = cached_dfs[0].copy()

    else:
        logger.info("Computing embeddings (cache not compatible)")

    with tempfile.TemporaryDirectory(prefix="skinmap_fusion_") as fusion_tmp_dir:
        for i, model_info in enumerate(models):
            if are_compatible:
                cached_entry = cached_results[i]
                if cached_entry is None:
                    raise RuntimeError(
                        f"Cache unexpectedly missing for compatible run: {model_info.model_path}"
                    )
                image_embs, text_embs, labels, cached_df = cached_entry

                if i == 0:
                    filtered_df = cached_df.copy()
                    dataset_labels = labels
            else:
                # Try loading from cache
                result = cache.load_single_model(model_info.model_path)

                if result is not None:
                    image_embs, text_embs, labels, cached_df = result

                    if i == 0:
                        filtered_df = cached_df.copy()
                        dataset_labels = labels
                        corrupted_images = []
                else:
                    # Extract embeddings
                    logger.info(
                        f"Extracting for model {i+1}/{len(models)} ({model_info.model_type})"
                    )

                    model, processor, model_type = normalize_model_tuple(model_info)

                    if model_type == "ssl":
                        image_embs, text_embs, labels, corrupted, df_clean = (
                            extract_ssl_embeddings(
                                model,
                                filtered_df if i > 0 else df,
                                device,
                                batch_size,
                                max_samples,
                                dataset_col,
                                num_workers,
                            )
                        )
                    else:
                        image_embs, text_embs, labels, corrupted, df_clean = (
                            extract_clip_embeddings(
                                model,
                                processor,
                                filtered_df if i > 0 else df,
                                device,
                                batch_size,
                                max_samples,
                                dataset_col,
                                num_workers,
                            )
                        )

                    if i == 0:
                        dataset_labels = labels
                        corrupted_images = corrupted
                        filtered_df = df_clean  # Use the cleaned dataframe

                    # Save to cache
                    cache.save_single_model(
                        model_info.model_path, image_embs, text_embs, labels, df_clean
                    )

            image_embs = np.ascontiguousarray(image_embs, dtype=np.float32)
            text_embs = (
                None
                if text_embs is None
                else np.ascontiguousarray(text_embs, dtype=np.float32)
            )

            if num_samples is None:
                num_samples = image_embs.shape[0]
            elif image_embs.shape[0] != num_samples:
                raise RuntimeError(
                    f"DIMENSION MISMATCH in fusion pipeline:\n"
                    f"Model {i+1} has {image_embs.shape[0]} samples\n"
                    f"Expected: {num_samples} samples (from model 1)\n"
                    f"This indicates corrupted caches or extraction failure.\n"
                    f"Delete all model caches and re-run."
                )

            image_path = os.path.join(fusion_tmp_dir, f"image_{i}.npy")
            np.save(image_path, image_embs, allow_pickle=False)

            text_path = None
            if text_embs is not None:
                text_path = os.path.join(fusion_tmp_dir, f"text_{i}.npy")
                np.save(text_path, text_embs, allow_pickle=False)

            temp_records.append(
                {
                    "image_path": image_path,
                    "text_path": text_path,
                    "image_dim": image_embs.shape[1],
                    "text_dim": text_embs.shape[1] if text_embs is not None else 0,
                    "image_dtype": image_embs.dtype,
                    "text_dtype": text_embs.dtype if text_embs is not None else None,
                }
            )

            del image_embs
            del text_embs
            gc.collect()

        if num_samples is None:
            raise RuntimeError("No embeddings were generated for fusion")

        image_embeddings = _assemble_emb_matrix(
            temp_records,
            key="image",
            num_samples=num_samples,
        )

        text_records = [
            record for record in temp_records if record["text_path"] is not None
        ]
        text_embeddings = (
            _assemble_emb_matrix(text_records, key="text", num_samples=num_samples)
            if text_records
            else None
        )

    logger.info(
        f"Combined dims: image={image_embeddings.shape[1]}, "
        f"text={'None (SSL-only models)' if text_embeddings is None else text_embeddings.shape[1]}"
    )

    svd_image_model = None
    svd_text_model = None

    # Apply SVD if requested
    if svd_components is not None:
        logger.info(f"Applying SVD with {svd_components} components")

        svd_image = TruncatedSVD(n_components=svd_components, random_state=42)
        image_embeddings = svd_image.fit_transform(image_embeddings)
        svd_image_model = svd_image
        gc.collect()

        if text_embeddings is not None:
            svd_text = TruncatedSVD(n_components=svd_components, random_state=42)
            text_embeddings = svd_text.fit_transform(text_embeddings)
            svd_text_model = svd_text
            gc.collect()

            logger.info(
                f"Reduced dims: image={image_embeddings.shape[1]}, text={text_embeddings.shape[1]}"
            )
        else:
            logger.info(
                f"Reduced dims: image={image_embeddings.shape[1]}, text=None (SSL-only)"
            )

    logger.info("=" * 60)
    logger.info("Simple fusion complete!")
    logger.info("=" * 60)

    return (
        image_embeddings,
        text_embeddings,
        dataset_labels,
        corrupted_images,
        svd_image_model,
        svd_text_model,
        filtered_df.reset_index(drop=True),
    )


def _assemble_emb_matrix(
    records: List[dict],
    key: str,
    num_samples: int,
    chunk_rows: int = _CONCAT_CHUNK_ROWS,
) -> np.ndarray:
    """Stream embeddings from disk-backed chunks into a single matrix."""
    path_key = f"{key}_path"
    dim_key = f"{key}_dim"
    dtype_key = f"{key}_dtype"

    total_dim = sum(record[dim_key] for record in records if record[dim_key] > 0)

    dtype = next(
        (record[dtype_key] for record in records if record[dtype_key] is not None),
        np.float32,
    )
    if total_dim == 0:
        return np.empty((num_samples, 0), dtype=dtype)

    combined = np.empty((num_samples, total_dim), dtype=dtype)

    offset = 0
    for record in records:
        dim = record[dim_key]
        path = record[path_key]

        if not path or dim == 0:
            continue

        data = np.load(path, mmap_mode="r")
        for start in range(0, num_samples, chunk_rows):
            end = min(start + chunk_rows, num_samples)
            combined[start:end, offset : offset + dim] = data[start:end]

        offset += dim
        del data
        gc.collect()

    return combined
