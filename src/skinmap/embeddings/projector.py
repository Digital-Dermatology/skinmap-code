"""Trained projector pipeline for embedding fusion.

This module implements the learnable projection approach with proper handling of:
1. Pre-whitening embeddings before storing to ensure dimension consistency
2. Cache validation with whitening configuration
3. Clear separation of whitening and projection steps
"""

import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from src.embedding_fusion import (
    BuildSpec,
    DomainBalancedBatchSampler,
    PrecomputedDataset,
    Sample,
    WhitenSpec,
    apply_whiten,
    build_model,
    clip_loss,
    fit_pca,
    fit_whitener,
    l2_normalize,
    make_collate,
    train_one_epoch,
)

from ..cache.manager import EmbeddingCache
from ..loss.clip_loss import CLIPLoss
from ..models.loaders import ModelInfo, normalize_model_tuple
from ..utils.worker_utils import resolve_num_workers
from .extractors import extract_clip_embeddings, extract_ssl_embeddings


def _normalize_teacher_blocks(z_cat: np.ndarray, dims: List[int]) -> np.ndarray:
    """L2 normalize each teacher block in concatenated embeddings."""
    start = 0
    for dim in dims:
        block = z_cat[:, start : start + dim]
        z_cat[:, start : start + dim] = l2_normalize(block, axis=1)
        start += dim
    return z_cat


def _try_load_trained_projector(
    projector_model,
    cache: EmbeddingCache,
    whitening_stats: Dict,
    projector_dim: int,
    projector_type: str,
    teacher_names: List[str],
) -> bool:
    """Try to load a trained projector from cache if compatible.

    Args:
        projector_model: Newly initialized projector model to load weights into
        cache: EmbeddingCache instance
        whitening_stats: Current whitening statistics
        projector_dim: Expected projector output dimension
        projector_type: Expected projector type ('linear' or 'mlp')
        teacher_names: List of teacher model names/paths

    Returns:
        True if projector was successfully loaded from cache, False otherwise
    """
    import json
    import os
    from pathlib import Path

    # Construct cache path based on teacher names (same logic as run_name generation)
    # This matches the path where create_skinmap.py saves the projector
    if len(teacher_names) == 1:
        # Single model case
        model_name = (
            teacher_names[0].replace("assets/", "").replace("/", "_").replace("\\", "_")
        )
        cache_run_name = model_name
    else:
        # Multi-model case
        model_parts = []
        for name in teacher_names:
            if "/" in name:
                if name.startswith("/") or name.startswith("."):
                    model_parts.append("_".join(Path(name).parts[-2:]))
                else:
                    cleaned = name.replace("assets/", "").replace("/", "_")
                    model_parts.append(cleaned)
            else:
                model_parts.append(name)

        cache_run_name = "combined_" + "_".join(model_parts[:3])
        if len(teacher_names) > 3:
            cache_run_name += f"_and_{len(teacher_names)-3}_more"
        cache_run_name += (
            "-trained_projector"  # Matches the suffix from generate_run_name
        )

    projector_cache_dir = os.path.join(cache.output_dir, cache_run_name, "embeddings")
    projector_weights_path = os.path.join(projector_cache_dir, "projector_model.pth")
    projector_config_path = os.path.join(projector_cache_dir, "projector_config.json")

    # Check if files exist
    if not os.path.exists(projector_weights_path):
        logger.debug(f"No cached projector weights found at {projector_weights_path}")
        return False

    if not os.path.exists(projector_config_path):
        logger.debug(f"No cached projector config found at {projector_config_path}")
        return False

    # Load and verify config compatibility
    try:
        with open(projector_config_path, "r") as f:
            cached_config = json.load(f)

        # Check critical parameters match
        if cached_config.get("projector_type") != projector_type:
            logger.debug(
                f"Projector type mismatch: cached={cached_config.get('projector_type')}, "
                f"current={projector_type}"
            )
            return False

        if cached_config.get("projector_dim") != projector_dim:
            logger.debug(
                f"Projector dim mismatch: cached={cached_config.get('projector_dim')}, "
                f"current={projector_dim}"
            )
            return False

        # Check whitening dimensions match
        cached_dims = cached_config.get("teacher_dims", [])
        current_dims = whitening_stats.get("dims", [])
        if cached_dims != current_dims:
            logger.debug(
                f"Teacher dims mismatch: cached={cached_dims}, current={current_dims}"
            )
            return False

        # Load weights into model
        # Get device from model parameters (models don't have .device attribute)
        device = next(projector_model.parameters()).device
        state_dict = torch.load(projector_weights_path, map_location=device)
        projector_model.load_state_dict(state_dict)

        logger.info(f"✓ Loaded trained projector from cache: {projector_weights_path}")
        logger.info(f"  Architecture: {projector_type}, Output dim: {projector_dim}")
        logger.info(f"  Teacher dims: {current_dims}")

        return True

    except Exception as e:
        logger.warning(f"Failed to load cached projector: {e}")
        return False


def validate_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    structure_loss=None,
    k_values: List[int] = [1, 5, 10],
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """Validate model performance with retrieval metrics using chunked processing.

    Computes metrics in chunks to avoid creating massive N×N similarity matrices
    that cause OOM errors on large validation sets.

    Args:
        model: Projector model
        loader: Validation DataLoader
        device: Device to run on
        structure_loss: Optional structure loss for loss computation
        k_values: List of k values for recall@k metrics
        chunk_size: Process similarities in chunks of this size to save memory

    Returns:
        Dictionary of validation metrics
    """
    model.eval()

    all_zi = []
    all_zt = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            z_cat = batch["z_cat"].to(device, non_blocking=True)
            t_vec = batch["t_vec"].to(device, non_blocking=True)

            zi, zt, ls = model(z_cat, t_vec)

            # Compute loss
            if structure_loss is not None:
                loss_dict = structure_loss(
                    zi, zt, z_cat, t_vec, image_embeddings_aligned_alt=None
                )
                loss = loss_dict["overall_loss"]
            else:
                loss = clip_loss(zi, zt, ls)

            total_loss += float(loss.item()) * z_cat.size(0)
            total_samples += z_cat.size(0)

            # Move to CPU and convert to float16 to save memory
            all_zi.append(zi.cpu().half())
            all_zt.append(zt.cpu().half())

            # Delete GPU tensors immediately
            del z_cat, t_vec, zi, zt

        # Clear GPU cache after validation loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all embeddings on CPU with float16 to save memory
    all_zi = torch.cat(all_zi, dim=0)  # Keep as float16
    all_zt = torch.cat(all_zt, dim=0)  # Keep as float16

    n_samples = all_zi.size(0)
    requested_k = list(k_values)
    # Clamp k values to available samples to avoid topk errors on tiny splits.
    max_k = min(max(requested_k), n_samples) if n_samples else 0
    k_values = [k for k in requested_k if 0 < k <= n_samples]
    if not k_values and max_k > 0:
        # Ensure at least one k for metric computation.
        k_values = [max_k]

    # Initialize accumulators for chunked processing
    i2t_correct_at_k = {k: 0 for k in k_values}
    t2i_correct_at_k = {k: 0 for k in k_values}
    i2t_ranks_list = []
    t2i_ranks_list = []

    # Process in chunks to avoid creating full N×N matrix
    logger.info(f"  Computing retrieval metrics in chunks of {chunk_size}...")

    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        chunk_indices = range(chunk_start, chunk_end)

        # Get chunk embeddings (convert to float32 only for this chunk)
        zi_chunk = all_zi[chunk_start:chunk_end].float()

        # Compute similarities for this chunk against all text embeddings
        # Only creates chunk_size × n_samples matrix instead of n_samples × n_samples
        sims_chunk = zi_chunk @ all_zt.float().T  # [chunk_size, n_samples]

        # Get top-k for this chunk
        topk_values, topk_indices = sims_chunk.topk(max_k, dim=1, largest=True)

        # Check which retrievals are correct for this chunk
        chunk_labels = torch.arange(chunk_start, chunk_end).unsqueeze(
            1
        )  # [chunk_size, 1]

        for k in k_values:
            # Check if correct text is in top-k
            correct_mask = (topk_indices[:, :k] == chunk_labels).any(dim=1)
            i2t_correct_at_k[k] += correct_mask.sum().item()

        # Find rank of correct answer for each sample in chunk
        for i, global_idx in enumerate(chunk_indices):
            correct_idx = global_idx
            rank = (topk_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                i2t_ranks_list.append(rank[0].item())
            else:
                # Correct answer not in top-k, need to search full similarities
                rank = (
                    (sims_chunk[i].argsort(descending=True) == correct_idx)
                    .nonzero(as_tuple=True)[0][0]
                    .item()
                )
                i2t_ranks_list.append(rank)

        # Clear chunk data
        del zi_chunk, sims_chunk, topk_values, topk_indices, chunk_labels

    # Now do text-to-image retrieval (query with text, retrieve images)
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        chunk_indices = range(chunk_start, chunk_end)

        # Get chunk text embeddings
        zt_chunk = all_zt[chunk_start:chunk_end].float()

        # Compute similarities (text queries vs all images)
        sims_chunk = zt_chunk @ all_zi.float().T  # [chunk_size, n_samples]

        # Get top-k for this chunk
        topk_values, topk_indices = sims_chunk.topk(max_k, dim=1, largest=True)

        # Check which retrievals are correct
        chunk_labels = torch.arange(chunk_start, chunk_end).unsqueeze(1)

        for k in k_values:
            correct_mask = (topk_indices[:, :k] == chunk_labels).any(dim=1)
            t2i_correct_at_k[k] += correct_mask.sum().item()

        # Find rank of correct answer
        for i, global_idx in enumerate(chunk_indices):
            correct_idx = global_idx
            rank = (topk_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                t2i_ranks_list.append(rank[0].item())
            else:
                rank = (
                    (sims_chunk[i].argsort(descending=True) == correct_idx)
                    .nonzero(as_tuple=True)[0][0]
                    .item()
                )
                t2i_ranks_list.append(rank)

        # Clear chunk data
        del zt_chunk, sims_chunk, topk_values, topk_indices, chunk_labels

    # Compute final metrics
    metrics = {"val_loss": total_loss / max(1, total_samples)}

    for k in requested_k:
        k_eval = k
        if k_eval not in i2t_correct_at_k:
            k_eval = max_k if k_values else 0
        if k_eval == 0:
            i2t_recall = t2i_recall = 0.0
        else:
            i2t_recall = (i2t_correct_at_k.get(k_eval, 0) / n_samples) * 100
            t2i_recall = (t2i_correct_at_k.get(k_eval, 0) / n_samples) * 100
        metrics[f"i2t_recall@{k}"] = i2t_recall
        metrics[f"t2i_recall@{k}"] = t2i_recall
        metrics[f"avg_recall@{k}"] = (i2t_recall + t2i_recall) / 2

    # Compute mean/median ranks
    i2t_ranks_tensor = torch.tensor(i2t_ranks_list, dtype=torch.float32)
    t2i_ranks_tensor = torch.tensor(t2i_ranks_list, dtype=torch.float32)

    metrics["i2t_mean_rank"] = float(i2t_ranks_tensor.mean()) + 1  # +1 for 1-indexed
    metrics["i2t_median_rank"] = float(i2t_ranks_tensor.median()) + 1
    metrics["t2i_mean_rank"] = float(t2i_ranks_tensor.mean()) + 1
    metrics["t2i_median_rank"] = float(t2i_ranks_tensor.median()) + 1

    # Explicitly delete large tensors to free memory
    del all_zi, all_zt
    del i2t_ranks_tensor, t2i_ranks_tensor
    gc.collect()

    return metrics


def _load_or_extract_per_model_embeddings(
    models: List[ModelInfo],
    df: pd.DataFrame,
    device: torch.device,
    cache: EmbeddingCache,
    batch_size: int,
    max_samples: Optional[int],
    dataset_col: str,
    num_workers: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, List[int], pd.DataFrame]:
    """Load or extract per-model embeddings with caching.

    Returns:
        Tuple of (image_embeddings_list, text_embeddings_list, dataset_labels,
                  corrupted_indices, filtered_df)
    """
    all_image_embeddings = []
    all_text_embeddings = []
    all_labels = None
    corrupted_images = []
    filtered_df = df.copy()

    # Check cache compatibility first
    model_paths = [m.model_path for m in models]
    are_compatible, cached_dfs = cache.check_individual_compatibility(model_paths, df)

    if are_compatible:
        logger.info("Loading embeddings from compatible caches")
        for i, model_info in enumerate(models):
            result = cache.load_single_model(model_info.model_path)
            if result is None:
                raise RuntimeError(f"Failed to load cache for {model_info.model_path}")

            image_embs, text_embs, labels, _ = result
            all_image_embeddings.append(image_embs)
            all_text_embeddings.append(text_embs)

            if all_labels is None:
                all_labels = labels

        filtered_df = cached_dfs[0].copy()
        logger.info(f"Loaded {len(models)} models from cache")

    else:
        logger.info("Computing embeddings (caches not compatible or missing)")

        for i, model_info in enumerate(models):
            # Try loading from individual cache
            result = cache.load_single_model(model_info.model_path)

            if result is not None:
                image_embs, text_embs, labels, cached_df = result
                logger.info(f"Loaded model {i+1} from cache")

                # Update df if this is the first model
                if i == 0:
                    filtered_df = cached_df.copy()
                    all_labels = labels
                    corrupted_images = []

            else:
                # Extract embeddings
                logger.info(
                    f"Computing embeddings for model {i+1}/{len(models)} "
                    f"({model_info.model_type})"
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
                    all_labels = labels
                    corrupted_images = corrupted
                    filtered_df = df_clean  # Use the cleaned dataframe

                # Save to cache
                cache.save_single_model(
                    model_info.model_path,
                    image_embs,
                    text_embs,
                    labels,
                    df_clean,
                )

            all_image_embeddings.append(image_embs)
            all_text_embeddings.append(text_embs)

    return (
        all_image_embeddings,
        all_text_embeddings,
        all_labels,
        corrupted_images,
        filtered_df,
    )


def extract_combined_embeddings_with_projector(
    models: List[ModelInfo],
    df: pd.DataFrame,
    device: torch.device,
    cache: EmbeddingCache,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    dataset_col: str = "dataset_desc",
    num_workers: int = 4,
    # Projector parameters
    projector_dim: int = 768,
    projector_type: str = "linear",
    n_epochs: int = 5,
    learning_rate: float = 1e-3,
    use_domain_balanced: bool = True,
    pca_init_samples: Optional[int] = 100_000,
    max_whitening_dim: int = 768,
    skip_whitening: bool = False,
    loss_type: str = "clip",
    hierarchy_lambda: float = 5.0,
    hierarchy_levels: int = 1,
    hierarchy_warmup: int = 500,
    hierarchy_weighting: str = "none",
    hierarchy_margin: float = 0.0,
    consistency_lambda: float = 0.0,
    log_to_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    validation_split: float = 0.1,
    validate_every_n_epochs: int = 1,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[int],
    object,
    Dict,
    pd.DataFrame,
    Optional[object],
]:
    """Extract embeddings using trained projector pipeline.

    This function properly handles dimension consistency by pre-whitening embeddings
    before storing in samples, preventing mismatches with whitening matrices.

    Args:
        models: List of ModelInfo objects
        df: DataFrame with image paths and descriptions
        device: torch device
        cache: EmbeddingCache instance
        batch_size: Batch size for training
        max_samples: Maximum samples to process
        dataset_col: Column for dataset labels
        num_workers: DataLoader workers
        projector_dim: Output dimension
        projector_type: 'linear' or 'mlp'
        n_epochs: Training epochs
        learning_rate: Learning rate
        use_domain_balanced: Use domain-balanced batching
        pca_init_samples: Samples for PCA initialization
        max_whitening_dim: Max dimension for whitening (truncation)
        skip_whitening: Skip whitening entirely
        validation_split: Fraction of data to use for validation (default 0.1)
        validate_every_n_epochs: Run validation every N epochs (default 1).
            Set to 0 to skip validation.

    Returns:
        Tuple of (
            image_embeddings,
            text_embeddings,
            dataset_labels,
            corrupted_indices,
            projector_model,
            whitening_stats,
            filtered_df with corrupted/failed samples removed,
            wandb_run for logging downstream metrics (or None if not started)
        )
    """
    logger.info("=" * 60)
    logger.info("Trained Projector Pipeline")
    logger.info("=" * 60)

    wandb_run = None

    # Step 1: Load or extract per-model embeddings
    logger.info("Step 1: Loading/extracting per-model embeddings")
    (
        all_image_embeddings,
        all_text_embeddings,
        dataset_labels,
        corrupted_images,
        filtered_df,
    ) = _load_or_extract_per_model_embeddings(
        models, df, device, cache, batch_size, max_samples, dataset_col, num_workers
    )

    teacher_names = [m.model_path for m in models]
    original_dims = [embs.shape[1] for embs in all_image_embeddings]

    logger.info(f"Loaded {len(all_image_embeddings)} models with dims: {original_dims}")

    # Identify CLIP models for text embeddings
    clip_indices = []
    clip_text_embeddings = []
    for i, model in enumerate(models):
        if model.model_type != "ssl":
            clip_indices.append(i)
            clip_text_embeddings.append(all_text_embeddings[i])
            logger.info(f"Including text from model {i+1} ({model.model_path})")

    if not clip_text_embeddings:
        raise ValueError("At least one CLIP model required for text encoding")

    # Step 2: Prepare whitening or normalization
    if skip_whitening:
        logger.info("Step 2: Skipping whitening (L2 normalize only)")
        whitening_stats = _prepare_normalized_embeddings(
            all_image_embeddings, clip_text_embeddings, original_dims, clip_indices
        )
    else:
        logger.info("Step 2: Fitting whitening transformations")
        whitening_stats = _prepare_whitened_embeddings(
            all_image_embeddings,
            clip_text_embeddings,
            original_dims,
            clip_indices,
            max_whitening_dim,
        )

    if log_to_wandb:
        try:
            wandb_config = {
                "projector_type": projector_type,
                "projector_dim": projector_dim,
                "num_teachers": len(teacher_names),
                "teacher_dims": whitening_stats["dims"],
                "text_dim": sum(whitening_stats["text_whitening"]["dims"]),
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "use_domain_balanced": use_domain_balanced,
                "skip_whitening": skip_whitening,
                "max_whitening_dim": max_whitening_dim,
                "pca_init_samples": pca_init_samples,
            }

            derived_run_name = (
                wandb_run_name
                or f"projector-{projector_type}-{projector_dim}-models{len(teacher_names)}"
            )
            project_name = wandb_project or "SkinMapProjector"

            if wandb.run is None:
                wandb_run = wandb.init(
                    project=project_name,
                    name=derived_run_name,
                    config=wandb_config,
                )
            else:
                # Reuse existing wandb run from parent scope
                wandb_run = wandb.run
                wandb_run.config.update(wandb_config, allow_val_change=True)
                logger.info("Reusing existing wandb run for projector training")

            wandb.log(
                {
                    "projector/num_samples": len(filtered_df),
                    "projector/num_clip_models": len(clip_indices),
                },
            )
        except Exception as exc:
            logger.warning(f"Failed to initialize wandb for projector logging: {exc}")
            wandb_run = None
            log_to_wandb = False

    # Apply whitening to get the actual embeddings we'll use
    # This ensures dimensions match what the collate function expects
    logger.info("Step 3: Applying whitening to create final embedding matrices")

    whitened_image_embeddings = []
    for i, embs in enumerate(all_image_embeddings):
        embs_normed = l2_normalize(embs, axis=1)
        if not skip_whitening:
            embs_whitened = apply_whiten(
                embs_normed, whitening_stats["mu"][i], whitening_stats["W"][i]
            )
            whitened_image_embeddings.append(embs_whitened)
        else:
            whitened_image_embeddings.append(embs_normed)
        logger.info(
            f"  Model {i+1}: {embs.shape[1]} → {whitened_image_embeddings[-1].shape[1]}"
        )

    # Concatenate whitened embeddings and normalize blocks
    z_cat = np.concatenate(whitened_image_embeddings, axis=1)
    z_cat = _normalize_teacher_blocks(z_cat, whitening_stats["dims"])
    # Note: Don't delete whitened_image_embeddings yet - needed for training samples
    logger.info(f"Concatenated image embeddings: {z_cat.shape}")

    # Process text embeddings similarly
    whitened_text_embeddings = []
    for i, text_embs in enumerate(clip_text_embeddings):
        text_embs_normed = l2_normalize(text_embs, axis=1)
        if not skip_whitening:
            text_embs_whitened = apply_whiten(
                text_embs_normed,
                whitening_stats["text_whitening"]["mu"][i],
                whitening_stats["text_whitening"]["W"][i],
            )
            whitened_text_embeddings.append(text_embs_whitened)
        else:
            whitened_text_embeddings.append(text_embs_normed)

    text_concat = np.concatenate(whitened_text_embeddings, axis=1)
    del whitened_text_embeddings
    del clip_text_embeddings
    gc.collect()
    logger.info(f"Concatenated text embeddings: {text_concat.shape}")

    # Step 4: Initialize projector with PCA
    logger.info(f"Step 4: Initializing projector with PCA (dim={projector_dim})")

    if pca_init_samples and len(z_cat) > pca_init_samples:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(z_cat), size=pca_init_samples, replace=False)
        z_cat_sample = z_cat[sample_indices]
        text_concat_sample = text_concat[sample_indices]
    else:
        z_cat_sample = z_cat
        text_concat_sample = text_concat

    pca_image = fit_pca(z_cat_sample, projector_dim)
    pca_text = fit_pca(text_concat_sample, projector_dim)
    logger.info(f"PCA fitted: image {pca_image.shape}, text {pca_text.shape}")

    # Step 5: Build projector model (or load from cache)
    logger.info(f"Step 5: Building {projector_type} projector")

    spec = BuildSpec(
        teacher_names=teacher_names,
        teacher_dims=whitening_stats["dims"],  # Use whitened dims
        text_dim=text_concat.shape[1],
        out_dim=projector_dim,
        pca_image=pca_image,
        pca_text=pca_text,
        kind=projector_type,
    )

    projector_model = build_model(spec)
    projector_model.to(device)

    # Check if trained projector already exists in cache
    trained_projector_loaded = _try_load_trained_projector(
        projector_model,
        cache,
        whitening_stats,
        projector_dim,
        projector_type,
        teacher_names,
    )

    if trained_projector_loaded:
        logger.info("=" * 60)
        logger.info("LOADED TRAINED PROJECTOR FROM CACHE - SKIPPING TRAINING")
        logger.info("=" * 60)
        # Skip training - jump directly to inference
        n_epochs = 0  # This will cause the training loop to skip
    else:
        logger.info(
            f"Projector built: {sum(p.numel() for p in projector_model.parameters())} params"
        )

    # Step 6: Prepare training dataset
    # Store pre-whitened embeddings to ensure dimension consistency
    logger.info("Step 6: Preparing training dataset with pre-whitened embeddings")

    samples = []
    for idx in range(len(z_cat)):
        # Store PRE-WHITENED embeddings (not original!)
        teacher_embs_dict = {
            teacher_names[i]: whitened_image_embeddings[i][idx]
            for i in range(len(teacher_names))
        }

        source = (
            filtered_df.iloc[idx][dataset_col]
            if dataset_col in filtered_df.columns
            else "unknown"
        )

        text_vec = l2_normalize(text_concat[idx : idx + 1], axis=1)[0]

        sample = Sample(
            image_id=str(idx),
            text_vec=text_vec,
            source=source,
            teacher_embs=teacher_embs_dict,
        )
        samples.append(sample)

    # Now safe to delete whitened_image_embeddings since samples are created
    del whitened_image_embeddings
    gc.collect()

    logger.info(f"Created {len(samples)} samples")

    # Split into train and validation sets
    if validation_split > 0 and validation_split < 1.0:
        n_val = int(len(samples) * validation_split)
        n_train = len(samples) - n_val

        # Shuffle samples for random split
        import random

        random.seed(42)
        indices = list(range(len(samples)))
        random.shuffle(indices)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]

        logger.info(
            f"Split: {len(train_samples)} train, {len(val_samples)} val samples"
        )
    else:
        train_samples = samples
        val_samples = []
        logger.info(
            f"No validation split: using all {len(train_samples)} samples for training"
        )

    # Create train dataset
    train_dataset = PrecomputedDataset(train_samples, teacher_names)

    # Create train sampler
    if use_domain_balanced:
        sources = [s.source for s in train_samples]
        sampler = DomainBalancedBatchSampler(sources, batch_size, shuffle=True)
        logger.info(f"Domain-balanced sampler: {len(set(sources))} sources")
    else:
        sampler = RandomSampler(train_dataset)
        logger.info("Random sampler")

    # Collate function receives pre-whitened embeddings,
    # so we use identity transformations (whitening already applied)
    whiten_spec = WhitenSpec(
        mu=[np.zeros((1, dim), dtype=np.float32) for dim in whitening_stats["dims"]],
        W=[np.eye(dim, dtype=np.float32) for dim in whitening_stats["dims"]],
        dims=whitening_stats["dims"],
    )
    collate_fn = make_collate(whiten_spec)

    effective_workers = resolve_num_workers(min(4, num_workers))

    # Create train loader
    if use_domain_balanced:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=effective_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=effective_workers,
            pin_memory=True,
        )

    logger.info(f"Train DataLoader created: {len(train_loader)} batches/epoch")

    # Create validation loader if we have validation samples
    val_loader = None
    if val_samples:
        val_dataset = PrecomputedDataset(val_samples, teacher_names)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=effective_workers,
            pin_memory=True,
        )
        logger.info(f"Validation DataLoader created: {len(val_loader)} batches")

    # Step 7: Train projector
    logger.info(f"Step 7: Training projector for {n_epochs} epochs")

    optimizer = torch.optim.AdamW(
        projector_model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    structure_loss = None
    if loss_type.lower() == "structure":
        structure_loss = CLIPLoss(
            temperature=0.07,
            normalize_latents=True,
            warmup_steps=hierarchy_warmup,
            lambda_hierarchy=hierarchy_lambda,
            hierarchy_levels=hierarchy_levels,
            hierarchy_weighting=hierarchy_weighting,
            hierarchy_margin=hierarchy_margin,
            lambda_consistency=consistency_lambda,
            teacher_dims=whitening_stats["dims"],
            text_dims=whitening_stats["text_whitening"]["dims"],
        )
        logger.info(
            f"Using STRUCTURE loss: "
            f"lambda={hierarchy_lambda} levels={hierarchy_levels} "
            f"warmup={hierarchy_warmup} consistency={consistency_lambda}"
        )
        logger.info(
            f"Per-model hierarchy: {len(whitening_stats['dims'])} image models, "
            f"{len(whitening_stats['text_whitening']['dims'])} text models"
        )
    elif loss_type.lower() != "clip":
        raise ValueError(f"Unknown projector loss type: {loss_type}")

    last_epoch_loss = 0.0
    last_temp = float(projector_model.logit_scale.exp().item())
    best_val_recall = 0.0
    best_model_state = None

    if validate_every_n_epochs <= 0:
        logger.info("Validation disabled for projector training")

    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch+1}/{n_epochs}")

        # Training
        epoch_loss, extra_logs = train_one_epoch(
            projector_model,
            train_loader,
            optimizer,
            device=device,
            structure_loss=structure_loss,
            wandb_run=wandb_run,
            log_prefix="projector/train",
            epoch=epoch,
        )
        last_epoch_loss = epoch_loss
        logger.info(f"  Train Loss: {epoch_loss:.4f}")
        logit_scale = projector_model.logit_scale.exp().item()
        last_temp = logit_scale
        effective_temp = 1.0 / max(logit_scale, 1e-8)
        logger.info(
            f"  Logit scale: {logit_scale:.4f} (temperature={effective_temp:.4f})"
        )

        # Log training metrics
        if wandb_run is not None:
            metrics = {
                "projector/train/loss": epoch_loss,
                "projector/train/logit_scale": logit_scale,
                "projector/train/temperature": effective_temp,
                "projector/epoch": epoch + 1,
            }
            for idx, group in enumerate(optimizer.param_groups):
                metrics[f"projector/train/lr_group_{idx}"] = float(group.get("lr", 0.0))
            metrics.update(
                {f"projector/train/{k}": v for k, v in extra_logs.items()}
                if extra_logs
                else {}
            )
            wandb.log(metrics)

        # Validation
        if (
            val_loader is not None
            and validate_every_n_epochs > 0
            and (epoch + 1) % validate_every_n_epochs == 0
        ):
            logger.info("  Running validation...")
            val_metrics = validate_epoch(
                projector_model,
                val_loader,
                device,
                structure_loss=structure_loss,
            )

            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val I2T Recall@1: {val_metrics['i2t_recall@1']:.2f}%")
            logger.info(f"  Val T2I Recall@1: {val_metrics['t2i_recall@1']:.2f}%")
            logger.info(f"  Val Avg Recall@5: {val_metrics['avg_recall@5']:.2f}%")

            # Log validation metrics to wandb
            if wandb_run is not None:
                val_wandb_metrics = {
                    f"projector/val/{k}": v for k, v in val_metrics.items()
                }
                val_wandb_metrics["projector/epoch"] = epoch + 1
                wandb.log(val_wandb_metrics)

            # Track best model based on average recall@5
            if val_metrics["avg_recall@5"] > best_val_recall:
                best_val_recall = val_metrics["avg_recall@5"]
                best_model_state = {
                    k: v.cpu().clone() for k, v in projector_model.state_dict().items()
                }
                logger.info(f"  New best model! Avg Recall@5: {best_val_recall:.2f}%")

            # Clear GPU cache after validation to free memory before next epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Load best model if we tracked it
    if best_model_state is not None:
        projector_model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with Avg Recall@5: {best_val_recall:.2f}%")
        if wandb_run is not None:
            wandb.log(
                {
                    "projector/best_val_recall@5": best_val_recall,
                }
            )

    # Release training-only objects before inference
    del train_loader
    del train_dataset
    if val_loader is not None:
        del val_loader
        del val_dataset
    del sampler
    gc.collect()

    # Step 8: Extract final projected embeddings
    logger.info("Step 8: Extracting final projected embeddings")

    projector_model.eval()
    final_image_embeddings = []
    final_text_embeddings = []

    batch_size_inference = 512
    with torch.no_grad():
        for start_idx in tqdm(
            range(0, len(z_cat), batch_size_inference), desc="Projecting"
        ):
            end_idx = min(start_idx + batch_size_inference, len(z_cat))

            z_cat_batch = torch.from_numpy(z_cat[start_idx:end_idx]).float().to(device)
            t_vec_batch = (
                torch.from_numpy(text_concat[start_idx:end_idx]).float().to(device)
            )
            t_vec_batch = t_vec_batch / (
                t_vec_batch.norm(p=2, dim=-1, keepdim=True) + 1e-8
            )

            zi, zt, _ = projector_model(z_cat_batch, t_vec_batch)

            final_image_embeddings.append(zi.cpu().numpy())
            final_text_embeddings.append(zt.cpu().numpy())

    final_image_embeddings = np.vstack(final_image_embeddings)
    final_text_embeddings = np.vstack(final_text_embeddings)
    del z_cat, text_concat
    del samples
    gc.collect()

    logger.info(f"Final embeddings: {final_image_embeddings.shape}")
    logger.info("=" * 60)
    logger.info("Projector pipeline complete!")
    logger.info("=" * 60)

    if wandb_run is not None:
        wandb.log(
            {
                "projector/final_image_dim": final_image_embeddings.shape[1],
                "projector/final_text_dim": final_text_embeddings.shape[1],
            },
        )
        wandb_run.summary["projector/num_samples"] = len(final_image_embeddings)
        wandb_run.summary["projector/final_loss"] = float(last_epoch_loss)
        wandb_run.summary["projector/final_logit_scale"] = float(last_temp)

    return (
        final_image_embeddings,
        final_text_embeddings,
        dataset_labels,
        corrupted_images if corrupted_images else [],
        projector_model,
        whitening_stats,
        filtered_df.reset_index(drop=True),
        wandb_run,
    )


def _prepare_normalized_embeddings(
    all_image_embeddings: List[np.ndarray],
    clip_text_embeddings: List[np.ndarray],
    original_dims: List[int],
    clip_indices: List[int],
) -> Dict:
    """Prepare normalized (no whitening) embedding statistics."""
    text_whitening_stats = {
        "mu": [],
        "W": [],
        "dims": [emb.shape[1] for emb in clip_text_embeddings],
    }

    whitening_stats = {
        "mu": [],
        "W": [],
        "dims": original_dims,
        "original_dims": original_dims,
        "clip_indices": clip_indices,
        "text_whitening": text_whitening_stats,
    }

    return whitening_stats


def _prepare_whitened_embeddings(
    all_image_embeddings: List[np.ndarray],
    clip_text_embeddings: List[np.ndarray],
    original_dims: List[int],
    clip_indices: List[int],
    max_whitening_dim: int,
) -> Dict:
    """Fit whitening transformations with optional truncation."""
    # Fit text whitening
    text_whitening_stats = {"mu": [], "W": [], "dims": []}

    for i, text_embs in enumerate(clip_text_embeddings):
        model_dim = text_embs.shape[1]
        max_components = (
            max_whitening_dim
            if max_whitening_dim > 0 and model_dim > max_whitening_dim
            else None
        )

        logger.info(
            f"  Text whitening for CLIP {i+1}: dim={model_dim}, "
            f"using {max_components or model_dim} components"
        )

        text_embs_normed = l2_normalize(text_embs, axis=1)
        mu, W = fit_whitener(text_embs_normed, max_components=max_components)
        text_whitening_stats["mu"].append(mu)
        text_whitening_stats["W"].append(W)
        text_whitening_stats["dims"].append(W.shape[0])

    # Fit image whitening
    whitening_stats = {
        "mu": [],
        "W": [],
        "dims": [],
        "original_dims": original_dims,
        "clip_indices": clip_indices,
        "text_whitening": text_whitening_stats,
    }

    for i, embs in enumerate(all_image_embeddings):
        model_dim = embs.shape[1]
        max_components = (
            max_whitening_dim
            if max_whitening_dim > 0 and model_dim > max_whitening_dim
            else None
        )

        logger.info(
            f"  Image whitening for model {i+1}: dim={model_dim}, "
            f"using {max_components or model_dim} components"
        )

        embs_normed = l2_normalize(embs, axis=1)
        mu, W = fit_whitener(embs_normed, max_components=max_components)
        whitening_stats["mu"].append(mu)
        whitening_stats["W"].append(W)
        whitening_stats["dims"].append(W.shape[0])

    # Log dimension reduction
    original_total = sum(original_dims)
    truncated_total = sum(whitening_stats["dims"])
    if original_total > truncated_total:
        reduction_pct = 100 * (1 - truncated_total / original_total)
        logger.info(
            f"Dimension reduction: {original_total} → {truncated_total} "
            f"({reduction_pct:.1f}% reduction)"
        )

    return whitening_stats
