"""Cache management system for embeddings with validation."""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..data.preprocessing import normalize_multilabel_columns


@dataclass
class CacheConfig:
    """Configuration for embedding cache validation."""

    max_whitening_dim: int
    skip_whitening: bool
    projector_dim: Optional[int] = None
    projector_type: Optional[str] = None
    svd_components: Optional[int] = None
    fusion_method: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_whitening_dim": self.max_whitening_dim,
            "skip_whitening": self.skip_whitening,
            "projector_dim": self.projector_dim,
            "projector_type": self.projector_type,
            "svd_components": self.svd_components,
            "fusion_method": self.fusion_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        """Create from dictionary."""

        # Helper to convert numpy types to Python types
        def to_python(value):
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                # Handle numpy string arrays (scalars stored as 0-d arrays)
                if value.ndim == 0:
                    return value.item()
                return value.tolist()
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            if isinstance(value, np.bool_):
                return bool(value)
            return value

        return cls(
            max_whitening_dim=int(to_python(data.get("max_whitening_dim", 768))),
            skip_whitening=bool(to_python(data.get("skip_whitening", False))),
            projector_dim=to_python(data.get("projector_dim")),
            projector_type=to_python(data.get("projector_type")),
            svd_components=to_python(data.get("svd_components")),
            fusion_method=str(to_python(data.get("fusion_method", "none"))),
        )

    def is_compatible_with(self, other: "CacheConfig") -> bool:
        """Check if this cache config is compatible with another.

        Args:
            other: Another cache configuration

        Returns:
            True if configurations are compatible
        """
        # Fusion method must match
        if self.fusion_method != other.fusion_method:
            logger.debug(
                f"Fusion method mismatch: cached={self.fusion_method}, expected={other.fusion_method}"
            )
            return False

        # For trained projector, check projector settings
        if self.fusion_method == "trained_projector":
            # If cached config has None for projector settings (old cache bug),
            # try to infer from the actual saved projector files
            if self.projector_dim is None or self.projector_type is None:
                logger.warning(
                    "Cached projector config incomplete (projector_dim or projector_type is None). "
                    "This cache was created with an older version. Will attempt to load projector "
                    "from saved files instead of using combined cache."
                )
                return False

            if self.projector_dim != other.projector_dim:
                logger.debug(
                    f"Projector dim mismatch: cached={self.projector_dim}, expected={other.projector_dim}"
                )
                return False
            if self.projector_type != other.projector_type:
                logger.debug(
                    f"Projector type mismatch: cached={self.projector_type}, expected={other.projector_type}"
                )
                return False
            if self.max_whitening_dim != other.max_whitening_dim:
                logger.debug(
                    f"Max whitening dim mismatch: cached={self.max_whitening_dim}, expected={other.max_whitening_dim}"
                )
                return False
            if self.skip_whitening != other.skip_whitening:
                logger.debug(
                    f"Skip whitening mismatch: cached={self.skip_whitening}, expected={other.skip_whitening}"
                )
                return False

        # For SVD, check components
        if "svd_" in self.fusion_method:
            if self.svd_components != other.svd_components:
                logger.debug(
                    f"SVD components mismatch: cached={self.svd_components}, expected={other.svd_components}"
                )
                return False

        return True


def get_cache_config_hash(config: CacheConfig) -> str:
    """Generate a hash for cache configuration.

    Args:
        config: Cache configuration

    Returns:
        8-character hash string
    """
    config_str = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def get_single_model_cache_path(model_path: str, output_dir: str) -> str:
    """Generate cache path for a single model.

    Args:
        model_path: Path or name of the model
        output_dir: Base output directory

    Returns:
        Path to cache file
    """
    model_name_clean = (
        model_path.replace("assets/", "").replace("/", "_").replace("\\", "_")
    )
    cache_dir = os.path.join(output_dir, model_name_clean, "embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "embeddings.npz")


class EmbeddingCache:
    """Manager for embedding cache operations with validation."""

    def __init__(self, output_dir: str):
        """Initialize cache manager.

        Args:
            output_dir: Base output directory for caches
        """
        self.output_dir = output_dir

    def get_single_model_path(self, model_path: str) -> str:
        """Get cache path for a single model.

        Args:
            model_path: Model path or name

        Returns:
            Path to cache file
        """
        return get_single_model_cache_path(model_path, self.output_dir)

    def get_combined_cache_path(self, run_name: str) -> str:
        """Get cache path for combined embeddings.

        Args:
            run_name: Name of the run

        Returns:
            Path to combined cache file
        """
        cache_dir = os.path.join(self.output_dir, run_name, "embeddings")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "embeddings.npz")

    def save_single_model(
        self,
        model_path: str,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        dataset_labels: np.ndarray,
        df: pd.DataFrame,
    ) -> None:
        """Save embeddings for a single model.

        Args:
            model_path: Model path or name
            image_embeddings: Image embedding array
            text_embeddings: Text embedding array
            dataset_labels: Dataset label array
            df: DataFrame with metadata
        """
        cache_path = self.get_single_model_path(model_path)
        cache_dir = os.path.dirname(cache_path)
        df_cache_path = os.path.join(cache_dir, "dataframe.csv")

        logger.info(f"Saving single model cache to {cache_path}")
        np.savez(
            cache_path,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            dataset_labels=dataset_labels,
        )
        df.to_csv(df_cache_path, index=False)

    def load_single_model(
        self, model_path: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]:
        """Load embeddings for a single model.

        Args:
            model_path: Model path or name

        Returns:
            Tuple of (image_embeddings, text_embeddings, dataset_labels, df) or None
        """
        cache_path = self.get_single_model_path(model_path)
        cache_dir = os.path.dirname(cache_path)
        df_cache_path = os.path.join(cache_dir, "dataframe.csv")

        if not (os.path.exists(cache_path) and os.path.exists(df_cache_path)):
            logger.debug(f"Cache not found for {model_path}")
            return None

        try:
            with np.load(cache_path, allow_pickle=True) as data:
                image_embs = data["image_embeddings"]
                text_embs = data["text_embeddings"]
                labels = data["dataset_labels"]

            # Convert numpy-wrapped None back to actual None
            # (happens when SSL models save None for text_embeddings)
            if text_embs is not None:
                if text_embs.dtype == object and text_embs.shape == ():
                    # This is a 0-d array containing None
                    text_embs = None

            # Handle empty DataFrames
            try:
                df = pd.read_csv(df_cache_path)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()

            df = normalize_multilabel_columns(df)
            logger.info(f"Loaded cache for {model_path}: {len(df)} samples")
            return image_embs, text_embs, labels, df

        except Exception as e:
            logger.warning(f"Failed to load cache for {model_path}: {e}")
            return None

    def save_combined(
        self,
        run_name: str,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        dataset_labels: np.ndarray,
        config: CacheConfig,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save combined embeddings with configuration metadata.

        Args:
            run_name: Name of the run
            image_embeddings: Combined image embeddings
            text_embeddings: Combined text embeddings
            dataset_labels: Dataset labels
            config: Cache configuration for validation
            extra_data: Extra data to save (e.g., projector model, whitening stats)
        """
        cache_path = self.get_combined_cache_path(run_name)

        logger.info(
            f"Saving combined cache to {cache_path} "
            f"(fusion_method={config.fusion_method})"
        )

        # Calculate dimensions safely
        image_dim = 0
        if image_embeddings is not None and image_embeddings.size > 0:
            image_dim = image_embeddings.shape[1]

        text_dim = 0
        if text_embeddings is not None:
            # Check if it's a real array (not None wrapped in numpy array)
            if (
                hasattr(text_embeddings, "shape")
                and len(text_embeddings.shape) > 0
                and text_embeddings.shape[0] > 0
            ):
                text_dim = text_embeddings.shape[1]

        save_dict = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "dataset_labels": dataset_labels,
            "image_dim": image_dim,
            "text_dim": text_dim,
            **config.to_dict(),
        }

        if extra_data:
            save_dict.update(extra_data)

        np.savez(cache_path, **save_dict)

    def load_combined(
        self, run_name: str, expected_config: Optional[CacheConfig] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Load combined embeddings with configuration validation.

        Args:
            run_name: Name of the run
            expected_config: Expected configuration for validation (optional)

        Returns:
            Tuple of (image_embeddings, text_embeddings, dataset_labels, metadata) or None
        """
        cache_path = self.get_combined_cache_path(run_name)

        if not os.path.exists(cache_path):
            logger.debug(f"Combined cache not found: {cache_path}")
            return None

        try:
            with np.load(cache_path, allow_pickle=True) as data:
                # Load embeddings
                image_embs = data["image_embeddings"]
                text_embs = data["text_embeddings"]
                labels = data["dataset_labels"]

                # Convert numpy-wrapped None back to actual None
                if text_embs is not None:
                    if text_embs.dtype == object and text_embs.shape == ():
                        text_embs = None

                # Load configuration
                cached_config = CacheConfig.from_dict(dict(data))

                # Validate configuration if expected_config is provided
                if expected_config is not None:
                    if not cached_config.is_compatible_with(expected_config):
                        logger.info(
                            f"Cache configuration mismatch: "
                            f"cached={cached_config.to_dict()}, "
                            f"expected={expected_config.to_dict()}"
                        )
                        return None

                logger.info(
                    f"Loaded combined cache: {len(image_embs)} samples, "
                    f"fusion_method={cached_config.fusion_method}"
                )

                # Extract metadata
                image_dim = 0
                if image_embs is not None and image_embs.size > 0:
                    image_dim = image_embs.shape[1]

                text_dim = 0
                if text_embs is not None:
                    # text_embs could be a numpy None-like object (0-d array containing None)
                    # Check if it's actually a real array with data
                    if (
                        hasattr(text_embs, "shape")
                        and len(text_embs.shape) > 0
                        and text_embs.shape[0] > 0
                    ):
                        text_dim = text_embs.shape[1]

                corrupted_indices = data.get("corrupted_indices")
                if corrupted_indices is not None:
                    corrupted_indices = corrupted_indices.tolist()

                metadata = {
                    "config": cached_config,
                    "image_dim": int(data.get("image_dim", image_dim)),
                    "text_dim": int(data.get("text_dim", text_dim)),
                    "corrupted_indices": corrupted_indices or [],
                }

                return image_embs, text_embs, labels, metadata

        except Exception as e:
            logger.warning(f"Failed to load combined cache: {e}")
            return None

    def check_individual_compatibility(
        self, model_paths: List[str], df: pd.DataFrame
    ) -> Tuple[bool, List[Optional[pd.DataFrame]]]:
        """Check if individual model caches are compatible.

        Args:
            model_paths: List of model paths
            df: Expected dataframe structure

        Returns:
            Tuple of (compatible, list of cached dataframes or None)
        """
        cached_dfs = []

        for i, model_path in enumerate(model_paths):
            result = self.load_single_model(model_path)
            if result is None:
                logger.info(f"Missing cache for model {i+1}: {model_path}")
                return False, []

            _, _, _, cached_df = result
            cached_dfs.append(cached_df)

        # Check compatibility between caches
        if len(cached_dfs) > 1:
            first_df = cached_dfs[0]
            for i, cached_df in enumerate(cached_dfs[1:], 1):
                if len(cached_df) != len(first_df):
                    logger.warning(
                        f"Length mismatch: model 1 ({len(first_df)}) vs "
                        f"model {i+1} ({len(cached_df)})"
                    )
                    return False, []

                # Check if same set of images (order doesn't matter for ensemble)
                if "img_path" in cached_df.columns and "img_path" in first_df.columns:
                    first_paths = set(first_df["img_path"].values)
                    cached_paths = set(cached_df["img_path"].values)
                    if first_paths != cached_paths:
                        missing_in_cached = len(first_paths - cached_paths)
                        missing_in_first = len(cached_paths - first_paths)
                        logger.warning(
                            f"Image set mismatch between model 1 and model {i+1}: "
                            f"{missing_in_cached} images missing in model {i+1}, "
                            f"{missing_in_first} images missing in model 1"
                        )
                        return False, []
                    # If sets match but order differs, we need to reorder
                    # This is handled by the projection pipeline which aligns by img_path
                    if not cached_df["img_path"].equals(first_df["img_path"]):
                        logger.info(
                            f"Model {i+1} cache has same images as model 1 but in different order - will align during projection"
                        )

            logger.info(
                f"All {len(cached_dfs)} model caches are compatible "
                f"({len(first_df)} samples)"
            )

        return True, cached_dfs
