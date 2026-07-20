"""Data preprocessing utilities for SkinMap."""

import ast
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from loguru import logger
from PIL import Image
from tqdm import tqdm

from src.skinmap.constants import MULTILABEL_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def coerce_multilabel(value: Union[str, list, None, Any]) -> Optional[list[str]]:
    """Convert stringified lists to proper Python lists for multi-label columns.

    Args:
        value: Input value that may be a string, list, or None

    Returns:
        List of strings or None if value is empty/invalid
    """
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]

    if value is None or (not isinstance(value, list) and pd.isna(value)):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        # Common cases like "['A', 'B']" or '["A", "B"]'
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            # Fallback: split on comma for ad-hoc strings
            parsed = [
                part.strip(" '\"") for part in stripped.split(",") if part.strip(" '\"")
            ]
        else:
            if isinstance(parsed, (list, tuple, set)):
                parsed = [
                    str(item).strip()
                    for item in parsed
                    if item is not None and str(item).strip()
                ]
            elif parsed is None:
                return None
            else:
                parsed = [str(parsed).strip()]
        return parsed if parsed else None

    return None


def normalize_multilabel_columns(
    df: pd.DataFrame, columns: set[str] = MULTILABEL_COLUMNS
) -> pd.DataFrame:
    """Ensure multi-label columns are stored as actual lists for downstream processing.

    Args:
        df: DataFrame to process
        columns: Set of column names to normalize

    Returns:
        DataFrame with normalized multi-label columns
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(coerce_multilabel)
    return df


def get_stable_hash(img_path: str) -> str:
    """Generate a stable hash ID for an image path.

    Prefers the user-provided relative path (e.g., ``data/ISIC/190.jpg``) so hashes remain
    stable across different machines or mount points. Falls back to a path relative to the
    project root, and finally to the absolute path when necessary.

    Args:
        img_path: Path to the image file as stored in the dataframe

    Returns:
        A 16-character hexadecimal hash string
    """
    path_obj = Path(img_path)

    # Prefer the original relative string if possible (captures dataset context such as data/ISIC/…)
    if not path_obj.is_absolute():
        key = path_obj.as_posix()
    else:
        resolved = path_obj.resolve(strict=False)
        try:
            key = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            key = resolved.as_posix()

    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def create_thumbnail(
    img_path: str,
    thumbnail_dir: str,
    max_size: int = 256,
    quality: int = 85,
) -> Optional[str]:
    """Create a thumbnail for an image and return the relative path.

    Args:
        img_path: Path to the original image
        thumbnail_dir: Directory to save thumbnails
        max_size: Maximum dimension for thumbnail (default: 256)
        quality: JPEG quality (default: 85)

    Returns:
        Relative path to thumbnail (e.g., "thumbnail/hash.jpg") or None on failure
    """
    try:
        # Generate stable hash ID from path (works even if source image is temporarily unavailable)
        img_id = get_stable_hash(img_path)
        thumbnail_filename = f"{img_id}.jpg"
        thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)

        # Check if thumbnail already exists BEFORE validating source image
        # This allows reusing cached thumbnails even if source paths have changed
        if os.path.exists(thumbnail_path):
            logger.debug(f"Thumbnail already exists, reusing: {thumbnail_path}")
            return f"thumbnail/{thumbnail_filename}"

        # Only check source image if we need to create a new thumbnail
        if not os.path.exists(img_path):
            logger.debug(f"File does not exist: {img_path}")
            return None

        with Image.open(img_path) as im:
            im = im.convert("RGB")

            # Resize if image is too large
            if max(im.size) > max_size:
                ratio = max_size / max(im.size)
                new_size = tuple(int(dim * ratio) for dim in im.size)
                im = im.resize(new_size, Image.Resampling.LANCZOS)

            im.save(thumbnail_path, format="JPEG", quality=quality, optimize=True)
            return f"thumbnail/{thumbnail_filename}"

    except Exception as e:
        logger.debug(f"Failed to process image {img_path}: {e}")
        return None


def create_thumbnail_column_parallel(
    img_paths: pd.Series,
    thumbnail_dir: str,
    max_size: int = 256,
    quality: int = 85,
    max_workers: Optional[int] = None,
) -> pd.Series:
    """Create thumbnail files and return their paths for atlas parquet.

    Uses content-based hashing to generate stable, unique IDs for each image.
    This ensures:
    - No duplicates across different runs or datasets
    - Same image always gets same thumbnail ID
    - Thumbnails can be reused if they already exist

    Args:
        img_paths: Series of image paths
        thumbnail_dir: Directory to save thumbnails
        max_size: Maximum image dimension for resizing (default: 256)
        quality: JPEG quality for thumbnails (default: 85)
        max_workers: Number of workers for parallel processing (default: min(cpu_count(), 8))

    Returns:
        Series of thumbnail paths (e.g., "thumbnail/hash.jpg")
    """
    if max_workers is None:
        max_workers = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues

    os.makedirs(thumbnail_dir, exist_ok=True)

    total_images = len(img_paths)
    logger.info(
        f"Starting thumbnail generation: {total_images} images, {max_workers} workers "
        f"(max_size={max_size}, quality={quality})"
    )

    paths_list = img_paths.tolist()

    # Pre-check existing thumbnails
    existing_before = set()
    for img_path in paths_list:
        if img_path and os.path.exists(img_path):
            thumb_id = get_stable_hash(img_path)
            thumb_path = os.path.join(thumbnail_dir, f"{thumb_id}.jpg")
            if os.path.exists(thumb_path):
                existing_before.add(thumb_id)

    logger.info(
        f"Found {len(existing_before)} thumbnails that already exist and can be reused"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda img_path: create_thumbnail(
                        img_path, thumbnail_dir, max_size, quality
                    ),
                    paths_list,
                ),
                total=total_images,
                desc="Creating thumbnails",
                unit="img",
            )
        )

    # Statistics
    failed_count = sum(1 for r in results if r is None)
    success_count = total_images - failed_count

    unique_thumbnails = set()
    for result in results:
        if result is not None:
            thumb_id = result.split("/")[-1].replace(".jpg", "")
            unique_thumbnails.add(thumb_id)

    total_unique = len(unique_thumbnails)
    newly_created = total_unique - len(existing_before)
    reused_count = len(existing_before)

    success_rate = (success_count / total_images) * 100 if total_images > 0 else 0.0
    logger.info(
        f"Thumbnail generation complete: {success_count}/{total_images} successful ({success_rate:.1f}%), "
        f"{total_unique} unique thumbnails ({newly_created} newly created, {reused_count} reused), "
        f"{failed_count} failed"
    )

    if failed_count > 0:
        logger.warning(
            f"{failed_count} images failed to process and will have None paths"
        )

    return pd.Series(results, index=img_paths.index)
