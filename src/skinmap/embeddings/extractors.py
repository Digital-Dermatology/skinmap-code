"""Basic embedding extractors for CLIP and SSL models.

Strategy for corrupted samples:
1. Use dummy images during extraction (keeps DataLoader simple)
2. Mark corrupted indices with negative values (-idx - 1)
3. After extraction, filter out corrupted samples from embeddings AND dataframe in one step
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data.transforms import get_imagenet_transform
from ..utils.worker_utils import resolve_num_workers


class ImageDataset(Dataset):
    """Unified dataset for both SSL and CLIP models.

    Returns dummy images for corrupted files and marks them with negative indices.
    """

    def __init__(
        self, df: pd.DataFrame, transform=None, text_col: Optional[str] = None
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with img_path column
            transform: Optional image transform (required for SSL)
            text_col: Optional text column name (required for CLIP)
        """
        self.df = df
        self.transform = transform
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Try to load image
        try:
            img = Image.open(row["img_path"]).convert("RGB")

            # For CLIP, thumbnail to reduce memory
            if self.text_col is not None:
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)

            # Apply transform if provided (for SSL)
            if self.transform:
                img = self.transform(img)

            # Get text if needed
            text = None
            if self.text_col:
                text = row[self.text_col]
                if pd.isna(text):
                    logger.warning(f"NaN text at idx {idx}, using 'unknown'")
                    text = "unknown"

            return img, text, idx

        except Exception as e:
            logger.debug(f"Failed to load sample at idx {idx}: {e}")

            # Create dummy image
            dummy = Image.new("RGB", (224, 224))
            if self.transform:
                dummy = self.transform(dummy)

            # Return dummy with negative index to mark as corrupted
            # Use -idx - 1 to handle idx=0 correctly
            return dummy, "unknown" if self.text_col else None, -idx - 1


def _filter_corrupted(
    embeddings: np.ndarray,
    indices: np.ndarray,
    df: pd.DataFrame,
    text_embeddings: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int], pd.DataFrame]:
    """Filter out corrupted samples from embeddings and dataframe.

    Args:
        embeddings: Image embeddings array
        indices: Index array (negative values mark corrupted samples)
        df: DataFrame to filter
        text_embeddings: Optional text embeddings to filter

    Returns:
        Tuple of (filtered_image_embs, filtered_text_embs, corrupted_indices, filtered_df)
    """
    # Identify corrupted samples (negative indices)
    valid_mask = indices >= 0
    corrupted_mask = ~valid_mask

    # Extract corrupted indices (convert back from negative encoding)
    corrupted_indices = [-(idx + 1) for idx in indices[corrupted_mask]]

    # Filter to valid samples only
    valid_embeddings = embeddings[valid_mask]
    valid_text_embeddings = (
        text_embeddings[valid_mask] if text_embeddings is not None else None
    )
    valid_indices = indices[valid_mask]

    # Filter dataframe and reset index
    filtered_df = df.iloc[valid_indices].reset_index(drop=True)

    return valid_embeddings, valid_text_embeddings, corrupted_indices, filtered_df


def extract_ssl_embeddings(
    ssl_model,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    dataset_col: str = "dataset_desc",
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], pd.DataFrame]:
    """Extract embeddings from SSL model, filtering out corrupted samples.

    Args:
        ssl_model: SSL model
        df: DataFrame with img_path column
        device: torch device
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to process
        dataset_col: Column name for dataset labels
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (image_embeddings, text_embeddings, dataset_labels, corrupted_indices, filtered_df)
        - image_embeddings: Valid embeddings only, corrupted removed
        - text_embeddings: Zeros (SSL models have no text encoder)
        - dataset_labels: Labels from filtered_df
        - corrupted_indices: Original indices that were corrupted
        - filtered_df: Dataframe with corrupted rows removed
    """
    transform = get_imagenet_transform()

    # Sample if needed
    df_work = df.copy()
    if max_samples and len(df_work) > max_samples:
        df_work = df_work.sample(n=max_samples, random_state=42)
    df_work = df_work.reset_index(drop=True)

    logger.info(f"Extracting SSL embeddings from {len(df_work)} samples")

    dataset = ImageDataset(df_work, transform=transform, text_col=None)

    def collate_fn(batch):
        # Images already transformed, text is None for SSL
        images, _, idxs = zip(*batch)
        return torch.stack(list(images)), list(idxs)

    worker_count = resolve_num_workers(num_workers)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    ssl_model.eval()
    all_embeddings = []
    all_indices = []

    with torch.no_grad():
        for batch_images, batch_indices in tqdm(
            dataloader, desc="Extracting SSL embeddings"
        ):
            batch_images = batch_images.to(device)
            emb = ssl_model(batch_images)
            # Normalize embeddings
            emb = torch.nn.functional.normalize(emb, dim=-1, p=2)
            all_embeddings.append(emb.cpu())
            all_indices.extend(batch_indices)

    # Concatenate all embeddings (handle empty case)
    if len(all_embeddings) == 0:
        # Empty dataframe - return empty arrays
        return (
            np.array([], dtype=np.float32).reshape(0, 0),
            None,  # SSL has no text embeddings
            np.array([], dtype=np.int64),
            [],
            df_work,
        )

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    indices = np.array(all_indices)

    # Filter corrupted samples
    valid_embeddings, _, corrupted_indices, filtered_df = _filter_corrupted(
        embeddings, indices, df_work
    )

    # SSL models return None for text embeddings
    text_embeddings = None

    # Get dataset labels from filtered dataframe
    dataset_labels = filtered_df[dataset_col].values

    logger.info(
        f"SSL extraction complete: {len(valid_embeddings)} valid (dim={valid_embeddings.shape[1]}), "
        f"{len(corrupted_indices)} corrupted removed"
    )

    return (
        valid_embeddings,
        text_embeddings,
        dataset_labels,
        corrupted_indices,
        filtered_df,
    )


def extract_clip_embeddings(
    model,
    processor,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    dataset_col: str = "dataset_desc",
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], pd.DataFrame]:
    """Extract embeddings from CLIP model, filtering out corrupted samples.

    Args:
        model: CLIP model
        processor: CLIP processor
        df: DataFrame with img_path and description columns
        device: torch device
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to process
        dataset_col: Column name for dataset labels and text
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (image_embeddings, text_embeddings, dataset_labels, corrupted_indices, filtered_df)
        - image_embeddings: Valid embeddings only, corrupted removed
        - text_embeddings: Valid text embeddings, corrupted removed
        - dataset_labels: Labels from filtered_df
        - corrupted_indices: Original indices that were corrupted
        - filtered_df: Dataframe with corrupted rows removed
    """
    # Sample if needed
    df_work = df.copy()
    if max_samples and len(df_work) > max_samples:
        df_work = df_work.sample(n=max_samples, random_state=42)
    df_work = df_work.reset_index(drop=True)

    logger.info(f"Extracting CLIP embeddings from {len(df_work)} samples")

    dataset = ImageDataset(df_work, transform=None, text_col=dataset_col)

    def collate_fn(batch):
        # Images are PIL, not transformed yet
        images, texts, idxs = zip(*batch)
        image_inputs = processor(images=list(images), return_tensors="pt", padding=True)
        text_inputs = processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        )
        return image_inputs, text_inputs, list(idxs)

    worker_count = resolve_num_workers(num_workers)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=worker_count,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Unwrap DDP if needed
    actual_model = model.module if hasattr(model, "module") else model
    actual_model.to(device)
    actual_model.eval()

    image_embs = []
    text_embs = []
    all_indices = []

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for image_inputs, text_inputs, idxs in tqdm(
            dataloader, desc="Extracting CLIP embeddings"
        ):
            # Move to device
            image_inputs = {
                k: v.to(device, non_blocking=True) for k, v in image_inputs.items()
            }
            text_inputs = {
                k: v.to(device, non_blocking=True) for k, v in text_inputs.items()
            }

            img_feats = actual_model.get_image_features(**image_inputs)
            txt_feats = actual_model.get_text_features(**text_inputs)

            image_embs.append(img_feats.cpu())
            text_embs.append(txt_feats.cpu())
            all_indices.extend(idxs)

    # Concatenate (handle empty case)
    if len(image_embs) == 0:
        # Empty dataframe - return empty arrays
        return (
            np.array([], dtype=np.float32).reshape(0, 0),
            np.array([], dtype=np.float32).reshape(0, 0),
            np.array([], dtype=np.int64),
            [],
            df_work,
        )

    image_embeddings = torch.cat(image_embs, dim=0).numpy()
    text_embeddings = torch.cat(text_embs, dim=0).numpy()
    indices = np.array(all_indices)

    # Filter corrupted samples
    valid_image_embeddings, valid_text_embeddings, corrupted_indices, filtered_df = (
        _filter_corrupted(image_embeddings, indices, df_work, text_embeddings)
    )

    # Get dataset labels from filtered dataframe
    dataset_labels = filtered_df[dataset_col].values

    logger.info(
        f"CLIP extraction complete: {len(valid_image_embeddings)} valid "
        f"(img_dim={valid_image_embeddings.shape[1]}, txt_dim={valid_text_embeddings.shape[1]}), "
        f"{len(corrupted_indices)} corrupted removed"
    )

    return (
        valid_image_embeddings,
        valid_text_embeddings,
        dataset_labels,
        corrupted_indices,
        filtered_df,
    )
