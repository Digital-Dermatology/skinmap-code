import argparse
import os
import random
import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import glasbey
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
import umap
import wandb
from loguru import logger
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    CLIPConfig,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    get_linear_schedule_with_warmup,
)

Image.MAX_IMAGE_PIXELS = None


# ----------------------------------------
# Configuration & Setup
# ----------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP with distributed training"
    )

    # Model and data
    parser.add_argument(
        "--model_name",
        type=str,
        # suinleelab/monet # openai/clip-vit-base-patch32
        default="suinleelab/monet",
    )
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./assets/")

    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=77)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Model configs
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="If set, initialize the CLIP model with random weights instead of loading pretrained weights.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["clip", "siglip"],
        default="clip",
        help="Which loss to use: 'clip'=softmax contrastive, 'siglip'=pairwise sigmoid",
    )

    # System params
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--wandb_project", type=str, default="SkinMap")
    parser.add_argument("--val_frac", type=float, default=0.01)

    # Visualization parameters
    parser.add_argument("--vis_epochs", type=int, default=10)
    parser.add_argument(
        "--vis_samples",
        type=int,
        default=100_000,
        help="Number of samples to use for visualization",
    )
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", type=str, default="cosine")
    parser.add_argument(
        "--dataset_col",
        type=str,
        default="dataset_desc",
        help="Column name for dataset labels in visualization",
    )
    parser.add_argument(
        "--holdout_datasets",
        type=str,
        nargs="+",
        default=["PAD-UFES-20", "DDI"],
        help="Datasets to exclude from training (hold-out for evaluation)",
    )

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )

    return parser.parse_args()


def setup_logger(level: str):
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------------------
# Distributed Setup
# ----------------------------------------
def setup_distributed() -> Tuple[torch.device, int, int]:
    """Setup distributed training environment"""
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"  # not to enforce timeout
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=7_200_000),  # was 1_800_000
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

        logger.info(f"Initialized distributed training on rank {rank}/{world_size}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank, world_size = 0, 1
        logger.info(f"Single GPU training on device: {device}")

    return device, rank, world_size


def cleanup_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def gather_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Gather embeddings from all GPUs for contrastive learning.
    This is essential for distributed CLIP training.

    Args:
        embeddings: Local embeddings from current GPU [batch_size, embed_dim]

    Returns:
        Gathered embeddings from all GPUs [world_size * batch_size, embed_dim]
    """
    if not torch.distributed.is_initialized():
        return embeddings

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return embeddings

    # Gather embeddings from all GPUs
    gathered_embeddings = [torch.zeros_like(embeddings) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_embeddings, embeddings)

    # Concatenate all embeddings
    return torch.cat(gathered_embeddings, dim=0)


# ----------------------------------------
# Model & Optimizer
# ----------------------------------------
def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return int(params)


def load_model_and_processor(
    model_name: str,
    device: torch.device,
    rank: Optional[int] = None,
    random_init: bool = False,
    loss_type: str = "clip",
):
    # For local checkpoints, extract original model name for processor
    processor_model_name = model_name
    if model_name.startswith("assets/"):
        if "openai_clip-vit-base-patch32" in model_name:
            processor_model_name = "openai/clip-vit-base-patch32"
        elif "monet" in model_name:
            processor_model_name = "suinleelab/monet"

    processor_exceptions: list[Exception] = []
    processor_candidates = [processor_model_name]
    if processor_model_name != model_name:
        processor_candidates.insert(0, model_name)

    def _load_processor(candidate: str, *, local_only: bool = False):
        load_kwargs = {"local_files_only": local_only}
        try:
            return CLIPProcessor.from_pretrained(candidate, **load_kwargs)
        except Exception as base_exc:
            try:
                tokenizer = CLIPTokenizer.from_pretrained(candidate, **load_kwargs)
                image_processor = CLIPImageProcessor.from_pretrained(
                    candidate, **load_kwargs
                )
                return CLIPProcessor(
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                )
            except Exception:
                raise base_exc

    processor = None
    for candidate in processor_candidates:
        local_only = Path(candidate).exists()
        try:
            processor = _load_processor(candidate, local_only=local_only)
            logger.info(
                f"Loaded CLIP processor from {candidate} (local_only={local_only})"
            )
            break
        except Exception as exc:  # pragma: no cover - runtime safeguard
            processor_exceptions.append(exc)
            logger.warning(
                f"Failed to load CLIP processor from {candidate} (local_only={local_only}): {exc}"
            )
            continue

    if processor is None:  # pragma: no cover - defensive fallback
        raise RuntimeError(
            f"Failed to load CLIP processor for {model_name}: {processor_exceptions[-1]}"
        ) from processor_exceptions[-1]

    if random_init:
        # load architecture config but no weights
        config = CLIPConfig.from_pretrained(model_name)
        model = CLIPModel(config)
        logger.info(f"Initializing CLIP from config ({model_name}) with random weights")
    else:
        model = CLIPModel.from_pretrained(model_name)
        logger.info(f"Loaded pretrained CLIP weights from {model_name}")
    model.to(device)
    logger.info(f"Number of parameters of CLIP: {count_trainable_parameters(model):,}")

    # if we'll be using SigLIP, add a learnable bias term (initialized to paper's b=–10)
    # this will automatically be picked up by the optimizer
    if loss_type == "siglip":
        model.logit_bias = torch.nn.Parameter(torch.tensor(-10.0, device=device))

    # Wrap in DDP for distributed training BEFORE compiling
    # This ensures proper gradient synchronization across GPUs
    if torch.distributed.is_initialized():
        model = DDP(model, device_ids=[rank], output_device=rank)
        logger.info(f"Wrapped model in DDP on rank {rank}")

    # Compile model for better performance and memory efficiency (PyTorch 2.0+)
    # Note: Compilation happens AFTER DDP wrapping for proper distributed training
    try:
        logger.info("Compiling model with torch.compile() for optimized performance...")
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        logger.warning(f"torch.compile() failed: {e}. Continuing without compilation.")

    return model, processor


def create_optimizer_and_scheduler(args, model, total_steps):
    # Handle DDP wrapper
    actual_model = model.module if hasattr(model, "module") else model

    # Standard weight decay setup
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [
                p
                for n, p in actual_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in actual_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(param_groups, lr=args.lr)

    # Linear warmup + cosine decay
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return optimizer, scheduler


# ----------------------------------------
# Data Loading
# ----------------------------------------
class ImageTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: CLIPProcessor, transform):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.df.at[idx, "img_path"]
        text = self.df.at[idx, "description"]

        # Handle NaN or None text values
        if pd.isna(text) or text is None:
            logger.warning(f"NaN/None text found at idx {idx} for image: {img_path}")
            logger.warning(f"Row data: {self.df.iloc[idx].to_dict()}")
            text = "a dermatology image"

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Unable to load image: " + str(img_path))
            img = Image.new("RGB", (256, 256))

        if self.transform:
            img = self.transform(img)

        return img, text


def collate_fn(batch, processor, max_length):
    images, texts = zip(*batch)

    # Process batch
    inputs = processor(
        images=list(images),
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return inputs


def create_dataloaders(args, processor, rank):
    # Load and shuffle data
    df = (
        pd.read_csv(args.data_csv)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )

    # Filter out samples with NaN descriptions
    original_size = len(df)
    df = df[df["description"].notna()].reset_index(drop=True)
    filtered_count = original_size - len(df)
    if filtered_count > 0 and rank == 0:
        logger.warning(
            f"Filtered out {filtered_count} samples with NaN descriptions. "
            f"Training on {len(df)} samples."
        )

    # Ensure we have the required columns for visualization
    if args.dataset_col not in df.columns:
        logger.warning(f"Column '{args.dataset_col}' not found, creating dummy labels")
        df[args.dataset_col] = "unknown"

    # CRITICAL: Filter out hold-out datasets (PAD-UFES-20 and DDI by default)
    # These datasets are reserved for evaluation and should NEVER be used for training
    original_size = len(df)
    holdout_datasets = args.holdout_datasets

    if holdout_datasets:
        # Filter out any rows containing the holdout dataset names
        mask = ~df[args.dataset_col].str.contains(
            "|".join(holdout_datasets), case=False, na=False
        )
        df = df[mask].reset_index(drop=True)

        filtered_count = original_size - len(df)
        if filtered_count > 0:
            logger.warning(
                f"Filtered out {filtered_count} samples from hold-out datasets "
                f"({', '.join(holdout_datasets)}). Training on {len(df)} samples."
            )
        else:
            logger.info(
                f"No hold-out datasets ({', '.join(holdout_datasets)}) found. "
                f"Training on all {len(df)} samples."
            )
    else:
        logger.info(
            f"No hold-out datasets specified. Training on all {len(df)} samples."
        )

    # Train/val split
    val_size = int(len(df) * args.val_frac)
    val_df = df.iloc[:val_size] if val_size > 0 else None
    train_df = df.iloc[val_size:]

    # Safety check: Verify no holdout datasets leaked into training set
    if holdout_datasets and rank == 0:
        for dataset_name in holdout_datasets:
            train_count = (
                train_df[args.dataset_col]
                .str.contains(dataset_name, case=False, na=False)
                .sum()
            )
            if train_count > 0:
                raise ValueError(
                    f"CRITICAL ERROR: Found {train_count} samples from hold-out dataset "
                    f"'{dataset_name}' in training set! This should never happen."
                )
        logger.info("✓ Verified: No hold-out datasets in training set")

    # Get image size from processor
    img_size = processor.feature_extractor.size["shortest_edge"]

    # Simple augmentations
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
        ]
    )

    # Create datasets
    train_dataset = ImageTextDataset(train_df, processor, train_transform)
    val_dataset = (
        ImageTextDataset(val_df, processor, val_transform)
        if val_df is not None
        else None
    )

    # Create samplers
    train_sampler = (
        DistributedSampler(train_dataset)
        if torch.distributed.is_initialized()
        else None
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, processor, args.max_length),
    )

    val_loader = None
    if val_dataset is not None and rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda b: collate_fn(b, processor, args.max_length),
        )

    return train_loader, val_loader, len(train_dataset)


# ----------------------------------------
# Training & Validation
# ----------------------------------------
def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor across processes and return mean"""
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= torch.distributed.get_world_size()
    return tensor


def train_epoch(epoch, args, model, optimizer, scheduler, loader, device, scaler, rank):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # set epoch for distributed sampler
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    iterator = tqdm(loader, desc=f"Epoch {epoch}") if rank == 0 else loader
    for step, batch in enumerate(iterator):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            # get model (handle DDP wrapper)
            actual_model = model.module if hasattr(model, "module") else model
            image_embeds = actual_model.get_image_features(
                pixel_values=pixel_values,
            )
            text_embeds = actual_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Gather embeddings from all GPUs for distributed training
            # This is crucial for contrastive learning across the full batch
            if torch.distributed.is_initialized():
                # Gather all embeddings
                all_image_embeds = gather_embeddings(image_embeds)
                all_text_embeds = gather_embeddings(text_embeds)

                # Calculate the rank offset for proper label assignment
                local_batch_size = pixel_values.size(0)
                rank = torch.distributed.get_rank()
                labels_offset = rank * local_batch_size
            else:
                all_image_embeds = image_embeds
                all_text_embeds = text_embeds
                local_batch_size = pixel_values.size(0)
                labels_offset = 0

            # compute logits
            logit_scale = actual_model.logit_scale.exp()
            logits_per_image = logit_scale * image_embeds @ all_text_embeds.t()
            logits_per_text = logit_scale * text_embeds @ all_image_embeds.t()

            # compute loss (either CLIP or SigLIP)
            if args.loss_type == "clip":
                # Labels need to account for the offset in distributed training
                labels = torch.arange(local_batch_size, device=device) + labels_offset
                loss_img = F.cross_entropy(logits_per_image, labels)
                loss_txt = F.cross_entropy(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2
            else:
                # SigLIP: pairwise-sigmoid over all image–text pairs
                # For distributed training, we need to create labels for the full gathered batch
                global_batch_size = all_image_embeds.size(0)
                eye = torch.zeros(local_batch_size, global_batch_size, device=device)
                eye[:, labels_offset : labels_offset + local_batch_size] = torch.eye(
                    local_batch_size, device=device
                )
                pair_labels = eye * 2 - 1  # +1 on diag, -1 off-diag
                # include bias if present
                bias = getattr(actual_model, "logit_bias", 0.0)
                logits = logits_per_image + bias
                # sum over all pairs, normalize by local batch size
                loss = -(F.logsigmoid(pair_labels * logits).sum()) / local_batch_size

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # logging
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "step": epoch * len(loader) + step,
                }
            )
            iterator.set_description(f"Loss: {loss.item():.4f}")

    # Synchronize and log epoch metrics
    avg_loss = total_loss / num_batches
    loss_tensor = torch.tensor(avg_loss, device=device)
    avg_loss = all_reduce_mean(loss_tensor).item()

    if rank == 0:
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch/train_loss": avg_loss, "epoch": epoch})

    return avg_loss


def validate(args, model, loader, device, epoch):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get model (handle DDP wrapper)
            actual_model = model.module if hasattr(model, "module") else model

            # Forward pass
            image_embeds = actual_model.get_image_features(pixel_values=pixel_values)
            text_embeds = actual_model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Compute logits and loss
            logit_scale = actual_model.logit_scale.exp()
            logits_per_image = logit_scale * image_embeds @ text_embeds.t()
            logits_per_text = logit_scale * text_embeds @ image_embeds.t()

            batch_size = pixel_values.size(0)
            labels = torch.arange(batch_size, device=device)

            loss_img = F.cross_entropy(logits_per_image, labels)
            loss_txt = F.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    wandb.log({"epoch/val_loss": avg_loss, "epoch": epoch})

    return avg_loss


# ----------------------------------------
# Embedding & Visualization
# ----------------------------------------
def extract_embeddings(
    model,
    processor,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 512,
    max_samples: Optional[int] = 5000,
    dataset_col: str = "dataset_desc",
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Faster extraction using DataLoader, mixed precision, and image-fault failsafe."""
    # Subsample if needed
    df_sample = (
        df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        if max_samples and len(df) > max_samples
        else df.reset_index(drop=True)
    )

    logger.info(f"Extracting embeddings from {len(df_sample)} samples")
    logger.info(f"Columns available: {list(df_sample.columns)}")
    logger.info(f"Using dataset_col: {dataset_col}")

    # Custom Dataset
    class ClipDataset(torch.utils.data.Dataset):
        def __init__(self, data_frame: pd.DataFrame):
            self.df = data_frame

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            try:
                img = Image.open(row["img_path"]).convert("RGB")
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                text = row[dataset_col]
                if pd.isna(text):
                    logger.warning(
                        f"NaN text found at idx {idx} (img: {row.get('img_path', 'N/A')}), using 'unknown'"
                    )
                    text = "unknown"
            except Exception as e:
                # Return a sentinel for faulty images or text
                logger.error(
                    f"Unable to load sample {idx} (img: {row.get('img_path', 'N/A')}): {e}"
                )
                return None, None, idx
            return img, text, idx

    # Collate function for batching, filters out faulty images
    def collate_fn(batch):
        # Separate valid and faulty entries
        valid = [(img, txt, idx) for img, txt, idx in batch if img is not None]
        faulty = [idx for img, txt, idx in batch if img is None]

        if not valid:
            # All images faulty in this batch
            return None, None, [], faulty

        images, texts, idxs = zip(*valid)
        image_inputs = processor(images=list(images), return_tensors="pt", padding=True)
        text_inputs = processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        )
        return image_inputs, text_inputs, list(idxs), faulty

    # DataLoader for parallel loading
    loader = torch.utils.data.DataLoader(
        ClipDataset(df_sample),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    image_embs, text_embs, labels = [], [], []
    corrupted = []

    # Unwrap DDP if needed
    actual_model = model.module if hasattr(model, "module") else model
    actual_model.to(device)
    actual_model.eval()

    # Use mixed precision for speed
    total_batches = 0
    valid_batches = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in tqdm(loader, total=len(loader)):
            total_batches += 1
            if batch is None:
                logger.debug("Batch is None, skipping")
                continue
            image_inputs, text_inputs, idxs, faulty_idxs = batch
            corrupted.extend(faulty_idxs)
            if not idxs:
                logger.debug("No valid indices in batch, skipping")
                continue
            try:
                # Move inputs to device
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
                labels.extend(idxs)
                valid_batches += 1
            except Exception as e:
                # If inference fails, mark all these indices as corrupted
                logger.warning(f"Inference failed for batch: {e}")
                corrupted.extend(idxs)
                continue

    logger.info(f"Processed {total_batches} batches, {valid_batches} were valid")

    # Concatenate and convert to numpy
    if image_embs:
        image_embeddings = torch.cat(image_embs, dim=0).numpy()
        text_embeddings = torch.cat(text_embs, dim=0).numpy()
    else:
        raise ValueError("All batches were empty, something went wrong!")
    return image_embeddings, text_embeddings, np.array(labels), corrupted


def create_embedding_visualization(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    dataset_labels: np.ndarray,
    output_dir: str,
    args,
    run_name: Optional[str] = None,
):
    """Create UMAP visualization of embeddings colored by dataset"""
    if run_name is None:
        run_name = wandb.run.name
    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        # Use single thread for reproducibility
        random_state=42,
        n_jobs=1,
    )

    # Plot 1: Color by dataset
    # fit only on the image data (since we're only interested in this)
    embedding_2d = reducer.fit_transform(image_embeddings)
    counts = pd.Series(dataset_labels).value_counts()
    classes_sorted = counts.index.tolist()

    plt.figure(figsize=(8, 6))
    colors = glasbey.create_palette(palette_size=len(classes_sorted))
    for i, dataset in enumerate(classes_sorted):
        mask = dataset_labels == dataset
        dataset_color = [colors[list(classes_sorted).index(dataset)]]
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=dataset_color,
            edgecolors=dataset_color,
            linewidths=0.6,
            label=dataset,
            alpha=0.6,
            s=10,
        )

    plt.title("Embeddings Colored by Dataset")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    vis_path = os.path.join(output_dir, run_name, "embedding_by_dataset.png")
    plt.savefig(vis_path, dpi=300, bbox_inches="tight")
    plt.close()

    if wandb.run is not None:
        wandb.log({"SkinMap/EmbeddingsByDataset": wandb.Image(vis_path)})

    # Log the same but interactive with Plotly
    log_umap_plotly(embedding_2d, dataset_labels)

    # Plot 2: Color by modality (Image vs Text)
    # create the combined embeddings (image+text) and refit
    all_embeddings = np.vstack([image_embeddings, text_embeddings])
    modality_labels = np.array(
        ["Image"] * len(image_embeddings) + ["Text"] * len(text_embeddings)
    )
    embedding_2d = reducer.fit_transform(all_embeddings)

    plt.figure(figsize=(8, 6))
    image_mask = modality_labels == "Image"
    plt.scatter(
        embedding_2d[image_mask, 0],
        embedding_2d[image_mask, 1],
        c="blue",
        label="Image",
        alpha=0.6,
        s=20,
    )
    text_mask = modality_labels == "Text"
    plt.scatter(
        embedding_2d[text_mask, 0],
        embedding_2d[text_mask, 1],
        c="red",
        label="Text",
        alpha=0.6,
        s=20,
    )

    plt.title("Embeddings Colored by Modality")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()

    plt.tight_layout()
    vis_path = os.path.join(output_dir, run_name, "embedding_by_modality.png")
    plt.savefig(vis_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved embedding visualization to {vis_path}")

    if wandb.run is not None:
        wandb.log({"SkinMap/EmbeddingsByModality": wandb.Image(vis_path)})


def log_umap_plotly(
    embedding_2d: np.ndarray,
    dataset_labels: np.ndarray,
):
    df = pd.DataFrame(
        {
            "UMAP 1": embedding_2d[:, 0],
            "UMAP 2": embedding_2d[:, 1],
            "Dataset": dataset_labels,
        }
    )
    counts = df["Dataset"].value_counts()
    category_order = counts.index.tolist()

    fig = px.scatter(
        df,
        x="UMAP 1",
        y="UMAP 2",
        color="Dataset",
        category_orders={"Dataset": category_order},
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data={"Dataset": True},
        opacity=0.6,
        height=600,
        width=800,
        title="UMAP Projection of Image Embeddings by Dataset",
    )
    fig.update_layout(
        template="plotly_white",
        legend_title_text="Dataset",
        legend=dict(
            itemsizing="trace",
            title_font_size=14,
            font_size=12,
            bordercolor="#ccc",
            borderwidth=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=0.3, color="DarkSlateGrey")))

    if wandb.run is not None:
        wandb.log({"SkinMap/EmbeddingsByDatasetInteractive": fig})


def plot_embeddings(
    args,
    model,
    processor,
    device,
    run_name: Optional[str] = None,
):
    try:
        logger.info("Creating embedding visualization...")
        df = pd.read_csv(args.data_csv)
        image_emb, text_emb, label_indices, _ = extract_embeddings(
            model,
            processor,
            df,
            device,
            batch_size=64,
            max_samples=args.vis_samples,
            dataset_col=args.dataset_col,
        )

        # Convert indices back to dataset labels
        df_sample = (
            df.sample(n=args.vis_samples, random_state=42).reset_index(drop=True)
            if args.vis_samples and len(df) > args.vis_samples
            else df.reset_index(drop=True)
        )
        dataset_labels = df_sample.iloc[label_indices][args.dataset_col].values

        create_embedding_visualization(
            image_emb,
            text_emb,
            dataset_labels,
            args.output_dir,
            args,
            run_name=run_name,
        )
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        import traceback

        logger.error(traceback.format_exc())


# ----------------------------------------
# Main Training Loop
# ----------------------------------------
def main():
    args = parse_args()
    setup_logger(args.log_level)
    set_seed(args.seed)

    device, rank, world_size = setup_distributed()
    logger.info(f"Device: {device} | Rank: {rank}/{world_size}")

    if rank == 0:
        wandb.init(project=args.wandb_project, config=vars(args))
        run_name = (
            f"{args.model_name.replace('/', '_')}-{args.loss_type}-{wandb.run.name}"
        )
        if args.random_init:
            run_name += "-random"
        wandb.run.name = run_name

        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, wandb.run.name), exist_ok=True)
        logger.info("Configuration: {}", args)

    try:
        model, processor = load_model_and_processor(
            args.model_name,
            device,
            rank,
            random_init=args.random_init,
            loss_type=args.loss_type,
        )
        train_loader, val_loader, train_size = create_dataloaders(args, processor, rank)

        total_steps = (train_size * args.epochs) // (args.batch_size * world_size)
        optimizer, scheduler = create_optimizer_and_scheduler(args, model, total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

        if rank == 0:
            plot_embeddings(args=args, model=model, processor=processor, device=device)

        for epoch in range(1, args.epochs + 1):
            _ = train_epoch(
                epoch,
                args,
                model,
                optimizer,
                scheduler,
                train_loader,
                device,
                scaler,
                rank,
            )

            if rank == 0:
                if val_loader is not None:
                    _ = validate(args, model, val_loader, device, epoch)

                # Save checkpoint based on frequency
                if epoch % args.checkpoint_frequency == 0:
                    checkpoint_dir = os.path.join(
                        args.output_dir, wandb.run.name, f"epoch_{epoch}"
                    )
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(checkpoint_dir)
                    processor.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

            if rank == 0 and epoch % args.vis_epochs == 0:
                plot_embeddings(
                    args=args, model=model, processor=processor, device=device
                )

        logger.info("Training completed!")
        wandb.finish()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
