"""Model loading utilities for CLIP and SSL models."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from loguru import logger

# SSL model names
SSL_MODEL_NAMES = [
    "dino_qderma",
    "ibot_qderma",
    "mae_qderma",
    "simclr_qderma",
    "byol_qderma",
    "colorme_qderma",
    "panderm_base",
    "panderm_large",
]


@dataclass
class ModelInfo:
    """Container for model information."""

    model: Any
    processor: Optional[Any]
    model_type: str  # 'clip' or 'ssl'
    model_path: str  # Original path/name

    def to_tuple(self) -> Tuple[Any, Optional[Any], str]:
        """Convert to tuple format for backward compatibility."""
        return (self.model, self.processor, self.model_type)


def normalize_model_tuple(model_info) -> Tuple[Any, Optional[Any], str]:
    """Normalize model info to (model, processor, model_type) format.

    Args:
        model_info: Either (model, processor) or (model, processor, model_type) or ModelInfo

    Returns:
        Tuple of (model, processor, model_type) where model_type defaults to 'clip'
    """
    if isinstance(model_info, ModelInfo):
        return model_info.to_tuple()

    if len(model_info) == 3:
        return model_info
    else:
        model, processor = model_info
        return model, processor, "clip"


def load_multiple_models(
    model_names_or_paths: str,
    device: torch.device,
) -> List[ModelInfo]:
    """Load multiple models (CLIP and/or SSL models).

    Args:
        model_names_or_paths: Comma-separated string of model names/paths
        device: torch device

    Returns:
        List of ModelInfo objects

    Raises:
        RuntimeError: If a model fails to load
    """
    # Import here to avoid circular dependencies
    from src.core.src.pkg.embedder import Embedder
    from src.train_clip import load_model_and_processor

    models = []
    model_list = [name.strip() for name in model_names_or_paths.split(",")]

    for model_source in model_list:
        logger.info(f"Loading model: {model_source}")

        # Check if this is an SSL model
        if model_source in SSL_MODEL_NAMES:
            logger.info(f"Detected SSL model: {model_source}")
            ssl_model, info, config = Embedder.load_pretrained(
                model_source,
                return_info=True,
                n_head_layers=0,
            )
            ssl_model.to(device)
            models.append(
                ModelInfo(
                    model=ssl_model,
                    processor=None,
                    model_type="ssl",
                    model_path=model_source,
                )
            )
        else:
            # Load as CLIP model
            try:
                model, processor = load_model_and_processor(model_source, device)
                models.append(
                    ModelInfo(
                        model=model,
                        processor=processor,
                        model_type="clip",
                        model_path=model_source,
                    )
                )
            except Exception as e:
                # Handle local checkpoints
                if model_source.startswith("assets/"):
                    model, processor = _load_local_checkpoint(model_source, device)
                    models.append(
                        ModelInfo(
                            model=model,
                            processor=processor,
                            model_type="clip",
                            model_path=model_source,
                        )
                    )
                else:
                    logger.error(f"Failed to load model {model_source}: {e}")
                    raise RuntimeError(f"Failed to load model {model_source}") from e

    logger.info(f"Successfully loaded {len(models)} models")
    return models


def _load_local_checkpoint(model_source: str, device: torch.device) -> Tuple[Any, Any]:
    """Load a local CLIP checkpoint with inferred processor.

    Args:
        model_source: Path to local checkpoint
        device: torch device

    Returns:
        Tuple of (model, processor)

    Raises:
        ValueError: If original model name cannot be inferred
    """
    from transformers import CLIPModel, CLIPProcessor

    # Infer original model name from path
    original_model_name = None
    if "suinleelab_monet" in model_source:
        original_model_name = "suinleelab/monet"
    elif "openai_clip-vit-base-patch32" in model_source:
        original_model_name = "openai/clip-vit-base-patch32"
    elif "openai_clip-vit-large-patch14" in model_source:
        original_model_name = "openai/clip-vit-large-patch14"

    if not original_model_name:
        raise ValueError(
            f"Could not infer original model name for local checkpoint: {model_source}"
        )

    logger.info(
        f"Loading processor from {original_model_name} for checkpoint {model_source}"
    )

    processor = CLIPProcessor.from_pretrained(original_model_name)
    model = CLIPModel.from_pretrained(model_source)
    model.to(device)

    return model, processor
