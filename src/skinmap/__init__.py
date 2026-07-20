"""SkinMap: Modular framework for creating medical image atlases with multi-model embeddings."""

__version__ = "2.0.0"

from . import cache, data, embeddings, evaluation, models, utils, visualization

__all__ = [
    "cache",
    "data",
    "embeddings",
    "evaluation",
    "models",
    "utils",
    "visualization",
]
