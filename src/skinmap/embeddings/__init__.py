"""Embedding extraction and fusion utilities."""

from .extractors import extract_clip_embeddings, extract_ssl_embeddings
from .fusion import combine_embeddings_simple
from .projector import extract_combined_embeddings_with_projector

__all__ = [
    "combine_embeddings_simple",
    "extract_clip_embeddings",
    "extract_combined_embeddings_with_projector",
    "extract_ssl_embeddings",
]
