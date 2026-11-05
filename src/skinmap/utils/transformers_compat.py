"""Compatibility helpers for the CLIP feature API across ``transformers`` majors.

``transformers >= 5`` changed ``CLIPModel.get_image_features`` /
``get_text_features`` to return a ``BaseModelOutputWithPooling`` output object
instead of a bare tensor. This module centralises the unwrap so every call site
stays version-agnostic and the logic lives in exactly one place.

Historically this shim was copy-pasted into three modules and *omitted* at
several call sites (``downstream.py``, ``build_quality_anchors.py``,
``train_clip.py``), which crashed under the ``transformers 5.3.0`` shipped by
``Dockerfile.train`` -- route every ``get_*_features`` result through here.
"""

from __future__ import annotations

import torch


def coerce_clip_features(feats, embeds_attr: str):
    """Return the projected CLIP feature tensor from a ``get_*_features`` call.

    Args:
        feats: the value returned by ``model.get_image_features`` /
            ``get_text_features``. Under ``transformers < 5`` this is already a
            tensor; under ``>= 5`` it is a ``BaseModelOutputWithPooling`` whose
            projected embedding lives in ``embeds_attr``
            (``image_embeds`` / ``text_embeds``) or in ``pooler_output``.
        embeds_attr: ``"image_embeds"`` or ``"text_embeds"``.

    Note:
        ``last_hidden_state`` is intentionally never used as a fallback: its CLS
        token has the encoder hidden dim, not the projection dim, and would
        silently corrupt the embedding space.
    """
    if torch.is_tensor(feats):
        return feats
    for attr in (embeds_attr, "pooler_output"):
        val = getattr(feats, attr, None)
        if val is not None:
            return val
    kind = embeds_attr.split("_")[0]
    raise TypeError(
        f"get_{kind}_features returned {type(feats).__name__} with neither "
        f"'{embeds_attr}' nor 'pooler_output'; cannot extract the projected "
        f"embedding (unsupported transformers version?)."
    )


def coerce_image_features(feats):
    """Convenience wrapper: ``coerce_clip_features(feats, "image_embeds")``."""
    return coerce_clip_features(feats, "image_embeds")


def coerce_text_features(feats):
    """Convenience wrapper: ``coerce_clip_features(feats, "text_embeds")``."""
    return coerce_clip_features(feats, "text_embeds")
