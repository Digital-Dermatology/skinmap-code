"""Integration tests that hit real SSL checkpoints (opt-in)."""

import os

import pytest
import torch

from src.skinmap.models.loaders import load_multiple_models


@pytest.mark.integration
def test_dino_qderma_embedding_dim_real():
    """Ensure dino_qderma embeddings retain the concatenated transformer layers."""
    if not os.environ.get("SKINMAP_RUN_SSL_INTEGRATION"):
        pytest.skip(
            "Set SKINMAP_RUN_SSL_INTEGRATION=1 to run SSL integration tests (downloads checkpoints)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_multiple_models("dino_qderma", device)
    assert len(models) == 1
    ssl_model = models[0].model.to(device).eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        emb = ssl_model(dummy, n_layers=1)
        emb_full = ssl_model(dummy, n_layers=4)

    assert emb.shape[0] == 1
    embed_dim = getattr(getattr(ssl_model, "model", ssl_model), "embed_dim", None)
    assert embed_dim is not None, "Underlying ViT missing embed_dim attribute"
    assert emb.shape[1] == embed_dim
    assert emb_full.shape[1] == embed_dim * 4
