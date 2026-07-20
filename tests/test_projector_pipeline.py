from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from src.create_skinmap import persist_artifacts
from src.embedding_fusion import train_one_epoch
from src.skinmap.embeddings.projector import _normalize_teacher_blocks
from src.skinmap.loss.clip_loss import CLIPLoss


def test_normalize_teacher_blocks_l2():
    rng = np.random.default_rng(0)
    # Three samples with two teachers: dims 2 and 3
    z_cat = rng.normal(size=(3, 5)).astype(np.float32)
    dims = [2, 3]

    normalized = _normalize_teacher_blocks(z_cat.copy(), dims)

    start = 0
    for dim in dims:
        block = normalized[:, start : start + dim]
        norms = np.linalg.norm(block, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        start += dim


def test_persist_artifacts_writes_whitening_arrays(tmp_path):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()

    projector_model = nn.Linear(3, 2, bias=False)
    whitening_stats = {
        "mu": [np.zeros((1, 3), dtype=np.float32)],
        "W": [np.eye(3, dtype=np.float32)],
        "dims": [3],
        "clip_indices": [0],
        "text_whitening": {
            "mu": [np.zeros((1, 2), dtype=np.float32)],
            "W": [np.eye(2, dtype=np.float32)],
            "dims": [2],
        },
    }

    artifacts = {"projector_model": projector_model, "whitening_stats": whitening_stats}
    args = SimpleNamespace(projector_type="linear", projector_dim=2)

    persist_artifacts(
        emb_dir=str(emb_dir),
        artifacts=artifacts,
        args=args,
        model_names=["modelA"],
        text_embeddings=np.zeros((10, 2), dtype=np.float32),
    )

    whitening_path = emb_dir / "whitening_stats.npz"
    assert whitening_path.exists()

    with np.load(whitening_path, allow_pickle=True) as data:
        assert "mu" in data.files
        assert "W" in data.files
        assert "text_mu" in data.files
        assert "text_W" in data.files


class _DummyLoader:
    """Minimal loader yielding one batch with precomputed tensors."""

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1


def _build_dummy_batch(batch_size=4, img_dim=6, txt_dim=6):
    z_cat = torch.randn(batch_size, img_dim, dtype=torch.float32)
    t_vec = torch.randn(batch_size, txt_dim, dtype=torch.float32)
    batch = {"z_cat": z_cat, "t_vec": t_vec, "source": ["a"] * batch_size}

    class _DummyModel(nn.Module):
        def __init__(self, img_dim, txt_dim):
            super().__init__()
            self.img_head = nn.Linear(img_dim, img_dim, bias=False)
            self.txt_head = nn.Linear(txt_dim, txt_dim, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, z_cat, t_vec):
            return self.img_head(z_cat), self.txt_head(t_vec), self.logit_scale.exp()

    model = _DummyModel(img_dim, txt_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, optimizer, _DummyLoader(batch)


def test_train_one_epoch_clip_loss():
    model, optimizer, loader = _build_dummy_batch()
    avg_loss, aux = train_one_epoch(model, loader, optimizer, device="cpu")
    assert avg_loss > 0
    assert aux == {}


def test_train_one_epoch_structure_loss():
    model, optimizer, loader = _build_dummy_batch()
    structure_loss = CLIPLoss(
        temperature=0.07,
        normalize_latents=True,
        warmup_steps=0,
        lambda_hierarchy=1.0,
        hierarchy_levels=1,
        hierarchy_weighting="none",
        hierarchy_margin=0.0,
        lambda_consistency=0.5,
    )
    avg_loss, aux = train_one_epoch(
        model, loader, optimizer, device="cpu", structure_loss=structure_loss
    )
    assert avg_loss > 0
    assert "clip_loss" in aux or "hierarchy_loss" in aux or "consistency_loss" in aux
