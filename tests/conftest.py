"""Shared pytest fixtures for SkinMap."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

# Ensure numba/umap don't attempt to cache into read-only site-packages during tests
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
NUMBA_CACHE_DIR = Path(__file__).resolve().parent / ".numba_cache"
NUMBA_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))


@pytest.fixture
def device():
    """CPU device fixture reused across tests."""
    return torch.device("cpu")


class DummyClipModel(torch.nn.Module):
    """Lightweight CLIP-like model for tests."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.logit_scale = torch.nn.Parameter(torch.ones([]))

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_image_features(self, pixel_values: torch.Tensor):
        batch = pixel_values.size(0)
        return torch.ones(batch, self.embed_dim)

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch = input_ids.size(0)
        return torch.ones(batch, self.embed_dim)


class DummySSLModel(torch.nn.Module):
    """Simple SSL model stub returning normalized embeddings."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor):
        batch = images.size(0)
        features = torch.arange(self.embed_dim, dtype=torch.float32)
        features = features / (features.norm() + 1e-6)
        return features.expand(batch, -1)

    __call__ = forward


class DummyProcessor:
    """Minimal CLIP processor stand-in."""

    def __call__(
        self,
        images=None,
        text=None,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ):
        if images is not None:
            batch = len(images)
            tensor = torch.randn(batch, 3, 224, 224)
            return {"pixel_values": tensor}
        if text is not None:
            batch = len(text)
            return {
                "input_ids": torch.ones(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }
        raise ValueError("Either images or text must be provided")


@pytest.fixture
def mock_clip_model():
    return DummyClipModel()


@pytest.fixture
def mock_ssl_model():
    return DummySSLModel()


@pytest.fixture
def mock_processor():
    return DummyProcessor()


@pytest.fixture
def temp_dir(tmp_path: Path) -> str:
    """Return temporary directory path as string."""
    return str(tmp_path)


@pytest.fixture
def sample_df(tmp_path: Path) -> pd.DataFrame:
    """Create a sample dataframe with valid image paths."""
    records = []
    for i in range(10):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 20 % 255, 0, 0)).save(img_path)
        records.append(
            {
                "img_path": str(img_path),
                "description": f"Image {i}",
                "dataset_desc": "dataset_a" if i % 2 == 0 else "dataset_b",
            }
        )
    return pd.DataFrame.from_records(records)


@pytest.fixture
def sample_df_with_corrupted(tmp_path: Path) -> Tuple[pd.DataFrame, int]:
    """Sample dataframe with a mix of valid and invalid paths."""
    valid_path = tmp_path / "valid.jpg"
    Image.new("RGB", (32, 32), color="red").save(valid_path)

    df = pd.DataFrame(
        {
            "img_path": [
                str(valid_path),
                "/nonexistent/corrupt1.jpg",
                str(valid_path),
                "/nonexistent/corrupt2.jpg",
            ],
            "description": ["valid", "bad1", "valid2", "bad2"],
            "dataset_desc": ["setA"] * 4,
        }
    )
    return df, 2


@pytest.fixture
def sample_embeddings():
    """Return deterministic embeddings + labels for cache tests."""
    rng = np.random.default_rng(0)
    num = 8
    image = rng.standard_normal((num, 16)).astype(np.float32)
    text = rng.standard_normal((num, 16)).astype(np.float32)
    labels = np.arange(num, dtype=np.int64)
    return image, text, labels


@pytest.fixture
def multilabel_df(tmp_path: Path) -> pd.DataFrame:
    """Create dataframe with heterogeneous multilabel column values."""
    img_paths = []
    for i in range(5):
        img_path = tmp_path / f"ml_{i}.jpg"
        Image.new("RGB", (16, 16), color=(i * 40 % 255, 0, 0)).save(img_path)
        img_paths.append(str(img_path))

    return pd.DataFrame(
        {
            "img_path": img_paths,
            "origin": [
                "['Europe', 'Asia']",
                "Africa, Europe",
                ["North America", "South America"],
                None,
                " [ 'Oceania' ] ",
            ],
        }
    )


@pytest.fixture(autouse=True)
def patch_model_loading(monkeypatch, mock_clip_model, mock_processor, mock_ssl_model):
    """Autouse fixture to patch heavy model loading with lightweight stubs."""
    if os.environ.get("SKINMAP_RUN_SSL_INTEGRATION"):
        # Allow real checkpoints to be loaded for integration runs
        yield
        return

    def _mock_loader(model_name, device, model_path=None):
        return mock_clip_model, mock_processor

    import src.train_clip

    monkeypatch.setattr(src.train_clip, "load_model_and_processor", _mock_loader)

    def _mock_load_pretrained(*args, **kwargs):
        return mock_ssl_model, SimpleNamespace(), {}

    import src.core.src.pkg.embedder as embedder_module

    monkeypatch.setattr(
        embedder_module.Embedder, "load_pretrained", _mock_load_pretrained
    )
    yield
