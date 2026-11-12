"""The embedding pipeline must find its teachers inside a Hugging Face cache snapshot.

In the cache every file is a symlink into ``blobs/``. Following those links moved
the config dir into ``blobs/`` and renamed teacher checkpoints to hashes, so the
published SkinMap model failed to load from ``from_pretrained``.
"""

import json

from skinmap.pipeline.combined_embedder import CombinedEmbeddingPipeline


def _hf_style_snapshot(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    weights = tmp_path / "snapshots" / "abc" / "weights"
    (weights / "ssl_teachers").mkdir(parents=True)
    (blobs / "cfg").write_text(json.dumps({"models": []}))
    (blobs / "dino").write_bytes(b"")
    (weights / "embedding_pipeline_config.json").symlink_to(blobs / "cfg")
    (weights / "ssl_teachers" / "dino_vit-base-patch16.pth").symlink_to(blobs / "dino")
    return weights


def test_teacher_paths_stay_inside_the_snapshot(tmp_path, monkeypatch):
    weights = _hf_style_snapshot(tmp_path)
    monkeypatch.chdir(tmp_path)  # relative sources must not depend on the cwd
    monkeypatch.setattr(CombinedEmbeddingPipeline, "_load_models", lambda self: None)  # paths only, no weights
    pipe = CombinedEmbeddingPipeline.from_config(weights / "embedding_pipeline_config.json")
    assert pipe.config_dir == weights
    found = pipe._resolve_model_path("ssl_teachers/dino_vit-base-patch16.pth")
    assert found == weights / "ssl_teachers" / "dino_vit-base-patch16.pth"
    assert found.stem == "dino_vit-base-patch16"  # the loader picks the model by this name
