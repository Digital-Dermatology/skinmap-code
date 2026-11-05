"""The package installs and its entry points start."""

import subprocess
import sys

import pytest

import skinmap


def test_version():
    assert skinmap.__version__


def test_import_skinmap_loads_no_heavy_dependency():
    code = "import sys, skinmap; print('umap' in sys.modules or 'faiss' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("name", ["create_skinmap", "run_analysis", "train_clip"])
def test_pipeline_entry_points_start(name):
    out = subprocess.run(
        [sys.executable, "-m", f"skinmap.pipeline.{name}", "--help"], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
