"""SkinMap: Modular framework for creating medical image atlases with multi-model embeddings."""

import importlib

__version__ = "2.0.0"

# Each subpackage loads on first access (PEP 562), so `import skinmap` does not pull in
# UMAP, FAISS or scikit-learn.
__all__ = [
    "cache",
    "data",
    "embeddings",
    "evaluation",
    "models",
    "utils",
    "visualization",
]


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # cache so subsequent access skips __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
