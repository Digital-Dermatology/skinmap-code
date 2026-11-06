import io
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from PIL import Image
from torch import nn

from core.src.pkg.embedder import Embedder
from skinmap.pipeline.create_skinmap import SSL_MODEL_NAMES, get_imagenet_transform
from skinmap.pipeline.train_clip import load_model_and_processor

try:
    import joblib
except ImportError:  # pragma: no cover - fallback for older sklearn versions
    from sklearn.externals import joblib  # type: ignore


logger = logging.getLogger(__name__)


def _coerce_clip_features(feats, embeds_attr: str):
    """Return the projected feature tensor from a CLIP ``get_*_features`` call.

    Some transformers versions return a ``BaseModelOutputWithPooling`` rather
    than a tensor. In that case the projected embedding lives in the ``*_embeds``
    field, or in ``pooler_output`` (which holds the projected vector for the
    shipped CLIP checkpoints). ``last_hidden_state`` is not used as a fallback:
    its CLS token has the encoder hidden dim, not the projection dim, so it would
    silently corrupt the embedding. Shared by the image and text paths.
    """
    if isinstance(feats, torch.Tensor):
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


@dataclass
class ClipModelWrapper:
    name: str
    model: nn.Module
    processor: Any  # CLIPProcessor

    def embed(self, image: Image.Image, device: torch.device) -> np.ndarray:
        prepared = image.copy().convert("RGB")
        prepared.thumbnail((512, 512), Image.Resampling.LANCZOS)
        inputs = self.processor(images=[prepared], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.model.eval()
        with torch.inference_mode():
            feats = _coerce_clip_features(
                self.model.get_image_features(**inputs), "image_embeds"
            )
            # Normalize to match training pipeline
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.detach().cpu().numpy()


@dataclass
class SSLModelWrapper:
    name: str
    model: nn.Module
    transform: Any  # torchvision.transforms.Compose

    def embed(self, image: Image.Image, device: torch.device) -> np.ndarray:
        prepared = image.copy().convert("RGB")
        tensor = self.transform(prepared).unsqueeze(0).to(device)
        self.model.eval()
        with torch.inference_mode():
            feats = self.model(tensor)
            feats = torch.nn.functional.normalize(feats, dim=-1, p=2)
        return feats.detach().cpu().numpy()


class CombinedEmbeddingPipeline:
    """
    Runtime helper that mirrors the embedding combination used to build SkinMap.

    It loads the individual models, the saved SVD projection (if present),
    and the FAISS index so that uploaded images can be embedded and searched
    in the same latent space.
    """

    def __init__(self, config: dict, config_path: Path, device: Optional[str] = None):
        self.config = config
        self.config_path = config_path.resolve()
        self.config_dir = self.config_path.parent.resolve()
        self.device = torch.device(device or "cpu")
        self._faiss_lock = threading.Lock()

        self.skinmap_root_path = self._resolve_skinmap_root(config.get("skinmap_root"))
        self.output_dir = self._resolve_output_dir(config.get("output_dir"))
        self._model_search_roots = self._build_model_search_roots()

        self.clip_models: List[ClipModelWrapper] = []
        self.ssl_models: List[SSLModelWrapper] = []
        self.models_in_order: List[Tuple[str, Any]] = (
            []
        )  # (type, wrapper) preserving config order

        self._load_models()

        # Load SVD (legacy pipeline) or projector (new pipeline)
        self.svd_image = self._load_joblib(config.get("svd", {}).get("image"))
        self.projector_model = None
        self.whitening_stats = None
        self._load_projector_if_available()

        artifacts = config.get("artifacts", {})
        self._embeddings_npz_path = self._resolve_artifact(
            artifacts.get("embeddings_npz")
        )
        self._faiss_index_path = self._resolve_artifact(artifacts.get("faiss_index"))
        self.faiss_index = self._load_faiss(self._faiss_index_path)
        self.vector_dim = config.get("vector", {}).get("dimension")
        self.index_metric = "cosine"  # Matches build_faiss_index default
        self.umap_model = self._load_umap(config.get("umap", {}).get("model"))

    @classmethod
    def from_config(cls, path: str | os.PathLike, device: Optional[str] = None):
        path = Path(path)
        with open(path, "r") as f:
            config = json.load(f)
        return cls(config=config, config_path=path, device=device)

    def _resolve_artifact(self, rel_path: Optional[str]) -> Optional[Path]:
        if rel_path is None:
            return None
        return (self.output_dir / rel_path).resolve()

    def _resolve_skinmap_root(self, raw_root: Optional[str]) -> Optional[Path]:
        candidate_roots: List[Path] = []

        def _add_candidate(path_like: Optional[str | os.PathLike]):
            if not path_like:
                return
            path_obj = Path(path_like)
            if not path_obj.is_absolute():
                path_obj = (self.config_dir / path_obj).resolve()
            else:
                path_obj = path_obj.resolve()
            candidate_roots.append(path_obj)

        if raw_root:
            _add_candidate(raw_root)
            raw_path = Path(raw_root)
            if not raw_path.is_absolute():
                _add_candidate(self.config_dir.parent / raw_path)

        # Walk up the config directory hierarchy to find the project root.
        for ancestor in self.config_dir.parents:
            candidate_roots.append(ancestor.resolve())

        seen: set[str] = set()
        for candidate in candidate_roots:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if not candidate.exists():
                continue
            # The project root holds the assets folder.
            if (candidate / "assets").is_dir():
                return candidate

        logger.warning(
            "Unable to resolve SkinMap project root from config %s; "
            "nearest-neighbor upload search may fail.",
            self.config_path,
        )
        return None

    def _resolve_with_bases(
        self,
        path_value: str | os.PathLike,
        bases: Iterable[Optional[Path]],
        *,
        must_exist: bool = False,
    ) -> Optional[Path]:
        path_obj = Path(path_value)
        candidates: List[Path] = []

        if path_obj.is_absolute():
            candidates.append(path_obj)
        else:
            candidates.append((Path.cwd() / path_obj).resolve())
            for base in bases:
                if base is None:
                    continue
                candidates.append((base / path_obj).resolve())

        for candidate in candidates:
            if candidate.exists():
                return candidate
        if must_exist:
            return None
        return candidates[0] if candidates else None

    def _resolve_output_dir(self, raw_output: Optional[str]) -> Path:
        if not raw_output:
            return self.config_dir

        bases: List[Optional[Path]] = [
            self.skinmap_root_path,
            self.config_dir,
            (
                self.config_dir.parent
                if self.config_dir.parent != self.config_dir
                else None
            ),
        ]
        resolved = self._resolve_with_bases(raw_output, bases, must_exist=True)
        if resolved is None or not resolved.exists():
            logger.warning(
                "Configured output_dir '%s' could not be resolved relative to %s; "
                "falling back to %s",
                raw_output,
                self.config_path,
                self.config_dir,
            )
            return self.config_dir
        return resolved

    def _build_model_search_roots(self) -> List[Path]:
        roots: List[Path] = []
        for candidate in (
            self.output_dir,
            self.output_dir.parent,
            self.skinmap_root_path,
            self.skinmap_root_path / "assets" if self.skinmap_root_path else None,
            self.config_dir,
        ):
            if candidate is None:
                continue
            resolved = candidate.resolve()
            if resolved not in roots:
                roots.append(resolved)
        return roots

    def _resolve_model_path(self, source: str) -> Optional[Path]:
        resolved = self._resolve_with_bases(
            source, self._model_search_roots, must_exist=False
        )
        if resolved is not None and resolved.exists():
            return resolved
        return None

    def _load_models(self):
        models_config: Sequence[dict] = self.config.get("models", [])
        if not models_config:
            raise ValueError("No model definitions found in configuration.")

        ssl_transform = get_imagenet_transform()
        for entry in models_config:
            source = entry.get("source")
            if source is None:
                continue
            resolved_path = self._resolve_model_path(source)
            resolved_source = str(resolved_path) if resolved_path else source
            model_type = entry.get("type") or (
                "ssl" if source in SSL_MODEL_NAMES else "clip"
            )
            if model_type == "ssl":
                load_target = (
                    resolved_source
                    if resolved_path and entry.get("is_local")
                    else source
                )
                model, _info, _cfg = Embedder.load_pretrained(
                    load_target,
                    return_info=True,
                    n_head_layers=0,
                )
                model.to(self.device)
                wrapper = SSLModelWrapper(
                    name=source, model=model, transform=ssl_transform
                )
                self.ssl_models.append(wrapper)
                self.models_in_order.append(("ssl", wrapper))
            else:
                model, processor = load_model_and_processor(
                    resolved_source, self.device
                )
                wrapper = ClipModelWrapper(
                    name=source, model=model, processor=processor
                )
                self.clip_models.append(wrapper)
                self.models_in_order.append(("clip", wrapper))

    def _load_joblib(self, rel_path: Optional[str]):
        resolved = self._resolve_artifact(rel_path)
        if resolved is None or not resolved.exists():
            return None
        try:
            logger.info("Loading model from %s", resolved)
            return joblib.load(resolved)
        except Exception as exc:
            logger.warning("Failed to load model from %s: %s", resolved, exc)
            return None

    def _load_umap(self, rel_path: Optional[str]):
        model = self._load_joblib(rel_path)
        if model is None:
            # Retry with NNDescent monkey-patch for pynndescent version mismatch.
            # Newer pynndescent adds a `quantization` attribute that older versions
            # lack, causing joblib.load to fail.
            resolved = self._resolve_artifact(rel_path)
            if resolved is not None and resolved.exists():
                try:
                    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
                    import pynndescent

                    NNDescent = pynndescent.NNDescent
                    if not hasattr(NNDescent, "quantization"):
                        NNDescent.quantization = None  # type: ignore[attr-defined]
                        logger.info(
                            "Patched NNDescent with missing 'quantization' attribute"
                        )
                    if not hasattr(NNDescent, "_min_distance"):
                        NNDescent._min_distance = None  # type: ignore[attr-defined]
                        logger.info(
                            "Patched NNDescent with missing '_min_distance' attribute"
                        )
                    model = joblib.load(resolved)
                except Exception as exc:
                    logger.warning(
                        "Failed to load UMAP model even after NNDescent patch: %s", exc
                    )
                    model = None
        if model is not None:
            logger.info("UMAP model loaded for upload projections")
        return model

    def _load_projector_if_available(self):
        """Load trained projector model if available in config."""
        projector_cfg = self.config.get("projector", {})
        if not projector_cfg.get("used", False):
            logger.info("No trained projector in config; using legacy SVD pipeline")
            return

        # Load projector config
        projector_config_path = self._resolve_artifact(projector_cfg.get("config"))
        if projector_config_path is None or not projector_config_path.exists():
            logger.warning("Projector config not found; falling back to SVD pipeline")
            return

        try:
            with open(projector_config_path, "r") as f:
                proj_config = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to load projector config: {exc}")
            return

        # Load whitening stats
        whitening_path = self._resolve_artifact(projector_cfg.get("whitening_stats"))
        if whitening_path is None or not whitening_path.exists():
            logger.warning("Whitening stats not found; falling back to SVD pipeline")
            return

        try:
            with np.load(whitening_path, allow_pickle=True) as data:
                files = set(data.files)
                dims = list(data["dims"])
                n_models = int(data["n_models"]) if "n_models" in files else len(dims)

                # Load mu/W arrays if they exist (skip_whitening=False case)
                mu_list = []
                W_list = []
                if n_models > 0:
                    if "mu" in files:
                        mu_list = list(data["mu"])
                    else:
                        # Try to load individual mu_i files
                        try:
                            mu_list = [data[f"mu_{i}"] for i in range(n_models)]
                        except KeyError:
                            logger.warning(
                                "No mu whitening parameters found (skip_whitening=True?)"
                            )

                    if "W" in files:
                        W_list = list(data["W"])
                    else:
                        # Try to load individual W_i files
                        try:
                            W_list = [data[f"W_{i}"] for i in range(n_models)]
                        except KeyError:
                            logger.warning(
                                "No W whitening parameters found (skip_whitening=True?)"
                            )

                clip_indices = (
                    list(data["clip_indices"]) if "clip_indices" in files else []
                )

                # Load text whitening if available
                text_mu_list = []
                text_W_list = []
                text_dims = []
                if "text_mu" in files and "text_W" in files:
                    text_mu_list = list(data["text_mu"])
                    text_W_list = list(data["text_W"])
                    text_dims = (
                        list(data["text_dims"])
                        if "text_dims" in files
                        else [w.shape[0] for w in text_W_list]
                    )
                elif "n_text_models" in files:
                    n_text = int(data["n_text_models"])
                    if n_text > 0:
                        try:
                            text_mu_list = [data[f"text_mu_{i}"] for i in range(n_text)]
                            text_W_list = [data[f"text_W_{i}"] for i in range(n_text)]
                            text_dims = (
                                list(data["text_dims"])
                                if "text_dims" in files
                                else [w.shape[0] for w in text_W_list]
                            )
                        except KeyError:
                            logger.warning("No text whitening parameters found")

                self.whitening_stats = {
                    "mu": mu_list,
                    "W": W_list,
                    "dims": dims,
                    "clip_indices": clip_indices,
                }

                if text_mu_list and text_W_list:
                    self.whitening_stats["text_whitening"] = {
                        "mu": text_mu_list,
                        "W": text_W_list,
                        "dims": text_dims,
                    }
                    logger.info(
                        f"Loaded text whitening stats for {len(text_mu_list)} CLIP models"
                    )
                else:
                    self.whitening_stats["text_whitening"] = None

            if len(mu_list) > 0:
                logger.info(
                    f"Loaded whitening stats for {len(self.whitening_stats['mu'])} teachers"
                )
            else:
                logger.info("Whitening was skipped during training (--skip_whitening)")
            logger.info(
                f"CLIP model indices for text: {self.whitening_stats['clip_indices']}"
            )
        except Exception as exc:
            logger.warning(f"Failed to load whitening stats: {exc}")
            return

        # Load projector model
        projector_model_path = self._resolve_artifact(projector_cfg.get("model"))
        if projector_model_path is None or not projector_model_path.exists():
            logger.warning("Projector model not found; falling back to SVD pipeline")
            return

        try:
            from skinmap.pipeline.embedding_fusion import BuildSpec, build_model

            # Calculate actual text input dimension from CLIP models
            # text_dim in config is the OUTPUT dimension, but we need INPUT dimension
            clip_indices = self.whitening_stats.get("clip_indices", [])
            if clip_indices and "text_dims" in self.whitening_stats:
                # Use text_dims from whitening stats (these are the actual input dims)
                text_input_dim = sum(self.whitening_stats["text_dims"])
                logger.info(f"Calculated text input dimension: {text_input_dim}")
            else:
                # Fallback: calculate from teacher_dims at clip_indices
                teacher_dims = proj_config["teacher_dims"]
                text_input_dim = (
                    sum(teacher_dims[i] for i in clip_indices)
                    if clip_indices
                    else teacher_dims[0]
                )
                logger.info(
                    f"Calculated text input dimension from teacher_dims: {text_input_dim}"
                )

            spec = BuildSpec(
                teacher_names=proj_config.get("model_paths", []),
                teacher_dims=proj_config["teacher_dims"],
                text_dim=text_input_dim,  # Use calculated INPUT dimension, not output
                out_dim=proj_config["projector_dim"],
                pca_image=None,
                pca_text=None,
                kind=proj_config["projector_type"],
            )

            self.projector_model = build_model(spec)
            self.projector_model.load_state_dict(
                torch.load(
                    projector_model_path, map_location=self.device, weights_only=False
                )
            )
            self.projector_model.to(self.device)
            self.projector_model.eval()
            logger.info(
                f"Loaded trained projector ({proj_config['projector_type']}, dim={proj_config['projector_dim']})"
            )
        except Exception as exc:
            logger.warning(f"Failed to load projector model: {exc}")
            self.projector_model = None
            self.whitening_stats = None

    def _load_faiss(self, path: Optional[Path]):
        if path is None or not path.exists():
            return None
        try:
            logger.info("Loading FAISS index from %s", path)
            return faiss.read_index(str(path))
        except Exception as exc:
            logger.warning("Failed to load FAISS index from %s: %s", path, exc)
            return None

    def _ensure_faiss_index(self) -> faiss.Index:
        index = self.faiss_index
        if index is not None:
            return index
        if self._embeddings_npz_path is None or not self._embeddings_npz_path.exists():
            raise RuntimeError(
                "FAISS index not available and embeddings_npz artifact is missing."
            )
        with self._faiss_lock:
            if self.faiss_index is not None:
                return self.faiss_index
            self.faiss_index = self._build_faiss_from_embeddings(
                self._embeddings_npz_path
            )
            return self.faiss_index

    def _build_faiss_from_embeddings(self, npz_path: Path) -> faiss.Index:
        logger.info(
            "Building FAISS index on the fly from %s; this may take several minutes.",
            npz_path,
        )
        try:
            data = np.load(npz_path, mmap_mode="r")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embeddings NPZ from {npz_path}: {exc}"
            ) from exc

        if "image_embeddings" not in data:
            data.close()
            raise RuntimeError(
                f"Embeddings NPZ at {npz_path} does not contain 'image_embeddings'."
            )

        embeddings = data["image_embeddings"]
        if embeddings.ndim != 2:
            data.close()
            raise RuntimeError("Embeddings array must be 2-dimensional.")

        dim = embeddings.shape[1]
        if self.vector_dim is not None and self.vector_dim != dim:
            logger.warning(
                "Configured vector dimension (%s) does not match embeddings (%s); using %s.",
                self.vector_dim,
                dim,
                dim,
            )

        index = faiss.IndexFlatIP(dim)
        batch_size = 8192
        total = embeddings.shape[0]
        try:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = np.array(embeddings[start:end], dtype=np.float32, copy=False)
                norms = np.linalg.norm(batch, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                batch = batch / norms
                index.add(batch)
                if (start // batch_size) % 50 == 0:
                    logger.info(
                        "FAISS build progress: %d/%d vectors (%.1f%%)",
                        end,
                        total,
                        (end / total) * 100.0,
                    )
        finally:
            data.close()
        logger.info("FAISS index build complete (%d vectors).", total)

        target_path = self._faiss_index_path
        if target_path is not None:
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(index, str(target_path))
                logger.info("Cached FAISS index to %s", target_path)
                self._faiss_index_path = target_path
            except Exception as exc:
                logger.warning(
                    "Failed to write FAISS index to %s: %s. Continuing without persistence.",
                    target_path,
                    exc,
                )
        return index

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed a PIL image into the combined latent space.

        Uses trained projector if available, otherwise falls back to SVD.
        """
        if not self.models_in_order:
            raise RuntimeError("No models loaded for embedding.")

        features: List[np.ndarray] = []

        # Iterate models in the same order as in config (critical for projector)
        for model_type, model_wrapper in self.models_in_order:
            if model_type == "clip":
                features.append(model_wrapper.embed(image, self.device))
            else:  # ssl
                # SSL transforms expect tensor input; ensure consistent preprocessing
                prepared = image.copy().convert("RGB")
                features.append(model_wrapper.embed(prepared, self.device))

        # Now apply either projector or SVD based on what's available
        if self.projector_model is not None and self.whitening_stats is not None:
            # Use trained projector pipeline
            combined = self._apply_projector_pipeline(features)
        else:
            # Legacy SVD pipeline
            combined = np.concatenate(
                [feat.astype(np.float32) for feat in features], axis=1
            )

            if self.svd_image is not None:
                combined = self.svd_image.transform(combined)

        if combined.ndim == 2:
            return combined[0]
        return combined

    def _apply_projector_pipeline(self, features: List[np.ndarray]) -> np.ndarray:
        """Apply whitening + projector to per-model features."""
        from skinmap.pipeline.embedding_fusion import apply_whiten, l2_normalize

        # Check if whitening was used (mu/W arrays present)
        has_whitening = (
            self.whitening_stats is not None
            and "mu" in self.whitening_stats
            and "W" in self.whitening_stats
            and len(self.whitening_stats["mu"]) > 0
            and len(self.whitening_stats["W"]) > 0
        )

        # Step 1: L2 normalize and optionally whiten each model's embeddings
        processed = []
        for i, feat in enumerate(features):
            # L2 normalize
            feat_norm = l2_normalize(feat, axis=1)

            # Whiten if whitening stats are available
            if has_whitening:
                feat_processed = apply_whiten(
                    feat_norm,
                    self.whitening_stats["mu"][i],
                    self.whitening_stats["W"][i],
                )
            else:
                # No whitening - projector was trained on raw L2-normalized embeddings
                feat_processed = feat_norm

            processed.append(feat_processed)

        # Step 2: Concatenate
        z_cat = np.concatenate(processed, axis=1).astype(np.float32)

        # Step 3: L2-normalize each teacher block (matches training pipeline)
        dims = self.whitening_stats["dims"]
        start = 0
        for dim in dims:
            block = z_cat[:, start : start + dim]
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            z_cat[:, start : start + dim] = block / norms
            start += dim

        # Step 4: Project through trained model
        z_cat_tensor = torch.from_numpy(z_cat).to(self.device)
        with torch.inference_mode():
            projected = self.projector_model.img_head(z_cat_tensor)

        return projected.detach().cpu().numpy()

    def embed_bytes(self, data: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return self.embed_image(image)

    # Default prompt templates for text search.  CLIP models are trained on
    # image-caption pairs so bare keywords ("melanoma") live in a very
    # different part of the embedding space than descriptive captions.
    # Wrapping the query in templates and averaging the resulting embeddings
    # brings the query much closer to the image distribution.
    DEFAULT_TEXT_TEMPLATES: List[str] = [
        "This image shows {}",
        "a clinical photograph of {}",
        "a dermatoscopic image of {}",
        "a medical image showing {}",
        "a close-up photograph of {} on skin",
    ]

    def encode_text(
        self,
        text: str,
        templates: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Encode a text query into the shared embedding space.

        When *templates* is provided (or defaults are used), the query is
        formatted into each template, encoded separately, and the resulting
        embeddings are averaged and re-normalized.  Pass ``templates=["{}"]``
        to encode the raw text without any prompt engineering.

        For projector pipeline: extracts text embeddings from CLIP models only,
        concatenates them, and projects through the text head.

        For SVD pipeline: uses first CLIP model only.
        """
        if not self.clip_models:
            raise RuntimeError("No CLIP models loaded. Cannot encode text.")

        if templates is None:
            templates = self.DEFAULT_TEXT_TEMPLATES

        # Expand the query into all templates
        prompts = [t.format(text) for t in templates]

        # Encode all prompts and average. With a trained projector, all prompts
        # go through each CLIP teacher in one batched forward (C forwards, not
        # N*C). The legacy pipeline falls back to per-prompt encoding.
        if self.projector_model is not None and self.whitening_stats is not None:
            per_prompt = self._encode_texts_projector(prompts)
        else:
            per_prompt = np.stack(
                [self._encode_single_text(p) for p in prompts], axis=0
            )
        avg = per_prompt.mean(axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        return avg.astype(np.float32)

    def _encode_texts_projector(self, texts: List[str]) -> np.ndarray:
        """Batched projector-path text encoding.

        Returns an ``(N, d_out)`` array of projected, L2-normalized text
        embeddings, one per input string. All prompts pass through each CLIP
        teacher in a single padded forward. CLIP's EOS-token pooling is
        right-padding invariant, so this matches per-prompt encoding.
        """
        from skinmap.pipeline.embedding_fusion import apply_whiten, l2_normalize

        clip_indices = self.whitening_stats.get("clip_indices", [])
        if not clip_indices:
            clip_indices = list(range(len(self.clip_models)))
        clip_index_set = set(clip_indices)
        text_whitening = self.whitening_stats.get("text_whitening")

        per_model = []  # each (N, d_i)
        clip_idx_counter = 0
        for i, (_model_type, model_wrapper) in enumerate(self.models_in_order):
            if i not in clip_index_set:
                continue
            inputs = model_wrapper.processor(
                text=list(texts), return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                text_emb = _coerce_clip_features(
                    model_wrapper.model.get_text_features(**inputs), "text_embeds"
                )
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            feat = text_emb.cpu().numpy()  # (N, d_i)
            if text_whitening is not None:
                feat = apply_whiten(
                    l2_normalize(feat, axis=1),
                    text_whitening["mu"][clip_idx_counter],
                    text_whitening["W"][clip_idx_counter],
                )
            per_model.append(feat)
            clip_idx_counter += 1

        # Concatenate per-teacher blocks, L2-normalize each row, project.
        text_concat = np.concatenate(per_model, axis=1).astype(np.float32)
        norms = np.linalg.norm(text_concat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        text_concat = text_concat / norms
        text_tensor = torch.from_numpy(text_concat).to(self.device)
        with torch.inference_mode():
            projected = self.projector_model.txt_head(text_tensor)
        return projected.cpu().numpy()  # (N, d_out)

    def _encode_single_text(self, text: str) -> np.ndarray:
        """Encode a single text string (no template expansion)."""
        if self.projector_model is not None and self.whitening_stats is not None:
            # Projector pipeline: concatenate text from CLIP models only
            from skinmap.pipeline.embedding_fusion import apply_whiten, l2_normalize

            clip_indices = self.whitening_stats.get("clip_indices", [])
            if not clip_indices:
                # Fallback: use all CLIP models if indices not available
                clip_indices = list(range(len(self.clip_models)))

            text_features = []
            clip_idx_counter = 0
            for i, (model_type, model_wrapper) in enumerate(self.models_in_order):
                if i in clip_indices:
                    # This is a CLIP model, extract text embedding
                    inputs = model_wrapper.processor(text=[text], return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        text_emb = _coerce_clip_features(
                            model_wrapper.model.get_text_features(**inputs),
                            "text_embeds",
                        )
                        # Normalize
                        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
                        text_feat = text_emb.cpu().numpy()

                        # Apply text whitening if available
                        text_whitening = self.whitening_stats.get("text_whitening")
                        if text_whitening is not None:
                            text_feat_normed = l2_normalize(text_feat, axis=1)
                            text_feat_whitened = apply_whiten(
                                text_feat_normed,
                                text_whitening["mu"][clip_idx_counter],
                                text_whitening["W"][clip_idx_counter],
                            )
                            text_features.append(text_feat_whitened[0])
                        else:
                            text_features.append(text_feat[0])

                        clip_idx_counter += 1

            # Concatenate text features from all CLIP models
            text_concat = np.concatenate(text_features).astype(np.float32)

            # L2-normalize the concatenated text vector (matches training pipeline)
            norm = np.linalg.norm(text_concat)
            if norm > 0:
                text_concat = text_concat / norm

            # Project through text head
            text_tensor = torch.from_numpy(text_concat).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                projected = self.projector_model.txt_head(text_tensor)

            return projected.cpu().numpy()[0]
        else:
            # Legacy SVD pipeline or no projector: use first CLIP model
            clip_model = self.clip_models[0]
            inputs = clip_model.processor(text=[text], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                text_emb = _coerce_clip_features(
                    clip_model.model.get_text_features(**inputs), "text_embeds"
                )
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

            # If SVD available, apply it
            text_feat = text_emb.cpu().numpy()
            if self.svd_image is not None:
                # Note: this assumes SVD was fitted on concatenated features
                # For single model, this might not work well
                pass

            return text_feat[0]

    def find_nearest_neighbors(
        self, vector: np.ndarray, k: int = 16
    ) -> Tuple[List[int], List[float]]:
        index = self._ensure_faiss_index()
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        similarities, indices = index.search(vec.reshape(1, -1), k)
        flat_indices = indices.reshape(-1).tolist()
        flat_distances = (1.0 - similarities.reshape(-1)).tolist()
        return flat_indices, flat_distances

    def project_vector(self, vector: np.ndarray) -> Optional[np.ndarray]:
        if self.umap_model is None:
            logger.warning(
                "Upload embedding pipeline has no UMAP reducer; skipping projection. "
                "Ensure the embedding pipeline config includes a saved UMAP model."
            )
            return None
        try:
            projected = self.umap_model.transform(vector.reshape(1, -1))
            if projected.ndim == 2 and projected.shape[1] >= 2:
                return projected[0, :2].astype(float)
        except Exception as exc:
            logger.warning("Failed to project vector with UMAP: %s", exc)
        return None

    def search_image(
        self, image_bytes: bytes, k: int = 16
    ) -> Tuple[List[int], List[float], Optional[np.ndarray]]:
        vector = self.embed_bytes(image_bytes)
        indices, distances = self.find_nearest_neighbors(vector, k=k)
        coords = self.project_vector(vector)
        return indices, distances, coords
