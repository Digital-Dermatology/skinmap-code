import torch
import torch.nn.functional as F
from torch import nn


def hierarchical_consistency_reg(
    original_embeddings,
    aligned_embeddings,
    levels=3,
    temperature=0.1,
    margin=0.0,
    eps=1e-12,
    weighting: str = "inverse",
    recenter: bool = True,
):
    """
    Compute hierarchical consistency loss between original and aligned embeddings.
    Preserves relationships at multiple scales.
    """
    with torch.amp.autocast("cuda", enabled=False):
        # normalize embeddings for numerical stability
        original_norm = F.normalize(original_embeddings, p=2, dim=-1)
        aligned_norm = F.normalize(aligned_embeddings, p=2, dim=-1)
        if recenter:
            original_norm = original_norm - original_norm.mean(0, keepdim=True)
            aligned_norm = aligned_norm - aligned_norm.mean(0, keepdim=True)
        # compute similarity matrices with temperature scaling
        original_sim = (original_norm @ original_norm.T) / temperature
        aligned_sim = (aligned_norm @ aligned_norm.T) / temperature
        # apply robust softmax with stable log-sum-exp trick
        orig_hierarchy = F.softmax(original_sim, dim=-1)
        aligned_hierarchy = F.softmax(aligned_sim, dim=-1)
        total_loss = 0
        for level in range(1, levels + 1):
            orig_hierarchy = torch.matrix_power(orig_hierarchy, level)
            aligned_hierarchy = torch.matrix_power(aligned_hierarchy, level)
            # use Jensen-Shannon divergence instead of KL (more stable)
            m = 0.5 * (orig_hierarchy + aligned_hierarchy)
            js_div = 0.5 * (
                F.kl_div(
                    (aligned_hierarchy + eps).log(), m + eps, reduction="batchmean"
                )
                + F.kl_div((orig_hierarchy + eps).log(), m + eps, reduction="batchmean")
            )
            # add a margin in order to keep focusing on big improvements rather than small
            js_div = F.relu(js_div - margin)
            # apply scaling to keep loss magnitudes reasonable
            if weighting == "none":
                scaled_loss = js_div
            elif weighting == "inverse":
                # lower weight for higher levels
                scaled_loss = js_div * (1.0 / level)
            else:
                raise ValueError(f"Unknown weighting: {weighting}")
            total_loss += scaled_loss
    return total_loss / levels


class CLIPLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        uni_temperature: float = 0.05,
        normalize_latents: bool = True,
        warmup_steps: int = 500,
        lambda_hierarchy: float = 10,
        hierarchy_levels: int = 1,
        hierarchy_weighting: str = "none",
        hierarchy_margin: float = 0.0,
        lambda_consistency: float = 0.0,
        teacher_dims: list = None,
        text_dims: list = None,
    ):
        super().__init__()
        self.train_step = 0
        self.warmup_steps = warmup_steps

        self.temperature = temperature
        self.uni_temperature = uni_temperature
        self.normalize_latents = normalize_latents

        self.lambda_hierarchy_base = lambda_hierarchy
        self.hierarchy_levels = hierarchy_levels
        self.hierarchy_weighting = hierarchy_weighting
        self.hierarchy_margin = hierarchy_margin
        self.lambda_consistency = lambda_consistency

        # Per-model dimensions for splitting concatenated embeddings
        self.teacher_dims = teacher_dims
        self.text_dims = text_dims

        # for schedulers (i.e. will be updated)
        self.register_buffer(
            "lambda_hierarchy",
            torch.tensor(lambda_hierarchy, dtype=torch.float32),
        )

        # static for torch.compile
        self.use_structure = lambda_hierarchy > 0
        self.use_consistency = lambda_consistency > 0

    def name(self):
        name = "CLIPLoss"
        name += f"(temp={self.temperature}"
        name += f", norm={self.normalize_latents}"
        if self.lambda_hierarchy > 0:
            name += f", lambda_hierarchy={self.lambda_hierarchy_base}"
            name += f", hierarchy_levels={self.hierarchy_levels}"
            name += f", warmup_steps={self.warmup_steps}"
        if self.lambda_consistency > 0:
            name += f", lambda_consistency={self.lambda_consistency}"
        name += ")"
        return name

    def step(self):
        if self.use_structure and self.warmup_steps > 0:
            new_lambda = self.lambda_hierarchy_base * min(
                1.0, self.train_step / self.warmup_steps
            )
            self.lambda_hierarchy.fill_(new_lambda)
        self.train_step += 1

    def _compute_per_model_hierarchy(
        self,
        concatenated_original: torch.Tensor,
        aligned_embeddings: torch.Tensor,
        dims: list,
    ):
        """
        Compute hierarchical consistency loss per-model to avoid preserving
        spurious cross-model correlations.

        Args:
            concatenated_original: Concatenated embeddings [M1 | M2 | ...], shape (B, d1+d2+...)
            aligned_embeddings: Projected embeddings in shared space, shape (B, d_out)
            dims: List of dimensions [d1, d2, ...] for splitting concatenated_original

        Returns:
            Average hierarchical loss across models
        """
        if dims is None or len(dims) <= 1:
            # Single model or no dims specified - use old behavior (full concatenation)
            return hierarchical_consistency_reg(
                concatenated_original,
                aligned_embeddings,
                levels=self.hierarchy_levels,
                temperature=self.temperature,
                weighting=self.hierarchy_weighting,
                margin=self.hierarchy_margin,
            )

        # Split concatenated embeddings by model and compute loss per-model
        losses = []
        start = 0
        for dim in dims:
            # Extract this model's embeddings from concatenation
            model_original = concatenated_original[:, start : start + dim]

            # Compute hierarchical consistency for this model
            # The aligned embeddings are the same for all models (shared projection space)
            loss = hierarchical_consistency_reg(
                model_original,
                aligned_embeddings,
                levels=self.hierarchy_levels,
                temperature=self.temperature,
                weighting=self.hierarchy_weighting,
                margin=self.hierarchy_margin,
            )
            losses.append(loss)
            start += dim

        # Average across models
        return torch.stack(losses).mean()

    def forward(
        self,
        image_embeddings_aligned: torch.Tensor,
        text_embeddings_aligned: torch.Tensor,
        image_embeddings_original: torch.Tensor,
        text_embeddings_original: torch.Tensor,
        image_embeddings_aligned_alt: torch.Tensor = None,
    ):
        if self.normalize_latents:
            image_embeddings_aligned = F.normalize(image_embeddings_aligned, p=2, dim=1)
            text_embeddings_aligned = F.normalize(text_embeddings_aligned, p=2, dim=1)

        # Standard CLIP contrastive loss
        logits = image_embeddings_aligned @ text_embeddings_aligned.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        clip_loss = (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        ) / 2

        loss_dict = {"clip_loss": clip_loss.detach()}
        total_loss = clip_loss
        if self.use_structure:
            # Compute per-model hierarchical consistency to avoid preserving
            # spurious cross-model correlations
            img_hierarchy_loss = self._compute_per_model_hierarchy(
                image_embeddings_original,
                image_embeddings_aligned,
                self.teacher_dims,
            )
            txt_hierarchy_loss = self._compute_per_model_hierarchy(
                text_embeddings_original,
                text_embeddings_aligned,
                self.text_dims,
            )
            hierarchy_loss = (img_hierarchy_loss + txt_hierarchy_loss) / 2
            loss_dict["hierarchy_loss_wo_lambda"] = hierarchy_loss.detach()
            loss_dict["hierarchy_loss"] = (
                self.lambda_hierarchy * hierarchy_loss.detach()
            )
            total_loss += self.lambda_hierarchy * hierarchy_loss

        # Consistency loss between primary and alternative image embeddings
        if self.use_consistency and image_embeddings_aligned_alt is not None:
            if self.normalize_latents:
                image_embeddings_aligned_alt = F.normalize(
                    image_embeddings_aligned_alt, p=2, dim=1
                )

            # Compute cosine similarity and minimize distance (1 - cosine_similarity)
            cos_sim = F.cosine_similarity(
                image_embeddings_aligned, image_embeddings_aligned_alt, dim=-1
            )
            consistency_loss = (1 - cos_sim).mean()
            loss_dict["consistency_loss_wo_lambda"] = consistency_loss.detach()
            loss_dict["consistency_loss"] = (
                self.lambda_consistency * consistency_loss.detach()
            )
            total_loss += self.lambda_consistency * consistency_loss

        loss_dict["overall_loss"] = total_loss
        return loss_dict
