from __future__ import annotations

import torch
import torch.nn.functional as F


def selected_channel(logits: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
    return logits[torch.arange(logits.shape[0], device=logits.device), task_id][:, None]


def known_organ_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    task_id: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
):
    selected = selected_channel(logits, task_id)
    probability = torch.sigmoid(selected)
    intersection = (probability * target).sum(dim=(1, 2, 3, 4))
    denominator = probability.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    per_sample_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    per_sample_bce = F.binary_cross_entropy_with_logits(
        selected, target, reduction="none"
    ).flatten(1).mean(dim=1)
    if sample_weights is None:
        sample_weights = torch.ones_like(per_sample_dice)
    sample_weights = sample_weights.to(device=per_sample_dice.device, dtype=per_sample_dice.dtype)
    normalizer = sample_weights.sum().clamp_min(1e-8)
    dice = (per_sample_dice * sample_weights).sum() / normalizer
    bce = (per_sample_bce * sample_weights).sum() / normalizer
    return dice + bce, {"dice": dice.detach(), "bce": bce.detach()}


def consistency_loss(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    known_task: torch.Tensor | None = None,
):
    student_probability = torch.sigmoid(student_logits)
    mask = teacher_probability >= 0.6
    if known_task is not None:
        unknown_channels = torch.ones_like(mask)
        unknown_channels[
            torch.arange(mask.shape[0], device=mask.device), known_task
        ] = False
        mask &= unknown_channels
    if not mask.any():
        return student_logits.sum() * 0.0
    return F.mse_loss(student_probability[mask], teacher_probability[mask])


def pseudo_label_loss(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    selected_mask: torch.Tensor,
    quality: torch.Tensor,
    mode: str = "legacy",
    local_radius: int = 3,
):
    if not selected_mask.any():
        return student_logits.sum() * 0.0
    voxel_loss = F.binary_cross_entropy_with_logits(
        student_logits, teacher_probability.detach(), reduction="none"
    )
    if mode == "local":
        kernel_size = 2 * local_radius + 1
        local_quality = F.max_pool3d(
            quality.detach(), kernel_size=kernel_size, stride=1,
            padding=local_radius,
        )
        local_mask = local_quality > 0
        weighted = voxel_loss * local_mask.float() * local_quality
    else:
        local_mask = selected_mask
        local_quality = quality.detach()
        weighted = voxel_loss * selected_mask.float() * local_quality
    per_class = []
    for organ in range(student_logits.shape[1]):
        active = local_mask[:, organ]
        if active.any():
            if mode in ("safe", "local"):
                # Keep reliability as a real multiplier. The legacy denominator
                # cancels most of the risk score and overweights sparse classes.
                denominator = active.float().sum().clamp_min(1.0)
            elif mode == "legacy":
                denominator = (active.float() * local_quality[:, organ]).sum().clamp_min(1.0)
            else:
                raise ValueError(f"Unknown pseudo loss mode: {mode}")
            per_class.append(weighted[:, organ].sum() / denominator)
        elif mode in ("safe", "local"):
            per_class.append(student_logits[:, organ].sum() * 0.0)
    return torch.stack(per_class).mean() if per_class else student_logits.sum() * 0.0


def soft_exclusion_loss(logits: torch.Tensor):
    probability = torch.sigmoid(logits)
    summed = probability.sum(dim=1)
    return torch.relu(summed - 1.0).square().mean()
