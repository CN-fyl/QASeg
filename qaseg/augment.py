from __future__ import annotations

import torch


def weak_intensity_augment(image: torch.Tensor) -> torch.Tensor:
    scale = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(0.95, 1.05)
    shift = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(-0.025, 0.025)
    return (image * scale + shift + torch.randn_like(image) * 0.01).clamp(0.0, 1.0)


def strong_intensity_augment(image: torch.Tensor) -> torch.Tensor:
    scale = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(0.8, 1.2)
    shift = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(-0.1, 0.1)
    gamma = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device).uniform_(0.7, 1.5)
    augmented = (image * scale + shift).clamp(0.0, 1.0).pow(gamma)
    return (augmented + torch.randn_like(image) * 0.03).clamp(0.0, 1.0)
