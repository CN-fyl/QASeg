from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


_MODEL_PARENT = Path(__file__).resolve().parents[2]
MODEL_ROOT = (
    Path(__file__).resolve().parents[3] / "models"
    if _MODEL_PARENT.name == "optimization" else _MODEL_PARENT
)
NNUNET_ROOT = MODEL_ROOT / "nnUNet"
NNUNET_TOOLS = NNUNET_ROOT / "project_tools"
sys.path.insert(0, str(NNUNET_ROOT))
sys.path.insert(0, str(NNUNET_TOOLS))

from train_masked_protocol_v2 import build_network


class SourceStyleBank(nn.Module):
    """EMA source statistics used for shallow cross-source feature intervention."""

    def __init__(self, channels: int, num_sources: int = 4, momentum: float = 0.95):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(num_sources, channels))
        self.register_buffer("std", torch.ones(num_sources, channels))
        self.register_buffer("initialized", torch.zeros(num_sources, dtype=torch.bool))

    @torch.no_grad()
    def update(self, feature: torch.Tensor, source_id: torch.Tensor):
        current_mean = feature.mean(dim=(2, 3, 4))
        current_std = feature.std(dim=(2, 3, 4)).clamp_min(1e-4)
        for source in source_id.unique().tolist():
            selected = source_id == source
            mean = current_mean[selected].mean(dim=0).float()
            std = current_std[selected].mean(dim=0).float()
            if not self.initialized[source]:
                self.mean[source].copy_(mean)
                self.std[source].copy_(std)
                self.initialized[source] = True
            else:
                self.mean[source].lerp_(mean, 1.0 - self.momentum)
                self.std[source].lerp_(std, 1.0 - self.momentum)

    def intervene(self, feature, source_id, target_source, alpha):
        if self.training:
            self.update(feature.detach(), source_id)
        dims = (2, 3, 4)
        instance_mean = feature.mean(dim=dims, keepdim=True)
        instance_std = feature.std(dim=dims, keepdim=True).clamp_min(1e-4)
        fallback_mean = instance_mean.flatten(2).mean(dim=2)
        fallback_std = instance_std.flatten(2).mean(dim=2)
        target_mean = torch.where(
            self.initialized[target_source, None], self.mean[target_source], fallback_mean
        ).to(feature.dtype)[:, :, None, None, None]
        target_std = torch.where(
            self.initialized[target_source, None], self.std[target_source], fallback_std
        ).to(feature.dtype)[:, :, None, None, None]
        transferred = (feature - instance_mean) / instance_std * target_std + target_mean
        return (1.0 - alpha) * feature + alpha * transferred


class ICRSegNNUNet(nn.Module):
    """Existing Masked nnU-Net with optional shallow source-style intervention."""

    def __init__(self, num_sources: int = 4):
        super().__init__()
        self.network = build_network()
        first_channels = self.network.encoder.output_channels[0]
        self.style_bank = SourceStyleBank(first_channels, num_sources)
        self.risk_feature_channels = self.network.encoder.output_channels[-1]

    def _target_sources(self, source_id: torch.Tensor) -> torch.Tensor:
        offsets = torch.randint(1, self.style_bank.mean.shape[0], source_id.shape, device=source_id.device)
        return (source_id + offsets) % self.style_bank.mean.shape[0]

    def sample_target_sources(self, source_id: torch.Tensor) -> torch.Tensor:
        return self._target_sources(source_id)

    @staticmethod
    def sample_intervention_alpha(
        image: torch.Tensor, alpha_range: tuple[float, float]
    ) -> torch.Tensor:
        return torch.empty(
            (image.shape[0], 1, 1, 1, 1), device=image.device, dtype=image.dtype
        ).uniform_(*alpha_range)

    def forward(
        self,
        image: torch.Tensor,
        return_features: bool = False,
        intervene: bool = False,
        source_id: torch.Tensor | None = None,
        target_source: torch.Tensor | None = None,
        alpha_range: tuple[float, float] = (0.25, 0.75),
        alpha: torch.Tensor | None = None,
    ):
        skips = []
        x = image
        for stage_index, stage in enumerate(self.network.encoder.stages):
            x = stage(x)
            if stage_index == 0 and intervene:
                if source_id is None:
                    raise ValueError("source_id is required for feature intervention")
                if target_source is None:
                    target_source = self._target_sources(source_id)
                if alpha is None:
                    alpha = self.sample_intervention_alpha(image, alpha_range)
                x = self.style_bank.intervene(x, source_id, target_source, alpha)
            elif stage_index == 0 and self.training and source_id is not None:
                self.style_bank.update(x.detach(), source_id)
            skips.append(x)
        logits = self.network.decoder(skips)
        if return_features:
            return logits, skips[-1]
        return logits

    def inference_model(self) -> nn.Module:
        """Return the plain nnU-Net used at inference time."""
        return self.network
