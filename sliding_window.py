from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


def _starts(size: int, patch: int, overlap: float):
    if size <= patch:
        return [0]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.no_grad()
def sliding_window_logits(
    model, image: torch.Tensor, roi_size=(80, 160, 160), overlap=0.5,
    accumulation_device: str | torch.device | None = None,
    sw_batch_size: int = 1,
    accumulation_dtype: torch.dtype | None = None,
):
    if image.shape[0] != 1:
        raise ValueError("The lightweight sliding-window helper expects batch size 1")
    if sw_batch_size < 1:
        raise ValueError("sw_batch_size must be positive")
    original = image.shape[2:]
    padding = []
    for current, target in reversed(list(zip(original, roi_size))):
        padding.extend((0, max(0, target - current)))
    padded = F.pad(image, padding)
    spatial = padded.shape[2:]
    starts = [_starts(size, patch, overlap) for size, patch in zip(spatial, roi_size)]
    output = count = None
    output_device = image.device if accumulation_device is None else torch.device(accumulation_device)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = image.device
    positions = list(itertools.product(*starts))
    for offset in range(0, len(positions), sw_batch_size):
        batch_positions = positions[offset:offset + sw_batch_size]
        patches = torch.cat([
            padded[:, :, z:z + roi_size[0], y:y + roi_size[1], x:x + roi_size[2]]
            for z, y, x in batch_positions
        ])
        batch_logits = model(patches.to(model_device, non_blocking=True))
        if output is None:
            output_dtype = accumulation_dtype or batch_logits.dtype
            output = torch.zeros(
                (1, batch_logits.shape[1], *spatial), device=output_device,
                dtype=output_dtype,
            )
            count = torch.zeros(
                (1, 1, *spatial), device=output_device, dtype=output_dtype
            )
        for index, (z, y, x) in enumerate(batch_positions):
            logits = batch_logits[index:index + 1].to(
                device=output_device, dtype=output.dtype
            )
            output[:, :, z:z + roi_size[0], y:y + roi_size[1], x:x + roi_size[2]] += logits
            count[:, :, z:z + roi_size[0], y:y + roi_size[1], x:x + roi_size[2]] += 1
    output = output / count.clamp_min(1)
    return output[:, :, :original[0], :original[1], :original[2]]
