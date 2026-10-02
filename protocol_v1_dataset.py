from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


ORGAN_ORDER = ("liver", "spleen", "kidney", "pancreas")
PATCH_SIZE = (80, 160, 160)


def stratified_subset(records, limit):
    if limit is None or limit >= len(records):
        return records
    if limit < 1:
        raise ValueError("limit must be positive")
    groups = {}
    for record in records:
        groups.setdefault(int(record["task_id"]), []).append(record)
    group_keys = sorted(groups)
    if limit < len(group_keys):
        raise ValueError(f"limit={limit} cannot cover all {len(group_keys)} organ sources")
    base, remainder = divmod(limit, len(group_keys))
    selected = []
    for position, key in enumerate(group_keys):
        bucket = groups[key]
        count = min(len(bucket), base + (position < remainder))
        indices = np.linspace(0, len(bucket) - 1, count, dtype=int)
        selected.extend(bucket[index] for index in indices)
    return selected


def normalize_ct(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, -325.0, 325.0).astype(np.float32)
    return (image + 325.0) / 650.0


def pad_to_shape(array: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    padding = [(0, max(0, size - current)) for current, size in zip(array.shape, target)]
    return np.pad(array, padding, mode="constant")


def random_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size=PATCH_SIZE,
    foreground_sampling_probability: float = 0.8,
):
    image = pad_to_shape(image, patch_size)
    label = pad_to_shape(label, patch_size)
    foreground = np.argwhere(label > 0)
    use_foreground = (
        len(foreground) and random.random() < foreground_sampling_probability
    )
    center = foreground[random.randrange(len(foreground))] if use_foreground else None
    starts = []
    for axis, patch in enumerate(patch_size):
        maximum = image.shape[axis] - patch
        if maximum <= 0:
            starts.append(0)
        elif center is None:
            starts.append(random.randint(0, maximum))
        else:
            low = max(0, int(center[axis]) - patch + 1)
            high = min(int(center[axis]), maximum)
            starts.append(random.randint(low, high))
    slices = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    return image[slices], label[slices]


class ProtocolV1Dataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        training=False,
        limit=None,
        balance_sources=False,
        foreground_sampling_probability: float | tuple[float, ...] = 0.8,
    ):
        records = json.loads(Path(manifest).read_text(encoding="utf-8"))["cases"]
        self.records = [record for record in records if record["split"] == split]
        for record in self.records:
            record.setdefault("task_id", int(record["channel"]))
        self.records = stratified_subset(self.records, limit)
        if not self.records:
            raise RuntimeError(f"No cases found for split={split}")
        self.training = training
        self.balance_sources = balance_sources and split == "train"
        if isinstance(foreground_sampling_probability, (tuple, list)):
            probabilities = tuple(float(value) for value in foreground_sampling_probability)
            if len(probabilities) != 4 or any(value < 0 or value > 1 for value in probabilities):
                raise ValueError("foreground_sampling_probability must contain four values in [0,1]")
            self.foreground_sampling_probabilities = probabilities
        else:
            value = float(foreground_sampling_probability)
            if not 0.0 <= value <= 1.0:
                raise ValueError("foreground_sampling_probability must be in [0,1]")
            self.foreground_sampling_probabilities = (value,) * 4
        self.source_groups = {
            task_id: [record for record in self.records if record["task_id"] == task_id]
            for task_id in sorted({record["task_id"] for record in self.records})
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.balance_sources:
            task_ids = sorted(self.source_groups)
            task_id = task_ids[index % len(task_ids)]
            record = self.source_groups[task_id][np.random.randint(len(self.source_groups[task_id]))]
        else:
            record = self.records[index]
        image = sitk.GetArrayFromImage(sitk.ReadImage(record["image_preprocessed"]))
        result = {
            "image": normalize_ct(image)[None],
            "task_id": np.int64(record["task_id"]),
            "identifier": record["identifier"],
            "source": record["source"],
            "full_label": record["full_label"],
            "image_preprocessed": record["image_preprocessed"],
        }
        if record.get("partial_label_preprocessed"):
            label = sitk.GetArrayFromImage(sitk.ReadImage(record["partial_label_preprocessed"]))
            label = (label == int(record["known_label"])).astype(np.float32)
            if self.training:
                image_patch, label_patch = random_patch(
                    result["image"][0],
                    label,
                    foreground_sampling_probability=self.foreground_sampling_probabilities[
                        int(record["task_id"])
                    ],
                )
                if random.random() < 0.5:
                    image_patch, label_patch = image_patch[:, :, ::-1], label_patch[:, :, ::-1]
                if random.random() < 0.5:
                    image_patch, label_patch = image_patch[:, ::-1, :], label_patch[:, ::-1, :]
                result["image"] = np.ascontiguousarray(image_patch[None])
                label = np.ascontiguousarray(label_patch)
            result["label"] = label[None]
        return result


def as_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
