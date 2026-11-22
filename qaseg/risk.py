from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import cc3d
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


STAT_NAMES = (
    "mean_probability",
    "low_probability",
    "augmentation_consistency",
    "teacher_student_agreement",
    "boundary_certainty",
    "history_stability",
    "log_volume",
    "compactness",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "nearest_organ_distance",
)
STAT_INDEX = {name: index for index, name in enumerate(STAT_NAMES)}


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    components, count = cc3d.connected_components(
        np.ascontiguousarray(mask), connectivity=6, return_N=True,
    )
    return components, int(count)


def _profile_measure(profile_timer, name: str):
    return profile_timer.measure(name) if profile_timer is not None else nullcontext()


@dataclass
class CandidateBatch:
    evidence: torch.Tensor
    batch_index: torch.Tensor
    organ_index: torch.Tensor
    masks: list[torch.Tensor]
    quality_target: torch.Tensor | None = None
    precision_target: torch.Tensor | None = None
    recall_target: torch.Tensor | None = None

    def __len__(self):
        return len(self.masks)


class RiskEstimator(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        num_organs=4,
        num_sources=4,
        embedding_dim=8,
        disabled_evidence: tuple[str, ...] = (),
        use_organ_embedding: bool = True,
        use_source_embedding: bool = True,
    ):
        super().__init__()
        unknown = set(disabled_evidence) - set(STAT_NAMES)
        if unknown:
            raise ValueError(f"Unknown risk evidence: {sorted(unknown)}")
        self.feature_channels = feature_channels
        self.disabled_evidence = tuple(disabled_evidence)
        self.use_organ_embedding = use_organ_embedding
        self.use_source_embedding = use_source_embedding
        self.organ_embedding = nn.Embedding(num_organs, embedding_dim)
        self.source_embedding = nn.Embedding(num_sources, embedding_dim)
        input_dim = feature_channels + len(STAT_NAMES) + embedding_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(self, evidence, organ_index, source_index):
        if self.disabled_evidence:
            evidence = evidence.clone()
            statistics = evidence[:, self.feature_channels:]
            for name in self.disabled_evidence:
                statistics[:, STAT_INDEX[name]] = 0.0
        organ_embedding = self.organ_embedding(organ_index)
        source_embedding = self.source_embedding(source_index)
        if not self.use_organ_embedding:
            organ_embedding = torch.zeros_like(organ_embedding)
        if not self.use_source_embedding:
            source_embedding = torch.zeros_like(source_embedding)
        embedded = torch.cat([
            evidence,
            organ_embedding,
            source_embedding,
        ], dim=1)
        logits = self.mlp(embedded)
        output = torch.sigmoid(logits)
        return {
            "quality": output[:, 0],
            "precision": output[:, 1],
            "recall": output[:, 2],
            "good": output[:, 3],
            "good_logit": logits[:, 3],
        }


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    pooled = F.avg_pool3d(mask.float()[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    return mask & (pooled < 0.999)


def _masked_feature_pool(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(mask.float()[None, None], size=feature.shape[1:], mode="nearest")[0, 0]
    denominator = resized.sum().clamp_min(1.0)
    return (feature * resized[None]).sum(dim=(1, 2, 3)) / denominator


def _matched_history_iou(mask: torch.Tensor, historical_mask: torch.Tensor) -> torch.Tensor:
    components, count = _connected_components(historical_mask.detach().cpu().numpy())
    if count == 0:
        return mask.new_tensor(0.0, dtype=torch.float32)
    current = mask.detach().cpu().numpy()
    intersections = np.bincount(components[current], minlength=count + 1)
    intersections[0] = 0
    best_component_index = int(np.argmax(intersections))
    best_intersection = int(intersections[best_component_index])
    if best_intersection == 0:
        return mask.new_tensor(0.0, dtype=torch.float32)
    component_sizes = np.bincount(components.ravel(), minlength=count + 1)
    union = int(current.sum() + component_sizes[best_component_index] - best_intersection)
    return mask.new_tensor(best_intersection / max(1, union), dtype=torch.float32)


def _matched_history_iou_numpy(
    current: np.ndarray,
    components: np.ndarray,
    component_sizes: np.ndarray,
    current_size: int,
) -> float:
    count = len(component_sizes) - 1
    if count == 0:
        return 0.0
    intersections = np.bincount(components[current], minlength=count + 1)
    intersections[0] = 0
    best_component = int(np.argmax(intersections))
    best_intersection = int(intersections[best_component])
    if best_intersection == 0:
        return 0.0
    union = int(current_size + component_sizes[best_component] - best_intersection)
    return best_intersection / max(1, union)


def _matched_target_counts(
    current: np.ndarray,
    target: np.ndarray,
    components: np.ndarray,
    component_sizes: np.ndarray,
) -> tuple[int, int]:
    count = len(component_sizes) - 1
    if count <= 1:
        return int(np.count_nonzero(current & target)), int(component_sizes[1]) if count else 0
    overlaps = np.bincount(components[current], minlength=count + 1)
    best_component = int(np.argmax(overlaps[1:])) + 1
    return int(overlaps[best_component]), int(component_sizes[best_component])


def _matched_target_component(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match a prediction to one anatomical GT component (not the whole bilateral organ)."""
    components, count = _connected_components(target.detach().cpu().numpy())
    if count <= 1:
        return target
    current = mask.detach().cpu().numpy()
    overlaps = np.bincount(components[current], minlength=count + 1)
    best_component = int(np.argmax(overlaps[1:])) + 1
    return torch.from_numpy(components == best_component).to(device=mask.device)


def _organ_centroids(probability: torch.Tensor) -> list[torch.Tensor | None]:
    spatial_shape = probability.shape[1:]
    centroids = []
    for organ in range(probability.shape[0]):
        mask = probability[organ] >= 0.5
        if not mask.any():
            centroids.append(None)
            continue
        coordinates = torch.where(mask)
        centroids.append(torch.stack([
            coordinates[axis].float().mean() / max(1, spatial_shape[axis] - 1)
            for axis in range(3)
        ]))
    return centroids


def _region_stats(
    teacher_probability: torch.Tensor,
    student_probability: torch.Tensor,
    strong_teacher_probability: torch.Tensor,
    mask: torch.Tensor,
    organ: int,
    historical_probability: torch.Tensor | None,
    history_stability_value: float | None = None,
    organ_centroids: list[torch.Tensor | None] | None = None,
):
    probability = teacher_probability[organ][mask].float()
    mean_probability = probability.mean()
    low_probability = torch.quantile(probability, 0.1) if probability.numel() > 1 else probability.mean()
    consistency = 1.0 - (
        teacher_probability[organ][mask].float()
        - strong_teacher_probability[organ][mask].float()
    ).abs().mean()
    teacher_student = 1.0 - (
        teacher_probability[organ][mask].float()
        - student_probability[organ][mask].float()
    ).abs().mean()
    boundary = _boundary(mask)
    boundary_probability = teacher_probability[:, boundary].float()
    if boundary_probability.numel():
        entropy = -(boundary_probability.clamp(1e-6, 1 - 1e-6) * boundary_probability.clamp(1e-6, 1 - 1e-6).log()
                    + (1 - boundary_probability).clamp(1e-6, 1 - 1e-6) * (1 - boundary_probability).clamp(1e-6, 1 - 1e-6).log()).mean()
        boundary_certainty = 1.0 - entropy / 0.693147
    else:
        boundary_certainty = mean_probability.new_tensor(0.5)
    if history_stability_value is not None:
        history_stability = mean_probability.new_tensor(history_stability_value)
    elif historical_probability is None:
        history_stability = mean_probability.new_tensor(0.5)
    else:
        historical_mask = (
            historical_probability[organ]
            if historical_probability.dtype == torch.bool
            else historical_probability[organ] >= 0.5
        )
        history_stability = _matched_history_iou(mask, historical_mask)
    volume_fraction = mask.float().mean().clamp_min(1e-6)
    log_volume = torch.log10(volume_fraction) / 6.0 + 1.0
    compactness = mask.float().sum() / boundary.float().sum().clamp_min(1.0)
    coordinates = torch.where(mask)
    spatial_shape = mask.shape
    centroid = torch.stack([
        coordinates[axis].float().mean() / max(1, spatial_shape[axis] - 1)
        for axis in range(3)
    ])
    other_distances = []
    if organ_centroids is None:
        organ_centroids = _organ_centroids(teacher_probability)
    for other_organ, other_centroid in enumerate(organ_centroids):
        if other_organ != organ and other_centroid is not None:
            other_distances.append(
                torch.linalg.vector_norm(centroid - other_centroid) / (3.0 ** 0.5)
            )
    nearest_distance = (
        torch.stack(other_distances).min() if other_distances
        else mean_probability.new_tensor(0.5)
    )
    return torch.stack([
        mean_probability,
        low_probability,
        consistency.clamp(0, 1),
        teacher_student.clamp(0, 1),
        boundary_certainty.clamp(0, 1),
        history_stability.clamp(0, 1),
        log_volume.clamp(0, 1),
        (compactness / 10.0).clamp(0, 1),
        *centroid.clamp(0, 1),
        nearest_distance.clamp(0, 1),
    ])


def extract_candidates(
    teacher_probability: torch.Tensor,
    student_probability: torch.Tensor,
    strong_teacher_probability: torch.Tensor,
    features: torch.Tensor,
    source_id: torch.Tensor,
    known_task: torch.Tensor,
    threshold: float | tuple[float, ...] | list[float],
    include_known: bool,
    target: torch.Tensor | None = None,
    min_voxels: int = 16,
    max_components_per_class: int = 8,
    historical_probability: torch.Tensor | None = None,
    forbidden_mask: torch.Tensor | None = None,
    profile_timer=None,
) -> CandidateBatch:
    device = teacher_probability.device
    evidence, batch_indices, organ_indices, masks = [], [], [], []
    quality_targets, precision_targets, recall_targets = [], [], []
    for batch_index in range(teacher_probability.shape[0]):
        batch_organ_centroids = None
        for organ in range(teacher_probability.shape[1]):
            is_known = organ == int(known_task[batch_index])
            if include_known != is_known:
                continue
            with _profile_measure(profile_timer, "candidate_foreground_to_cpu"):
                if isinstance(threshold, (tuple, list)):
                    foreground_threshold = threshold[organ]
                else:
                    foreground_threshold = threshold
                foreground = teacher_probability[batch_index, organ] >= foreground_threshold
                if forbidden_mask is not None:
                    blocked = forbidden_mask[batch_index]
                    if blocked.ndim == foreground.ndim + 1:
                        blocked = blocked[0]
                    foreground = foreground & ~blocked.bool()
                foreground = foreground.detach().cpu().numpy()
            with _profile_measure(profile_timer, "candidate_component_labeling"):
                components, count = _connected_components(foreground)
                all_component_sizes = np.bincount(components.ravel(), minlength=count + 1)
                component_sizes = all_component_sizes[1:]
                ranked_components = np.argsort(component_sizes)[::-1][:max_components_per_class] + 1
            history_components = history_component_sizes = None
            if historical_probability is not None:
                with _profile_measure(profile_timer, "candidate_history_to_cpu"):
                    history_mask = (
                        historical_probability[batch_index, organ] >= 0.5
                    ).detach().cpu().numpy()
                with _profile_measure(profile_timer, "candidate_history_labeling"):
                    history_components, history_count = _connected_components(history_mask)
                    history_component_sizes = np.bincount(
                        history_components.ravel(), minlength=history_count + 1,
                    )
            target_array = target_components = target_component_sizes = None
            if target is not None:
                target_channel = organ if target.shape[1] > 1 else 0
                with _profile_measure(profile_timer, "candidate_target_to_cpu"):
                    target_array = target[batch_index, target_channel].bool().detach().cpu().numpy()
                with _profile_measure(profile_timer, "candidate_target_labeling"):
                    target_components, target_count = _connected_components(target_array)
                    target_component_sizes = np.bincount(
                        target_components.ravel(), minlength=target_count + 1,
                    )
            for component in ranked_components.tolist():
                if component_sizes[component - 1] < min_voxels:
                    continue
                with _profile_measure(profile_timer, "candidate_mask_to_gpu"):
                    mask_array = components == component
                    mask = torch.from_numpy(mask_array).to(device=device)
                with _profile_measure(profile_timer, "candidate_feature_pool"):
                    pooled = _masked_feature_pool(features[batch_index], mask)
                history_stability_value = None
                if history_components is not None:
                    with _profile_measure(profile_timer, "candidate_history_match"):
                        history_stability_value = _matched_history_iou_numpy(
                            mask_array, history_components, history_component_sizes,
                            int(component_sizes[component - 1]),
                        )
                if batch_organ_centroids is None:
                    with _profile_measure(profile_timer, "candidate_organ_centroids"):
                        batch_organ_centroids = _organ_centroids(
                            teacher_probability[batch_index]
                        )
                with _profile_measure(profile_timer, "candidate_region_stats"):
                    stats = _region_stats(
                        teacher_probability[batch_index], student_probability[batch_index],
                        strong_teacher_probability[batch_index], mask, organ,
                        None if historical_probability is None else historical_probability[batch_index],
                        history_stability_value=history_stability_value,
                        organ_centroids=batch_organ_centroids,
                    )
                evidence.append(torch.cat([pooled, stats]))
                batch_indices.append(batch_index)
                organ_indices.append(organ)
                masks.append(mask)
                if target is not None:
                    with _profile_measure(profile_timer, "candidate_target_match"):
                        intersection_count, truth_count = _matched_target_counts(
                            mask_array, target_array, target_components, target_component_sizes,
                        )
                        intersection = mask.new_tensor(intersection_count, dtype=torch.float32)
                        mask_count = mask.new_tensor(component_sizes[component - 1], dtype=torch.float32)
                        truth_count = mask.new_tensor(truth_count, dtype=torch.float32)
                        precision = intersection / mask_count.clamp_min(1)
                        recall = intersection / truth_count.clamp_min(1)
                        quality = 2.0 * intersection / (mask_count + truth_count).clamp_min(1)
                        quality_targets.append(quality)
                        precision_targets.append(precision)
                        recall_targets.append(recall)
    with _profile_measure(profile_timer, "candidate_assemble"):
        feature_dim = features.shape[1] + len(STAT_NAMES)
        empty_evidence = torch.empty((0, feature_dim), device=device)
        return CandidateBatch(
            evidence=torch.stack(evidence) if evidence else empty_evidence,
            batch_index=torch.tensor(batch_indices, device=device, dtype=torch.long),
            organ_index=torch.tensor(organ_indices, device=device, dtype=torch.long),
            masks=masks,
            quality_target=torch.stack(quality_targets) if quality_targets else None,
            precision_target=torch.stack(precision_targets) if precision_targets else None,
            recall_target=torch.stack(recall_targets) if recall_targets else None,
        )


def risk_supervision_loss(prediction: dict, candidates: CandidateBatch, good_threshold: float):
    if len(candidates) == 0:
        return prediction["quality"].sum() * 0.0
    regression = (
        F.smooth_l1_loss(prediction["quality"], candidates.quality_target)
        + 0.5 * F.smooth_l1_loss(prediction["precision"], candidates.precision_target)
        + 0.5 * F.smooth_l1_loss(prediction["recall"], candidates.recall_target)
    )
    good_target = (candidates.quality_target >= good_threshold).float()
    classification = F.binary_cross_entropy_with_logits(prediction["good_logit"], good_target)
    return regression + 0.5 * classification


def select_pseudo_labels(
    candidates: CandidateBatch,
    prediction: dict,
    teacher_probability: torch.Tensor,
    quality_threshold: float | tuple[float, ...] | list[float],
    consistency_threshold: float,
    max_regions_per_class: int,
    quota_fraction: float = 1.0,
    min_regions_per_class: int = 1,
    class_balance: bool = True,
    confidence_threshold: float = 0.0,
    precision_threshold: float | tuple[float, ...] | list[float] = 0.0,
    history_threshold: float | tuple[float, ...] | list[float] = 0.0,
    good_threshold: float = 0.0,
):
    selected = torch.zeros_like(teacher_probability, dtype=torch.bool)
    quality_map = torch.zeros_like(teacher_probability)
    counts = {organ: 0 for organ in range(teacher_probability.shape[1])}
    if len(candidates) == 0:
        return selected, quality_map, counts
    statistics = candidates.evidence[:, -len(STAT_NAMES):]
    consistency = statistics[:, STAT_INDEX["augmentation_consistency"]]
    confidence = statistics[:, STAT_INDEX["mean_probability"]]
    history_stability = statistics[:, STAT_INDEX["history_stability"]]
    if isinstance(quality_threshold, (tuple, list)):
        thresholds = torch.tensor(quality_threshold, device=prediction["quality"].device)
        candidate_threshold = thresholds[candidates.organ_index]
    else:
        candidate_threshold = quality_threshold
    if isinstance(precision_threshold, (tuple, list)):
        precision_thresholds = torch.tensor(
            precision_threshold, device=prediction["quality"].device
        )[candidates.organ_index]
    else:
        precision_thresholds = precision_threshold
    if isinstance(history_threshold, (tuple, list)):
        history_thresholds = torch.tensor(
            history_threshold, device=prediction["quality"].device
        )[candidates.organ_index]
    else:
        history_thresholds = history_threshold
    valid = (
        (prediction["quality"] >= candidate_threshold)
        & (consistency >= consistency_threshold)
        & (confidence >= confidence_threshold)
        & (history_stability >= history_thresholds)
    )
    if isinstance(precision_threshold, (tuple, list)):
        precision_gate_active = any(value > 0 for value in precision_threshold)
    else:
        precision_gate_active = precision_threshold > 0
    if precision_gate_active:
        if "precision" not in prediction:
            raise ValueError("precision_threshold requires a risk prediction with precision")
        valid &= prediction["precision"] >= precision_thresholds
    if good_threshold > 0:
        if "good" not in prediction:
            raise ValueError("good_threshold requires a risk prediction with good")
        valid &= prediction["good"] >= good_threshold
    valid_counts = [
        int((valid & (candidates.organ_index == organ)).sum())
        for organ in range(teacher_probability.shape[1])
    ]
    positive_counts = [count for count in valid_counts if count > 0]
    median_count = int(torch.tensor(positive_counts).float().median()) if positive_counts else 0
    for organ in range(teacher_probability.shape[1]):
        indices = torch.where(valid & (candidates.organ_index == organ))[0]
        if len(indices) == 0:
            continue
        ranking = indices[torch.argsort(prediction["quality"][indices], descending=True)]
        if class_balance:
            quota = min(
                max_regions_per_class,
                max(min_regions_per_class, int(quota_fraction * median_count)),
            )
        else:
            quota = len(ranking)
        ranking = ranking[:quota]
        for candidate_index in ranking.tolist():
            batch_index = int(candidates.batch_index[candidate_index])
            mask = candidates.masks[candidate_index]
            if not mask.any():
                continue
            selected[batch_index, organ][mask] = True
            reliability = torch.minimum(
                prediction["quality"][candidate_index], confidence[candidate_index]
            )
            if "precision" in prediction:
                reliability = torch.minimum(
                    reliability, prediction["precision"][candidate_index]
                )
            if "good" in prediction:
                reliability = torch.minimum(reliability, prediction["good"][candidate_index])
            quality_map[batch_index, organ][mask] = reliability.detach().to(quality_map.dtype)
            counts[organ] += 1
    conflicts = selected.sum(dim=1, keepdim=True) > 1
    if conflicts.any():
        scores = quality_map * teacher_probability.detach()
        winner = scores.argmax(dim=1, keepdim=True)
        winner_mask = torch.zeros_like(selected).scatter_(1, winner, True)
        selected &= ~conflicts | winner_mask
        quality_map *= selected
        counts = {
            organ: sum(
                bool(selected[int(candidates.batch_index[index]), organ][candidates.masks[index]].any())
                for index in range(len(candidates))
                if int(candidates.organ_index[index]) == organ
            )
            for organ in range(teacher_probability.shape[1])
        }
    return selected, quality_map, counts
