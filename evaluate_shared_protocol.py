from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from medpy.metric.binary import __surface_distances


ORGAN_ORDER = ("liver", "spleen", "kidney", "pancreas")
ORGAN_TO_LABEL = {"liver": 1, "spleen": 2, "kidney": 3, "pancreas": 4}


def finite_mean(values):
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def segmentation_metrics(prediction, reference, spacing_zyx):
    prediction, reference = prediction.astype(bool), reference.astype(bool)
    if not prediction.any() and not reference.any():
        return {
            "dice": float("nan"), "hd_mm": float("nan"), "hd95_mm": float("nan"),
            "asd_mm": float("nan"),
            "sensitivity": float("nan"), "precision": float("nan"), "iou": float("nan"),
        }
    if not prediction.any() or not reference.any():
        return {
            "dice": 0.0,
            "hd_mm": float("nan"),
            "hd95_mm": float("nan"),
            "asd_mm": float("nan"),
            "sensitivity": 0.0 if reference.any() else float("nan"),
            "precision": 0.0 if prediction.any() else float("nan"),
            "iou": 0.0,
        }
    true_positive = np.logical_and(prediction, reference).sum()
    false_positive = np.logical_and(prediction, ~reference).sum()
    false_negative = np.logical_and(~prediction, reference).sum()
    distance_to_reference = __surface_distances(
        prediction, reference, spacing_zyx, connectivity=1
    )
    distance_to_prediction = __surface_distances(
        reference, prediction, spacing_zyx, connectivity=1
    )
    surface_distances = np.concatenate((distance_to_reference, distance_to_prediction))
    return {
        "dice": float(2.0 * true_positive / (prediction.sum() + reference.sum())),
        "hd_mm": float(surface_distances.max()),
        "hd95_mm": float(np.percentile(surface_distances, 95)),
        "asd_mm": float(surface_distances.mean()),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "precision": float(true_positive / (true_positive + false_positive)),
        "iou": float(true_positive / (true_positive + false_positive + false_negative)),
    }


def parameter_counts(model):
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def restore(mask, preprocessed_path, reference, output):
    source = sitk.ReadImage(str(preprocessed_path))
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(source)
    image = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output), True)
    return sitk.GetArrayFromImage(image)


def restore_probability(probability, preprocessed_path, reference, output):
    """Restore a continuous probability map with linear interpolation."""
    source = sitk.ReadImage(str(preprocessed_path))
    image = sitk.GetImageFromArray(np.asarray(probability, dtype=np.float32))
    image.CopyInformation(source)
    image = sitk.Resample(
        image, reference, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output), True)
    return sitk.GetArrayFromImage(image)


def append_case_rows(rows, item, masks):
    reference_image = sitk.ReadImage(str(item["full_label"]))
    reference = sitk.GetArrayFromImage(reference_image)
    spacing = tuple(reversed(reference_image.GetSpacing()))
    task_id = item["channel"] if "channel" in item else item["task_id"]
    known_organ = ORGAN_ORDER[int(task_id)]
    for organ in ORGAN_ORDER:
        values = segmentation_metrics(
            masks[organ], reference == ORGAN_TO_LABEL[organ], spacing
        )
        row = {
            "identifier": item["identifier"], "source": item["source"], "organ": organ,
            "known_in_training": organ == known_organ,
        }
        row.update(values)
        rows.append(row)


def write_metrics(rows, output: Path, durations, peak_gpu_memory_gb, metadata=None):
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[("all_sources", "all", row["organ"])].append(row)
        groups[(row["source"], "all", row["organ"])].append(row)
        groups[(row["source"], "known" if row["known_in_training"] else "unlabeled", row["organ"])].append(row)
    summary = {
        "macro_dice": finite_mean([row["dice"] for row in rows]),
        "macro_hd_mm": finite_mean([row["hd_mm"] for row in rows]),
        "macro_hd95_mm": finite_mean([row["hd95_mm"] for row in rows]),
        "macro_asd_mm": finite_mean([row["asd_mm"] for row in rows]),
        "macro_sensitivity": finite_mean([row["sensitivity"] for row in rows]),
        "macro_precision": finite_mean([row["precision"] for row in rows]),
        "macro_iou": finite_mean([row["iou"] for row in rows]),
        "n_test_cases": len({row["identifier"] for row in rows}),
        "mean_inference_seconds": finite_mean(durations),
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "per_group": [
            {"source": key[0], "subset": key[1], "organ": key[2], "n_cases": len(group),
             "dice": finite_mean([r["dice"] for r in group]),
             "hd_mm": finite_mean([r["hd_mm"] for r in group]),
             "hd95_mm": finite_mean([r["hd95_mm"] for r in group]),
             "asd_mm": finite_mean([r["asd_mm"] for r in group]),
             "sensitivity": finite_mean([r["sensitivity"] for r in group]),
             "precision": finite_mean([r["precision"] for r in group]),
             "iou": finite_mean([r["iou"] for r in group])}
            for key, group in sorted(groups.items())
        ],
    }
    if metadata:
        summary.update(metadata)
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return summary
