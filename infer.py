from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATASET_ROOT = ROOT
COMMON_ROOT = PROJECT_ROOT / "models" / "common"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATASET_ROOT))
sys.path.insert(0, str(COMMON_ROOT))

from protocol_v1_dataset import ProtocolV1Dataset
from evaluate_shared_protocol import (
    ORGAN_ORDER, append_case_rows, parameter_counts, restore, write_metrics,
)
from reproducibility import seed_everything
from sliding_window import sliding_window_logits

from icrseg import ICRSegNNUNet


def parse_four_values(text: str, name: str, minimum=0.0, maximum=None) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) == 1:
        values *= len(ORGAN_ORDER)
    invalid = len(values) != len(ORGAN_ORDER) or any(value < minimum for value in values)
    if maximum is not None:
        invalid = invalid or any(value > maximum for value in values)
    if invalid:
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must contain one or four values >= {minimum}{suffix}")
    return values


def remove_small_components(mask: np.ndarray, minimum_voxels: int) -> np.ndarray:
    if minimum_voxels <= 1:
        return mask
    components, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel())
    keep = np.where(sizes >= minimum_voxels)[0]
    keep = keep[keep != 0]
    return np.isin(components, keep)


def probabilities_to_masks(
    probability: torch.Tensor,
    thresholds: tuple[float, ...],
    minimum_voxels: tuple[int, ...],
) -> dict[str, np.ndarray]:
    threshold_tensor = probability.new_tensor(thresholds)[:, None, None, None]
    active = probability >= threshold_tensor
    scores = probability.masked_fill(~active, -1.0)
    winner = scores.argmax(dim=0)
    any_active = active.any(dim=0)
    masks = {}
    for channel, organ in enumerate(ORGAN_ORDER):
        mask = ((winner == channel) & any_active).cpu().numpy()
        masks[organ] = remove_small_components(mask, minimum_voxels[channel])
    return masks


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


def main():
    parser = argparse.ArgumentParser(description="ICRSeg inference with the student nnU-Net only")
    parser.add_argument("--manifest", type=Path, default=Path("/path/to/project2/protocol/manifest.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-prediction", action="store_true")
    parser.add_argument(
        "--no-probabilities", action="store_true",
        help="Save segmentation masks and metrics without writing probability maps",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--sw-accumulation-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--amp", choices=("none", "bfloat16", "float16"), default="bfloat16",
        help="Mixed-precision inference dtype; bfloat16 matches the training AMP setup",
    )
    parser.add_argument(
        "--organ-thresholds", default="0.5",
        help="One shared or four liver,spleen,kidney,pancreas validation-frozen thresholds",
    )
    parser.add_argument(
        "--min-component-voxels", default="0",
        help="One shared or four validation-frozen component-size cutoffs",
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    thresholds = parse_four_values(args.organ_thresholds, "organ-thresholds", maximum=1.0)
    minimum_voxels = tuple(
        int(value) for value in parse_four_values(args.min_component_voxels, "min-component-voxels")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = ICRSegNNUNet()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("inference_state_dict")
    if state_dict is None:
        student_state = checkpoint.get("student_state_dict", checkpoint)
        prefix = "network."
        state_dict = {
            key.removeprefix(prefix): value for key, value in student_state.items()
            if key.startswith(prefix)
        }
    wrapper.inference_model().load_state_dict(state_dict)
    model = wrapper.inference_model().to(device).eval()
    del wrapper

    dataset = ProtocolV1Dataset(args.manifest, args.split, training=False, limit=args.limit)
    rows, durations = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for index, item in enumerate(dataset, 1):
        mask_paths = {
            organ: args.output / "predictions" / organ / f"{item['identifier']}.nii.gz"
            for organ in ORGAN_ORDER
        }
        probability_paths = {
            organ: args.output / "probabilities" / organ / f"{item['identifier']}.nii.gz"
            for organ in ORGAN_ORDER
        }
        if (
            args.continue_prediction
            and all(path.is_file() for path in mask_paths.values())
            and (
                args.no_probabilities
                or all(path.is_file() for path in probability_paths.values())
            )
        ):
            masks = {
                organ: sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
                for organ, path in mask_paths.items()
            }
        else:
            image = torch.from_numpy(item["image"][None]).float()
            reference = sitk.ReadImage(item["full_label"])
            start = time.perf_counter()
            amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.amp)
            autocast = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast:
                logits = sliding_window_logits(
                    model, image,
                    accumulation_device=args.sw_accumulation_device,
                    sw_batch_size=args.sw_batch_size,
                )[0]
            durations.append(time.perf_counter() - start)
            probabilities = torch.sigmoid(logits)
            masks = probabilities_to_masks(probabilities, thresholds, minimum_voxels)
            for organ in ORGAN_ORDER:
                masks[organ] = restore(
                    masks[organ], item["image_preprocessed"], reference, mask_paths[organ]
                )
                if not args.no_probabilities:
                    restore_probability(
                        probabilities[ORGAN_ORDER.index(organ)].float().detach().cpu().numpy(),
                        item["image_preprocessed"], reference, probability_paths[organ]
                    )
        append_case_rows(rows, item, masks)
        print(f"[{index}/{len(dataset)}] {item['identifier']}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    metadata = parameter_counts(model)
    metadata["seed"] = args.seed
    summary = write_metrics(rows, args.output, durations, peak, metadata)
    print(json.dumps(summary, allow_nan=True))


if __name__ == "__main__":
    main()
