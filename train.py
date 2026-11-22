from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # Keep training usable in minimal environments.
    tqdm = None


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATASET_ROOT = ROOT
COMMON_ROOT = PROJECT_ROOT / "models" / "common"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATASET_ROOT))
sys.path.insert(0, str(COMMON_ROOT))

from protocol_v1_dataset import ORGAN_ORDER, ProtocolV1Dataset, as_device
from reproducibility import seed_everything, seeded_generator, seed_worker
from sliding_window import sliding_window_logits

from qaseg import QASegNNUNet, RiskEstimator
from qaseg.augment import strong_intensity_augment, weak_intensity_augment
from qaseg.losses import consistency_loss, known_organ_loss, pseudo_label_loss, soft_exclusion_loss
from qaseg.risk import STAT_INDEX, STAT_NAMES, extract_candidates, risk_supervision_loss, select_pseudo_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train QASeg on SharedProtocol")
    parser.add_argument("--manifest", type=Path, default=Path("/path/to/project2/protocol/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("/path/to/project2/results/QASeg/SharedProtocol/full/model"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--run-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--persistent-workers", dest="persistent_workers", action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-persistent-workers", dest="persistent_workers", action="store_false",
        help="Disable persistent DataLoader workers (useful for smoke tests)",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--risk-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--diagnose-gradients", action="store_true", default=True)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument(
        "--checkpoint-epochs", default="150,200,250,300,350,400,450,500",
        help="Comma-separated 1-based epochs whose validated model weights are retained",
    )
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--sw-batch-size", type=int, default=4)
    parser.add_argument("--sw-accumulation-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument(
        "--validation-thresholds", default="0.5",
        help="One shared or four liver,spleen,kidney,pancreas validation thresholds",
    )

    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--pseudo-start-epoch", type=int, default=100)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--pseudo-ema-decay", type=float, default=0.999)
    parser.add_argument("--history-start-epoch", type=int, default=25)
    parser.add_argument("--history-snapshot-every", type=int, default=10)
    parser.add_argument("--candidate-threshold", type=float, default=0.9)
    parser.add_argument(
        "--candidate-thresholds", default="0.90,0.90,0.90,0.85",
        help="Optional liver,spleen,kidney,pancreas candidate-generation thresholds",
    )
    parser.add_argument("--risk-train-threshold", type=float, default=0.5)
    parser.add_argument("--quality-threshold", type=float, default=0.9)
    parser.add_argument(
        "--quality-thresholds", default="0.90,0.90,0.95,0.85",
        help="Optional liver,spleen,kidney,pancreas absolute quality thresholds",
    )
    parser.add_argument("--good-quality-threshold", type=float, default=0.7)
    parser.add_argument("--consistency-threshold", type=float, default=0.9)
    parser.add_argument("--pseudo-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--pseudo-precision-threshold", type=float, default=0.9)
    parser.add_argument(
        "--pseudo-precision-thresholds", default="0.90,0.90,0.90,0.85",
        help="Optional liver,spleen,kidney,pancreas pseudo-label precision thresholds",
    )
    parser.add_argument("--pseudo-history-threshold", type=float, default=0.75)
    parser.add_argument(
        "--pseudo-history-thresholds", default="0.75,0.75,0.80,0.75",
        help="Optional liver,spleen,kidney,pancreas history stability thresholds",
    )
    parser.add_argument("--pseudo-good-threshold", type=float, default=0.0)
    parser.add_argument("--max-regions-per-class", type=int, default=2)
    parser.add_argument("--quota-fraction", type=float, default=1.0)
    parser.add_argument("--min-regions-per-class", type=int, default=1)
    parser.add_argument("--min-candidate-voxels", type=int, default=16)
    parser.add_argument("--max-candidates-per-class", type=int, default=8)
    parser.add_argument("--risk-update-every", type=int, default=1)

    parser.add_argument("--lambda-intervention", type=float, default=0.5)
    parser.add_argument("--supervised-loss-weight", type=float)
    parser.add_argument(
        "--supervised-loss-weights", default="1.0,1.5,1.0,1.0",
        help="Optional liver,spleen,kidney,pancreas supervised loss weights",
    )
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--lambda-pseudo", type=float, default=0.1)
    parser.add_argument("--pseudo-ramp-epochs", type=int, default=20)
    parser.add_argument(
        "--pseudo-loss-mode", choices=("legacy", "safe", "local"), default="local"
    )
    parser.add_argument("--pseudo-local-radius", type=int, default=3)
    parser.add_argument("--lambda-exclusion", type=float, default=0.01)
    parser.add_argument("--intervention-alpha-min", type=float, default=0.25)
    parser.add_argument("--intervention-alpha-max", type=float, default=0.75)
    parser.add_argument("--foreground-sampling-probability", type=float)
    parser.add_argument(
        "--foreground-sampling-probabilities", default="0.8,0.9,0.8,0.8",
        help="Optional liver,spleen,kidney,pancreas foreground sampling probabilities",
    )
    parser.add_argument("--disable-intervention", action="store_true")
    parser.add_argument("--disable-risk", action="store_true")
    parser.add_argument(
        "--pseudo-selection", choices=("risk", "risk-basic", "confidence", "none"), default="risk",
        help="Region selector; confidence is the fixed-confidence mechanism baseline",
    )
    parser.add_argument(
        "--ablation", choices=("full", "A0", "A1", "A2", "A3"), default="full",
        help=(
            "Reproducible ablation preset. A0 is supervised-only with the "
            "shared strong intensity augmentation; A1 adds intervention and "
            "teacher consistency; A2 additionally adds basic risk pseudo labels; "
            "A3/full uses the complete risk selector."
        ),
    )
    parser.add_argument("--disable-class-balance", action="store_true")
    parser.add_argument(
        "--disable-evidence", default="",
        help=f"Comma-separated risk evidence to zero. Choices: {','.join(STAT_NAMES)}",
    )
    parser.add_argument("--disable-source-embedding", action="store_true")
    parser.add_argument("--disable-organ-embedding", action="store_true")
    return parser.parse_args()


def apply_ablation_preset(args):
    """Apply named ablation semantics before datasets/models are constructed."""
    if args.ablation == "A0":
        # A0 keeps the common strong intensity augmentation but removes every
        # proposed method-specific objective.
        args.disable_intervention = True
        args.disable_risk = True
        args.pseudo_selection = "none"
        args.lambda_intervention = 0.0
        args.lambda_consistency = 0.0
        args.lambda_pseudo = 0.0
    elif args.ablation == "A1":
        # A1 = A0 + cross-source appearance intervention + EMA consistency.
        # Risk estimation and both ordinary/risk pseudo-label paths stay off.
        args.disable_intervention = False
        args.disable_risk = True
        args.pseudo_selection = "none"
        args.lambda_intervention = 0.5
        args.lambda_consistency = 0.1
        args.lambda_pseudo = 0.0
    elif args.ablation == "A2":
        # A2 has risk-gated pseudo labels, without A3's organ-specific gates,
        # historical/precision gates, or class-balanced quota controller.
        args.disable_intervention = False
        args.disable_risk = False
        args.pseudo_selection = "risk-basic"
        args.disable_class_balance = True
        args.pseudo_confidence_threshold = 0.0
        args.pseudo_precision_threshold = 0.0
        args.pseudo_history_threshold = 0.0
        args.pseudo_good_threshold = 0.0
    elif args.ablation == "A3":
        args.disable_intervention = False
        args.disable_risk = False
        args.pseudo_selection = "risk"
    return args


def set_seed(seed: int):
    seed_everything(seed)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_checkpoint_epochs(value: str) -> set[int]:
    try:
        epochs = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as error:
        raise ValueError("--checkpoint-epochs must be comma-separated positive integers") from error
    if not epochs or any(epoch < 1 for epoch in epochs):
        raise ValueError("--checkpoint-epochs must contain at least one positive epoch")
    return epochs


def format_timestamp(value: datetime) -> str:
    return value.astimezone().isoformat(timespec="seconds")


class PhaseTimer:
    """Synchronize only in profiling runs so semantic block timings are comparable."""

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = enabled
        self.device = device
        self.values = defaultdict(list)

    def synchronize(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name: str):
        if not self.enabled:
            yield
            return
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.values[name].append(time.perf_counter() - started)

    def add(self, name: str, seconds: float):
        if self.enabled:
            self.values[name].append(seconds)

    def summary(self):
        return {
            name: {
                "count": len(values),
                "total": float(np.sum(values)),
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
            }
            for name, values in self.values.items()
            if values
        }


def organ_thresholds(value: str, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != len(ORGAN_ORDER) or any(value < 0 or value > 1 for value in values):
        raise ValueError(f"{name} must contain four values in [0,1]")
    return values


def organ_weights(value: str, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != len(ORGAN_ORDER) or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain four non-negative values")
    return values


def configured_organ_thresholds(
    args, vector_name: str, scalar_name: str, default_scalar: float, label: str
) -> tuple[float, ...]:
    vector = getattr(args, vector_name)
    scalar = getattr(args, scalar_name)
    if vector and scalar == default_scalar:
        return organ_thresholds(vector, label)
    if scalar is not None:
        if not 0 <= scalar <= 1:
            raise ValueError(f"--{scalar_name.replace('_', '-')} must be in [0,1]")
        return (scalar,) * len(ORGAN_ORDER)
    if vector:
        return organ_thresholds(vector, label)
    return (default_scalar,) * len(ORGAN_ORDER)


def validation_thresholds(args) -> tuple[float, ...]:
    values = tuple(float(value) for value in args.validation_thresholds.split(","))
    if len(values) == 1:
        values *= len(ORGAN_ORDER)
    if len(values) != len(ORGAN_ORDER) or any(value < 0 or value > 1 for value in values):
        raise ValueError("validation-thresholds must contain one or four values in [0,1]")
    return values


@torch.no_grad()
def update_ema(teacher: torch.nn.Module, student: torch.nn.Module, decay: float):
    teacher_parameters = dict(teacher.named_parameters())
    for name, parameter in student.named_parameters():
        teacher_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    teacher_buffers = dict(teacher.named_buffers())
    for name, buffer in student.named_buffers():
        teacher_buffers[name].copy_(buffer)


def dice_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    intersection = (prediction * target).sum()
    return float((2.0 * intersection / (prediction.sum() + target.sum() + 1e-8)).cpu())


def interactive_progress_enabled() -> bool:
    """Avoid terminal control characters in redirected training logs."""
    return tqdm is not None and sys.stderr.isatty()


@torch.no_grad()
def validate(model, dataset, device, thresholds, sw_batch_size, sw_accumulation_device):
    model.eval()
    known_values = {organ: [] for organ in ORGAN_ORDER}
    full_values = {organ: [] for organ in ORGAN_ORDER}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    validation_iterator = (
        tqdm(
            loader,
            total=len(loader),
            desc="Validation",
            unit="case",
            position=1,
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        if interactive_progress_enabled() else loader
    )
    validation_started = time.perf_counter()
    validation_start_time = datetime.now().astimezone()
    print(
        f"[validation] start | cases={len(loader)} "
        f"| started={format_timestamp(validation_start_time)}",
        flush=True,
    )
    for case_index, batch in enumerate(validation_iterator, 1):
        logits = sliding_window_logits(
            model, batch["image"].float(),
            accumulation_device=sw_accumulation_device,
            sw_batch_size=sw_batch_size,
        )
        task = int(batch["task_id"][0])
        probability = torch.sigmoid(logits)
        known_prediction = probability[:, task:task + 1].gt(thresholds[task]).float()
        known_values[ORGAN_ORDER[task]].append(
            dice_score(known_prediction.cpu(), batch["label"].float().cpu())
        )

        image_reference = sitk.ReadImage(batch["image_preprocessed"][0])
        full_label = sitk.ReadImage(batch["full_label"][0])
        full_label = sitk.Resample(
            full_label, image_reference, sitk.Transform(), sitk.sitkNearestNeighbor,
            0, sitk.sitkUInt8,
        )
        full_label = torch.from_numpy(sitk.GetArrayFromImage(full_label))
        for organ, name in enumerate(ORGAN_ORDER):
            prediction = probability[:, organ].gt(thresholds[organ]).float()
            target = (full_label == organ + 1).float()[None]
            full_values[name].append(dice_score(prediction.cpu(), target))
        if case_index == 1 or case_index % 10 == 0 or case_index == len(loader):
            elapsed = time.perf_counter() - validation_started
            remaining = elapsed / case_index * (len(loader) - case_index)
            finish = datetime.now().astimezone() + timedelta(seconds=remaining)
            print(
                f"[validation] case={case_index}/{len(loader)} "
                f"| elapsed={format_duration(elapsed)} "
                f"| remaining={format_duration(remaining)} "
                f"| finish={format_timestamp(finish)}",
                flush=True,
            )
    known_per_organ = {
        organ: float(np.mean(scores)) if scores else float("nan")
        for organ, scores in known_values.items()
    }
    full_per_organ = {
        organ: float(np.mean(scores)) if scores else float("nan")
        for organ, scores in full_values.items()
    }
    return {
        "known_dice": known_per_organ,
        "mean_known_dice": float(np.nanmean(list(known_per_organ.values()))),
        "full_dice": full_per_organ,
        "mean_full_dice": float(np.nanmean(list(full_per_organ.values()))),
    }


def risk_prediction(
    risk_estimator, candidates, source_id, detach_evidence=True, deterministic=False,
):
    evidence = candidates.evidence.detach() if detach_evidence else candidates.evidence
    candidate_sources = source_id[candidates.batch_index]
    was_training = risk_estimator.training
    if deterministic:
        risk_estimator.eval()
    try:
        return risk_estimator(evidence, candidates.organ_index, candidate_sources)
    finally:
        if deterministic:
            risk_estimator.train(was_training)


def checkpoint_payload(
    epoch, student, teacher, history_teacher, risk_estimator,
    optimizer, risk_optimizer, scaler, loader_generator, best, args,
):
    return {
        "epoch": epoch,
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "history_teacher_state_dict": history_teacher.state_dict(),
        "risk_state_dict": risk_estimator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "risk_optimizer_state_dict": risk_optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator_state": loader_generator.get_state(),
        "best_mean_full_dice": best,
        "organ_order": ORGAN_ORDER,
        "config": vars(args),
    }


def main():
    args = apply_ablation_preset(parse_args())
    if args.persistent_workers and args.workers == 0:
        raise ValueError("--persistent-workers requires --workers > 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be non-negative")
    if args.supervised_loss_weight is not None and args.supervised_loss_weight < 0:
        raise ValueError("--supervised-loss-weight must be non-negative")
    if args.foreground_sampling_probability is not None and not 0 <= args.foreground_sampling_probability <= 1:
        raise ValueError("--foreground-sampling-probability must be in [0,1]")
    if args.pseudo_ramp_epochs < 0:
        raise ValueError("--pseudo-ramp-epochs must be non-negative")
    if args.pseudo_local_radius < 0:
        raise ValueError("--pseudo-local-radius must be non-negative")
    for name in (
        "pseudo_confidence_threshold", "pseudo_precision_threshold",
        "pseudo_history_threshold", "pseudo_good_threshold",
    ):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.pseudo_ema_decay is not None and not 0 <= args.pseudo_ema_decay < 1:
        raise ValueError("--pseudo-ema-decay must be in [0,1)")
    if args.run_epochs is not None and args.run_epochs < 1:
        raise ValueError("--run-epochs must be positive")
    if args.pseudo_start_epoch < args.warmup_epochs:
        raise ValueError("pseudo-start-epoch must be >= warmup-epochs")
    retained_checkpoint_epochs = parse_checkpoint_epochs(args.checkpoint_epochs)
    if args.disable_risk and args.pseudo_selection in ("risk", "risk-basic"):
        args.pseudo_selection = "none"
    if not 0 <= args.intervention_alpha_min <= args.intervention_alpha_max <= 1:
        raise ValueError("intervention alpha range must satisfy 0 <= min <= max <= 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    supervised_loss_weights = organ_weights(
        args.supervised_loss_weights, "supervised-loss-weights"
    )
    if args.supervised_loss_weight is not None:
        supervised_loss_weights = (args.supervised_loss_weight,) * len(ORGAN_ORDER)
    supervised_loss_weights = torch.tensor(
        supervised_loss_weights, device=device, dtype=torch.float32
    )
    foreground_sampling_probabilities = organ_thresholds(
        args.foreground_sampling_probabilities,
        "foreground-sampling-probabilities",
    )
    if args.foreground_sampling_probability is not None:
        foreground_sampling_probabilities = (
            args.foreground_sampling_probability,
        ) * len(ORGAN_ORDER)

    train_set = ProtocolV1Dataset(
        args.manifest,
        "train",
        training=True,
        limit=args.train_limit,
        balance_sources=True,
        foreground_sampling_probability=foreground_sampling_probabilities,
    )
    val_set = ProtocolV1Dataset(args.manifest, "val", training=False, limit=args.val_limit)
    loader_generator = seeded_generator(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
        "persistent_workers": args.persistent_workers,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(train_set, **loader_kwargs)

    student = QASegNNUNet().to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    teacher.requires_grad_(False)
    history_teacher = copy.deepcopy(student).to(device).eval()
    history_teacher.requires_grad_(False)
    if args.compile:
        for module in (student, teacher, history_teacher):
            module.compile(mode=args.compile_mode)
    disabled_evidence = tuple(name for name in args.disable_evidence.split(",") if name)
    risk_estimator = RiskEstimator(
        student.risk_feature_channels,
        disabled_evidence=disabled_evidence,
        use_source_embedding=not args.disable_source_embedding,
        use_organ_embedding=not args.disable_organ_embedding,
    ).to(device)
    optimizer = torch.optim.SGD(
        student.parameters(), lr=args.learning_rate, momentum=0.99,
        nesterov=True, weight_decay=3e-5,
    )
    risk_optimizer = torch.optim.AdamW(risk_estimator.parameters(), lr=args.risk_learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )

    def abort_nonfinite(kind: str, epoch: int, step: int, values: dict):
        diagnostic = {
            "kind": kind,
            "epoch": epoch,
            "step": step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "amp_scale": scaler.get_scale(),
            "task_id": task_id.detach().cpu().tolist(),
            "values": values,
        }
        (args.output / "nonfinite_diagnostic.json").write_text(
            json.dumps(diagnostic, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )
        torch.save({
            **diagnostic,
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }, args.output / "nonfinite_diagnostic.pt")
        raise FloatingPointError(
            f"Non-finite {kind} at epoch={epoch}, step={step}; "
            f"diagnostic saved under {args.output}"
        )

    def record_amp_overflow(epoch: int, step: int, gradient_norm: float):
        event = {
            "kind": "amp_gradient_overflow",
            "epoch": epoch,
            "step": step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "amp_scale": scaler.get_scale(),
            "gradient_norm": gradient_norm,
            "task_id": task_id.detach().cpu().tolist(),
        }
        with (args.output / "amp_overflows.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=True) + "\n")

    start_epoch, best = 0, float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["student_state_dict"])
        teacher.load_state_dict(checkpoint["teacher_state_dict"])
        history_teacher.load_state_dict(
            checkpoint.get("history_teacher_state_dict", checkpoint["teacher_state_dict"])
        )
        risk_estimator.load_state_dict(checkpoint["risk_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        risk_optimizer.load_state_dict(checkpoint["risk_optimizer_state_dict"])
        if scaler.is_enabled() and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "loader_generator_state" in checkpoint:
            loader_generator.set_state(checkpoint["loader_generator_state"])
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_random_state_all"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state_all"])
        start_epoch = checkpoint["epoch"] + 1
        # Older checkpoints selected by known-organ Dice, which is not
        # comparable to the complete four-organ validation criterion.
        best = checkpoint.get("best_mean_full_dice", best)

    history_path = args.output / "history.jsonl"
    global_step = start_epoch * max(1, len(loader))
    organ_candidate_thresholds = configured_organ_thresholds(
        args, "candidate_thresholds", "candidate_threshold", 0.9, "candidate-thresholds"
    )
    organ_quality_thresholds = configured_organ_thresholds(
        args, "quality_thresholds", "quality_threshold", 0.9, "quality-thresholds"
    )
    organ_precision_thresholds = configured_organ_thresholds(
        args, "pseudo_precision_thresholds", "pseudo_precision_threshold", 0.9,
        "pseudo-precision-thresholds",
    )
    organ_history_thresholds = configured_organ_thresholds(
        args, "pseudo_history_thresholds", "pseudo_history_threshold", 0.75,
        "pseudo-history-thresholds",
    )
    if args.ablation == "A2":
        # A2 uses one shared candidate/quality gate and no A3-specific
        # precision, history, or organ-balanced controls.
        organ_candidate_thresholds = (args.candidate_threshold,) * len(ORGAN_ORDER)
        organ_quality_thresholds = (args.quality_threshold,) * len(ORGAN_ORDER)
        organ_precision_thresholds = (0.0,) * len(ORGAN_ORDER)
        organ_history_thresholds = (0.0,) * len(ORGAN_ORDER)
    organ_validation_thresholds = validation_thresholds(args)
    stage_duration_history = defaultdict(list)
    validation_duration_history = []
    default_stage_seconds = {
        "warmup": 100.0,
        "risk_calibration": 200.0,
        "pseudo_label": 255.0,
    }

    def stage_for_epoch(epoch_index: int) -> str:
        if epoch_index < args.warmup_epochs:
            return "warmup"
        if epoch_index < args.pseudo_start_epoch:
            return "risk_calibration"
        return "pseudo_label"

    stop_epoch = args.epochs
    if args.run_epochs is not None:
        stop_epoch = min(args.epochs, start_epoch + args.run_epochs)
    total_epochs = max(0, stop_epoch - start_epoch)
    training_started = time.perf_counter()
    show_progress = interactive_progress_enabled()
    progress_bar = (
        tqdm(
            total=total_epochs,
            desc="QASeg training",
            unit="epoch",
            position=0,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        if show_progress else None
    )
    start_message = (
        f"[QASeg] start epoch {start_epoch + 1}/{stop_epoch} "
        f"| total={total_epochs} | started={format_timestamp(datetime.now().astimezone())}"
    )
    if show_progress:
        tqdm.write(start_message)
    else:
        print(start_message, flush=True)

    for epoch in range(start_epoch, stop_epoch):
        epoch_started = time.perf_counter()
        timer = PhaseTimer(args.profile_timing, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        student.train()
        risk_estimator.train()
        if epoch > start_epoch and epoch % args.history_snapshot_every == 0:
            history_teacher.load_state_dict(teacher.state_dict())
            history_teacher.eval()
        stage = stage_for_epoch(epoch)
        pseudo_stage_index = max(0, epoch - args.pseudo_start_epoch)
        pseudo_ramp = (
            min(1.0, (pseudo_stage_index + 1) / args.pseudo_ramp_epochs)
            if stage == "pseudo_label" and args.pseudo_ramp_epochs > 0 else 1.0
        )
        effective_lambda_pseudo = (
            args.lambda_pseudo * pseudo_ramp if stage == "pseudo_label" else 0.0
        )
        accumulators = defaultdict(list)
        progress = epoch / max(1, args.epochs)
        learning_rate = args.learning_rate * (1.0 - progress) ** 0.9
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        data_wait_started = time.perf_counter()
        step_total = args.steps_per_epoch or len(loader)
        step_bar = (
            tqdm(
                total=step_total,
                desc=f"Epoch {epoch + 1}/{stop_epoch} [{stage}]",
                unit="step",
                position=1,
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if show_progress else None
        )
        for step, batch in enumerate(loader):
            if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
                break
            timer.add("data_wait", time.perf_counter() - data_wait_started)
            with timer.measure("data_to_device"):
                batch = as_device(batch, device)
                image = batch["image"].float()
                target = batch["label"].float()
                task_id = batch["task_id"].long()
                source_id = task_id
            with timer.measure("augmentations"):
                weak_image = weak_intensity_augment(image)
                strong_image = strong_intensity_augment(image)
                target_source = (
                    student.sample_target_sources(source_id)
                    if not args.disable_intervention else None
                )
                intervention_alpha = (
                    student.sample_intervention_alpha(
                        image, (args.intervention_alpha_min, args.intervention_alpha_max)
                    )
                    if not args.disable_intervention else None
                )

            optimizer.zero_grad(set_to_none=True)
            intervention_loss_value = 0.0
            intervention_logits_detached = None
            with timer.measure("student_intervention_forward_backward"):
                with torch.autocast(
                    device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
                ):
                    if not args.disable_intervention:
                        intervention_logits = student(
                            image, return_features=True, intervene=True, source_id=source_id,
                            target_source=target_source, alpha=intervention_alpha,
                        )[0]
                        intervention_loss, _ = known_organ_loss(intervention_logits, target, task_id)
                        if not torch.isfinite(intervention_loss.detach()).all():
                            abort_nonfinite(
                                "intervention_loss", epoch, step + 1,
                                {"intervention": float(intervention_loss.detach())},
                            )
                        intervention_logits_detached = intervention_logits.detach()
                        intervention_loss_value = float(intervention_loss.detach())
                if not args.disable_intervention:
                    scaler.scale(args.lambda_intervention * intervention_loss).backward()
                    del intervention_logits, intervention_loss

            with torch.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
            ):
                with timer.measure("student_strong_forward"):
                    strong_logits = student(
                        strong_image,
                        source_id=source_id if not args.disable_intervention else None,
                    )
                    supervised_loss, _ = known_organ_loss(
                        strong_logits,
                        target,
                        task_id,
                        sample_weights=supervised_loss_weights[task_id],
                    )

                risk_training_active = (
                    stage != "warmup"
                    and not args.disable_risk
                    and args.pseudo_selection in ("risk", "risk-basic")
                )
                pseudo_selection_active = stage == "pseudo_label" and args.pseudo_selection != "none"
                risk_evidence_active = risk_training_active or pseudo_selection_active
                teacher_forward_active = args.lambda_consistency > 0 or risk_evidence_active
                with torch.no_grad():
                    if teacher_forward_active:
                        with timer.measure("teacher_weak_forward"):
                            teacher_logits, teacher_features = teacher(weak_image, return_features=True)
                            teacher_probability = torch.sigmoid(teacher_logits)
                    else:
                        teacher_probability = None
                        teacher_features = None
                    if risk_evidence_active:
                        with timer.measure("teacher_evidence_forwards"):
                            strong_teacher_probability = torch.sigmoid(teacher(strong_image))
                            historical_probability = (
                                torch.sigmoid(history_teacher(weak_image))
                                if epoch >= args.history_start_epoch else None
                            )
                    else:
                        strong_teacher_probability = None
                        historical_probability = None
                with timer.measure("consistency_exclusion_losses"):
                    consistency = (
                        consistency_loss(strong_logits, teacher_probability, task_id)
                        if teacher_probability is not None
                        else strong_logits.sum() * 0.0
                    )
                    exclusion = soft_exclusion_loss(strong_logits)

                pseudo_loss = strong_logits.sum() * 0.0
                selected_counts = {organ: 0 for organ in range(len(ORGAN_ORDER))}
                if pseudo_selection_active:
                    with timer.measure("pseudo_candidates_selection"):
                        unknown_candidates = extract_candidates(
                            teacher_probability, torch.sigmoid(strong_logits), strong_teacher_probability,
                            teacher_features, source_id, task_id, organ_candidate_thresholds,
                            include_known=False, min_voxels=args.min_candidate_voxels,
                            max_components_per_class=args.max_candidates_per_class,
                            historical_probability=historical_probability,
                            forbidden_mask=target,
                            profile_timer=timer if args.profile_timing else None,
                        )
                        if len(unknown_candidates):
                            if args.pseudo_selection in ("risk", "risk-basic"):
                                unknown_prediction = risk_prediction(
                                    risk_estimator, unknown_candidates, source_id,
                                    detach_evidence=True, deterministic=True,
                                )
                            else:
                                confidence = unknown_candidates.evidence[
                                    :, -len(STAT_NAMES) + STAT_INDEX["mean_probability"]
                                ]
                                unknown_prediction = {"quality": confidence}
                            selected_mask, quality_map, selected_counts = select_pseudo_labels(
                                unknown_candidates, unknown_prediction, teacher_probability,
                                organ_quality_thresholds, args.consistency_threshold,
                                args.max_regions_per_class,
                                args.quota_fraction, args.min_regions_per_class,
                                class_balance=(
                                    not args.disable_class_balance
                                    and args.pseudo_selection != "risk-basic"
                                ),
                                confidence_threshold=(
                                    0.0 if args.pseudo_selection == "risk-basic"
                                    else args.pseudo_confidence_threshold
                                ),
                                precision_threshold=(
                                    (0.0,) * len(ORGAN_ORDER)
                                    if args.pseudo_selection == "risk-basic"
                                    else organ_precision_thresholds
                                ),
                                history_threshold=(
                                    (0.0,) * len(ORGAN_ORDER)
                                    if args.pseudo_selection == "risk-basic"
                                    else organ_history_thresholds
                                ),
                                good_threshold=args.pseudo_good_threshold,
                            )
                            pseudo_loss = pseudo_label_loss(
                                strong_logits, teacher_probability, selected_mask, quality_map,
                                mode=args.pseudo_loss_mode,
                                local_radius=args.pseudo_local_radius,
                            )

                segmentation_loss = (
                    supervised_loss
                    + args.lambda_consistency * consistency
                    + effective_lambda_pseudo * pseudo_loss
                    + args.lambda_exclusion * exclusion
                )

                finite_losses = {
                    "supervised": supervised_loss,
                    "consistency": consistency,
                    "pseudo": pseudo_loss,
                    "exclusion": exclusion,
                    "segmentation": segmentation_loss,
                }
                if not all(torch.isfinite(value.detach()).all() for value in finite_losses.values()):
                    abort_nonfinite(
                        "segmentation_loss", epoch, step + 1,
                        {name: float(value.detach()) for name, value in finite_losses.items()},
                    )

            with timer.measure("student_segmentation_backward_step"):
                scaler.scale(segmentation_loss).backward()
                gradient_norm_value = None
                if args.diagnose_gradients or args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    parameter_grad_norms = [
                        torch.linalg.vector_norm(
                            parameter.grad.detach(), ord=2, dtype=torch.float64
                        )
                        for parameter in student.parameters()
                        if parameter.grad is not None
                    ]
                    gradient_norm = torch.linalg.vector_norm(
                        torch.stack(parameter_grad_norms), ord=2
                    ) if parameter_grad_norms else torch.zeros((), device=device, dtype=torch.float64)
                    gradient_norm_value = float(gradient_norm.detach())
                    if not np.isfinite(gradient_norm_value):
                        if amp_enabled:
                            record_amp_overflow(epoch, step + 1, gradient_norm_value)
                        else:
                            abort_nonfinite(
                                "student_gradient", epoch, step + 1,
                                {
                                    "gradient_norm": gradient_norm_value,
                                    "supervised": float(supervised_loss.detach()),
                                    "segmentation": float(segmentation_loss.detach()),
                                },
                            )
                    elif args.max_grad_norm > 0 and gradient_norm_value > args.max_grad_norm:
                        clip_coefficient = args.max_grad_norm / (gradient_norm_value + 1e-6)
                        for parameter in student.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(clip_coefficient)
                amp_scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                amp_step_skipped = scaler.get_scale() < amp_scale_before

            risk_loss_value = 0.0
            risk_candidates_count = 0
            risk_mae_value = float("nan")
            confidence_mae_value = float("nan")
            risk_candidate_counts_by_organ = {
                organ: 0 for organ in range(len(ORGAN_ORDER))
            }
            risk_quality_error_sums_by_organ = {
                organ: 0.0 for organ in range(len(ORGAN_ORDER))
            }
            confidence_quality_error_sums_by_organ = {
                organ: 0.0 for organ in range(len(ORGAN_ORDER))
            }
            risk_quality_counts_by_organ = {
                organ: 0 for organ in range(len(ORGAN_ORDER))
            }
            if (
                risk_training_active
                and global_step % args.risk_update_every == 0
            ):
                with timer.measure("risk_candidate_generation"):
                    with torch.no_grad():
                        original_candidates = extract_candidates(
                            teacher_probability, torch.sigmoid(strong_logits.detach()),
                            strong_teacher_probability, teacher_features, source_id, task_id,
                            args.risk_train_threshold, include_known=True, target=target,
                            min_voxels=args.min_candidate_voxels,
                            max_components_per_class=args.max_candidates_per_class,
                            historical_probability=historical_probability,
                            profile_timer=timer if args.profile_timing else None,
                        )
                        if args.disable_intervention:
                            intervention_candidates = None
                        else:
                            teacher_intervention_logits, teacher_intervention_features = teacher(
                                image, return_features=True, intervene=True, source_id=source_id,
                                target_source=target_source, alpha=intervention_alpha,
                            )
                            teacher_intervention_probability = torch.sigmoid(teacher_intervention_logits)
                            historical_intervention_probability = (
                                torch.sigmoid(history_teacher(
                                    image, intervene=True, source_id=source_id,
                                    target_source=target_source, alpha=intervention_alpha,
                                )) if epoch >= args.history_start_epoch else None
                            )
                            intervention_candidates = extract_candidates(
                                teacher_intervention_probability,
                                torch.sigmoid(intervention_logits_detached),
                                teacher_probability,
                                teacher_intervention_features,
                                target_source,
                                task_id,
                                args.risk_train_threshold,
                                include_known=True,
                                target=target,
                                min_voxels=args.min_candidate_voxels,
                                max_components_per_class=args.max_candidates_per_class,
                                historical_probability=historical_intervention_probability,
                                profile_timer=timer if args.profile_timing else None,
                            )
                candidate_sets = [(original_candidates, source_id)]
                if intervention_candidates is not None:
                    candidate_sets.append((intervention_candidates, target_source))
                active_sets = [(candidates, sources) for candidates, sources in candidate_sets if len(candidates)]
                if active_sets:
                    with timer.measure("risk_optimizer_step"):
                        rng_devices = (
                            [device.index if device.index is not None else torch.cuda.current_device()]
                            if device.type == "cuda" else []
                        )
                        # Risk-head dropout must not perturb the augmentation RNG
                        # used by the segmentation branch on the following batch.
                        with torch.random.fork_rng(devices=rng_devices):
                            risk_optimizer.zero_grad(set_to_none=True)
                            losses = []
                            risk_errors = []
                            confidence_errors = []
                            for candidates, candidate_sources in active_sets:
                                prediction = risk_prediction(
                                    risk_estimator, candidates, candidate_sources,
                                    detach_evidence=True,
                                )
                                losses.append(risk_supervision_loss(
                                    prediction, candidates, args.good_quality_threshold
                                ))
                                risk_errors.append(
                                    (prediction["quality"] - candidates.quality_target).abs().mean()
                                )
                                confidence_errors.append(
                                    (
                                        candidates.evidence[:, -len(STAT_NAMES) + STAT_INDEX["mean_probability"]]
                                        - candidates.quality_target
                                    ).abs().mean()
                                )
                                quality_errors = (
                                    prediction["quality"] - candidates.quality_target
                                ).abs()
                                confidence_quality_errors = (
                                    candidates.evidence[
                                        :, -len(STAT_NAMES) + STAT_INDEX["mean_probability"]
                                    ] - candidates.quality_target
                                ).abs()
                                for organ in range(len(ORGAN_ORDER)):
                                    organ_mask = candidates.organ_index == organ
                                    organ_count = int(organ_mask.sum())
                                    if organ_count == 0:
                                        continue
                                    risk_candidate_counts_by_organ[organ] += organ_count
                                    risk_quality_counts_by_organ[organ] += organ_count
                                    risk_quality_error_sums_by_organ[organ] += float(
                                        quality_errors[organ_mask].sum().detach()
                                    )
                                    confidence_quality_error_sums_by_organ[organ] += float(
                                        confidence_quality_errors[organ_mask].sum().detach()
                                    )
                            risk_loss = torch.stack(losses).mean()
                            risk_loss.backward()
                            risk_optimizer.step()
                        risk_loss_value = float(risk_loss.detach())
                        risk_candidates_count = sum(len(candidates) for candidates, _ in active_sets)
                        risk_mae_value = float(torch.stack(risk_errors).mean().detach())
                        confidence_mae_value = float(torch.stack(confidence_errors).mean().detach())

            if teacher_forward_active:
                with timer.measure("ema_update"):
                    teacher_decay = (
                        args.pseudo_ema_decay
                        if stage == "pseudo_label" and args.pseudo_ema_decay is not None
                        else args.ema_decay
                    )
                    update_ema(teacher, student, teacher_decay)

            accumulators["total"].append(float(segmentation_loss.detach()))
            accumulators["supervised"].append(float(supervised_loss.detach()))
            accumulators["intervention"].append(intervention_loss_value)
            accumulators["consistency"].append(float(consistency.detach()))
            accumulators["pseudo"].append(float(pseudo_loss.detach()))
            accumulators["exclusion"].append(float(exclusion.detach()))
            accumulators["risk"].append(risk_loss_value)
            accumulators["risk_candidates"].append(risk_candidates_count)
            for organ, count in risk_candidate_counts_by_organ.items():
                accumulators[f"risk_candidates_{ORGAN_ORDER[organ]}"].append(count)
            if gradient_norm_value is not None and np.isfinite(gradient_norm_value):
                accumulators["gradient_norm"].append(gradient_norm_value)
            accumulators["amp_step_skipped"].append(float(amp_step_skipped))
            if np.isfinite(risk_mae_value):
                accumulators["risk_quality_mae"].append(risk_mae_value)
                accumulators["confidence_quality_mae"].append(confidence_mae_value)
            for organ, count in risk_quality_counts_by_organ.items():
                if count == 0:
                    continue
                organ_name = ORGAN_ORDER[organ]
                accumulators[f"risk_quality_mae_{organ_name}"].append(
                    risk_quality_error_sums_by_organ[organ] / count
                )
                accumulators[f"confidence_quality_mae_{organ_name}"].append(
                    confidence_quality_error_sums_by_organ[organ] / count
                )
            for organ, count in selected_counts.items():
                accumulators[f"selected_{ORGAN_ORDER[organ]}"] .append(count)
            global_step += 1
            data_wait_started = time.perf_counter()
            completed_steps = step + 1
            if step_bar is not None:
                step_bar.update(1)
                step_bar.set_postfix(
                    loss=f"{float(segmentation_loss.detach()):.4f}",
                    lr=f"{learning_rate:.2e}",
                    refresh=False,
                )
            if (
                completed_steps == 1
                or completed_steps % args.log_every_steps == 0
                or (args.steps_per_epoch is not None and completed_steps == args.steps_per_epoch)
            ):
                print(json.dumps({
                    "epoch": epoch,
                    "stage": stage,
                    "step": completed_steps,
                    "steps": args.steps_per_epoch or len(loader),
                    "supervised": float(supervised_loss.detach()),
                    "pseudo": float(pseudo_loss.detach()),
                    "risk_candidates": risk_candidates_count,
                    "gradient_norm": gradient_norm_value,
                    "amp_step_skipped": amp_step_skipped,
                    "amp_scale": scaler.get_scale(),
                }), flush=True)

        if step_bar is not None:
            step_bar.close()

        record = {
            "epoch": epoch,
            "stage": stage,
            "learning_rate": learning_rate,
            "effective_lambda_pseudo": effective_lambda_pseudo,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "loss": {key: float(np.mean(value)) for key, value in accumulators.items()},
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
            ),
        }
        stage_duration_history[stage].append(record["epoch_seconds"])
        if args.profile_timing:
            record["timing_seconds"] = timer.summary()
        should_validate = (
            not args.skip_validation
            and ((epoch + 1) % args.val_every == 0 or epoch + 1 == stop_epoch)
        )
        if should_validate:
            validation_started = time.perf_counter()
            validation = validate(
                student, val_set, device, organ_validation_thresholds,
                args.sw_batch_size, args.sw_accumulation_device,
            )
            validation_seconds = time.perf_counter() - validation_started
            validation_duration_history.append(validation_seconds)
            record["validation_seconds"] = validation_seconds
            record.update(validation)
            mean_dice = validation["mean_full_dice"]
            new_best = max(best, mean_dice)
            payload = checkpoint_payload(
                epoch, student, teacher, history_teacher, risk_estimator, optimizer, risk_optimizer,
                scaler, loader_generator, new_best, args
            )
            if epoch + 1 in retained_checkpoint_epochs:
                torch.save({
                    "epoch": epoch,
                    "student_state_dict": student.state_dict(),
                    "validation": validation,
                    "config": vars(args),
                }, args.output / f"checkpoint_epoch_{epoch:04d}.pt")
            if mean_dice > best:
                best = mean_dice
                torch.save(payload, args.output / "checkpoint_best.pt")

        stage_seconds_estimate = {
            name: (
                float(np.mean(stage_duration_history[name]))
                if stage_duration_history[name]
                else default_seconds
            )
            for name, default_seconds in default_stage_seconds.items()
        }
        remaining_seconds = sum(
            stage_seconds_estimate[stage_for_epoch(future_epoch)]
            for future_epoch in range(epoch + 1, stop_epoch)
        )
        validation_seconds_estimate = (
            float(np.mean(validation_duration_history))
            if validation_duration_history else 900.0
        )
        if not args.skip_validation:
            remaining_validations = sum(
                (future_epoch + 1) % args.val_every == 0
                or future_epoch + 1 == stop_epoch
                for future_epoch in range(epoch + 1, stop_epoch)
            )
            remaining_seconds += remaining_validations * validation_seconds_estimate
        estimated_finish = datetime.now().astimezone() + timedelta(seconds=remaining_seconds)
        record["eta"] = {
            "remaining_hours": remaining_seconds / 3600.0,
            "estimated_finish_at": estimated_finish.isoformat(timespec="seconds"),
            "stage_epoch_seconds": stage_seconds_estimate,
            "validation_seconds": validation_seconds_estimate,
        }
        latest_payload = checkpoint_payload(
            epoch, student, teacher, history_teacher, risk_estimator, optimizer, risk_optimizer,
            scaler, loader_generator, best, args
        )
        torch.save(latest_payload, args.output / "checkpoint_latest.pt")
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=True) + "\n")
        print(json.dumps(record, allow_nan=True), flush=True)
        epoch_wall_seconds = time.perf_counter() - epoch_started
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"epoch={epoch + 1}/{stop_epoch} "
                f"time={format_duration(epoch_wall_seconds)} "
                f"left={format_duration(remaining_seconds)} "
                f"finish={format_timestamp(estimated_finish)}",
                refresh=True,
            )
            tqdm.write(
                f"[QASeg] epoch {epoch + 1}/{stop_epoch} done "
                f"| stage={stage} | time={format_duration(epoch_wall_seconds)} "
                f"| remaining={format_duration(remaining_seconds)} "
                f"| finish={format_timestamp(estimated_finish)}"
            )
        else:
            print(
                f"[QASeg] epoch {epoch + 1}/{stop_epoch} done "
                f"| stage={stage} | time={format_duration(epoch_wall_seconds)} "
                f"| remaining={format_duration(remaining_seconds)} "
                f"| finish={format_timestamp(estimated_finish)}",
                flush=True,
            )

    if progress_bar is not None:
        progress_bar.close()
    total_wall_seconds = time.perf_counter() - training_started
    completion_message = (
        f"[QASeg] training complete | elapsed={format_duration(total_wall_seconds)} "
        f"| finished={format_timestamp(datetime.now().astimezone())}"
    )
    if show_progress:
        tqdm.write(completion_message)
    else:
        print(completion_message, flush=True)


if __name__ == "__main__":
    main()
