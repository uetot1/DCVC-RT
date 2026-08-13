"""Train the video component of the proposed DCVC-RT VCM system.

Protocol
--------
* RGB source frames are converted to full-range BT.709 YCbCr 4:4:4 for DMCI/DMC.
* Frozen pretrained DMCI reconstructs frame 0; it is never optimized.
* DMC codes frames 1..T and is optimized with ``R + lambda_feature * D_feature``.
* ``D_feature`` matches YOLOv5 Detect-input task-pyramid layers 17/20/23,
  inspired by TransTIC's multi-level FPN feature distortion.
* Only cloned YOLOv5 layers 0..4 are trainable; layers 5..23 are the frozen
  pretrained task back end.
* Base QP is sampled uniformly from integers [0, 63]. Lambda is interpolated
  log-linearly, and evaluation uses the fixed points 0, 21, 42, 63.
* Launch with ``torchrun`` for DistributedDataParallel training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# DCVC-RT's optional fused CUDA extension is inference-only and contains
# parameter fusion that is not an autograd implementation. Force the pure
# PyTorch path before importing any codec module.
os.environ["DCVC_FORCE_TORCH"] = "1"

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.models.image_model import DMCI
from src.models.vcm_loss import VCMLoss
from src.models.vcm_system import QP_OFFSETS, VideoVCMSystem
from src.models.video_model import DMC
from src.models.yolov5_extractor import (
    DEFAULT_CLONED_FRONTEND_LAST_LAYER,
    DEFAULT_FEATURE_LAYER_INDICES,
    ddp_find_unused_parameters,
    install_cloned_frontend,
    load_yolov5,
)
from src.utils.dataset import VideoSequenceDataset
from src.utils.detection_map import DetectionMAP


DEFAULT_VALIDATION_QPS = (0, 21, 42, 63)
DEFAULT_CURRICULUM_FRAMES = (3, 5, 7)
DEFAULT_CURRICULUM_START_EPOCHS = (1, 3, 6)
FIXED_METRICS = (
    "total_loss",
    "estimated_bpp",
    "feature_mse",
    "weighted_feature_mse",
    "lambda_rd",
    "lambda_feature",
    "distortion_weight",
    "rgb_mse",
    "rgb_psnr",
    "y_mse",
    "chroma_mse",
    "base_qp",
    "coding_qp",
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    enabled: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class TrainingSchedule:
    name: str
    num_frames: int


@dataclass(frozen=True)
class RestoredState:
    start_epoch: int = 1
    best_proxy_score: float = float("inf")
    best_map5095: float = float("-inf")
    optimizer_steps: int = 0
    epoch_history: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    mean: dict[str, float]
    by_qp: dict[int, dict[str, float]]
    proxy_score: float
    map50: float | None = None
    map5095: float | None = None


def get_schedule(stage: str) -> TrainingSchedule:
    frame_counts = {"vimeo7": 7, "reds8": 8}
    if stage not in frame_counts:
        raise ValueError(f"Unknown training stage: {stage}")
    return TrainingSchedule(stage, frame_counts[stage])


def get_epoch_num_frames(args: argparse.Namespace, epoch: int) -> int:
    schedule = get_schedule(args.training_stage)
    if schedule.name != "vimeo7":
        return schedule.num_frames
    counts = tuple(args.vimeo_curriculum_frames)
    starts = tuple(args.vimeo_curriculum_start_epochs)
    if len(counts) != len(starts) or not counts:
        raise ValueError("Vimeo curriculum frame/start lists must be non-empty and equal length")
    if starts[0] != 1 or any(a >= b for a, b in zip(starts, starts[1:])):
        raise ValueError("Vimeo curriculum starts must begin at 1 and strictly increase")
    if any(a >= b for a, b in zip(counts, counts[1:])):
        raise ValueError("Vimeo curriculum frame counts must strictly increase")
    if counts[0] < 2 or counts[-1] != schedule.num_frames:
        raise ValueError(f"Vimeo curriculum must end at {schedule.num_frames} frames")
    active = counts[0]
    for start, count in zip(starts, counts, strict=True):
        if epoch < start:
            break
        active = count
    return active


def interpolate_lambda(
    base_qp: int,
    lambda_min: float,
    lambda_max: float,
    lambda_scale: float = 1.0,
) -> float:
    """Log-linearly interpolate a feature-distortion multiplier."""
    if not 0 <= int(base_qp) <= 63:
        raise ValueError("base_qp must be in [0, 63]")
    if not 0 < lambda_min <= lambda_max or lambda_scale <= 0:
        raise ValueError("lambda range and lambda_scale must be positive")
    position = int(base_qp) / 63.0
    base = math.exp(
        math.log(lambda_min)
        + position * (math.log(lambda_max) - math.log(lambda_min))
    )
    return float(lambda_scale * base)


def init_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if enabled:
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank, local_rank, world_size, device, enabled)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int, deterministic: bool) -> None:
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or None


def checkpoint_state(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "dmc_state_dict",
            checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)),
        )
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Could not find a model state dictionary in {path}")
    normalized = {key.removeprefix("module."): value for key, value in state.items()}
    return normalized, metadata


def load_image_weights(model: DMCI, path: str | Path) -> None:
    state, _ = checkpoint_state(path)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_video_initialization(system: VideoVCMSystem, path: str | Path) -> None:
    state, metadata = checkpoint_state(path)
    system.dmc.load_state_dict(state, strict=True)
    cloned_state = metadata.get("cloned_frontend_state_dict")
    if cloned_state is not None:
        expected_state = system.objective.cloned_frontend_state_dict()
        compatible_state = {
            key: value
            for key, value in cloned_state.items()
            if key in expected_state and value.shape == expected_state[key].shape
        }
        missing = sorted(set(expected_state) - set(compatible_state))
        if missing:
            print(
                "Video initialization contains an incompatible cloned front end; "
                "DMC weights were loaded and clone layers 0..4 remain initialized "
                "from the frozen YOLO teacher."
            )
        else:
            system.objective.load_cloned_frontend_state_dict(compatible_state)
            ignored = len(cloned_state) - len(compatible_state)
            if ignored:
                print(
                    "Migrated cloned YOLO layers 0..4 from a legacy checkpoint; "
                    f"ignored {ignored} state entries from obsolete deeper layers."
                )


def make_loader(
    args: argparse.Namespace,
    context: DistributedContext,
    root_dir: str,
    list_file: str | Path | None,
    training: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = VideoSequenceDataset(
        root_dir,
        list_file=list_file,
        crop_size=args.crop_size,
        num_frames=get_schedule(args.training_stage).num_frames,
        training=training,
        samples_per_sequence=(
            args.samples_per_sequence
            if training
            else args.validation_samples_per_sequence
        ),
        crop_mode=args.crop_mode,
        aware_crop_probability=args.aware_crop_probability,
        return_targets=(not training and args.select_checkpoint_by_map),
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if training and context.enabled
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        # Curriculum changes dataset.num_frames between epochs. Fresh workers
        # are required so worker-side dataset copies observe that value.
        persistent_workers=False,
    )
    return loader, sampler


def make_optimizer(system: VideoVCMSystem, args: argparse.Namespace) -> optim.AdamW:
    groups: list[dict[str, Any]] = []
    sources = (
        ("dmc", system.dmc.named_parameters(), args.learning_rate),
        (
            "cloned_frontend",
            system.objective.cloned_frontend_named_parameters(),
            args.frontend_learning_rate,
        ),
    )
    for source, named_parameters, learning_rate in sources:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            leaf = name.rsplit(".", 1)[-1]
            target = (
                no_decay
                if parameter.ndim <= 1 or leaf == "bias" or leaf.startswith("q_")
                else decay
            )
            target.append(parameter)
        if decay:
            groups.append(
                {
                    "params": decay,
                    "lr": learning_rate,
                    "weight_decay": args.weight_decay,
                    "source": source,
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                    "source": source,
                }
            )
    if not groups:
        raise RuntimeError("No trainable DMC or cloned-front-end parameters")
    return optim.AdamW(groups)


def unwrap(model: VideoVCMSystem | DDP) -> VideoVCMSystem:
    return model.module if isinstance(model, DDP) else model


def global_boolean(value: bool, context: DistributedContext) -> bool:
    flag = torch.tensor(int(value), device=context.device, dtype=torch.int32)
    if context.enabled:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def parameter_grad_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum())
    return math.sqrt(squared)


def combine_chunk_metrics(
    chunks: list[tuple[int, dict[str, torch.Tensor]]],
    seed_metrics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    frame_count = sum(length for length, _ in chunks)
    combined = {
        key: sum(length * metrics[key] for length, metrics in chunks) / frame_count
        for key in chunks[0][1]
    }
    combined.update(seed_metrics)
    return combined


def train_gop(
    model: VideoVCMSystem | DDP,
    frames: torch.Tensor,
    base_qp: int,
    lambda_feature: float,
    args: argparse.Namespace,
    context: DistributedContext,
    synchronize_last_chunk: bool,
) -> tuple[bool, dict[str, torch.Tensor] | None]:
    module = unwrap(model)
    seed_metrics = module.prepare_gop(frames[:, 0], base_qp)
    p_frame_count = frames.shape[1] - 1
    chunk_size = p_frame_count if args.tbptt_steps <= 0 else args.tbptt_steps
    chunks: list[tuple[int, dict[str, torch.Tensor]]] = []
    for first in range(1, frames.shape[1], chunk_size):
        last = min(first + chunk_size, frames.shape[1])
        is_last_chunk = last == frames.shape[1]
        should_sync = synchronize_last_chunk and is_last_chunk
        sync_context = nullcontext() if should_sync or not isinstance(model, DDP) else model.no_sync()
        with sync_context:
            chunk_loss, chunk_metrics = model(
                frames[:, first:last],
                int(base_qp),
                first,
                float(lambda_feature),
            )
            finite_loss = global_boolean(bool(torch.isfinite(chunk_loss)), context)
            if not finite_loss:
                return False, None
            weight = (last - first) / p_frame_count / args.accumulation_steps
            (chunk_loss * weight).backward()
        chunks.append((last - first, chunk_metrics))
        if args.tbptt_steps > 0 and not is_last_chunk:
            module.detach_dpb()
    return True, combine_chunk_metrics(chunks, seed_metrics)


def reduce_metrics(
    metric_sums: dict[str, float],
    count: int,
    context: DistributedContext,
) -> dict[str, float]:
    if count <= 0:
        raise RuntimeError("No valid batches were processed")
    keys = tuple(sorted(metric_sums))
    values = torch.tensor(
        [metric_sums[key] for key in keys] + [float(count)],
        device=context.device,
        dtype=torch.float64,
    )
    if context.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_count = float(values[-1].item())
    return {key: float(values[index].item() / total_count) for index, key in enumerate(keys)}


def aggregate_entries(entries: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    if not entries:
        raise RuntimeError("No validation batches were processed")
    return {
        key: sum(float(entry[key]) for entry in entries) / len(entries)
        for key in entries[0]
    }


@torch.no_grad()
def _as_yolo_image(tensor: torch.Tensor):
    return tensor.detach().clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()


def validate(
    system: VideoVCMSystem,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    map_detector=None,
) -> ValidationResult:
    system.eval()
    full_frames = get_schedule(args.training_stage).num_frames
    loader.dataset.set_num_frames(full_frames)
    entries_by_qp: dict[int, list[dict[str, torch.Tensor]]] = {
        qp: [] for qp in args.validation_qps
    }
    map_evaluators = (
        {qp: DetectionMAP() for qp in args.validation_qps}
        if map_detector is not None
        else None
    )
    next_image_id = {qp: 0 for qp in args.validation_qps}
    for batch_index, batch in enumerate(loader):
        if args.max_validation_batches is not None and batch_index >= args.max_validation_batches:
            break
        if isinstance(batch, dict):
            frames = batch["frames"].to(device, non_blocking=True)
            targets = batch["targets"]
            has_annotations = bool(batch["has_annotations"].all())
        else:
            frames = batch.to(device, non_blocking=True)
            targets = None
            has_annotations = False
        for base_qp in args.validation_qps:
            lambda_feature = interpolate_lambda(
                base_qp,
                args.lambda_min,
                args.lambda_max,
                args.lambda_scale,
            )
            seed_metrics = system.prepare_gop(frames[:, 0], base_qp)
            _, metrics = system(
                frames[:, 1:],
                base_qp,
                1,
                lambda_feature,
            )
            metrics.update(seed_metrics)
            entries_by_qp[base_qp].append(
                {key: value.detach().cpu() for key, value in metrics.items()}
            )
            if map_detector is not None and has_annotations:
                reconstructed = system.reconstruct_gop(frames, base_qp)
                detections = map_detector(
                    [
                        _as_yolo_image(reconstructed[0, frame_index])
                        for frame_index in range(reconstructed.shape[1])
                    ],
                    size=args.map_detector_size,
                ).xyxy
                for frame_index, prediction in enumerate(detections):
                    target_boxes = targets[frame_index]["boxes"][0]
                    target_classes = targets[frame_index]["classes"][0]
                    prediction = prediction.detach().cpu()
                    map_evaluators[base_qp].add(
                        image_id=next_image_id[base_qp],
                        predicted_boxes=prediction[:, :4],
                        predicted_scores=prediction[:, 4],
                        predicted_classes=prediction[:, 5].long(),
                        target_boxes=target_boxes,
                        target_classes=target_classes,
                    )
                    next_image_id[base_qp] += 1
    by_qp = {qp: aggregate_entries(entries) for qp, entries in entries_by_qp.items()}
    mean = {
        key: sum(metrics[key] for metrics in by_qp.values()) / len(by_qp)
        for key in next(iter(by_qp.values()))
    }
    mean_log_objective = sum(
        VideoVCMSystem.selection_score(metrics) for metrics in by_qp.values()
    ) / len(by_qp)
    proxy_score = math.exp(mean_log_objective)
    map50 = map5095 = None
    if map_evaluators is not None and all(evaluator.image_ids for evaluator in map_evaluators.values()):
        map_metrics = {qp: evaluator.compute() for qp, evaluator in map_evaluators.items()}
        map50 = sum(metrics["map50"] for metrics in map_metrics.values()) / len(map_metrics)
        map5095 = sum(metrics["map5095"] for metrics in map_metrics.values()) / len(map_metrics)
        for qp, metrics in map_metrics.items():
            by_qp[qp]["map50"] = float(metrics["map50"])
            by_qp[qp]["map5095"] = float(metrics["map5095"])
        mean["map50"] = float(map50)
        mean["map5095"] = float(map5095)
    return ValidationResult(mean, by_qp, proxy_score, map50, map5095)


class TrainingLogger:
    """Durable epoch logs and a run-specific plot generated during cleanup.

    Every checkpoint embeds ``history``. A resumed Kaggle run can therefore
    rebuild a cumulative CSV and plot even when only the checkpoint survived.
    """

    PLOT_METRICS = (
        ("total_loss", "Total loss"),
        ("estimated_bpp", "Estimated BPP"),
        ("feature_mse", "Feature MSE"),
    )

    def __init__(
        self,
        directory: Path,
        args: argparse.Namespace,
        metric_names: tuple[str, ...],
        previous_history: tuple[dict[str, Any], ...] = (),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        run_name = f"video_vcm_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}Z"
        self.metric_names = metric_names
        self.validation_qps = tuple(args.validation_qps)
        self.csv_path = directory / f"{run_name}.csv"
        self.batch_path = directory / f"{run_name}_batches.jsonl"
        self.plot_path = directory / f"{run_name}_training_curves.png"
        self.latest_csv_path = directory / "latest_training_history.csv"
        self.latest_plot_path = directory / "latest_training_curves.png"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.batch_file = self.batch_path.open("w", encoding="utf-8")
        self.fieldnames = [
            "epoch",
            "active_num_frames",
            "optimizer_steps",
            "dmc_learning_rate",
            "frontend_learning_rate",
            "proxy_score",
            "validation_map50",
            "validation_map5095",
            "grad_norm",
            "dmc_grad_norm",
            "frontend_grad_norm",
            "skipped_batches",
            "epoch_seconds",
            "gpu_peak_memory_mib",
            *[f"train_{key}" for key in metric_names],
            *[f"val_{key}" for key in metric_names],
            *[
                f"val_qp{qp}_{key}"
                for qp in self.validation_qps
                for key in metric_names
            ],
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.history: list[dict[str, Any]] = []
        self._plot_warning_reported = False
        for previous_row in previous_history:
            row = {key: previous_row.get(key) for key in self.fieldnames}
            self.writer.writerow(row)
            self.history.append(row)
        self._flush_epoch_log()

    def _flush_epoch_log(self) -> None:
        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())
        shutil.copy2(self.csv_path, self.latest_csv_path)

    @staticmethod
    def _numeric_series(
        history: list[dict[str, Any]],
        key: str,
    ) -> tuple[list[int], list[float]]:
        epochs: list[int] = []
        values: list[float] = []
        for row in history:
            value = row.get(key)
            if value in (None, ""):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            epochs.append(int(row["epoch"]))
            values.append(numeric)
        return epochs, values

    def update_plot(self) -> None:
        """Atomically render loss, rate and feature curves for the whole run."""
        if not self.history:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt

            figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
            for axis, (metric, title) in zip(axes, self.PLOT_METRICS, strict=True):
                train_epochs, train_values = self._numeric_series(
                    self.history,
                    f"train_{metric}",
                )
                validation_epochs, validation_values = self._numeric_series(
                    self.history,
                    f"val_{metric}",
                )
                if train_values:
                    axis.plot(
                        train_epochs,
                        train_values,
                        marker="o",
                        linewidth=2,
                        label="train",
                    )
                if validation_values:
                    axis.plot(
                        validation_epochs,
                        validation_values,
                        marker="s",
                        linewidth=2,
                        label="validation (4-QP mean)",
                    )
                axis.set_title(title)
                axis.set_xlabel("Epoch")
                axis.grid(True, alpha=0.3)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
                if train_values or validation_values:
                    axis.legend()

            last_epoch = int(self.history[-1]["epoch"])
            figure.suptitle(
                f"DCVC-RT VCM training curves - through epoch {last_epoch}"
            )
            figure.tight_layout()
            temporary = self.plot_path.with_suffix(".png.tmp")
            figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
            plt.close(figure)
            temporary.replace(self.plot_path)
            shutil.copy2(self.plot_path, self.latest_plot_path)
        except Exception as error:  # a plot failure must not kill a long run
            if not self._plot_warning_reported:
                print(f"warning: could not update training plot: {error}")
                self._plot_warning_reported = True

    def log_batch(self, record: dict[str, Any]) -> None:
        self.batch_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.batch_file.flush()

    def log_epoch(
        self,
        metadata: dict[str, Any],
        train_metrics: dict[str, float],
        validation: ValidationResult | None,
    ) -> None:
        row = dict(metadata)
        row["validation_map50"] = validation.map50 if validation else None
        row["validation_map5095"] = validation.map5095 if validation else None
        row.update({f"train_{key}": train_metrics.get(key) for key in self.metric_names})
        if validation is not None:
            row.update({f"val_{key}": validation.mean.get(key) for key in self.metric_names})
            for qp, metrics in validation.by_qp.items():
                row.update(
                    {
                        f"val_qp{qp}_{key}": metrics.get(key)
                        for key in self.metric_names
                    }
                )
        row = {key: row.get(key) for key in self.fieldnames}
        self.writer.writerow(row)
        self.history.append(row)
        self._flush_epoch_log()

    def close(self) -> None:
        if self.history:
            self.update_plot()
        self.csv_file.close()
        self.batch_file.close()


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def gather_rng_states(context: DistributedContext) -> list[dict[str, Any]]:
    local = rng_state()
    if not context.enabled:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    return [state for state in gathered if state is not None]


def restore_rng_state(states: list[dict[str, Any]] | None, context: DistributedContext) -> None:
    if not states or context.rank >= len(states):
        return
    state = states[context.rank]
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    epoch: int,
    system: VideoVCMSystem,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    validation: ValidationResult | None,
    best_proxy_score: float,
    best_map5095: float,
    optimizer_steps: int,
    rng_states: list[dict[str, Any]],
    epoch_history: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": 13,
        "epoch": epoch,
        "training_stage": args.training_stage,
        "dmc_state_dict": system.dmc.state_dict(),
        "cloned_frontend_state_dict": system.objective.cloned_frontend_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "optimizer_steps": optimizer_steps,
        "best_proxy_score": best_proxy_score,
        "best_map5095": best_map5095,
        "train_metrics": train_metrics,
        "validation_metrics": validation.mean if validation else None,
        "validation_metrics_by_qp": validation.by_qp if validation else None,
        "validation_proxy_score": validation.proxy_score if validation else None,
        "rng_states": rng_states,
        "epoch_history": list(epoch_history),
        "protocol": {
            "training_scope": "DMC plus cloned YOLO layers 0..4",
            "frozen": ["DCVC-RT DMCI", "YOLO teacher", "YOLO task backend"],
            "colour": "full-range BT.709 YCbCr 4:4:4 inside codec; RGB inside YOLO",
            "objective": "estimated_bpp + lambda_feature * multi_level_feature_mse",
            "gaussian_rate_scale_domain": "coder table [0.11,16], 128 log levels",
            "codec_training_implementation": "autograd-safe PyTorch path",
            "base_qp_sampling": "uniform integer [0, 63] per rank and batch",
            "qp_offsets": QP_OFFSETS,
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "lambda_scale": args.lambda_scale,
            "validation_qps": tuple(args.validation_qps),
            "crop_size": args.crop_size,
            "crop_mode": args.crop_mode,
            "aware_crop_probability": args.aware_crop_probability,
            "select_checkpoint_by_map": args.select_checkpoint_by_map,
            "validation_samples_per_sequence": args.validation_samples_per_sequence,
            "map_detector_size": args.map_detector_size,
            "map_confidence_threshold": args.map_confidence_threshold,
            "map_nms_iou_threshold": args.map_nms_iou_threshold,
            "map_max_detections": args.map_max_detections,
            "tbptt_steps": args.tbptt_steps,
            "vimeo_curriculum_frames": tuple(args.vimeo_curriculum_frames),
            "vimeo_curriculum_start_epochs": tuple(
                args.vimeo_curriculum_start_epochs
            ),
        },
        "feature_objective": {
            "task_model": args.task_model,
            "layer_indices": system.objective.feature_layer_indices,
            "topology": system.objective.feature_topology,
            "cloned_frontend_last_layer": (
                system.objective.cloned_frontend_last_layer
            ),
            "frozen_task_backend_layers": "5..23 plus Detect",
            "normalized_layer_weights": system.objective.layer_weights.detach().cpu().tolist(),
            "teacher_frontend": "frozen_pretrained",
            "reconstruction_frontend": (
                "jointly_trainable_clone"
                if system.objective.train_cloned_frontend
                else "frozen_clone_ablation"
            ),
            "batch_norm_statistics": "frozen",
            "inspiration": "TransTIC multi-level FPN feature distortion",
        },
        "optimizer_config": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "frontend_learning_rate": args.frontend_learning_rate,
            "weight_decay": args.weight_decay,
            "train_cloned_frontend": system.objective.train_cloned_frontend,
            "accumulation_steps": args.accumulation_steps,
            "grad_clip": args.grad_clip,
            "lr_milestones": tuple(args.lr_milestones),
            "lr_gamma": args.lr_gamma,
        },
        "source_checkpoints": {
            "image_checkpoint_sha256": sha256_file(args.image_checkpoint),
            "video_initialization_sha256": (
                sha256_file(args.video_init) if args.video_init else None
            ),
            "yolov5_weights_sha256": (
                sha256_file(args.yolov5_weights) if args.yolov5_weights else None
            ),
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_training(
    path: str | None,
    system: VideoVCMSystem,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    args: argparse.Namespace,
) -> RestoredState:
    if path is None:
        return RestoredState()
    # Keep serialized RNG byte tensors on CPU; module/optimizer loaders move
    # parameter state to the appropriate device.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    schema_version = checkpoint.get("schema_version")
    if schema_version not in (12, 13):
        raise ValueError(
            "Only schema-12/13 checkpoints can be resumed; use --video-init otherwise"
        )
    if schema_version == 12 and (
        args.crop_mode != "random" or args.select_checkpoint_by_map
    ):
        raise ValueError(
            "Schema-12 used random crops and proxy checkpoint selection. Start the "
            "new object-aware/mAP protocol with --video-init <old-checkpoint>; "
            "do not use --resume."
        )
    if checkpoint.get("training_stage") != args.training_stage:
        raise ValueError("Resume checkpoint training stage does not match --training-stage")
    saved_protocol = checkpoint.get("protocol", {})
    expected_protocol = {
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "lambda_scale": args.lambda_scale,
        "validation_qps": tuple(args.validation_qps),
        "tbptt_steps": args.tbptt_steps,
        "vimeo_curriculum_frames": tuple(args.vimeo_curriculum_frames),
        "vimeo_curriculum_start_epochs": tuple(
            args.vimeo_curriculum_start_epochs
        ),
        "crop_size": args.crop_size,
        "crop_mode": args.crop_mode,
        "aware_crop_probability": args.aware_crop_probability,
        "select_checkpoint_by_map": args.select_checkpoint_by_map,
        "validation_samples_per_sequence": args.validation_samples_per_sequence,
        "map_detector_size": args.map_detector_size,
        "map_confidence_threshold": args.map_confidence_threshold,
        "map_nms_iou_threshold": args.map_nms_iou_threshold,
        "map_max_detections": args.map_max_detections,
    }
    for key, expected in expected_protocol.items():
        saved = saved_protocol.get(key)
        if saved is not None and saved != expected:
            raise ValueError(
                f"Resume checkpoint {key}={saved!r} does not match current {expected!r}"
            )
    saved_feature = checkpoint.get("feature_objective", {})
    if tuple(saved_feature.get("layer_indices", ())) != system.objective.feature_layer_indices:
        raise ValueError("Resume checkpoint feature layers do not match current settings")
    if int(saved_feature.get("cloned_frontend_last_layer", -1)) != (
        system.objective.cloned_frontend_last_layer
    ):
        raise ValueError("Resume checkpoint cloned front-end topology does not match")
    saved_weights = tuple(saved_feature.get("normalized_layer_weights", ()))
    current_weights = tuple(system.objective.layer_weights.detach().cpu().tolist())
    if len(saved_weights) != len(current_weights) or any(
        not math.isclose(float(saved), float(current), rel_tol=1e-7, abs_tol=1e-9)
        for saved, current in zip(saved_weights, current_weights)
    ):
        raise ValueError("Resume checkpoint feature-layer weights do not match")
    saved_optimizer = checkpoint.get("optimizer_config", {})
    if bool(saved_optimizer.get("train_cloned_frontend")) != system.objective.train_cloned_frontend:
        raise ValueError("Resume checkpoint cloned-front-end setting does not match")
    expected_optimizer = {
        "learning_rate": args.learning_rate,
        "frontend_learning_rate": args.frontend_learning_rate,
        "weight_decay": args.weight_decay,
        "accumulation_steps": args.accumulation_steps,
        "grad_clip": args.grad_clip,
        "lr_milestones": tuple(args.lr_milestones),
        "lr_gamma": args.lr_gamma,
    }
    for key, expected in expected_optimizer.items():
        saved = saved_optimizer.get(key)
        if isinstance(expected, tuple) and saved is not None:
            saved = tuple(saved)
        if saved is not None and saved != expected:
            raise ValueError(
                f"Resume checkpoint optimizer {key}={saved!r} does not match "
                f"current {expected!r}"
            )
    saved_image_hash = checkpoint.get("source_checkpoints", {}).get(
        "image_checkpoint_sha256"
    )
    if saved_image_hash and saved_image_hash != sha256_file(args.image_checkpoint):
        raise ValueError("Resume checkpoint was created with a different DMCI checkpoint")
    system.dmc.load_state_dict(checkpoint["dmc_state_dict"], strict=True)
    system.objective.load_cloned_frontend_state_dict(
        checkpoint["cloned_frontend_state_dict"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint.get("rng_states"), context)
    return RestoredState(
        start_epoch=int(checkpoint["epoch"]) + 1,
        best_proxy_score=float(checkpoint.get("best_proxy_score", float("inf"))),
        best_map5095=float(checkpoint.get("best_map5095", float("-inf"))),
        optimizer_steps=int(checkpoint.get("optimizer_steps", 0)),
        epoch_history=tuple(checkpoint.get("epoch_history", ())),
    )


def copy_checkpoint(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def prune_periodic_checkpoints(directory: Path, keep: int) -> None:
    checkpoints: list[tuple[int, Path]] = []
    for path in directory.glob("epoch_*.pt"):
        match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    for _, path in checkpoints[:-keep] if keep else checkpoints:
        path.unlink()


def validate_args(args: argparse.Namespace) -> None:
    schedule = get_schedule(args.training_stage)
    if args.resume and args.video_init:
        raise ValueError("Use --resume or --video-init, not both")
    if not args.resume and not args.video_init:
        raise ValueError("A new run requires --video-init with pretrained DCVC-RT DMC weights")
    if not Path(args.image_checkpoint).is_file():
        raise FileNotFoundError(f"Image checkpoint not found: {args.image_checkpoint}")
    if args.crop_size % 16:
        raise ValueError("crop_size must be divisible by 16")
    if args.crop_size < 128:
        raise ValueError("crop_size must be at least 128")
    if not 0.0 <= args.aware_crop_probability <= 1.0:
        raise ValueError("aware_crop_probability must be in [0, 1]")
    if args.validation_samples_per_sequence < 1:
        raise ValueError("validation_samples_per_sequence must be positive")
    if args.map_detector_size < 32 or args.map_detector_size % 32:
        raise ValueError("map_detector_size must be a positive multiple of 32")
    if not 0.0 <= args.map_confidence_threshold <= 1.0:
        raise ValueError("map_confidence_threshold must be in [0, 1]")
    if not 0.0 < args.map_nms_iou_threshold <= 1.0:
        raise ValueError("map_nms_iou_threshold must be in (0, 1]")
    if args.map_max_detections < 1:
        raise ValueError("map_max_detections must be positive")
    if args.batch_size_per_gpu != 1:
        raise ValueError("DCVC-RT sequence training currently requires batch_size_per_gpu=1")
    if args.accumulation_steps < 1 or args.tbptt_steps < 0:
        raise ValueError("accumulation_steps must be positive and tbptt_steps non-negative")
    if args.learning_rate <= 0 or args.frontend_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("weight_decay must be non-negative and grad_clip positive")
    if not args.lr_milestones or any(epoch < 1 for epoch in args.lr_milestones):
        raise ValueError("lr_milestones must contain positive epoch numbers")
    if tuple(sorted(set(args.lr_milestones))) != tuple(args.lr_milestones):
        raise ValueError("lr_milestones must be unique and strictly increasing")
    if not 0 < args.lr_gamma <= 1:
        raise ValueError("lr_gamma must be in (0, 1]")
    if args.epochs < 1 or args.validate_every < 1:
        raise ValueError("epochs and validate_every must be positive")
    if args.save_every < 0 or args.keep_periodic_checkpoints < 0:
        raise ValueError("save_every and keep_periodic_checkpoints must be non-negative")
    if args.log_every < 1:
        raise ValueError("log_every must be positive")
    if len(args.validation_qps) != 4 or len(set(args.validation_qps)) != 4:
        raise ValueError("Exactly four distinct validation QPs are required")
    if any(not 0 <= qp <= 63 for qp in args.validation_qps):
        raise ValueError("Validation QPs must be in [0, 63]")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max_batches must be positive")
    if args.max_validation_batches is not None and args.max_validation_batches < 1:
        raise ValueError("max_validation_batches must be positive")
    interpolate_lambda(0, args.lambda_min, args.lambda_max, args.lambda_scale)
    get_epoch_num_frames(args, 1)
    if schedule.name == "reds8" and not (args.video_init or args.resume):
        raise ValueError("REDS fine-tuning requires a video initialization")


def run_training(args: argparse.Namespace) -> None:
    validate_args(args)
    context = init_distributed()
    logger: TrainingLogger | None = None
    try:
        seed_everything(args.seed, context.rank, args.deterministic)
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)

        train_loader, train_sampler = make_loader(
            args, context, args.data_dir, args.train_list, training=True
        )
        validation_loader = None
        if context.is_main and args.val_dir:
            validation_loader, _ = make_loader(
                args, context, args.val_dir, args.val_list, training=False
            )

        image_model = DMCI()
        load_image_weights(image_model, args.image_checkpoint)
        objective = VCMLoss(
            args.task_model,
            args.feature_layer_indices,
            args.feature_layer_weights,
            yolov5_repository=args.yolov5_repo,
            yolov5_weights=args.yolov5_weights,
            train_cloned_frontend=args.train_cloned_frontend,
            cloned_frontend_last_layer=args.cloned_frontend_last_layer,
        )
        system = VideoVCMSystem(image_model, DMC(), objective).to(context.device)
        if args.video_init:
            load_video_initialization(system, args.video_init)

        optimizer = make_optimizer(system, args)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=tuple(args.lr_milestones),
            gamma=args.lr_gamma,
        )
        restored = restore_training(
            args.resume,
            system,
            optimizer,
            scheduler,
            context,
            args,
        )
        trainable_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        dmc_parameters = [parameter for parameter in system.dmc.parameters() if parameter.requires_grad]
        frontend_parameters = [
            parameter
            for parameter in system.objective.reconstruction_extractor.parameters()
            if parameter.requires_grad
        ]

        if context.enabled:
            find_unused_parameters = ddp_find_unused_parameters(args.tbptt_steps)
            model: VideoVCMSystem | DDP = DDP(
                system,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                output_device=context.local_rank if context.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            model = system

        map_detector = None
        if context.is_main and args.select_checkpoint_by_map:
            if (
                validation_loader is None
                or not validation_loader.dataset.has_complete_object_annotations
            ):
                raise ValueError(
                    "--select-checkpoint-by-map requires every validation sequence "
                    "to have MOT annotations at <sequence>/../gt/gt.txt; do not use "
                    "the unlabeled MOT17 test split for validation mAP"
                )
            map_detector = load_yolov5(
                args.task_model,
                repository=args.yolov5_repo,
                weights=args.yolov5_weights,
            ).to(context.device).eval()
            map_detector.conf = args.map_confidence_threshold
            map_detector.iou = args.map_nms_iou_threshold
            map_detector.max_det = args.map_max_detections
            for parameter in map_detector.parameters():
                parameter.requires_grad_(False)

        checkpoint_dir = Path(args.checkpoint_dir)
        if context.is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            metric_names = (*FIXED_METRICS, *system.objective.layer_metric_names, "i_rgb_mse", "i_rgb_psnr")
            logger = TrainingLogger(
                checkpoint_dir / "logs",
                args,
                metric_names,
                previous_history=restored.epoch_history,
            )
            run_config = {
                **vars(args),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_revision": git_revision(),
                "world_size": context.world_size,
                "effective_batch_size": (
                    context.world_size
                    * args.batch_size_per_gpu
                    * args.accumulation_steps
                ),
                "qp_offsets": QP_OFFSETS,
                "image_checkpoint_sha256": sha256_file(args.image_checkpoint),
                "video_initialization_sha256": (
                    sha256_file(args.video_init) if args.video_init else None
                ),
                "color_pipeline": "RGB -> full-range BT.709 YCbCr444 -> codec -> RGB -> YOLO",
                "codec_training_implementation": "autograd-safe PyTorch path",
                "custom_cuda_inference_disabled": True,
                "gaussian_rate_scale_domain": "coder table [0.11,16], 128 log levels",
                "ddp_find_unused_parameters": ddp_find_unused_parameters(
                    args.tbptt_steps
                ),
                "feature_topology": system.objective.feature_topology,
                "cloned_frontend_layers": (
                    f"0..{system.objective.cloned_frontend_last_layer}"
                ),
            }
            (checkpoint_dir / "run_config.json").write_text(
                json.dumps(run_config, indent=2), encoding="utf-8"
            )
            print(
                f"stage={args.training_stage}, world_size={context.world_size}, "
                f"train_samples={len(train_loader.dataset)}, effective_batch="
                f"{run_config['effective_batch_size']}, qp_offsets={list(QP_OFFSETS)}"
            )

        best_proxy_score = restored.best_proxy_score
        best_map5095 = restored.best_map5095
        optimizer_steps = restored.optimizer_steps
        latest_path = checkpoint_dir / "latest.pt"
        for epoch in range(restored.start_epoch, args.epochs + 1):
            epoch_start = time.perf_counter()
            active_num_frames = get_epoch_num_frames(args, epoch)
            train_loader.dataset.set_num_frames(active_num_frames)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            metric_sums: dict[str, float] = {}
            valid_batches = 0
            skipped_batches = 0
            micro_in_window = 0
            grad_norms: list[float] = []
            dmc_grad_norms: list[float] = []
            frontend_grad_norms: list[float] = []
            total_batches = len(train_loader)
            if args.max_batches is not None:
                total_batches = min(total_batches, args.max_batches)
            progress = tqdm(
                total=total_batches,
                desc=f"epoch {epoch}/{args.epochs} ({active_num_frames} frames)",
                disable=not context.is_main,
            )
            for batch_index, frames in enumerate(train_loader):
                if batch_index >= total_batches:
                    break
                is_last_batch = batch_index + 1 == total_batches
                will_step = micro_in_window + 1 >= args.accumulation_steps or is_last_batch
                base_qp = random.randint(0, 63)
                lambda_feature = interpolate_lambda(
                    base_qp,
                    args.lambda_min,
                    args.lambda_max,
                    args.lambda_scale,
                )
                valid, details = train_gop(
                    model,
                    frames.to(context.device, non_blocking=True),
                    base_qp,
                    lambda_feature,
                    args,
                    context,
                    synchronize_last_chunk=will_step,
                )
                if not valid or details is None:
                    skipped_batches += 1
                    micro_in_window = 0
                    optimizer.zero_grad(set_to_none=True)
                    progress.update(1)
                    continue

                valid_batches += 1
                micro_in_window += 1
                for key, value in details.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach())

                if will_step:
                    if micro_in_window < args.accumulation_steps:
                        correction = args.accumulation_steps / micro_in_window
                        for parameter in trainable_parameters:
                            if parameter.grad is not None:
                                parameter.grad.mul_(correction)
                    dmc_norm = parameter_grad_norm(dmc_parameters)
                    frontend_norm = parameter_grad_norm(frontend_parameters)
                    total_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        args.grad_clip,
                        error_if_nonfinite=False,
                    )
                    gradients_finite = global_boolean(
                        bool(torch.isfinite(total_norm_tensor)), context
                    )
                    if gradients_finite:
                        optimizer.step()
                        optimizer_steps += 1
                        total_norm = float(total_norm_tensor)
                        grad_norms.append(total_norm)
                        dmc_grad_norms.append(dmc_norm)
                        frontend_grad_norms.append(frontend_norm)
                        if (
                            context.is_main
                            and logger is not None
                            and optimizer_steps % args.log_every == 0
                        ):
                            logger.log_batch(
                                {
                                    "epoch": epoch,
                                    "batch_index": batch_index,
                                    "optimizer_step": optimizer_steps,
                                    "active_num_frames": active_num_frames,
                                    "base_qp": base_qp,
                                    "lambda_feature": lambda_feature,
                                    "loss": float(details["total_loss"].detach()),
                                    "estimated_bpp": float(details["estimated_bpp"].detach()),
                                    "feature_mse": float(details["feature_mse"].detach()),
                                    "rgb_psnr": float(details["rgb_psnr"].detach()),
                                    "grad_norm_pre_clip": total_norm,
                                    "dmc_grad_norm": dmc_norm,
                                    "frontend_grad_norm": frontend_norm,
                                }
                            )
                    else:
                        skipped_batches += micro_in_window
                    optimizer.zero_grad(set_to_none=True)
                    micro_in_window = 0

                progress.set_postfix(
                    loss=f"{float(details['total_loss'].detach()):.4f}",
                    bpp=f"{float(details['estimated_bpp'].detach()):.4f}",
                    qp=base_qp,
                )
                progress.update(1)
            progress.close()

            train_metrics = reduce_metrics(metric_sums, valid_batches, context)
            skipped_tensor = torch.tensor(skipped_batches, device=context.device)
            if context.enabled:
                # Invalidity is synchronized before backward, so every rank
                # skips the same logical batch; do not multiply by world size.
                dist.all_reduce(skipped_tensor, op=dist.ReduceOp.MAX)
            skipped_total = int(skipped_tensor.item())
            if context.enabled:
                dist.barrier()

            validation = None
            should_validate = (
                validation_loader is not None
                and (epoch % args.validate_every == 0 or epoch == args.epochs)
            )
            if context.is_main and should_validate:
                if map_detector is not None:
                    install_cloned_frontend(
                        map_detector,
                        system.objective.cloned_frontend_state_dict(),
                        system.objective.cloned_frontend_last_layer,
                    )
                validation = validate(
                    system,
                    validation_loader,
                    args,
                    context.device,
                    map_detector=map_detector,
                )
            if context.enabled:
                dist.barrier()

            epoch_dmc_lr = next(
                group["lr"] for group in optimizer.param_groups if group["source"] == "dmc"
            )
            epoch_frontend_lr = next(
                (
                    group["lr"]
                    for group in optimizer.param_groups
                    if group["source"] == "cloned_frontend"
                ),
                0.0,
            )
            scheduler.step()
            rng_states = gather_rng_states(context)
            epoch_seconds = time.perf_counter() - epoch_start
            if context.is_main:
                proxy_score = (
                    validation.proxy_score
                    if validation is not None
                    else float(train_metrics["total_loss"])
                )
                improved = validation is not None and proxy_score < best_proxy_score
                if improved:
                    best_proxy_score = proxy_score
                improved_map = (
                    validation is not None
                    and validation.map5095 is not None
                    and validation.map5095 > best_map5095
                )
                if improved_map:
                    best_map5095 = float(validation.map5095)
                peak_memory = (
                    torch.cuda.max_memory_allocated(context.device) / 2**20
                    if context.device.type == "cuda"
                    else 0.0
                )
                epoch_metadata = {
                    "epoch": epoch,
                    "active_num_frames": active_num_frames,
                    "optimizer_steps": optimizer_steps,
                    "dmc_learning_rate": epoch_dmc_lr,
                    "frontend_learning_rate": epoch_frontend_lr,
                    "proxy_score": proxy_score,
                    "grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
                    "dmc_grad_norm": (
                        sum(dmc_grad_norms) / len(dmc_grad_norms) if dmc_grad_norms else 0.0
                    ),
                    "frontend_grad_norm": (
                        sum(frontend_grad_norms) / len(frontend_grad_norms)
                        if frontend_grad_norms
                        else 0.0
                    ),
                    "skipped_batches": skipped_total,
                    "epoch_seconds": epoch_seconds,
                    "gpu_peak_memory_mib": peak_memory,
                }
                assert logger is not None
                logger.log_epoch(epoch_metadata, train_metrics, validation)
                save_checkpoint(
                    latest_path,
                    epoch,
                    system,
                    optimizer,
                    scheduler,
                    args,
                    train_metrics,
                    validation,
                    best_proxy_score,
                    best_map5095,
                    optimizer_steps,
                    rng_states,
                    logger.history,
                )
                if improved:
                    copy_checkpoint(latest_path, checkpoint_dir / "best_proxy.pt")
                if improved_map:
                    copy_checkpoint(latest_path, checkpoint_dir / "best_map.pt")
                if args.save_every and epoch % args.save_every == 0:
                    copy_checkpoint(latest_path, checkpoint_dir / f"epoch_{epoch}.pt")
                    prune_periodic_checkpoints(checkpoint_dir, args.keep_periodic_checkpoints)
                print(
                    f"epoch {epoch}: loss={train_metrics['total_loss']:.6f}, "
                    f"bpp={train_metrics['estimated_bpp']:.6f}, "
                    f"feature={train_metrics['feature_mse']:.6f}, "
                    f"rgb_psnr={train_metrics['rgb_psnr']:.3f}, "
                    f"proxy={proxy_score:.6f}, "
                    f"map5095={validation.map5095 if validation else None}, "
                    f"skipped={skipped_total}"
                )
                if validation is not None:
                    for qp, metrics in validation.by_qp.items():
                        print(
                            f"  val qp={qp}: bpp={metrics['estimated_bpp']:.6f}, "
                            f"feature={metrics['feature_mse']:.6f}, "
                            f"rgb_psnr={metrics['rgb_psnr']:.3f}, "
                            f"map5095={metrics.get('map5095')}"
                        )
                if context.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(context.device)
            if context.enabled:
                # Rank 0 can spend noticeable time writing a large atomic
                # checkpoint. Keep all ranks aligned before the next epoch or
                # process-group teardown.
                dist.barrier()
    finally:
        if logger is not None:
            logger.close()
        cleanup_distributed(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-stage", choices=("vimeo7", "reds8"), default="vimeo7")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-list")
    parser.add_argument("--val-dir")
    parser.add_argument("--val-list")
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--video-init")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint-dir", default="checkpoints/vcm_vimeo7")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--batch-size-per-gpu",
        "--batch-size",
        dest="batch_size_per_gpu",
        type=int,
        default=1,
    )
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--tbptt-steps",
        type=int,
        default=0,
        help="0 keeps the complete temporal graph; use 2 or 3 only if memory requires truncation",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--crop-mode",
        choices=("random", "auto", "object", "motion"),
        default="random",
        help=(
            "Stage-2 recommendation: auto uses MOT GT boxes when available, "
            "otherwise temporal-motion crops"
        ),
    )
    parser.add_argument(
        "--aware-crop-probability",
        type=float,
        default=0.8,
        help="Fraction of training samples using object/motion-aware crop",
    )
    parser.add_argument("--samples-per-sequence", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--frontend-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=(8, 12))
    parser.add_argument("--lr-gamma", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-min", type=float, default=1.0)
    parser.add_argument("--lambda-max", type=float, default=768.0)
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=0.006,
        help=(
            "Feature-loss calibration multiplier; rerun the 0.003/0.006/0.01 "
            "pilot for the selected feature topology before the final run"
        ),
    )
    parser.add_argument(
        "--vimeo-curriculum-frames",
        type=int,
        nargs="+",
        default=DEFAULT_CURRICULUM_FRAMES,
    )
    parser.add_argument(
        "--vimeo-curriculum-start-epochs",
        type=int,
        nargs="+",
        default=DEFAULT_CURRICULUM_START_EPOCHS,
    )
    parser.add_argument(
        "--validation-qps",
        type=int,
        nargs=4,
        default=DEFAULT_VALIDATION_QPS,
        metavar=("QP1", "QP2", "QP3", "QP4"),
    )
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--max-validation-batches", type=int, default=25)
    parser.add_argument(
        "--validation-samples-per-sequence",
        type=int,
        default=4,
        help="Deterministic clips spread from start to end of each validation video",
    )
    parser.add_argument(
        "--select-checkpoint-by-map",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create best_map.pt from mean validation mAP@[.5:.95] over four QPs",
    )
    parser.add_argument("--map-detector-size", type=int, default=640)
    parser.add_argument("--map-confidence-threshold", type=float, default=0.001)
    parser.add_argument("--map-nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--map-max-detections", type=int, default=300)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--keep-periodic-checkpoints", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--task-model", default="yolov5s")
    parser.add_argument("--yolov5-repo")
    parser.add_argument("--yolov5-weights")
    parser.add_argument(
        "--feature-layer-indices",
        type=int,
        nargs="+",
        default=DEFAULT_FEATURE_LAYER_INDICES,
    )
    parser.add_argument(
        "--cloned-frontend-last-layer",
        type=int,
        default=DEFAULT_CLONED_FRONTEND_LAST_LAYER,
        help="Keep 4 for the five-layer Learned Scalable front-end protocol",
    )
    parser.add_argument("--feature-layer-weights", type=float, nargs="+")
    parser.add_argument(
        "--train-cloned-frontend",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
