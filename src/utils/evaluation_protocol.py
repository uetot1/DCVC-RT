"""Shared metadata for reproducible machine-oriented codec evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


EXTERNAL_SEED_PROTOCOL = (
    "external seed excluded; all coded P-frames included"
)
ALL_FRAMES_PROTOCOL = "all frames and complete bitstream included"


def _relative_name(path: Path, root_dir: Path) -> str:
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def evaluation_id(dataset, sequences=None, progress_description: str | None = None) -> str:
    """Fingerprint the exact sequence/frame/label set used by every codec.

    Frame contents are represented by their relative paths and byte sizes to
    avoid hashing many gigabytes for each run. Ground-truth text is hashed in
    full because labels are small and directly affect mAP.
    """
    digest = hashlib.sha256()
    digest.update(b"dcvc-rt-vcm-evaluation-v1\0")
    selected_sequences = list(dataset if sequences is None else sequences)
    total_frames = sum(sequence.frame_count for sequence in selected_sequences)
    progress = (
        tqdm(total=total_frames, desc=progress_description, unit="frame")
        if progress_description
        else None
    )
    for sequence in selected_sequences:
        sequence_metadata = {
            "name": sequence.name,
            "fps": sequence.fps,
            "width": sequence.width,
            "height": sequence.height,
            "frame_count": sequence.frame_count,
        }
        digest.update(
            json.dumps(
                sequence_metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\0")
        for frame_path, label_path in zip(
            sequence.frame_paths,
            sequence.label_paths,
            strict=True,
        ):
            frame_record = {
                "path": _relative_name(frame_path, dataset.root_dir),
                "bytes": frame_path.stat().st_size,
                "label": _relative_name(label_path, dataset.root_dir),
            }
            digest.update(
                json.dumps(
                    frame_record,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\0")
            digest.update(label_path.read_bytes())
            digest.update(b"\0")
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return f"sha256:{digest.hexdigest()}"


def detector_config(
    model_name: str,
    detector_size: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    weights: str | Path | None = None,
) -> dict[str, Any]:
    if weights is None:
        weights_id = f"torch-hub:{model_name}:ultralytics/yolov5:v7.0"
    else:
        weights_path = Path(weights).expanduser()
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Detector weights not found while fingerprinting: {weights_path}"
            )
        weights_digest = hashlib.sha256()
        with weights_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                weights_digest.update(chunk)
        weights_id = f"sha256:{weights_digest.hexdigest()}"
    return {
        "model": str(model_name),
        "weights_id": weights_id,
        "weights_scope": "pretrained initialization and frozen task backend",
        "input_size": int(detector_size),
        "confidence_threshold": float(confidence_threshold),
        "nms_iou_threshold": float(nms_iou_threshold),
        "max_detections": int(max_detections),
        "feature_repository": "ultralytics/yolov5:v7.0",
    }


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Fingerprint a model component independently of checkpoint packaging."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def dataset_summary(dataset, sequences=None) -> dict[str, int]:
    sequences = list(dataset if sequences is None else sequences)
    return {
        "sequences": len(sequences),
        "source_frames": sum(sequence.frame_count for sequence in sequences),
    }
