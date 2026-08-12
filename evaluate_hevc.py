"""Evaluate an HEVC x265 anchor with actual bitrate and object-detection mAP.

This script is evaluation-only. It does not import or modify the VCM training
loop. For each of four QPs it performs:

RGB frames -> FFmpeg BT.709 YUV 4:4:4 10-bit -> x265 -> FFmpeg RGB
-> frozen YOLOv5 -> mAP.

The all-frame protocol counts the complete independently decodable HEVC
bitstream and evaluates every frame, matching ``evaluate_vcm.py``. x265 is
configured as Low-Delay P: one first I-frame, no B-frames, and all following
frames are P-frames. YUV 4:2:0 is still available only for experiments whose
original source domain is YUV420.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models.yolov5_extractor import load_yolov5
from src.utils.detection_map import DetectionMAP
from src.utils.evaluation_protocol import (
    ALL_FRAMES_PROTOCOL,
    dataset_summary,
    detector_config,
    evaluation_id,
)
from src.utils.vcm_eval_dataset import AnnotatedVideoDataset, VideoSequence


RATE_POINT_COUNT = 4
PROGRESS_SCHEMA_VERSION = 1
PROTOCOL = ALL_FRAMES_PROTOCOL


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def resolve_executable(value: str, label: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise FileNotFoundError(f"{label} executable not found: {value}")


def run_command(command: list[str], label: str) -> str:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}.\n"
            f"Command: {subprocess.list2cmdline(command)}\n{tail}"
        )
    return completed.stdout


def x265_version(encoder: str) -> str:
    """Capture the encoder identity so a resumed/evaluated run is reproducible."""
    return run_command([encoder, "--version"], "x265 version").strip()


def ffconcat_quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def write_concat_file(sequence: VideoSequence, output_path: Path) -> None:
    lines = ["ffconcat version 1.0"]
    lines.extend(f"file '{ffconcat_quote(path)}'" for path in sequence.frame_paths)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def raw_frame_bytes(
    width: int,
    height: int,
    bit_depth: int,
    chroma_format: str,
) -> int:
    bytes_per_sample = 1 if bit_depth == 8 else 2
    samples_per_pixel = 3.0 if chroma_format == "444" else 1.5
    return int(width * height * samples_per_pixel * bytes_per_sample)


def x265_profile(bit_depth: int, chroma_format: str) -> str:
    """Return the explicit HEVC profile required by the raw input format."""
    profiles = {
        ("420", 8): "main",
        ("420", 10): "main10",
        ("444", 8): "main444-8",
        ("444", 10): "main444-10",
    }
    return profiles[(chroma_format, bit_depth)]


def rgb_to_yuv(
    ffmpeg: str,
    sequence: VideoSequence,
    concat_path: Path,
    yuv_path: Path,
    bit_depth: int,
    chroma_format: str,
) -> None:
    pixel_formats = {
        ("420", 8): "yuv420p",
        ("420", 10): "yuv420p10le",
        ("444", 8): "yuv444p",
        ("444", 10): "yuv444p10le",
    }
    pixel_format = pixel_formats[(chroma_format, bit_depth)]
    output_range = "pc" if chroma_format == "444" else "tv"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vsync",
        "0",
        "-frames:v",
        str(sequence.frame_count),
        "-vf",
        f"scale=in_range=pc:out_range={output_range}:out_color_matrix=bt709",
        "-pix_fmt",
        pixel_format,
        "-f",
        "rawvideo",
        "-y",
        str(yuv_path),
    ]
    run_command(command, f"FFmpeg RGB-to-YUV for {sequence.name}")
    expected_size = (
        raw_frame_bytes(sequence.width, sequence.height, bit_depth, chroma_format)
        * sequence.frame_count
    )
    actual_size = yuv_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{sequence.name}: converted YUV has {actual_size} bytes; "
            f"expected {expected_size}. Check FFmpeg frame enumeration."
        )


def x265_encode(
    encoder: str,
    sequence: VideoSequence,
    input_yuv: Path,
    bitstream_path: Path,
    qp: int,
    bit_depth: int,
    chroma_format: str,
    preset: str,
    extra_arguments: list[str],
) -> str:
    rounded_fps = int(round(sequence.fps))
    if abs(sequence.fps - rounded_fps) > 1e-6:
        raise ValueError(
            f"x265 requires an integer frame rate; {sequence.name} uses {sequence.fps}"
        )
    bitstream_path.parent.mkdir(parents=True, exist_ok=True)
    input_format = "i444" if chroma_format == "444" else "i420"
    command = [
        encoder,
        "--input",
        str(input_yuv),
        "--input-res",
        f"{sequence.width}x{sequence.height}",
        "--fps",
        str(rounded_fps),
        "--frames",
        str(sequence.frame_count),
        "--input-csp",
        input_format,
        "--input-depth",
        str(bit_depth),
        "--output-depth",
        str(bit_depth),
        "--profile",
        x265_profile(bit_depth, chroma_format),
        "--qp",
        str(qp),
        "--preset",
        preset,
        # Low-Delay P: one first I-frame then P-frames only. Fixed GOP and
        # scenecut disable source-dependent intra refreshes for reproducibility.
        "--bframes",
        "0",
        "--keyint",
        str(sequence.frame_count),
        "--min-keyint",
        str(sequence.frame_count),
        "--scenecut",
        "0",
        "--no-open-gop",
        "--repeat-headers",
        "--log-level",
        "warning",
        *extra_arguments,
        "--output",
        str(bitstream_path),
    ]
    return run_command(command, f"x265 encode for {sequence.name} at QP {qp}")


def x265_decode(
    ffmpeg: str,
    bitstream_path: Path,
    reconstructed_yuv: Path,
    bit_depth: int,
    chroma_format: str,
) -> None:
    """Decode x265 elementary stream to raw YUV for the shared RGB/YOLO path."""
    pixel_formats = {
        ("420", 8): "yuv420p",
        ("420", 10): "yuv420p10le",
        ("444", 8): "yuv444p",
        ("444", 10): "yuv444p10le",
    }
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(bitstream_path),
            "-map",
            "0:v:0",
            "-frames:v",
            str(sequence.frame_count),
            "-pix_fmt",
            pixel_formats[(chroma_format, bit_depth)],
            "-f",
            "rawvideo",
            "-y",
            str(reconstructed_yuv),
        ],
        f"FFmpeg x265 decode for {bitstream_path.name}",
    )
    expected_size = raw_frame_bytes(
        sequence.width, sequence.height, bit_depth, chroma_format
    ) * sequence.frame_count
    if reconstructed_yuv.stat().st_size != expected_size:
        raise RuntimeError(
            f"{sequence.name}: x265 decoder produced "
            f"{reconstructed_yuv.stat().st_size} bytes; expected {expected_size}"
        )


def yuv_to_rgb_frames(
    ffmpeg: str,
    sequence: VideoSequence,
    reconstructed_yuv: Path,
    output_dir: Path,
    bit_depth: int,
    chroma_format: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pixel_formats = {
        ("420", 8): "yuv420p",
        ("420", 10): "yuv420p10le",
        ("444", 8): "yuv444p",
        ("444", 10): "yuv444p10le",
    }
    pixel_format = pixel_formats[(chroma_format, bit_depth)]
    input_range = "pc" if chroma_format == "444" else "tv"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        "-video_size",
        f"{sequence.width}x{sequence.height}",
        "-framerate",
        str(sequence.fps),
        "-i",
        str(reconstructed_yuv),
        "-frames:v",
        str(sequence.frame_count),
        "-vf",
        f"scale=in_range={input_range}:out_range=pc:in_color_matrix=bt709",
        "-pix_fmt",
        "rgb24",
        "-start_number",
        "0",
        "-y",
        str(output_dir / "%08d.png"),
    ]
    run_command(command, f"FFmpeg YUV-to-RGB for {sequence.name}")
    frames = sorted(output_dir.glob("*.png"))
    if len(frames) != sequence.frame_count:
        raise RuntimeError(
            f"{sequence.name}: decoded {len(frames)} RGB frames; "
            f"expected {sequence.frame_count}"
        )
    return frames


def as_yolo_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


@torch.inference_mode()
def evaluate_reconstructions(
    detector,
    evaluator: DetectionMAP,
    dataset: AnnotatedVideoDataset,
    sequence: VideoSequence,
    reconstructed_frames: list[Path],
    detector_size: int,
    first_image_id: int,
    protocol_key: str,
    progress_label: str,
    detector_batch_size: int,
) -> int:
    first_frame = 0 if protocol_key == "all-frames" else 1
    image_id = first_image_id
    if detector_batch_size < 1:
        raise ValueError("detector_batch_size must be positive")

    with tqdm(
        total=sequence.frame_count - first_frame,
        desc=progress_label,
        unit="frame",
        leave=False,
    ) as frame_progress:
        for batch_start in range(
            first_frame, sequence.frame_count, detector_batch_size
        ):
            batch_indices = list(
                range(
                    batch_start,
                    min(batch_start + detector_batch_size, sequence.frame_count),
                )
            )
            # AutoShape accepts a list of RGB arrays and performs independent
            # letterboxing/NMS per image. Batch inference changes throughput,
            # not the evaluated frames, labels, or codec rate.
            batch_detections = detector(
                [as_yolo_image(reconstructed_frames[index]) for index in batch_indices],
                size=detector_size,
            ).xyxy
            if len(batch_detections) != len(batch_indices):
                raise RuntimeError(
                    f"YOLO returned {len(batch_detections)} outputs for "
                    f"a batch of {len(batch_indices)} frames"
                )
            for frame_index, detections in zip(
                batch_indices, batch_detections, strict=True
            ):
                target_boxes, target_classes = dataset.load_ground_truth(
                    sequence.label_paths[frame_index],
                    sequence.width,
                    sequence.height,
                )
                if (
                    len(target_classes)
                    and int(target_classes.max()) >= len(detector.names)
                ):
                    raise ValueError(
                        f"{sequence.label_paths[frame_index]} contains class "
                        f"{int(target_classes.max())}, but the task model only "
                        f"defines {len(detector.names)} classes"
                    )
                detections = detections.detach().cpu()
                evaluator.add(
                    image_id=image_id,
                    predicted_boxes=detections[:, :4],
                    predicted_scores=detections[:, 4],
                    predicted_classes=detections[:, 5].long(),
                    target_boxes=target_boxes,
                    target_classes=target_classes,
                )
                image_id += 1
            frame_progress.update(len(batch_indices))
    return image_id


def aggregate_rate(records: list[dict]) -> dict[str, float | int]:
    total_bits = sum(record["actual_bits"] for record in records)
    total_pixels = sum(
        record["coded_frames"] * record["width"] * record["height"]
        for record in records
    )
    total_duration = sum(
        record["coded_frames"] / record["fps"] for record in records
    )
    return {
        "actual_bits": int(total_bits),
        "actual_bpp": float(total_bits / total_pixels),
        "kbps": float(total_bits / total_duration / 1000.0),
        "coded_frames": int(sum(record["coded_frames"] for record in records)),
    }


def checkpoint_identity(
    args: argparse.Namespace,
    dataset: AnnotatedVideoDataset,
    sequences: list[VideoSequence],
    encoder_version: str,
) -> dict:
    """Return the inputs that must agree before an evaluation can resume."""
    return {
        "evaluation_id": evaluation_id(
            dataset,
            sequences,
            progress_description="Preparing HEVC evaluation identity",
        ),
        "qps": list(args.qps),
        "protocol": "all-frames",
        "bit_depth": args.bit_depth,
        "chroma_format": args.chroma_format,
        "x265_preset": args.preset,
        "x265_extra_arg": list(args.x265_extra_arg),
        "x265_version": encoder_version,
        "task_model": args.task_model,
        "detector_size": args.detector_size,
        "confidence_threshold": args.confidence_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "max_detections": args.max_detections,
        "yolov5_weights": str(args.yolov5_weights or "torch-hub-default"),
    }


def save_progress_checkpoint(path: Path, state: dict) -> None:
    """Atomically save progress after each finished sequence.

    DetectionMAP contains the raw predictions/labels required to calculate a
    dataset-level mAP after a resumed run, so it is intentionally saved along
    with the completed bitrate records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_progress_checkpoint(path: Path, identity: dict) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("schema_version") != PROGRESS_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported HEVC progress checkpoint: {path}. Remove it and restart."
        )
    if state.get("identity") != identity:
        raise RuntimeError(
            "The HEVC progress checkpoint was made with a different dataset, "
            "QP list, x265 configuration, or detector configuration. "
            "Choose a new --progress-checkpoint or restart without --resume."
        )
    return state


def evaluate_hevc(args: argparse.Namespace) -> None:
    if len(set(args.qps)) != RATE_POINT_COUNT:
        raise ValueError("Exactly four distinct HEVC QPs are required")
    if not torch.cuda.is_available():
        raise RuntimeError("YOLO mAP evaluation requires CUDA")

    encoder = resolve_executable(args.x265_encoder, "x265 encoder")
    ffmpeg = resolve_executable(args.ffmpeg, "FFmpeg")
    encoder_version = x265_version(encoder)
    print(f"x265 encoder: {encoder_version}")

    dataset = AnnotatedVideoDataset(args.data_dir, args.dataset_manifest)
    sequences = list(dataset)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError("No evaluation sequences were selected")
    if args.chroma_format == "420" and any(
        sequence.width % 2 or sequence.height % 2 for sequence in sequences
    ):
        raise ValueError("YUV420 evaluation requires even width and height")

    device = torch.device(f"cuda:{args.cuda_index}")
    detector = load_yolov5(
        args.task_model,
        repository=args.yolov5_repo,
        weights=args.yolov5_weights,
    ).to(device).eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    detector.conf = args.confidence_threshold
    detector.iou = args.nms_iou_threshold
    detector.max_det = args.max_detections

    method_name = safe_name(args.method_name)
    bitstream_root = Path(args.bitstream_dir) / method_name
    log_root = Path(args.encoder_log_dir) / method_name
    reconstruction_root = Path(args.reconstruction_dir) / method_name
    work_parent = Path(args.work_dir) if args.work_dir else None
    if work_parent is not None:
        work_parent.mkdir(parents=True, exist_ok=True)

    identity = checkpoint_identity(args, dataset, sequences, encoder_version)
    progress_path = (
        Path(args.progress_checkpoint)
        if args.progress_checkpoint
        else Path(args.output_dir) / f"{method_name}_progress.pt"
    )
    state = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "identity": identity,
        "points": [],
        "active": None,
    }
    if args.resume:
        if progress_path.is_file():
            state = load_progress_checkpoint(progress_path, identity)
            print(f"Resuming HEVC evaluation from {progress_path}")
        else:
            print(f"No HEVC progress checkpoint at {progress_path}; starting fresh")

    points = list(state["points"])
    completed_qps = {int(point["base_qp"]) for point in points}
    for qp in args.qps:
        if qp in completed_qps:
            print(f"{args.method_name}: x265 QP {qp} already complete; skipping")
            continue

        active = state.get("active")
        if active is not None and int(active["qp"]) == qp:
            evaluator = active["evaluator"]
            records = active["records"]
            next_image_id = int(active["next_image_id"])
            completed_names = set(active["completed_sequence_names"])
            print(
                f"{args.method_name}: resuming QP {qp}; "
                f"{len(completed_names)}/{len(sequences)} sequences complete"
            )
        else:
            evaluator = DetectionMAP()
            records = []
            next_image_id = 0
            completed_names = set()
            state["active"] = {
                "qp": qp,
                "evaluator": evaluator,
                "records": records,
                "next_image_id": next_image_id,
                "completed_sequence_names": [],
            }
            save_progress_checkpoint(progress_path, state)

        pending_sequences = [
            sequence for sequence in sequences if sequence.name not in completed_names
        ]
        progress = tqdm(
            pending_sequences,
            total=len(sequences),
            initial=len(completed_names),
            desc=f"{args.method_name}: x265 QP {qp}",
        )
        for sequence in progress:
            with tempfile.TemporaryDirectory(
                prefix=f"hevc_{safe_name(sequence.name)}_",
                dir=work_parent,
            ) as temporary_name:
                temporary = Path(temporary_name)
                concat_path = temporary / "frames.ffconcat"
                input_yuv = temporary / "input.yuv"
                reconstructed_yuv = temporary / "reconstructed.yuv"
                reconstructed_dir = temporary / "reconstructed"
                bitstream_path = (
                    bitstream_root
                    / f"qp_{qp:02d}"
                    / f"{safe_name(sequence.name)}.hevc"
                )

                write_concat_file(sequence, concat_path)
                progress.set_postfix_str(f"{sequence.name}: RGB -> YUV")
                progress.refresh()
                rgb_to_yuv(
                    ffmpeg,
                    sequence,
                    concat_path,
                    input_yuv,
                    args.bit_depth,
                    args.chroma_format,
                )
                progress.set_postfix_str(f"{sequence.name}: x265 encode")
                progress.refresh()
                encoder_log = x265_encode(
                    encoder,
                    sequence,
                    input_yuv,
                    bitstream_path,
                    qp,
                    args.bit_depth,
                    args.chroma_format,
                    args.preset,
                    args.x265_extra_arg,
                )
                log_path = (
                    log_root
                    / f"qp_{qp:02d}"
                    / f"{safe_name(sequence.name)}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(encoder_log, encoding="utf-8")

                progress.set_postfix_str(f"{sequence.name}: x265 decode -> RGB")
                progress.refresh()
                x265_decode(
                    ffmpeg,
                    sequence,
                    bitstream_path,
                    reconstructed_yuv,
                    args.bit_depth,
                    args.chroma_format,
                )
                reconstructed_frames = yuv_to_rgb_frames(
                    ffmpeg,
                    sequence,
                    reconstructed_yuv,
                    reconstructed_dir,
                    args.bit_depth,
                    args.chroma_format,
                )
                next_image_id = evaluate_reconstructions(
                    detector,
                    evaluator,
                    dataset,
                    sequence,
                    reconstructed_frames,
                    args.detector_size,
                    next_image_id,
                    "all-frames",
                    f"{args.method_name}: QP {qp} YOLO {sequence.name}",
                    args.detector_batch_size,
                )
                if args.save_reconstructions:
                    destination = (
                        reconstruction_root
                        / f"qp_{qp:02d}"
                        / safe_name(sequence.name)
                    )
                    shutil.copytree(
                        reconstructed_dir,
                        destination,
                        dirs_exist_ok=True,
                    )

                actual_bits = bitstream_path.stat().st_size * 8
                excluded_seed_bits = 0
                coded_frames = sequence.frame_count
                records.append(
                    {
                        "name": sequence.name,
                        "bitstream_file": str(bitstream_path),
                        "full_bitstream_bits": bitstream_path.stat().st_size * 8,
                        "excluded_seed_bits": excluded_seed_bits,
                        "actual_bits": actual_bits,
                        "actual_bpp": actual_bits
                        / (coded_frames * sequence.width * sequence.height),
                        "kbps": actual_bits
                        * sequence.fps
                        / (1000.0 * coded_frames),
                        "fps": sequence.fps,
                        "width": sequence.width,
                        "height": sequence.height,
                        "coded_frames": coded_frames,
                    }
                )
                completed_names.add(sequence.name)
                # The current sequence has completed encoding, decoding and
                # mAP evaluation. If Colab stops after this point, --resume
                # reuses it and only starts the next sequence.
                state["active"] = {
                    "qp": qp,
                    "evaluator": evaluator,
                    "records": records,
                    "next_image_id": next_image_id,
                    "completed_sequence_names": sorted(completed_names),
                }
                save_progress_checkpoint(progress_path, state)

        points.append(
            {
                "base_qp": qp,
                **aggregate_rate(records),
                **evaluator.compute(),
                "sequences": records,
            }
        )
        completed_qps.add(qp)
        state["points"] = points
        state["active"] = None
        save_progress_checkpoint(progress_path, state)

    protocol = PROTOCOL
    rate_source = "complete independently decodable x265 HEVC bitstream bytes including headers"
    detector_metadata = detector_config(
        args.task_model,
        args.detector_size,
        args.confidence_threshold,
        args.nms_iou_threshold,
        args.max_detections,
        args.yolov5_weights,
    )
    output = {
        "schema_version": 7,
        "method": args.method_name,
        "codec": "HEVC x265 reference encoder",
        "codec_config": {
            "encoder": encoder,
            "encoder_version": encoder_version,
            "configuration_name": args.configuration_name,
            "qps": list(args.qps),
            "preset": args.preset,
            "profile": x265_profile(args.bit_depth, args.chroma_format),
            "coding_structure": "Low-Delay P (one first I-frame, no B-frames)",
            "bit_depth": args.bit_depth,
            "chroma_format": "4:4:4" if args.chroma_format == "444" else "4:2:0",
            "color_conversion": (
                "BT.709 RGB full <-> YUV full"
                if args.chroma_format == "444"
                else "BT.709 RGB full <-> YUV420 limited"
            ),
            "extra_arguments": list(args.x265_extra_arg),
        },
        "protocol": protocol,
        "comparison_scope": "end-to-end VCM system",
        "rate_source": rate_source,
        "rate_points": RATE_POINT_COUNT,
        "task": "object_detection",
        "task_model": args.task_model,
        "ground_truth": "normalized YOLO labels from evaluation manifest",
        "evaluation_id": evaluation_id(dataset, sequences),
        "dataset": dataset_summary(dataset, sequences),
        "detector_config": detector_metadata,
        "machine_frontend": {
            "type": "pretrained_yolov5_frontend",
            "weights_id": detector_metadata["weights_id"],
            "trainable_during_codec_training": False,
            "task_backend": "frozen_pretrained_yolov5",
        },
        "points": points,
    }
    output_path = Path(args.output_dir) / f"{method_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if progress_path.is_file() and not args.keep_progress_checkpoint:
        progress_path.unlink()
    print(f"Saved HEVC actual-rate mAP results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument(
        "--x265-encoder",
        default="x265",
        help="Path to the x265 CLI executable, or an executable available on PATH",
    )
    parser.add_argument(
        "--configuration-name",
        default="x265 HEVC Low-Delay P RGB444 10-bit",
        help="Descriptive name recorded in the result JSON",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--qps",
        type=int,
        nargs=RATE_POINT_COUNT,
        default=(22, 27, 32, 37),
        metavar=("QP1", "QP2", "QP3", "QP4"),
    )
    parser.add_argument("--bit-depth", type=int, choices=(8, 10), default=10)
    parser.add_argument(
        "--chroma-format",
        choices=("420", "444"),
        default="444",
        help="Use 444 for RGB/PNG sources; use 420 only for a YUV420-source protocol",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="x265 speed/compression preset, recorded in the result JSON",
    )
    parser.add_argument(
        "--x265-extra-arg",
        action="append",
        default=[],
        help="Repeat for each additional x265 option, e.g. --x265-extra-arg=--aq-mode=0",
    )
    parser.add_argument("--method-name", default="hevc_x265_ldp_rgb444_10bit")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the automatic per-sequence HEVC progress checkpoint",
    )
    parser.add_argument(
        "--progress-checkpoint",
        help="Path for HEVC progress state (default: <output-dir>/<method>_progress.pt)",
    )
    parser.add_argument(
        "--keep-progress-checkpoint",
        action="store_true",
        help="Keep the progress checkpoint after a successful final result",
    )
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--task-model", default="yolov5s")
    parser.add_argument("--yolov5-repo")
    parser.add_argument("--yolov5-weights")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument(
        "--detector-batch-size",
        type=int,
        default=16,
        help="YOLO inference frames per GPU batch; changes speed only (default: 16)",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--bitstream-dir", default="output/hevc_bitstreams")
    parser.add_argument("--encoder-log-dir", default="output/hevc_logs")
    parser.add_argument("--output-dir", default="output/hevc_evaluation")
    parser.add_argument("--work-dir")
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument(
        "--reconstruction-dir",
        default="output/hevc_reconstructions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_hevc(parse_args())
