"""All-frame machine-task evaluation for DCVC-RT and the proposed VCM codec.

Training remains differentiable and uses ``DMC.forward_train``. This script is
evaluation-only: frozen DMCI codes frame 0, DMC codes all P-frames, and actual
sequence-container bytes include both. Reconstructed BT.709 YCbCr is converted
to RGB before measuring detector mAP against real labels at four or more rate points.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as transforms
from tqdm import tqdm

from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.models.yolov5_extractor import install_cloned_frontend, load_yolov5
from src.utils.bd_rate import compute_bd_metric, compute_bd_rate, pareto_front
from src.utils.detection_map import DetectionMAP
from src.utils.evaluation_protocol import (
    ALL_FRAMES_PROTOCOL,
    dataset_summary,
    detector_config,
    evaluation_id,
    state_dict_sha256,
)
from src.utils.vcm_bitstream import FRAME_HEADER, VCMSequenceReader, VCMSequenceWriter
from src.utils.vcm_eval_dataset import AnnotatedVideoDataset, VideoSequence
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb


QP_OFFSETS = (0, 8, 0, 4, 0, 4, 0, 4)
MIN_RATE_POINT_COUNT = 4
TWO_ENTROPY_CODER_PIXEL_THRESHOLD = 1280 * 720
TEMPORAL_BIN_ORDER = (
    "frame_0",
    "frames_1_7",
    "frames_8_31",
    "frames_32_63",
    "frames_64_plus",
)


def temporal_bin_name(frame_index: int) -> str:
    """Map a zero-based frame index to the long-sequence diagnostic bins."""
    frame_index = int(frame_index)
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if frame_index == 0:
        return "frame_0"
    if frame_index <= 7:
        return "frames_1_7"
    if frame_index <= 31:
        return "frames_8_31"
    if frame_index <= 63:
        return "frames_32_63"
    return "frames_64_plus"


def load_codec_checkpoint(model, path: str | Path, state_key: str | None = None) -> dict:
    """Load official or project checkpoint weights and return its metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            state_key or "dmc_state_dict",
            checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)),
        )
    else:
        state = checkpoint
    model.load_state_dict(
        {key.removeprefix("module."): value for key, value in state.items()}
    )
    return checkpoint if isinstance(checkpoint, dict) else {}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "method"


def pad_frame(frame: torch.Tensor, model, device: torch.device) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    frame = frame.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
    padding_right, padding_bottom = model.get_padding_size(
        frame.shape[-2],
        frame.shape[-1],
        16,
    )
    return F.pad(frame, (0, padding_right, 0, padding_bottom), mode="replicate")


def as_yolo_image(frame: torch.Tensor) -> np.ndarray:
    return (
        frame.detach()
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .clip(0, 1)
        * 255
    ).astype(np.uint8)


def coding_qp(base_qp: int, frame_index: int) -> int:
    return base_qp + QP_OFFSETS[frame_index % len(QP_OFFSETS)]


def use_two_entropy_coders(width: int, height: int) -> bool:
    return width * height > TWO_ENTROPY_CODER_PIXEL_THRESHOLD


@torch.inference_mode()
def encode_sequence(
    image_model: DMCI,
    model: DMC,
    dataset: AnnotatedVideoDataset,
    sequence: VideoSequence,
    base_qp: int,
    output_path: Path,
    device: torch.device,
    reset_interval: int,
) -> list[float]:
    """Encode one DMCI I-frame and every following DMC P-frame."""
    two_entropy_coders = use_two_entropy_coders(sequence.width, sequence.height)
    image_model.set_use_two_entropy_coders(two_entropy_coders)
    model.set_use_two_entropy_coders(two_entropy_coders)
    model.clear_dpb()
    model.set_curr_poc(0)
    seed_rgb = pad_frame(dataset.load_frame(sequence.frame_paths[0]), image_model, device)
    encoded_i = image_model.compress(rgb2ycbcr(seed_rgb), base_qp)
    model.add_ref_frame(feature=None, frame=encoded_i["x_hat"])
    estimated_entropy_bits = [float(encoded_i["estimated_entropy_bits"])]

    with VCMSequenceWriter(
        output_path,
        width=sequence.width,
        height=sequence.height,
        fps=sequence.fps,
        coded_frames=sequence.frame_count,
        external_seed=False,
        two_entropy_coders=two_entropy_coders,
        reset_interval=reset_interval,
    ) as writer:
        writer.write_frame(base_qp, encoded_i["bit_stream"])
        last_qp = 0
        for frame_index in range(1, sequence.frame_count):
            frame_rgb = pad_frame(
                dataset.load_frame(sequence.frame_paths[frame_index]),
                model,
                device,
            )
            if reset_interval > 0 and frame_index % reset_interval == 1:
                model.prepare_feature_adaptor_i(last_qp)
            qp = coding_qp(base_qp, frame_index)
            encoded = model.compress(rgb2ycbcr(frame_rgb), qp)
            writer.write_frame(qp, encoded["bit_stream"])
            estimated_entropy_bits.append(float(encoded["estimated_entropy_bits"]))
            last_qp = qp
    torch.cuda.synchronize(device)
    return estimated_entropy_bits


@torch.inference_mode()
def decode_and_evaluate_sequence(
    image_model: DMCI,
    model: DMC,
    detector,
    evaluator: DetectionMAP,
    dataset: AnnotatedVideoDataset,
    sequence: VideoSequence,
    bitstream_path: Path,
    device: torch.device,
    detector_size: int,
    first_image_id: int,
    reconstruction_dir: Path | None,
    temporal_evaluators: dict[str, DetectionMAP] | None = None,
) -> int:
    """Decode a sequence and add ground-truth detection results to mAP."""
    model.clear_dpb()
    model.set_curr_poc(0)
    with VCMSequenceReader(bitstream_path) as reader:
        header = reader.header
        if (header.width, header.height) != (sequence.width, sequence.height):
            raise ValueError(f"Bitstream resolution mismatch for {sequence.name}")
        if header.coded_frames != sequence.frame_count:
            raise ValueError(f"Bitstream frame-count mismatch for {sequence.name}")
        if header.external_seed:
            raise ValueError("All-frame evaluation requires a coded DMCI I-frame")

        sps = {
            "height": sequence.height,
            "width": sequence.width,
            "ec_part": int(header.two_entropy_coders),
            "use_ada_i": 0,
        }
        image_id = first_image_id
        for frame_index, packet in enumerate(reader.frames()):
            if frame_index == 0:
                decoded = image_model.decompress(packet.bitstream, sps, packet.qp)
                model.add_ref_frame(feature=None, frame=decoded["x_hat"])
            else:
                if (
                    header.reset_interval > 0
                    and frame_index % header.reset_interval == 1
                ):
                    model.reset_ref_feature()
                decoded = model.decompress(packet.bitstream, sps, packet.qp)
            reconstructed_ycbcr = decoded["x_hat"][
                :,
                :,
                : sequence.height,
                : sequence.width,
            ]
            reconstructed = ycbcr2rgb(reconstructed_ycbcr)
            detections = detector(
                [as_yolo_image(reconstructed)],
                size=detector_size,
            ).xyxy[0].detach().cpu()

            target_boxes, target_classes = dataset.load_ground_truth(
                sequence.label_paths[frame_index],
                sequence.width,
                sequence.height,
            )
            if len(target_classes) and int(target_classes.max()) >= len(detector.names):
                raise ValueError(
                    f"{sequence.label_paths[frame_index]} contains class "
                    f"{int(target_classes.max())}, but the task model only "
                    f"defines {len(detector.names)} classes"
                )
            evaluator.add(
                image_id=image_id,
                predicted_boxes=detections[:, :4],
                predicted_scores=detections[:, 4],
                predicted_classes=detections[:, 5].long(),
                target_boxes=target_boxes,
                target_classes=target_classes,
            )
            if temporal_evaluators is not None:
                temporal_evaluators[temporal_bin_name(frame_index)].add(
                    image_id=image_id,
                    predicted_boxes=detections[:, :4],
                    predicted_scores=detections[:, 4],
                    predicted_classes=detections[:, 5].long(),
                    target_boxes=target_boxes,
                    target_classes=target_classes,
                )

            if reconstruction_dir is not None:
                output_path = (
                    reconstruction_dir
                    / safe_name(sequence.name)
                    / sequence.frame_paths[frame_index].with_suffix(".png").name
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                transforms.to_pil_image(reconstructed[0].float().cpu()).save(output_path)
            image_id += 1

    torch.cuda.synchronize(device)
    return image_id


def sequence_rate_record(
    sequence: VideoSequence,
    bitstream_path: Path,
    estimated_entropy_bits_by_frame: list[float],
) -> dict:
    coded_frames = sequence.frame_count
    actual_bits = bitstream_path.stat().st_size * 8
    if len(estimated_entropy_bits_by_frame) != coded_frames:
        raise ValueError(
            f"Estimated-rate frame count mismatch for {sequence.name}: "
            f"{len(estimated_entropy_bits_by_frame)} != {coded_frames}"
        )
    actual_bits_by_frame = container_bits_by_frame(bitstream_path)
    if len(actual_bits_by_frame) != coded_frames:
        raise ValueError(f"Container frame count mismatch for {sequence.name}")
    estimated_entropy_bits = float(sum(estimated_entropy_bits_by_frame))
    total_pixels = coded_frames * sequence.width * sequence.height
    if estimated_entropy_bits <= 0:
        raise ValueError(f"Non-positive estimated entropy rate for {sequence.name}")
    temporal_bins: dict[str, dict[str, float | int]] = {}
    pixels_per_frame = sequence.width * sequence.height
    for frame_index, (frame_actual_bits, frame_estimated_bits) in enumerate(
        zip(actual_bits_by_frame, estimated_entropy_bits_by_frame, strict=True)
    ):
        name = temporal_bin_name(frame_index)
        entry = temporal_bins.setdefault(
            name,
            {
                "actual_bits": 0,
                "estimated_entropy_bits": 0.0,
                "pixels": 0,
                "coded_frames": 0,
            },
        )
        entry["actual_bits"] += int(frame_actual_bits)
        entry["estimated_entropy_bits"] += float(frame_estimated_bits)
        entry["pixels"] += pixels_per_frame
        entry["coded_frames"] += 1

    return {
        "name": sequence.name,
        "bitstream_file": str(bitstream_path),
        "actual_bits": actual_bits,
        "actual_bpp": actual_bits / total_pixels,
        "estimated_entropy_bits": float(estimated_entropy_bits),
        "estimated_bpp": float(estimated_entropy_bits / total_pixels),
        "actual_to_estimated_bpp_ratio": float(actual_bits / estimated_entropy_bits),
        "actual_minus_estimated_bits": float(actual_bits - estimated_entropy_bits),
        "kbps": actual_bits * sequence.fps / (1000.0 * coded_frames),
        "fps": sequence.fps,
        "width": sequence.width,
        "height": sequence.height,
        "coded_frames": coded_frames,
        "temporal_bins": temporal_bins,
    }


def container_bits_by_frame(bitstream_path: Path) -> list[int]:
    """Return exact container bits per frame, assigning sequence header to I."""
    with VCMSequenceReader(bitstream_path) as reader:
        sequence_header_bits = reader.file.tell() * 8
        frame_bits = [
            (FRAME_HEADER.size + len(packet.bitstream)) * 8
            for packet in reader.frames()
        ]
    if not frame_bits:
        raise ValueError(f"Container has no coded frames: {bitstream_path}")
    frame_bits[0] += sequence_header_bits
    if sum(frame_bits) != bitstream_path.stat().st_size * 8:
        raise RuntimeError(f"Per-frame container accounting failed: {bitstream_path}")
    return frame_bits


def aggregate_rate(sequence_records: list[dict]) -> dict[str, float | int]:
    total_bits = sum(record["actual_bits"] for record in sequence_records)
    estimated_entropy_bits = sum(
        record["estimated_entropy_bits"] for record in sequence_records
    )
    if estimated_entropy_bits <= 0:
        raise ValueError("Aggregate estimated entropy rate must be positive")
    total_pixels = sum(
        record["coded_frames"] * record["width"] * record["height"]
        for record in sequence_records
    )
    total_duration = sum(
        record["coded_frames"] / record["fps"] for record in sequence_records
    )
    return {
        "actual_bits": int(total_bits),
        "actual_bpp": float(total_bits / total_pixels),
        "estimated_entropy_bits": float(estimated_entropy_bits),
        "estimated_bpp": float(estimated_entropy_bits / total_pixels),
        "actual_to_estimated_bpp_ratio": float(total_bits / estimated_entropy_bits),
        "actual_minus_estimated_bits": float(total_bits - estimated_entropy_bits),
        "kbps": float(total_bits / total_duration / 1000.0),
        "coded_frames": int(
            sum(record["coded_frames"] for record in sequence_records)
        ),
    }


def aggregate_temporal_rate(sequence_records: list[dict]) -> dict[str, dict]:
    totals: dict[str, dict[str, float | int]] = {}
    for record in sequence_records:
        for name, values in record["temporal_bins"].items():
            entry = totals.setdefault(
                name,
                {
                    "actual_bits": 0,
                    "estimated_entropy_bits": 0.0,
                    "pixels": 0,
                    "coded_frames": 0,
                },
            )
            for key in entry:
                entry[key] += values[key]

    rates = {}
    for name in TEMPORAL_BIN_ORDER:
        if name not in totals:
            continue
        entry = totals[name]
        estimated = float(entry["estimated_entropy_bits"])
        pixels = int(entry["pixels"])
        actual = int(entry["actual_bits"])
        rates[name] = {
            **entry,
            "actual_bpp": float(actual / pixels),
            "estimated_bpp": float(estimated / pixels),
            "actual_to_estimated_bpp_ratio": (
                float(actual / estimated) if estimated > 0 else None
            ),
        }
    return rates


def temporal_diagnostics(
    rates: dict[str, dict],
    evaluators: dict[str, DetectionMAP],
) -> dict[str, dict]:
    diagnostics = {}
    for name in TEMPORAL_BIN_ORDER:
        if name not in rates:
            continue
        evaluator = evaluators[name]
        record = dict(rates[name])
        if evaluator.image_ids and evaluator.ground_truth:
            record.update(evaluator.compute())
            record["status"] = "ok"
        else:
            record.update(
                {
                    "evaluated_images": len(evaluator.image_ids),
                    "map50": None,
                    "map5095": None,
                    "status": "no_ground_truth_objects",
                }
            )
        diagnostics[name] = record
    return diagnostics


def evaluate_codec(args: argparse.Namespace) -> None:
    if len(args.qps) < MIN_RATE_POINT_COUNT or len(set(args.qps)) != len(args.qps):
        raise ValueError("At least four distinct base QPs are required")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Actual DMC bitstream coding requires CUDA. Training with "
            "forward_train remains separate and differentiable."
        )

    device = torch.device(f"cuda:{args.cuda_index}")
    dataset = AnnotatedVideoDataset(args.data_dir, args.dataset_manifest)
    sequences = list(dataset)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError("No evaluation sequences were selected")
    short_sequences = [
        sequence.name
        for sequence in sequences
        if sequence.frame_count < args.minimum_sequence_frames
    ]
    if short_sequences:
        raise ValueError(
            f"{len(short_sequences)} selected sequences have fewer than "
            f"--minimum-sequence-frames={args.minimum_sequence_frames}; "
            f"first: {short_sequences[0]}"
        )

    image_model = DMCI().to(device).eval()
    load_codec_checkpoint(image_model, args.image_ckpt, state_key="dmci_state_dict")
    model = DMC().to(device).eval()
    checkpoint = load_codec_checkpoint(model, args.video_ckpt)
    try:
        image_model.update(force_zero_thres=args.force_zero_thres)
        model.update(force_zero_thres=args.force_zero_thres)
    except ImportError as error:
        raise RuntimeError(
            "Actual bitstream evaluation requires the MLCodec_extensions_cpp "
            "entropy-coder extension. Build src/cpp first."
        ) from error
    if args.codec_precision == "fp16":
        image_model.half()
        model.half()

    detector = load_yolov5(
        args.task_model,
        repository=args.yolov5_repo,
        weights=args.yolov5_weights,
    )
    detector_metadata = detector_config(
        args.task_model,
        args.detector_size,
        args.confidence_threshold,
        args.nms_iou_threshold,
        args.max_detections,
        args.yolov5_weights,
    )
    feature_objective = checkpoint.get("feature_objective", {})
    cloned_frontend_state = checkpoint.get("cloned_frontend_state_dict")
    if cloned_frontend_state is not None:
        trained_task_model = feature_objective.get("task_model")
        if trained_task_model is not None and trained_task_model != args.task_model:
            raise ValueError(
                "Checkpoint cloned front end was trained for "
                f"{trained_task_model}, not --task-model {args.task_model}"
            )
        trained_backend_weights = checkpoint.get("source_checkpoints", {}).get(
            "yolov5_weights_sha256"
        )
        if (
            trained_backend_weights is not None
            and trained_backend_weights != detector_metadata["weights_id"]
        ):
            raise ValueError(
                "Evaluation must use the same frozen YOLOv5 weights as training: "
                f"checkpoint={trained_backend_weights}, "
                f"evaluation={detector_metadata['weights_id']}"
            )
        last_frontend_layer = feature_objective.get(
            "cloned_frontend_last_layer"
        )
        if last_frontend_layer is None:
            last_frontend_layer = feature_objective.get("last_backbone_layer")
        if last_frontend_layer is None:
            last_frontend_layer = max(
                int(key.split(".", 1)[0]) for key in cloned_frontend_state
            )
        last_frontend_layer = int(last_frontend_layer)
        install_cloned_frontend(
            detector,
            cloned_frontend_state,
            last_frontend_layer,
        )
        machine_frontend = {
            "type": "checkpoint_cloned_yolov5_frontend",
            "weights_id": state_dict_sha256(cloned_frontend_state),
            "trainable_during_codec_training": bool(
                checkpoint.get("optimizer_config", {}).get(
                    "train_cloned_frontend",
                    False,
                )
            ),
            "last_frontend_layer": last_frontend_layer,
            "feature_layer_indices": list(
                feature_objective.get("layer_indices", ())
            ),
            "normalized_layer_weights": feature_objective.get(
                "normalized_layer_weights"
            ),
            "task_backend": "frozen_pretrained_yolov5",
            "task_backend_weights_id": detector_metadata["weights_id"],
            "feature_topology": feature_objective.get("topology"),
        }
        print(
            "Evaluation detector uses the trained cloned YOLO front end "
            f"(layers 0..{last_frontend_layer}) and frozen task back end."
        )
    else:
        machine_frontend = {
            "type": "pretrained_yolov5_frontend",
            "weights_id": detector_metadata["weights_id"],
            "trainable_during_codec_training": False,
            "task_backend": "frozen_pretrained_yolov5",
            "task_backend_weights_id": detector_metadata["weights_id"],
            "checkpoint_without_clone": True,
            "evaluation_role": "pretrained_frontend_anchor",
        }
        print(
            "Checkpoint has no cloned YOLO front end; evaluation falls back "
            "to the pretrained detector (anchor/frozen-feature protocol)."
        )
    detector = detector.to(device).eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    del checkpoint, cloned_frontend_state
    detector.conf = args.confidence_threshold
    detector.iou = args.nms_iou_threshold
    detector.max_det = args.max_detections

    method_name = safe_name(args.method_name)
    bitstream_root = Path(args.bitstream_dir) / method_name
    reconstruction_root = (
        Path(args.reconstruction_dir) / method_name
        if args.save_reconstructions
        else None
    )

    points = []
    for base_qp in args.qps:
        evaluator = DetectionMAP()
        temporal_evaluators = {
            name: DetectionMAP() for name in TEMPORAL_BIN_ORDER
        }
        sequence_records = []
        next_image_id = 0
        progress = tqdm(sequences, desc=f"{method_name}: base QP {base_qp}")
        for sequence in progress:
            bitstream_path = (
                bitstream_root
                / f"qp_{base_qp:02d}"
                / f"{safe_name(sequence.name)}.bin"
            )
            estimated_entropy_bits_by_frame = encode_sequence(
                image_model,
                model,
                dataset,
                sequence,
                base_qp,
                bitstream_path,
                device,
                args.reset_interval,
            )
            next_image_id = decode_and_evaluate_sequence(
                image_model,
                model,
                detector,
                evaluator,
                dataset,
                sequence,
                bitstream_path,
                device,
                args.detector_size,
                next_image_id,
                (
                    reconstruction_root / f"qp_{base_qp:02d}"
                    if reconstruction_root is not None
                    else None
                ),
                temporal_evaluators,
            )
            sequence_records.append(
                sequence_rate_record(
                    sequence,
                    bitstream_path,
                    estimated_entropy_bits_by_frame,
                )
            )

        temporal_rates = aggregate_temporal_rate(sequence_records)
        point = {
            "base_qp": base_qp,
            **aggregate_rate(sequence_records),
            **evaluator.compute(),
            "sequences": sequence_records,
            "temporal_diagnostics": temporal_diagnostics(
                temporal_rates,
                temporal_evaluators,
            ),
        }
        points.append(point)

    output = {
        "schema_version": 7,
        "method": args.method_name,
        "codec": "DCVC-RT DMCI + DMC all-frame VCM",
        "codec_config": {
            "image_checkpoint": str(Path(args.image_ckpt).resolve()),
            "checkpoint": str(Path(args.video_ckpt).resolve()),
            "base_qps": list(args.qps),
            "qp_offsets": list(QP_OFFSETS),
            "two_entropy_coders": (
                "enabled when source width*height exceeds 1280*720"
            ),
            "reset_interval": args.reset_interval,
            "codec_precision": args.codec_precision,
            "external_seed": False,
            "color_pipeline": "RGB -> full-range BT.709 YCbCr444 -> codec -> RGB",
            "temporal_bins": list(TEMPORAL_BIN_ORDER),
            "minimum_sequence_frames": args.minimum_sequence_frames,
        },
        "protocol": ALL_FRAMES_PROTOCOL,
        "comparison_scope": "end-to-end VCM system",
        "rate_source": "actual sequence-container bytes including headers",
        "rate_points": len(points),
        "task": "object_detection",
        "task_model": args.task_model,
        "ground_truth": "normalized YOLO labels from evaluation manifest",
        "evaluation_id": evaluation_id(dataset, sequences),
        "dataset": dataset_summary(dataset, sequences),
        "detector_config": detector_metadata,
        "machine_frontend": machine_frontend,
        "points": points,
    }
    output_path = Path(args.output_dir) / f"{method_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved actual-bitstream mAP results to {output_path}")


def load_results(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") not in (6, 7):
        raise ValueError(f"{path} does not use all-frame evaluation schema v6/v7")
    if len(data.get("points", [])) < MIN_RATE_POINT_COUNT:
        raise ValueError(
            f"{path} must contain at least {MIN_RATE_POINT_COUNT} rate points"
        )
    required_metadata = (
        "evaluation_id",
        "task_model",
        "protocol",
        "ground_truth",
        "detector_config",
        "machine_frontend",
        "comparison_scope",
        "rate_source",
    )
    missing = [key for key in required_metadata if data.get(key) is None]
    if missing:
        raise ValueError(f"{path} is missing evaluation metadata: {missing}")
    return data


def validate_compatible_results(anchor: dict, candidate: dict) -> None:
    for key in (
        "evaluation_id",
        "task_model",
        "protocol",
        "ground_truth",
        "detector_config",
        "comparison_scope",
    ):
        anchor_value = json.dumps(
            anchor.get(key),
            sort_keys=True,
            separators=(",", ":"),
        )
        candidate_value = json.dumps(
            candidate.get(key),
            sort_keys=True,
            separators=(",", ":"),
        )
        if anchor_value != candidate_value:
            raise ValueError(
                f"Anchor and candidate must use the same {key}: "
                f"{anchor.get(key)!r} != {candidate.get(key)!r}"
            )
    anchor_frames = {int(point["coded_frames"]) for point in anchor["points"]}
    candidate_frames = {
        int(point["coded_frames"])
        for point in candidate["points"]
    }
    if len(anchor_frames) != 1 or len(candidate_frames) != 1:
        raise ValueError("Evaluated frame count must not change across rate points")
    if anchor_frames != candidate_frames:
        raise ValueError(
            "Anchor and candidate must evaluate the same number of frames: "
            f"{anchor_frames} != {candidate_frames}"
        )


def curve_arrays(data: dict, rate_key: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    rates = np.asarray([point[rate_key] for point in data["points"]], dtype=np.float64)
    quality = np.asarray([point[metric] for point in data["points"]], dtype=np.float64)
    return pareto_front(rates, quality)


def save_curve_csv(
    anchor: dict,
    candidate: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "base_qp",
                "actual_bpp",
                "kbps",
                "map50",
                "map5095",
            ),
        )
        writer.writeheader()
        for data in (anchor, candidate):
            for point in data["points"]:
                writer.writerow(
                    {
                        key: value
                        for key, value in {
                            "method": data["method"],
                            "base_qp": point["base_qp"],
                            "actual_bpp": point["actual_bpp"],
                            "kbps": point["kbps"],
                            "map50": point["map50"],
                            "map5095": point["map5095"],
                        }.items()
                    }
                )


def plot_rd_curve(
    anchor: dict,
    candidate: dict,
    rate_key: str,
    metric: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("RD-curve plotting requires matplotlib") from error

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for data, marker in ((anchor, "o"), (candidate, "s")):
        points = sorted(data["points"], key=lambda point: point[rate_key])
        rates = [point[rate_key] for point in points]
        quality = [point[metric] for point in points]
        axis.plot(rates, quality, marker=marker, linewidth=2, label=data["method"])
        for point in points:
            axis.annotate(
                f"q={point['base_qp']}",
                (point[rate_key], point[metric]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_xlabel("Actual BPP" if rate_key == "actual_bpp" else "Actual bitrate (kbps)")
    axis.set_ylabel("mAP@0.5" if metric == "map50" else "mAP@[0.5:0.95]")
    axis.set_title("VCM Rate-Accuracy Curve")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def compare_bd_rate(args: argparse.Namespace) -> None:
    anchor = load_results(args.anchor_results)
    candidate = load_results(args.candidate_results)
    validate_compatible_results(anchor, candidate)
    anchor_rate, anchor_quality = curve_arrays(anchor, args.rate, args.metric)
    candidate_rate, candidate_quality = curve_arrays(candidate, args.rate, args.metric)

    result = {
        "anchor": anchor["method"],
        "candidate": candidate["method"],
        "comparison_scope": anchor["comparison_scope"],
        "anchor_machine_frontend": anchor["machine_frontend"],
        "candidate_machine_frontend": candidate["machine_frontend"],
        "rate": args.rate,
        "metric": args.metric,
        "bd_rate_percent": compute_bd_rate(
            anchor_rate,
            anchor_quality,
            candidate_rate,
            candidate_quality,
        ),
        "bd_metric": compute_bd_metric(
            anchor_rate,
            anchor_quality,
            candidate_rate,
            candidate_quality,
        ),
        "interpretation": "negative BD-rate means bitrate saving at equal mAP",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bd_rate_map.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    save_curve_csv(anchor, candidate, output_dir / "rd_points.csv")
    plot_rd_curve(
        anchor,
        candidate,
        args.rate,
        "map50",
        output_dir / f"rd_curve_{args.rate}_map50.png",
    )
    plot_rd_curve(
        anchor,
        candidate,
        args.rate,
        "map5095",
        output_dir / f"rd_curve_{args.rate}_map5095.png",
    )
    print(json.dumps(result, indent=2))


def summarize_training(args: argparse.Namespace) -> None:
    summary = []
    for path in sorted(Path(args.log_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            summary.append({"run": path.stem, "epochs": len(rows), "final": rows[-1]})
    output_path = Path(args.output_dir) / "training_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved training summary to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("codec", "bdrate", "training"))
    parser.add_argument("--data-dir", help="Root containing evaluation frames and labels")
    parser.add_argument("--dataset-manifest", help="Full-resolution evaluation manifest JSON")
    parser.add_argument("--image-ckpt", help="Frozen official DCVC-RT DMCI checkpoint")
    parser.add_argument("--video-ckpt", help="Official DMC or trained VCM checkpoint")
    parser.add_argument("--method-name", default="dcvc_rt_vcm")
    parser.add_argument(
        "--qps",
        type=int,
        nargs="+",
        default=(0, 21, 42, 63),
        choices=range(64),
        metavar="QP",
        help="At least four distinct DCVC-RT base QPs; all supplied points are evaluated",
    )
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--force-zero-thres", type=float)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument(
        "--codec-precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="FP16 matches the official DCVC-RT evaluation path",
    )
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument(
        "--minimum-sequence-frames",
        type=int,
        default=2,
        help="Use 100 for the required long-sequence drift evaluation",
    )
    parser.add_argument("--task-model", default="yolov5s")
    parser.add_argument("--yolov5-repo")
    parser.add_argument("--yolov5-weights")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--bitstream-dir", default="output/bitstreams")
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--reconstruction-dir", default="output/reconstructions")
    parser.add_argument("--anchor-results")
    parser.add_argument("--candidate-results")
    parser.add_argument("--rate", default="actual_bpp", choices=("actual_bpp", "kbps"))
    parser.add_argument("--metric", default="map5095", choices=("map50", "map5095"))
    parser.add_argument("--log-dir", default="checkpoints/vcm_video/logs")
    parser.add_argument("--output-dir", default="output/evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "codec":
        if arguments.reset_interval < 0:
            raise ValueError("--reset-interval must be non-negative")
        if arguments.minimum_sequence_frames < 2:
            raise ValueError("--minimum-sequence-frames must be at least 2")
        if (
            not arguments.data_dir
            or not arguments.dataset_manifest
            or not arguments.image_ckpt
            or not arguments.video_ckpt
        ):
            raise ValueError(
                "codec mode requires --data-dir, --dataset-manifest, "
                "--image-ckpt and --video-ckpt"
            )
        evaluate_codec(arguments)
    elif arguments.mode == "bdrate":
        if not arguments.anchor_results or not arguments.candidate_results:
            raise ValueError(
                "bdrate mode requires --anchor-results and --candidate-results"
            )
        compare_bd_rate(arguments)
    else:
        summarize_training(arguments)
