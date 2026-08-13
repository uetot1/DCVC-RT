"""Compare HEVC, Learned Scalable and the proposed codec using BD-rate-mAP.

Each input JSON must contain at least four rate points. It may be an evaluation
schema produced by ``evaluate_vcm.py`` or a compact external-result file:

{
  "method": "HEVC",
  "evaluation_id": "same_dataset_same_frames_v1",
  "task_model": "yolov5s",
  "protocol": "all frames and complete bitstream included",
  "comparison_scope": "end-to-end VCM system",
  "machine_frontend": {
    "type": "pretrained_or_trained_frontend",
    "weights_id": "sha256:..."
  },
  "points": [
    {
      "rate_label": "QP 37",
      "actual_bpp": 0.03,
      "kbps": 120.0,
      "map50": 0.42,
      "map5095": 0.23
    }
  ]
}

The four points must be Pareto-optimal: bitrate and mAP must increase together.
BD-rate is computed with HEVC and Learned Scalable as anchors. A negative value
means that the candidate uses fewer bits at equal mAP.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


MIN_RATE_POINT_COUNT = 4
METHOD_ORDER = ("hevc", "learned_scalable", "proposed")
METHOD_TITLES = {
    "hevc": "HEVC",
    "learned_scalable": "Learned Scalable (base layer)",
    "proposed": "DCVC-RT VCM (proposed)",
}
RATE_ALIASES = {
    "actual_bpp": ("actual_bpp", "bpp", "rate"),
    "kbps": ("kbps", "bitrate_kbps"),
}
METRIC_ALIASES = {
    "map50": ("map50", "mAP50", "map_50", "map@0.5"),
    "map5095": (
        "map5095",
        "mAP5095",
        "map_50_95",
        "map@[0.5:0.95]",
        "mAP",
    ),
}
FRAME_COUNT_ALIASES = ("evaluated_frames", "coded_frames")
SHARED_METADATA_KEYS = (
    "evaluation_id",
    "task_model",
    "protocol",
    "ground_truth",
    "detector_config",
    "comparison_scope",
)
REQUIRED_METADATA_KEYS = (
    "method",
    "codec",
    "rate_source",
    *SHARED_METADATA_KEYS,
    "machine_frontend",
)


def _read_number(
    point: dict[str, Any],
    aliases: tuple[str, ...],
    path: Path,
    point_index: int,
) -> float:
    for key in aliases:
        if key in point:
            value = float(point[key])
            if not np.isfinite(value):
                raise ValueError(
                    f"{path}: point {point_index} field '{key}' must be finite"
                )
            return value
    raise ValueError(
        f"{path}: point {point_index} is missing one of {list(aliases)}"
    )


def _point_label(point: dict[str, Any], point_index: int) -> str:
    if "rate_label" in point:
        return str(point["rate_label"])
    if "base_qp" in point:
        return f"QP {point['base_qp']}"
    if "qp" in point:
        return f"QP {point['qp']}"
    return f"R{point_index + 1}"


def _read_frame_count(
    point: dict[str, Any],
    path: Path,
    point_index: int,
) -> int:
    value = _read_number(
        point,
        FRAME_COUNT_ALIASES,
        path,
        point_index,
    )
    if value <= 0 or int(value) != value:
        raise ValueError(
            f"{path}: point {point_index} evaluated frame count "
            "must be a positive integer"
        )
    return int(value)


def load_method_results(
    path: str | Path,
    fallback_method: str,
    rate_key: str,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    result_path = Path(path)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    missing_metadata = [
        key
        for key in REQUIRED_METADATA_KEYS
        if data.get(key) is None
    ]
    if missing_metadata:
        raise ValueError(
            f"{result_path} is missing required evaluation metadata: "
            f"{missing_metadata}"
        )
    points = data.get("points")
    if not isinstance(points, list) or len(points) < MIN_RATE_POINT_COUNT:
        raise ValueError(
            f"{result_path} must contain at least {MIN_RATE_POINT_COUNT} entries in 'points'"
        )

    normalized_points = []
    for point_index, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"{result_path}: point {point_index} must be an object")
        normalized = {
            "rate_label": _point_label(point, point_index),
            rate_key: _read_number(
                point,
                RATE_ALIASES[rate_key],
                result_path,
                point_index,
            ),
            "evaluated_frames": _read_frame_count(
                point,
                result_path,
                point_index,
            ),
        }
        if normalized[rate_key] <= 0:
            raise ValueError(
                f"{result_path}: point {point_index} rate must be positive"
            )
        for metric in metrics:
            normalized[metric] = _read_number(
                point,
                METRIC_ALIASES[metric],
                result_path,
                point_index,
            )
            if not 0.0 <= normalized[metric] <= 1.0:
                raise ValueError(
                    f"{result_path}: point {point_index} {metric} must "
                    "use the fractional [0, 1] scale"
                )
        normalized_points.append(normalized)

    return {
        "method": str(data.get("method") or fallback_method),
        "codec": data.get("codec"),
        "codec_config": data.get("codec_config"),
        "rate_source": data.get("rate_source"),
        "source_file": str(result_path.resolve()),
        "evaluation_id": data.get("evaluation_id"),
        "task_model": data.get("task_model"),
        "protocol": data.get("protocol"),
        "ground_truth": data.get("ground_truth"),
        "detector_config": data.get("detector_config"),
        "comparison_scope": data.get("comparison_scope"),
        "machine_frontend": data.get("machine_frontend"),
        "points": normalized_points,
    }


def validate_shared_protocol(results: list[dict[str, Any]]) -> None:
    """Reject dataset, detector, ground-truth and framing mismatches."""
    for metadata_key in SHARED_METADATA_KEYS:
        values = [result.get(metadata_key) for result in results]
        serialized = [
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in values
        ]
        if len(set(serialized)) != 1:
            mapping = {
                result["method"]: result.get(metadata_key) for result in results
            }
            raise ValueError(
                f"All methods must use the same {metadata_key}: {mapping}"
            )

    frame_counts = {}
    for result in results:
        counts = {
            point["evaluated_frames"]
            for point in result["points"]
        }
        if len(counts) != 1:
            raise ValueError(
                f"{result['method']} changes evaluated frame count across "
                f"rate points: {sorted(counts)}"
            )
        frame_counts[result["method"]] = counts.pop()
    if len(set(frame_counts.values())) != 1:
        raise ValueError(
            f"All methods must evaluate the same frame count: {frame_counts}"
        )


def validate_metric_scale(results: list[dict[str, Any]], metric: str) -> None:
    for result in results:
        if any(not 0.0 <= point[metric] <= 1.0 for point in result["points"]):
            raise ValueError(
                f"{result['method']} {metric} must use fractional [0, 1] values"
            )


def curve_arrays(
    result: dict[str, Any],
    rate_key: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    from src.utils.bd_rate import pareto_front

    rates = np.asarray(
        [point[rate_key] for point in result["points"]],
        dtype=np.float64,
    )
    quality = np.asarray(
        [point[metric] for point in result["points"]],
        dtype=np.float64,
    )
    return pareto_front(rates, quality)


def compare_pair(
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    rate_key: str,
    metric: str,
) -> dict[str, Any]:
    from src.utils.bd_rate import compute_bd_metric, compute_bd_rate

    anchor_rate, anchor_quality = curve_arrays(anchor, rate_key, metric)
    candidate_rate, candidate_quality = curve_arrays(candidate, rate_key, metric)
    quality_min = max(float(anchor_quality.min()), float(candidate_quality.min()))
    quality_max = min(float(anchor_quality.max()), float(candidate_quality.max()))
    return {
        "anchor": anchor["method"],
        "candidate": candidate["method"],
        "rate": rate_key,
        "metric": metric,
        "overlap_quality_min": quality_min,
        "overlap_quality_max": quality_max,
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


def save_rd_points(
    results: list[dict[str, Any]],
    rate_key: str,
    metrics: tuple[str, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ("method", "rate_label", rate_key, *metrics)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for point in result["points"]:
                writer.writerow(
                    {
                        "method": result["method"],
                        "rate_label": point["rate_label"],
                        rate_key: point[rate_key],
                        **{metric: point[metric] for metric in metrics},
                    }
                )


def save_comparison_csv(comparisons: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "metric",
            "rate",
            "anchor",
            "candidate",
            "overlap_quality_min",
            "overlap_quality_max",
            "bd_rate_percent",
            "bd_metric",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(
                {key: comparison[key] for key in fieldnames}
            )


def plot_rd_curve(
    results: list[dict[str, Any]],
    rate_key: str,
    metric: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("RD-curve plotting requires matplotlib") from error

    markers = ("o", "s", "^")
    figure, axis = plt.subplots(figsize=(8.0, 5.8))
    for result, marker in zip(results, markers, strict=True):
        points = sorted(result["points"], key=lambda point: point[rate_key])
        rates = [point[rate_key] for point in points]
        quality = [point[metric] for point in points]
        axis.plot(
            rates,
            quality,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=result["method"],
        )
        for point in points:
            axis.annotate(
                point["rate_label"],
                (point[rate_key], point[metric]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_xlabel(
        "Actual BPP" if rate_key == "actual_bpp" else "Actual bitrate (kbps)"
    )
    axis.set_ylabel(
        "mAP@0.5" if metric == "map50" else "mAP@[0.5:0.95]"
    )
    axis.set_title("HEVC vs Learned Scalable vs Proposed: Rate-Accuracy")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_templates(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for method_key in METHOD_ORDER:
        points = []
        for index in range(MIN_RATE_POINT_COUNT):
            points.append(
                {
                    "rate_label": f"R{index + 1}",
                    "actual_bpp": None,
                    "kbps": None,
                    "map50": None,
                    "map5095": None,
                    "evaluated_frames": None,
                }
            )
        template = {
            "schema_version": 1,
            "method": METHOD_TITLES[method_key],
            "codec": "replace_with_codec_and_version",
            "codec_config": {},
            "evaluation_id": "replace_with_shared_dataset_and_frame_set_id",
            "task_model": "yolov5s",
            "ground_truth": "normalized YOLO labels from evaluation manifest",
            "detector_config": {
                "model": "yolov5s",
                "weights_id": (
                    "torch-hub:yolov5s:ultralytics/yolov5:v7.0"
                ),
                "weights_scope": "pretrained initialization and frozen task backend",
                "input_size": 640,
                "confidence_threshold": 0.001,
                "nms_iou_threshold": 0.6,
                "max_detections": 300,
                "feature_repository": "ultralytics/yolov5:v7.0",
            },
            "protocol": "all frames and complete bitstream included",
            "comparison_scope": "end-to-end VCM system",
            "rate_source": "replace_with_actual_bitstream_measurement",
            "machine_frontend": {
                "type": "replace_with_pretrained_or_trained_frontend",
                "weights_id": "replace_with_sha256_or_checkpoint_id",
                "task_backend": "replace_with_frozen_task_backend",
            },
            "points": points,
        }
        path = output_dir / f"{method_key}_results_template.json"
        path.write_text(json.dumps(template, indent=2), encoding="utf-8")
        print(f"Created {path}")


def run_comparison(args: argparse.Namespace) -> None:
    metrics = tuple(args.metrics)
    results_by_key = {
        "hevc": load_method_results(
            args.hevc_results,
            METHOD_TITLES["hevc"],
            args.rate,
            metrics,
        ),
        "learned_scalable": load_method_results(
            args.learned_scalable_results,
            METHOD_TITLES["learned_scalable"],
            args.rate,
            metrics,
        ),
        "proposed": load_method_results(
            args.proposed_results,
            METHOD_TITLES["proposed"],
            args.rate,
            metrics,
        ),
    }
    results = [results_by_key[key] for key in METHOD_ORDER]
    validate_shared_protocol(results)

    comparisons = []
    pair_keys = (
        ("hevc", "learned_scalable"),
        ("hevc", "proposed"),
        ("learned_scalable", "proposed"),
    )
    for metric in metrics:
        validate_metric_scale(results, metric)
        for anchor_key, candidate_key in pair_keys:
            comparisons.append(
                compare_pair(
                    results_by_key[anchor_key],
                    results_by_key[candidate_key],
                    args.rate,
                    metric,
                )
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "rate": args.rate,
        "metrics": list(metrics),
        "methods": [
            {
                key: result.get(key)
                for key in (
                    "method",
                    "codec",
                    "codec_config",
                    "rate_source",
                    "source_file",
                    "evaluation_id",
                    "task_model",
                    "protocol",
                    "ground_truth",
                    "detector_config",
                    "comparison_scope",
                    "machine_frontend",
                )
            }
            for result in results
        ],
        "comparisons": comparisons,
    }
    json_path = output_dir / "codec_bd_rate_comparison.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    save_rd_points(
        results,
        args.rate,
        metrics,
        output_dir / "codec_rd_points.csv",
    )
    save_comparison_csv(
        comparisons,
        output_dir / "codec_bd_rate_comparison.csv",
    )
    for metric in metrics:
        plot_rd_curve(
            results,
            args.rate,
            metric,
            output_dir / f"rd_curve_{args.rate}_{metric}.png",
        )

    print(json.dumps(comparisons, indent=2))
    print(f"Saved comparison to {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hevc-results", help="HEVC four-point result JSON")
    parser.add_argument(
        "--learned-scalable-results",
        help="Learned Scalable base-layer four-point result JSON",
    )
    parser.add_argument(
        "--proposed-results",
        help="Proposed DCVC-RT VCM four-point result JSON",
    )
    parser.add_argument(
        "--rate",
        choices=("actual_bpp", "kbps"),
        default="kbps",
    )
    parser.add_argument(
        "--metrics",
        choices=("map50", "map5095"),
        nargs="+",
        default=("map50", "map5095"),
    )
    parser.add_argument("--output-dir", default="output/codec_comparison")
    parser.add_argument(
        "--write-templates",
        action="store_true",
        help="Create three empty four-point input templates and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_templates:
        write_templates(Path(args.output_dir))
        return
    missing = [
        option
        for option, value in (
            ("--hevc-results", args.hevc_results),
            ("--learned-scalable-results", args.learned_scalable_results),
            ("--proposed-results", args.proposed_results),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Comparison requires {', '.join(missing)} "
            "(or use --write-templates)"
        )
    run_comparison(args)


if __name__ == "__main__":
    main()
