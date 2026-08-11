"""Validate video-training sequences before launching a multi-GPU run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.dataset import VideoSequenceDataset


def main(args: argparse.Namespace) -> None:
    dataset = VideoSequenceDataset(
        args.root,
        list_file=args.list_file,
        crop_size=args.crop_size,
        num_frames=args.frames,
        training=args.training,
        samples_per_sequence=args.samples_per_sequence,
    )
    sequence_count = len(dataset.sequence_ids)
    checked = min(sequence_count, args.check_sequences or sequence_count)
    for sequence_id in dataset.sequence_ids[:checked]:
        sequence_dir = Path(args.root) / sequence_id
        frame_paths = dataset._frame_paths(sequence_dir)  # validation-only inspection
        if len(frame_paths) < args.frames:
            raise ValueError(
                f"{sequence_dir} has {len(frame_paths)} frames; {args.frames} required"
            )
        with dataset._load_rgb(frame_paths[0]) as image:
            width, height = image.size
        if width < args.crop_size or height < args.crop_size:
            raise ValueError(
                f"{sequence_dir} is {width}x{height}, smaller than crop {args.crop_size}"
            )
    sample = dataset[0]
    summary = {
        "status": "PASS",
        "sequences": sequence_count,
        "dataset_samples": len(dataset),
        "checked_sequences": checked,
        "frames_per_sample": int(sample.shape[0]),
        "channels": int(sample.shape[1]),
        "crop_height": int(sample.shape[2]),
        "crop_width": int(sample.shape[3]),
        "samples_per_sequence": args.samples_per_sequence,
        "training": args.training,
        "value_min": float(sample.min()),
        "value_max": float(sample.max()),
    }
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--list-file")
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--samples-per-sequence", type=int, default=1)
    parser.add_argument("--training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--check-sequences",
        type=int,
        help="Limit structural inspection; omit to check every listed sequence",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
