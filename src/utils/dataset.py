"""Dataset loader for the two video-training stages used by DCVC-RT.

It supports both standard Vimeo-90K septuplets (seven frames per directory)
and REDS sharp sequences containing 100 consecutive frames per directory.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageChops, ImageStat
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _natural_key(path: Path) -> list[tuple[int, int | str]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    ]


class VideoSequenceDataset(Dataset):
    """Return a consistently transformed contiguous clip from each directory.

    ``root_dir`` contains sequence directories. A list entry is a sequence
    path relative to ``root_dir``. If the list is omitted or cannot be found,
    sequence directories are discovered recursively.
    """

    def __init__(
        self,
        root_dir: str | Path,
        list_file: str | Path | None = None,
        crop_size: int = 256,
        num_frames: int = 8,
        training: bool = True,
        samples_per_sequence: int = 1,
        crop_mode: str = "random",
        aware_crop_probability: float = 0.8,
        return_targets: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.crop_size = int(crop_size)
        self.num_frames = int(num_frames)
        self.max_num_frames = self.num_frames
        self.training = bool(training)
        self.samples_per_sequence = int(samples_per_sequence)
        self.crop_mode = str(crop_mode)
        self.aware_crop_probability = float(aware_crop_probability)
        self.return_targets = bool(return_targets)
        self._mot_gt_cache: dict[Path, dict[int, tuple[tuple[float, float, float, float], ...]]] = {}

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Video sequences directory not found: {self.root_dir}")
        if self.num_frames < 2:
            raise ValueError("num_frames must contain one reference seed and at least one P-frame")
        if self.samples_per_sequence < 1:
            raise ValueError("samples_per_sequence must be at least 1")
        if self.crop_mode not in {"random", "auto", "object", "motion"}:
            raise ValueError("crop_mode must be random, auto, object, or motion")
        if not 0.0 <= self.aware_crop_probability <= 1.0:
            raise ValueError("aware_crop_probability must be in [0, 1]")

        list_path = self._resolve_list_path(list_file)
        if list_path is not None:
            self.sequence_ids = [
                line.strip()
                for line in list_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            self.sequence_ids = self._discover_sequences()

        if not self.sequence_ids:
            raise RuntimeError(f"No frame sequences found under {self.root_dir}")

        first_sequence = self.root_dir / self.sequence_ids[0]
        first_count = len(self._frame_paths(first_sequence))
        if first_count < self.num_frames:
            raise ValueError(
                f"{first_sequence} contains {first_count} frames, but the clip requires "
                f"{self.num_frames}."
            )

    def _resolve_list_path(self, list_file: str | Path | None) -> Path | None:
        if list_file is None:
            return None
        candidate = Path(list_file)
        candidates = (candidate, self.root_dir / candidate, self.root_dir.parent / candidate)
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"Sequence list not found: {list_file}")

    def _discover_sequences(self) -> list[str]:
        directories = {
            frame.parent
            for frame in self.root_dir.rglob("*")
            if frame.is_file() and frame.suffix.lower() in IMAGE_EXTENSIONS
        }
        return sorted(str(path.relative_to(self.root_dir)) for path in directories)

    @staticmethod
    def _frame_paths(sequence_dir: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in sequence_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=_natural_key,
        )

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _mot_gt_path(sequence_dir: Path) -> Path:
        return sequence_dir.parent / "gt" / "gt.txt"

    def _mot_ground_truth(
        self,
        sequence_dir: Path,
    ) -> dict[int, tuple[tuple[float, float, float, float], ...]]:
        """Read visible pedestrian boxes from a MOTChallenge ``gt.txt`` file."""
        gt_path = self._mot_gt_path(sequence_dir)
        if gt_path in self._mot_gt_cache:
            return self._mot_gt_cache[gt_path]
        boxes: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        if gt_path.is_file():
            for line in gt_path.read_text(encoding="utf-8").splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) < 6:
                    continue
                frame_number = int(float(fields[0]))
                left, top, width, height = map(float, fields[2:6])
                confidence = float(fields[6]) if len(fields) > 6 else 1.0
                class_id = int(float(fields[7])) if len(fields) > 7 else 1
                visibility = float(fields[8]) if len(fields) > 8 else 1.0
                if confidence <= 0 or class_id != 1 or visibility <= 0:
                    continue
                if width > 1 and height > 1:
                    boxes[frame_number].append(
                        (left, top, left + width, top + height)
                    )
        result = {key: tuple(value) for key, value in boxes.items()}
        self._mot_gt_cache[gt_path] = result
        return result

    @staticmethod
    def _frame_number(path: Path) -> int | None:
        match = re.search(r"(\d+)$", path.stem)
        return int(match.group(1)) if match else None

    def _clip_boxes(
        self,
        sequence_dir: Path,
        frame_paths: list[Path],
        left: int,
        top: int,
        crop_width: int,
        crop_height: int,
        flipped: bool,
    ) -> list[dict[str, torch.Tensor]]:
        ground_truth = self._mot_ground_truth(sequence_dir)
        targets = []
        for frame_path in frame_paths:
            frame_number = self._frame_number(frame_path)
            transformed = []
            for x1, y1, x2, y2 in ground_truth.get(frame_number, ()):
                x1 = max(0.0, min(float(crop_width), x1 - left))
                x2 = max(0.0, min(float(crop_width), x2 - left))
                y1 = max(0.0, min(float(crop_height), y1 - top))
                y2 = max(0.0, min(float(crop_height), y2 - top))
                if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                    continue
                if flipped:
                    x1, x2 = crop_width - x2, crop_width - x1
                transformed.append((x1, y1, x2, y2))
            boxes = torch.tensor(transformed, dtype=torch.float32).reshape(-1, 4)
            targets.append(
                {
                    "boxes": boxes,
                    # MOT17 pedestrian maps to COCO/YOLO class 0 (person).
                    "classes": torch.zeros(len(boxes), dtype=torch.long),
                }
            )
        return targets

    def _object_crop(
        self,
        sequence_dir: Path,
        frame_paths: list[Path],
        width: int,
        height: int,
    ) -> tuple[int, int] | None:
        ground_truth = self._mot_ground_truth(sequence_dir)
        candidates = []
        for path in frame_paths:
            frame_number = self._frame_number(path)
            candidates.extend(ground_truth.get(frame_number, ()))
        if not candidates:
            return None
        x1, y1, x2, y2 = random.choice(candidates)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        jitter_x = random.uniform(-0.2, 0.2) * self.crop_size
        jitter_y = random.uniform(-0.2, 0.2) * self.crop_size
        left = round(center_x + jitter_x - self.crop_size * 0.5)
        top = round(center_y + jitter_y - self.crop_size * 0.5)
        return (
            max(0, min(width - self.crop_size, left)),
            max(0, min(height - self.crop_size, top)),
        )

    def _validation_object_crop(
        self,
        sequence_dir: Path,
        frame_paths: list[Path],
        width: int,
        height: int,
    ) -> tuple[int, int] | None:
        """Deterministically center validation on the largest visible object."""
        ground_truth = self._mot_ground_truth(sequence_dir)
        candidates = []
        for path in frame_paths:
            frame_number = self._frame_number(path)
            candidates.extend(ground_truth.get(frame_number, ()))
        if not candidates:
            return None
        x1, y1, x2, y2 = max(
            candidates,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        )
        left = round((x1 + x2 - self.crop_size) * 0.5)
        top = round((y1 + y2 - self.crop_size) * 0.5)
        return (
            max(0, min(width - self.crop_size, left)),
            max(0, min(height - self.crop_size, top)),
        )

    def _motion_crop(
        self,
        images: list[Image.Image],
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """Choose the most temporally active of several random crop candidates."""
        difference = ImageChops.difference(images[0], images[-1]).convert("L")
        candidates = [
            (
                random.randint(0, width - self.crop_size),
                random.randint(0, height - self.crop_size),
            )
            for _ in range(8)
        ]
        return max(
            candidates,
            key=lambda position: ImageStat.Stat(
                difference.crop(
                    (
                        position[0],
                        position[1],
                        position[0] + self.crop_size,
                        position[1] + self.crop_size,
                    )
                ).resize((32, 32))
            ).mean[0],
        )

    @property
    def has_object_annotations(self) -> bool:
        return any(
            self._mot_gt_path(self.root_dir / sequence_id).is_file()
            for sequence_id in self.sequence_ids
        )

    @property
    def has_complete_object_annotations(self) -> bool:
        return all(
            self._mot_gt_path(self.root_dir / sequence_id).is_file()
            for sequence_id in self.sequence_ids
        )

    def __len__(self) -> int:
        return len(self.sequence_ids) * self.samples_per_sequence

    def set_num_frames(self, num_frames: int):
        """Change the temporal crop length without rebuilding the DataLoader."""
        num_frames = int(num_frames)
        if not 2 <= num_frames <= self.max_num_frames:
            raise ValueError(
                f"num_frames must be in [2, {self.max_num_frames}], got {num_frames}"
            )
        self.num_frames = num_frames

    def __getitem__(self, index: int) -> torch.Tensor | dict[str, object]:
        sequence_index = index % len(self.sequence_ids)
        sample_index = index // len(self.sequence_ids)
        sequence_dir = self.root_dir / self.sequence_ids[sequence_index]
        frame_paths = self._frame_paths(sequence_dir)
        if len(frame_paths) < self.num_frames:
            raise ValueError(
                f"{sequence_dir} contains {len(frame_paths)} frames; "
                f"{self.num_frames} contiguous frames are required"
            )
        max_start = len(frame_paths) - self.num_frames
        if self.training:
            first_frame = random.randint(0, max_start)
        elif self.samples_per_sequence == 1:
            first_frame = 0
        else:
            first_frame = round(
                sample_index * max_start / (self.samples_per_sequence - 1)
            )
        images = [
            self._load_rgb(path)
            for path in frame_paths[first_frame:first_frame + self.num_frames]
        ]

        width, height = images[0].size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(
                f"Crop size {self.crop_size} exceeds frame size {width}x{height} "
                f"in {sequence_dir}"
            )

        selected_paths = frame_paths[first_frame:first_frame + self.num_frames]
        flipped = False
        if self.training:
            crop_height = crop_width = self.crop_size
            aware = random.random() < self.aware_crop_probability
            position = None
            if aware and self.crop_mode in {"auto", "object"}:
                position = self._object_crop(
                    sequence_dir, selected_paths, width, height
                )
            if aware and position is None and self.crop_mode in {"auto", "motion"}:
                position = self._motion_crop(images, width, height)
            if position is None:
                top, left, crop_height, crop_width = RandomCrop.get_params(
                    images[0], output_size=(self.crop_size, self.crop_size)
                )
            else:
                left, top = position
            flipped = random.random() < 0.5
            if flipped:
                images = [transforms.hflip(image) for image in images]
        else:
            crop_height = crop_width = self.crop_size
            position = (
                self._validation_object_crop(
                    sequence_dir, selected_paths, width, height
                )
                if self.crop_mode in {"auto", "object"}
                else None
            )
            if position is None:
                top = (height - crop_height) // 2
                left = (width - crop_width) // 2
            else:
                left, top = position

        frames = torch.stack(
            [
                transforms.to_tensor(
                    transforms.crop(image, top, left, crop_height, crop_width)
                )
                for image in images
            ]
        )
        if not self.return_targets:
            return frames
        return {
            "frames": frames,
            "targets": self._clip_boxes(
                sequence_dir,
                selected_paths,
                left,
                top,
                crop_width,
                crop_height,
                flipped,
            ),
            "has_annotations": self._mot_gt_path(sequence_dir).is_file(),
        }


# Kept as a compatibility import for evaluation scripts and older checkpoints.
VimeoSeptupletDataset = VideoSequenceDataset
