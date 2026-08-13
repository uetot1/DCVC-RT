from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.utils.dataset import VideoSequenceDataset


class ObjectAwareDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_dir = self.root / "train" / "MOT17-02-FRCNN" / "img1"
        self.gt_dir = self.image_dir.parent / "gt"
        self.image_dir.mkdir(parents=True)
        self.gt_dir.mkdir()
        for frame_number in range(1, 9):
            image = Image.new("RGB", (640, 480), color=(0, 0, 0))
            image.save(self.image_dir / f"{frame_number:06d}.jpg")
        # A pedestrian near the bottom-right makes a center crop unable to see it.
        self.gt_dir.joinpath("gt.txt").write_text(
            "\n".join(
                f"{frame},1,500,350,80,100,1,1,1"
                for frame in range(1, 9)
            ),
            encoding="utf-8",
        )
        self.list_file = self.root / "sequences.txt"
        self.list_file.write_text(
            "train/MOT17-02-FRCNN/img1\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_object_crop_returns_mot_targets_for_every_frame(self):
        random.seed(7)
        dataset = VideoSequenceDataset(
            self.root,
            self.list_file,
            crop_size=256,
            num_frames=8,
            training=True,
            crop_mode="object",
            aware_crop_probability=1.0,
            return_targets=True,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["frames"].shape), (8, 3, 256, 256))
        self.assertTrue(sample["has_annotations"])
        self.assertTrue(all(len(target["boxes"]) > 0 for target in sample["targets"]))
        self.assertTrue(
            all(int(target["classes"][0]) == 0 for target in sample["targets"])
        )

    def test_validation_crop_is_deterministic(self):
        dataset = VideoSequenceDataset(
            self.root,
            self.list_file,
            crop_size=256,
            num_frames=8,
            training=False,
            crop_mode="auto",
            return_targets=True,
        )
        first = dataset[0]
        second = dataset[0]
        self.assertTrue(first["frames"].equal(second["frames"]))
        self.assertTrue(
            all(
                first_target["boxes"].equal(second_target["boxes"])
                for first_target, second_target in zip(
                    first["targets"], second["targets"], strict=True
                )
            )
        )

    def test_validation_samples_cover_sequence_timeline(self):
        # Extend the sequence so deterministic validation clips can start at
        # different temporal positions.
        for frame_number in range(9, 17):
            image = Image.new(
                "RGB",
                (640, 480),
                color=(frame_number, frame_number, frame_number),
            )
            image.save(self.image_dir / f"{frame_number:06d}.jpg")
        dataset = VideoSequenceDataset(
            self.root,
            self.list_file,
            crop_size=256,
            num_frames=8,
            training=False,
            samples_per_sequence=3,
            crop_mode="random",
        )
        self.assertEqual(len(dataset), 3)
        self.assertFalse(dataset[0].equal(dataset[2]))


if __name__ == "__main__":
    unittest.main()
