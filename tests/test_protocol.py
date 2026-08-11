from __future__ import annotations

import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:  # Local packaging environments may omit CUDA/PyTorch.
    raise unittest.SkipTest("PyTorch is required for model protocol tests") from error

from evaluate_hevc import raw_frame_bytes
from evaluate_vcm import (
    container_bits_by_frame,
    temporal_bin_name,
    use_two_entropy_coders,
)
from src.models.entropy_models import BitEstimator, GaussianEncoder
from src.models.vcm_loss import multi_level_feature_mse
from src.models.vcm_system import QP_OFFSETS
from src.models.yolov5_extractor import (
    BACKBONE_ABLATION_FEATURE_LAYER_INDICES,
    DEFAULT_CLONED_FRONTEND_LAST_LAYER,
    DEFAULT_FEATURE_LAYER_INDICES,
    _run_yolov5_graph,
    ddp_find_unused_parameters,
)
from src.utils.evaluation_protocol import state_dict_sha256
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb
from src.utils.vcm_bitstream import VCMSequenceReader, VCMSequenceWriter
from train_vcm_final import get_epoch_num_frames, interpolate_lambda


class ColourTransformTests(unittest.TestCase):
    def test_bt709_round_trip(self):
        torch.manual_seed(7)
        rgb = torch.rand(2, 3, 16, 16)
        reconstructed = ycbcr2rgb(rgb2ycbcr(rgb))
        self.assertTrue(torch.allclose(rgb, reconstructed, atol=2e-6, rtol=0.0))

    def test_three_channels_required(self):
        with self.assertRaises(ValueError):
            rgb2ycbcr(torch.rand(1, 1, 4, 4))


class TrainingProtocolTests(unittest.TestCase):
    def test_dcvc_rt_qp_offsets(self):
        self.assertEqual(QP_OFFSETS, (0, 8, 0, 4, 0, 4, 0, 4))
        self.assertEqual(63 + max(QP_OFFSETS), 71)

    def test_lambda_endpoints(self):
        self.assertAlmostEqual(interpolate_lambda(0, 1.0, 768.0, 0.006), 0.006)
        self.assertAlmostEqual(interpolate_lambda(63, 1.0, 768.0, 0.006), 4.608)
        middle = interpolate_lambda(21, 1.0, 768.0, 0.006)
        self.assertGreater(middle, 0.006)
        self.assertLess(middle, 4.608)

    def test_fifteen_epoch_curriculum(self):
        args = Namespace(
            training_stage="vimeo7",
            vimeo_curriculum_frames=(3, 5, 7),
            vimeo_curriculum_start_epochs=(1, 3, 6),
        )
        self.assertEqual(get_epoch_num_frames(args, 1), 3)
        self.assertEqual(get_epoch_num_frames(args, 3), 5)
        self.assertEqual(get_epoch_num_frames(args, 6), 7)
        self.assertEqual(get_epoch_num_frames(args, 15), 7)

    def test_gaussian_rate_uses_coder_scale_bounds(self):
        encoder = GaussianEncoder()
        symbols = torch.zeros(1, 1, 1, 1)
        low = encoder.forward_rate(symbols, torch.full_like(symbols, 1e-9))
        scale_min = encoder.forward_rate(
            symbols,
            torch.full_like(symbols, encoder.scale_min),
        )
        high = encoder.forward_rate(symbols, torch.full_like(symbols, 100.0))
        scale_max = encoder.forward_rate(
            symbols,
            torch.full_like(symbols, encoder.scale_max),
        )
        self.assertTrue(torch.equal(low, scale_min))
        self.assertTrue(torch.equal(high, scale_max))
        half_rate = encoder.forward_rate(symbols.half(), torch.ones_like(symbols).half())
        self.assertEqual(half_rate.dtype, torch.float32)
        self.assertTrue(torch.isfinite(half_rate).all())

    def test_bit_estimator_rate_stays_fp32_for_half_input(self):
        estimator = BitEstimator(qp_num=72, channel=2).half()
        rate = estimator.forward_rate(torch.zeros(1, 2, 1, 1).half(), qp=21)
        self.assertEqual(rate.dtype, torch.float32)
        self.assertTrue(torch.isfinite(rate).all())

    def test_two_entropy_coder_threshold(self):
        self.assertFalse(use_two_entropy_coders(1280, 720))
        self.assertTrue(use_two_entropy_coders(1920, 1080))

    def test_tbptt_enables_ddp_unused_parameter_detection(self):
        self.assertFalse(ddp_find_unused_parameters(0))
        self.assertTrue(ddp_find_unused_parameters(2))
        with self.assertRaises(ValueError):
            ddp_find_unused_parameters(-1)

    def test_main_and_ablation_feature_topologies(self):
        self.assertEqual(DEFAULT_FEATURE_LAYER_INDICES, (17, 20, 23))
        self.assertEqual(BACKBONE_ABLATION_FEATURE_LAYER_INDICES, (4, 6, 9))
        self.assertEqual(DEFAULT_CLONED_FRONTEND_LAST_LAYER, 4)

    def test_long_sequence_temporal_bins(self):
        expected = {
            0: "frame_0",
            1: "frames_1_7",
            7: "frames_1_7",
            8: "frames_8_31",
            31: "frames_8_31",
            32: "frames_32_63",
            63: "frames_32_63",
            64: "frames_64_plus",
            100: "frames_64_plus",
        }
        for frame_index, name in expected.items():
            self.assertEqual(temporal_bin_name(frame_index), name)


class _GraphLayer(torch.nn.Module):
    def __init__(self, index, source, operation):
        super().__init__()
        self.i = index
        self.f = source
        self.operation = operation

    def forward(self, value):
        return self.operation(value)


class YOLOGraphTests(unittest.TestCase):
    def test_skip_routing_matches_yolov5_graph_semantics(self):
        layers = torch.nn.ModuleList(
            [
                _GraphLayer(0, -1, lambda value: value * 2),
                _GraphLayer(1, -1, lambda value: value + 1),
                _GraphLayer(2, [-1, 0], lambda values: torch.cat(values, dim=1)),
            ]
        )
        selected = _run_yolov5_graph(
            layers,
            torch.ones(1, 1, 2, 2),
            frozenset({0, 2}),
            frozenset({2}),
        )
        expected = torch.cat(
            (torch.full((1, 1, 2, 2), 3.0), torch.full((1, 1, 2, 2), 2.0)),
            dim=1,
        )
        self.assertTrue(torch.equal(selected[2], expected))


class FeatureLossTests(unittest.TestCase):
    def test_equal_weight_multi_level_mse(self):
        teacher = tuple(torch.zeros(1, 1, size, size) for size in (8, 4, 2))
        student = (
            torch.ones_like(teacher[0]),
            torch.full_like(teacher[1], 2.0),
            torch.full_like(teacher[2], 3.0),
        )
        weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        total, levels = multi_level_feature_mse(student, teacher, weights)
        self.assertTrue(torch.allclose(levels, torch.tensor([1.0, 4.0, 9.0])))
        self.assertAlmostEqual(float(total), 14.0 / 3.0, places=6)

    def test_frontend_state_fingerprint_changes_with_weights(self):
        first = {"0.weight": torch.zeros(2, 2)}
        second = {"0.weight": torch.ones(2, 2)}
        self.assertEqual(state_dict_sha256(first), state_dict_sha256(first))
        self.assertNotEqual(state_dict_sha256(first), state_dict_sha256(second))


class BitstreamAndAnchorTests(unittest.TestCase):
    def test_all_frame_container(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.vcm"
            with VCMSequenceWriter(
                path,
                width=64,
                height=32,
                fps=30.0,
                coded_frames=2,
                external_seed=False,
                two_entropy_coders=True,
                reset_interval=32,
            ) as writer:
                writer.write_frame(21, b"intra")
                writer.write_frame(29, b"predicted")
            with VCMSequenceReader(path) as reader:
                self.assertFalse(reader.header.external_seed)
                self.assertTrue(reader.header.two_entropy_coders)
                self.assertEqual(reader.header.reset_interval, 32)
                packets = list(reader.frames())
            self.assertEqual([packet.qp for packet in packets], [21, 29])
            self.assertEqual([packet.bitstream for packet in packets], [b"intra", b"predicted"])
            frame_bits = container_bits_by_frame(path)
            self.assertEqual(sum(frame_bits), path.stat().st_size * 8)
            self.assertEqual(len(frame_bits), 2)

    def test_raw_frame_sizes(self):
        self.assertEqual(raw_frame_bytes(16, 16, 8, "420"), 384)
        self.assertEqual(raw_frame_bytes(16, 16, 10, "444"), 1536)


if __name__ == "__main__":
    unittest.main()
