"""CLI regression tests for dense rate-point evaluation."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import evaluate_hevc
import evaluate_vcm


class VariableRatePointTests(unittest.TestCase):
    def test_hevc_parser_accepts_more_than_four_qps(self):
        argv = [
            "evaluate_hevc.py",
            "--data-dir",
            "data",
            "--dataset-manifest",
            "manifest.json",
            "--qps",
            "29",
            "33",
            "37",
            "41",
            "45",
            "50",
        ]
        with patch.object(sys, "argv", argv):
            args = evaluate_hevc.parse_args()
        self.assertEqual(args.qps, [29, 33, 37, 41, 45, 50])

    def test_vcm_parser_accepts_more_than_four_qps(self):
        argv = [
            "evaluate_vcm.py",
            "--mode",
            "codec",
            "--qps",
            "0",
            "14",
            "28",
            "42",
            "56",
            "63",
        ]
        with patch.object(sys, "argv", argv):
            args = evaluate_vcm.parse_args()
        self.assertEqual(args.qps, [0, 14, 28, 42, 56, 63])


if __name__ == "__main__":
    unittest.main()
