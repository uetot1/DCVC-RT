from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.vcm_bitstream import VCMSequenceReader, VCMSequenceWriter


class AllFrameContainerTests(unittest.TestCase):
    def test_first_packet_is_coded_intra(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.vcm"
            with VCMSequenceWriter(
                path,
                width=64,
                height=32,
                fps=30.0,
                coded_frames=2,
                reset_interval=32,
            ) as writer:
                writer.write_frame(21, b"intra")
                writer.write_frame(29, b"predicted")
            with VCMSequenceReader(path) as reader:
                self.assertFalse(reader.header.external_seed)
                self.assertEqual(reader.header.reset_interval, 32)
                packets = list(reader.frames())
            self.assertEqual([packet.qp for packet in packets], [21, 29])
            self.assertEqual([packet.bitstream for packet in packets], [b"intra", b"predicted"])


if __name__ == "__main__":
    unittest.main()
