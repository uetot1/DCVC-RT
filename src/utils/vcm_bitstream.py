"""Versioned sequence container for actual DMCI + DMC frame bitstreams.

When ``external_seed`` is false, packet 0 is DMCI and all later packets are DMC.
The deterministic ordering avoids adding a frame-type byte to every packet.
VCM2 records the DCVC-RT feature-reset interval; the reader remains compatible
with legacy VCM1 files, which imply no periodic reset.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


LEGACY_MAGIC = b"VCM1"
MAGIC = b"VCM2"
LEGACY_SEQUENCE_HEADER = struct.Struct(">4sHHfIB")
SEQUENCE_HEADER = struct.Struct(">4sHHfIHB")
FRAME_HEADER = struct.Struct(">BI")
FLAG_EXTERNAL_SEED = 1
FLAG_TWO_ENTROPY_CODERS = 2


@dataclass(frozen=True)
class SequenceHeader:
    width: int
    height: int
    fps: float
    coded_frames: int
    external_seed: bool
    two_entropy_coders: bool
    reset_interval: int


@dataclass(frozen=True)
class FramePacket:
    qp: int
    bitstream: bytes


class VCMSequenceWriter:
    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        coded_frames: int,
        external_seed: bool = False,
        two_entropy_coders: bool = False,
        reset_interval: int = 0,
    ):
        if not 0 < width < 65536 or not 0 < height < 65536:
            raise ValueError("Sequence dimensions must fit unsigned 16-bit fields")
        if fps <= 0 or coded_frames <= 0:
            raise ValueError("fps and coded_frames must be positive")
        if not 0 <= reset_interval < 65536:
            raise ValueError("reset_interval must fit an unsigned 16-bit field")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file: BinaryIO = self.path.open("wb")
        self.expected_frames = int(coded_frames)
        self.written_frames = 0
        flags = (
            (FLAG_EXTERNAL_SEED if external_seed else 0)
            | (FLAG_TWO_ENTROPY_CODERS if two_entropy_coders else 0)
        )
        self.file.write(
            SEQUENCE_HEADER.pack(
                MAGIC,
                int(width),
                int(height),
                float(fps),
                int(coded_frames),
                int(reset_interval),
                flags,
            )
        )

    def write_frame(self, qp: int, bitstream: bytes) -> None:
        if not 0 <= qp <= 255:
            raise ValueError("qp must fit an unsigned byte")
        if self.written_frames >= self.expected_frames:
            raise RuntimeError("More frames were written than declared in the sequence header")
        self.file.write(FRAME_HEADER.pack(int(qp), len(bitstream)))
        self.file.write(bitstream)
        self.written_frames += 1

    def close(self) -> None:
        if self.file.closed:
            return
        self.file.close()
        if self.written_frames != self.expected_frames:
            raise RuntimeError(
                f"Container declares {self.expected_frames} frames but "
                f"{self.written_frames} were written"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.file.close()


class VCMSequenceReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file: BinaryIO = self.path.open("rb")
        magic = self.file.read(4)
        if len(magic) != 4:
            raise ValueError(f"Truncated sequence header: {self.path}")
        if magic == MAGIC:
            raw_header = magic + self.file.read(SEQUENCE_HEADER.size - 4)
            if len(raw_header) != SEQUENCE_HEADER.size:
                raise ValueError(f"Truncated sequence header: {self.path}")
            (
                _,
                width,
                height,
                fps,
                coded_frames,
                reset_interval,
                flags,
            ) = SEQUENCE_HEADER.unpack(raw_header)
        elif magic == LEGACY_MAGIC:
            raw_header = magic + self.file.read(LEGACY_SEQUENCE_HEADER.size - 4)
            if len(raw_header) != LEGACY_SEQUENCE_HEADER.size:
                raise ValueError(f"Truncated sequence header: {self.path}")
            _, width, height, fps, coded_frames, flags = LEGACY_SEQUENCE_HEADER.unpack(
                raw_header
            )
            reset_interval = 0
        else:
            raise ValueError(f"Invalid VCM bitstream magic in {self.path}")
        self.header = SequenceHeader(
            width=width,
            height=height,
            fps=fps,
            coded_frames=coded_frames,
            external_seed=bool(flags & FLAG_EXTERNAL_SEED),
            two_entropy_coders=bool(flags & FLAG_TWO_ENTROPY_CODERS),
            reset_interval=reset_interval,
        )

    def frames(self) -> Iterator[FramePacket]:
        for _ in range(self.header.coded_frames):
            raw_header = self.file.read(FRAME_HEADER.size)
            if len(raw_header) != FRAME_HEADER.size:
                raise ValueError(f"Truncated frame header in {self.path}")
            qp, length = FRAME_HEADER.unpack(raw_header)
            bitstream = self.file.read(length)
            if len(bitstream) != length:
                raise ValueError(f"Truncated frame payload in {self.path}")
            yield FramePacket(qp=qp, bitstream=bitstream)
        if self.file.read(1):
            raise ValueError(f"Unexpected trailing bytes in {self.path}")

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
