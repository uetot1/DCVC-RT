"""Complete train-time DCVC-RT VCM system.

The intra codec and teacher detector are frozen. Only DMC and, by default, the
cloned YOLO front end receive gradients. Codec tensors are BT.709 YCbCr 4:4:4;
task features and RGB diagnostics are computed after conversion back to RGB.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .image_model import DMCI
from .vcm_loss import VCMLoss
from .video_model import DMC
from ..utils.transforms import rgb2ycbcr, ycbcr2rgb


QP_INDEX_MAP = (0, 1, 0, 2, 0, 2, 0, 2)
QP_SHIFTS = (0, 8, 4)
QP_OFFSETS = tuple(QP_SHIFTS[index] for index in QP_INDEX_MAP)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


class VideoVCMSystem(nn.Module):
    """Frozen DMCI + trainable DMC + multi-level YOLO feature objective."""

    def __init__(self, image_model: DMCI, dmc: DMC, objective: VCMLoss):
        super().__init__()
        self.image_model = image_model
        self.dmc = dmc
        self.objective = objective
        for parameter in self.image_model.parameters():
            parameter.requires_grad_(False)
        self.image_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_model.eval()
        self.objective.train(mode)
        return self

    @torch.no_grad()
    def prepare_gop(self, seed_rgb: torch.Tensor, base_qp: int) -> dict[str, torch.Tensor]:
        """Code frame 0 with frozen DMCI and seed DMC with the reconstruction."""
        seed_ycbcr = rgb2ycbcr(seed_rgb)
        reconstructed_ycbcr = self.image_model.forward_reconstruction(
            seed_ycbcr,
            int(base_qp),
        )
        self.dmc.clear_dpb()
        self.dmc.set_curr_poc(0)
        self.dmc.add_ref_frame(feature=None, frame=reconstructed_ycbcr.detach())
        reconstructed_rgb = ycbcr2rgb(reconstructed_ycbcr)
        rgb_mse = F.mse_loss(reconstructed_rgb, seed_rgb)
        return {
            "i_rgb_mse": rgb_mse,
            "i_rgb_psnr": psnr_from_mse(rgb_mse),
        }

    def detach_dpb(self) -> None:
        self.dmc.detach_dpb()

    @torch.no_grad()
    def reconstruct_gop(
        self,
        rgb_frames: torch.Tensor,
        base_qp: int,
    ) -> torch.Tensor:
        """Reconstruct a validation GOP without computing the training loss."""
        if rgb_frames.ndim != 5 or rgb_frames.shape[2] != 3:
            raise ValueError("rgb_frames must have shape [B, T, 3, H, W]")
        seed_ycbcr = rgb2ycbcr(rgb_frames[:, 0])
        reconstructed_ycbcr = self.image_model.forward_reconstruction(
            seed_ycbcr,
            int(base_qp),
        )
        self.dmc.clear_dpb()
        self.dmc.set_curr_poc(0)
        self.dmc.add_ref_frame(feature=None, frame=reconstructed_ycbcr.detach())
        reconstructions = [ycbcr2rgb(reconstructed_ycbcr)]
        for frame_index in range(1, rgb_frames.shape[1]):
            coding_qp = int(base_qp) + QP_OFFSETS[frame_index % len(QP_OFFSETS)]
            reconstructed_ycbcr, _ = self.dmc.forward_train(
                rgb2ycbcr(rgb_frames[:, frame_index]),
                coding_qp,
            )
            reconstructions.append(ycbcr2rgb(reconstructed_ycbcr))
        return torch.stack(reconstructions, dim=1)

    def forward(
        self,
        rgb_frames: torch.Tensor,
        base_qp: int,
        start_frame_index: int,
        lambda_feature: float,
        frame_loss_weights: Sequence[float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train a contiguous P-frame chunk after :meth:`prepare_gop`.

        ``start_frame_index`` is the frame index in the GOP (frame 0 is the
        DMCI-coded I-frame). It selects DCVC-RT's published feature-adaptor QP
        shift pattern ``0,8,0,4,0,4,...``.
        """
        if rgb_frames.ndim != 5 or rgb_frames.shape[2] != 3:
            raise ValueError("rgb_frames must have shape [B, T, 3, H, W]")
        if rgb_frames.shape[1] < 1:
            raise ValueError("A training chunk must contain at least one P-frame")
        if start_frame_index < 1:
            raise ValueError("start_frame_index must be at least 1")
        if frame_loss_weights is None:
            frame_loss_weights = (1.0,) * rgb_frames.shape[1]
        if len(frame_loss_weights) != rgb_frames.shape[1]:
            raise ValueError("frame_loss_weights must match the chunk length")

        entries: list[dict[str, torch.Tensor]] = []
        for local_index in range(rgb_frames.shape[1]):
            frame_index = start_frame_index + local_index
            coding_qp = int(base_qp) + QP_OFFSETS[frame_index % len(QP_OFFSETS)]
            original_rgb = rgb_frames[:, local_index]
            original_ycbcr = rgb2ycbcr(original_rgb)
            reconstructed_ycbcr, estimated_bpp = self.dmc.forward_train(
                original_ycbcr,
                coding_qp,
            )
            reconstructed_rgb = ycbcr2rgb(reconstructed_ycbcr)
            details = self.objective(
                original_rgb,
                reconstructed_rgb,
                estimated_bpp,
                lambda_feature,
                distortion_weight=float(frame_loss_weights[local_index]),
                return_details=True,
            )
            rgb_mse = F.mse_loss(reconstructed_rgb, original_rgb)
            y_mse = F.mse_loss(reconstructed_ycbcr[:, :1], original_ycbcr[:, :1])
            chroma_mse = F.mse_loss(
                reconstructed_ycbcr[:, 1:],
                original_ycbcr[:, 1:],
            )
            details.update(
                {
                    "rgb_mse": rgb_mse,
                    "rgb_psnr": psnr_from_mse(rgb_mse),
                    "y_mse": y_mse,
                    "chroma_mse": chroma_mse,
                    "base_qp": details["total_loss"].new_tensor(float(base_qp)),
                    "coding_qp": details["total_loss"].new_tensor(float(coding_qp)),
                    "lambda_feature": details["total_loss"].new_tensor(
                        float(lambda_feature)
                    ),
                }
            )
            entries.append(details)

        metrics = {
            key: torch.stack([entry[key] for entry in entries]).mean()
            for key in entries[0]
        }
        return metrics["total_loss"], metrics

    def trainable_named_parameters(self):
        yield from ((f"dmc.{name}", parameter) for name, parameter in self.dmc.named_parameters())
        yield from (
            (f"cloned_frontend.{name}", parameter)
            for name, parameter in self.objective.cloned_frontend_named_parameters()
        )

    @staticmethod
    def selection_score(metrics: dict[str, float]) -> float:
        """Return log objective for scale-invariant aggregation across QPs.

        Averaging raw objectives lets the largest-lambda QP dominate. Averaging
        their logarithms (equivalently, using a geometric mean) is invariant to
        a fixed multiplicative scale at any individual rate point.
        """
        return math.log(max(float(metrics["total_loss"]), 1e-12))


def validate_qp_pattern() -> None:
    if QP_OFFSETS != (0, 8, 0, 4, 0, 4, 0, 4):
        raise RuntimeError("DCVC-RT feature-adaptor QP pattern changed unexpectedly")
    if max(QP_OFFSETS) + 63 > 71:
        raise RuntimeError("QP shift exceeds the DCVC-RT DMC parameter range")


validate_qp_pattern()
