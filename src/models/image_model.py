# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Frozen DCVC-RT intra model used to create and code GOP reference frames."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from ..layers.cuda_inference import round_and_to_int8, should_use_custom_kernel


G_CH_SRC = 3 * 8 * 8
G_CH_ENC_DEC = 368


class IntraEncoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.enc_1 = DepthConvBlock(G_CH_SRC, G_CH_ENC_DEC)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(G_CH_ENC_DEC, G_CH_ENC_DEC),
            DepthConvBlock(G_CH_ENC_DEC, G_CH_ENC_DEC),
            DepthConvBlock(G_CH_ENC_DEC, G_CH_ENC_DEC),
            DepthConvBlock(G_CH_ENC_DEC, G_CH_ENC_DEC),
            DepthConvBlock(G_CH_ENC_DEC, G_CH_ENC_DEC),
            DepthConvBlock(G_CH_ENC_DEC, G_CH_ENC_DEC),
            nn.Conv2d(G_CH_ENC_DEC, channels, 3, stride=2, padding=1),
        )

    def forward(self, image: torch.Tensor, quant_step: torch.Tensor) -> torch.Tensor:
        output = F.pixel_unshuffle(image, 8)
        if not should_use_custom_kernel(image):
            output = self.enc_1(output)
            output = output * quant_step
            return self.enc_2(output)
        output = self.enc_1(output, quant_step=quant_step)
        return self.enc_2(output)


class IntraDecoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(channels, G_CH_ENC_DEC),
            *[DepthConvBlock(G_CH_ENC_DEC, G_CH_ENC_DEC) for _ in range(12)],
        )
        self.dec_2 = DepthConvBlock(G_CH_ENC_DEC, G_CH_SRC)

    def forward(self, latent: torch.Tensor, quant_step: torch.Tensor) -> torch.Tensor:
        if not should_use_custom_kernel(latent):
            output = self.dec_1(latent)
            output = output * quant_step
            output = self.dec_2(output)
            return F.pixel_shuffle(output, 8)

        output = self.dec_1[0](latent)
        for layer in self.dec_1[1:-1]:
            output = layer(output)
        output = self.dec_1[-1](output, quant_step=quant_step)
        output = self.dec_2(output)
        return F.pixel_shuffle(output, 8)


class DMCI(CompressionModel):
    """DCVC-RT intra codec.

    The model is always frozen by the VCM training script. ``forward_reconstruction``
    follows the codec reconstruction path without writing entropy bytes, while
    ``compress``/``decompress`` are used for all-frame actual-bitstream evaluation.
    """

    def __init__(self, channels: int = 256, z_channel: int = 128):
        super().__init__(z_channel=z_channel)
        self.enc = IntraEncoder(channels)
        self.hyper_enc = nn.Sequential(
            DepthConvBlock(channels, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, channels),
        )
        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(channels, channels * 2),
            DepthConvBlock(channels * 2, channels * 2),
            DepthConvBlock(channels * 2, channels * 2),
            nn.Conv2d(channels * 2, channels * 2 + 2, 1),
        )
        self.y_spatial_prior_reduction = nn.Conv2d(channels * 2 + 2, channels, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(
            channels * 2, channels * 2, force_adaptor=True
        )
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(
            channels * 2, channels * 2, force_adaptor=True
        )
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(
            channels * 2, channels * 2, force_adaptor=True
        )
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(channels * 2, channels * 2),
            DepthConvBlock(channels * 2, channels * 2),
            DepthConvBlock(channels * 2, channels * 2),
            nn.Conv2d(channels * 2, channels * 2, 1),
        )
        self.dec = IntraDecoder(channels)
        self.q_scale_enc = nn.Parameter(
            torch.ones((self.get_qp_num(), G_CH_ENC_DEC, 1, 1))
        )
        self.q_scale_dec = nn.Parameter(
            torch.ones((self.get_qp_num(), G_CH_ENC_DEC, 1, 1))
        )

    def _analysis_synthesis(self, image: torch.Tensor, qp: int):
        if not 0 <= int(qp) < self.get_qp_num():
            raise ValueError(f"DMCI QP must be in [0, 63], got {qp}")
        q_enc = self.q_scale_enc[qp : qp + 1]
        q_dec = self.q_scale_dec[qp : qp + 1]
        y = self.enc(image, q_enc)
        z = self.hyper_enc(self.pad_for_y(y))
        z_hat, z_hat_write = round_and_to_int8(z)
        params = self.y_prior_fusion(self.hyper_dec(z_hat))
        y_height, y_width = y.shape[-2:]
        params = params[:, :, :y_height, :y_width].contiguous()
        prior = self.compress_prior_4x(
            y,
            params,
            self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptor_1,
            self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3,
            self.y_spatial_prior,
        )
        *rate_tensors, y_hat = prior
        x_hat = self.dec(y_hat, q_dec).clamp(0.0, 1.0)
        return x_hat, z_hat_write, rate_tensors

    def forward_reconstruction(self, image: torch.Tensor, qp: int) -> torch.Tensor:
        """Return the quantized frozen I-frame reconstruction without entropy I/O."""
        reconstruction, _, _ = self._analysis_synthesis(image, qp)
        return reconstruction

    def compress(self, image: torch.Tensor, qp: int):
        device = image.device
        reconstruction, z_hat_write, rate_tensors = self._analysis_synthesis(image, qp)
        (
            y_q_w_0,
            y_q_w_1,
            y_q_w_2,
            y_q_w_3,
            s_w_0,
            s_w_1,
            s_w_2,
            s_w_3,
        ) = rate_tensors

        rate_z = self.bit_estimator_z.forward_rate(
            z_hat_write.to(dtype=image.dtype),
            qp,
        ).sum()
        rate_y = sum(
            self.gaussian_encoder.forward_rate(symbols, scales).sum()
            for symbols, scales in (
                (y_q_w_0, s_w_0),
                (y_q_w_1, s_w_1),
                (y_q_w_2, s_w_2),
                (y_q_w_3, s_w_3),
            )
        )

        self.entropy_coder.reset()
        self.bit_estimator_z.encode_z(z_hat_write, qp)
        for symbols, scales in (
            (y_q_w_0, s_w_0),
            (y_q_w_1, s_w_1),
            (y_q_w_2, s_w_2),
            (y_q_w_3, s_w_3),
        ):
            self.gaussian_encoder.encode_y(symbols, scales)
        self.entropy_coder.flush()
        bit_stream = self.entropy_coder.get_encoded_stream()
        if image.is_cuda:
            torch.cuda.synchronize(device=device)
        return {
            "bit_stream": bit_stream,
            "x_hat": reconstruction,
            "estimated_entropy_bits": float((rate_z + rate_y).detach().cpu()),
        }

    def decompress(self, bit_stream: bytes, sps: dict, qp: int):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        q_dec = self.q_scale_dec[qp : qp + 1]
        self.entropy_coder.set_use_two_entropy_coders(sps["ec_part"] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps["height"], sps["width"], 64)
        y_height, y_width = self.get_downsampled_shape(
            sps["height"], sps["width"], 16
        )
        self.bit_estimator_z.decode_z(z_size, qp)
        z_hat = self.bit_estimator_z.get_z(z_size, device, dtype)
        params = self.y_prior_fusion(self.hyper_dec(z_hat))
        params = params[:, :, :y_height, :y_width].contiguous()
        y_hat = self.decompress_prior_4x(
            params,
            self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptor_1,
            self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3,
            self.y_spatial_prior,
        )
        return {"x_hat": self.dec(y_hat, q_dec).clamp(0.0, 1.0)}
