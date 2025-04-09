"""
Disentanglement via Latent Quantization
 - https://arxiv.org/abs/2305.18378
Code adapted from Jax version in https://github.com/kylehkhsu/latent_quantization
"""

from __future__ import annotations
from typing import Callable, List
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import Tensor, int32, nn, int64
from torch.nn import Module
from torch.optim import Optimizer

from tokenizer.tokenizer_image.vq.blocks import Encoder, Decoder

from tokenizer.tokenizer_image.lq.lq_vit_model import (
    LQ_16_vit_s128, LQ_16_vit_b64, LQ_16_vit_l32,
    LQ_8_vit_s128, LQ_8_vit_b64, LQ_8_vit_l32
)





@dataclass
class ModelArgs:
    levels: List[int] = field(default_factory=lambda: [8, 5, 5, 5])
    codebook_show_usage: bool = True
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    # codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25

# helper functions
def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class LatentQuantize(Module):
    def __init__(
        self,
        levels: List[int] | int,
        dim: int | None = None,
        commitment_loss_weight: float | None = 0.1,
        quantization_loss_weight: float | None = 0.1,
        num_codebooks: int = 1,
        codebook_dim: int = -1,
        keep_num_codebooks_dim: bool | None = None,
        optimize_values: bool | None = True,
        in_place_codebook_optimizer: Callable[
            ..., Optimizer
        ] = None,  # Optimizer used to update the codebook embedding if using learnable_codebook
    ):
        """
        Initializes the LatentQuantization module.

        Args:
            levels (List[int]|init): The number of levels per codebook.
                If an int is provided, it is used for all codebooks.
            dim (int): The dimensionality of the input tensor.
                The input tensor is expected to be of shape [B D ...]
            num_codebooks (int): The number of codebooks to use.
                (default is 1)
            codebook_dim (int): the dimension of the codebook.
                If levels is a list, codebook_dim is the length of the list.
                (default to -1) 
            keep_num_codebooks_dim (Optional[bool]): Whether to keep the number of codebooks dimension in the output tensor. If not provided, it is set to True if num_codebooks > 1, otherwise False.
            optimize_values (Optional[bool]): Whether to optimize the values of the codebook. If not provided, it is set to True.
        """
        super().__init__()

        self.dim = dim if dim is not None else len(levels)
        self.in_place_codebook_optimizer = in_place_codebook_optimizer
        _levels = torch.tensor(levels, dtype=int32)

        # if levels is an int, use it for all codebooks
        if isinstance(levels, int):
            try:
                _levels = _levels.repeat(codebook_dim)
            except RuntimeError as e:
                raise e
        self.register_buffer(
            "commitment_loss_weight",
            torch.tensor(commitment_loss_weight, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "quantization_loss_weight",
            torch.tensor(quantization_loss_weight, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(
            torch.concat([torch.tensor([1], dtype=int32), _levels[:-1]], dim=0), dim=0
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.codebook_dim = codebook_dim if codebook_dim > 0 else len(_levels)

        effective_codebook_dim = self.codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = (
            keep_num_codebooks_dim if keep_num_codebooks_dim else num_codebooks > 1
        )
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        has_projections = self.dim != effective_codebook_dim
        if has_projections:
            self.project_in = (
                nn.Linear(self.dim, effective_codebook_dim)
                if has_projections
                else nn.Identity()
            )
            self.project_out = (
                nn.Linear(effective_codebook_dim, self.dim)
                if has_projections
                else nn.Identity()
            )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        values_per_latent = [
            torch.linspace(-1, 1, level)
            if level % 2 == 1
            else torch.arange(level) / (level // 2) - 1
            for level in _levels
        ]  # ensure zero is in the middle and start is always -1.0

        # test, and check whether it would be in the parameters of the model or not
        if optimize_values:
            self.values_per_latent = nn.ParameterList(
                [nn.Parameter(values) for values in values_per_latent]
            )
            if in_place_codebook_optimizer is not None:
                self.in_place_codebook_optimizer = in_place_codebook_optimizer(
                    self.values_per_latent
                )
        else:
            self.values_per_latent = values_per_latent  # are there any scenarios where this would have its gradients updated?
        
        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)


    def quantization_loss(self, z: Tensor, zhat: Tensor, reduce="mean") -> Tensor:
        """Computes the quantization loss."""
        return F.mse_loss(zhat.detach(), z, reduction=reduce)

    def commitment_loss(self, z: Tensor, zhat: Tensor, reduce="mean") -> Tensor:
        """Computes the commitment loss."""
        return F.mse_loss(z.detach(), zhat, reduction=reduce)

    def bound(self, z, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d).
        n is the per-dimension level
        if n is odd -> [- (n-1)//2, (n-1)//2] 
        if n is even -> [- n//2, n//2 - 1]
        """
        z = rearrange(z, "b d ... -> b ... d")
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        res = (z + shift).tanh() * half_l - offset
        return rearrange(res, "b ... d -> b d ...")

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z.
        The quantization is done by measuring the distance between the input and the codebook values per latent dimension
        and returning the index of the closest codebook value.
        """

        def distance(x, y):
            return torch.abs(x - y)
        

        index_per_dim = torch.stack(
            [
                torch.argmin(
                    distance(z[..., i, None], self.values_per_latent[i]), dim=-1
                )
                for i in range(self.codebook_dim)
            ],
            dim=-1,
        )
        quantize = torch.stack(
            [
                self.values_per_latent[i][index_per_dim[..., i]]
                for i in range(self.codebook_dim)
            ],
            dim=-1,
        )

        # STE may cause little difference compared with directly selecting from index_per_dim
        quantize = z + (quantize - z).detach()
        # half_width = self._levels // 2 / 2  # Renormalize to [-0.5, 0.5].
        return quantize, index_per_dim  # / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        """scale and shift zhat from [-0.5, 0.5] to [0, level_per_dim]"""
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        """normalize zhat to [-0.5, 0.5]
        if n is odd: [0, n-1]-> [-1, 1] 
        """
        half_width = self._levels // 2
        return (zhat - half_width) / half_width / 2

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` which contains the number per latent to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat.round().to(int32) * self._basis).sum(dim=-1).round().to(int64)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""
        indices = rearrange(indices, "... -> ... 1")
        level_indices = (indices // self._basis) % self._levels

        codes = torch.stack(
            [
                self.values_per_latent[i][level_indices[..., i]]
                for i in range(self.codebook_dim)
            ],
            dim=-1,
        )
        # codes = self._scale_and_shift_inverse(level_indices.to(int64))
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if self.has_projections and project_out:
            codes = self.project_out(codes)

        codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def quantize_and_project(self, z: Tensor, is_img_or_video, ps) -> Tensor:
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)
        out = unpack_one(out, ps, "b * d")
        out = rearrange(out, "b ... d -> b d ...")

        indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")
        return codes, out, indices

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        z = self.bound(z)
        original_input = z
        should_inplace_optimize = self.in_place_codebook_optimizer is not None

        z = rearrange(z, "b d ... -> b ... d")
        z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        # project in
        if self.has_projections:
            z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes, index_per_dim = self.quantize(z)
        indices = self.codes_to_indices(codes)

        # assert (indices == self.level_indices_to_indices(index_per_dim)).all(), f"{indices.shape}, {index_per_dim.shape}"
        # assert (index_per_dim == self.indices_to_level_indices(indices)).all(), f"{indices.shape}, {index_per_dim.shape}"

        codes = rearrange(codes, "b n c d -> b n (c d)")

        if self.has_projections:
            out = self.project_out(codes)
        else:
            out = codes
        # print(out.shape)
        out = unpack_one(out, ps, "b * d")
        out = rearrange(out, "b ... d -> b d ...")
        # print(out.shape)

        # assert (out == rearrange(codes, "b (h w) c -> b c h w", h=32)).all()

        indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        if should_inplace_optimize and self.training and not self.optimize_values:
            # update codebook
            loss = (
                self.commitment_loss(z, out)
                if self.commitment_loss_weight != 0
                else torch.tensor(0.0)
            )
            loss += (
                self.quantization_loss(z, out)
                if self.quantization_loss_weight != 0
                else torch.tensor(0.0)
            )
            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()
            # quantize again
            codes = self.quantize(z)
            indices = self.codes_to_indices(codes)
            codes = rearrange(codes, "b n c d -> b n (c d)")
            out = self.project_out(codes)

            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

            if not self.keep_num_codebooks_dim:
                indices = rearrange(indices, "... 1 -> ...")

        # calculate losses
        commitment_loss = (
            self.commitment_loss(original_input, out)
            if self.training and self.commitment_loss_weight != 0
            else torch.tensor(0.0)
        )
        quantization_loss = (
            self.quantization_loss(original_input, out)
            if self.training and self.quantization_loss_weight != 0
            else torch.tensor(0.0)
        )

        loss = (
            self.commitment_loss_weight * commitment_loss
            + self.quantization_loss_weight * quantization_loss
        )

        return out, indices, loss

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered.to(int64)
    
    def level_indices_to_indices(self, level_indices: torch.Tensor):
        indices = (level_indices * self._basis).sum(dim=-1)
        return indices.to(int64)


class LQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)

        self.quantize = LatentQuantize(levels=config.levels, commitment_loss_weight=config.commit_loss_beta, quantization_loss_weight=1.0)
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


#################################################################################
#                              LQ Model Configs                                 #
#################################################################################
def LQ_8(**kwargs):
    return LQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def LQ_16(**kwargs):
    return LQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


LQ_models = {'LQ-16': LQ_16, 
             'LQ-8': LQ_8, 
             'LQ-16-vit-s128': LQ_16_vit_s128, 
             'LQ-16-vit-b64': LQ_16_vit_b64, 
             'LQ-16-vit-l32': LQ_16_vit_l32,
             'LQ-8-vit-s128': LQ_8_vit_s128,
             'LQ-8-vit-b64': LQ_8_vit_b64, 
             'LQ-8-vit-l32': LQ_8_vit_l32,
            }