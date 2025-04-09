"""
This is the implementation of the continuous form of CNN+Transformer VQGAN Mixed Structure 
Reference:
# LightningDiT: https://github.com/hustvl/LightningDiT/blob/main/tokenizer/vavae.py
# LDM: https://github.com/CompVis/latent-diffusion/blob/main/ldm/
"""
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, reduce, pack, unpack

import numpy as np

from tokenizer.tokenizer_image.vq.blocks import (
    ViTEncoder, ViTDecoder, Encoder, Decoder,
    ViTEncoder2D, ViTDecoder2D,
    ChannelDownsampleResidual, ChannelUpsampleResidual,
)


# the KL related parameters are set by default according to
# https://github.com/CompVis/latent-diffusion/blob/main/configs/autoencoder/autoencoder_kl_16x16x16.yaml
# the rest are set the same as our VQ model
@dataclass
class VAEVitModelArgs:
    # for latent compression
    latent_embed_dim: int = 16
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    model_size: str = 'small'
    encoder_size: str = None 
    decoder_size: str = None
    num_latent_tokens: int = 256
    nb_z_channels: int = 256   # the dimension of the intermediate downsample/upsample (neighbor) to/from codebook dimension
    dropout_p: float = 0.0

    # the setting for initializing the 1d queries for the 2dto1d encoder
    multi_level_query_init: bool = False
    learnable_1d_query_init: bool = False

    # the initialization for the 2d queries. the "level" corresponds to the 
    # "level" division for "multi_level_query_init". It assumes 1d tokens have levels
    # all false means simply using the global average of the 1d tokens to initialize
    # the 2d queries.
    last_level_2d_query_init: bool = False
    multi_level_2d_query_init: bool = False
    learnable_2d_query_init: bool = False

    # tricks for the CNN 2d decoder
    adaptive_gn: bool = False
    d2s_up: bool = False
    res_up_down_sample: bool = False
    downsample_match_channel: bool = False
    upsample_match_channel: bool = False
    res_codebook_updown_sample: bool = False
    downsample_improve: bool = False
    # whether to use attention in the 2d encoder or decoder
    # suggested not to. May be unstable and slower
    use_attn: bool = True
    # rope 2d only supports the 1dto2d decoder queries (since Q-former)
    rope_2d: bool = False

    # distillation setting
    distill_depth: int = None
    # whether to distill from encoder. Not tested yet.
    encoder_2d_distill: bool = False

    # for semantic distillation regularization
    # the default 768 is for dino-v2 base
    out_inner_dim: int = 768

    fea_rec_loss_type: str = "cosine"
    fea_rec_loss_weight: float = 1.0


class VAEVitModel(nn.Module):
    def __init__(self, config: VAEVitModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
                        ch_mult=config.encoder_ch_mult, 
                        z_channels=config.nb_z_channels, 
                        dropout=config.dropout_p, 
                        use_attn=config.use_attn,
                        res_down_sample=config.res_up_down_sample,
                        downsample_match_channel=config.downsample_match_channel,
                        )

        if config.encoder_2d_distill:
            # setting is from REPA
            self.distill_mlp = nn.Sequential(
                    nn.Linear(config.nb_z_channels, config.nb_z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.nb_z_channels * 4, config.nb_z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.nb_z_channels * 4, config.out_inner_dim),
                    )


        # set the size of the transformer encoder/decoder size
        encoder_size = config.model_size if config.encoder_size is None else config.encoder_size
        decoder_size = config.model_size if config.decoder_size is None else config.decoder_size
        # when encoder size or decoder size is given, model size should be none
        if config.encoder_size is not None or config.decoder_size is not None:
            assert config.model_size is None
        
        if config.encoder_2d_distill:
            assert config.distill_depth is None



        self.s2to1encoder = ViTEncoder(
                               model_size=encoder_size, 
                               num_latent_tokens=config.num_latent_tokens, 
                               token_size=config.nb_z_channels,
                               dropout=config.dropout_p, 
                               patch_size=2**(len(config.encoder_ch_mult) - 1),
                               multi_level_query_init=config.multi_level_query_init,
                               learnable_1d_query_init=config.learnable_1d_query_init,
                               downsample_improve=config.downsample_improve,
                               )

        self.s1to2decoder = ViTDecoder(
                            model_size=decoder_size, 
                            num_latent_tokens=config.num_latent_tokens, 
                            token_size=config.nb_z_channels, 
                            dropout=config.dropout_p,
                            patch_size=2**(len(config.decoder_ch_mult) - 1),
                            last_level_2d_query_init=config.last_level_2d_query_init,
                            multi_level_2d_query_init=config.multi_level_2d_query_init,
                            learnable_2d_query_init=config.learnable_2d_query_init,
                            out_inner_feat=config.distill_depth is not None,
                            out_inner_depth=config.distill_depth,
                            out_inner_dim=config.out_inner_dim,
                            )

        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, 
                               z_channels=config.nb_z_channels, 
                               dropout=config.dropout_p,
                               adaptive_gn=config.adaptive_gn,
                               d2s_up=config.d2s_up,
                               use_attn=config.use_attn,
                               res_up_sample=config.res_up_down_sample,
                               upsample_match_channel=config.upsample_match_channel,
                               )

        self.num_latent_tokens = config.num_latent_tokens

        if self.config.res_codebook_updown_sample:
            if self.config.downsample_improve:
                self.quant_conv = ChannelDownsampleResidual(self.s2to1encoder.width, config.latent_embed_dim)
            else:
                self.quant_conv = ChannelDownsampleResidual(config.nb_z_channels, config.latent_embed_dim)

            # for vae, the latent dimension is [mean, logvar], after sampling the dimnsion will be half
            self.post_quant_conv = ChannelUpsampleResidual(config.latent_embed_dim // 2, self.config.nb_z_channels)
        else:
            self.quant_conv = nn.Conv2d(self.config.nb_z_channels, config.latent_embed_dim, 1)
            # for vae, the latent dimension is [mean, logvar], after sampling the dimnsion will be half
            self.post_quant_conv = nn.Conv2d(config.latent_embed_dim // 2, self.config.nb_z_channels, 1)
        
        self.freeze_but_2d_decoder_flag = False

        def nan_hook(self, inp, output):
            if not isinstance(output, torch.Tensor):
                return
            if torch.isnan(output).any():
                print(f"NaN detected in {self}")
                raise RuntimeError("NaN detected")

        # for name, module in self.named_modules():
        #     module.register_forward_hook(nan_hook)

        self.apply(self._init_weights)


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        wrap_modules = []
        # Add encoder layers
        for layer in self.s2to1encoder.transformer:
            wrap_modules.append(layer)
        # Add decoder layers
        for layer in self.s1to2decoder.transformer:
            wrap_modules.append(layer)

        return wrap_modules
    

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def get_encoder_last_layer(self):
        return self.quant_conv.weight
    
    def get_distill_layer(self):
        return self.s1to2decoder.transformer[self.s1to2decoder.out_inner_depth].linear2.weight
 

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, 
               return_latent_only=True, 
               return_feat=False,   # for linear probe
               return_fix_dim_feat=False,   # the fix dimension is the nb_z_channel
               num_en_q_level=None, 
               causal_type=None,        # deprecated
               random_mix_reg=False,    # deprecated
               replace_ratio=None,      # deprecated
               global_step=None,        # prior model related, deprecated
               max_steps=None,          # prior model related, deprecated
               ):
        # causal_type = causal_type if causal_type is not None else self.config.causal_type
        if return_feat:
            assert (not return_latent_only) and (not return_fix_dim_feat)
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level, 
                causal_type=causal_type, 
                return_feat=True)
            # return the feature of exactly the same width as the vit encoder
            return h
        
        if return_fix_dim_feat:
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level,
                causal_type=causal_type,
                return_feat=False)
            # return the feature of the fix width (e.g. 256) before further downsampled to codebook dim
            return h

        s = self.encoder(x)
        h = self.s2to1encoder(s, num_q_level=num_en_q_level, causal_type=causal_type)

        h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        if return_latent_only:
            return posterior, None

        return posterior, s

    def decode(self, z, ret_inner_feat=False):
        z = self.post_quant_conv(z)
        if ret_inner_feat:
            rec_spatial, inner_feat = self.s1to2decoder(z, ret_inner_feat=True)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, inner_feat
        else:
            rec_spatial = self.s1to2decoder(z)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec

    def forward(
            self, 
            input, 
            num_en_q_level=None, 
            causal_type=None, 
            ret_posteriors=False,
            # rec_loss=True, # used for shortcut direct reconstruction.
            ret_inner_feat=False,
            random_mix_reg=False,   # deprecated
            replace_ratio=None,     # deprecated
            global_step=None,       # prior model related, deprecated
            max_steps=None,         # prior model related, deprecated
            ):
        posteriors, spatial = self.encode(
                            input, 
                            return_latent_only=False, 
                            num_en_q_level=num_en_q_level, 
                            causal_type=causal_type,
                            random_mix_reg=random_mix_reg,
                            replace_ratio=replace_ratio,
                            global_step=global_step,
                            max_steps=max_steps
                            )
        z = posteriors.sample()
        if ret_inner_feat:
            if self.config.encoder_2d_distill:
                inner_feat = rearrange(spatial, 'b c h w -> b (h w) c')
                inner_feat = self.distill_mlp(inner_feat)
                dec = self.decode(z)
            else:
                dec, inner_feat = self.decode(z, ret_inner_feat=True)
        else:
            dec = self.decode(z)

        return dec, \
               posteriors if ret_posteriors else None, \
               inner_feat if ret_inner_feat else None

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean

def compute_cosinesim_loss(feat1, feat2, dim):
    cos_sim = F.cosine_similarity(feat1, feat2, dim=dim)
    loss = 1 - cos_sim
    return torch.mean(loss)  



