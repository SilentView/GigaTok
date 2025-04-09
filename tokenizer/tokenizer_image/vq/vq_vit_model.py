# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
#   REPA: https://github.com/sihyun-yu/REPA
#   DETR: https://github.com/facebookresearch/detr
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, reduce, pack, unpack

import numpy as np


from tokenizer.tokenizer_image.vq.blocks import (
    ViTEncoder, ViTDecoder, Encoder, Decoder, ViTDecoder_V2,
    ViTEncoder2D, ViTDecoder2D,
    ChannelDownsampleResidual, ChannelUpsampleResidual,
)
from tokenizer.tokenizer_image.vq.gptc import (
    GPTC_models
)

from tokenizer.tokenizer_image.vq.lfq import (
    LFQ
)



def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


@dataclass
class VQVitModelPlusArgs:

    # for quantization
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    # tricks for lfq. deprecated
    use_lfq: bool = False
    bernoulli_sample: bool = False
    eval_deterministic: bool = False

    # SimVQ trick, deprecated
    simvq: bool = False
    codebook_transform: str = None
    freeze_codebook: bool = False
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    model_size: str = 'small'
    encoder_size: str = None 
    decoder_size: str = None
    num_latent_tokens: int = 256
    z_channels: int = 256   # the dimension of the intermediate downsample towards codebook dimension
    dropout_p: float = 0.0

    # use rope for the decoder Q-former attention. 
    use_rope: bool = False
    use_qk_norm: bool = False

    # TODO: remove option. flash attention is automatically used when calling scaled_dot_product_attention
    use_flash_attn: bool = False

    # the setting for initializing the 1d queries for the 2dto1d encoder
    multi_level_query_init: bool = False
    learnable_1d_query_init: bool = False
    rope_1d: bool = False

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

    # the rotation trick for quantizer. The influence is limited
    rot: bool = False

    # for stochastic quantization. Closed by default
    stochastic: bool = False
    stochastic_temperature: float = 0.03

    # distillation setting
    distill_depth: int = None
    # whether to distill from encoder. Not tested yet.
    # (to be deleted)
    encoder_2d_distill: bool = False

    # for semantic distillation regularization
    # the default 768 is for dino-v2 base
    out_inner_dim: int = 768

    fea_rec_loss_type: str = "cosine"
    fea_rec_loss_weight: float = 1.0

    # for gptc model, which tries to utilize AR prior for 
    # training tokenizers. The effect is limited and this feature
    # is deprecated.
    # for ar prior model
    with_prior_model: bool = False
    prior_model_config: dict = None




class VQVitModelPlus(nn.Module):
    def __init__(self, config: VQVitModelPlusArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
                        ch_mult=config.encoder_ch_mult, 
                        z_channels=config.z_channels, 
                        dropout=config.dropout_p, 
                        use_attn=config.use_attn,
                        res_down_sample=config.res_up_down_sample,
                        downsample_match_channel=config.downsample_match_channel,
                        )

        if config.encoder_2d_distill:
            # setting is from REPA
            self.distill_mlp = nn.Sequential(
                    nn.Linear(config.z_channels, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.out_inner_dim),
                    )


        # set the size of the transformer encoder/decoder size
        encoder_size = config.model_size if config.encoder_size is None else config.encoder_size
        decoder_size = config.model_size if config.decoder_size is None else config.decoder_size
       
        # when encoder size or decoder size is given, model size should be none
        if config.encoder_size is not None or config.decoder_size is not None:
            assert config.model_size is None
        
        if config.encoder_2d_distill:
            assert config.distill_depth is None



        self.s2to1encoder = ViTEncoder(model_size=encoder_size, num_latent_tokens=config.num_latent_tokens, 
                               token_size=config.z_channels, dropout=config.dropout_p, 
                               patch_size=2**(len(config.encoder_ch_mult) - 1),
                               multi_level_query_init=config.multi_level_query_init,
                               learnable_1d_query_init=config.learnable_1d_query_init,
                               rope_1d=config.rope_1d,
                               downsample_improve=config.downsample_improve,
                               use_qk_norm=config.use_qk_norm,
                               use_flash_attn=config.use_flash_attn,
                               )

        if config.use_rope:
            # V2 model is specifically designed for rope2d
            self.s1to2decoder = ViTDecoder_V2(model_size=decoder_size, num_latent_tokens=config.num_latent_tokens, 
                                token_size=config.z_channels, dropout=config.dropout_p,
                                patch_size=2**(len(config.decoder_ch_mult) - 1),
                                last_level_2d_query_init=config.last_level_2d_query_init,
                                multi_level_2d_query_init=config.multi_level_2d_query_init,
                                learnable_2d_query_init=config.learnable_2d_query_init,
                                rope_2d=True,
                                use_qk_norm=config.use_qk_norm,
                                use_flash_attn=config.use_flash_attn,
                                )
        
        else:
            self.s1to2decoder = ViTDecoder(model_size=decoder_size, num_latent_tokens=config.num_latent_tokens, 
                                token_size=config.z_channels, dropout=config.dropout_p,
                                patch_size=2**(len(config.decoder_ch_mult) - 1),
                                last_level_2d_query_init=config.last_level_2d_query_init,
                                multi_level_2d_query_init=config.multi_level_2d_query_init,
                                learnable_2d_query_init=config.learnable_2d_query_init,
                                out_inner_feat=config.distill_depth is not None,
                                out_inner_depth=config.distill_depth,
                                out_inner_dim=config.out_inner_dim,
                                use_qk_norm=config.use_qk_norm,
                                use_flash_attn=config.use_flash_attn,
                                )

        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, 
                               z_channels=config.z_channels, 
                               dropout=config.dropout_p,
                               adaptive_gn=config.adaptive_gn,
                               d2s_up=config.d2s_up,
                               use_attn=config.use_attn,
                               res_up_sample=config.res_up_down_sample,
                               upsample_match_channel=config.upsample_match_channel,
                               )

        self.num_latent_tokens = config.num_latent_tokens
        # scale = self.s2to1encoder.width ** -0.5
        # self.latent_tokens = nn.Parameter(
        #     scale * torch.randn(self.num_latent_tokens, self.s2to1encoder.width))

        # the weight initialization seems to have ignored post_quant_conv and pre_quant_conv
        # and it potentially affects the prior model(deprecated) training
        # but weight initialization is also ignored(?) in llamagen implementation
        # currently this setting can just work.
        self.apply(self._init_weights)

        if self.config.with_prior_model:
            if self.config.use_lfq:
                raise NotImplementedError("LFQ is not implemented yet")
            else:
                self.quantize = VectorQuantizerWithPM(
                                                config.codebook_size, config.codebook_embed_dim, 
                                                config.commit_loss_beta, config.entropy_loss_ratio,
                                                config.codebook_l2_norm, config.codebook_show_usage,
                                                rot=config.rot, stochastic=config.stochastic,
                                                stochastic_temperature=config.stochastic_temperature,
                                                prior_model_config=config.prior_model_config,
                                                simvq=config.simvq,
                                                codebook_transform=config.codebook_transform,
                                                freeze_codebook=config.freeze_codebook,
                                        )

            
        else:
            if self.config.use_lfq:
                self.quantize = LFQ(
                    dim=config.codebook_embed_dim,
                    beta=config.commit_loss_beta,
                    entropy_loss_ratio=config.entropy_loss_ratio,
                    n_e=config.codebook_size,
                )
            else:
                self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                                config.commit_loss_beta, config.entropy_loss_ratio,
                                                config.codebook_l2_norm, config.codebook_show_usage,
                                                rot=config.rot, stochastic=config.stochastic,
                                                stochastic_temperature=config.stochastic_temperature,
                                                eval_deterministic=config.eval_deterministic,
                                                simvq=config.simvq,
                                                codebook_transform=config.codebook_transform,
                                                freeze_codebook=config.freeze_codebook,
                                                )

        if self.config.res_codebook_updown_sample:

            if self.config.downsample_improve:
                self.quant_conv = ChannelDownsampleResidual(self.s2to1encoder.width, config.codebook_embed_dim)
            else:
                self.quant_conv = ChannelDownsampleResidual(config.z_channels, config.codebook_embed_dim)

            self.post_quant_conv = ChannelUpsampleResidual(config.codebook_embed_dim, self.config.z_channels)
        else:
            self.quant_conv = nn.Conv2d(self.config.z_channels, config.codebook_embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, self.config.z_channels, 1)
        
        self.freeze_but_2d_decoder_flag = False

        def nan_hook(self, inp, output):
            if not isinstance(output, torch.Tensor):
                return
            if torch.isnan(output).any():
                print(f"NaN detected in {self}")
                raise RuntimeError("NaN detected")

        # for name, module in self.named_modules():
        #     module.register_forward_hook(nan_hook)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        wrap_modules = []
        # Add encoder layers
        for layer in self.s2to1encoder.transformer:
            wrap_modules.append(layer)
        # Add decoder layers
        for layer in self.s1to2decoder.transformer:
            wrap_modules.append(layer)

        return wrap_modules
        

    # def eval(self):
        # delete unused modules for inferencing
        # - semantic distillation mlp
        # - ar prior model

        # if self.config.encoder_2d_distill:
        #     del self.distill_mlp
        
        # if self.config.distill_depth is not None:
        #     del self.s1to2decoder.distill_mlp
        
        # if self.config.with_prior_model:
        #     del self.quantize.prior_model
        # super().eval()
    

    def freeze_but_2d_decoder(self):
        """deprecated"""
        for param in self.parameters():
            param.requires_grad = False

        set_requires_grad(True, self.decoder)
        self.freeze_but_2d_decoder_flag = True
    
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
               return_code=True, 
               return_feat=False, 
               return_cont_feat=False,      # return the feature before quantization
               return_fix_dim_feat=False,   # the fix dimension is the conv out before to codebook dim
               num_en_q_level=None, 
               causal_type=None,        # deprecated
               random_mix_reg=False,    # deprecated
               replace_ratio=None,      # deprecated
               global_step=None,        # prior model related, deprecated
               max_steps=None,          # prior model related, deprecated
               ):
        # causal_type = causal_type if causal_type is not None else self.config.causal_type
        if return_feat:
            assert (not return_code) and (not return_fix_dim_feat)
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level, 
                causal_type=causal_type, 
                return_feat=True)
            # return the feature of exactly the same width as the vit encoder
            return h, None, None
        
        if return_cont_feat:
            assert (not return_code) and (not return_fix_dim_feat)
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level,
                causal_type=causal_type,
                return_feat=False)
            # return the feature before quantization
            h = self.quant_conv(h)
            return h, None, None
        
        if return_fix_dim_feat:
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level,
                causal_type=causal_type,
                return_feat=False)
            # return the feature of the fix width (e.g. 256) before further downsampled to codebook dim
            return h, None, None


        s = self.encoder(x)
        h = self.s2to1encoder(s, num_q_level=num_en_q_level, causal_type=causal_type)
        # print("s shape:", s.shape)

        h = self.quant_conv(h)
        if self.training and self.config.with_prior_model:
            quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio,
                                                   global_step=global_step, max_steps=max_steps)
        else:
            quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio)

        if return_code:
            return quant, emb_loss, info
       
        return quant, emb_loss, s

    def decode(
            self, quant, 
            ret_inner_feat=False, # the feature passed through a MLP for alignment loss
            return_feat=False,    # the feature for linear probe
            ):
        quant = self.post_quant_conv(quant)
        if ret_inner_feat:
            rec_spatial, inner_feat = self.s1to2decoder(quant, ret_inner_feat=True)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial, inner_feat
        elif return_feat:
            # specifically for linear probe
            _, inner_feat = self.s1to2decoder(quant, return_feat=True)
            # pixel_dec = self.decoder(rec_spatial)
            return None, None, inner_feat
        else:
            rec_spatial = self.s1to2decoder(quant)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec, rec_spatial = self.decode(quant_b)
        return dec

    def forward(
            self, 
            input, 
            num_en_q_level=None, 
            causal_type=None, 
            rec_loss=True, 
            ret_inner_feat=False,
            random_mix_reg=False,
            replace_ratio=None,
            global_step=None,
            max_steps=None,
            ):
        quant, diff, spatial = self.encode(
                                    input, 
                                    return_code=False, 
                                    num_en_q_level=num_en_q_level, 
                                    causal_type=causal_type,
                                    random_mix_reg=random_mix_reg,
                                    replace_ratio=replace_ratio,
                                    global_step=global_step,
                                    max_steps=max_steps
                                    )
        if ret_inner_feat:
            if self.config.encoder_2d_distill:
                inner_feat = rearrange(spatial, 'b c h w -> b (h w) c')
                inner_feat = self.distill_mlp(inner_feat)
                dec, rec_spatial = self.decode(quant)
            else:
                dec, rec_spatial, inner_feat = self.decode(quant, ret_inner_feat=True)
        else:
            dec, rec_spatial = self.decode(quant)

        if self.training:
            if rec_loss:
                if self.config.fea_rec_loss_type == "cosine":
                    fea_rec_loss = self.config.fea_rec_loss_weight * compute_cosinesim_loss(spatial.detach(), rec_spatial, 1)
                elif self.config.fea_rec_loss_type == "mse":
                    fea_rec_loss = self.config.fea_rec_loss_weight * F.mse_loss(spatial.detach(), rec_spatial)
            else:
                fea_rec_loss = 0

        if self.training:
            if rec_loss:
                dir_dec = self.decoder(spatial)
            else:
                dir_dec = None
            
            if ret_inner_feat:
                return [dec, dir_dec], [diff, fea_rec_loss], inner_feat
            return [dec, dir_dec], [diff, fea_rec_loss]

        return dec, diff



@dataclass
class VQVitModel2DPlusArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    model_size: str = 'small'
    num_latent_tokens: int = 256
    encoder_size: str = None 
    decoder_size: str = None
    transformer_layer_type: str = "TransformerDecoderLayer"
    z_channels: int = 256
    dropout_p: float = 0.0

    adaptive_gn: bool = False
    d2s_up: bool = False

    rot: bool = False
    distill_depth: int = None

    encoder_2d_distill: bool = False

    # for semantic distillation regularization
    # the default 768 is for dino-v2 base
    out_inner_dim: int = 768

    fea_rec_loss_type: str = "cosine"
    fea_rec_loss_weight: float = 1.0
    use_attn: bool = True


class VQVitModel2DPlus(nn.Module):
    def __init__(self, config: VQVitModelPlusArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
                        ch_mult=config.encoder_ch_mult, 
                        z_channels=config.z_channels, 
                        dropout=config.dropout_p, 
                        use_attn=config.use_attn,
                        )

        if config.encoder_2d_distill:
            self.distill_mlp = nn.Sequential(
                    nn.Linear(config.z_channels, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.out_inner_dim),
                    )


        if config.encoder_size is not None:
            encoder_size = config.encoder_size
        else:
            encoder_size = config.model_size
        
        if config.decoder_size is not None:
            decoder_size = config.decoder_size
        else:
            decoder_size = config.model_size
        
        # when encoder size or decoder size is given, model size should be none
        if config.encoder_size is not None or config.decoder_size is not None:
            assert config.model_size is None
        
        if config.encoder_2d_distill:
            assert config.distill_depth is None



        self.s2dencoder = ViTEncoder2D(
            model_size=encoder_size, 
            token_size=config.z_channels, dropout=config.dropout_p, 
            patch_size=2**(len(config.encoder_ch_mult) - 1),
            transformer_layer_type=config.transformer_layer_type,
            )

        self.s2ddecoder = ViTDecoder2D(
            model_size=decoder_size,
            token_size=config.z_channels, dropout=config.dropout_p,
            patch_size=2**(len(config.decoder_ch_mult) - 1),
            out_inner_feat=config.distill_depth is not None,
            out_inner_depth=config.distill_depth,
            out_inner_dim=config.out_inner_dim,
            transformer_layer_type=config.transformer_layer_type,
            )

        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, 
            z_channels=config.z_channels, 
            dropout=config.dropout_p,
            adaptive_gn=config.adaptive_gn,
            d2s_up=config.d2s_up,
            use_attn=config.use_attn
            )

        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage,
                                        rot=config.rot
                                        )
        self.quant_conv = nn.Conv2d(self.config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, self.config.z_channels, 1)

        def nan_hook(self, inp, output):
            if not isinstance(output, torch.Tensor):
                return
            if torch.isnan(output).any():
                print(f"NaN detected in {self}")
                raise RuntimeError("NaN detected")

        for name, module in self.named_modules():
            module.register_forward_hook(nan_hook)
    
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
               return_code=True, 
               return_feat=False, 
               random_mix_reg=False,
               replace_ratio=0.1,
               **kwargs
               ):
        # causal_type = causal_type if causal_type is not None else self.config.causal_type
        if return_feat:
            s = self.encoder(x)
            h = self.s2dencoder(s, return_feat=True)
            # return the feature before quantization
            return h, None, None

        s = self.encoder(x)
        h = self.s2dencoder(s)
        # print("s shape:", s.shape)

        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio)

        if return_code:
            return quant, emb_loss, info
       
        return quant, emb_loss, s

    def decode(self, quant, ret_inner_feat=False, return_feat=False):
        quant = self.post_quant_conv(quant)
        if ret_inner_feat:
            rec_spatial, inner_feat = self.s2ddecoder(quant, ret_inner_feat=True)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial, inner_feat
        elif return_feat:
            # specifically for linear probe or visualization (don not go through mlp)
            _, feat = self.s2ddecoder(quant, return_feat=True)
            # pixel_dec = self.decoder(rec_spatial)
            return _, feat
        else:
            rec_spatial = self.s2ddecoder(quant)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec, rec_spatial = self.decode(quant_b)
        return dec

    def forward(
            self, 
            input, 
            num_en_q_level=None, 
            causal_type=None, 
            rec_loss=True, 
            ret_inner_feat=False,
            random_mix_reg=False,
            replace_ratio=None,
            global_step=None,
            max_steps=None,
            ):
        quant, diff, spatial = self.encode(
                                    input, 
                                    return_code=False, 
                                    random_mix_reg=random_mix_reg,
                                    replace_ratio=replace_ratio
                                    )
        if ret_inner_feat:
            if self.config.encoder_2d_distill:
                inner_feat = rearrange(spatial, 'b c h w -> b (h w) c')
                inner_feat = self.distill_mlp(inner_feat)
                dec, rec_spatial = self.decode(quant)
            else:
                dec, rec_spatial, inner_feat = self.decode(quant, ret_inner_feat=True)
        else:
            dec, rec_spatial = self.decode(quant)
        # if torch.isnan(dec).any():
        #     print("nan in dec")
        # if torch.isnan(rec_spatial).any():
        #     print("nan in rec_spatial")

        if self.training:
            if rec_loss:
                if self.config.fea_rec_loss_type == "cosine":
                    fea_rec_loss = self.config.fea_rec_loss_weight * compute_cosinesim_loss(spatial.detach(), rec_spatial, 1)
                elif self.config.fea_rec_loss_type == "mse":
                    fea_rec_loss = self.config.fea_rec_loss_weight * F.mse_loss(spatial.detach(), rec_spatial)
            else:
                fea_rec_loss = 0

        if self.training:
            if rec_loss:
                dir_dec = self.decoder(spatial)
            else:
                dir_dec = None
            
            if ret_inner_feat:
                return [dec, dir_dec], [diff, fea_rec_loss], inner_feat
            return [dec, dir_dec], [diff, fea_rec_loss]

        return dec, diff




class VectorQuantizer(nn.Module):
    def __init__(
            self, 
            n_e, 
            e_dim, 
            beta, 
            entropy_loss_ratio, 
            l2_norm, 
            show_usage, 
            rot=False,
            stochastic=False,
            stochastic_temperature=1.0,
            eval_deterministic=False,
            simvq=False,
            codebook_transform=None,
            freeze_codebook=False,
            ):
        """
        Args:
            n_e: the size of the codebook
            e_dim: the dimension of the codebook vectors
            beta: the commitment loss weight
            entropy_loss_ratio: the ratio of the entropy loss to the commitment loss
            l2_norm: whether to normalize the codebook vectors
            show_usage: whether to show the usage of the codebook vectors
            rot: whether to use rotation trick
            stochastic: whether to use stochastic quantization
            stochastic_temperature: the temperature of the stochastic quantization
            eval_deterministic: whether to use deterministic quantization in evaluation mode
            simvq: whether to use simvq https://arxiv.org/abs/2411.02038
            codebook_transform: the transform to apply to the codebook vectors,
                choices from [ None, "linear", "mlp"]
            freeze_codebook: whether to freeze the codebook vectors
        """

        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.rot = rot
        self.stochastic = stochastic
        self.eval_deterministic = eval_deterministic

        self.simvq = simvq
        self.codebook_transform = codebook_transform
        self.freeze_codebook = freeze_codebook


        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

        if self.stochastic:
            if stochastic_temperature > 0: # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else: # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))
        
        if self.simvq:
            if codebook_transform == "linear":
                codebook_transform = nn.Linear(self.e_dim, self.e_dim, bias=False)
            elif codebook_transform == "mlp":
                codebook_transform = nn.Sequential(
                    nn.Linear(self.e_dim, self.e_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.e_dim * 4, self.e_dim),
                )
            else:
                raise ValueError("codebook_transform: {} Not Acceptable".format(codebook_transform))
            self.codebook_transform = codebook_transform

            if self.freeze_codebook:
                self.embedding.weight.requires_grad = False

    def get_emb(self):
        if self.simvq:
            return self.codebook_transform(self.embedding.weight)
        else:
            return self.embedding.weight

    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        # from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
        torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    
    def forward(self, z, random_replace=False, replace_ratio=0.1):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        if self.stochastic:
            # sample the softmaxed cosine similarity
            # reference: LARP
            assert self.l2_norm, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, embedding)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                min_encoding_indices = torch.argmax(probs, dim=-1)

            else:
                min_encoding_indices = torch.multinomial(probs, 1)
                min_encoding_indices = min_encoding_indices.squeeze(-1)
        else:
            # look up by l2 distance, argmin
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

            min_encoding_indices = torch.argmin(d, dim=1)   # (b*h*w)

        z_q = embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            if self.entropy_loss_ratio > 0:
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            else:
                entropy_loss = 0

        b, h, w, c = z.shape
        if self.rot:
            # adapted from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
            b, h, w, c = z.shape
            z = z / torch.norm(z, dim=-1, keepdim=True)
            # assert self.l2_norm, "Rot requires l2 normalization"
            z = rearrange(z, 'b h w c-> (b h w) c')
            z_q= rearrange(z_q, 'b h w c -> (b h w) c')
            pre_norm_q = self.get_very_efficient_rotation(z / (torch.norm(z, dim=1, keepdim=True) + 1e-6),
                                                            z_q / (torch.norm(z_q, dim=1, keepdim=True) + 1e-6),
                                                            z.unsqueeze(1)).squeeze()
            z_q = pre_norm_q * (
                    torch.norm(z_q, dim=1, keepdim=True) / (torch.norm(z, dim=1, keepdim=True) + 1e-6)).detach()
            z_q = rearrange(z_q, '(b h w) c -> b h w c', b=b, h=h, w=w)
        else:
            # preserve gradients
            z_q = z + (z_q - z).detach()

        if random_replace and self.training:
            # randomly replace the quantized vectors with the continuous input
            z = rearrange(z, '(b h w) c -> b h w c', b=b, h=h, w=w)
            mask = torch.bernoulli(torch.full(z.shape[:-1], replace_ratio)).unsqueeze(-1).to(z.device)  # replace_ratio chance of replacement
            z_q = torch.where(mask.bool(), z, z_q)

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, [vq_loss, commit_loss, entropy_loss, codebook_usage], (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q



class VectorQuantizerWithPM(nn.Module):
    def __init__(
            self, 
            n_e, 
            e_dim, 
            beta, 
            entropy_loss_ratio, 
            l2_norm, 
            show_usage, 
            rot=False,
            stochastic=False,
            stochastic_temperature=1.0,
            eval_deterministic=False,
            simvq=False,
            codebook_transform=None,
            freeze_codebook=False,
            prior_model_config=None
            ):
        """
        Args:
            n_e: the size of the codebook
            e_dim: the dimension of the codebook vectors
            beta: the commitment loss weight
            entropy_loss_ratio: the ratio of the entropy loss to the commitment loss
            l2_norm: whether to normalize the codebook vectors
            show_usage: whether to show the usage of the codebook vectors
            rot: whether to use rotation trick
            stochastic: whether to use stochastic quantization
            stochastic_temperature: the temperature of the stochastic quantization
            eval_deterministic: whether to use deterministic quantization in evaluation mode
            simvq: whether to use simvq https://arxiv.org/abs/2411.02038
            codebook_transform: the transform to apply to the codebook vectors,
                choices from [ None, "linear", "mlp"]
            freeze_codebook: whether to freeze the codebook vectors
            prior_model_config: the config for the prior model

        - prior ar model for ntp regularization
            - returns the prior model loss along with other codebook loss
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.rot = rot
        self.stochastic = stochastic
        self.prior_model_config = prior_model_config
        self.eval_deterministic = eval_deterministic

        self.simvq = simvq
        self.codebook_transform = codebook_transform
        self.freeze_codebook = freeze_codebook

        # prior model training config 
        if prior_model_config is not None:
            self.prior_n_rounds = prior_model_config["train_args"]["n_rounds"]
            self.prior_no_grad_before_last_round = prior_model_config["train_args"]["no_grad_before_last_round"]
            self.prior_avg_loss_over_rounds = prior_model_config["train_args"]["avg_loss_over_rounds"]
            self.use_mix_ss = prior_model_config["train_args"]["use_mix_ss"]
            self.mix_ss_max_ratio = prior_model_config["train_args"]["mix_ss_max_ratio"]
            self.mix_ss_peak_steps_ratio = prior_model_config["train_args"]["mix_ss_peak_steps_ratio"]
            self.prior_latent_ce_temperature = prior_model_config["train_args"].get("latent_ce_temperature", 1.0)
 

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

        if self.stochastic:
            if stochastic_temperature > 0: # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else: # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        if prior_model_config is None:
            self.prior_model = None
        else:
            prior_model_additional_args = {
                'n_ind': self.e_dim, 
                'n_classes': self.n_e
            }

            self.ar_prior_loss_weight = prior_model_config["train_args"].get('prior_loss_weight', 0.06)
            if prior_model_config["train_args"].get('no_dropout', False):
                prior_model_additional_args['embd_pdrop'] = 0.0
                prior_model_additional_args['resid_pdrop'] = 0.0
                prior_model_additional_args['attn_pdrop'] = 0.0
                print(f"Warning: prior_loss is using no dropout")
            
            # initialize
            self.prior_model = GPTC_models[self.prior_model_config['name']](
                    **prior_model_config['init_args'], 
                    **prior_model_additional_args
                )

        if self.simvq:
            if codebook_transform == "linear":
                codebook_transform = nn.Linear(self.e_dim, self.e_dim, bias=False)
            elif codebook_transform == "mlp":
                codebook_transform = nn.Sequential(
                    nn.Linear(self.e_dim, self.e_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.e_dim * 4, self.e_dim),
                )
            else:
                raise ValueError("codebook_transform: {} Not Acceptable".format(codebook_transform))
            self.codebook_transform = codebook_transform

            if self.freeze_codebook:
                self.embedding.weight.requires_grad = False

        
    def get_emb(self):
        if self.simvq:
            return self.codebook_transform(self.embedding.weight)
        else:
            return self.embedding.weight


    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        # from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
        torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    def logits_to_token_embedding_with_ss(
            self, 
            logits, 
            ar_input_staring_from_idx_1, 
            global_step,
            max_steps,
            mask=None):
        """
        adapted from https://github.com/hywang66/LARP/
        """
        # logits: (b, n - 1, codebook_size), sequence index from 1 to n-1 (inclusive)
        # ar_input_staring_from_idx_1: (b, n - 1, d=16), requires_grad=True
        if mask is None:
            b, n_minus_1, _ = logits.size()
            if self.use_mix_ss:
                ss_ratio = (global_step / (max_steps * self.mix_ss_peak_steps_ratio )) * self.mix_ss_max_ratio
                ss_ratio = min(ss_ratio, self.mix_ss_max_ratio)
            else:
                ss_ratio = 1.0

            mask = torch.rand(b, n_minus_1, 1, device=logits.device) < ss_ratio
            mask = mask.expand(-1, -1, self.e_dim) # (b, n - 1, d=16)

        with torch.autocast(device_type='cuda', enabled=False):
            logits = logits.float()
            probs = F.softmax(logits, dim=-1) # (b, n - 1, codebook_size)
            indices = torch.multinomial(probs.view(-1, self.n_e), 1).view(*probs.size()[:-1]) # (b, n - 1)
        token_embedding = F.embedding(indices, self.get_emb()) # (b, n - 1, d=16)
        token_embedding = torch.where(mask, token_embedding, ar_input_staring_from_idx_1)

        return token_embedding

    def calculate_logits_and_ar_pred_cont(self, prior_model_output):
        ar_pred_cont = prior_model_output # (b, n, d=16)
        # the prior_model_output and the embedding should have been normalized (-1 dim)
        logits = F.linear(prior_model_output, self.get_emb())[:, 1:]
        logits = logits.mul_(1 / self.prior_latent_ce_temperature)
        logits = logits.contiguous() # (b, n - 1, codebook_size)
        return logits, ar_pred_cont


    def prior_ar_predict_n_rounds_ss(
            self, 
            ar_input, 
            global_step,
            max_steps,
        ):
        """
        adapted from https://github.com/hywang66/LARP/
        """
        prior_model = self.prior_model
        n_rounds = self.prior_n_rounds
        no_grad_before_last_round = self.prior_no_grad_before_last_round

        b, n, _ = ar_input.size()
        n_minus_1 = n - 1
        if self.use_mix_ss:
            peak_steps_ratio = torch.tensor(self.mix_ss_peak_steps_ratio, dtype=torch.float32)
            max_ratio = torch.tensor(self.mix_ss_max_ratio, dtype=torch.float32)

            ss_ratio = (global_step / (max_steps * peak_steps_ratio)) * max_ratio
            ss_ratio = torch.min(ss_ratio, max_ratio)
        else:
            ss_ratio = torch.tensor(1.0, dtype=torch.float32)

        mask_ss = torch.rand(b, n_minus_1, 1, device=ar_input.device) < ss_ratio
        mask_ss = mask_ss.expand(-1, -1, self.e_dim) # (b, n - 1, d=16)

        logits_all_rounds = []
        next_ar_input = ar_input # (b, n, d=16)
        for i in range(n_rounds):
            if no_grad_before_last_round and i < n_rounds - 1:
                # we can not use "with torch.no_grad()" here due to a pytorch's bug!
                # https://github.com/pytorch/pytorch/issues/112583
                prior_model.requires_grad_(False)
                prior_model_output = prior_model.ar_predict(next_ar_input.detach()) # (b, n - 1, codebook_size)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)
                prior_model.requires_grad_(True)
            else:
                prior_model_output = prior_model.ar_predict(next_ar_input) # (b, n, d=16)(1 orig + n - 1 pred)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)   # (b, n - 1, codebook_size)
                logits_all_rounds.append(logits)


            if i < n_rounds - 1:
                token_embedding = self.logits_to_token_embedding_with_ss(
                                            logits, 
                                            ar_input[:, 1:], 
                                            global_step=global_step,
                                            max_steps=max_steps,
                                            mask=mask_ss) # (b, n - 1, d=16)
                next_ar_input = torch.cat([ar_input[:, :1], token_embedding], dim=1) # (b, n, d=16)

        if self.prior_avg_loss_over_rounds:
            logits_all_rounds = torch.stack(logits_all_rounds, dim=0) # (n_rounds, b, n - 1, codebook_size)

        else:
            logits_all_rounds = torch.stack([logits_all_rounds[-1]], dim=0) # (1, b, n - 1, codebook_size)

        return logits_all_rounds, ar_pred_cont, next_ar_input # here the next_ar_input is actually the last round's ar_input

    def calculate_prior_loss_with_pred(
            self, 
            encode_output, 
            indices,
            global_step, 
            max_steps,
            return_sampled_indices=False,
            sample_temperature=1.0,
        ):
        """
        adapted from https://github.com/hywang66/LARP/
        """
        B = encode_output.size(0)
        ar_input = encode_output # (b, n, d) normalized
        labels = indices[:, 1:].contiguous() # (b, n - 1)
        logits_all_rounds, ar_pred_cont, regularized_z_ss = self.prior_ar_predict_n_rounds_ss(
                                                                    ar_input, 
                                                                    global_step=global_step, 
                                                                    max_steps=max_steps,
                                                                ) # regularized_z_ss: (b, n, d=16)
        labels_all_rounds = labels.unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1).contiguous() # (n_rounds or 1, b, n - 1)
        
        loss_latent_ce = F.cross_entropy(logits_all_rounds.view(-1, self.n_e), labels_all_rounds.view(-1))
        # return_dict['loss_latent_ce'] = loss_latent_ce
        # topk_accuracies = utils.calculate_topk_accuracy(logits_all_rounds[0], labels, topk=(1, 5), prepend='prior_')
        # return_dict.update(topk_accuracies)

        if return_sampled_indices:
            # sample the indices from the last round prediction because it is closer
            # to the downstream gpt prediction error pattern
            sampled_indices = torch.multinomial(F.softmax(logits_all_rounds[-1] / sample_temperature, dim=-1), 1).squeeze(-1)


        return loss_latent_ce 


    
    def forward(
            self, 
            z, 
            max_steps=None, # for pm training
            global_step=None,
            random_replace=False, 
            replace_ratio=0.1,
            ):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        b, h, w, c = z.shape
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()


        if self.stochastic:
            # sample the softmaxed cosine similarity
            # reference: LARP
            assert self.l2_norm, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, embedding)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                min_encoding_indices = torch.argmax(probs, dim=-1)
            else:
                min_encoding_indices = torch.multinomial(probs, 1)
                min_encoding_indices = min_encoding_indices.squeeze(-1)
        else:
            # look up by l2 distance, argmin
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

            min_encoding_indices = torch.argmin(d, dim=1)

        z_q = embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        ar_prior_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            if self.entropy_loss_ratio > 0:
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            else:
                entropy_loss = 0

        if self.rot:
            # adapted from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
            b, h, w, c = z.shape
            z = z / torch.norm(z, dim=-1, keepdim=True)
            # assert self.l2_norm, "Rot requires l2 normalization"
            z = rearrange(z, 'b h w c-> (b h w) c')
            z_q= rearrange(z_q, 'b h w c -> (b h w) c')
            pre_norm_q = self.get_very_efficient_rotation(z / (torch.norm(z, dim=1, keepdim=True) + 1e-6),
                                                            z_q / (torch.norm(z_q, dim=1, keepdim=True) + 1e-6),
                                                            z.unsqueeze(1)).squeeze()
            z_q = pre_norm_q * (
                    torch.norm(z_q, dim=1, keepdim=True) / (torch.norm(z, dim=1, keepdim=True) + 1e-6)).detach()
            z_q = rearrange(z_q, '(b h w) c -> b h w c', b=b, h=h, w=w)
        else:
            # preserve gradients
            z_q = z + (z_q - z).detach()
        
        if self.prior_model is not None and self.training:
            # ar prior training must be put after straight-through estimator, so that the gradients can be backpropagated
            assert global_step is not None and max_steps is not None, \
                "global_step and max_steps must be provided when using prior model"
            # when quantizing, there are only 2 dimensions, now change back to 
            # B N C for AR trianing
            min_indices = rearrange(min_encoding_indices, '(b n) -> b n', b=b)
            ar_prior_loss = self.ar_prior_loss_weight * self.calculate_prior_loss_with_pred(
                                rearrange(z_q, 'b h w c -> b (h w) c'),
                                indices=min_indices,
                                global_step=global_step, 
                                max_steps=max_steps
                                )
        else:
            ar_prior_loss = None


        if random_replace and self.training:
            # randomly replace the quantized vectors with the continuous input
            z = rearrange(z, '(b h w) c -> b h w c', b=b, h=h, w=w)
            mask = torch.bernoulli(torch.full(z.shape[:-1], replace_ratio)).unsqueeze(-1).to(z.device)  # replace_ratio chance of replacement
            z_q = torch.where(mask.bool(), z, z_q)
        

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, ar_prior_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q



def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """
    modified from llamagen and magvit
    Args:
        affinity: (b, n, n), the affinity matrix, where affinity[i, j] is the affinity 
                between encoed vector i and codebook vector j
        loss_type: how to turn the affinity into probability distribution
    """
    # shape: (b n) n
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    # target_probs.shape: (b, n, n), and sum(target_probs, dim=-1) = 1
    avg_probs = torch.mean(target_probs, dim=0) # (,n)
    # average entropy corresponeds (negatively) to the diversity of indices for a single position
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    # sample entropy is the confidence for the quantization process
    # (bn, n) -> (bn) -> avg 
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

def compute_cosinesim_loss(feat1, feat2, dim):
    cos_sim = F.cosine_similarity(feat1, feat2, dim=dim)
    loss = 1 - cos_sim
    return torch.mean(loss)  




