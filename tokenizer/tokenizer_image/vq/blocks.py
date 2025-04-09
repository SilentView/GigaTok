"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
"""
from typing import Optional, List
import os
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict

from utils.rope import precompute_freqs_cis_2d, apply_rotary_emb, precompute_freqs_cis

# check the version of pytorch, 
# if pytorch version >= 2.2.0, then flash_attention can be used 
if torch.__version__ >= "2.2.0":
    HAS_FLASH_ATTENTION_V2 = True
    # print("flash_attention v2 can be used.")
else:
    HAS_FLASH_ATTENTION_V2 = False
    # print("flash_attention v2 is not supported.")


############################################################################
## Util Functions
############################################################################


def sample_multi_level_1d_tokens(x, n_level, first_grow="row"):
    """
    Args:
        x: [B, C, H, W]
        n_level: token_length = 2^n_level = 1 + 2^0 + ... + 2^(n_level - 1)
            corresponding to token_group_idx: 0, 1, ..., n_level (summed to be n_level + 1 groups)
    Returns:
        x_1d: [B, 2^n_level, C]
    note: log_2(W) >= n_level // 2 if first_grow == 'row'
    on the condition of: 
        if first_grow == 'row':
            W % 2^(n_level//2) == 0 and H % 2^((n_level-1)//2) == 0
        elif first_grow == 'col':
            H % 2^(n_level//2) == 0 and W % 2^((n_level-1)//2) == 0
    """

    def _get_kernel_size(token_group_idx, max_h, max_w, first_grow="row"):
        # level_idx should start from 0
        level_idx = max(token_group_idx - 1, 0)
        if first_grow == "row":
            # first split in the row dimension (first increase w_block_num)
            w_block_num = 2 ** ((level_idx + 1) // 2)
            h_block_num = 2 ** (level_idx // 2)
            w_block_size = max_w // w_block_num
            h_block_size = max_h // h_block_num
        elif first_grow == "col":
            # first split in the col dimension (first increase h_block_num)
            h_block_num = 2 ** ((level_idx + 1) // 2)
            w_block_num = 2 ** (level_idx // 2)
            h_block_size = max_h // h_block_num
            w_block_size = max_w // w_block_num
        else:
            raise ValueError(f"Invalid first_grow: {first_grow}, choose from 'row' or 'col'")
        assert w_block_size > 0 and h_block_size > 0, f"Invalid level idx: {level_idx}, max_h: {max_h}, max_w: {max_w}"
        return (h_block_size, w_block_size)

    x_groups = []
    for i in range(n_level + 1):
        kernel_size = _get_kernel_size(i, x.shape[2], x.shape[3], first_grow=first_grow)
        cur_group_x = F.avg_pool2d(x, kernel_size=kernel_size)
        cur_group_x = rearrange(cur_group_x, "b c h w -> b (h w) c")
        x_groups.append(cur_group_x)

    x_1d = torch.cat(x_groups, dim=1)
    return x_1d


def last_level_1d_features_to_2d_maps(x_1d, n_level, max_h, max_w, first_grow="row"):
    """
    Args:
        x_1d: [B, 2^n_level, C]
        n_level: token_length = 2^n_level = 1 + 2^0 + ... + 2^(n_level - 1)
            corresponding to token_group_idx: 0, 1, ..., n_level (summed to be n_level + 1 groups)
    Returns:
        x_dilated: [B, C, max_h, max_w]
    """
    assert x_1d.shape[1] == 2 ** n_level, f"Invalid x_1d shape: {x_1d.shape}"
    if n_level == 1:
        last_level_features = x_1d[:, 0:2, :].mean(dim=1, keepdim=True)
    else:
        last_level_features = x_1d[:, -2**(n_level - 1):, :]
    if first_grow == "row":
        # first split in the row dimension (first increase w_block_num)
        w_block_num = 2 ** (n_level // 2)
        h_block_num = 2 ** ((n_level - 1)// 2)
        w_block_size = max_w // w_block_num
        h_block_size = max_h // h_block_num
    elif first_grow == "col":
        # first split in the col dimension (first increase h_block_num)
        h_block_num = 2 ** (n_level // 2)
        w_block_num = 2 ** ((n_level - 1)// 2)
        h_block_size = max_h // h_block_num
        w_block_size = max_w // w_block_num
    
    last_level_features = rearrange(last_level_features, "b (h w) c -> b c h w", h=h_block_num, w=w_block_num)
    # dilate the last level features by w_block_size and h_block_size in a differentiable way
    x_dilated = F.interpolate(last_level_features, size=(max_h, max_w), mode='nearest')

    return x_dilated


def multi_level_1d_features_to_2d_maps_avg(x_1d, n_level, max_h, max_w, first_grow="row"):
    """
    Args:
        x_1d: [B, 2^n_level, C]
        n_level: token_length = 2^n_level = 1 + 2^0 + ... + 2^(n_level - 1)
            corresponding to token_group_idx: 0, 1, ..., n_level (summed to be n_level + 1 groups)
    Returns:
        x_dilated: [B, C, max_h, max_w]
    """
    assert x_1d.shape[1] == 2 ** n_level, f"Invalid x_1d shape: {x_1d.shape}"
    # print("this multi_level_1d_features_to_2d_maps_avg function is called ")

    for level_idx in range(n_level):
        start_idx = 0
        end_idx = 0
        if level_idx == 0:
            start_idx = 0
            end_idx = 2
            cur_level_features = x_1d[:, start_idx:end_idx, :].mean(dim=1, keepdim=True)
        else:
            start_idx = end_idx
            end_idx = start_idx + 2**level_idx
            # 1 + 2^(level_idx - 1) : 2 + 2^(level)
            cur_level_features = x_1d[:, start_idx:end_idx, :]

        if first_grow == "row":
            w_block_num = 2 ** ((level_idx + 1) // 2)
            h_block_num = 2 ** (level_idx // 2)

        if first_grow == "col":
            h_block_num = 2 ** ((level_idx + 1) // 2)
            w_block_num = 2 ** (level_idx // 2)


        cur_level_features = rearrange(cur_level_features, "b (h w) c -> b c h w", h=h_block_num, w=w_block_num)
        x_dilated = F.interpolate(cur_level_features, size=(max_h, max_w), mode='nearest')
        if level_idx == 0:
            x_result = x_dilated
        else:
            x_result += x_dilated
    
    x_result = x_result / (n_level)

    return x_result



def get_attn_mask(x_1d, causal_type="per-token"):
    """
    Generates an attention mask based on the given causal type.
    
    Args:
        x_1d: Tensor of shape [2^n_level, B, C], where 2^n_level represents the number of tokens.
        causal_type: A string indicating the type of causal mask to generate. 
                     Options are "per-token" or "per-level".
    
    Returns:
        A binary attention mask of shape [2^n_level, 2^n_level] with dtype torch.bool.
    """
    
    seq_len = x_1d.shape[0]  # This is 2^n_level
    n_level = int(np.round(np.log2(seq_len)))  # Calculate n_level from the sequence length

    if causal_type == "per-token":
        # Generate a token-wise (lower triangular) causal mask
        attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    
    elif causal_type == "per-level":
        # Generate a level-wise causal mask
        attn_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        
        # Define the starting index for each level
        for level_idx in range(n_level + 1):
            level_start =  0 if level_idx == 0 else 2 ** level_idx
            level_end = 2 ** (level_idx + 1)
            
            # Allow attention within the same level and all previous levels
            attn_mask[level_start:level_end, :level_end] = False
    
    else:
        raise ValueError(f"Unknown causal_type: {causal_type}. Choose 'per-token' or 'per-level'.")
    
    return attn_mask.to(x_1d.device)


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size,
                            w * block_size)

    return x



############################################################################
## Layer Component Module
############################################################################

def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class D2SUpsampler(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out
 


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        # self.lin = nn.Linear(z_channels, in_filters * 2)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps
    
    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        # quantizer = F.adaptive_avg_pool2d(quantizer, (1, 1))
        ### calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps #not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)
       
        x = self.gn(x)
        x = scale * x + bias

        return x


class AttentionCustom(nn.Module):
    def __init__(
            self, 
            dim,
            n_head,
            resid_dropout_p,
            use_rope=False,
            use_qk_norm=False,
            use_flash_attn=False,
            no_bias=False,
            attn_dropout_p=0
        ):
        """
        This custom attention block supports the following modifications:
        - ROPE
        - QK Norm
        - Flash attention
        Currently, the dimension of the key and value is the same as the query.
        """
        super().__init__()

        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        if self.use_flash_attn:
            print("Using flash attention!")
        # flash attention can be switched to normal attention for inference
        # rasie error only when training and use_flash_attn is True and 
        # flash attention is not installed
        assert HAS_FLASH_ATTENTION_V2 or (not self.use_flash_attn) or (not self.training), \
            "Flash attention is not installed and cannot be used when training"
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_qkv_dim =  3 * self.n_head * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.k_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.v_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.wo = nn.Linear(dim, dim, bias=not no_bias)

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor,
        value: torch.Tensor,
        freqs_cis: torch.Tensor = None, 
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        """
        The q, k, v will be projected into multiple heads.
        """
        seqlen, bsz, _ = query.shape

        # rearrange, (L, B, D) -> (B, L, D)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        xq = self.q_proj(query)
        xk = self.k_proj(key)
        xv = self.v_proj(value)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)
    
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.use_rope:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        else:
            assert freqs_cis is None, "Attention Module is not using ROPE but freqs_cis is not None. Check your setting!"
        

        # (B, L, H, D) -> (B, H, L, D)
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.use_flash_attn:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, 
                    attn_mask=attn_mask, 
                    is_causal=is_causal, # is_causal=False is for KV cache
                    dropout_p=self.attn_dropout_p if self.training else 0)
        else:
            output = F.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=attn_mask, 
                is_causal=is_causal, # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0)
            
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))

        # rearrange, (B, L, D) -> (L, B, D)
        return output.transpose(0, 1)


class SelfAttentionCustom(nn.Module):
    def __init__(
            self, 
            dim,
            n_head,
            resid_dropout_p,
            use_rope=False,
            use_qk_norm=False,
            no_bias=False,
            use_flash_attn=False,
            attn_dropout_p=0
        ):
        """
        This custom attention block supports the following modifications:
        - ROPE
        - QK Norm
        - Flash attention
        """
        super().__init__()

        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        # flash attention can be switched to normal attention for inference
        # rasie error only when training and use_flash_attn is True and 
        # flash attention is not installed
        assert HAS_FLASH_ATTENTION_V2 or (not use_flash_attn) or (not self.training), \
            "Flash attention is not installed and cannot be used when training"
        assert dim % n_head == 0

        if self.use_flash_attn:
            print("Using flash attention!")

        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_qkv_dim =  3 * self.n_head * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_qkv_dim, bias=not no_bias)
        self.wo = nn.Linear(dim, dim, bias=not no_bias)

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor = None, 
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        seqlen, bsz, _ = x.shape

        # rearrange, (L, B, D) -> (B, L, D)
        x = x.transpose(0, 1)

        xq, xk, xv = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)
    
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.use_rope:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        else:
            assert freqs_cis is None, "Attention Module is not using ROPE but freqs_cis is not None. Check your setting!"
        
        # (B, L, H, D) -> (B, H, L, D)
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.use_flash_attn:
            # Shape: (batch_size, num_heads, seq_length, head_dim)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, 
                    attn_mask=mask, 
                    is_causal=is_causal, # is_causal=False is for KV cache
                    dropout_p=self.attn_dropout_p if self.training else 0)            
        else:
            output = F.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=mask, 
                is_causal=is_causal, # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0)            
            
        # (B, H, L, D) -> (B, L, H*D)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))

        # rearrange back, (B, L, D) -> (L, B, D)
        return output.transpose(0, 1)





############################################################################
## Layer Module
############################################################################

class TransformerDecoderLayer(nn.Module):
    """
    This is the Q-former layer from DETR.
    """
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1,
                 activation=nn.GELU, normalize_before=True, query_rope=False,
                 use_qk_norm=False, use_flash_attn=False
                 ):
        super().__init__()
        self.query_rope = query_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        if self.query_rope:
            self.self_attn = SelfAttentionCustom(
                                d_model, 
                                nhead, 
                                resid_dropout_p=dropout, 
                                use_rope=True,
                            )
            self.multihead_attn = AttentionCustom(
                                d_model,
                                nhead,
                                resid_dropout_p=dropout,
                                use_rope=True,
                            )
        elif self.use_qk_norm or self.use_flash_attn:
            self.self_attn = AttentionCustom(
                                d_model, 
                                nhead, 
                                resid_dropout_p=dropout, 
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
            self.multihead_attn = AttentionCustom(
                                d_model,
                                nhead,
                                resid_dropout_p=dropout,
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        dim_feedforward = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before



    def with_pos_embed(self, tensor, pos: Optional[Tensor], rope=False):
        if rope:
            return apply_rotary_emb(tensor, pos, bs_first=False)
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        
        if self.query_rope:
            tgt2 = self.self_attn(
                tgt2,
                freqs_cis=query_pos,
                mask=tgt_mask,
            )
        elif self.use_qk_norm or self.use_flash_attn:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.use_qk_norm or self.use_flash_attn:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos, rope=self.query_rope),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos, rope=self.query_rope),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        if self.query_rope:
            tgt2 = self.self_attn(
                tgt2,
                query_pos,
                tgt_mask
            )
        elif self.use_qk_norm or self.use_flash_attn:
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        if torch.isnan(tgt2).any():
            print("tgt2 is nan")
            print("q shape and values:", q.shape, q.min().item(), q.max().item())
            print("k shape and values:", k.shape, k.min().item(), k.max().item())
            print("tgt shape and values:", tgt.shape, tgt.min().item(), tgt.max().item())
            tmp = self.norm1(tgt)
            print("after norm tgt shape and values:", tmp.shape, tmp.min().item(), tmp.max().item())

            print("q is nan:", torch.isnan(q).any())
            print("k is nan:", torch.isnan(k).any())
            print("tgt is nan:", torch.isnan(tgt).any())

            # check any nans in the weights
            for name, param in self.self_attn.named_parameters():
                if torch.isnan(param).any():
                    print(f"{name} has nan values")
            torch.set_printoptions(threshold=10_000)
            print(q)
            print(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        if self.use_qk_norm or self.use_flash_attn:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos, rope=self.query_rope),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos, rope=self.query_rope),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
 
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerLayer(nn.Module):
    """
    This is the standard Transformer layer. Currently only supports absolute positional embedding.
    # TODO: support RoPE 2d
    """
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1,
                 activation=nn.GELU, normalize_before=True, use_rope=False,
                 use_qk_norm=False,
                 ):
        super().__init__()
        self.use_rope = use_rope
        assert not self.use_rope, "QformerLayerVar does not support RoPE yet"
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.self_attn = AttentionCustom(
                                d_model, 
                                nhead, 
                                resid_dropout_p=dropout, 
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        dim_feedforward = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before



    def with_pos_embed(self, tensor, pos: Optional[Tensor], rope=False):
        if rope:
            return apply_rotary_emb(tensor, pos, bs_first=False)
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
    ):
        if self.use_qk_norm:
            q = k = tgt
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = tgt
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        if self.use_qk_norm:
            tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask)
        else:
            tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        assert pos is None, "TransformerLayer does not support injected positional embedding"
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, pos)

class QformerLayerVar(nn.Module):
    """
    This is the variant of Q-former layer from DETR. It is merely used for comparison.
    This layer has 2 multi-head attention layers, and the positional embedding follows the original DETR.
    But there is not reference feature map any more. All attentions are self-attention.
    """
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1,
                 activation=nn.GELU, normalize_before=True, use_rope=False,
                 use_qk_norm=False,
                 ):
        super().__init__()
        self.use_rope = use_rope
        assert not self.use_rope, "QformerLayerVar does not support RoPE yet"
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.self_attn = AttentionCustom(
                                d_model, 
                                nhead, 
                                resid_dropout_p=dropout, 
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
            self.multihead_attn = AttentionCustom(
                                d_model,
                                nhead,
                                resid_dropout_p=dropout,
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        dim_feedforward = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before



    def with_pos_embed(self, tensor, pos: Optional[Tensor], rope=False):
        if rope:
            return apply_rotary_emb(tensor, pos, bs_first=False)
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
    ):
        
        if self.use_qk_norm:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.use_qk_norm :
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos, rope=self.use_rope),
                                    key=self.with_pos_embed(tgt, pos),
                                    value=tgt, attn_mask=tgt_mask)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos, rope=self.use_rope),
                                    key=self.with_pos_embed(tgt, pos),
                                    value=tgt, attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        if self.use_qk_norm:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt2, pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        if self.use_qk_norm:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos, rope=self.use_rope),
                                    key=self.with_pos_embed(tgt2, pos),
                                    value=tgt2, attn_mask=tgt_mask)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos, rope=self.use_rope),
                                    key=self.with_pos_embed(tgt2, pos),
                                    value=tgt2, attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)[0]
 
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, pos)




class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

############################################################################
## Part Module
############################################################################

class ViTEncoder(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            num_latent_tokens=32, 
            token_size=256, 
            dropout=0.0,
            multi_level_query_init=False,
            learnable_1d_query_init=False,
            rope_1d=False,
            downsample_improve=False,
            use_qk_norm=False,
            use_flash_attn=False
            ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn

        # TODO: if verify the downsample_improve, then hard code it as a must
        self.downsample_improve = downsample_improve

        self.multi_level_query_init = multi_level_query_init
        self.learnable_1d_query_init = learnable_1d_query_init

        self.rope_1d = rope_1d

        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
                "xl": 1280,
                "xxl": 1536,
                "xxxl": 2560
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 6,
                "base": 12,
                "large": 24,
                "xl": 36,
                "xxl": 48,
                "xxxl": 48
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
                "xl": 20,
                "xxl": 24,
                "xxxl": 40
            }[self.model_size]
        
        self.encoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        assert not (self.multi_level_query_init and self.learnable_1d_query_init)

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, 1, self.width))
        
        if self.rope_1d:
            self.freqs_cis_1d = precompute_freqs_cis(
                                    self.grid_size ** 2, 
                                    self.width // self.num_heads, 
                                    base=10000, 
                                    cls_token_num=0)
        else:
            self.latent_token_positional_embedding = nn.Parameter(
                scale * torch.randn(self.num_latent_tokens, 1, self.width))

        if self.learnable_1d_query_init:
            self.query_1d = nn.Parameter(
                scale * torch.randn(self.num_latent_tokens, 1, self.width)
            )
        else:
            self.global_proj = nn.Linear(self.width, self.width, bias=True)

        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(TransformerDecoderLayer(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
                query_rope=self.rope_1d, use_qk_norm=self.use_qk_norm, use_flash_attn=self.use_flash_attn,
            ))
        
        if downsample_improve:
            self.ln_post = nn.Identity()
            self.conv_out = nn.Identity()
        else:
            self.ln_post = nn.LayerNorm(self.width)
            self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, x, num_q_level=None, causal_type=None, return_feat=False):
        bs = x.shape[0]
        if self.multi_level_query_init:
            x = x.reshape(x.shape[0], x.shape[1], -1)   # B, C, H, W -> B, C, H*W
            x = x.permute(2, 0, 1) # shape = [grid ** 2, B, width]
            x = self.encoder_embed(self.ln_pre(x))
            n_levels = round(np.log2(self.num_latent_tokens))
            latent_tokens = sample_multi_level_1d_tokens(
                                rearrange(x, '(h w) b c -> b c h w', h=self.grid_size), 
                                n_levels, 
                                first_grow="row")
            # if os.environ.get("LOCAL_RANK", 0) == 0:
            #     print(latent_tokens.shape)
            #     print(x.shape)
            latent_tokens = latent_tokens.permute(1, 0, 2)  # num_latent_tokens, B, width
            latent_tokens = self.global_proj(latent_tokens)
        elif self.learnable_1d_query_init:
            x = x.reshape(x.shape[0], x.shape[1], -1)   # B, C, H, W -> B, C, H*W
            x = x.permute(2, 0, 1) # shape = [grid ** 2, B, width]
            x = self.encoder_embed(self.ln_pre(x))
            # n_levels = round(np.log2(self.num_latent_tokens))
            latent_tokens = self.query_1d.repeat(1, bs, 1).to(x.dtype)
        else:
            x = x.reshape(x.shape[0], x.shape[1], -1)   # B, C, H, W -> B, C, H*W
            x = x.permute(2, 0, 1) # shape = [grid ** 2, B, width]
            x = self.encoder_embed(self.ln_pre(x))
            # class embeddings and positional embeddings

            # shape: (num_tokens, B, c), 
            # note: using nn.MultiheadAttention with default batch_first=True, means the input should also be 
            #       sequence length first
            latent_tokens = self.global_proj(x.mean(dim=0, keepdim=True)).repeat(self.num_latent_tokens, 1, 1)
        

        # select a certain number of latent_tokens
        if num_q_level is None:
            selected_token_num = self.num_latent_tokens
        else:
            if causal_type == "per-level":
                selected_token_num = 2**num_q_level
            elif causal_type == "per-token":
                selected_token_num = num_q_level
            else:
                selected_token_num = num_q_level
                # raise ValueError(f"Unknown causal_type: {causal_type}. Choose 'per-token' or 'per-level'.")

        latent_tokens = latent_tokens[:selected_token_num]

        pos_embed = self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]

        if self.rope_1d:
            query_pos = self.freqs_cis_1d.to(x.dtype)[:selected_token_num].to(x.device)
        else:
            query_pos = self.latent_token_positional_embedding.to(x.dtype)[:selected_token_num]

        # get the attn_mask
        if causal_type is None:
            attn_mask = None
        elif causal_type == "per-level":
            attn_mask = get_attn_mask(latent_tokens, causal_type=causal_type)
        elif causal_type == "per-token":
            attn_mask = get_attn_mask(latent_tokens, causal_type=causal_type)
        else:
            attn_mask = None


        # x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            latent_tokens = self.transformer[i](latent_tokens, x, pos=pos_embed, query_pos=query_pos, tgt_mask=attn_mask)
        
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, 1, selected_token_num)  # LND -> NDL
        if return_feat:
            return latent_tokens[:, :, :, :selected_token_num]
        latent_tokens = self.conv_out(latent_tokens)

        return latent_tokens


class ViTEncoder2D(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            token_size=256, 
            dropout=0.0,
            transformer_layer_type="QformerLayerVar"
            ):
        super().__init__()
        self.transformer_layer_type = transformer_layer_type
        assert transformer_layer_type in \
            ["QformerLayerVar", "TransformerLayer", "TransformerDecoderLayer"]
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.token_size = token_size

        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
                "xl": 1280,
                "xxl": 1536,
                "xxxl": 2560
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 6,
                "base": 12,
                "large": 24,
                "xl": 36,
                "xxl": 48,
                "xxxl": 48
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
                "xl": 20,
                "xxl": 24,
                "xxxl": 40
            }[self.model_size]
        
        self.encoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, 1, self.width))

        self.transformer = nn.ModuleList()

        layer_cls = eval(self.transformer_layer_type)
        for i in range(self.num_layers):
            self.transformer.append(layer_cls(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, x, return_feat=False):
        bs = x.shape[0]

        x = x.reshape(x.shape[0], x.shape[1], -1)   # B, C, H, W -> B, C, H*W
        x = x.permute(2, 0, 1) # shape = [grid ** 2, B, width] or LND
        x = self.encoder_embed(self.ln_pre(x))

        # shape: (num_tokens, B, c), 
        # note: using nn.MultiheadAttention with default batch_first=True, means the input should also be 
        #       sequence length first
        pos_embed = self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]

        latent_tokens = x
        if self.transformer_layer_type == "TransformerLayer":
            # this is the original transformer layer, use absolute position embedding
            latent_tokens = latent_tokens + pos_embed
        for i in range(self.num_layers):
            if self.transformer_layer_type == "TransformerDecoderLayer":
                latent_tokens = self.transformer[i](
                                    latent_tokens, 
                                    latent_tokens, 
                                    pos=pos_embed, 
                                    query_pos=pos_embed, 
                                    tgt_mask=None
                                    )
            elif self.transformer_layer_type == "TransformerLayer":
                latent_tokens = self.transformer[i](
                                    latent_tokens
                                    )
            elif self.transformer_layer_type == "QformerLayerVar":
                latent_tokens = self.transformer[i](
                                    latent_tokens, 
                                    pos=pos_embed
                                    )
            # latent_tokens = self.transformer[i](x, pos=pos_embed)
        
        latent_tokens = self.ln_post(latent_tokens)
        # 2D shape; L N D -> N D H W
        latent_tokens = rearrange(latent_tokens, '(h w) b c -> b c h w', h=self.grid_size, w=self.grid_size)
        if return_feat:
            return latent_tokens
        latent_tokens = self.conv_out(latent_tokens)

        return latent_tokens


    

class ViTDecoder_V2(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            num_latent_tokens=32, 
            token_size=256, 
            dropout=0.0,
            last_level_2d_query_init=False,
            multi_level_2d_query_init=False,
            learnable_2d_query_init=False,
            rope_2d=True
            ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.last_level_2d_query_init = last_level_2d_query_init
        self.multi_level_2d_query_init = multi_level_2d_query_init
        self.learnable_2d_query_init = learnable_2d_query_init
        # rope_2d will be effective only for 2d query embedding
        self.rope_2d = rope_2d
        assert not (self.last_level_2d_query_init and self.multi_level_2d_query_init)
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 6,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        if self.learnable_2d_query_init:
            self.q_2d = nn.Parameter(scale * torch.randn(self.grid_size ** 2, 1, self.width))
        
        if self.rope_2d:
            self.freqs_cis = precompute_freqs_cis_2d(self.grid_size, self.width // self.num_heads, base=10000, cls_token_num=0)
            # self.freqs_cis = precompute_freqs_cis_2d(self.grid_size, self.width, cls_token_num=0)
        else:
            self.positional_embedding = nn.Parameter(
                    scale * torch.randn(self.grid_size ** 2, 1, self.width))
        # add mask token and query pos embed
        # self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, 1, self.width))
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(TransformerDecoderLayer(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
                query_rope=self.rope_2d,
            ))
        self.ln_post = nn.LayerNorm(self.width)

        self.conv_out = nn.Conv2d(self.width, token_size, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        # assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        selected_latent_tokens = W
        x = z_quantized.reshape(N, C*H, W).permute(2, 0, 1) # LND
        x = self.decoder_embed(self.ln_pre(x))

        seq_len, bs, _ = x.shape    # shape: (num_latent_tokens, B, c)

        if self.last_level_2d_query_init:
            n_level = round(np.log2(selected_latent_tokens))
            # (B, C, grid_size, grid_size)
            latent_tokens = last_level_1d_features_to_2d_maps(
                x_1d=rearrange(x, 'l b c -> b l c', b=bs),
                n_level=n_level,
                max_h=self.grid_size,
                max_w=self.grid_size,
                first_grow="row",
            )
            latent_tokens = rearrange(latent_tokens, 'b c h w -> (h w) b c').to(x.dtype)
            # (grid_size**2, B, c)
        elif self.multi_level_2d_query_init:
            n_level = round(np.log2(selected_latent_tokens))
            # (B, C, grid_size, grid_size)
            latent_tokens = multi_level_1d_features_to_2d_maps_avg(
                x_1d=rearrange(x, 'l b c -> b l c', b=bs),
                n_level=n_level,
                max_h=self.grid_size,
                max_w=self.grid_size,
                first_grow="row",
            )
            latent_tokens = rearrange(latent_tokens, 'b c h w -> (h w) b c').to(x.dtype)
        elif self.learnable_2d_query_init:
            latent_tokens = self.q_2d.repeat(1, bs, 1).to(x.dtype)
        else:
            latent_tokens = x[:1, :, :].repeat(self.grid_size**2, 1, 1).to(x.dtype)

        # mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        # x = x + self.latent_token_positional_embedding[:seq_len]
        # x = torch.cat([mask_tokens, x], dim=1)

        if self.rope_2d:
            query_pos = self.freqs_cis.to(x.dtype).to(x.device)
        else:
            query_pos = self.positional_embedding.repeat(1, bs, 1).to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        pos_embed = self.latent_token_positional_embedding[:selected_latent_tokens].repeat(1, bs, 1).to(x.dtype)

        for i in range(self.num_layers):
            latent_tokens = self.transformer[i](latent_tokens, x, pos=pos_embed, query_pos=query_pos)

        latent_tokens = self.ln_post(latent_tokens)
        # L N D -> N D H W
        latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, self.grid_size, self.grid_size)
        latent_tokens = self.conv_out(latent_tokens.contiguous())
        return latent_tokens


class ViTDecoder(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            num_latent_tokens=32, 
            token_size=256, 
            dropout=0.0,
            last_level_2d_query_init=False,
            multi_level_2d_query_init=False,
            learnable_2d_query_init=False,
            q_upsample=False,
            out_inner_feat=False,
            out_inner_dim=768,   # for dino-v2
            out_inner_depth=None,
            use_qk_norm=False,
            use_flash_attn=False,
            ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.last_level_2d_query_init = last_level_2d_query_init
        self.multi_level_2d_query_init = multi_level_2d_query_init
        self.learnable_2d_query_init = learnable_2d_query_init
        self.out_inner_feat = out_inner_feat
        self.out_inner_depth = out_inner_depth

        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn

        assert not (self.last_level_2d_query_init and self.multi_level_2d_query_init)
        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
                "xl": 1280,
                "xxl": 1536,
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 6,
                "base": 12,
                "large": 24,
                "xl": 36,
                "xxl": 48,
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
                "xl": 20,
                "xxl": 24,
            }[self.model_size]
 

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        if self.learnable_2d_query_init:
            self.q_2d = nn.Parameter(scale * torch.randn(self.grid_size ** 2, 1, self.width))

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, 1, self.width))
        # add mask token and query pos embed
        # self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, 1, self.width))
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(TransformerDecoderLayer(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
                use_qk_norm=self.use_qk_norm, use_flash_attn=self.use_flash_attn
            ))
        self.ln_post = nn.LayerNorm(self.width)

        self.conv_out = nn.Conv2d(self.width, token_size, kernel_size=3, stride=1, padding=1)

        self.q_upsample = q_upsample
        if self.q_upsample:
            self.upsampler = QFormerUpsample(
                                width=self.width, 
                                nhead=self.num_heads, 
                                dropout=dropout, 
                                num_queries=num_latent_tokens
                            )
        
        if out_inner_feat:
            self.distill_mlp = nn.Sequential(
                    nn.Linear(self.width, self.width * 4),
                    nn.SiLU(),
                    nn.Linear(self.width * 4, self.width * 4),
                    nn.SiLU(),
                    nn.Linear(self.width * 4, out_inner_dim),
                    )

    
    def forward(
            self, 
            z_quantized, 
            ret_inner_feat=False,   # return inner feature(through mlp) for distillation
            return_feat=False,      # return feature for linear probe
            ):
        N, C, H, W = z_quantized.shape
        # assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        selected_latent_tokens = W
        x = z_quantized.reshape(N, C*H, W).permute(2, 0, 1) # LND
        x = self.decoder_embed(self.ln_pre(x))
        # upsample
        if self.q_upsample:
            x = self.upsampler(x)
            selected_latent_tokens = self.num_latent_tokens

        seq_len, bs, _ = x.shape    # shape: (num_latent_tokens, B, c)

        if self.last_level_2d_query_init:
            n_level = round(np.log2(selected_latent_tokens))
            # (B, C, grid_size, grid_size)
            latent_tokens = last_level_1d_features_to_2d_maps(
                x_1d=rearrange(x, 'l b c -> b l c', b=bs),
                n_level=n_level,
                max_h=self.grid_size,
                max_w=self.grid_size,
                first_grow="row",
            )
            latent_tokens = rearrange(latent_tokens, 'b c h w -> (h w) b c').to(x.dtype)
            # (grid_size**2, B, c)
        elif self.multi_level_2d_query_init:
            n_level = round(np.log2(selected_latent_tokens))
            # (B, C, grid_size, grid_size)
            latent_tokens = multi_level_1d_features_to_2d_maps_avg(
                x_1d=rearrange(x, 'l b c -> b l c', b=bs),
                n_level=n_level,
                max_h=self.grid_size,
                max_w=self.grid_size,
                first_grow="row",
            )
            latent_tokens = rearrange(latent_tokens, 'b c h w -> (h w) b c').to(x.dtype)
        elif self.learnable_2d_query_init:
            latent_tokens = self.q_2d.repeat(1, bs, 1).to(x.dtype)
        else:
            latent_tokens = x[:1, :, :].repeat(self.grid_size**2, 1, 1).to(x.dtype)

        # mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype
        # x = x + self.latent_token_positional_embedding[:seq_len]
        # x = torch.cat([mask_tokens, x], dim=1)

        query_pos = self.positional_embedding.repeat(1, bs, 1).to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        pos_embed = self.latent_token_positional_embedding[:selected_latent_tokens].repeat(1, bs, 1).to(x.dtype)

        for i in range(self.num_layers):
            latent_tokens = self.transformer[i](latent_tokens, x, pos=pos_embed, query_pos=query_pos)
            if self.out_inner_feat and ret_inner_feat and (i + 1) == self.out_inner_depth:
                inner_feat = self.distill_mlp(latent_tokens)

            elif self.out_inner_feat and return_feat and (i + 1) == self.out_inner_depth:
                # return the feature without distill_mlp, used for linear probe
                assert not (ret_inner_feat and return_feat), \
                    "ret_inner_feat and return_feat cannot be True at the same time"
                inner_feat = latent_tokens
                # L N D -> N L D
                inner_feat = inner_feat.permute(1, 0, 2)
                # assert inner_feat.shape[1] == self.width, \
                #     f"inner_feat.shape[1]={inner_feat.shape[1]} != self.width={self.width},"\
                #     f"current latent_tokens.shape={latent_tokens.shape}"\
                #     f"current x.shape={x.shape}"
                return None, inner_feat

        latent_tokens = self.ln_post(latent_tokens)
        # L N D -> N D H W
        latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, self.grid_size, self.grid_size)
        latent_tokens = self.conv_out(latent_tokens.contiguous())
        if self.out_inner_feat and (ret_inner_feat or return_feat):
            # L N D -> N L D
            inner_feat = inner_feat.permute(1, 0, 2)
            return latent_tokens, inner_feat
        else:
            return latent_tokens


class ViTDecoder2D(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            token_size=256, 
            dropout=0.0,
            out_inner_feat=False,
            out_inner_dim=768,   # for dino-v2
            out_inner_depth=None,
            transformer_layer_type="QformerLayerVar"
            ):
        super().__init__()
        self.transformer_layer_type = transformer_layer_type
        assert self.transformer_layer_type in \
            ["QformerLayerVar", "TransformerLayer","TransformerDecoderLayer" ]

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.token_size = token_size
        self.out_inner_feat = out_inner_feat
        self.out_inner_depth = out_inner_depth
        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
                "xl": 1280,
                "xxl": 1536,
                "xxxl": 2560
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 6,
                "base": 12,
                "large": 24,
                "xl": 36,
                "xxl": 48,
                "xxxl": 48
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
                "xl": 20,
                "xxl": 24,
                "xxxl": 40
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, 1, self.width))
        self.transformer = nn.ModuleList()
        layer_cls = eval(self.transformer_layer_type)
        for i in range(self.num_layers):
            self.transformer.append(layer_cls(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
            ))
        self.ln_post = nn.LayerNorm(self.width)

        self.conv_out = nn.Conv2d(self.width, token_size, kernel_size=3, stride=1, padding=1)
       
        if out_inner_feat:
            self.distill_mlp = nn.Sequential(
                    nn.Linear(self.width, self.width * 4),
                    nn.SiLU(),
                    nn.Linear(self.width * 4, self.width * 4),
                    nn.SiLU(),
                    nn.Linear(self.width * 4, out_inner_dim),
                    )

    
    def forward(self, z_quantized, ret_inner_feat=False):
        N, C, H, W = z_quantized.shape
        selected_latent_tokens = W
        x = z_quantized.reshape(N, C, H*W).permute(2, 0, 1) # LND
        x = self.decoder_embed(self.ln_pre(x))

        seq_len, bs, _ = x.shape    # shape: (num_latent_tokens, B, c)
        latent_tokens = x

        pos_embed = self.positional_embedding.repeat(1, bs, 1).to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        if self.transformer_layer_type == "TransformerLayer":
                    # this is the original transformer layer, use absolute position embedding
                    latent_tokens = latent_tokens + pos_embed

        for i in range(self.num_layers):
            if self.transformer_layer_type == "TransformerDecoderLayer":
                latent_tokens = self.transformer[i](
                                    latent_tokens, 
                                    latent_tokens, 
                                    pos=pos_embed, 
                                    query_pos=pos_embed, 
                                    tgt_mask=None
                                    )
            elif self.transformer_layer_type == "TransformerLayer":
                latent_tokens = self.transformer[i](
                                    latent_tokens
                                    )
            elif self.transformer_layer_type == "QformerLayerVar":
                latent_tokens = self.transformer[i](
                                    latent_tokens, 
                                    pos=pos_embed
                                    )
            if self.out_inner_feat and ret_inner_feat and (i + 1) == self.out_inner_depth:
                inner_feat = self.distill_mlp(latent_tokens)

        latent_tokens = self.ln_post(latent_tokens)
        # L N D -> N D H W
        latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, self.grid_size, self.grid_size)
        latent_tokens = self.conv_out(latent_tokens.contiguous())
        if self.out_inner_feat and ret_inner_feat:
            # L N D -> N L D
            inner_feat = inner_feat.permute(1, 0, 2)
            return latent_tokens, inner_feat
        else:
            return latent_tokens



class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256,
                 use_attn=True, res_down_sample=False, downsample_match_channel=False
                 ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.use_attn = use_attn
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        prev_out_dim = ch

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res
            res_block = nn.ModuleList()
            if self.use_attn:
                attn_block = nn.ModuleList()
            # block_in = ch*in_ch_mult[i_level]
            block_in = prev_out_dim
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if self.use_attn:
                    if i_level == self.num_resolutions - 1:
                        attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            if self.use_attn:
                conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                if res_down_sample:
                    assert resamp_with_conv, 'res_down_sample only support resamp_with_conv'
                    if downsample_match_channel:
                        conv_block.downsample = DownsamplerWithPixunshuffleResidual(
                                                    block_in, 
                                                    ch*ch_mult[i_level+1],
                                                )
                        prev_out_dim = ch*ch_mult[i_level+1]
                    else:
                        conv_block.downsample = DownsamplerWithPixunshuffleResidual(block_in)
                        prev_out_dim = block_out
                else:
                    conv_block.downsample = Downsample(block_in, resamp_with_conv)
                    prev_out_dim = block_out

            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        if self.use_attn:
            self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if self.use_attn and len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3, 
                 adaptive_gn=False, d2s_up=False, use_attn=True,
                 res_up_sample=False, upsample_match_channel=False
                 ):
        """
        adaptive_gn: whether to use adaptive group normalization as in MAGVIT-v2
        d2s_up: whether to use depth_to_space for up sampling
        res_up: whether to use residual non-parametric depth-to-space when upsampling
        """
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.adaptive_gn = adaptive_gn
        self.d2s_up = d2s_up

        self.use_attn = use_attn

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        if self.use_attn:
            self.mid.append(AttnBlock(block_in, norm_type=norm_type))

        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        prev_out_dim = block_in

        self.conv_blocks = nn.ModuleList()
        if self.adaptive_gn:
            self.adaptive = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            if self.use_attn:
                attn_block = nn.ModuleList()
            block_in = prev_out_dim
            block_out = ch*ch_mult[i_level]
            if self.adaptive_gn:
                self.adaptive.append(AdaptiveGroupNorm(z_channels, block_in))
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if self.use_attn and i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            if self.use_attn:
                conv_block.attn = attn_block
            # upsample
            if i_level != 0:
                if d2s_up:
                    assert not res_up_sample,'d2s_up or res_up_sample can not be True at the same time'
                    conv_block.upsample = D2SUpsampler(block_in)
                    prev_out_dim = block_in
                elif res_up_sample:
                    if upsample_match_channel:
                        conv_block.upsample = UpsamplerWithPixshuffleDupResidual(
                                                    block_in,
                                                    ch*ch_mult[i_level-1],
                                                )
                        prev_out_dim = ch*ch_mult[i_level-1]
                    else:
                        conv_block.upsample = UpsamplerWithPixshuffleDupResidual(block_in)
                        prev_out_dim = block_in
                else:
                    conv_block.upsample = Upsample(block_in, resamp_with_conv)
                    prev_out_dim = block_in

            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):

        if self.adaptive_gn:
            style = z.clone()

        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            if self.adaptive_gn:
                ### pass in each resblock first adaGN
                try:
                    h = self.adaptive[i_level](h, style)
                except Exception as e:
                    error_info = str(e) + f"Showing the h shape: {h.shape}, {style.shape}"
                    raise ValueError(error_info)
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if self.use_attn and len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


   

############################################################################
## Deprecated
############################################################################


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class QFormerUpsample(nn.Module):
    """
    Cross attn module as an Upsampler
    """
    def __init__(
        self,
        width,
        nhead,
        dropout,
        num_queries=256,
        mlp_ratio=4.0,
        activation=nn.GELU,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, nhead, dropout=dropout)
        self.cross_attn= nn.MultiheadAttention(width, nhead, dropout=dropout)
        scale = width ** -0.5
        self.query_1d = nn.Parameter(
            scale * torch.randn(num_queries, 1, width)
        )

        dim_feedforward = int(width * mlp_ratio)
        self.linear1 = nn.Linear(width, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, width)

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()

    def forward(self, x):
        x_2 = self.norm1(x) # N, B, C
        x_2 = self.self_attn(x_2, x_2, value=x_2)[0]
        x = x + self.dropout1(x_2)
        x = self.norm2(x)

        x = self.cross_attn(query=self.query_1d, key=x, value=x)[0]
        x_2 = self.norm3(x)
        x_2 = self.linear2(self.dropout(self.activation(self.linear1(x_2))))
        x = x + self.dropout3(x_2)
        return x




class UpsamplerWithPixshuffleDupResidual(nn.Module):
    """
    Using pixshuffle for upsampling, and with a residual connction.
    The residual is the depth to space + channel duplicated value
    """
    def __init__(
        self,
        dim,
        dim_out=None,
        factor=2,    # the spatial upsampling factor
    ):
        super().__init__()

        # for the main upsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.factor = factor
        self.conv1 = nn.Conv2d(dim, self.dim_out * factor**2, (3, 3), padding=1)

        # for residual non-parameteric connections
        # note we
        assert self.dim_out * factor**2 % dim == 0
        self.repeats = self.dim_out * factor**2 // dim


    def forward(self, x: torch.Tensor):
        """
        input_image: [B C H W]
        """
        # we use the implementation from efficientvit/models/nn/ops: first duplicate then shuffle
        # but this is not exactly the same as the LARP paper presents(first shuffle then duplicate).
        residual = x.repeat_interleave(self.repeats, dim=1)
        residual = F.pixel_shuffle(residual, self.factor)
 
        out = self.conv1(x)
        out = F.pixel_shuffle(out, self.factor)
        return out + residual

class DownsamplerWithPixunshuffleResidual(nn.Module):
    """
    Using pixshuffle for upsampling, and with a residual connction.
    The residual is the depth to space + channel duplicated value
    """
    def __init__(
        self,
        dim,
        dim_out=None,
        factor=2,    # the spatial downsampling factor
    ):
        super().__init__()

        # for the main downsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.factor = factor
        self.conv1 = nn.Conv2d(dim, self.dim_out // factor**2, (3, 3), padding=1)

        # for residual non-parameteric connections
        # note we
        self.factor = factor
        assert dim * factor**2 % self.dim_out == 0
        self.group_size = dim * factor**2 // self.dim_out


    def forward(self, x: torch.Tensor):
        """
        input_image: [B C H W]
        """
        residual = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = residual.shape
        residual = residual.view(B, self.dim_out, self.group_size, H, W)
        residual = residual.mean(dim=2)
 
        out = self.conv1(x)
        out = F.pixel_unshuffle(out, self.factor)
        return out + residual


class ChannelDownsampleResidual(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
    ):
        """
        Down sample the width by a conv and a shortcut with channel averaging
        """
        super().__init__()
        # for the main downsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.norm = Normalize(dim)
        self.nonlinear = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, self.dim_out,  (3, 3), padding=1)

        self.in_channels = dim
        self.out_channels = dim_out
        assert self.in_channels % self.out_channels == 0
        self.group_size = self.in_channels // self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x.view(B, self.out_channels, self.group_size, H, W)
        residual = residual.mean(dim=2)

        x = self.norm(x)
        x = self.nonlinear(x)
        x = self.conv1(x)

        return x + residual


class ChannelUpsampleResidual(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
    ):
        """
        Up sample the width by a conv and a shortcut with channel duplication
        """
        super().__init__()
        # for the main downsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.conv1 = nn.Conv2d(dim, self.dim_out,  (3, 3), padding=1)

        self.in_channels = dim
        self.out_channels = dim_out
        assert self.out_channels % self.in_channels == 0
        self.repeats = self.out_channels // self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.repeat_interleave(self.repeats, dim=1)
        x = self.conv1(x)

        return x + residual



class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class QFormerUpsample(nn.Module):
    """
    Cross attn module as an Upsampler
    """
    def __init__(
        self,
        width,
        nhead,
        dropout,
        num_queries=256,
        mlp_ratio=4.0,
        activation=nn.GELU,
    ):
        super().__init__()
        self.cross_attn= nn.MultiheadAttention(width, nhead, dropout=dropout)
        scale = width ** -0.5
        self.num_queries = num_queries
        self.query_1d = nn.Parameter(
            scale * torch.randn(num_queries, 1, width)
        )

        self.pos_emb = nn.Parameter(
            scale * torch.randn(num_queries, 1, width)
        )

        dim_feedforward = int(width * mlp_ratio)
        self.linear1 = nn.Linear(width, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, width)

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        # self.dropout1 = nn.Dropout(dropout)

        self.activation = activation()

    def forward(self, x):
        if x.shape[0] == self.num_queries:
            return x
        else:
            x_2 = self.norm1(x)
            x_2 = self.pos_emb[:x.shape[0]].repeat(1, x.shape[1], 1).to(x.dtype) + x_2
            query = self.query_1d[x.shape[0]:].repeat(1, x.shape[1], 1).to(x.dtype)
            x_2 = self.cross_attn(query=query, key=x_2, value=x_2)[0]
            x_2 = self.norm2(x_2)
            x_2 = self.linear2(self.dropout(self.activation(self.linear1(x_2))))
            # concat at the first dimension
            x = torch.cat([x, x_2], dim=0)
            return x