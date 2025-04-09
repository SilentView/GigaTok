"""
Codes adapted from https://github.com/hywang66/LARP/

This is the continuous gpt model. It will take token embedding as input, and output continuous features
The features will be used to calculate the logits using similarity.
Currently, this gptc is only for regularization when training tokenizers.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F




@dataclass
class GPTCConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    max_seq_len: int = 1024
    n_ind: int = 16 # number of input dim
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    detach_x: bool = False
    detach_target: bool = True
    l2_normalized: bool = True
    n_classes: int = -1
    fully_separated: bool = False



class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

        self.p_attn_drop = config.attn_pdrop

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v)) # (2, B, nh, T, hs)
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        if hasattr(F, "scaled_dot_product_attention") and torch.__version__ >= "2.1.0":
            is_causal = layer_past is None 
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_attn_drop, is_causal=is_causal)  
        else:
            raise NotImplementedError("scaled_dot_product_attention not available in this version of PyTorch")
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present  


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPTC(nn.Module):
    """  the continuous GPT model"""
    def __init__(self, config: GPTCConfig) -> None:
        super().__init__()
        # input embedding stem
        self.input_proj = nn.Linear(config.n_ind, config.n_embd)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_seq_len, config.n_embd) * 0.02) 
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.apply(self._init_weights)
        self.config = config
        self.max_seq_length = config.max_seq_len
        self.detach_x = config.detach_x
        self.detach_target = config.detach_target
        self.l2_normalized = config.l2_normalized

        self.n_classes = config.n_classes
        self.fully_separated = config.fully_separated
        assert not (self.detach_x and self.detach_target), 'Cannot detach both x and target'
        self.head = nn.Linear(config.n_embd, config.n_ind)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, targets=None):
        # forward the GPTC model
        # x: (b, n, n_ind)
        token_embeddings = self.input_proj(x) # (b, n, d)

        x = self.drop(token_embeddings + self.pos_emb[:, :token_embeddings.shape[1], :])

        x = self.blocks(x)

        x = self.ln_f(x)
        pred = self.head(x) # (b, n, n_ind)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if self.diff_loss:
                loss = self.diff_loss_module.loss(pred, targets)
            else:
                loss = F.mse_loss(pred, targets)

        return pred, loss
    
    def compute_prior_loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, n, n_ind) 
        if self.l2_normalized:
            x = F.normalize(x, p=2, dim=-1)
            
        target = x[:, 1:]
        if self.detach_target:
            target = target.detach()

        x = x[:, :-1]
        if self.detach_x:
            x = x.detach()

        _, loss = self.forward(x, targets=target)

        return loss

    def ar_predict(self, x: torch.Tensor) -> torch.Tensor:
        # make ar prediction using teacher forcing
        # x: (b, n, n_ind)
        x = x[:, :-1] # (b, n-1, n_ind)
        pred, _ = self.forward(x) # (b, n-1, n_ind)
        full_pred = torch.cat([x[:, :1], pred], dim=1) # (b, n, n_ind)

        if self.l2_normalized:
            full_pred = F.normalize(full_pred, p=2, dim=-1)
        return full_pred
    


#################################################################################
#                                 GPTC Configs                                  #
#################################################################################   

def GPTC_L(**kwargs):
    return GPTC(GPTCConfig(n_layer=24, n_head=16, n_embd=1024, **kwargs)) # 316.4M?

def GPTC_B(**kwargs):
    return GPTC(GPTCConfig(n_layer=12, n_head=12, n_embd=768, **kwargs)) # 85.9M

def GPTC_M(**kwargs):
    return GPTC(GPTCConfig(n_layer=12, n_head=8, n_embd=512, **kwargs)) # 38.4M

def GPTC_S(**kwargs):
    return GPTC(GPTCConfig(n_layer=12, n_head=6, n_embd=384, **kwargs)) # 21.7M

def GPTC_XS(**kwargs):
    return GPTC(GPTCConfig(n_layer=6, n_head=6, n_embd=384, **kwargs)) # 11.1M

def GPTC_XXS(**kwargs):
    return GPTC(GPTCConfig(n_layer=6, n_head=4, n_embd=256, **kwargs)) # 5.0M

GPTC_models = {
    'gptc-L': GPTC_L,
    'gptc-B': GPTC_B,
    'gptc-M': GPTC_M,
    'gptc-S': GPTC_S,
    'gptc-XS': GPTC_XS,
    'gptc-XXS': GPTC_XXS
}

'''
# Count number of parameters

import torch
import models
from utils import compute_num_params
compute_num_params(models.make({'name': 'gptc-S', 'args': {}}))

'''

