"""
Modified from 
https://github.com/TencentARC/Open-MAGVIT2/blob//taming/modules/vqvae/lookup_free_quantize.py
"""

from typing import List

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, reduce, pack, unpack

import numpy as np



def entropy(prob):
    return (-prob * torch.log(prob + 1e-5)).sum(dim=-1)

# class

def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y


def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benefit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.

    The drawback is larger floating point errors
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))

def entropy_loss(
    logits,
    mask=None,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    if mask is not None:
        # avg_probs = probs[mask].mean(tuple(range(probs.ndim - 1)))
        # avg_probs = einx.mean("... D -> D", probs[mask])

        avg_probs = masked_mean(probs, mask)
        # avg_probs = einx.mean("... D -> D", avg_probs)
    else:
        avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        # sample_entropy = sample_entropy[mask].mean()
        sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss


def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class LFQ(nn.Module):
    """
    Modified from 
    https://github.com/TencentARC/Open-MAGVIT2/blob/taming/modules/vqvae/lookup_free_quantize.py
    """
    def __init__(
        self,
        dim,
        beta, 
        entropy_loss_ratio, 
        n_e=None,   # codebook size
        num_codebooks = 1,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        token_factorization = False,
        factorized_bits = [9, 9],
        show_usage = True,
    ):
        super().__init__()

        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.show_usage = show_usage

        # some assert validations
        assert dim is not None or n_e is not None, \
            'either dim or codebook_size must be specified for LFQ'

        assert n_e is None or np.log2(n_e).is_integer(), \
            f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** np.ceil(np.log2(n_e))})'

        self.n_e = default(n_e, lambda: 2 ** dim)
        self.e_dim = int(np.log2(n_e))

        codebook_dims = self.e_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.has_projections = has_projections

        self.dim = dim
        self.e_dim = self.e_dim
        self.num_codebooks = num_codebooks
        
        # for entropy loss
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # for no auxiliary loss, during inference
        self.token_factorization = token_factorization
        if not self.token_factorization: #for first stage model
            # used for bits to indices
            self.register_buffer('mask', 2 ** torch.arange(self.e_dim), persistent=False)
        else:
            self.factorized_bits = factorized_bits
            self.register_buffer("pre_mask", 2** torch.arange(factorized_bits[0]), persistent=False)
            self.register_buffer("post_mask", 2**torch.arange(factorized_bits[1]), persistent=False)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # codes
        all_codes = torch.arange(n_e)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0

        self.register_buffer('codebook', codebook, persistent = False)


        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    @property
    def dtype(self):
        return self.codebook.dtype
    
    def indices_to_bits(self, x):
        """
        x: long tensor of indices

        returns big endian bits (bool, True for 1, False for -1)
        """
        mask = 2 ** torch.arange(self.e_dim, device=x.device, dtype=torch.long)
        # x is now big endian bits, the last dimension being the bits
        x = (x.unsqueeze(-1) & mask) != 0
        return x

    def get_codebook_entry(self, x, shape, order): #0610
        if self.token_factorization:
            if order == "pre":
                mask = 2 ** torch.arange(self.factorized_bits[0], device=x.device, dtype=torch.long)
            else:
                mask = 2 ** torch.arange(self.factorized_bits[1], device=x.device, dtype=torch.long)
        else:
            mask = 2 ** torch.arange(self.e_dim, device=x.device, dtype=torch.long)
        
        # indices_to_bits
        x = (x.unsqueeze(-1) & mask) != 0
        x = x * 2.0 - 1.0 #back to the float
        ## scale back to the 
        b, c, h, w = shape
        x = rearrange(x, "(b h w) c -> b h w c", h=h, w=w, b=b)
        x = rearrange(x, "b h w c -> b c h w")
        return x

    def bits_to_indices(self, bits):
        """
        bits: bool tensor of big endian bits, where the last dimension is the bit dimension

        returns indices, which are long integers from 0 to self.codebook_size
        """
        assert bits.shape[-1] == self.e_dim
        indices = 2 ** torch.arange(
            0,
            self.e_dim,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)
    
    def decode(self, x):
        """
        x: ... NH
            where NH is number of codebook heads
            A longtensor of codebook indices, containing values from
            0 to self.codebook_size
        """
        x = self.indices_to_bits(x)
        # to some sort of float
        x = x.to(self.dtype)
        # -1 or 1
        x = x * 2 - 1
        x = rearrange(x, "... NC Z-> ... (NC Z)")
        return x

    def forward(
        self,
        z,
        mask = None,
        return_loss = True,
        stochastic = False,
        **kwargs
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        for k, v in kwargs.items():
            assert v is None or not v, f'unsupported parameter {k}'

        z = rearrange(z, 'b d ... -> b ... d')
        z, ps = pack_one(z, 'b * d')
        # split out number of codebooks

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)


        codebook_value = torch.Tensor([1.0]).to(device=z.device, dtype=z.dtype)

        if stochastic:
            z = torch.sigmoid(z)
            quantized = torch.bernoulli(z)
            quantized = (quantized - 0.5) * 2.0 * codebook_value # -1 or 1
        else:
            quantized = torch.where(z > 0, codebook_value, -codebook_value) # higher than 0 filled 

        # calculate indices
        if self.token_factorization:
            indices_pre = reduce((quantized[..., :self.factorized_bits[0]] > 0).int() * self.pre_mask.int(), "b n c d -> b n c", "sum")
            indices_post = reduce((quantized[..., self.factorized_bits[0]:] > 0).int() * self.post_mask.int(), "b n c d -> b n c", "sum")
        else:
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

        # entropy aux loss

        if self.training and return_loss:
            logits = 2 * einsum('... i d, j d -> ... i j', z, self.codebook)
            # the same as euclidean distance up to a constant
            per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                logits = logits,
                sample_minimization_weight = self.sample_minimization_weight,
                batch_maximization_weight = self.batch_maximization_weight
            )
            avg_probs = self.zero
            if self.entropy_loss_ratio > 0:
                entropy_aux_loss = self.entropy_loss_ratio * entropy_aux_loss
            else:
                entropy_aux_loss = 0
        else:
            # logits = 2 * einsum('... i d, j d -> ... i j', x, self.codebook)
            # probs = F.softmax(logits / 0.01, -1)
            # avg_probs = reduce(probs, "b n c d -> b d", "mean")
            # avg_probs = torch.sum(avg_probs, 0) #batch dimension
            # if not training, just return dummy 0
            per_sample_entropy = codebook_entropy = self.zero
            ## calculate the codebook_entropy needed for one batch evaluation
            entropy_aux_loss = self.zero
            avg_probs = self.zero

        # commit loss

        if self.training:
            commit_loss = F.mse_loss(z, quantized.detach(), reduction = 'none')

            if exists(mask):
                commit_loss = commit_loss[mask]

            commit_loss = commit_loss.mean()
            commit_loss = self.beta * commit_loss
        else:
            commit_loss = self.zero


        # use straight-through gradients (optionally with custom activation fn) if training

        quantized = z + (quantized - z).detach() #transfer to quantized

        # merge back codebook dim

        quantized = rearrange(quantized, 'b n c d -> b n (c d)')

        # reconstitute image or video dimensions, i.e. b c h w or b c t h w
        quantized = unpack_one(quantized, ps, 'b * d')
        quantized = rearrange(quantized, 'b ... d -> b d ...')

        
        if self.token_factorization:
            indices_pre = unpack_one(indices_pre, ps, "b * c")
            indices_post = unpack_one(indices_post, ps, "b * c")
            indices_pre = indices_pre.flatten()
            indices_post = indices_post.flatten()
            indices = (indices_pre, indices_post)
        else:
            indices = unpack_one(indices, ps, 'b * c')
            indices = indices.flatten() # b * h * w

        codebook_usage = 0
        if self.show_usage and self.training:
            cur_len = indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = indices 
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        perplexity = None
        min_encodings = None

        return quantized, (commit_loss, entropy_aux_loss, codebook_usage), (perplexity, min_encodings, indices)