# Modified from:
#   R3GAN: https://github.com/brownvc/R3GAN
#   vit-pytorch: https://github.com/lucidrains/vit-pytorch/blob/c3018d14339fe57912a27ff59f0c170f40880874/vit_pytorch/efficient.py#L9
"""
This version will use the R3GAN modification to modernize the discriminator.
Specifically, we will use the ResNet structure, and R3 losses stablize the training.
We also plan to further implement a ViT version for better scalability.
But for ViT, the varying input resolution is till to be solved
"""
import functools
import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from tokenizer.tokenizer_image.vq.blocks import ResnetBlock, Encoder
# from autoregressive.models.gpt_1d import TransformerBlock, ModelArgs, RMSNorm


class ResNetDiscriminator(nn.Module):
    """
    This discriminator uses the ResenetBlock directly from llamagen
    It is to validate whether R3GAN loss can really stablize the training so that 
    we can directly use common working structure from other domain
    """
    pass


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        """
        This transformer uses pre-normailization
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)



class PatchViTDiscriminator(nn.Module):
    """
    Modified from 
    https://github.com/lucidrains/vit-pytorch/vit_pytorch/efficient.py#L9
    """
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            dropout=0.0,
            ):
        super().__init__()
        self.model_size = model_size
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

        # Currently only suportting 1:1 ratio
        # TODO: more flexible implementations
        img_h, img_w = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        num_patches = img_h // patch_height * img_w // patch_width
        patch_dim = img_h * img_w * 3 // num_patches

        self.patch_num_h = img_h // patch_height
        self.patch_num_w = img_w // patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.width),
            nn.LayerNorm(self.width),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.width))
        self.dropout = nn.Dropout(dropout)

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # TODO: use its own config, not the borrowed one from gpt
        # dim, depth, heads, dim_head, mlp_dim, dropout = 0.
        self.transformer = Transformer(
                    dim=self.width,
                    depth=self.num_layers,
                    heads=self.num_heads,
                    dim_head=self.width // self.num_heads,
                    mlp_dim=self.width * 4,
                    dropout=dropout,
                )
        self.norm = nn.LayerNorm(self.width)
        # output to logits
        self.head = nn.Linear(self.width, 1)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # we don't use cls token beacause we use patch discriminator
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n]
        x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # x = self.to_latent(x)
        x = self.norm(x)
        x = self.head(x)

        # from B, N, C to B, C, H, W
        x = rearrange(x, 'b (h w) c -> b c h w', h = self.patch_num_h, w = self.patch_num_w)
        return x

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)
        # nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)



