# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   SeD: https://github.com/lbc12345/SeD/tree/main/models
import functools
import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm
from torch.nn import functional as F
import math

from einops import rearrange

from tokenizer.tokenizer_image.module_attention_SeD import ModifiedSpatialTransformer
from tokenizer.tokenizer_image.discriminator_patchgan import BlurPool



class PatchGANSeDiscriminatorV3(nn.Module):
    def __init__(
            self, 
            input_nc=3, 
            ndf=64, 
            semantic_dim=768, 
            semantic_size=16, 
            use_bias=True, 
            norm_layer=None,
            nheads=1, 
            dhead=64,
            kw=4,
            blur_ds=False
            ):
        """Construct a PatchGAN discriminator
        A relatively weak version compared to original semantic discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if norm_layer == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_layer == "group":
            # group norm 
            norm_layer = nn.GroupNorm
        else:
            norm_layer = nn.Identity

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        use_bias = True

        self.semantic_size = semantic_size

        padw = 1        
        # ss = [128, 64, 32, 31, 30]  # PatchGAN's spatial size
        # comment: wrong. ss = [128, 64, 32, 32, 32]
        # cs = [64, 128, 256, 512, 1]  # PatchGAN's channel size

        norm = spectral_norm

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.conv_first = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw) if not blur_ds \
                        else nn.Sequential(
                            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
                            BlurPool(ndf, stride=2),
                        )
        self.norm0 = norm_layer(ndf)

        self.conv1 = norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias)) \
                    if not blur_ds else nn.Sequential(
                        norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=1, padding=padw)),
                        BlurPool(ndf * 2, stride=2),
                    )

        # upscale = math.ceil(64 / semantic_size)
        # self.att1 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=ndf * 2, up_factor=upscale)

        self.norm1 = norm_layer(ndf * 2)
        # ex_ndf = int(semantic_dim / upscale**2)
        # self.conv11 = norm(nn.Conv2d(ndf * 2 + ex_ndf, ndf * 2, kernel_size=3, stride=1, padding=padw, bias=use_bias)) 

        
        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias)) \
                    if not blur_ds else nn.Sequential(
                        norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=1, padding=padw)),
                        BlurPool(ndf * 4, stride=2),
                    )

        # upscale = math.ceil(32 / semantic_size)
        # self.att2 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=ndf * 4, up_factor=upscale)
        
        self.norm2 = norm_layer(ndf * 4)
        # ex_ndf = int(semantic_dim / upscale**2)
        # self.conv21 = norm(nn.Conv2d(ndf * 4 + ex_ndf, ndf * 4, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        upscale = math.ceil(32 / semantic_size)
        self.att3 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=ndf * 8, up_factor=upscale, is_last=True and kw==4)
        print("attention dhead:", dhead)

        self.norm3 = norm_layer(ndf * 8)
        
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv31 = norm(nn.Conv2d(ndf * 8 + ex_ndf, ndf * 8, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv_last = nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)
        
        init_weights(self, init_type='normal')

    def forward(self, input, semantic):
        """Standard forward."""
        semantic = torch.ones_like(semantic, device=input.device)
        if semantic.dim() == 3:
            # N L D -> N C H W
            semantic = rearrange(semantic, 'N (H W) D -> N D H W', H=self.semantic_size, W=self.semantic_size)

        input = self.conv_first(input)
        input = self.norm0(input)
        input = self.lrelu(input)
        
        input = self.conv1(input)
        input = self.norm1(input)
        input = self.lrelu(input)
        
        input = self.conv2(input)
        input = self.norm2(input)
        input = self.lrelu(input)
        
        input = self.conv3(input)
        se = self.att3(semantic, input)
        input = self.norm3(input)
        input = self.lrelu(self.conv31(torch.cat([input, se], dim=1)))
            
        input = self.conv_last(input)
        return input



class PatchGANSeDiscriminatorV2(nn.Module):
    def __init__(
            self, 
            input_nc=3, 
            ndf=64, 
            semantic_dim=768, 
            semantic_size=16, 
            use_bias=True, 
            norm_layer=None,
            nheads=1, 
            dhead=64,
            kw=4,
            blur_ds=False
            ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if norm_layer == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_layer == "group":
            # group norm 
            norm_layer = nn.GroupNorm
        else:
            norm_layer = nn.Identity

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        use_bias = True

        self.semantic_size = semantic_size

        padw = 1        
        # ss = [128, 64, 32, 31, 30]  # PatchGAN's spatial size
        # comment: wrong. ss = [128, 64, 32, 32, 32]
        # cs = [64, 128, 256, 512, 1]  # PatchGAN's channel size


        norm = spectral_norm

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.conv_first = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw) if not blur_ds \
                        else nn.Sequential(
                            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
                            BlurPool(ndf, stride=2),
                        )
        self.norm0 = norm_layer(ndf)

        self.conv1 = norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias)) \
                    if not blur_ds else nn.Sequential(
                        norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=1, padding=padw)),
                        BlurPool(ndf * 2, stride=2),
                    )

        upscale = math.ceil(64 / semantic_size)
        self.att1 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=ndf * 2, up_factor=upscale)

        self.norm1 = norm_layer(ndf * 2)
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv11 = norm(nn.Conv2d(ndf * 2 + ex_ndf, ndf * 2, kernel_size=3, stride=1, padding=padw, bias=use_bias)) 

        
        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias)) \
                    if not blur_ds else nn.Sequential(
                        norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=1, padding=padw)),
                        BlurPool(ndf * 4, stride=2),
                    )

        upscale = math.ceil(32 / semantic_size)
        self.att2 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=ndf * 4, up_factor=upscale)
        
        self.norm2 = norm_layer(ndf * 4)
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv21 = norm(nn.Conv2d(ndf * 4 + ex_ndf, ndf * 4, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        upscale = math.ceil(32 / semantic_size)
        self.att3 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=ndf * 8, up_factor=upscale, is_last=True and kw==4)

        self.norm3 = norm_layer(ndf * 8)
        
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv31 = norm(nn.Conv2d(ndf * 8 + ex_ndf, ndf * 8, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv_last = nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)
        
        init_weights(self, init_type='normal')

    def forward(self, input, semantic):
        """Standard forward."""

        if semantic.dim() == 3:
            # N L D -> N C H W
            semantic = rearrange(semantic, 'N (H W) D -> N D H W', H=self.semantic_size, W=self.semantic_size)

        input = self.conv_first(input)
        input = self.norm0(input)
        input = self.lrelu(input)
        
        input = self.conv1(input)
        se = self.att1(semantic, input)
        input = self.norm1(input)
        input = self.lrelu(self.conv11(torch.cat([input, se], dim=1)))
        
        input = self.conv2(input)
        se = self.att2(semantic, input)
        input = self.norm2(input)
        input = self.lrelu(self.conv21(torch.cat([input, se], dim=1)))
        
        input = self.conv3(input)
        se = self.att3(semantic, input)
        input = self.norm3(input)
        input = self.lrelu(self.conv31(torch.cat([input, se], dim=1)))
            
        input = self.conv_last(input)
        return input



class PatchGANSeDiscriminator(nn.Module):
    def __init__(
            self, 
            input_nc=3, 
            ndf=64, 
            semantic_dim=768, 
            semantic_size=16, 
            use_bias=True, 
            nheads=1, 
            dhead=64,
            ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()

        self.semantic_size = semantic_size

        kw = 4
        padw = 1        
        # ss = [128, 64, 32, 31, 30]  # PatchGAN's spatial size
        # comment: wrong. ss = [128, 64, 32, 32, 32]
        # cs = [64, 128, 256, 512, 1]  # PatchGAN's channel size

        norm = spectral_norm

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.conv_first = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)

        self.conv1 = norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = math.ceil(64 / semantic_size)
        self.att1 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=128, up_factor=upscale)
        
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv11 = norm(nn.Conv2d(ndf * 2 + ex_ndf, ndf * 2, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = math.ceil(32 / semantic_size)
        self.att2 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=256, up_factor=upscale)
        
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv21 = norm(nn.Conv2d(ndf * 4 + ex_ndf, ndf * 4, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        upscale = math.ceil(32 / semantic_size)
        self.att3 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=512, up_factor=upscale, is_last=True)
        
        ex_ndf = int(semantic_dim / upscale**2)
        self.conv31 = norm(nn.Conv2d(ndf * 8 + ex_ndf, ndf * 8, kernel_size=3, stride=1, padding=padw, bias=use_bias))
        
        self.conv_last = nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)
        
        init_weights(self, init_type='normal')

    def forward(self, input, semantic):
        """Standard forward."""

        if semantic.dim() == 3:
            # N L D -> N C H W
            semantic = rearrange(semantic, 'N (H W) D -> N D H W', H=self.semantic_size, W=self.semantic_size)

        input = self.conv_first(input)
        input = self.lrelu(input)
        
        input = self.conv1(input)
        se = self.att1(semantic, input)
        input = self.lrelu(self.conv11(torch.cat([input, se], dim=1)))
        
        input = self.conv2(input)
        se = self.att2(semantic, input)
        input = self.lrelu(self.conv21(torch.cat([input, se], dim=1)))
        
        input = self.conv3(input)
        se = self.att3(semantic, input)
        input = self.lrelu(self.conv31(torch.cat([input, se], dim=1)))
            
        input = self.conv_last(input)
        return input

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)