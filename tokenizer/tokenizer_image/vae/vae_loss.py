# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
#   LightningDiT: https://github.com/hustvl/LightningDiT/blob/main/tokenizer/vavae.py
#   LDM: https://github.com/CompVis/latent-diffusion/blob/main/ldm/
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import numpy as np

from tokenizer.tokenizer_image.lpips import ResNet50ImgSim, LPIPS, DinoV2ImgSim
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminatorV2 as PatchGANDiscriminatorV2
from tokenizer.tokenizer_image.discriminator_stylegan import Discriminator as StyleGANDiscriminator
from tokenizer.tokenizer_image.discriminator_patchgan_SeD import PatchGANSeDiscriminatorV3
from tokenizer.tokenizer_image.discriminator_patchr3gan import PatchViTDiscriminator


from tokenizer.tokenizer_image.discriminator_dino import DinoDisc as DINODiscriminator
from tokenizer.tokenizer_image.diffaug import DiffAug

from utils.resume_log import wandb_cache_file_append

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

########################################
# R3GAN
# code modified from R3GAN: https://github.com/brownvc/R3GAN
########################################
def zero_centered_gradient_penalty(samples, critics):
    gradient, = torch.autograd.grad(
                    outputs=critics.sum(), 
                    inputs=samples, 
                    create_graph=True,
                    )
    return gradient.square().sum([1, 2, 3])


def r3gan_d_loss(
        logits_real, 
        logits_fake, 
        samples_real, 
        samples_fake,
        gamma=15,   # this may need to be tuned
        ema=None,
        iter=None,
        ):
    assert ema is None
    # assert iter is None
    # Relativistic discriminator loss
    relat_logits = logits_real - logits_fake
    adv_loss = nn.functional.softplus(-relat_logits)

    # R1 gradient penalty
    r1_penalty = zero_centered_gradient_penalty(samples=samples_real, critics=logits_real)
    # R2 gradient penalty
    r2_penalty = zero_centered_gradient_penalty(samples=samples_fake, critics=logits_fake)

    try:
        disc_loss = torch.mean(adv_loss) + torch.mean((gamma / 2) * (r1_penalty + r2_penalty))
    except:
        print("adv_loss shape", adv_loss.shape)
        print("r1_penalty shape", r1_penalty.shape)
        print("r2_penalty shape", r2_penalty.shape)
        exit()
    return disc_loss


def r3gan_gen_loss(logits_real, logits_fake):
    relat_logits = logits_fake - logits_real 
    adv_loss = nn.functional.softplus(-relat_logits)

    return torch.mean(adv_loss)


########################################
# hinge
########################################
def loss_hinge_dis(dis_fake, dis_real, ema=None, it=None):
  if ema is not None:
    # track the prediction
    ema.update(torch.mean(dis_fake).item(), 'D_fake', it)
    ema.update(torch.mean(dis_real).item(), 'D_real', it)

  loss_real = F.relu(1. - dis_real)
  loss_fake = F.relu(1. + dis_fake)
  return torch.mean(loss_real), torch.mean(loss_fake)


def hinge_d_loss(logits_real, logits_fake, ema=None, iter=None):
    if ema is not None:
        # track the prediction
        ema.update(torch.mean(logits_fake).item(), 'D_fake', iter)
        ema.update(torch.mean(logits_real).item(), 'D_real', iter)

    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

# LeCam Regularziation loss
# from https://github.com/google/lecam-gan/
def lecam_reg(dis_real, dis_fake, ema):
  reg = torch.mean(F.relu(dis_real - ema.D_fake).pow(2)) + \
        torch.mean(F.relu(ema.D_real - dis_fake).pow(2))
  return reg


# Simple wrapper that applies EMA to losses.
# from https://github.com/google/lecam-gan/blob/f9af9485eda4637b25e694c142ce8e6992eb7243/third_party/utils.py#L636C1-L661C60
class ema_losses(object):
    def __init__(self, init=1000., decay=0.99, start_itr=0):
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay
        self.start_itr = start_itr

    def update(self, cur, mode, itr):
        if itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        if mode == 'G_loss':
          self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == 'D_loss_real':
          self.D_loss_real = self.D_loss_real*decay + cur*(1 - decay)
        elif mode == 'D_loss_fake':
          self.D_loss_fake = self.D_loss_fake*decay + cur*(1 - decay)
        elif mode == 'D_real':
          self.D_real = self.D_real*decay + cur*(1 - decay)
        elif mode == 'D_fake':
          self.D_fake = self.D_fake*decay + cur*(1 - decay)


class VAELossSDAlign(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, 
                 disc_weight=1.0,  # the base adv loss weight for adaptive calculation of the generator 
                 disc_adaptive_weight=False,  # whether to adpvtively adjust the adv loss weight for generator
                 disc_factor=1.0,  # the adv loss weight for the discriminator and generator
                 gen_adv_loss='hinge', 
                 reconstruction_loss='l1', 
                 reconstruction_weight=1.0, 
                 perceptual_weight=1.0, 
                 norm="batch",kw=4, blur_ds=False, lecam=False, lecam_weight=0.001,
                 # For VAE, sem loss must be used with adaptive scale, but in the initial stage, nll_grad can be nan
                 # So we need to start later
                 sem_reg_start=0,  
                 proj_weight=0.5, 
                 sem_weight_upper_bound=1e4,
                 sem_adaptive_weight=False,
                 perceptual_model="vgg",      # for perceptual loss setting
                 gamma=15,  # (deprecated) for r3gan R1+R2 panelty, 
                 logvar_init=0.0, kl_weight=1.0,
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan", "patchgan_SeD", "patchganv2", "dinodisc", "patchvit"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating", "r3gan"]

        self.disc_type = disc_type

        self.kl_weight = kl_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # specially for r3gan
        self.gamma = gamma

        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
                norm=norm,
                kw=kw,
                blur_ds=blur_ds
            )
        elif disc_type == "patchganv2":
            self.discriminator = PatchGANDiscriminatorV2(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
                norm=norm,
                kw=kw,
                blur_ds=blur_ds,
                use_semantic_input=use_semantic_input
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        elif disc_type == "patchgan_SeD":
            self.discriminator = PatchGANSeDiscriminatorV3(
                input_nc=disc_in_channels, 
                ndf=disc_dim,
                kw=kw,
                blur_ds=blur_ds,
                dhead=dhead,
            )
        elif disc_type == "dinodisc":
            aug_prob = 1.0
            self.discriminator = DINODiscriminator(norm_type="bn")  # default 224 otherwise crop
            self.daug = DiffAug(prob=aug_prob, cutout=0.2)
        elif disc_type == "patchvit":
            self.discriminator = PatchViTDiscriminator(
                model_size="base",
            )
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        elif disc_loss == "r3gan":
            self.disc_loss = r3gan_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        
        self.disc_loss_type = disc_loss
        self.disc_type = disc_type
        self.gen_adv_loss_type = gen_adv_loss

        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight
        self.disc_factor = disc_factor

        self.proj_weight = proj_weight
        self.sem_adaptive_weight = sem_adaptive_weight
        self.sem_weight_upper_bound = sem_weight_upper_bound
        self.sem_reg_start = sem_reg_start


        assert gen_adv_loss in ["hinge", "non-saturating", "r3gan"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        elif gen_adv_loss == "r3gan":
            self.gen_adv_loss = r3gan_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        if perceptual_model == "resent50":
            self.perceptual_loss = ResNet50ImgSim().eval()
        elif perceptual_model == "dinov2-s":
            self.perceptual_loss = DinoV2ImgSim().eval()
        elif perceptual_model == "vgg":
            self.perceptual_loss = LPIPS().eval()
        else:
            raise ValueError(f"Unknown perceptual model '{perceptual_model}'.")
        self.perceptual_weight = perceptual_weight

        # reconstruction loss
        assert reconstruction_loss == "l1"
        # if reconstruction_loss == "l1":
        #     self.rec_loss = F.l1_loss
        # elif reconstruction_loss == "l2":
        #     self.rec_loss = F.mse_loss
        # else:
        #     raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # Special config for logging
        self.log_update_cache_generator = []
        self.log_update_cache_discriminator = []

        self.log_update_cache_multi_level = {}

        self.lecam = lecam
        if self.lecam:
            self.ema_logits = ema_losses(start_itr=self.discriminator_iter_start + 1000)
        else:
            self.ema_logits = None
        self.lecam_weight = lecam_weight


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer, upper_bound=1e4):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        # print("nll_grads norm:", torch.norm(nll_grads))
        # print("g_grads norm:", torch.norm(g_grads))
        if torch.isnan(nll_grads).any() or torch.isnan(g_grads).any():
            d_weight = 1.0
        else:
            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, upper_bound).detach()
        return d_weight

    def forward(self, 
                inputs, reconstructions, posteriors,
                optimizer_idx, global_step, exp_dir, last_layer=None, encoder_last_layer=None,
                logger=None, log_every=100, ckpt_every=500, num_en_q_level=None, causal_type=None,
                check_nan_loss=True, inner_feat=None, sem_enc_feat=None,
                weights=None, 
                ):
        rank = dist.get_rank() 
        node_rank = int(os.environ.get('NODE_RANK', 0))
        B, C, H, W = inputs.shape
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            # discriminator loss
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            if global_step < self.discriminator_iter_start:
                logits_fake = 0.0
                generator_adv_loss = 0.0
                disc_adaptive_weight = 1.0
            else:
                if self.disc_type == "dinodisc":
                    fade_blur_schedule = 0
                    logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous(), fade_blur_schedule))
                else:
                    logits_fake = self.discriminator(reconstructions.contiguous())

                if self.gen_adv_loss_type == "r3gan":
                    logits_real = self.discriminator(inputs.contiguous())
                    generator_adv_loss = self.gen_adv_loss(logits_real, logits_fake)
                else:
                    generator_adv_loss = self.gen_adv_loss(logits_fake)
                
                if self.disc_adaptive_weight and last_layer is not None:
                    disc_adaptive_weight = self.calculate_adaptive_weight(nll_loss, generator_adv_loss, last_layer=last_layer)
                else:
                    disc_adaptive_weight = 1
            
            
            # compute semantic distill loss
            # projection loss
            proj_loss = 0.
            if global_step >= self.sem_reg_start:
                if inner_feat is not None and sem_enc_feat is not None:
                    assert inner_feat.shape == sem_enc_feat.shape, f"inner_feat.shape: {inner_feat.shape}, sem_enc_feat.shape: {sem_enc_feat.shape}"
                    bsz = inner_feat.shape[0]
                    # TODO: the for loop is ugly, change to array calculation
                    for j, (z_j, z_tilde_j) in enumerate(zip(inner_feat, sem_enc_feat)):
                        z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                        z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                        proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                    proj_loss /= bsz
            
            if self.sem_adaptive_weight and global_step >= self.sem_reg_start:
                # similar to adversarial loss, because the nll_loss has a very large scale
                # we want to make sure the semantic regularization has the same level of affect 
                # in terms of gradients.
                # The solution follws: 
                # - https://github.com/hustvl/LightningDiT/issues/1#issuecomment-2576817610
                # - https://github.com/hustvl/LightningDiT/issues/1#issuecomment-2576896899
                sem_loss_weight = self.calculate_adaptive_weight(
                        nll_loss, 
                        proj_loss, 
                        last_layer=encoder_last_layer,
                        upper_bound=self.sem_weight_upper_bound
                        )
            else:
                sem_loss_weight = 1.0


            loss = weighted_nll_loss + self.kl_weight * kl_loss + \
                (disc_adaptive_weight * disc_weight) * self.disc_factor * generator_adv_loss  + \
                sem_loss_weight * self.proj_weight * proj_loss

            if check_nan_loss:
                if torch.isnan(loss).any():
                    error_info = ""
                    rec_loss = torch.mean(self.rec_weight * rec_loss)
                    p_loss = torch.mean(self.perceptual_weight * p_loss)
                    generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                    sem_reg_loss = sem_loss_weight * self.proj_weight * proj_loss
                    # check any nan in reconstruction input
                    if torch.isnan(inputs).any():
                        error_info += "input contains nan\n"
                    if torch.isnan(reconstructions).any():
                        error_info += "reconstructions contains nan\n"
                    error_info += (f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
                                f"generator_adv_loss: {generator_adv_loss:.4f}, "
                                f"disc_adaptive_weight: {disc_adaptive_weight:.4e}, disc_weight: {disc_weight:.4f}\n"
                                f"sem_reg_loss: {sem_reg_loss:.4e}, sem_loss_weight: {sem_loss_weight:.4e}, "
                                f"kl_loss: {kl_loss:.4e}, weighted_nll_loss: {weighted_nll_loss:.4e}\n"
                                )
                    get_ip_cmd = """hostname -I | awk '{split($0, a, " "); print a[1]}'"""
                    ip_addr = os.popen(get_ip_cmd).read().strip()
                    error_info += f"ip: {ip_addr}\n"
                    error_info += f"current iteration:{global_step}"
                    raise RuntimeError(error_info)
            
            if rank == 0 and node_rank == 0 and (global_step % log_every == 0):
                rec_loss = torch.mean(self.rec_weight * rec_loss)
                p_loss = torch.mean(self.perceptual_weight * p_loss)
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
                            f"generator_adv_loss: {generator_adv_loss:.4f}, " 
                            f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"proj_loss: {proj_loss:.4f}, "
                            f"kl_loss: {kl_loss:.4e}, "
                            f"weighted_nll_loss: {weighted_nll_loss:.4e}, "
                            f"loss: {loss:.2e}, "
                            f"disc_adaptive_weight: {disc_adaptive_weight:.4e}, "
                            f"sem_loss_weight: {sem_loss_weight:.4e}"
                            )

                # update to wandb
                update_info = {
                    "(Generator)rec_loss": rec_loss,
                    "(Generator)perceptual_loss": p_loss,
                    "(Generator)generator_adv_loss": generator_adv_loss,
                    "(Generator)disc_adaptive_weight": disc_adaptive_weight,
                    "(Generator)disc_weight": disc_weight,
                    "iteration": global_step,
                    "(Generator)proj_loss": proj_loss,
                    "(Generator)kl_loss": kl_loss,
                    "(Generator)sem_loss_weight": sem_loss_weight,
                }

                if num_en_q_level is not None and causal_type == "per-level":
                    multi_level_update_info = {
                        key: np.mean(value)
                        for key, value in self.log_update_cache_multi_level.items()
                    }
                    update_info.update(multi_level_update_info)

                    multi_level_info = "\n".join(
                        [f"{key}: {np.mean(value):.4f}, " for key, value in multi_level_update_info.items()]
                    )
                    logger.info(multi_level_info)

                self.log_update_cache_multi_level = {}
                self.log_update_cache_generator.append(update_info)


            if rank == 0 and node_rank == 0 and (global_step % ckpt_every == 0 and global_step > 0):
                # update to wandb
                wandb_cache_file_append(self.log_update_cache_generator, exp_dir)
                self.log_update_cache_generator = []

            return loss

        # discriminator update
        if optimizer_idx == 1:
            if global_step < self.discriminator_iter_start:
                return 0.0

            if self.disc_type == "dinodisc":
                fade_blur_schedule = 0
                # add blur since disc is too strong
                logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous().detach(), fade_blur_schedule))
                logits_real = self.discriminator(self.daug.aug(inputs.contiguous().detach(), fade_blur_schedule))
            elif self.disc_loss_type == "r3gan":
                samples_real = inputs.contiguous().detach().clone().requires_grad_(True)
                samples_fake = reconstructions.contiguous().detach().clone().requires_grad_(True)
                logits_real = self.discriminator(samples_real)
                logits_fake = self.discriminator(samples_fake)
            else:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_weight = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            if self.disc_loss_type == "r3gan":
                d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake, 
                                                                  samples_fake=samples_fake,
                                                                  samples_real=samples_real,
                                                                  gamma=self.gamma,
                                                                  ema=self.ema_logits, iter=global_step,
                                                                  )
                lecam_regularization = 0.0
            else:
                d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake, ema=self.ema_logits, iter=global_step)
                lecam_regularization = self.lecam_weight * lecam_reg(logits_real, logits_fake, ema=self.ema_logits) \
                                        if self.lecam and global_step > self.discriminator_iter_start else 0.0

            if global_step % log_every == 0:
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                logger.info(f"(Discriminator) " 
                            f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f},")

                update_info = {
                    "(Discriminator)discriminator_adv_loss": d_adversarial_loss,
                    "(Discriminator)disc_weight": disc_weight,
                    "(Discriminator)logits_real": logits_real,
                    "(Discriminator)logits_fake": logits_fake,
                    "iteration": global_step,
                }
                self.log_update_cache_discriminator.append(update_info)

            rank = dist.get_rank()             
            if rank == 0 and (global_step % ckpt_every == 0 and global_step > 0):
                # update to wandb
                wandb_cache_file_append(self.log_update_cache_discriminator, exp_dir)
                self.log_update_cache_discriminator = []

            return d_adversarial_loss + lecam_regularization


# class VAELoss(nn.Module):
#     def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
#                  disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
#                  gen_adv_loss='hinge', reconstruction_loss='l1', reconstruction_weight=1.0, 
#                  perceptual_weight=1.0,
#                  norm="batch",kw=4, blur_ds=False, lecam=False, lecam_weight=0.001,
#                  proj_weight=0.5, 
#                  disc_semantic_type="local",  # (deprecated)choosing from "local" or "global", for Semantic discriminator
#                  use_semantic_input=False,    # (deprecated)choosing from "local" or "global", for Semantic discriminator
#                  perceptual_model="vgg",      # for perceptual loss setting
#                  gamma=15,  # (deprecated) for r3gan R1+R2 panelty, 
#                  logvar_init=0.0, kl_weight=1.0,
#     ):
#         super().__init__()
#         # discriminator loss
#         assert disc_type in ["patchgan", "stylegan", "patchgan_SeD", "patchganv2", "dinodisc", "patchvit"]
#         assert disc_loss in ["hinge", "vanilla", "non-saturating", "r3gan"]

#         self.disc_type = disc_type
#         self.disc_semantic_type = disc_semantic_type
#         self.use_semantic_input = use_semantic_input

#         self.use_direct_rec_loss = use_direct_rec_loss
#         assert not use_direct_rec_loss
#         self.kl_weight = kl_weight
#         # output log variance
#         self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

#         # specially for r3gan
#         self.gamma = gamma

#         if disc_type == "patchgan":
#             self.discriminator = PatchGANDiscriminator(
#                 input_nc=disc_in_channels, 
#                 n_layers=disc_num_layers,
#                 ndf=disc_dim,
#                 norm=norm,
#                 kw=kw,
#                 blur_ds=blur_ds
#             )
#         elif disc_type == "patchganv2":
#             self.discriminator = PatchGANDiscriminatorV2(
#                 input_nc=disc_in_channels, 
#                 n_layers=disc_num_layers,
#                 ndf=disc_dim,
#                 norm=norm,
#                 kw=kw,
#                 blur_ds=blur_ds,
#                 use_semantic_input=use_semantic_input
#             )
#         elif disc_type == "stylegan":
#             self.discriminator = StyleGANDiscriminator(
#                 input_nc=disc_in_channels, 
#                 image_size=image_size,
#             )
#         elif disc_type == "patchgan_SeD":
#             self.discriminator = PatchGANSeDiscriminatorV3(
#                 input_nc=disc_in_channels, 
#                 ndf=disc_dim,
#                 kw=kw,
#                 blur_ds=blur_ds,
#                 dhead=dhead,
#             )
#         elif disc_type == "dinodisc":
#             aug_prob = 1.0
#             self.discriminator = DINODiscriminator(norm_type="bn")  # default 224 otherwise crop
#             self.daug = DiffAug(prob=aug_prob, cutout=0.2)
#         elif disc_type == "patchvit":
#             self.discriminator = PatchViTDiscriminator(
#                 model_size="base",
#             )
#         else:
#             raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")

#         if disc_loss == "hinge":
#             self.disc_loss = hinge_d_loss
#         elif disc_loss == "vanilla":
#             self.disc_loss = vanilla_d_loss
#         elif disc_loss == "non-saturating":
#             self.disc_loss = non_saturating_d_loss
#         elif disc_loss == "r3gan":
#             self.disc_loss = r3gan_d_loss
#         else:
#             raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        
#         self.disc_loss_type = disc_loss
#         self.disc_type = disc_type
#         self.gen_adv_loss_type = gen_adv_loss

#         self.discriminator_iter_start = disc_start
#         self.disc_weight = disc_weight
#         self.disc_adaptive_weight = disc_adaptive_weight

#         self.proj_weight = proj_weight

#         assert gen_adv_loss in ["hinge", "non-saturating", "r3gan"]
#         # gen_adv_loss
#         if gen_adv_loss == "hinge":
#             self.gen_adv_loss = hinge_gen_loss
#         elif gen_adv_loss == "non-saturating":
#             self.gen_adv_loss = non_saturating_gen_loss
#         elif gen_adv_loss == "r3gan":
#             self.gen_adv_loss = r3gan_gen_loss
#         else:
#             raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

#         # perceptual loss
#         if perceptual_model == "resent50":
#             self.perceptual_loss = ResNet50ImgSim().eval()
#         elif perceptual_model == "dinov2-s":
#             self.perceptual_loss = DinoV2ImgSim().eval()
#         elif perceptual_model == "vgg":
#             self.perceptual_loss = LPIPS().eval()
#         else:
#             raise ValueError(f"Unknown perceptual model '{perceptual_model}'.")
#         self.perceptual_weight = perceptual_weight

#         # reconstruction loss
#         if reconstruction_loss == "l1":
#             self.rec_loss = F.l1_loss
#         elif reconstruction_loss == "l2":
#             self.rec_loss = F.mse_loss
#         else:
#             raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
#         self.rec_weight = reconstruction_weight

#         # iteration to stop using auxiliary loss
#         self.aux_loss_end = aux_loss_end

#         # Special config for logging
#         self.log_update_cache_generator = []
#         self.log_update_cache_discriminator = []

#         self.log_update_cache_multi_level = {}

#         self.lecam = lecam
#         if self.lecam:
#             self.ema_logits = ema_losses(start_itr=self.discriminator_iter_start + 1000)
#         else:
#             self.ema_logits = None
#         self.lecam_weight = lecam_weight


#     def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
#         nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
#         g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
#         print("nll_grads norm:", torch.norm(nll_grads))
#         print("g_grads norm:", torch.norm(g_grads))

#         d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
#         d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
#         return d_weight.detach()

#     def forward(self, 
#                 inputs, reconstructions, posteriors,
#                 optimizer_idx, global_step, exp_dir, last_layer=None, 
#                 logger=None, log_every=100, ckpt_every=500, num_en_q_level=None, causal_type=None,
#                 check_nan_loss=True, inner_feat=None, sem_enc_feat=None,
#                 weights=None
#                 ):
#         rank = dist.get_rank() 
#         node_rank = int(os.environ.get('NODE_RANK', 0))
#         B, C, H, W = inputs.shape
#         # generator update
#         if optimizer_idx == 0:
#             # reconstruction loss
#             rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

#             # perceptual loss
#             p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
#             p_loss = torch.mean(p_loss)

#             # discriminator loss
#             disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
#             if global_step < self.discriminator_iter_start:
#                 logits_fake = 0
#                 generator_adv_loss = 0
#             else:
#                 if "SeD" in self.disc_type or self.use_semantic_input:
#                     assert sem_enc_feat is not None, "Semantic discriminator needs sem_enc_feat as input"
#                     if self.disc_semantic_type == "global":
#                         global_sem_enc_feat = sem_enc_feat.detach()
#                         if sem_enc_feat.dim() == 3:
#                             # B (H W) C -> B 1 C -> B (H W) C
#                             global_sem_enc_feat = global_sem_enc_feat.mean(dim=1, keepdim=True)
#                             global_sem_enc_feat = global_sem_enc_feat.expand(-1, sem_enc_feat.shape[1], -1)
#                         else:
#                             # B C H W -> B C 1 1 -> B C H W
#                             global_sem_enc_feat = global_sem_enc_feat.mean(dim=(2, 3), keepdim=True)
#                             global_sem_enc_feat = global_sem_enc_feat.expand(-1, -1, sem_enc_feat.shape[2], sem_enc_feat.shape[3])

#                         disc_sem_feat = global_sem_enc_feat
#                     elif self.disc_semantic_type == "local":
#                         disc_sem_feat = sem_enc_feat.detach()
#                     else:
#                         raise ValueError("disc_semantic_type must be global or local")

#                     logits_fake = self.discriminator(reconstructions.contiguous().detach(), disc_sem_feat)
#                 elif self.disc_type == "dinodisc":
#                     fade_blur_schedule = 0
#                     logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous(), fade_blur_schedule))
#                 else:
#                     logits_fake = self.discriminator(reconstructions.contiguous())
#                 if self.gen_adv_loss_type == "r3gan":
#                     logits_real = self.discriminator(inputs.contiguous())
#                     generator_adv_loss = self.gen_adv_loss(logits_real, logits_fake)
#                 else:
#                     generator_adv_loss = self.gen_adv_loss(logits_fake)
                
#             if self.disc_adaptive_weight:
#                 null_loss = self.rec_weight * rec_loss + self.perceptual_weight * p_loss
#                 disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
#             else:
#                 disc_adaptive_weight = 1
            
            
#             # compute semantic distill loss
#             # projection loss
#             proj_loss = 0.
#             if inner_feat is not None and sem_enc_feat is not None:
#                 assert inner_feat.shape == sem_enc_feat.shape, f"inner_feat.shape: {inner_feat.shape}, sem_enc_feat.shape: {sem_enc_feat.shape}"
#                 bsz = inner_feat.shape[0]
#                 # TODO: the for loop is ugly, change to array calculation
#                 for j, (z_j, z_tilde_j) in enumerate(zip(inner_feat, sem_enc_feat)):
#                     z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
#                     z_j = torch.nn.functional.normalize(z_j, dim=-1) 
#                     proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
#                 proj_loss /= bsz


#             rec_loss = self.rec_weight * rec_loss + self.perceptual_weight * p_loss 
#             nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
#             weighted_nll_loss = nll_loss
#             if weights is not None:
#                 weighted_nll_loss = weights*nll_loss
#             # weighted_nll_loss = torch.sum(weighted_nll_loss) / B
#             # nll_loss = torch.sum(nll_loss) / B 
#             kl_loss = posteriors.kl()
#             kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

#             loss = weighted_nll_loss + self.kl_weight * kl_loss + \
#                 disc_adaptive_weight * disc_weight * generator_adv_loss  + \
#                 self.proj_weight * proj_loss

#             # BUG!
#             # loss = self.rec_weight * rec_loss + \
#             #     self.perceptual_weight * p_loss + \
#             #     disc_adaptive_weight * disc_weight * generator_adv_loss  + \
#             #     self.proj_weight * proj_loss

#             if check_nan_loss:
#                 if torch.isnan(loss).any():
#                     error_info = ""
#                     rec_loss = self.rec_weight * rec_loss
#                     p_loss = self.perceptual_weight * p_loss
#                     generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
#                     # check any nan in reconstruction input
#                     if torch.isnan(inputs).any():
#                         error_info += "input contains nan\n"
#                     if torch.isnan(reconstructions).any():
#                         error_info += "reconstructions contains nan\n"
#                     error_info += (f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
#                                 f"generator_adv_loss: {generator_adv_loss:.4f}, "
#                                 f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}\n")
#                     get_ip_cmd = """hostname -I | awk '{split($0, a, " "); print a[1]}'"""
#                     ip_addr = os.popen(get_ip_cmd).read().strip()
#                     error_info += f"ip: {ip_addr}\n"
#                     error_info += f"current iteration:{global_step}"
#                     raise RuntimeError(error_info)
            
#             if rank == 0 and node_rank == 0 and (global_step % log_every == 0):
#                 rec_loss = self.rec_weight * rec_loss
#                 p_loss = self.perceptual_weight * p_loss
#                 generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
#                 logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
#                             f"generator_adv_loss: {generator_adv_loss:.4f}, " 
#                             f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}, "
#                             f"proj_loss: {proj_loss:.4f}, "
#                             )

#                 # update to wandb
#                 update_info = {
#                     "(Generator)rec_loss": rec_loss,
#                     "(Generator)perceptual_loss": p_loss,
#                     "(Generator)generator_adv_loss": generator_adv_loss,
#                     "(Generator)disc_adaptive_weight": disc_adaptive_weight,
#                     "(Generator)disc_weight": disc_weight,
#                     "iteration": global_step,
#                     "(Generator)proj_loss": proj_loss,
#                 }

#                 if num_en_q_level is not None and causal_type == "per-level":
#                     multi_level_update_info = {
#                         key: np.mean(value)
#                         for key, value in self.log_update_cache_multi_level.items()
#                     }
#                     update_info.update(multi_level_update_info)

#                     multi_level_info = "\n".join(
#                         [f"{key}: {np.mean(value):.4f}, " for key, value in multi_level_update_info.items()]
#                     )
#                     logger.info(multi_level_info)

#                 self.log_update_cache_multi_level = {}
#                 self.log_update_cache_generator.append(update_info)


#             if rank == 0 and node_rank == 0 and (global_step % ckpt_every == 0 and global_step > 0):
#                 # update to wandb
#                 wandb_cache_file_append(self.log_update_cache_generator, exp_dir)
#                 self.log_update_cache_generator = []

#             return loss

#         # discriminator update
#         if optimizer_idx == 1:
#             if global_step < self.discriminator_iter_start:
#                 return 0.0

#             if "SeD" in self.disc_type or self.use_semantic_input:
#                 # Semantic discriminator needs sem_enc_feat as input
#                 assert sem_enc_feat is not None, "Semantic discriminator needs sem_enc_feat as input"

#                 # use only global feature 
#                 # B C H W -> B C 1 1 -> B C H W
#                 if self.disc_semantic_type == "global":
#                     global_sem_enc_feat = sem_enc_feat.detach()
#                     if sem_enc_feat.dim() == 3:
#                         # B (H W) C -> B 1 C -> B (H W) C
#                         global_sem_enc_feat = global_sem_enc_feat.mean(dim=1, keepdim=True)
#                         global_sem_enc_feat = global_sem_enc_feat.expand(-1, sem_enc_feat.shape[1], -1)
#                     else:
#                         assert sem_enc_feat.dim() == 4, "sem_enc_feat.dim() must be 3 or 4"
#                         # B C H W -> B C 1 1 -> B C H W
#                         global_sem_enc_feat = global_sem_enc_feat.mean(dim=(2, 3), keepdim=True)
#                         global_sem_enc_feat = global_sem_enc_feat.expand(-1, -1, sem_enc_feat.shape[2], sem_enc_feat.shape[3])
#                     disc_sem_feat = global_sem_enc_feat
#                 elif self.disc_semantic_type == "local":
#                     disc_sem_feat = sem_enc_feat.detach()
#                 else:
#                     raise ValueError("disc_semantic_type must be global or local")


#                 # TODO: check if this detach() is leading to problems
#                 logits_real = self.discriminator(inputs.contiguous().detach(), disc_sem_feat)
#                 logits_fake = self.discriminator(reconstructions.contiguous().detach(), disc_sem_feat)
#             elif self.disc_type == "dinodisc":
#                 fade_blur_schedule = 0
#                 # add blur since disc is too strong
#                 logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous().detach(), fade_blur_schedule))
#                 logits_real = self.discriminator(self.daug.aug(inputs.contiguous().detach(), fade_blur_schedule))
#             elif self.disc_loss_type == "r3gan":
#                 samples_real = inputs.contiguous().detach().clone().requires_grad_(True)
#                 samples_fake = reconstructions.contiguous().detach().clone().requires_grad_(True)
#                 logits_real = self.discriminator(samples_real)
#                 logits_fake = self.discriminator(samples_fake)
#             else:
#                 logits_real = self.discriminator(inputs.contiguous().detach())
#                 logits_fake = self.discriminator(reconstructions.contiguous().detach())

#             disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)

#             if self.disc_loss_type == "r3gan":
#                 d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake, 
#                                                                   samples_fake=samples_fake,
#                                                                   samples_real=samples_real,
#                                                                   gamma=self.gamma,
#                                                                   ema=self.ema_logits, iter=global_step,
#                                                                   )
#                 lecam_regularization = 0.0
#             else:
#                 d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake, ema=self.ema_logits, iter=global_step)
#                 lecam_regularization = self.lecam_weight * lecam_reg(logits_real, logits_fake, ema=self.ema_logits) \
#                                         if self.lecam and global_step > self.discriminator_iter_start else 0.0

#             if global_step % log_every == 0:
#                 logits_real = logits_real.detach().mean()
#                 logits_fake = logits_fake.detach().mean()
#                 logger.info(f"(Discriminator) " 
#                             f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
#                             f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f},")

#                 update_info = {
#                     "(Discriminator)discriminator_adv_loss": d_adversarial_loss,
#                     "(Discriminator)disc_weight": disc_weight,
#                     "(Discriminator)logits_real": logits_real,
#                     "(Discriminator)logits_fake": logits_fake,
#                     "iteration": global_step,
#                 }
#                 self.log_update_cache_discriminator.append(update_info)

#             rank = dist.get_rank()             
#             if rank == 0 and (global_step % ckpt_every == 0 and global_step > 0):
#                 # update to wandb
#                 wandb_cache_file_append(self.log_update_cache_discriminator, exp_dir)
#                 self.log_update_cache_discriminator = []

#             return d_adversarial_loss + lecam_regularization