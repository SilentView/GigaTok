# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/lqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from tokenizer.tokenizer_image.discriminator_stylegan import Discriminator as StyleGANDiscriminator

from utils.resume_log import wandb_cache_file_append



def hinge_d_loss(logits_real, logits_fake):
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


class LQLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0, perceptual_weight=1.0, aux_loss_end=0,
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight

        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight

        # iteration to stop using auxiliary loss
        self.aux_loss_end = aux_loss_end


        # Special config for logging
        self.log_update_cache_generator = []
        self.log_update_cache_discriminator = []


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, inter_loss_set, inputs, all_reconstructions, optimizer_idx, global_step, exp_dir, last_layer=None, 
                logger=None, log_every=100, ckpt_every=500):
        assert len(inter_loss_set) == 2
        assert isinstance(all_reconstructions, list)
        reconstructions, direct_reconstructions = all_reconstructions
        codebook_loss, feature_rec_loss = inter_loss_set
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())
            direct_rec_loss = self.rec_loss(inputs.contiguous(), direct_reconstructions.contiguous())

            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            direct_p_loss = self.perceptual_loss(inputs.contiguous(), direct_reconstructions.contiguous())
            p_loss = torch.mean(p_loss)
            direct_p_loss = torch.mean(direct_p_loss)

            # discriminator loss
            logits_fake = self.discriminator(reconstructions.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)
            direct_logits_fake = self.discriminator(direct_reconstructions.contiguous())
            direct_generator_adv_loss = self.gen_adv_loss(direct_logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * (rec_loss + direct_rec_loss) + self.perceptual_weight * p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            
            if global_step >= self.aux_loss_end:
                direct_rec_loss = direct_rec_loss * 0.0
                feature_rec_loss = feature_rec_loss * 0.0
                direct_p_loss = direct_p_loss * 0.0
                direct_generator_adv_loss = direct_generator_adv_loss * 0.0

            loss = self.rec_weight * (rec_loss + direct_rec_loss) + \
                self.perceptual_weight * (p_loss + direct_p_loss) + \
                disc_adaptive_weight * disc_weight * (generator_adv_loss + direct_generator_adv_loss) + \
                codebook_loss + feature_rec_loss
            
            if global_step % log_every == 0:
                rec_loss = self.rec_weight * rec_loss
                direct_rec_loss = self.rec_weight * direct_rec_loss
                p_loss = self.perceptual_weight * p_loss
                direct_p_loss = self.perceptual_weight * direct_p_loss
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                direct_generator_adv_loss = disc_adaptive_weight * disc_weight * direct_generator_adv_loss
                logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, direct_rec_loss: {direct_rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, direct_p_loss: {direct_p_loss:.4f}, "
                            f"(Generator)codebook_loss: {codebook_loss} "
                            f"feature_rec_loss: {feature_rec_loss:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, direct_generator_adv_loss: {direct_generator_adv_loss:.4f}, "
                            f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}")

                # update to wandb
                update_info = {
                    "(Generator)rec_loss": rec_loss,
                    "(Generator)perceptual_loss": p_loss,
                    "(Generator)codebook_loss": codebook_loss,
                    "(Generator)generator_adv_loss": generator_adv_loss,
                    "(Generator)disc_adaptive_weight": disc_adaptive_weight,
                    "(Generator)disc_weight": disc_weight,
                    "iteration": global_step,
                }
                self.log_update_cache_generator.append(update_info)

            rank = dist.get_rank() 
            node_rank = int(os.environ.get('NODE_RANK', 0))
            if rank == 0 and node_rank == 0 and (global_step % ckpt_every == 0 and global_step > 0):
                # update to wandb
                wandb_cache_file_append(self.log_update_cache_generator, exp_dir)
                self.log_update_cache_generator = []

            return loss

        # discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            direct_logits_fake = self.discriminator(direct_reconstructions.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)
            d_d_adversarial_loss = disc_weight * self.disc_loss(logits_real, direct_logits_fake)
            if global_step >= self.aux_loss_end:
                d_d_adversarial_loss = d_d_adversarial_loss * 0.0
            
            if global_step % log_every == 0:
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                direct_logits_fake = direct_logits_fake.detach().mean()
                logger.info(f"(Discriminator) " 
                            f"discriminator_adv_loss: {d_adversarial_loss:.4f}, d_d_adversarial_loss: {d_d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}, direct_logits_fake: {direct_logits_fake:.4f}")

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

            return d_adversarial_loss + d_d_adversarial_loss