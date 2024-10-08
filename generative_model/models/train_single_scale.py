# train_single_scale.py

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from models import calc_gradient_penalty, reset_grads


def train_single_scale(D, G, reals, generators, noise_maps, input_from_prev_scale,
                       noise_amplitudes, opt, optimizerD, optimizerG):
    """Train the models at a single scale."""
    real = reals[0].to(opt.device)
    fixed_noise = torch.randn_like(real).to(opt.device)

    for epoch in tqdm(range(opt.niter)):
        # Update Discriminator
        for _ in range(opt.Dsteps):
            D.zero_grad()
            output_real = D(real)
            fake = G(fixed_noise, input_from_prev_scale)
            output_fake = D(fake.detach())
            errD = -torch.mean(output_real) + torch.mean(output_fake)
            errD.backward()
            optimizerD.step()

            # Gradient penalty
            gradient_penalty = calc_gradient_penalty(D, real, fake.detach(), opt.lambda_grad, opt.device)
            gradient_penalty.backward()
            optimizerD.step()

        # Update Generator
        for _ in range(opt.Gsteps):
            G.zero_grad()
            fake = G(fixed_noise, input_from_prev_scale)
            output = D(fake)
            errG = -torch.mean(output)
            errG.backward()
            optimizerG.step()

    return fixed_noise, fake.detach(), G
