# train.py

# Code inspired by https://github.com/tamarott/SinGAN
import os
import torch
import torch.optim as optim
from tqdm import tqdm

from models import init_models, reset_grads
from train_single_scale import train_single_scale


def train(real, opt):
    """ Wrapper function for training. Calculates necessary scales then calls train_single_scale on each. """
    generators = []
    noise_maps = []
    noise_amplitudes = []

    reals = [real]  # Since we're not doing multi-scale training in this example

    input_from_prev_scale = torch.zeros_like(real, device=opt.device)

    opt.stop_scale = len(reals)

    os.makedirs(f"{opt.out_}/state_dicts", exist_ok=True)

    # Training Loop
    for current_scale in range(0, opt.stop_scale):
        opt.outf = f"{opt.out_}/{current_scale}"
        os.makedirs(opt.outf, exist_ok=True)

        opt.nc_current = real.shape[1]

        use_softmax = False  # Adjust as needed

        # Initialize models
        D, G = init_models(opt, use_softmax)

        # Define optimizers
        optimizerD = optim.Adam(D.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        # Actually train the current scale
        z_opt, input_from_prev_scale, G = train_single_scale(D, G, reals, generators, noise_maps,
                                                             input_from_prev_scale, noise_amplitudes, opt,
                                                             optimizerD, optimizerG)

        # Reset grads and save current scale
        G = reset_grads(G, False)
        G.eval()
        D = reset_grads(D, False)
        D.eval()

        generators.append(G)
        noise_maps.append(z_opt)
        noise_amplitudes.append(opt.noise_amp)

        torch.save(noise_maps, f"{opt.out_}/noise_maps.pth")
        torch.save(generators, f"{opt.out_}/generators.pth")
        torch.save(reals, f"{opt.out_}/reals.pth")
        torch.save(noise_amplitudes, f"{opt.out_}/noise_amplitudes.pth")

        torch.save(G.state_dict(), f"{opt.out_}/state_dicts/G_{current_scale}.pth")
        torch.save(D.state_dict(), f"{opt.out_}/state_dicts/D_{current_scale}.pth")

        del D, G

    return generators, noise_maps, reals, noise_amplitudes
