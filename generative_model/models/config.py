# config.py

import torch


class Config:
    """Configuration class to hold all hyperparameters."""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nfc = 64  # Number of feature channels
        self.min_nfc = 32
        self.ker_size = 3  # Kernel size
        self.num_layer = 5  # Number of layers
        self.padd_size = 1  # Padding size
        self.scales = [0.9, 0.8, 0.7, 0.6, 0.5]  # Scales for multi-scale training
        self.stop_scale = len(self.scales)
        self.nc_current = None  # To be set based on data
        self.netG = ""  # Path to a pre-trained Generator model
        self.netD = ""  # Path to a pre-trained Discriminator model
        self.outf = ""  # Output folder for current scale
        self.out_ = ""  # Main output folder
        self.noise_amp = 0.1
        self.retrain = False
        self.niter = 2000  # Number of iterations per scale
        self.gamma = 0.1
        self.lr_g = 0.0005  # Learning rate for Generator
        self.lr_d = 0.0005  # Learning rate for Discriminator
        self.beta1 = 0.5
        self.Gsteps = 3  # Number of Generator updates per iteration
        self.Dsteps = 3  # Number of Discriminator updates per iteration
        self.lambda_grad = 0.1
        self.alpha = 10
        self.use_multiple_inputs = False
