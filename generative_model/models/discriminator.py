# discriminator.py

# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn

from conv_block import ConvBlock


class Level_WDiscriminator(nn.Module):
    """ Patch-based 3D Discriminator. """
    def __init__(self, opt):
        super().__init__()
        N = int(opt.nfc)
        dim = 3  # Working with 3D data
        kernel = tuple(opt.ker_size for _ in range(dim))
        self.head = ConvBlock(opt.nc_current, N, kernel, padd=opt.padd_size, stride=1, dim=dim)
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, kernel, padd=opt.padd_size, stride=1, dim=dim)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, kernel, padd=opt.padd_size, stride=1, dim=dim)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        self.tail = nn.Conv3d(N, 1, kernel_size=kernel, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
