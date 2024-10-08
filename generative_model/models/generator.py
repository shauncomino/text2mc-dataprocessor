# generator.py

# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_block import ConvBlock


class Level_GeneratorConcatSkip2CleanAdd(nn.Module):
    """ Patch-based 3D Generator with skip connections. """
    def __init__(self, opt, use_softmax=True):
        super().__init__()
        self.use_softmax = use_softmax
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

        self.tail = nn.Sequential(
            nn.Conv3d(N, opt.nc_current, kernel_size=kernel, stride=1, padding=opt.padd_size)
        )

    def forward(self, x, y, temperature=1):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        if self.use_softmax:
            x = F.softmax(x * temperature, dim=1)  # Apply temperature if needed
        # Adjust y to match the size of x
        diff_depth = y.shape[2] - x.shape[2]
        diff_height = y.shape[3] - x.shape[3]
        diff_width = y.shape[4] - x.shape[4]
        y = y[:, :, diff_depth//2:y.shape[2]-diff_depth//2, diff_height//2:y.shape[3]-diff_height//2, diff_width//2:y.shape[4]-diff_width//2]
        return x + y
