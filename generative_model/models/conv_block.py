# conv_block.py

# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """ Conv block containing Conv3d, BatchNorm3d, and LeakyReLU Layers. """
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, dim=3):
        super().__init__()
        if dim == 3:
            self.add_module("conv", nn.Conv3d(in_channel, out_channel, kernel_size=ker_size,
                                              stride=stride, padding=padd))
            self.add_module("norm", nn.BatchNorm3d(out_channel))
        else:
            raise NotImplementedError("Only 3D Conv Layers are supported.")

        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
