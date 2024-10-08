# __init__.py

# Contains code based on https://github.com/tamarott/SinGAN
import torch

from generator import Level_GeneratorConcatSkip2CleanAdd
from discriminator import Level_WDiscriminator


def weights_init(m):
    """ Initialize weights for Conv and Norm Layers. """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Norm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_models(opt, use_softmax=True):
    """ Initialize Generator and Discriminator. """
    # Generator initialization:
    G = Level_GeneratorConcatSkip2CleanAdd(opt, use_softmax=use_softmax).to(opt.device)
    G.apply(weights_init)
    if opt.netG != "":
        G.load_state_dict(torch.load(opt.netG))
    print(G)

    # Discriminator initialization:
    D = Level_WDiscriminator(opt).to(opt.device)
    D.apply(weights_init)
    if opt.netD != "":
        D.load_state_dict(torch.load(opt.netD))
    print(D)

    return D, G
