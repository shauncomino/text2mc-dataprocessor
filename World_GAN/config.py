# Code based on https://github.com/tamarott/SinGAN
import argparse
import random
from typing import List, Literal, Optional, Union
import numpy as np
import torch
from torch import cuda
from tap import Tap
import pickle
import json
import os 

from utils import set_seed, load_pkl


class Config(Tap):
    tok2block_filepath: str = os.path.join("tok2block.json")
    block2tok_filepath: str = os.path.join("block2tok.json")
    # game: Literal["minecraft"] = "minecraft"  # Which game is to be used? ONLY MINECRAFT
    not_cuda: bool = True  # disables cuda 
    netG: str = ""  # path to netG (to continue training)
    netD: str = ""  # path to netD (to continue training)
    manualSeed: Optional[int] = None
    out: str = os.path.join("output") # output directory
    input_dir: str = os.path.join("input", "minecraft")  # input directory
    input_name: str = "batch_101_2618.h5"  # input level filename
    # input level names (if multiple inputs are used)
    input_names: List[str] = ["lvl_1-1.txt", "lvl_1-2.txt"]
    # use mulitple inputs for training (use --input-names instead of --input-name)
    use_multiple_inputs: bool = False # Original: false 

    # if minecraft is used, which coords are used from the world? Which world do we save to?
    input_area_name: str = "ruins"  # needs to be a string from the coord dictionary in input folder
    output_dir: str = os.path.join("output", "minecraft")  # folder with worlds
    output_name: str = "Gen_Empty_World"  # name of the world to generate in
    sub_coords: List[float] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # defines which coords of the full coord are are
    # taken (if float -> percentage, if int -> absolute)

    nfc: int = 64  # number of filters for conv layers
    ker_size: int = 3  # kernel size for conv layers
    num_layer: int = 3  # number of layers
    scales: List[float] = [0.75, 0.5, 0.25]  # Scales descending (< 1 and > 0)
    noise_update: float = 0.1  # additive noise weight
    # use reflection padding? (makes edges random)
    pad_with_noise: bool = False
    #ORIGINAL: MUST CHANGE BACK!!! 
    # niter: int = 4000  # number of epochs to train per scale
    niter: int = 10  
    gamma: float = 0.1  # scheduler gamma
    lr_g: float = 0.0005  # generator learning rate
    lr_d: float = 0.0005  # discriminator learning rate
    beta1: float = 0.5  # optimizer beta
    Gsteps: int = 3  # generator inner steps
    Dsteps: int = 3  # discriminator inner steps
    lambda_grad: float = 0.1  # gradient penalty weight
    alpha: int = 100  # reconstruction loss weight
    token_list: List[str] = ['!', '#', '-', '1', '@', 'C', 'S',
                             'U', 'X', 'g', 'k', 't']  # default list of 1-1

    repr_type: str = "block2vec"  # Which representation type to use, currently [None, block2vec, autoencoder]

    def __init__(self,
                   underscores_to_dashes: bool = False,
                 explicit_bool: bool = False,
                 *args,
                 **kwargs,):
        super().__init__(underscores_to_dashes, explicit_bool, args, kwargs)

    def process_args(self):

        with open(self.tok2block_filepath, 'r') as f:
            self.tok2block = json.load(f)

        with open(self.block2tok_filepath, 'r') as f:
            self.block2tok = json.load(f)
        self.device = torch.device("cpu" if self.not_cuda else "cuda:0")
        if cuda.is_available() and self.not_cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")

        if self.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.manualSeed)
        set_seed(self.manualSeed)

        # Defaults for other namespace values that will be overwritten during runtime
        self.nc_current = 12  # n tokens of level 1-1
        #if not hasattr(self, "out_"):
            #self.out_ = "%s/%s/" % (self.out, self.input_name[:-4])
        self.outf = "0"  # changes with each scale trained
        # number of scales is implicitly defined
        self.num_scales = len(self.scales)
        self.noise_amp = 1.0  # noise amp for lowest scale always starts at 1
        self.seed_road = None  # for mario kart seed roads after training
        # which scale to stop on - usually always last scale defined
        self.stop_scale = self.num_scales + 1
        coord_dict = load_pkl('primordial_coords_dict', 'input/minecraft/')
        tmp_coords = coord_dict[self.input_area_name]
        sub_coords = [(self.sub_coords[0], self.sub_coords[1]),
                      (self.sub_coords[2], self.sub_coords[3]),
                      (self.sub_coords[4], self.sub_coords[5])]
        self.coords = []
        for i, (start, end) in enumerate(sub_coords):
            curr_len = tmp_coords[i][1] - tmp_coords[i][0]
            if isinstance(start, float):
                tmp_start = curr_len * start + tmp_coords[i][0]
                tmp_end = curr_len * end + tmp_coords[i][0]
            elif isinstance(start, int):
                tmp_start = tmp_coords[i][0] + start
                tmp_end = tmp_coords[i][0] + end
            else:
                AttributeError("Unexpected type for sub_coords")
                tmp_start = tmp_coords[i][0]
                tmp_end = tmp_coords[i][1]

            self.coords.append((int(tmp_start), int(tmp_end)))
        if not self.repr_type:
            self.block2repr = None
        elif self.repr_type == "block2vec":
            self.block2repr = {}
            embeddings_pkl_path = os.path.join("..", "block2vec", "output", "block2vec", "representations.pkl")
            with open(embeddings_pkl_path, 'rb') as f:
                embeddings_dict = pickle.load(f)
                for key, value in embeddings_dict.items(): 
                    print("key is: ", key)
                    print("value is: ", value)
                    self.block2repr[key] = value 
            
        else:
            AttributeError("unexpected repr_type, use [None, block2vec, autoencoder]")
