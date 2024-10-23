import os
import json
from loguru import logger
import numpy as np
import torch
import h5py
import re
from torch.utils.data.dataset import Dataset


class WorldGANDataset(Dataset):
    def __init__(self, input_directory, tok2block_filepath: str):
        super().__init__()
        self.input_directory = input_directory
        self.files = os.listdir(input_directory)
        with open(tok2block_filepath, 'r') as file:
            self.tok2block = json.load(file)

    def __getitem__(self, index):
        # Abort forward pass with context window too small here 
        h5_filename = self.files[index]
        level, uniques, props = self.read_level_from_file(h5_filename, block2repr=None, repr_type=None)
        return level 
    
    def read_level_from_file(self, h5_filename, coords, block2repr, repr_type):
        """ coords is ((y0,yend), (z0,zend), (x0,xend)) """
        build_array = None 
        with h5py.File(os.path.join(self.input_directory, h5_filename), "r") as file:
            keys = file.keys()
            if len(keys) == 0:
                logger.info("%s failed loading: no keys." % h5_filename)
                exit(1)
            else: 
                build_array = np.array(file[list(keys)[0]][()], dtype=np.int32)
                logger.info("%s loaded." % h5_filename)  
                x_max, y_max, z_max = build_array.shape
                coords = ((0, x_max), (0, y_max), (0, z_max))
        print("coords are ", coords)
        if repr_type == "block2vec":
            # Read Representations
            uniques = [u for u in block2repr.keys()]
            props = [None for _ in range(len(uniques))]
            dim = len(block2repr[uniques[0]])  # all are the same size

            level = torch.zeros(
                (1, dim, coords[0][1] - coords[0][0], coords[1][1] - coords[1][0], coords[2][1] - coords[2][0]))
        else:
            uniques = []
            props = []
            # Init level with zeros
            level = torch.zeros((coords[0][1] - coords[0][0], coords[1][1] - coords[1][0], coords[2][1] - coords[2][0]))

        # Inject
        pattern = r'\[(.*?)\]'
        for j in range(coords[0][0], coords[0][1]):
            for k in range(coords[1][0], coords[1][1]):
                for l in range(coords[2][0], coords[2][1]):
                    block_tok = build_array[(j, k, l)]
                    if (str(block_tok) == "4000"): 
                        block_tok = 3714
                    block_name = re.sub(pattern, '', self.tok2block[str(block_tok)])
                    #print("block is ", block_tok, ": ", block_name) 

                    if repr_type == "block2vec":
                        level[0, :, j - coords[0][0], k - coords[1][0], l - coords[2][0]] = block2repr[block_name]
                        if not props[uniques.index(block_name)]:
                            props[uniques.index(block_name)] = re.search(pattern, block_name)

                    else:
                        if block_name not in uniques:
                            uniques.append(block_name)
                            props.append(re.search(pattern, block_name))
                        level[j - coords[0][0], k - coords[1][0], l - coords[2][0]] = uniques.index(block_name)
     
        if repr_type == "block2vec":
            # For block2vec, directly use representation vectors
            oh_level = level
        else:
            # Else we need the one hot encoding
            oh_level = torch.zeros((1, len(uniques),) + level.shape)
            for i, tok in enumerate(uniques):
                oh_level[0, i] = (level == i)

            if repr_type == "autoencoder":
                # Autoencoder takes a one hot encoded level as input to get representations
                device = next(block2repr["encoder"].parameters()).device
                oh_level = block2repr["encoder"](oh_level.to(device))
                if isinstance(oh_level, tuple):
                    oh_level = oh_level[0].detach()
                else:
                    oh_level = oh_level.detach()

        return oh_level, uniques, props

    def __len__(self):
        return self.files.shape[0]
