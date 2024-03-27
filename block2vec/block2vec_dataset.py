import os
from collections import defaultdict
from itertools import product
from typing import Tuple
from loguru import logger
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset


class Block2VecDataset(Dataset):

    def __init__(self, build, neighbor_radius: int = 1):
        super().__init__()
        self.build = build
        self.x_lims, self.y_lims, self.z_lims = self.get_build_dimensions()
        padding = 2 * neighbor_radius  # one token on each side
        self.x_dim = self.x_lims[1] - self.x_lims[0] - padding
        self.y_dim = self.y_lims[1] - self.y_lims[0] - padding
        self.z_dim = self.z_lims[1] - self.z_lims[0] - padding
        logger.info("Cutting {} x {} x {} volume", self.x_dim, self.y_dim, self.z_dim)
        
        self.neighbor_radius = neighbor_radius
        self._read_blocks()
        self._init_discards()

    """ Store discard probabilities for each token """
    def _init_discards(self):
        threshold = 0.001
        token_frequencies = list(self.block_frequency.values())
        freq = np.array(token_frequencies) / sum(token_frequencies)
        self.discards = 1.0 - (np.sqrt(freq / threshold) + 1) * (threshold / freq)

    """ Read dimensions of the build tensor """
    def get_build_dimensions(self):
        # Read tensor shape to determine size in each dimension  
        x_max, y_max, z_max = self.build.shape

        # Format tuples for the limits in each dimension
        x_lims = [0, x_max - 1] 
        y_lims = [0, y_max - 1]
        z_lims = [0, z_max - 1]

        return x_lims, y_lims, z_lims
    

    """ Read the blocks of the build """
    def _read_blocks(self):
        self.block_frequency = defaultdict(int)
        
        # Iterate over build coordinates, from (0, full build size) for each dimension
        coordinates = [(x, y, z) for x, y, z in product(range(self.x_lims[0], self.x_lims[1] - 1),
                                                range(self.y_lims[0], self.y_lims[1] - 1),
                                                range(self.z_lims[0], self.z_lims[1] - 1))]

        logger.info("Collecting {} blocks", len(self))
        
        # Collect counts for each block 
        for block_tok in tqdm([self.get_block_at(*coord) for coord in coordinates]):
            self.block_frequency[block_tok] += 1

        logger.info("Found the following blocks {blocks}", blocks=dict(self.block_frequency))

        # Quick reference from block tokens to index and vice versa
        self.block2idx = dict()
        self.idx2block = dict()

        for tok in self.block_frequency.keys():
            block_idx = len(self.block2idx)
            self.block2idx[tok] = block_idx
            self.idx2block[block_idx] = tok

    
    """ Returns target and context blocks """
    def __getitem__(self, index):
        coords = self._idx_to_coords(index)
        block = self.get_block_at(*coords)
        target = self.block2idx[block]
        if np.random.rand() < self.discards[target]:
            return self.__getitem__(np.random.randint(self.__len__()))
        neighbor_blocks = self.get_block_neighbors(*coords)
        context = np.array([self.block2idx[n] for n in neighbor_blocks])
        return target, context

    def _idx_to_coords(self, index):
        z = index % (self.z_dim + 1)
        y = int(((index - z) / (self.z_dim + 1)) % (self.y_dim + 1))
        x = int(((index - z) / (self.z_dim + 1) - y) / (self.y_dim + 1))
        x += self.x_lims[0] + self.neighbor_radius
        y += self.y_lims[0] + self.neighbor_radius
        z += self.z_lims[0] + self.neighbor_radius
        return x, y, z
    
    """ Gets specific block from build tensor at a given coordinate """
    def get_block_at(self, x, y, z):
        block_tok = self.build[x, y, z].item()
        return block_tok 

    """ Gets neighboring blocks from build tensor for a block at given coordinate """
    def get_block_neighbors(self, x, y, z):
        neighbor_coords = [(x + x_diff, y + y_diff, z + z_diff) for x_diff, y_diff, z_diff in product(list(
            range(-self.neighbor_radius, self.neighbor_radius + 1)), repeat=3) if x_diff != 0 or y_diff != 0 or z_diff != 0]

        return [self.get_block_at(*coord) for coord in neighbor_coords]
    
    """ Visalization of target and neighbor block context for documentation """
    def plot_coords(self, target_coord, context_coords): 
        x, y, z = zip(*context_coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
       
        ax.scatter(x, y, z, color='red')
        ax.scatter(*target_coord, color='blue')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()

    def __len__(self):
        return self.x_dim * self.y_dim * self.z_dim
