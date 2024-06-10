import os
from collections import defaultdict
from itertools import product
from typing import Tuple
import json
from loguru import logger
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset


class Block2VecDataset(Dataset):

    def __init__(self, builds, tok2block_filename: str, neighbor_radius: int):
        super().__init__()
        self.builds = builds
        self.neighbor_radius = neighbor_radius
        print(tok2block_filename)
        print(neighbor_radius)
        with open(tok2block_filename, 'r') as file:
            self.tok2block = json.load(file)
        logger.info("Received {} builds in dataset.", builds.shape[0])

    """ Store discard probabilities for each token """
    def _init_discards(self):
        threshold = 0.001
        token_frequencies = list(self.block_frequency.values())
        freq = np.array(token_frequencies) / sum(token_frequencies)
        self.discards = 1.0 - (np.sqrt(freq / threshold) + 1) * (threshold / freq)

    
    """ Return context and target blocks of build """
    def _get_coords(self, build):
        print("type is: " )
        print(type(build))
        self.block_frequency = defaultdict(int)
        
        x_max, y_max, z_max = build.shape

        # All valid coordinates
        coords = np.array([(x, y, z) for x, y, z in product(range(0, x_max),
            range(0, y_max), range(0, z_max))])
        

        # Need to come back to this: this may be faster 
        """
        x, y, z = np.arange(x_max), np.arange(y_max), np.arange(z_max)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')  # Create 3D grid
        coords_array = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        """
        
        # All valid target block coordinates
        target_coords = [(x, y, z) for x, y, z in product(range(self.neighbor_radius, x_max - self.neighbor_radius),
            range(self.neighbor_radius, y_max - self.neighbor_radius), range(self.neighbor_radius, z_max - self.neighbor_radius))]
        target_coords = np.array(target_coords) 
        
        # All valid context block coordiantes 
        context_coords = []
        for target_coord in target_coords:
            neighbors = [
                (target_coord[0] + i, target_coord[1] + j, target_coord[2] + k)
                for i, j, k in product(range(-self.neighbor_radius, self.neighbor_radius + 1), repeat=3)
                if (i, j, k) != (0, 0, 0)
            ]
            context_coords.append(np.array(neighbors))

        return coords, np.array(target_coords), np.array(context_coords)

    """ Get blocks from build """
    def _get_blocks(self, build, coords, target_coords, context_coords): 
        logger.info("Found {} blocks in build", len(coords))

        blocks = [build[coord[0], coord[1], coord[2]] for coord in coords]

        target_blocks = [build[target_coord[0], target_coord[1], target_coord[2]] for target_coord in target_coords]

        context_blocks = []
        for context_coord_list in context_coords: 
            context_blocks.append([build[context_coord[0], context_coord[1], context_coord[2]] for context_coord in context_coord_list])

        return blocks, target_blocks, context_blocks

    """ Size stuff"""
    def _store_sizes(self, blocks): 
        # Collect counts for each block 
  
        for block_tok in blocks: 
            block_name = self.tok2block[str(block_tok)]

            self.block_frequency[block_name] += 1

        logger.info("Found the following blocks {blocks}", blocks=dict(self.block_frequency))
       
        self.block2idx = dict()
        self.idx2block = dict()

        for tok in self.block_frequency.keys():
            block_idx = len(self.block2idx)
            self.block2idx[tok] = block_idx
            self.idx2block[block_idx] = tok

    """ Returns target and context block lists """
    def __getitem__(self, index):
        # Abort forward pass with context window too small here 
        build = self.builds[index]
        coords, target_coords, context_coords = self._get_coords(build) 
        blocks, target_blocks, context_blocks = self._get_blocks(build, coords, target_coords, context_coords)
        print("blocks are")
        print(blocks)

        print("target blocks are")
        print(target_blocks)

        print("context blocks are")
        print(context_blocks)
        self._store_sizes(blocks) 

        return target_blocks, context_blocks
    
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
        return self.builds.shape[0]
