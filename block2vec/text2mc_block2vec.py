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
        coordinates = [
            (x, y, z)
            for x, y, z in product(
                range(self.x_lims[0], self.x_lims[1] - 1),
                range(self.y_lims[0], self.y_lims[1] - 1),
                range(self.z_lims[0], self.z_lims[1] - 1),
            )
        ]

        logger.info("Collecting {} blocks", len(self))

        # Collect counts for each block
        for block_tok in tqdm([self.get_block_at(*coord) for coord in coordinates]):
            self.block_frequency[block_tok] += 1

        logger.info(
            "Found the following blocks {blocks}", blocks=dict(self.block_frequency)
        )

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
        neighbor_coords = [
            (x + x_diff, y + y_diff, z + z_diff)
            for x_diff, y_diff, z_diff in product(
                list(range(-self.neighbor_radius, self.neighbor_radius + 1)), repeat=3
            )
            if x_diff != 0 or y_diff != 0 or z_diff != 0
        ]

        return [self.get_block_at(*coord) for coord in neighbor_coords]

    """ Visalization of target and neighbor block context for documentation """

    def plot_coords(self, target_coord, context_coords):
        x, y, z = zip(*context_coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

        ax.scatter(x, y, z, color="red")
        ax.scatter(*target_coord, color="blue")

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")

        plt.show()

    def __len__(self):
        return self.x_dim * self.y_dim * self.z_dim


import math
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple
import json
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from fuzzywuzzy import process
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib import rcParams
from block2vec_dataset import Block2VecDataset
from image_annotations_3d import ImageAnnotations3D
from skip_gram_model import SkipGramModel
from sklearn.metrics import ConfusionMatrixDisplay
from tap import Tap
from torch.utils.data import DataLoader
import umap

""" Default arguments for Block2Vec """


class Block2VecArgs(Tap):
    emb_dimension: int = 32
    epochs: int = 30
    batch_size: int = 256
    initial_lr: float = 1e-3
    neighbor_radius: int = 1

    output_path: str = os.path.join("output", "block2vec")
    token_to_block_filename: str = "tok_to_block.json"
    textures_directory: str = os.path.join("textures")
    embeddings_txt_filename: str = "embeddings.txt"
    embeddings_npy_filename: str = "embeddings.npy"
    embeddings_pkl_filename: str = "representations.pkl"
    embeddings_scatterplot_filename: str = "scatter_3d.png"
    embeddings_dist_matrix_filename: str = "dist_matrix.png"


class Block2Vec(pl.LightningModule):

    def __init__(self, build_paths, **kwargs):
        super().__init__()
        self.args: Block2VecArgs = Block2VecArgs().from_dict(kwargs)
        self.save_hyperparameters()
        self.build_paths = build_paths
        self.dataset = Block2VecDataset(
            self.build_paths,
            neighbor_radius=self.args.neighbor_radius,
        )
        with open(self.args.token_to_block_filename, "r") as file:
            self.tok2block = json.load(file)

        # Convert string keys back to integers
        self.tok2block = {int(key): value for key, value in self.tok2block.items()}

        self.emb_size = len(self.dataset.block2idx)
        self.model = SkipGramModel(self.emb_size, self.args.emb_dimension)
        self.textures = dict()
        self.learning_rate = self.args.initial_lr

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        loss = self.forward(*batch)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            math.ceil(len(self.dataset) / self.args.batch_size) * self.args.epochs,
        )
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            # This needs to be fixed.
            # num_workers=os.cpu_count() or 1,
            num_workers=0,
        )

    """ Plot and save embeddings at end of each training epoch """

    def on_train_epoch_end(self):
        embedding_dict = self.save_embedding(
            self.dataset.idx2block, self.args.output_path
        )
        self.create_confusion_matrix(self.dataset.idx2block, self.args.output_path)
        self.plot_embeddings(embedding_dict, self.args.output_path)

    """ Reads texture .png file for a given token """

    def read_texture(self, block: str):
        if block not in self.textures and block != "air":
            texture_candidates = Path(self.args.textures_directory).glob("*.png")

            match = process.extractOne(block, texture_candidates)

            if match is not None:
                logger.info("Matches {} with {} texture file", block, match[0])
                self.textures[block] = plt.imread(match[0])

        if block not in self.textures:
            self.textures[block] = np.ones(shape=[16, 16, 3])

        return self.textures[block]

    """ Save generated embeddings to a file """

    def save_embedding(self, id2block: Dict[int, str], output_path: str):
        embeddings = self.model.target_embeddings.weight
        # embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        embeddings = embeddings.cpu().data.numpy()
        embedding_dict = {}

        with open(
            os.path.join(output_path, self.args.embeddings_txt_filename), "w"
        ) as f:

            f.write("%d %d\n" % (len(id2block), self.args.emb_dimension))
            for wid, w in id2block.items():
                e = " ".join(map(lambda x: str(x), embeddings[wid]))
                embedding_dict[self.tok2block[w]] = torch.from_numpy(embeddings[wid])
                f.write("%s %s\n" % (self.tok2block[w], e))

        np.save(
            os.path.join(output_path, self.args.embeddings_npy_filename), embeddings
        )

        with open(
            os.path.join(output_path, self.args.embeddings_pkl_filename), "wb"
        ) as f:
            pickle.dump(embedding_dict, f)

        return embedding_dict

    """ Plot generated block embeddings """

    def plot_embeddings(self, embedding_dict: Dict[str, np.ndarray], output_path: str):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        legend = [label for label in embedding_dict.keys()]
        texture_imgs = [self.read_texture(block) for block in legend]
        embeddings = torch.stack(list(embedding_dict.values())).numpy()
        if embeddings.shape[-1] != 3:
            embeddings_3d = umap.UMAP(
                n_neighbors=2, min_dist=0.3, n_components=3
            ).fit_transform(embeddings)
        else:
            embeddings_3d = embeddings
        for embedding in embeddings_3d:
            ax.scatter(*embedding, alpha=0)
        ia = ImageAnnotations3D(embeddings_3d, texture_imgs, legend, ax, fig)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_path, self.args.embeddings_scatterplot_filename),
            dpi=300,
        )
        plt.close("all")

    def create_confusion_matrix(self, id2block: Dict[int, str], output_path: str):
        rcParams.update({"font.size": 6})
        names = []
        dists = np.zeros((len(id2block), len(id2block)))
        for i, b1 in id2block.items():
            names.append(b1)
            for j, b2 in id2block.items():
                dists[i, j] = F.mse_loss(
                    self.model.target_embeddings.weight.data[i],
                    self.model.target_embeddings.weight.data[j],
                )
        confusion_display = ConfusionMatrixDisplay(dists, display_labels=names)
        confusion_display.plot(include_values=False, xticks_rotation="vertical")
        confusion_display.ax_.set_xlabel("")
        confusion_display.ax_.set_ylabel("")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_path, self.args.embeddings_dist_matrix_filename)
        )
        plt.close()
