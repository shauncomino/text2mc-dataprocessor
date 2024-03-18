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
    
    def __init__(self, build, **kwargs):
        super().__init__()
        self.args: Block2VecArgs = Block2VecArgs().from_dict(kwargs)
        self.save_hyperparameters()
        self.build = build 
        self.dataset = Block2VecDataset(
            self.build,
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
            math.ceil(len(self.dataset) / self.args.batch_size) *
            self.args.epochs,
        )
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            # This needs to be fixed. 
            #num_workers=os.cpu_count() or 1,
            num_workers = 0
        )

    """ Plot and save embeddings at end of each training epoch """
    def on_train_epoch_end(self):
        embedding_dict = self.save_embedding(
            self.dataset.idx2block, self.args.output_path
        )
        self.create_confusion_matrix(
            self.dataset.idx2block, self.args.output_path)
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
        
        with open(os.path.join(output_path, self.args.embeddings_txt_filename), "w") as f:

            f.write("%d %d\n" % (len(id2block), self.args.emb_dimension))
            for wid, w in id2block.items():
                e = " ".join(map(lambda x: str(x), embeddings[wid]))
                embedding_dict[self.tok2block[w]] = torch.from_numpy(embeddings[wid])
                f.write("%s %s\n" % (self.tok2block[w], e))
        
        np.save(os.path.join(output_path, self.args.embeddings_npy_filename), embeddings)
        
        with open(os.path.join(output_path, self.args.embeddings_pkl_filename), "wb") as f:
            pickle.dump(embedding_dict, f)

        return embedding_dict

    """ Plot generated block embeddings """
    def plot_embeddings(self, embedding_dict: Dict[str, np.ndarray], output_path: str):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        legend = [label
                  for label in embedding_dict.keys()]
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
        plt.savefig(os.path.join(output_path, self.args.embeddings_scatterplot_filename), dpi=300)
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
        confusion_display.plot(include_values=False,
                               xticks_rotation="vertical")
        confusion_display.ax_.set_xlabel("")
        confusion_display.ax_.set_ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, self.args.embeddings_dist_matrix_filename))
        plt.close()
