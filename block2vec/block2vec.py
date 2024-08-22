import math
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple
import re 
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

from block2vec_dataset import custom_collate_fn
from image_annotations_3d import ImageAnnotations3D
from skip_gram_model import SkipGramModel
from sklearn.metrics import ConfusionMatrixDisplay
from tap import Tap
from torch.utils.data import DataLoader
import umap

""" Arguments for Block2Vec """
class Block2VecArgs(Tap):
    max_build_dim: int = 100
    emb_dimension: int = 32
    epochs: int = 3
    batch_size: int = 2
    num_workers: int = 2
    initial_lr: float = 1e-3
    context_radius: int = 2
    output_path: str = os.path.join("output", "block2vec") 
    tok2block_filepath: str = "../world2vec/tok2block.json"
    block2texture_filepath: str = "../world2vec/block2texture.json"
    hdf5s_directory = "hdf5s"
    textures_directory: str = os.path.join("textures") 
    embeddings_txt_filename: str = "embeddings.txt"
    embeddings_json_filename: str = "embeddings.json"
    embeddings_npy_filename: str = "embeddings.npy"
    embeddings_pkl_filename: str = "representations.pkl"
    embeddings_scatterplot_filename: str = "scatter_3d.png"
    embeddings_dist_matrix_filename: str = "dist_matrix.png"
      


class Block2Vec(pl.LightningModule):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.args: Block2VecArgs = Block2VecArgs().from_dict(kwargs)
        self.save_hyperparameters()
        
        with open(self.args.tok2block_filepath, "r") as file:
            self.tok2block = json.load(file)
        
        with open(self.args.block2texture_filepath, "r") as file:
            self.block2texture = json.load(file)
        
        self.dataset = Block2VecDataset(
            directory=self.args.hdf5s_directory,
            tok2block=self.tok2block, 
            context_radius=self.args.context_radius,
            max_build_dim=self.args.max_build_dim
        )
        self.model = SkipGramModel(len(self.tok2block), self.args.emb_dimension)
        self.textures = dict()
        self.learning_rate = self.args.initial_lr

    def forward(self, target_blocks, context_blocks, **kwargs) -> torch.Tensor:
        target_blocks = torch.tensor(target_blocks)
        context_blocks =  torch.tensor(context_blocks) 

        return self.model(target_blocks, context_blocks, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        print("The length of the batch is: %d" % len(batch))

        total_batch_loss = torch.tensor(0.0, requires_grad=True) # Initialize total batch loss

        if (len(batch)) == 0: 
            #return torch.tensor(0.0, requires_grad=True)
            logger.info("Zero eligible builds in batch. Returing None for batch loss.")
            return None 
        
        # For each item in the batch, which is the tuple: (target_blocks_list, context_blocks_list) for ONE build
        for item in batch: 
            target_blocks, context_blocks, build_name = item  # The target and context block lists for ONE build
            print("Calculating loss for %s" % build_name)
            loss = self.forward(target_blocks, context_blocks)  # Loss returned from the SkipGram model for that entire build
            total_batch_loss =  total_batch_loss + loss  # Accumulate the loss
        
        # Option 1: Average the loss across the batch
        average_loss = total_batch_loss / len(batch)
        self.log("loss", average_loss)
        return average_loss
        
        # Option 2: Sum the loss across the batch 
        # self.log("loss", total_batch_loss)
        # return total_batch_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            math.ceil(len(self.dataset) / self.args.batch_size) *
            self.args.epochs,
        )
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset, batch_size=self.args.batch_size, collate_fn=custom_collate_fn, num_workers=self.args.num_workers, persistent_workers=True)

    """ Plot and save embeddings at end of each training epoch """
    def on_train_epoch_end(self):
        print("here")
        embedding_dict = self.save_embedding(
            self.tok2block, self.args.output_path
        )
        #self.create_confusion_matrix(
            #self.dataset.idx2block, self.args.output_path)
        # self.plot_embeddings(embedding_dict, self.args.output_path)

    """ Reads texture .png file for a given token """
    def read_texture(self, block: str):
        if block not in self.block2texture:
            self.textures[block] = np.ones(shape=[16, 16, 3])
        else: 
            self.textures[block] = plt.imread(self.block2texture[block])
        """
        if block not in self.block and block != "air":
            texture_candidates = Path(self.args.textures_directory).glob("*.png")
            cleaned_text = re.sub(r'\[.*?\]', '', block)
            match = process.extractOne(cleaned_text, texture_candidates)
            
            if match is not None:
                logger.info("Matches {} with {} texture file", block, match[0])
                self.textures[block] = plt.imread(match[0])

        if block not in self.textures:
            self.textures[block] = np.ones(shape=[16, 16, 3])
        """
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
                e = " ".join(map(lambda x: str(x), embeddings[int(wid)]))
                embedding_dict[self.tok2block[str(wid)]] = torch.from_numpy(embeddings[int(wid)])
                f.write("%s %s\n" % (self.tok2block[str(wid)], e))
        #print(embedding_dict)
        np.save(os.path.join(output_path, self.args.embeddings_npy_filename), embeddings)
        
        # Create a copy of the embedding_dict with tensors converted to lists
        embedding_dict_copy = {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in embedding_dict.items()
        }
    
        # Write the modified copy to the JSON file
        with open(os.path.join(output_path, self.args.embeddings_json_filename), 'w') as f:
            json.dump(embedding_dict_copy, f, indent=4)
        
        with open(os.path.join(output_path, self.args.embeddings_pkl_filename), "wb") as f:
            pickle.dump(embedding_dict, f)

        return embedding_dict

    """ Plot generated block embeddings """
    def plot_embeddings(self, embedding_dict: Dict[str, np.ndarray], output_path: str):
        # Increase the figure size for better visibility
        fig = plt.figure(figsize=(55, 55))
        ax = fig.add_subplot(111, projection="3d")
        
        # Prepare the legend and corresponding textures
        legend = [label for label in embedding_dict.keys()]
        texture_imgs = [self.read_texture(block) for block in legend]
        
        # Convert embeddings to numpy array
        embeddings = torch.stack(list(embedding_dict.values())).numpy()
        
        # If embeddings are not 3-dimensional, reduce them to 3D using UMAP
        if embeddings.shape[-1] != 3:
            embeddings_3d = umap.UMAP(
                n_neighbors=10, min_dist=0.9, n_components=3
            ).fit_transform(embeddings)
        else:
            embeddings_3d = embeddings
        
        # Adjust scatter plot details
        scatter = ax.scatter(
            embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
            c=np.arange(len(embeddings_3d)), cmap='Spectral', s=2, alpha=0.6
        )
        
        # Add color bar for better visualization of different clusters
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        
        # If texture images are necessary, use the ImageAnnotations3D class
        ia = ImageAnnotations3D(embeddings_3d, texture_imgs, legend, ax, fig)
        
        # Tightly layout the plot and save it
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, self.args.embeddings_scatterplot_filename), dpi=300)
        plt.close("all")

    def create_confusion_matrix(self, id2block: Dict[int, str], output_path: str):
        rcParams.update({"font.size": 1})
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
 