import os
import numpy as np
from loguru import logger
import multiprocessing as mp 
import traceback 
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import math
import json
from block2vec_dataset import Block2VecDataset
from skip_gram_model import SkipGramModel
from block2vec_args import Block2VecArgs
from image_annotations_2d import ImageAnnotations2D
import umap


model = SkipGramModel(3716, Block2VecArgs.emb_dimension)
model.load_state_dict(torch.load(os.path.join(Block2VecArgs.checkpoints_directory, Block2VecArgs.model_savefile_name)))
print(model.output.weight)