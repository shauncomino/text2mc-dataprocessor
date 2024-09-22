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


files = []
count = 0
for filename in os.listdir(Block2VecArgs.hdf5s_directory):
    if filename.endswith(".h5"):
        files.append(filename)
        count += 1

train_size = int(0.7 * count)
val_size = int(0.1 * count)
test_size = count - train_size - val_size

i = 0
with open('train_dataset_files.txt', 'w') as f:
    for i in range (0, train_size): 
        f.write(f"{files[i]}\n")
with open('val_dataset_files.txt', 'w') as f:
    for i in range (0, val_size): 
        f.write(f"{files[i]}\n")
with open('test_dataset_files.txt', 'w') as f:
    for i in range (0, test_size): 
        f.write(f"{files[i]}\n")

    """  
with open('filenames.txt', 'r') as f:
    filenames = [line.strip() for line in f.readlines()]
"""