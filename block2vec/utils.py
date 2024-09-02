import random
import pickle
import numpy as np
import torch
import torchvision
from torch.nn.functional import interpolate, grid_sample
import matplotlib.pyplot as plt


min_blocks = 10 
max_length = 13

target = np.array([9, 0, 0, 0, 0])


target = np.pad(
    target, 
    pad_width=(0, 10),
    mode='constant', 
    constant_values=-1)
                   
context = np.array([
    [[1, 2], [3, 4], [5, 6]],
    [[7, 8], [9, 10], [11, 12]]
])


context = np.pad(
    context, 
    pad_width=((0, max_length), (0, 0), (0, 0)),
    mode='constant',
    constant_values=-1)

print(target)
print(context)

