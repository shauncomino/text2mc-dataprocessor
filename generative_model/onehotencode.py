import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import one_hot
import pandas as pd
import json

file_path = './batch_10_250.h5'
with h5py.File(file_path, 'r') as file:
            build_folder_in_hdf5 = list(file.keys())[0]
            data = file[build_folder_in_hdf5][()]

# using numpy (works but takes lots of memory)
try:           
    num_classes = 3714
    onehot_encoded = np.eye(num_classes, dtype=np.int8)[data]
    print(onehot_encoded)
except Exception as e:
    print("np eye: " + e)

# using torch (doesn't work)
# try:           
#     tensor = torch.from_numpy(data)
#     onehot_encoded = one_hot(tensor, num_classes)
#     print(onehot_encoded)
# except Exception as e:
#     print("torch one_hot: " + str(e))    

# # using sklearn (doesn't work)
# try:           
#     onehot_encoder = OneHotEncoder(sparse_output=False)
#     onehot_encoded = onehot_encoder.fit_transform(data)
#     print(onehot_encoded)
# except Exception as e:
#     print("onehot_encoder: " + str(e))

# # using sklearn and reshaping data (works but loses two dimensions)
# try:           
#     onehot_encoder = OneHotEncoder(sparse_output=False)
#     onehot_encoded = onehot_encoder.fit_transform(data.reshape(-1, 1))
#     print(onehot_encoded)
# except Exception as e:
#     print("onehot_encoder data reshape: " + str(e))

# #using pandas (doesn't work)
# try:
#     onehot_encoded = pd.get_dummies(data)
#     print(onehot_encoded)
# except Exception as e:
#     print("pandas get_dummies: " + str(e))

