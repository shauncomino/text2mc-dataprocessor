import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from text2mcVAEDataset import text2mcVAEDataset
import scipy
import numpy as np
import os
import sys
import h5py
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import umap

# Add the vec2world and rendering directories to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vec2world'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rendering'))

from vec2world import convert_numpy_array_to_blocks, create_schematic_file
from render_single import process_hdf5_file as two_build_render # Import the function from render_single.py
from render_hdf5s import process_hdf5_file as one_build_render

class text2mcPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the paths to the embeddings and models
        self.EMBEDDINGS_FILE = "../block2vec/output/block2vec/embeddings_old.json"
        self.EMBEDDING_MODEL_PATH = "../block2vec/checkpoints/best_model.pth"


        # Update the path to the model checkpoint
        self.MODEL_PATH = "../best_models/USE_THIS_MODEL.pth"

        self.BLOCK_TO_TOK = "../world2vec/block2tok.json"

        self.SAVE_DIRECTORY = "../generated_builds/"

        # Load the pre-trained embeddings and the encoder and decoder models and set them to eval mode
        self.embeddings = json.load(open(self.EMBEDDINGS_FILE))
        self.block2tok = json.load(open(self.BLOCK_TO_TOK))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.air_token_id = self.block2tok["minecraft:air"]

        print("Loading model...")
        checkpoint = torch.load(self.MODEL_PATH, map_location=self.device, weights_only=True)
        print("Model loaded.")

        self.encoder = text2mcVAEEncoder().to(self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.encoder.eval()

        self.decoder = text2mcVAEDecoder().to(self.device)
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.decoder.eval()

    # 2. Embeds builds using trained embedding model
    def embed_builds(self, building1_path: str, building2_path: str):
        hdf5_files = [building1_path, building2_path]

        dataset = text2mcVAEDataset(file_paths=hdf5_files, block2embedding=self.embeddings, block2tok=self.block2tok, block_ignore_list=[102], fixed_size=(64, 64, 64))

        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        data_list = []
        for data, _ in data_loader:
            data = data.to(self.device)
            data_list.append(data)

        building1_data = data_list[0]
        building2_data = data_list[1]

        return building1_data, building2_data, dataset.embedding_matrix

    # Alternatively, embed a single build
    def embed_single(self, build_path: str):
        hdf5_files = [build_path]

        dataset = text2mcVAEDataset(file_paths=hdf5_files, block2embedding=self.embeddings, block2tok=self.block2tok, block_ignore_list=[102], fixed_size=(64, 64, 64))

        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        data_list = []
        for data, _ in data_loader:
            data = data.to(self.device)
            data_list.append(data)

        return data_list[0], dataset.embedding_matrix

    # 3. Sends the two embedded builds through the encoder portion of the VAE
    def encode_builds(self, embedding1, embedding2):
        '''
        Get the latent points for the two builds
        Do we need the mu and logvar for anything?
        '''
        z1, mu1, logvar1 = self.encoder(embedding1)
        z2, mu2, logvar2 = self.encoder(embedding2)

        return z1, z2

    # Alternatively, encode a single build
    def encode_single(self, build):
        z, mu, logvar = self.encoder(build)
        return z

    # 4. Linearly interpolate between those n-dimensional latent points to get other latent points connecting the two
    def interpolate_latent_points(self, z1, z2, num_interpolations=1):
        interpolations = []
        for alpha in np.linspace(0, 1, num_interpolations):
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolations.append(z_interp)

        return interpolations

    # 6. Convert those intermediate (embedded at this point) builds into tokens
    def embeddings_to_tokens(self, embedded_data, embedding_matrix):
        batch_size, embedding_dim, D, H, W = embedded_data.shape
        N = D * H * W
        embedded_data_flat = embedded_data.view(batch_size, embedding_dim, -1).permute(0, 2, 1).contiguous()
        embedded_data_flat = embedded_data_flat[0].detach().numpy()  # (N, Embedding_Dim)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embedding_matrix)
        distances, indices = nbrs.kneighbors(embedded_data_flat)
        tokens = indices.flatten().reshape(D, H, W)
        return tokens

    def decode_and_generate(self, interpolations, embedding_matrix, building1_path, building2_path, timestamp):
        # Create the main timestamped folder once
        main_folder_path = os.path.join(self.SAVE_DIRECTORY, timestamp)
        os.makedirs(main_folder_path, exist_ok=True)

        for i, z in enumerate(interpolations):
            recon_embedding, block_air_pred = self.decoder(z)
            # Convert to numpy array
            recon_embedding = recon_embedding.cpu()
            # Convert embeddings to tokens
            recon_tokens = self.embeddings_to_tokens(recon_embedding, embedding_matrix)  # 3D Array Of Integers
            # Apply block-air mask
            block_air_pred_labels = (block_air_pred.squeeze(1) >= 0.5).long()
            air_mask = (block_air_pred_labels == 0).cpu()  # Move air_mask to CPU
            # Ensure recon_tokens and air_mask are 3-dimensional
            if recon_tokens.ndim == 4:
                recon_tokens = recon_tokens.squeeze(0)
            if air_mask.ndim == 4:
                air_mask = air_mask.squeeze(0)
            # Assign air_token_id to air voxels
            recon_tokens[air_mask] = self.air_token_id
            # Convert to numpy array
            recon_tokens_np = recon_tokens  # Shape: (Depth, Height, Width)
            file_name = f"interpolation_{i:03d}"
            hdf5_file_path = os.path.join(main_folder_path, f"{file_name}.h5")
            with h5py.File(hdf5_file_path, 'w') as hdf5_file:
                hdf5_file.create_dataset('recon_tokens', data=recon_tokens_np)
            # Call functions from vec2world to convert tokens to blocks and save it as a schematic
            string_world = convert_numpy_array_to_blocks(recon_tokens_np)
            create_schematic_file(string_world, main_folder_path, file_name)

        # Save the tokens as an HDF5 file to use the rendering script.
        two_build_render(main_folder_path, building1_path, building2_path)

    # Alternatively, decode and generate a single build
    def decode_single(self, build, embedding_matrix):

        # Create a new folder with the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_directory = os.path.join(self.SAVE_DIRECTORY, timestamp)

        os.makedirs(save_directory, exist_ok=True)

        for z in build:
            recon_embedding, block_air_pred = self.decoder(z)

            # Convert to numpy array
            recon_embedding = recon_embedding.cpu()

            # Convert embeddings to tokens
            recon_tokens = self.embeddings_to_tokens(recon_embedding, embedding_matrix)  # 3D Array Of Integers

            # Apply block-air mask
            block_air_pred_labels = (block_air_pred.squeeze(1) >= 0.5).long()
            air_mask = (block_air_pred_labels == 0).cpu()  # Move air_mask to CPU

            # Ensure recon_tokens and air_mask are 3-dimensional
            if recon_tokens.ndim == 4:
                recon_tokens = recon_tokens.squeeze(0)
            if air_mask.ndim == 4:
                air_mask = air_mask.squeeze(0)

            # Assign air_token_id to air voxels
            recon_tokens[air_mask] = self.air_token_id

            # Convert to numpy array
            recon_tokens_np = recon_tokens  # Shape: (Depth, Height, Width)

            file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            hdf5_file_path = f"{save_directory}/{file_name}.h5"
            with h5py.File(hdf5_file_path, 'w') as hdf5_file:
                hdf5_file.create_dataset('recon_tokens', data=recon_tokens_np)

            # Call functions from vec2world to convert tokens to blocks and save it as a schematic
            string_world = convert_numpy_array_to_blocks(recon_tokens_np)
            create_schematic_file(string_world, save_directory, file_name)

        # Save the tokens as an HDF5 file to use the rendering script.
        one_build_render(save_directory)

    def predict(self, building1_path: str, building2_path: str):
        building1_embedding, building2_embedding, embedding_matrix = self.embed_builds(building1_path, building2_path)
        building1_latent, building2_latent = self.encode_builds(building1_embedding, building2_embedding)
        interpolations = self.interpolate_latent_points(building1_latent, building2_latent, num_interpolations=60)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.decode_and_generate(interpolations, embedding_matrix, building1_path, building2_path, timestamp)

def encode_and_reconstruct(build_paths: list):
    predictor = text2mcPredictor()
    latents = []

    for path in build_paths:
        build_embedding, embedding_matrix = predictor.embed_single(path)
        print("Encoding %s" % path)
        build_latent = predictor.encode_single(build_embedding)
        latents.append(build_latent)
        predictor.decode_single([build_latent], embedding_matrix)
    return latents


# paths = ["batch_118_3058.h5", # House 2
# "batch_322_8358.h5", # House 3
# "batch_624_16217.h5", # Castle 3
# "batch_32_829.h5", # House 4
# "batch_81_2089.h5", # House 5
# "batch_2_26.h5", # Tower 3
# "batch_109_2825.h5", # Tower 1
# "batch_61_1569.h5", # Castle 4
# "batch_481_12502.h5", # Castle 2
# "batch_52_1334.h5", # Tower 2
# "batch_305_7928.h5", # Terrain 1
# "batch_350_9081.h5", # House 6
# "batch_512_13289.h5", # Castle 1
# "batch_176_4551.h5"] # House 1


# labels = ["House 2",
# "House 3",
# "Castle 3",
# "House 4",
# "House 5",
# "Tower 3",
# "Tower 1",
# "Castle 4",
# "Castle 2",
# "Tower 2",
# "Terrain 1",
# "House 6",
# "Castle 1",
# "House 1"]

# for i in range(0, len(paths)):
#     paths[i] = os.path.join("../processed_builds", paths[i])

# latents = encode_and_reconstruct(paths)

# # Get a 2D plot of build latents
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)

# latent_stack = torch.stack(list(latents)).detach().numpy()
# num_builds, a, b, c, d, e = latent_stack.shape
# latent_stack = latent_stack.reshape(num_builds, a*b*c*d*e)

# if latent_stack.shape[-1] != 2:
#     latents_2d = umap.UMAP(n_neighbors=5, min_dist=0.2, n_components=2).fit_transform(latent_stack)
# else:
#     latents_2d = latent_stack

# index = 0
# for latent in latents_2d:
#     ax.scatter(*latent)
#     ax.annotate(labels[index], latent)
#     index += 1

# plt.tight_layout()
# plt.savefig("latent_plot.png", dpi=300)
# plt.close("all")
