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

# Add the vec2world and rendering directories to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vec2world'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rendering'))

from vec2world import convert_numpy_array_to_blocks, create_schematic_file
from render_single import process_hdf5_files  # Import the function from render_single.py


class text2mcPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the paths to the embeddings and models
        self.EMBEDDINGS_FILE = "../block2vec/output/block2vec/embeddings.json"
        self.EMBEDDING_MODEL_PATH = "../block2vec/checkpoints/best_model.pth"

        # Update the path to the model checkpoint
        self.MODEL_PATH = "checkpoints/best_model.pth"

        self.BLOCK_TO_TOK = "../world2vec/block2tok.json"

        self.SAVE_DIRECTORY = "../generated_builds/"

        # Load the pre-trained embeddings and the encoder and decoder models and set them to eval mode
        self.embeddings = json.load(open(self.EMBEDDINGS_FILE))
        self.block2tok = json.load(open(self.BLOCK_TO_TOK))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.air_token_id = self.block2tok["minecraft:air"]

        checkpoint = torch.load(self.MODEL_PATH, map_location=self.device, weights_only=True)

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

    # 3. Sends the two embedded builds through the encoder portion of the VAE
    def encode_builds(self, embedding1, embedding2):
        '''
        Get the latent points for the two builds
        Do we need the mu and logvar for anything?
        '''
        z1, mu1, logvar1 = self.encoder(embedding1)
        z2, mu2, logvar2 = self.encoder(embedding2)

        return z1, z2
        
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
    
    # 5. Send those intermediate latent points through the decoder portion of the VAE    
    def decode_and_generate(self, interpolations, embedding_matrix):
    
        # Create a new folder with the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.SAVE_DIRECTORY = os.path.join(self.SAVE_DIRECTORY, timestamp)
        os.makedirs(self.SAVE_DIRECTORY, exist_ok=True)

        for z in interpolations:
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

            hdf5_file_path = f"{self.SAVE_DIRECTORY}/{file_name}.h5"
            with h5py.File(hdf5_file_path, 'w') as hdf5_file:
                hdf5_file.create_dataset('recon_tokens', data=recon_tokens_np)
            
            # Call functions from vec2world to convert tokens to blocks and save it as a schematic
            string_world = convert_numpy_array_to_blocks(recon_tokens_np)
            create_schematic_file(string_world, self.SAVE_DIRECTORY, file_name)
        
        # Save the tokens as an HDF5 file to use the rendering script.
        process_hdf5_files(self.SAVE_DIRECTORY)

def main():
    building1_path = "batch_108_2789.h5"
    building2_path = "batch_116_3001.h5"

    predictor = text2mcPredictor()
    
    building1_embedding, building2_embedding, embedding_matrix = predictor.embed_builds(building1_path, building2_path)
    building1_latent, building2_latent = predictor.encode_builds(building1_embedding, building2_embedding)
    interpolations = predictor.interpolate_latent_points(building1_latent, building2_latent, num_interpolations=60)
    predictor.decode_and_generate(interpolations, embedding_matrix)

if __name__ == "__main__":
    main()