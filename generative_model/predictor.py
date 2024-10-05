import json
import torch
import torch.nn as nn
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from text2mcVAEDataset import text2mcVAEDataset
import scipy
import numpy as np
import os
import h5py
import vec2world
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

class text2mcPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the paths to the embeddings and models
        self.EMBEDDINGS_FILE = "../block2vec/output/block2vec/embeddings.json"
        self.EMBEDDING_MODEL_PATH = "../block2vec/checkpoints/best_model.pth"

        self.ENCODER_MODEL_PATH = "generative_model/checkpoints/encoder.pth"
        self.DECODER_MODEL_PATH = "generative_model/checkpoints/decoder.pth"

        self.BLOCK_TO_TOK = "../world2vec/block2tok.json"

        self.SAVE_DIRECTORY = "../generated_builds/"
        self.GENERATED_HDF5S_PATH = "../generated_hdf5s/"

        # Load the pre-trained embeddings and the encoder and decoder models and set them to eval mode
        self.embeddings = json.load(open(self.EMBEDDINGS_FILE))
        self.block2tok = json.load(open(self.BLOCK_TO_TOK))

        self.encoder = text2mcVAEEncoder()
        self.encoder.load_state_dict(torch.load(self.ENCODER_MODEL_PATH, weights_only=True))
        self.encoder.eval()

        self.decoder = text2mcVAEDecoder()
        self.decoder.load_state_dict(torch.load(self.DECODER_MODEL_PATH, weights_only=True))
        self.decoder.eval()

    # 1. Loads two builds from the dataset (user specified)
    # 2. Embeds builds using trained embedding model
    def embed_builds(self, building1_path: str, building2_path: str):
        hdf5_files = [building1_path, building2_path]

        dataset = text2mcVAEDataset(file_paths=hdf5_files, block2embedding=self.embeddings, block2tok=self.block2tok, block_ignore_list=[102], fixed_size=(64, 64, 64))
        
        '''
        Get the data and mask for the two builds by calling the __getitem__ method which converts the build into the embeddings.
        Do we need the mask part for anything or is that is only for figuring out the loss?
        '''
        building1_data, building1_mask = dataset.__getitem__(0)
        building2_data, building2_mask = dataset.__getitem__(1)

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
        embedded_data_flat = embedded_data_flat[0].numpy()  # (N, Embedding_Dim)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embedding_matrix)
        distances, indices = nbrs.kneighbors(embedded_data_flat)
        tokens = indices.flatten().reshape(D, H, W)
        return tokens
    
    # 5. Send those intermediate latent points through the decoder portion of the VAE    
    def decode_and_generate(self, interpolations, embedding_matrix):
        
        for z in interpolations:
            recon_embedding = self.decoder(z)

            # Convert to numpy array
            recon_embedding = recon_embedding.cpu()

            # Convert tokens to block names
            tokens = text2mcPredictor.embeddings_to_tokens(recon_embedding, embedding_matrix) # 3d Array Of Integers

            # Call functions from vec2world to convert tokens to blocks and same it as a schematics
            string_world = vec2world.convert_numpy_array_to_blocks(tokens)
            file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            vec2world.create_schematic_file(string_world, self.SAVE_DIRECTORY, file_name)

    def encode_build_find_latent_point(self, building_path):
        hdf5_files = [building_path]

        dataset = text2mcVAEDataset(file_paths=hdf5_files, block2embedding=self.embeddings, block2tok=self.block2tok, block_ignore_list=[102], fixed_size=(64, 64, 64))

        building_data, building_mask = dataset.__get__item(0)

        z, mu, logvar = self.encoder(building_data)

        return z

def main():
    building1_path = "rar_test5_Desert+Tavern+2.h5"
    building2_path = "rar_test6_Desert_Tavern.h5"

    predictor = text2mcPredictor()
    
    building1_embedding, building2_embedding, embedding_matrix = predictor.embed_builds(building1_path, building2_path)
    building1_latent, building2_latent = predictor.encode_builds(building1_embedding, building2_embedding)
    interpolations = predictor.interpolate_latent_points(building1_latent, building2_latent, num_interpolations=1)
    predictor.decode_and_generate(interpolations, embedding_matrix)

if __name__ == "__main__":
    main()