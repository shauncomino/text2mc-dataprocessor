import json
import h5py as h5
import torch
import torch.nn as nn
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from text2mcVAEDataset import text2mcVAEDataset
import scipy
import numpy as np
import os
import h5py
from datetime import datetime


class text2mcPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the paths to the embeddings and models
        self.EMBEDDINGS_FILE = "../block2vec/output/block2vec/embeddings.json"
        self.EMBEDDING_MODEL_PATH = "../block2vec/checkpoints/best_model.pth"

        self.ENCODER_MODEL_PATH = "generative_model/checkpoints/encoder.pth"
        self.DECODER_MODEL_PATH = "generative_model/checkpoints/decoder.pth"

        self.TOK_TO_BLOCK = "../world2vec/tok2block.json"
        self.BLOCK_TO_TOK = "../world2vec/block2tok.json"

        self.SAVE_DIRECTORY = "../generated_builds/"
        self.GENERATED_HDF5S_PATH = "../generated_hdf5s/"

        # Load the pre-trained embeddings and the encoder and decoder models and set them to eval mode
        self.embeddings = json.load(open(self.EMBEDDINGS_FILE))
        self.tok2block = json.load(open(self.TOK_TO_BLOCK))
        self.block2tok = json.load(open(self.BLOCK_TO_TOK))

        # self.encoder = text2mcVAEEncoder()
        # self.encoder.load_state_dict(torch.load(self.ENCODER_MODEL_PATH, weights_only=True))
        # self.encoder.eval()

        # self.decoder = text2mcVAEDecoder()
        # self.decoder.load_state_dict(torch.load(self.DECODER_MODEL_PATH, weights_only=True))
        # self.decoder.eval()

    # 1. Loads two builds from the dataset (user specified)
    # 2. Embeds builds using trained embedding model
    def embed_builds(self, building1_path: str, building2_path: str):
        hdf5_files = [building1_path, building2_path]

        dataset = text2mcVAEDataset(file_paths=hdf5_files, block2embedding=self.embeddings, block2tok=self.block2tok, block_ignore_list=[102], fixed_size=(256, 256, 256, 32))
        
        '''
        Get the data and mask for the two builds by calling the __getitem__ method which converts the build into the embeddings.
        Do we need the mask part for anything or is that is only for figuring out the loss?
        '''
        building1_data, building1_mask = dataset.__getitem__(0)
        building2_data, building2_mask = dataset.__getitem__(1)

        return building1_data, building2_data

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

    # 5. Send those intermediate latent points through the decoder portion of the VAE    
    def call_vae(self, interpolations, save_dir):
        
        interp_builds_paths = []
        
        recon_build = self.decoder(interpolations)
        # recon_build: (1, num_tokens, Depth, Height, Width)

        # Convert logits to predicted tokens
        recon_build = recon_build.argmax(dim=1)  # (1, Depth, Height, Width)

        # Convert to numpy array
        recon_build_np = recon_build.cpu().numpy().squeeze(0)  # (Depth, Height, Width)

        # Save the build as an HDF5 file
        save_path = os.path.join(save_dir, f'{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5')
        with h5py.File(save_path, 'w') as h5f:
            h5f.create_dataset('build', data=recon_build_np, compression='gzip')
        
        interp_builds_paths.append(save_path)

        print(f'Saved interpolated build at {save_path}')
        return interp_builds_paths

    # 6. Convert those intermediate (embedded at this point) builds into tokens
    def convert_intermediate_builds_to_tokens(self, interpolations):
        
        tokens = list(self.tok2block.keys())
        embeddings = np.array(list(self.embeddings.values()))

        token_arrays = []

        for z in interpolations:
            recon_build = self.decoder(z)
            # recon_build: (1, num_tokens, Depth, Height, Width)

            # Convert logits to predicted tokens
            recon_build = recon_build.argmax(dim=1)  # (1, Depth, Height, Width)

            # Convert to numpy array
            recon_build_np = recon_build.cpu().numpy().squeeze(0)  # (Depth, Height, Width)

            # Initialize an array to store the token representation
            token_array = np.empty(recon_build_np.shape, dtype=object)

            # Flatten the recon_build_np for distance computation
            recon_build_flat = recon_build_np.reshape(-1, recon_build_np.shape[-1])

            # Compute distances between recon_build_flat and pre-trained embeddings
            distances = cdist(recon_build_flat, embeddings, metric='euclidean')

            # Find the index of the closest embedding for each point
            closest_indices = np.argmin(distances, axis=1)

            # Map the closest indices to tokens
            closest_tokens = np.array(tokens)[closest_indices]

            # Reshape the token array to the original shape
            token_array = closest_tokens.reshape(recon_build_np.shape)

            token_arrays.append(token_array)

        return token_arrays


        pass
    
    # 7. Convert those token arrays into actual builds using vec2world
    # 8. Save the build to be accessible somewhere
    def save_predicted_build(self):
        file_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        pass

def main():
    building1_path = "rar_test5_Desert+Tavern+2.h5"
    building2_path = "rar_test6_Desert_Tavern.h5"

    predictor = text2mcPredictor()
    
    building1_embedding, building2_embedding = predictor.embed_builds(building1_path, building2_path)
    print(building1_embedding)
    #building1_latent, building2_latent = predictor.encode_builds(building1_embedding, building2_embedding)

    
if __name__ == "__main__":
    main()