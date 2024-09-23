import json
import h5py as h5
import torch
import torch.nn as nn
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from text2mcVAEDataset import text2mcVAEDataset
import scipy
import numpy as np

class text2mcPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the paths to the embeddings and models
        self.EMBEDDINGS_FILE = "../block2vec/output/block2vec/embeddings.json"
        self.EMBEDDING_MODEL_PATH = "../block2vec/checkpoints/best_model.pth"

        self.ENCODER_MODEL_PATH = "generative_model/checkpoints/encoder.pth"
        self.DECODER_MODEL_PATH = "generative_model/checkpoints/decoder.pth"

        self.BLOCK_TO_TOK = "../world2vec/block2tok.json"

        # Load the pre-trained embeddings and the encoder and decoder models and set them to eval mode
        self.embeddings = json.load(open(self.EMBEDDINGS_FILE))
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
    def interpolate_latent_points(self, z1, z2, num_steps):
        """
        Linearly interpolate between two n-dimensional latent points using LinearNDInterpolator.

        Args:
            point1 (np.ndarray): The first latent point (n-dimensional).
            point2 (np.ndarray): The second latent point (n-dimensional).
            num_steps (int): The number of interpolation steps.

        Returns:
            np.ndarray: An array of interpolated points.
        """

        """"
            READ THIS WE NEED SOME HELP
            what is z supposed to be?
        """

        # Ensure the points are numpy arrays
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Generate a series of values for t between 0 and 1
        t_values = np.linspace(0, 1, num_steps)

        # Create the interpolation function
        points = np.array([[0], [1]])
        values = np.vstack([point1, point2])
        interpolator = scipy.LinearNDInterpolator(points, values)

        # Compute the interpolated points
        interpolated_points = interpolator(t_values[:, None])

        return interpolated_points
        pass

    # 5. Send those intermediate latent points through the decoder portion of the VAE    
    def call_vae(self):
        pass

    # 6. Convert those intermediate (embedded at this point) builds into tokens
    def convert_intermediate_builds_to_tokens(self):
        pass

    # 7. Convert those token arrays into actual builds using vec2world
    # 8. Save the build to be accessible somewhere
    def save_predicted_build(self):
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