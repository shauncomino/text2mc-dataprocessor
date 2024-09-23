import json
import h5py as h5
import torch
import torch.nn as nn
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from text2mcVAEDataset import text2mcVAEDataset

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

        print(hdf5_files)
        print(self.BLOCK_TO_TOK)

        dataset = text2mcVAEDataset(file_paths=hdf5_files, block2embedding=self.embeddings, block2tok=self.block2tok, block_ignore_list=[102], fixed_size=(256, 256, 256, 32))
        
        building1_data, building1_mask = dataset.__getitem__(0)
        building2_data, building2_mask = dataset.__getitem__(1)

        return building1_data, building2_data

    # 3. Sends the two embedded builds through the encoder portion of the VAE
    def encode_builds(self, embedding1, embedding2):
        pass
        
    # 4. Linearly interpolate between those n-dimensional latent points to get other latent points connecting the two
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
    predictor.embed_builds(building1_path, building2_path)

if __name__ == "__main__":
    main()