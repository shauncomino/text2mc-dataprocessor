import json
import h5py as h5
import torch
import torch.nn as nn
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
class text2mcPredictor(nn.Module):
    EMBEDDINGS_FILE = "../block2vec/output/block2vec/embeddings.json"
    EMBEDDING_MODEL_PATH = "../block2vec/checkpoints/current_model.pth"

    def __init__(self):
        super().__init__()
        self.embeddings = json.load(open(self.EMBEDDINGS_FILE))
        self.embedding_model = torch.load(self.EMBEDDING_MODEL_PATH, weights_only=True)
        self.embedding_model.eval()
        
        #device_type = "cpu"
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(device_type)
        self.encoder = text2mcVAEEncoder().to(device)
        self.decoder = text2mcVAEDecoder().to(device)

    # 1. Loads two builds from the dataset (user specified)
    def load_two_builds_from_dataset(self, building1_path: str, building2_path: str):
        building1 = h5.File(building1_path, 'r')
        building2 = h5.File(building2_path, 'r')
        return building1, building2

    # 2. Embeds builds using trained embedding model
    def embed_builds(self, building1, building2):
        embedding1 = self.embedding_model(building1)
        embedding2 = self.embedding_model(building2)
        return embedding1, embedding2

    def embed(self, building: h5.File):
        return self.embedding_model(building)
    
    # 3. Sends the two embedded builds through the encoder portion of the VAE
    def encode_builds(self, embedding1, embedding2):
        encode1 = self.encoder(embedding1)
        encode2 = self.encoder(embedding2)
        
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
    building1, building2 = predictor.load_two_builds_from_dataset(building1_path, building2_path)

    print(building1)


if __name__ == "__main__":
    main()