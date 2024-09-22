import json
import h5py as h5

class text2mcPredictor():
    EMBEDDINGS_FILE = "../block2vec/output/block2vec/embeddings.json"

    def __init__(self):
        pass

    # 1. Loads two builds from the dataset (user specified)
    # 2. Embeds builds using trained embedding model
    def load_two_builds_from_dataset(self, building1_path: str, building2_path: str):
        building1 = h5.File(building1_path, 'r')
        building2 = h5.File(building2_path, 'r')
        embeddings = json.load(open(self.EMBEDDINGS_FILE))

    # 3. Sends the two embedded builds through the encoder portion of the VAE
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

building1_path = "rar_test5_Desert+Tavern+2.h5"
building2_path = "rar_test6_Desert_Tavern.h5"

predictor = text2mcPredictor()
predictor.load_two_builds_from_dataset(building1_path, building2_path)
