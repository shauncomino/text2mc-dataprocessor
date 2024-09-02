from tap import Tap
import os

""" Arguments for Block2Vec """
class Block2VecArgs(Tap):
    max_num_targets: int = 20
    build_limit: int = -1 # set to -1 for no limit 
    emb_dimension: int = 32
    epochs: int = 3
    batch_size: int = 2
    num_workers: int = 1
    initial_lr: float = 1e-3
    context_radius: int = 1
    output_path: str = os.path.join("output", "block2vec") 
    tok2block_filepath: str = "../world2vec/tok2block.json"
    block2texture_filepath: str = "../world2vec/block2texture.json"
    hdf5s_directory = "../processed_builds"
    checkpoints_directory = "checkpoints"
    model_savefile_name = "best_model.pth"
    textures_directory: str = os.path.join("textures") 
    embeddings_txt_filename: str = "embeddings.txt"
    embeddings_json_filename: str = "embeddings.json"
    embeddings_npy_filename: str = "embeddings.npy"
    embeddings_pkl_filename: str = "representations.pkl"
    embeddings_scatterplot_filename: str = "scatter_3d.png"
    embeddings_dist_matrix_filename: str = "dist_matrix.png"