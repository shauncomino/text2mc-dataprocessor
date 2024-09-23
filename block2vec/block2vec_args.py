from tap import Tap
import os

""" Arguments for Block2Vec """
class Block2VecArgs(Tap):
    max_num_targets: int = 20
    build_limit: int = -1 # set to -1 for no limit 
    emb_dimension: int = 32
    epochs: int = 3
    batch_size: int = 1
    num_workers: int = 1
    initial_lr: float = 1e-3
    context_radius: int = 1
    targets_per_batch: int = 100
    targets_per_build: int = 50
    output_path: str = os.path.join("output", "block2vec") 
    textures_directory: str = os.path.join("block_pngs")
    tok2block_filepath: str = "../world2vec/tok2block.json"
    block2texture_filepath: str = "../world2vec/block2texture.json"
    hdf5s_directory = "../processed_builds"
    checkpoints_directory = "checkpoints"
    model_savefile_name = "best_model.pth" 
    cur_model_safefile_name = "current_model.pth"
    embeddings_txt_filename: str = "embeddings.txt"
    embeddings_json_filename: str = "embeddings.json"
    embeddings_npy_filename: str = "embeddings.npy"
    embeddings_pkl_filename: str = "representations.pkl"
    embeddings_scatterplot_filename: str = "embeddings_scatterplot_2d.png"
    embeddings_dist_matrix_filename: str = "dist_matrix.png"
    blocks_to_plot: list = ["minecraft:cobblestone",
                                "minecraft:crafting_table",
                                "minecraft:furnace",
                                "minecraft:fletching_table",
                                "minecraft:glass",
                                "minecraft:smooth_stone",
                                "minecraft:stripped_acacia_log[axis=x]",
                                "minecraft:stripped_acacia_log[axis=y]",
                                "minecraft:stripped_acacia_log[axis=z]",
                                "minecraft:acacia_wood[axis=x]",
                                "minecraft:acacia_wood[axis=y]",
                                "minecraft:acacia_wood[axis=z]",
                                "minecraft:acacia_planks",
                                "minecraft:spruce_wood[axis=x]",
                                "minecraft:spruce_wood[axis=y]",
                                "minecraft:spruce_wood[axis=z]",
                                "minecraft:spruce_planks", 
                                "minecraft:stripped_dark_oak_wood[axis=x]",
                                "minecraft:stripped_dark_oak_wood[axis=y]",
                                "minecraft:stripped_dark_oak_wood[axis=z]",
                                "minecraft:dark_oak_planks",
                                "minecraft:birch_wood[axis=x]",
                                "minecraft:birch_wood[axis=y]",
                                "minecraft:birch_wood[axis=z]",
                                "minecraft:stripped_birch_log[axis=x]",
                                "minecraft:stripped_birch_log[axis=y]",
                                "minecraft:stripped_birch_log[axis=z]",
                                "minecraft:birch_planks", 
                                "minecraft:bookshelf",
                                "minecraft:dark_prismarine",
                                "minecraft:diamond_block",
                                "minecraft:emerald_block", 
                                "minecraft:beacon", 
                                "minecraft:gold_block",
                                "minecraft:glowstone"]