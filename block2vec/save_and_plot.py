import os
import numpy as np
from loguru import logger 
import torch
import json
from matplotlib import pyplot as plt
from skip_gram_model import SkipGramModel
from block2vec_args import Block2VecArgs
from image_annotations_2d import ImageAnnotations2D
import umap

def get_next_filepath(base_filepath): 
    base, extension = os.path.splitext(base_filepath)
    counter = 1
    new_filepath = base_filepath

    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{extension}"
        counter += 1

    return new_filepath

def plot_embeddings(embedding_dict: dict[str, np.ndarray]):
    logger.info("Plotting embeddings...")
    # Load block2texture dict 
    with open(Block2VecArgs.block2texture_filepath, "r") as file:
        block2texture = json.load(file)
    
    # Load block images  
    texture_imgs = [plt.imread(os.path.join(Block2VecArgs.textures_directory, block2texture[block])) for block in Block2VecArgs.blocks_to_plot]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    embeddings = torch.stack(list(embedding_dict.values())).numpy()
    if embeddings.shape[-1] != 2:
        embeddings_2d = umap.UMAP(n_neighbors=5, min_dist=0.2, n_components=2).fit_transform(embeddings) # n_components = dimension of plotted embeddings
    else:
        embeddings_2d = embeddings
    for embedding in embeddings_2d:
        ax.scatter(*embedding, alpha=0)
    ia = ImageAnnotations2D(embeddings_2d, texture_imgs, Block2VecArgs.blocks_to_plot, ax, fig)
    plt.tight_layout()
    filepath = get_next_filepath(os.path.join(Block2VecArgs.output_path, Block2VecArgs.embeddings_scatterplot_filename))
    plt.savefig(filepath, dpi=300)
    plt.close("all")
    logger.info("Finished plotting embeddings.")

def save_embeddings(embeddings, tok2block: dict[int, str]):
    embedding_dict = {}

    with open(os.path.join(Block2VecArgs.output_path, Block2VecArgs.embeddings_txt_filename), "w") as f:
        for tok, block_name in tok2block.items():
            e = " ".join(map(lambda x: str(x), embeddings[int(tok)]))
            embedding_dict[tok2block[str(tok)]] = torch.from_numpy(embeddings[int(tok)])
            f.write("%s %s\n" % (tok2block[str(tok)], e))
    np.save(os.path.join(Block2VecArgs.output_path, Block2VecArgs.embeddings_npy_filename), embeddings)
    # Create a copy of the embedding_dict with tensors converted to lists
    embedding_dict_copy = {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in embedding_dict.items()
    }

    # Write the modified copy to the JSON file
    with open(os.path.join(Block2VecArgs.output_path, Block2VecArgs.embeddings_json_filename), 'w') as f:
        json.dump(embedding_dict_copy, f)


def main():
    plot = True
    save  = True 

    with open(Block2VecArgs.tok2block_filepath, "r") as file:
        tok2block = json.load(file)

    model = SkipGramModel(len(tok2block), Block2VecArgs.emb_dimension)
    model.load_state_dict(torch.load(os.path.join(Block2VecArgs.checkpoints_directory, Block2VecArgs.cur_model_safefile_name)))

    embeddings = model.target_embeddings.weight
    embeddings = embeddings.cpu().data.numpy()
    
    embedding_dict = {}
    for tok, something in tok2block.items():
        e = " ".join(map(lambda x: str(x), embeddings[int(tok)]))
        embedding_dict[tok2block[str(tok)]] = torch.from_numpy(embeddings[int(tok)])

    if (plot): 
        plot_embeddings(embedding_dict)
    if (save): 
        save_embeddings(embeddings, tok2block)


if __name__ == "__main__":
    main()
