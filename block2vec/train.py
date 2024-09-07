import os
import numpy as np
from loguru import logger
import multiprocessing as mp 
import traceback 
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import math
import json
from block2vec_dataset import Block2VecDataset
from skip_gram_model import SkipGramModel
from block2vec_args import Block2VecArgs
from image_annotations_2d import ImageAnnotations2D
import umap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_collate_fn(batch):
    targets, contexts, build_names = zip(*batch)

    # Convert the targets and contexts to tensors
    targets = [torch.tensor(target, dtype=torch.int64).to(device) for target in targets if len(target) > 0]
    contexts = [torch.tensor(context, dtype=torch.int64).to(device) for context in contexts if len(context) > 0]

    logger.info("Processing %d build(s) in batch." % (len(targets)))

    # Handle empty batch 
    if len(targets) == 0 or len(contexts) == 0:
        return None

    # Pack the targets and contexts as variable-length sequences
    packed_targets = pack_sequence(targets, enforce_sorted=False)
    packed_contexts = pack_sequence(contexts, enforce_sorted=False)

    # Return the packed sequences, without the build names
    return packed_targets, packed_contexts, build_names

def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            try: 
                if batch is None:
                    continue 

                batch_targets, batch_contexts, _ = batch
                batch_targets, lengths = pad_packed_sequence(batch_targets, batch_first=True, padding_value=-1)
                batch_contexts, lengths = pad_packed_sequence(batch_contexts, batch_first=True, padding_value=-1)

                # Move to device 
                #batch_targets.to(device)
               # batch_contexts.to(device)

                optimizer.zero_grad()  # Clear previous gradients
                
                for item_targets, item_contexts in zip(batch_targets, batch_contexts):
                    build_loss = 0.0
                    for target_block, context_blocks in zip(item_targets, item_contexts): 
                        if target_block == -1: 
                            break
                        loss = model(target_block, context_blocks)  # Forward pass, returns loss
                        #loss = criterion(output, target_block)  # Compute loss
                        #running_loss += loss.item()
                        build_loss += loss

                    running_loss += build_loss.item()
                    build_loss.backward()  # Backpropagate
                    optimizer.step()  # Update internal model weights
                    scheduler.step()

            except Exception as e: 
                logger.error("Error occured processing training batch:")
                traceback.print_exc()

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            
            for batch in val_loader:
                try: 
                    if batch is None: 
                        continue 
                    
                    batch_targets, batch_contexts, _ = batch
                    batch_targets, lengths = pad_packed_sequence(batch_targets, batch_first=True, padding_value=-1)
                    batch_contexts, lengths = pad_packed_sequence(batch_contexts, batch_first=True, padding_value=-1)

                    #batch_targets.to(device)
                    #batch_contexts.to(device)

                    for item_targets, item_contexts in zip(batch_targets, batch_contexts):
                        build_loss = 0.0
                        for target_block, context_blocks in zip(item_targets, item_contexts): 
                            if target_block == -1: 
                                break 
                            loss = model(target_block, context_blocks)  # Forward pass, returns loss
                            #loss = criterion(output, target_block)  # Compute loss
                            #running_loss += loss.item()
                            build_loss += loss

                        #build_loss.backward()  # Backpropagate
                        # optimizer.step()  # Update weights
                        val_loss += build_loss.item()
                except Exception as e: 
                    logger.error("Error occured processing validation batch:")
                    traceback.print_exc()

            val_loss /= len(val_loader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(Block2VecArgs.checkpoints_directory, Block2VecArgs.model_savefile_name))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered.")
            break

def test(model, test_loader, device):
    mp.set_start_method('spawn', force=True) 

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
            
        for batch in test_loader:
            try: 
                if batch is None: 
                    continue 
                batch_targets, batch_contexts, _ = batch
                batch_targets, lengths = pad_packed_sequence(batch_targets, batch_first=True, padding_value=-1)
                batch_contexts, lengths = pad_packed_sequence(batch_contexts, batch_first=True, padding_value=-1)

                #batch_targets.to(device)
                #batch_contexts.to(device)
                
                for item_targets, item_contexts in zip(batch_targets, batch_contexts):
                    build_loss = 0.0
                    for target_block, context_blocks in zip(item_targets, item_contexts): 
                        if target_block == -1: 
                            break 
                        loss = model(target_block, context_blocks)  # Forward pass, returns loss
                        #loss = criterion(output, target_block)  # Compute loss
                        #running_loss += loss.item()
                        build_loss += loss
                    #build_loss.backward()  # Backpropagate
                    # optimizer.step()  # Update weights
                    test_loss += build_loss.item()
            except Exception as e: 
                logger.error("Error occured processing test batch:")
                traceback.print_exc()

    test_loss /= len(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")


def get_next_filepath(base_filepath): 
    base, extension = os.path.splitext(base_filepath)
    counter = 1
    new_filepath = base_filepath

    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{extension}"
        counter += 1

    return new_filepath

def plot_embeddings(embedding_dict: dict[str, np.ndarray]):
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
    plt.savefig(os.path.join(Block2VecArgs.output_path, get_next_filepath(Block2VecArgs.embeddings_scatterplot_filename)), dpi=300)
    plt.close("all")

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

    plot_embeddings(embedding_dict)

def main():
    mp.set_start_method('spawn', force=True) 

    with open(Block2VecArgs.tok2block_filepath, "r") as file:
        tok2block = json.load(file)

    # Initialize dataset
    dataset = Block2VecDataset(
        directory=Block2VecArgs.hdf5s_directory,
        tok2block=tok2block, 
        context_radius=Block2VecArgs.context_radius,
        max_num_targets=Block2VecArgs.max_num_targets, 
        build_limit=Block2VecArgs.build_limit
    )

    # Divide dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Block2VecArgs.batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=Block2VecArgs.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Block2VecArgs.batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=Block2VecArgs.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=Block2VecArgs.batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=Block2VecArgs.num_workers)

    # Initialize model
    model = SkipGramModel(len(tok2block), Block2VecArgs.emb_dimension)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Block2VecArgs.initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            math.ceil(len(dataset) / Block2VecArgs.batch_size) *
            Block2VecArgs.epochs,
        )

    # Training and validation  
    train(model, train_loader, val_loader, optimizer, scheduler, Block2VecArgs.epochs, device)
    model.load_state_dict(torch.load(os.path.join(Block2VecArgs.checkpoints_directory, Block2VecArgs.model_savefile_name)))
    
    # Testing 
    test(model, test_loader, device)

    # File save embeddings  
    embeddings = model.target_embeddings.weight
    # embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
    embeddings = embeddings.cpu().data.numpy()
    save_embeddings(embeddings, tok2block)


if __name__ == "__main__":
    main()
