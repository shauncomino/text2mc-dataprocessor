import os
import numpy as np
import traceback 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from tap import Tap
import json
from block2vec_dataset import Block2VecDataset
from skip_gram_model import SkipGramModel

NIL_DATAPOINT_INDICATOR = 7000

""" Arguments for Block2Vec """
class Block2VecArgs(Tap):
    max_build_dim: int = 100
    max_num_targets: int = 20
    build_limit: int = -1
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

    
def custom_collate_fn(batch):
    targets, contexts, build_names = zip(*batch)

    # Convert the targets and contexts to tensors
    targets = [torch.tensor(target, dtype=torch.int64) for target in targets if len(target) > 0]
    contexts = [torch.tensor(context, dtype=torch.int64) for context in contexts if len(context) > 0]

    print("Processing %d build(s) in batch." % (len(targets)))

    # Handle empty batch 
    if len(targets) == 0 or len(contexts) == 0:
        return None

    # Pack the targets and contexts as variable-length sequences
    packed_targets = pack_sequence(targets, enforce_sorted=False)
    packed_contexts = pack_sequence(contexts, enforce_sorted=False)

    # Return the packed sequences, without the build names
    return packed_targets, packed_contexts, build_names

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
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
                batch_targets.to(device)
                batch_contexts.to(device)

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

            except Exception as e: 
                print("Error occured processing training batch:")
                traceback.print_exc()

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}")

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

                    batch_targets.to(device)
                    batch_contexts.to(device)

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
                    print("Error occured processing validation batch:")
                    traceback.print_exc()

            val_loss /= len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(Block2VecArgs.checkpoints_directory, Block2VecArgs.model_savefile_name))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

def test(model, test_loader, criterion, device):
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

                batch_targets.to(device)
                batch_contexts.to(device)
                
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
                print("Error occured processing test batch:")
                traceback.print_exc()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    
def main():
    with open(Block2VecArgs.tok2block_filepath, "r") as file:
        tok2block = json.load(file)

    # Initialize dataset
    dataset = Block2VecDataset(
        directory=Block2VecArgs.hdf5s_directory,
        tok2block=tok2block, 
        context_radius=Block2VecArgs.context_radius,
        max_build_dim=Block2VecArgs.max_build_dim,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=Block2VecArgs.initial_lr)
    criterion = nn.CrossEntropyLoss()

    # Training and validation  
    train(model, train_loader, val_loader, optimizer, criterion, Block2VecArgs.epochs, device)
    model.load_state_dict(torch.load(os.path.join(Block2VecArgs.checkpoints_directory, Block2VecArgs.model_savefile_name)))
    
    # Testing 
    test(model, test_loader, criterion, device)

if __name__ == "__main__":
    main()
