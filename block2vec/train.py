import os
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
from block2vec_dataset import Block2VecDataset
from tap import Tap
import json
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
    context_radius: int = 5
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
    targets, contexts, build_names = [], [], []
    
    for item in batch:
        if item:
            target_blocks, context_blocks, build_name = item[0], item[1], item[2]
            if len(target_blocks) > 0 and len(context_blocks) > 0:

                # Pad target_blocks if necessary
                try: 
                    if len(target_blocks) < Block2VecArgs.max_num_targets:
                        print("there are %d targets" % len(target_blocks))
                        print("%d" % (Block2VecArgs.max_num_targets - len(target_blocks)))
                        print("dafug")
                        target = np.pad(
                            np.array(target_blocks),
                            pad_width=(0, Block2VecArgs.max_num_targets - len(target_blocks)),
                            mode='constant',
                            constant_values=-1)
                        
                        pad_arr = np.full((Block2VecArgs.max_num_targets - len(target_blocks), len(context_blocks[0])), 7)
                        print(target)
                        context = np.pad(
                            np.array(context_blocks),
                            pad_width=((0, Block2VecArgs.max_num_targets - len(target_blocks)), (0, 0), (0, 0)),
                            mode='constant',
                            constant_values=-1)
                    else:
                        target = np.array(target_blocks)
                        context = np.array(context_blocks)

                    targets.append(target)
                    contexts.append(context)
                    build_names.append(build_name)
                except: 
                    print("something messed up.")
                    print("context blocks:")
                    print(context_blocks)
                    print("target blocks:")
                    print(target_blocks)
                    print("build name:")
                    print(build_name)

    target = np.array(targets)
    context = np.array(contexts)



    print("target collate:")
    print(target)
    print("context collate:")
    print(context)

    return target, context, build_names

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    best_val_loss = float('inf')
    patience = 3
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch_targets, batch_contexts, _ = batch

            # Move to device
            batch_targets = torch.tensor(batch_targets).to(device)
            batch_contexts = torch.tensor(batch_contexts).to(device)

            optimizer.zero_grad()  # Clear previous gradients
            
            for item_targets, item_contexts in zip(batch_targets, batch_contexts):
                build_loss = 0.0
                print("item targets")
                print(item_targets)
                print("Item contexts:")
                print(item_contexts)
                for target_block, context_blocks in zip(item_targets, item_contexts): 
                    if target_block == -1: 
                        break
                    loss = model(target_block, context_blocks)  # Forward pass, returns loss
                    #loss = criterion(output, target_block)  # Compute loss
                    #running_loss += loss.item()
                    build_loss += loss

                running_loss += build_loss.item()
                build_loss.backward()  # Backpropagate
                optimizer.step()  # Update weights

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            
            for batch in val_loader:
                batch_targets, batch_contexts, _ = batch

                batch_targets = torch.tensor(batch_targets).to(device)
                batch_contexts = torch.tensor(batch_contexts).to(device)

                for item_targets, item_contexts in zip(batch_targets, batch_contexts):
                    build_loss = 0.0
                    for target_block, context_blocks in zip(item_targets, item_contexts): 
                        if (target_block) == -1: 
                            break 

                        loss = model(target_block, context_blocks)  # Forward pass, returns loss
                        #loss = criterion(output, target_block)  # Compute loss
                        #running_loss += loss.item()
                        build_loss += loss
                      

                    #build_loss.backward()  # Backpropagate
                    # optimizer.step()  # Update weights
                    val_loss += build_loss.item()

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
            batch_targets, batch_contexts, _ = batch

            batch_targets = torch.tensor(batch_targets).to(device)
            batch_contexts = torch.tensor(batch_contexts).to(device)
            
            
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

    # Split the dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Block2VecArgs.batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=Block2VecArgs.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Block2VecArgs.batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=Block2VecArgs.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=Block2VecArgs.batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=Block2VecArgs.num_workers)

    # Initialize the model
    model = SkipGramModel(len(tok2block), Block2VecArgs.emb_dimension)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=Block2VecArgs.initial_lr)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion, Block2VecArgs.epochs, device)

    # Load the best model for testing
    model.load_state_dict(torch.load(os.path.join(Block2VecArgs.checkpoints_directory, Block2VecArgs.model_savefile_name)))

    # Test the model
    test(model, test_loader, criterion, device)

if __name__ == "__main__":
    main()
