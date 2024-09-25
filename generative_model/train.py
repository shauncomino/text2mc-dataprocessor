import torch
import torch.optim as optim
import glob
import json
import os
from text2mcVAEDataset import text2mcVAEDataset
from encoder import text2mcVAEEncoder
from decoder import text2mcVAEDecoder
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import h5py
import numpy as np
import random


def main():
    # Get rank and world_size from environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    batch_size = 8
    num_epochs = 32

    # Paths and configurations
    checkpoint_path = r'/lustre/fs1/home/scomino/training/checkpoint.pth'
    block2tok_file_path = r'/lustre/fs1/home/scomino/text2mc/text2mc-dataprocessor/world2vec/tok2block.json'
    builds_folder_path = r'/lustre/fs1/groups/jaedo/processed_builds'
    build1_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_101_2606.h5'
    build2_path = r'/lustre/fs1/groups/jaedo/processed_builds/batch_118_3048.h5'
    save_dir = r'/lustre/fs1/home/scomino/training/interpolations'
    best_model_path = r'/lustre/fs1/home/scomino/training/best_model.pth'

    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize the process group
    dist.init_process_group(backend='nccl')

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if rank == 0:
        print(f"Using GPUs with CUDA, Rank: {rank}")

    # Load block2tok
    with open(block2tok_file_path, 'r') as j:
        block2tok = json.load(j)
        block2tok = dict((v, k) for k, v in block2tok.items())

    # Prepare the dataset
    hdf5_filepaths = glob.glob(os.path.join(builds_folder_path, '*.h5'))
    dataset = text2mcVAEDataset(file_paths=hdf5_filepaths, block2tok=block2tok, fixed_size=(128, 128, 128))

    # Get num_tokens from dataset
    num_tokens = dataset.num_tokens  # Total number of tokens
    embedding_dim = 32  # Choose an embedding dimension

    # Split the dataset into training, validation, and test sets
    dataset_size = len(dataset)
    validation_split = 0.2
    test_split = 0.1
    train_size = int((1 - validation_split - test_split) * dataset_size)
    val_size = int(validation_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)  # Ensure splits are consistent
    )

    # Create DistributedSamplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    # Initialize the model components
    encoder = text2mcVAEEncoder(num_tokens=num_tokens, embedding_dim=embedding_dim).to(device)
    decoder = text2mcVAEDecoder(num_tokens=num_tokens, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-2)
    scaler = GradScaler()  # Initialize the gradient scaler for mixed precision

    # Wrap models with DDP
    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[local_rank], output_device=local_rank)
    decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[local_rank], output_device=local_rank)

    start_epoch = 1
    best_val_loss = float('inf')

    # Load checkpoint if it exists (only on rank 0)
    if rank == 0 and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.module.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print(f"Rank {rank}: No checkpoint found, starting from scratch")

    # Broadcast the start_epoch and best_val_loss to all processes
    start_epoch_tensor = torch.tensor(start_epoch).to(device)
    best_val_loss_tensor = torch.tensor(best_val_loss).to(device)
    dist.broadcast(start_epoch_tensor, src=0)
    dist.broadcast(best_val_loss_tensor, src=0)
    start_epoch = start_epoch_tensor.item()
    best_val_loss = best_val_loss_tensor.item()


    # Adjust the loss function to use CrossEntropyLoss
    import torch.nn.functional as F

    def loss_function(recon_x, x, mu, logvar, mask):
        # recon_x: (Batch_Size, num_tokens, Depth, Height, Width)
        # x: (Batch_Size, Depth, Height, Width)  # Target tokens
        # mask: (Batch_Size, Depth, Height, Width)

        # Flatten the outputs and targets
        batch_size = x.size(0)
        recon_x = recon_x.view(batch_size, recon_x.size(1), -1)  # (Batch_Size, num_tokens, N)
        x = x.view(batch_size, -1)  # (Batch_Size, N)
        mask = mask.view(batch_size, -1)  # (Batch_Size, N)

        # Apply mask to targets (we don't need to mask the logits)
        x_masked = x * mask.long()

        # CrossEntropyLoss expects input: (N, C), target: (N)
        recon_x = recon_x.permute(0, 2, 1).reshape(-1, recon_x.size(1))  # (Batch_Size * N, num_tokens)
        x_masked = x_masked.reshape(-1)  # (Batch_Size * N)
        mask = mask.reshape(-1)  # (Batch_Size * N)

        # Compute loss only where mask is 1
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            recon_loss = torch.tensor(0.0, device=x.device)
        else:
            recon_loss = F.cross_entropy(recon_x[valid_indices], x_masked[valid_indices], reduction='sum')

        # KL Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + KLD

    # Function to interpolate between two builds and generate new builds
    def interpolate_and_generate(encoder, decoder, build1_path, build2_path, save_dir, epoch, num_interpolations=5):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            # Load the two builds
            dataset = text2mcVAEDataset(file_paths=[build1_path, build2_path], block2tok=block2tok, block_ignore_list=[])
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

            # Get the latent representations
            data_list = []
            mask_list = []
            for data, mask in data_loader:
                data_list.append(data.to(device))
                mask_list.append(mask.to(device))

            z_list = []
            for data in data_list:
                z, mu, logvar = encoder(data)
                z_list.append(z)

            # Linearly interpolate between z1 and z2
            z1 = z_list[0]
            z2 = z_list[1]

            interpolations = []
            for alpha in np.linspace(0, 1, num_interpolations):
                z_interp = (1 - alpha) * z1 + alpha * z2
                interpolations.append(z_interp)

            # Generate builds from interpolated latent vectors
            for idx, z in enumerate(interpolations):
                recon_build = decoder(z)
                # recon_build: (1, num_tokens, Depth, Height, Width)

                # Convert logits to predicted tokens
                recon_build = recon_build.argmax(dim=1)  # (1, Depth, Height, Width)

                # Convert to numpy array
                recon_build_np = recon_build.cpu().numpy().squeeze(0)  # (Depth, Height, Width)

                # Save the build as an HDF5 file
                save_path = os.path.join(save_dir, f'epoch_{epoch}_interp_{idx}.h5')
                with h5py.File(save_path, 'w') as h5f:
                    h5f.create_dataset('build', data=recon_build_np, compression='gzip')

                if rank == 0:
                    print(f'Saved interpolated build at {save_path}')

    # Training loop
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs + 1):
        train_sampler.set_epoch(epoch)
        encoder.train()
        decoder.train()
        total_loss = 0

        for batch_idx, (data, mask) in enumerate(train_loader):
            data, mask = data.to(device), mask.to(device)
            optimizer.zero_grad()

            # Mixed precision context
            with autocast():
                # Encode the data to get latent representation, mean, and log variance
                z, mu, logvar = encoder(data)

                # Decode the latent variable to reconstruct the input
                recon_batch = decoder(z)

                # Compute the loss
                loss = loss_function(recon_batch, data, mu, logvar, mask)

            # Scale loss and perform backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0 and rank == 0:
                print(f'Rank {rank}, Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}')

        # Average the training loss across all processes
        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_train_loss_tensor = torch.tensor(avg_train_loss).to(device)
        dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = avg_train_loss_tensor.item() / world_size

        if rank == 0:
            print(f'====> Epoch: {epoch} Average training loss: {avg_train_loss:.4f}')

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for data, mask in val_loader:
                data, mask = data.to(device), mask.to(device)
                z, mu, logvar = encoder(data)
                recon_batch = decoder(z)
                loss = loss_function(recon_batch, data, mu, logvar, mask)
                val_loss += loss.item()

        # Average the validation loss across all processes
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_loss_tensor = torch.tensor(avg_val_loss).to(device)
        dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = avg_val_loss_tensor.item() / world_size

        if rank == 0:
            print(f'====> Epoch: {epoch} Validation loss: {avg_val_loss:.4f}')

            # Check if validation loss improved, and save model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.module.state_dict(),
                    'decoder_state_dict': decoder.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, best_model_path)
                print(f'Saved new best model at epoch {epoch} with validation loss {avg_val_loss:.4f}')

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.module.state_dict(),
                'decoder_state_dict': decoder.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'seed': seed,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch}")

            # Interpolate and generate builds
            print(f'Interpolating between builds at the end of epoch {epoch}')
            try:
                interpolate_and_generate(encoder.module, decoder.module, build1_path, build2_path, save_dir, epoch, num_interpolations=5)
            except Exception as e:
                print(f"Unable to generate interpolations for this epoch due to error: {e}")

    # Clean up the process group
    dist.destroy_process_group()

if __name__ == '__main__':
    main()