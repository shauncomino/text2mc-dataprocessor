import torch
from text2mcVAE import text2mcVAE

# Instantiate the model
model = text2mcVAE()

# Calculate total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Assuming 32-bit floats, each parameter requires 4 bytes
bytes_per_param = 4
total_memory_bytes = total_params * bytes_per_param

# Convert total memory from bytes to more interpretable units (MB)
total_memory_MB = total_memory_bytes / (1024 ** 2)

print(f'Total trainable parameters: {total_params}')
print(f'Estimated memory requirement (in MB): {total_memory_MB:.2f}')