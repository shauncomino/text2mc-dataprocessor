import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModel(nn.Module):
    def __init__(self, num_embeddings: int, emb_dimension: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb_dimension = emb_dimension
        
        self.target_embeddings = nn.Embedding(num_embeddings, emb_dimension)
        self.output = nn.Linear(emb_dimension, num_embeddings)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.target_embeddings.weight.data, -
                      initrange, initrange)

    
    def forward(self, target_blocks, context_blocks):
        total_batch_loss = 0
       
        if target_blocks == None or context_blocks == None or len(target_blocks) == 0 or len(context_blocks) == 0: 
            print("Error. Did not receive target or context blocks in foward pass.")
            # Return 0 loss, using tensor to make sure backward pass can still do its thing 
            return torch.tensor(0.0, requires_grad=True)
        
        for idx in range (0, len(target_blocks)): 
            target = target_blocks[idx]
            emb_target = self.target_embeddings(target)

            score = self.output(emb_target)
            score = F.log_softmax(score, dim=-1)
            losses = torch.stack([F.nll_loss(score, context_block.long())
                                for context_block in context_blocks[idx]])
            total_batch_loss += losses.mean()

        return losses.mean()/len(target_blocks)
