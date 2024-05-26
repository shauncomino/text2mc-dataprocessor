import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModel(nn.Module):
    def __init__(self, emb_size: int, emb_dimension: int):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        
        self.target_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.output = nn.Linear(emb_dimension, emb_size)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.target_embeddings.weight.data, -
                      initrange, initrange)

    
    def forward(self, target_blocks, context_blocks):
        total_batch_loss = 0
        for idx in range (0, len(target_blocks)): 
            target = target_blocks[idx]
            emb_target = self.target_embeddings(target)

            score = self.output(emb_target)
            score = F.log_softmax(score, dim=-1)
            losses = torch.stack([F.nll_loss(score, context_block.long())
                                for context_block in context_blocks[idx]])
            total_batch_loss += losses.mean()

        return losses.mean()/len(target_blocks)
