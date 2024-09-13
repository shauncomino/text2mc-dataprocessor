import torch
from torch import nn
from torch.nn import functional as F
from decoder import (
    text2mcVAEAttentionBlock as text2mcVAEAttentionBlock,
    text2mcVAEResidualBlock as text2mcVAEResidualBlock,
)
from torch.nn import init


class text2mcVAEEmbedder(nn.Sequential):
    def __init__(self, emb_size: int, emb_dimension: int):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.target_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.output = nn.Linear(emb_dimension, emb_size)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.target_embeddings.weight.data, -
                      initrange, initrange)

    def forward(self, target_block, context_blocks):
        emb_target = self.target_embeddings(target_block)

        score = self.output(emb_target)
        score = F.log_softmax(score, dim=-1)

        losses = torch.stack([F.nll_loss(score, context_block.long())
                              for context_block in context_blocks])
        return losses.mean()

