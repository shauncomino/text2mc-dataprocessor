import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", ".."))

from typing import Tuple
import torch
import pytorch_lightning as pl
from block2vec import Block2Vec, Block2VecArgs


class TrainBlock2VecArgs(Block2VecArgs):
    debug: bool = False

    def process_args(self) -> None:
        super().process_args()
        os.makedirs(self.output_path, exist_ok=True)


def main():

    # Example tokenized build (5 x 5 x 5)
    # Tokens are available in tok_to_block.json
    example_build = torch.tensor(
        [
            [
                [1, 0, 3, 2, 1],
                [0, 3, 2, 1, 5],
                [3, 2, 1, 0, 3],
                [2, 1, 5, 3, 2],
                [1, 0, 3, 2, 1],
            ],
            [
                [0, 3, 2, 1, 0],
                [3, 2, 1, 5, 3],
                [2, 1, 0, 3, 2],
                [1, 0, 3, 2, 1],
                [0, 3, 2, 1, 0],
            ],
            [
                [3, 2, 1, 0, 3],
                [2, 1, 5, 3, 2],
                [1, 5, 3, 2, 1],
                [0, 3, 2, 1, 0],
                [3, 2, 1, 0, 3],
            ],
            [
                [2, 1, 0, 3, 2],
                [1, 5, 3, 2, 1],
                [6, 3, 2, 1, 0],
                [3, 2, 1, 0, 3],
                [2, 1, 0, 3, 2],
            ],
            [
                [1, 0, 3, 2, 1],
                [0, 6, 2, 1, 0],
                [3, 2, 1, 0, 3],
                [2, 1, 0, 3, 2],
                [1, 0, 6, 2, 1],
            ],
        ]
    )

    block2vec = Block2Vec(build=example_build)
    trainer = pl.Trainer(max_epochs=3, log_every_n_steps=1)
    trainer.fit(block2vec)


if __name__ == "__main__":
    main()
