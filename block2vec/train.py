import os
import sys
import numpy as np
import pytorch_lightning as pl
from block2vec import Block2Vec, Block2VecArgs

class TrainBlock2VecArgs(Block2VecArgs):
    debug: bool = False

    def process_args(self) -> None:
        super().process_args()
        os.makedirs(self.output_path, exist_ok=True)

def get_random_builds(x_dim, y_dim, z_dim, max_token_val, num_builds):
    builds = [np.random.randint(1, max_token_val, size=(x_dim, y_dim, z_dim)) for _ in range(num_builds)]
    return np.stack([build for build in builds])

def main():
    random_builds = get_random_builds(4, 4, 4, 5, 6)
    block2vec = Block2Vec(builds=random_builds)
    trainer = pl.Trainer(max_epochs=3, log_every_n_steps=1)
    trainer.fit(block2vec)

if __name__ == "__main__":
    main()
