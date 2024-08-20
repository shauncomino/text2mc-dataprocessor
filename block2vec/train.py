import os
import numpy as np
import h5py
import pytorch_lightning as pl
from block2vec import Block2Vec, Block2VecArgs

class TrainBlock2VecArgs(Block2VecArgs):
    debug: bool = False

    def process_args(self) -> None:
        super().process_args()
        os.makedirs(self.output_path, exist_ok=True)

"""
def get_random_builds(x_dim, y_dim, z_dim, max_token_val, num_builds):
    builds = [np.random.randint(1, max_token_val, size=(x_dim, y_dim, z_dim)) for _ in range(num_builds)]
    return np.stack([build for build in builds])

def slice_to_fixed_dim(arr, max_dim):
    x_dim, y_dim, z_dim = arr.shape
    
    x_slice = slice(0, min(x_dim, max_dim))
    y_slice = slice(0, min(y_dim, max_dim))
    z_slice = slice(0, min(z_dim, max_dim))
    
    return arr[x_slice, y_slice, z_slice]

def pad_to_fixed_dim(arr, fixed_dim):
    return np.pad(arr, [(0, fixed_dim - arr.shape[0]), (0, fixed_dim - arr.shape[1]), (0, fixed_dim - arr.shape[2])],
                                    'constant', constant_values=[(-1, -1), (-1, -1), (-1, -1)])

def get_builds(builds_dir):
    builds = []
    count = 0
    for filename in os.listdir(builds_dir):
        if str(filename).endswith(".h5"):
            with h5py.File(os.path.join(builds_dir, filename), "r") as f:
                if len(list(f.keys())) > 0:
                    builditem = f[list(f.keys())[0]][()]
                    builditem = np.array(builditem, dtype=np.int32)
                    builds.append(builditem)
                    count += 1
                    print("%d/%d builds loaded..." % (count, len(os.listdir(builds_dir))))
                else:
                    print("%s failed, no keys" % filename)
    padded_builds = []
    for build in builds: 
        build = slice_to_fixed_dim(build, 4)
        build = pad_to_fixed_dim(build, 4)
        padded_builds.append(build)
        
    return np.stack([build for build in padded_builds])
"""

def main():
    block2vec = Block2Vec()
    trainer = pl.Trainer(max_epochs=3, log_every_n_steps=1)
    trainer.fit(block2vec)

if __name__ == "__main__":
    main()
