import os
import h5py
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

def get_builds(builds_dir):
    builds = []
    x_max = 0
    y_max = 0
    z_max = 0
    count = 0
    for filename in os.listdir(builds_dir):
        if str(filename).endswith(".h5"):
            with h5py.File(os.path.join(builds_dir, filename), "r") as f:
                if len(list(f.keys())) > 0:
                    builds.append(f[list(f.keys())[0]][()])
                    if builds[len(builds) - 1].shape[0] > x_max:
                        x_max = builds[len(builds) - 1].shape[0]
                    if builds[len(builds) - 1].shape[1] > y_max:
                        y_max = builds[len(builds) - 1].shape[1]
                    if builds[len(builds) - 1].shape[2] > z_max:
                        z_max = builds[len(builds) - 1].shape[2]
                    count += 1
                    print("%d/%d builds loaded..." % (count, len(os.listdir(builds_dir))))
                else:
                    print("%s failed, no keys" % filename)
    # Pad builds with undefined blocks
    count = 0
    padded_builds = []
    for build in builds:
        padded_builds.append(np.pad(build, [(0, x_max - build.shape[0]), (0, y_max - build.shape[1]), (0, z_max - build.shape[2])],
                                    'constant', constant_values=[(-1, -1), (-1, -1), (-1, -1)]))
        count += 1
        print("%d/%d builds padded..." % (count, len(os.listdir(builds_dir))))
    print("Maximum dimensions: %dx%dx%d" % (x_max, y_max, z_max))
    return np.stack([build for build in padded_builds])


def main():
    build_list = get_builds("data")
    #random_builds = get_random_builds(4, 4, 4, 5, 6)
    block2vec = Block2Vec(builds=build_list)
    trainer = pl.Trainer(max_epochs=3, log_every_n_steps=1)
    trainer.fit(block2vec)

if __name__ == "__main__":
    main()
