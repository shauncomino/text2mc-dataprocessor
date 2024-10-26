import os
import sys
import h5py
from itertools import product
from predictor import predict
from predictor import text2mcPredictor

builds1 = ['/lustre/fs1/groups/jaedo/processed_builds/batch_164_4259.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_164_4256.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_224_5804.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_179_4650.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_478_12414.h5']
builds2 = ['/lustre/fs1/groups/jaedo/processed_builds/batch_443_11512.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_443_11493.h5',' /lustre/fs1/groups/jaedo/processed_builds/batch_691_17960_1.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_913_23712.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_909_23629.h5']

build_set = product(builds1, builds2)

predictor = text2mcPredictor()

for build1_path, build2_path in build_set:
    predictor.predict(build1_path, build2_path)
