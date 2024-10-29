import os
import sys
import h5py
from itertools import product
from predictor import text2mcPredictor

builds1 = ['/lustre/fs1/groups/jaedo/processed_builds/batch_389_10111.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_544_14142.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_196_5090.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_25_641.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_628_16325.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_397_10302.h5']
builds2 = ['/lustre/fs1/groups/jaedo/processed_builds/batch_404_10483.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_783_20352.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_155_4025.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_659_17131.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_224_5818.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_699_18155.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_471_12228.h5']

build_set = product(builds1, builds2)

predictor = text2mcPredictor()

for build1_path, build2_path in build_set:
    predictor.predict(build1_path, build2_path)
