import os
import sys
import h5py
from itertools import product
from predictor import text2mcPredictor

builds1 = ['/lustre/fs1/groups/jaedo/processed_builds/batch_229_5948.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_342_8886.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_103_2663.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_540_14037_1.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_389_10111.h5']
builds2 = ['/lustre/fs1/groups/jaedo/processed_builds/batch_254_6591.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_486_12614.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_868_22555_2.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_280_7257.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_303_7867.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_544_14142.h5']

build_set = product(builds1, builds2)

predictor = text2mcPredictor()

for build1_path, build2_path in build_set:
    predictor.predict(build1_path, build2_path)
