import os
import sys
import h5py
from itertools import product
from predictor import text2mcPredictor
import gc

builds1 = ['/lustre/fs1/groups/jaedo/processed_builds/batch_217_5637.h5',"/lustre/fs1/groups/jaedo/processed_builds/batch_225_5840.h5",'/lustre/fs1/groups/jaedo/processed_builds/batch_56_1440.h5',"/lustre/fs1/groups/jaedo/processed_builds/batch_319_8281.h5", "/lustre/fs1/groups/jaedo/processed_builds/batch_109_2825.h5"]
builds2 = ["/lustre/fs1/groups/jaedo/processed_builds/batch_404_10483.h5",'/lustre/fs1/groups/jaedo/processed_builds/batch_3_67.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_397_10302.h5','/lustre/fs1/groups/jaedo/processed_builds/batch_22_557.h5', '/lustre/fs1/groups/jaedo/processed_builds/batch_52_1334.h5']

r = text2mcPredictor()

for build1_path in builds1:
    for build2_path in builds2:
        predictor.predict(build1_path, build2_path)
        gc.collect()

